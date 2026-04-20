ALGO_NAME = 'BC_ACT_rgbd_InContextLearning'

import argparse
import os
import random
from distutils.util import strtobool
from functools import partial
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from act.evaluate_murad import (
    load_h5_trajectories,
    select_prompt_and_query,
    evaluate_with_prompts,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import tyro


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PickCube-v1"
    """the id of the environment"""
    demo_path: str = 'pickcube.trajectory.rgbd.pd_joint_delta_pos.cpu.h5'
    """the path of demo dataset (h5 file)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""

    # ACT specific arguments
    lr: float = 1e-4
    """the learning rate of the Action Chunking with Transformers"""
    kl_weight: float = 10
    """weight for the kl loss term"""
    temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed"""

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Max episode steps"""
    log_freq: int = 1
    """the frequency of logging the training metrics"""
    eval_freq: int = 500
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints"""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 100
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments"""

    # In-context learning arguments
    prompt_traj_start: int = 991
    """Starting trajectory index for prompts"""
    prompt_traj_end: int = 1002
    """Ending trajectory index for prompts (inclusive)"""
    num_prompt_trajs: int = 5
    """Number of prompt trajectories to use"""
    num_query_trajs: int = 5
    """Number of query trajectories to use"""

    # additional tags/configs
    demo_type: Optional[str] = None


class Agent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.kl_weight = args.kl_weight
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # CNN backbone
        from act.detr.backbone import build_backbone
        from act.detr.transformer import build_transformer
        from act.detr.detr_vae import ACT_DETRVAE
        from act.detr.icrt.action_tokenizer import build_hvq_encoder

        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_hvq_encoder(args)
        
        # ACT model
        self.model = ACT_DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=25,  # qpos(9) + qvel(9) + tcp_pose(7)
            action_dim=4,  # action dimension
            num_queries=args.num_queries,
        )

    def compute_loss(self, obs, action_seq):
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        if 'depth' in obs:
            obs['depth'] = obs['depth'].float()

        a_hat, vq_vae_loss = self.model(obs, action_seq)

        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['loss'] = loss_dict['l1'] + vq_vae_loss
        return loss_dict

    def get_action(self, obs):
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        a_hat, _ = self.model(obs)

        return a_hat


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Verify control mode matches
    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                demo_info = json.load(f)
                if 'control_mode' in demo_info['env_info']['env_kwargs']:
                    control_mode = demo_info['env_info']['env_kwargs']['control_mode']
                elif 'control_mode' in demo_info['episodes'][0]:
                    control_mode = demo_info['episodes'][0]['control_mode']
                else:
                    control_mode = args.control_mode
                assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has {control_mode}, args has {args.control_mode}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # ==================== Load H5 Trajectories ====================
    print(f"\n{'='*60}")
    print(f"Loading trajectories from H5 file: {args.demo_path}")
    print(f"{'='*60}")
    
    # Generate trajectory indices from range [prompt_traj_start, prompt_traj_end]
    available_traj_indices = list(range(args.prompt_traj_start, args.prompt_traj_end + 1))
    print(f"Available trajectories: {available_traj_indices}")
    
    # Load all trajectories in the range
    trajectories = load_h5_trajectories(args.demo_path, available_traj_indices)
    print(f"Loaded {len(trajectories['observations'])} trajectories")
    
    # ==================== Select Prompt and Query Trajectories ====================
    print(f"\n{'='*60}")
    print(f"Selecting prompt and query trajectories")
    print(f"{'='*60}")
    
    prompt_trajs, query_trajs, prompt_indices, query_indices = select_prompt_and_query(
        trajectories,
        num_prompt_trajs=args.num_prompt_trajs,
        num_query_trajs=args.num_query_trajs,
    )
    
    # Map local indices back to original trajectory indices
    prompt_traj_ids = [available_traj_indices[i] for i in prompt_indices]
    query_traj_ids = [available_traj_indices[i] for i in query_indices]
    
    print(f"Prompt trajectories (indices in H5): {prompt_traj_ids}")
    print(f"Query trajectories (indices in H5): {query_traj_ids}")
    
    # ==================== Agent Setup ====================
    print(f"\n{'='*60}")
    print(f"Setting up agent and loading checkpoint")
    print(f"{'='*60}")
    
    agent = Agent(args).to(device)
    
    # Load the pre-trained model
    checkpoint_path = "/home/retrocausal-train/Desktop/Maniskill_hvq/runs/Stack_cube_murad/checkpoints/35000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading model from {checkpoint_path}")
    
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
    elif 'ema_agent' in checkpoint:
        agent.load_state_dict(checkpoint['ema_agent'])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    print("Model loaded and set to eval mode")

    # ==================== Evaluation Setup ====================
    print(f"\n{'='*60}")
    print(f"Running in-context learning evaluation")
    print(f"{'='*60}")
    
    eval_kwargs = dict(
        stats=None,  # No normalization stats needed for in-context learning
        num_queries=args.num_queries,
        temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps,
        device=device,
        sim_backend=args.sim_backend
    )
    
    transforms = T.Compose([
        T.Resize((224, 224), antialias=True),
    ])

    # Optional: setup wandb/tensorboard for logging
    if args.track:
        import wandb
        config = vars(args)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=config,
            name=run_name,
            save_code=True,
            group="ACT_InContextLearning",
            tags=["act", "in-context-learning", "evaluation"]
        )
    
    writer = SummaryWriter(f"runs/{run_name}")

    # Run evaluation
    print(f"\nEvaluating on {len(query_trajs['observations'])} query trajectories...")
    eval_metrics = evaluate_with_prompts(
        agent,
        prompt_trajs,
        query_trajs,
        eval_kwargs,
        transforms=transforms,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Total evaluated samples: {len(eval_metrics['l1_error'])}")
    
    for k in sorted(eval_metrics.keys()):
        mean_value = np.mean(eval_metrics[k])
        std_value = np.std(eval_metrics[k])
        writer.add_scalar(f"eval/{k}", mean_value, 0)
        print(f"{k}: {mean_value:.6f} ± {std_value:.6f}")
    
    print(f"{'='*60}\n")

    writer.close()
    
    if args.track:
        wandb.finish()
    
    print("Evaluation completed!")