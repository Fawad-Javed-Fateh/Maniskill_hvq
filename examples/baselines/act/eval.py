ALGO_NAME = 'BC_ACT_rgbd'

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
from act.evaluate import evaluate
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import ACT_DETRVAE, DETRVAE
# from act.detr.detr_vae import build_encoder
from act.detr.icrt.action_tokenizer import build_lfq_encoder,build_hvq_encoder,build_mega_mind
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
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PickCube-v1"
    """the id of the environment"""
    demo_path: str = 'pickcube.trajectory.rgbd.pd_joint_delta_pos.cpu.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

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
    include_depth: bool = True

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
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1
    """the frequency of logging the training metrics"""
    eval_freq: int = 500
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 100
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, rgb=True, depth=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # resize the input image to be at least 224x224
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                resized_rgb = self.transforms(
                    cam_data["rgb"].permute(0, 3, 1, 2)
                )  # (1, 3, 224, 224)
                images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(
                    depth.permute(0, 3, 1, 2)
                )  # (1, 1, 224, 224)
                images_depth.append(resized_depth)

        rgb = torch.stack(images_rgb, dim=1) # (1, num_cams, C, 224, 224), uint8
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1) # (1, num_cams, C, 224, 224), float16

        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)
        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb
        elif self.include_rgb and self.include_depth:
            ret["rgb"] = rgb
            ret["depth"] = depth
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth
        return ret


class SmallDemoDataset_ACTPolicy(Dataset): # Load everything into memory
    def __init__(self, data_path, num_queries, num_traj, include_depth=True):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from act.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print('Raw trajectory loaded, start to pre-process the observations...')

        self.include_depth = include_depth
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # pre-trained models from torchvision.models expect input image to be at least 224x224

        # Pre-process the observations, make them align with the obs returned by the FlattenRGBDObservationWrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories['observations']:
            obs_traj_dict = self.process_obs(obs_traj_dict)
            obs_traj_dict_list.append(obs_traj_dict)
        trajectories['observations'] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict.keys())

        # Pre-process the actions
        for i in range(len(trajectories['actions'])):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done.')

        # When the robot reaches the goal state, its joints and gripper fingers need to remain stationary
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in range(self.num_traj):
            episode_len = trajectories['actions'][traj_idx].shape[0]
            self.slices += [
                (traj_idx, ts) for ts in range(episode_len)
            ]

        print(f"Length of Dataset: {len(self.slices)}")

        self.num_queries = num_queries
        self.trajectories = trajectories
        self.delta_control = 'delta' in args.control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        # get state at start_ts only
        state = self.trajectories['observations'][traj_idx]['state'][ts]
        # get num_queries actions
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.num_queries:
            if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
                # making the robot (arm and gripper) stay still
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)

        # normalize state and act_seq
        if not self.delta_control:
            state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # get rgb or rgbd data at start_ts and combine with state to form obs
        if self.include_depth:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            depth = self.trajectories['observations'][traj_idx]['depth'][ts]
            obs = dict(state=state, rgb=rgb, depth=depth)
        else:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            obs = dict(state=state, rgb=rgb)

        return {
            'observations': obs,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)

    def process_obs(self, obs_dict):
        # get rgbd data
        sensor_data = obs_dict.pop("sensor_data")
        del obs_dict["sensor_param"]
        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            rgb = torch.from_numpy(cam_data["rgb"]) # (ep_len, H, W, 3)
            resized_rgb = self.transforms(
                rgb.permute(0, 3, 1, 2)
            )  # (ep_len, 3, 224, 224); pre-trained models from torchvision.models expect input image to be at least 224x224
            images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = torch.Tensor(cam_data["depth"].astype(np.float32) / 1024).to(torch.float16) # (ep_len, H, W, 1)
                resized_depth = self.transforms(
                    depth.permute(0, 3, 1, 2)
                )  # (ep_len, 1, 224, 224); pre-trained models from torchvision.models expect input image to be at least 224x224
                images_depth.append(resized_depth)
        rgb = torch.stack(images_rgb, dim=1) # (ep_len, num_cams, 3, 224, 224) # still uint8
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1) # (ep_len, num_cams, 1, 224, 224) # float16

        # flatten the rest of the data which should just be state data
        obs_dict['extra'] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict['extra'].items()} # dirty fix for data that has one dimension (e.g. is_grasped)
        obs_dict = common.flatten_state_dict(obs_dict, use_torch=True)

        processed_obs = dict(state=obs_dict, rgb=rgb, depth=depth) if self.include_depth else dict(state=obs_dict, rgb=rgb)

        return processed_obs

    def get_norm_stats(self):
        all_state_data = []
        all_action_data = []
        for traj_idx, ts in self.slices:
            state = self.trajectories['observations'][traj_idx]['state'][ts]
            act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
            action_len = act_seq.shape[0]
            if action_len < self.num_queries:
                target_pos = act_seq[-1]
                act_seq = torch.cat([act_seq, target_pos.repeat(self.num_queries-action_len, 1)], dim=0)
            all_state_data.append(state)
            all_action_data.append(act_seq)

        all_state_data = torch.stack(all_state_data)
        all_action_data = torch.concatenate(all_action_data)

        # normalize obs (state) data
        state_mean = all_state_data.mean(dim=0, keepdim=True)
        state_std = all_state_data.std(dim=0, keepdim=True)
        state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

        # normalize action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

        stats = {"action_mean": action_mean, "action_std": action_std,
                 "state_mean": state_mean, "state_std": state_std,
                 "example_state": state}

        return stats


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space['state'].shape) == 1 # (obs_dim,)
        assert len(env.single_observation_space['rgb'].shape) == 4 # (num_cams, C, H, W)
        assert len(env.single_action_space.shape) == 1 # (act_dim,)
        #assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()

        self.state_dim = env.single_observation_space['state'].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # CNN backbone
        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        #encoder = build_lfq_encoder(args)
        encoder = build_mega_mind(args)
        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = ACT_DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
        )

    def compute_loss(self, obs, action_seq):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if args.include_depth:
            obs['depth'] = obs['depth'].float()

        # forward pass
        a_hat, vq_vae_loss = self.model(obs, action_seq)

        # compute l1 loss and kl loss
        # total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        # store all loss
        loss_dict = dict()
        loss_dict['l1'] = l1
        print('l1_loss',loss_dict['l1'])
        print("vq_vae_loss",vq_vae_loss)
        # loss_dict['kl'] = total_kld[0]
        # loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + vq_vae_loss
        loss_dict['loss'] = loss_dict['l1'] + vq_vae_loss
        return loss_dict

    def get_action(self, obs):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if args.include_depth:
            obs['depth'] = obs['depth'].float()

        # forward pass
        a_hat, _ = self.model(obs) # no action, sample from prior

        return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

# def save_ckpt(run_name, tag):
#     os.makedirs(f'/workspace/ManiSkill_release_new/runs/{run_name}/checkpoints', exist_ok=True)
#     ema.copy_to(ema_agent.parameters())
#     torch.save({
#         'norm_stats': dataset.norm_stats,
#         'agent': agent.state_dict(),
#         'ema_agent': ema_agent.state_dict(),
#     }, f'/workspace/ManiSkill_release_new/runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        run_name='eval_without_cte'
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgbd" if args.include_depth else "rgb", render_mode="rgb_array")
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=args.include_depth)]
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None, wrappers=wrappers)

    # dataloader setup (needed for normalization stats)
    dataset = SmallDemoDataset_ACTPolicy(args.demo_path, args.num_queries, num_traj=args.num_demos, include_depth=args.include_depth)
    
    obs_mode = "rgb+depth" if args.include_depth else "rgb"

    # agent setup
    agent = Agent(envs, args).to(device)

    # Load the pre-trained model
    checkpoint_path = "/home/retrocausal-train/Desktop/Maniskill_hvq/runs/without_cte/2000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading model from {checkpoint_path}")
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
    elif 'ema_agent' in checkpoint:
        # Use EMA agent if available (usually better)
        agent.load_state_dict(checkpoint['ema_agent'])
    else:
        # If checkpoint is just the state dict
        agent.load_state_dict(checkpoint)
    
    #agent.load_state_dict(checkpoint)
    agent.eval()

    # Evaluation setup
    eval_kwargs = dict(
        stats=dataset.norm_stats, 
        num_queries=args.num_queries, 
        temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps, 
        device=device, 
        sim_backend=args.sim_backend
    )

    # Optional: setup wandb/tensorboard for logging
    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=config,
            name=run_name,
            save_code=True,
            group="ACT_Eval",
            tags=["act", "evaluation"]
        )
    writer = SummaryWriter(f"runs/{run_name}")

    # Run evaluation on 100 episodes
    print("Starting evaluation on 100 episodes...")
    num_eval_episodes = 10000
    eval_metrics = evaluate(num_eval_episodes, agent, envs, eval_kwargs)

    print(f"\nEvaluated {len(eval_metrics['success_at_end'])} episodes")
    print("="*50)
    for k in eval_metrics.keys():
        mean_value = np.mean(eval_metrics[k])
        writer.add_scalar(f"eval/{k}", mean_value, 0)
        print(f"{k}: {mean_value:.4f}")
    print("="*50)

    envs.close()
    writer.close()
    
    if args.track:
        wandb.finish()