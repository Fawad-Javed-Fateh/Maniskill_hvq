from collections import defaultdict
import gymnasium
import numpy as np
import torch
import h5py
from typing import Dict, List, Tuple

from mani_skill.utils import common
from act.ur5_action_converter import convert_maniskill_to_ur5, ManiSkillToUR5Converter

def load_h5_trajectories(data_path: str, traj_indices: List[int]) -> Dict:
    """
    Load specific trajectories from H5 file.
    
    Args:
        data_path: Path to H5 file
        traj_indices: List of trajectory indices to load (e.g., [1000, 1001, ..., 1009])
    
    Returns:
        Dictionary with 'observations' and 'actions' lists
    """
    trajectories = {'observations': [], 'actions': []}
    
    with h5py.File(data_path, 'r') as f:
        for traj_idx in traj_indices:
            traj_key = f'traj_{traj_idx}'
            if traj_key not in f:
                print(f"Warning: {traj_key} not found in H5 file")
                continue
            
            traj_group = f[traj_key]
            
            # Load observations
            obs_dict = {}
            
            # Agent state (qpos, qvel)
            agent_qpos = torch.from_numpy(np.array(traj_group['obs/agent/qpos']))  # (T, 9)
            agent_qvel = torch.from_numpy(np.array(traj_group['obs/agent/qvel']))  # (T, 9)
            
            # Extra (tcp_pose)
            tcp_pose = torch.from_numpy(np.array(traj_group['obs/extra/tcp_pose']))  # (T, 7)
            
            # Sensor data (RGB only, no depth)
            base_camera_rgb = torch.from_numpy(np.array(traj_group['obs/sensor_data/base_camera/rgb']))  # (T, 128, 128, 3)
            hand_camera_rgb = torch.from_numpy(np.array(traj_group['obs/sensor_data/hand_camera/rgb']))  # (T, 128, 128, 3)
            
            # Combine multi-camera observations
            rgb = torch.stack([base_camera_rgb, hand_camera_rgb], dim=1)  # (T, 2, 128, 128, 3)
            
            # Flatten state (qpos, qvel, tcp_pose)
            state = torch.cat([agent_qpos, agent_qvel, tcp_pose], dim=-1)  # (T, 9+9+7=25)
            
            obs_dict['state'] = state
            obs_dict['rgb'] = rgb
            
            trajectories['observations'].append(obs_dict)
            
            # Load actions
            actions = torch.from_numpy(np.array(traj_group['actions']))  # (T-1, 4)
            trajectories['actions'].append(actions)
    
    return trajectories


def select_prompt_and_query(
    trajectories: Dict,
    num_prompt_trajs: int = 5,
    num_query_trajs: int = 5,
    prompt_traj_indices: List[int] = None,
    query_traj_indices: List[int] = None,
) -> Tuple[Dict, Dict, List[int], List[int]]:
    """
    Select prompt trajectories and query trajectories from loaded demonstrations.
    
    Args:
        trajectories: Dictionary with 'observations' and 'actions' lists
        num_prompt_trajs: Number of trajectories to use as prompts
        num_query_trajs: Number of trajectories to use as queries
        prompt_traj_indices: Specific indices to use as prompts (if None, randomly select)
        query_traj_indices: Specific indices to use as queries (if None, randomly select from remaining)
    
    Returns:
        (prompt_trajs, query_trajs, used_prompt_indices, used_query_indices)
    """
    total_trajs = len(trajectories['observations'])
    
    if prompt_traj_indices is None:
        prompt_traj_indices = list(range(min(num_prompt_trajs, total_trajs)))
    
    # Ensure query trajectories don't overlap with prompt trajectories
    available_for_query = [i for i in range(total_trajs) if i not in prompt_traj_indices]
    
    if query_traj_indices is None:
        query_traj_indices = available_for_query[:min(num_query_trajs, len(available_for_query))]
    
    # Extract prompt trajectories
    prompt_trajs = {
        'observations': [trajectories['observations'][i] for i in prompt_traj_indices],
        'actions': [trajectories['actions'][i] for i in prompt_traj_indices],
    }
    
    # Extract query trajectories
    query_trajs = {
        'observations': [trajectories['observations'][i] for i in query_traj_indices],
        'actions': [trajectories['actions'][i] for i in query_traj_indices],
    }
    
    return prompt_trajs, query_trajs, prompt_traj_indices, query_traj_indices


def preprocess_observations(obs_dict: Dict, transforms=None) -> Dict:
    """
    Preprocess observations to match expected format.
    Resize RGB to 224x224.
    """
    import torchvision.transforms as T
    
    if transforms is None:
        transforms = T.Compose([
            T.Resize((224, 224), antialias=True),
        ])
    
    processed = {}
    
    # Process RGB
    rgb = obs_dict['rgb'].float()  # (T, num_cams, 128, 128, 3)
    # Permute to (T, num_cams, 3, 128, 128) for transform
    rgb = rgb.permute(0, 1, 4, 2, 3)
    T_rgb, num_cams, C, H, W = rgb.shape
    # Reshape to apply transform
    rgb = rgb.reshape(T_rgb * num_cams, C, H, W)
    rgb = transforms(rgb)
    rgb = rgb.reshape(T_rgb, num_cams, C, 224, 224)
    processed['rgb'] = rgb
    
    processed['state'] = obs_dict['state']
    
    return processed


def evaluate_with_prompts(
    agent,
    prompt_trajs: Dict,
    query_trajs: Dict,
    eval_kwargs: Dict,
    transforms=None,
):
    """
    Evaluate agent using in-context learning.
    Use prompt trajectories as demonstrations and evaluate on query trajectories.
    
    Args:
        agent: The policy model
        prompt_trajs: Dictionary with prompt observations and actions
        query_trajs: Dictionary with query observations and actions
        eval_kwargs: Dictionary with evaluation hyperparameters
        transforms: Image transform function
    
    Returns:
        Dictionary with evaluation metrics (action predictions vs ground truth)
    """
    import torchvision.transforms as T
    
    stats, num_queries, temporal_agg, max_timesteps, device, sim_backend = eval_kwargs.values()
    
    if transforms is None:
        transforms = T.Compose([
            T.Resize((224, 224), antialias=True),
        ])
    
    agent.eval()
    eval_metrics = defaultdict(list)
    
    # Preprocess prompt trajectories
    prompt_obs_processed = []
    for obs_dict in prompt_trajs['observations']:
        obs_proc = preprocess_observations(obs_dict, transforms)
        # Move to device
        obs_proc = {k: common.to_tensor(v, device) for k, v in obs_proc.items()}
        prompt_obs_processed.append(obs_proc)
    
    prompt_actions = [common.to_tensor(a, device) for a in prompt_trajs['actions']]
    
    with torch.no_grad():
        # Iterate over query trajectories
        for query_idx, (query_obs_dict, query_actions) in enumerate(
            zip(query_trajs['observations'], query_trajs['actions'])
        ):
            # Preprocess query trajectory
            query_obs_proc = preprocess_observations(query_obs_dict, transforms)
            query_obs_proc = {k: common.to_tensor(v, device) for k, v in query_obs_proc.items()}
            query_actions = common.to_tensor(query_actions, device)
            
            # For each timestep in the query trajectory
            T_query = query_obs_proc['state'].shape[0]
            
            for ts in range(T_query - num_queries + 1):  # Ensure we have num_queries steps ahead
                # Extract observation at timestep ts
                obs_at_ts = {k: v[ts:ts+1] for k, v in query_obs_proc.items()}  # Add batch dim
                
                # Get ground truth actions
                actions_gt = query_actions[ts:ts+num_queries]  # (num_queries, act_dim)
                
                # Normalize RGB
                obs_at_ts['rgb'] = obs_at_ts['rgb'].float() / 255.0
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                obs_at_ts['rgb'] = normalize(obs_at_ts['rgb'])
                
                # Get action prediction from agent
                action_pred = agent.get_action(obs_at_ts)  # (1, num_queries, act_dim)
                action_pred = action_pred.squeeze(0)
                ur5_action=convert_maniskill_to_ur5(action_pred,
                                sequence_format=True,
                                return_dict=True,
                                action_scale=1.0)
                  # (num_queries, act_dim)
                print(len(ur5_action))
                
                # Compute L1 error
                l1_error = torch.abs(action_pred - actions_gt).mean().cpu().numpy()
                eval_metrics['l1_error'].append(l1_error)
                
                # Compute per-action-dim error
                for dim in range(actions_gt.shape[-1]):
                    l1_dim = torch.abs(action_pred[:, dim] - actions_gt[:, dim]).mean().cpu().numpy()
                    eval_metrics[f'l1_error_dim_{dim}'].append(l1_dim)
    
    agent.train()
    
    # Aggregate metrics
    for k in eval_metrics.keys():
        eval_metrics[k] = np.array(eval_metrics[k])
    
    return eval_metrics


def evaluate(n: int, agent, eval_envs, eval_kwargs):
    """
    Original evaluation function (kept for compatibility).
    """
    stats, num_queries, temporal_agg, max_timesteps, device, sim_backend = eval_kwargs.values()

    use_visual_obs = isinstance(eval_envs.single_observation_space.sample(), dict)
    delta_control = not stats
    if not delta_control:
        if sim_backend == "physx_cpu":
            pre_process = lambda s_obs: (s_obs - stats['state_mean'].cpu().numpy()) / stats['state_std'].cpu().numpy()
        else:
            pre_process = lambda s_obs: (s_obs - stats['state_mean']) / stats['state_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # create action table for temporal ensembling
    action_dim = eval_envs.action_space.shape[-1]
    num_envs = eval_envs.num_envs
    if temporal_agg:
        query_frequency = 1
        all_time_actions = torch.zeros([num_envs, max_timesteps, max_timesteps+num_queries, action_dim], device=device)
    else:
        query_frequency = num_queries
        actions_to_take = torch.zeros([num_envs, num_queries, action_dim], device=device)

    agent.eval()

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        ts, eps_count = 0, 0

        EP_PRINT_FREQ = 5  # print every 5 episodes

        while eps_count < n:
            # ------------------ preprocess obs ------------------
            if use_visual_obs:
                obs["state"] = pre_process(obs["state"]) if not delta_control else obs["state"]
                obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            else:
                obs = pre_process(obs) if not delta_control else obs
                obs = common.to_tensor(obs, device)

            # ------------------ query policy ------------------
            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)
                print(action_seq)
                print("*"*25)  # (num_envs, num_queries, act_dim)

            if temporal_agg:
                assert query_frequency == 1
                all_time_actions[:, ts, ts:ts + num_queries] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]

                actions_populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                actions_populated[max(0, ts + 1 - num_queries): ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, actions_populated]

                k = 0.01
                if ts < num_queries:
                    exp_weights = torch.exp(
                        -k * torch.arange(len(actions_for_curr_step[0]), device=device)
                    )
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = exp_weights[None, :, None].repeat(num_envs, 1, 1)

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            action = post_process(raw_action) if not delta_control else raw_action
            if sim_backend == "physx_cpu":
                action = action.cpu().numpy()

            # ------------------ env step ------------------
            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1

            # ------------------ episode end ------------------
            if truncated.any():
                assert truncated.all(), "All envs must truncate together"

                if isinstance(info["final_info"], dict):
                    ep_info = info["final_info"]["episode"]
                    for k, v in ep_info.items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                eps_count += num_envs
                ts = 0
                all_time_actions = torch.zeros(
                    [num_envs, max_timesteps, max_timesteps + num_queries, action_dim],
                    device=device,
                )

                # ------------------ print stats every 5 episodes ------------------
                if (eps_count // num_envs) % EP_PRINT_FREQ == 0:
                    print(f"\n📊 Eval stats after {eps_count} episodes:")
                    for k, v in eval_metrics.items():
                        last_vals = np.array(v[-EP_PRINT_FREQ:])
                        print(f"  {k}: {last_vals.mean():.4f}")

    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics