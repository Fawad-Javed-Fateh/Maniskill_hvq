"""
Helper functions to convert ManiSkill policy actions back to UR5 robot action format.

Supports outputs as:
- Dictionary with UR5 keys (human readable)
- Numpy arrays (efficient batch processing)
- PyTorch tensors (GPU-friendly, differentiable)
"""

import torch
import numpy as np
from typing import Dict, Union, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ManiSkillToUR5Converter:
    """Convert ManiSkill format actions to UR5 robot action dictionary."""
    
    # UR5 action keys in expected order
    UR5_ACTION_KEYS = [
        'robot0_base_pos_x', 'robot0_base_pos_y', 'robot0_base_pos_z',
        'robot0_base_quat_w', 'robot0_base_quat_x', 'robot0_base_quat_y', 'robot0_base_quat_z',
        'robot0_base_to_eef_pos_x', 'robot0_base_to_eef_pos_y', 'robot0_base_to_eef_pos_z',
        'robot0_base_to_eef_quat_w', 'robot0_base_to_eef_quat_x', 'robot0_base_to_eef_quat_y', 'robot0_base_to_eef_quat_z',
        'robot0_gripper_qpos'
    ]
    
    # Expected dimensions for each component
    BASE_POS_DIM = 3      # x, y, z
    BASE_QUAT_DIM = 4     # w, x, y, z
    EEF_POS_DIM = 3       # x, y, z
    EEF_QUAT_DIM = 4      # w, x, y, z
    GRIPPER_DIM = 1       # gripper_qpos
    
    TOTAL_DIM = BASE_POS_DIM + BASE_QUAT_DIM + EEF_POS_DIM + EEF_QUAT_DIM + GRIPPER_DIM  # 15
    
    @classmethod
    def convert_action(
        cls,
        maniskill_action: Union[torch.Tensor, np.ndarray],
        return_dict: bool = False,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        normalize: bool = False,
        action_scale: float = 1.0,
    ) -> Union[Dict[str, float], np.ndarray, torch.Tensor]:
        """
        Convert a single ManiSkill action to UR5 format.
        
        Args:
            maniskill_action: Action tensor/array of shape (4,) or (action_dim,)
                             If shape is (4,), it's expected to be delta actions for:
                             [gripper_action, ee_x, ee_y, ee_z] (common ManiSkill format)
            return_dict: If True, return dict with UR5 keys
            return_tensor: If True, return torch.Tensor; if False, return numpy array
            device: Device to place tensor on (e.g., 'cuda', 'cpu'). Ignored if return_dict=True
            normalize: If True, normalize quaternions to unit vectors
            action_scale: Scale factor for actions (useful for amplitude control)
        
        Returns:
            Dictionary with UR5 action keys, numpy array (15,), or torch tensor (15,)
        """
        # Convert to numpy if tensor
        if isinstance(maniskill_action, torch.Tensor):
            maniskill_action = maniskill_action.detach().cpu().numpy()
        
        maniskill_action = np.asarray(maniskill_action, dtype=np.float32)
        
        # Handle different action dimensions
        if maniskill_action.shape[-1] == 4:
            # Common ManiSkill format: [gripper_action, delta_ee_x, delta_ee_y, delta_ee_z]
            action_scaled = maniskill_action * action_scale
            
            ur5_action = np.zeros(cls.TOTAL_DIM, dtype=np.float32)
            
            # Base position (0:3) - typically fixed at origin for UR5 on fixed base
            ur5_action[0:3] = 0.0  # base_pos (x, y, z)
            
            # Base quaternion (3:7) - typically identity for fixed base
            ur5_action[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # base_quat (w, x, y, z)
            
            # End-effector position (7:10) - delta from action
            ur5_action[7:10] = action_scaled[1:4]  # ee_pos_xyz
            
            # End-effector quaternion (10:14) - identity (no rotation)
            ur5_action[10:14] = np.array([1.0, 0.0, 0.0, 0.0])  # ee_quat (w, x, y, z)
            
            # Gripper (14) - gripper action
            ur5_action[14] = action_scaled[0]  # gripper_qpos
            
        elif maniskill_action.shape[-1] == cls.TOTAL_DIM:
            # Already in full UR5 format
            ur5_action = maniskill_action.astype(np.float32) * action_scale
            
        else:
            raise ValueError(
                f"Expected action dimension 4 or {cls.TOTAL_DIM}, "
                f"got {maniskill_action.shape[-1]}"
            )
        
        # Normalize quaternions if requested
        if normalize:
            ur5_action = cls._normalize_quaternions(ur5_action)
        
        if return_dict:
            return cls._array_to_dict(ur5_action)
        elif return_tensor:
            if device is None:
                device = 'cpu'
            return torch.from_numpy(ur5_action).to(device)
        else:
            return ur5_action
    
    @classmethod
    def convert_action_sequence(
        cls,
        maniskill_actions: Union[torch.Tensor, np.ndarray],
        return_dict: bool = False,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        normalize: bool = False,
        action_scale: float = 1.0,
    ) -> Union[List[Dict[str, float]], np.ndarray, torch.Tensor]:
        """
        Convert a sequence of ManiSkill actions to UR5 format.
        
        Args:
            maniskill_actions: Action tensor/array of shape (T, action_dim)
                              where T is sequence length
            return_dict: If True, return list of dicts
            return_tensor: If True, return torch.Tensor; if False, return numpy array
            device: Device to place tensor on (e.g., 'cuda', 'cpu'). Ignored if return_dict=True
            normalize: If True, normalize quaternions
            action_scale: Scale factor for actions
        
        Returns:
            List of dictionaries, numpy array (T, 15), or torch tensor (T, 15)
        """
        # Store device and dtype info before converting
        input_device = None
        input_dtype = None
        if isinstance(maniskill_actions, torch.Tensor):
            input_device = maniskill_actions.device
            input_dtype = maniskill_actions.dtype
            maniskill_actions = maniskill_actions.detach().cpu().numpy()
        
        maniskill_actions = np.asarray(maniskill_actions, dtype=np.float32)
        
        if maniskill_actions.ndim != 2:
            raise ValueError(f"Expected 2D array, got {maniskill_actions.ndim}D")
        
        T = maniskill_actions.shape[0]
        
        if return_dict:
            ur5_actions = []
            for t in range(T):
                ur5_action = cls.convert_action(
                    maniskill_actions[t],
                    return_dict=True,
                    normalize=normalize,
                    action_scale=action_scale,
                )
                ur5_actions.append(ur5_action)
            return ur5_actions
        else:
            # Build as array first
            ur5_actions = np.zeros((T, cls.TOTAL_DIM), dtype=np.float32)
            for t in range(T):
                ur5_actions[t] = cls.convert_action(
                    maniskill_actions[t],
                    return_dict=False,
                    normalize=normalize,
                    action_scale=action_scale,
                )
            
            if return_tensor:
                if device is None:
                    # Use original device if input was tensor, otherwise cpu
                    device = input_device if input_device is not None else 'cpu'
                return torch.from_numpy(ur5_actions).to(device)
            else:
                return ur5_actions
    
    @classmethod
    def _array_to_dict(cls, ur5_action: np.ndarray) -> Dict[str, float]:
        """Convert action array to dictionary with UR5 keys."""
        if len(ur5_action) != cls.TOTAL_DIM:
            raise ValueError(f"Expected {cls.TOTAL_DIM} dimensions, got {len(ur5_action)}")
        
        return {
            key: float(ur5_action[i])
            for i, key in enumerate(cls.UR5_ACTION_KEYS)
        }
    
    @classmethod
    def _normalize_quaternions(cls, ur5_action: np.ndarray) -> np.ndarray:
        """Normalize quaternion components to unit length."""
        ur5_action = ur5_action.copy()
        
        # Normalize base quaternion (indices 3:7)
        base_quat = ur5_action[3:7]
        base_quat_norm = np.linalg.norm(base_quat)
        if base_quat_norm > 1e-8:
            ur5_action[3:7] = base_quat / base_quat_norm
        
        # Normalize EEF quaternion (indices 10:14)
        eef_quat = ur5_action[10:14]
        eef_quat_norm = np.linalg.norm(eef_quat)
        if eef_quat_norm > 1e-8:
            ur5_action[10:14] = eef_quat / eef_quat_norm
        
        return ur5_action
    
    @classmethod
    def dict_to_array(cls, action_dict: Dict[str, float]) -> np.ndarray:
        """Convert dictionary with UR5 keys back to array."""
        ur5_action = np.zeros(cls.TOTAL_DIM, dtype=np.float32)
        
        for i, key in enumerate(cls.UR5_ACTION_KEYS):
            if key not in action_dict:
                logger.warning(f"Missing key in action dict: {key}")
            ur5_action[i] = action_dict.get(key, 0.0)
        
        return ur5_action


def convert_maniskill_to_ur5(
    action_pred: Union[torch.Tensor, np.ndarray],
    sequence_format: bool = True,
    return_dict: bool = False,
    return_tensor: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    action_scale: float = 1.0,
) -> Union[Dict[str, float], List[Dict[str, float]], np.ndarray, torch.Tensor]:
    """
    Convenience function to convert ManiSkill actions to UR5 format.
    
    This is a wrapper around ManiSkillToUR5Converter for quick conversions.
    
    Example:
        >>> action_pred = torch.Size([30, 4])  # 30 timesteps, 4 action dims
        >>> ur5_actions = convert_maniskill_to_ur5(action_pred, sequence_format=True, return_tensor=True)
        >>> # ur5_actions is a tensor of shape (30, 15)
        
        >>> single_action = action_pred[0]  # shape (4,)
        >>> ur5_single = convert_maniskill_to_ur5(single_action, sequence_format=False, return_tensor=True)
        >>> # ur5_single is a tensor of shape (15,)
    
    Args:
        action_pred: ManiSkill action(s) as tensor or numpy array
        sequence_format: If True, treat input as sequence (T, action_dim);
                        if False, treat as single action (action_dim,)
        return_dict: If True, return dictionary format (dict output only)
        return_tensor: If True, return PyTorch tensor; if False, return numpy array
        device: Device to place tensor on (e.g., 'cuda', 'cpu'). Only used if return_tensor=True
        action_scale: Scale factor for actions
    
    Returns:
        - If sequence_format=True and return_dict=True: List of dicts
        - If sequence_format=True and return_tensor=True: torch tensor (T, 15)
        - If sequence_format=True and return_tensor=False: numpy array (T, 15)
        - If sequence_format=False and return_dict=True: single dict
        - If sequence_format=False and return_tensor=True: torch tensor (15,)
        - If sequence_format=False and return_tensor=False: numpy array (15,)
    """
    if sequence_format:
        return ManiSkillToUR5Converter.convert_action_sequence(
            action_pred, 
            return_dict=return_dict, 
            return_tensor=return_tensor,
            device=device,
            action_scale=action_scale
        )
    else:
        result = ManiSkillToUR5Converter.convert_action(
            action_pred, 
            return_dict=return_dict, 
            return_tensor=return_tensor,
            device=device,
            action_scale=action_scale
        )
        return result


# Example usage and integration
if __name__ == "__main__":
    # Example 1: Single action conversion to tensor
    print("=" * 70)
    print("Example 1: Single ManiSkill action as tensor")
    print("=" * 70)
    single_action = np.array([0.5, 0.1, -0.05, 0.02])
    ur5_single = convert_maniskill_to_ur5(single_action, sequence_format=False, return_tensor=True)
    print(f"\nInput (ManiSkill): {single_action}")
    print(f"Output (UR5 tensor): {ur5_single}")
    print(f"Shape: {ur5_single.shape}, Device: {ur5_single.device}, Dtype: {ur5_single.dtype}")
    
    # Example 2: Action sequence as tensor (most common case)
    print("\n" + "=" * 70)
    print("Example 2: Action sequence as tensor (30 timesteps)")
    print("=" * 70)
    action_seq = np.random.randn(30, 4) * 0.1
    ur5_seq_tensor = convert_maniskill_to_ur5(action_seq, sequence_format=True, return_tensor=True)
    print(f"\nInput shape: {action_seq.shape}")
    print(f"Output shape: {ur5_seq_tensor.shape}")
    print(f"Output dtype: {ur5_seq_tensor.dtype}")
    print(f"First action:\n{ur5_seq_tensor[0]}")
    
    # Example 3: GPU tensor output (if CUDA available)
    print("\n" + "=" * 70)
    print("Example 3: GPU tensor output (CUDA if available)")
    print("=" * 70)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    action_tensor = torch.randn(30, 4) * 0.1
    ur5_seq_gpu = convert_maniskill_to_ur5(action_tensor, sequence_format=True, return_tensor=True, device=device)
    print(f"\nInput device: {action_tensor.device}")
    print(f"Output shape: {ur5_seq_gpu.shape}")
    print(f"Output device: {ur5_seq_gpu.device}")
    
    # Example 4: Dict output (human-readable)
    print("\n" + "=" * 70)
    print("Example 4: Dictionary output (human-readable)")
    print("=" * 70)
    ur5_seq_dict = convert_maniskill_to_ur5(action_seq, sequence_format=True, return_dict=True)
    print(f"\nOutput: List of {len(ur5_seq_dict)} action dicts")
    print(f"\nFirst action dict:")
    for key, val in list(ur5_seq_dict[0].items())[:5]:
        print(f"  {key}: {val:.6f}")
    
    # Example 5: Numpy array output (for batch processing)
    print("\n" + "=" * 70)
    print("Example 5: Numpy array output (batch processing)")
    print("=" * 70)
    ur5_seq_array = convert_maniskill_to_ur5(action_seq, sequence_format=True, return_tensor=False)
    print(f"\nOutput shape: {ur5_seq_array.shape}")
    print(f"Output dtype: {ur5_seq_array.dtype}")
    print(f"\nExtract gripper commands: {ur5_seq_array[:, 14]}")
    
    # Example 6: Comparison of all output types
    print("\n" + "=" * 70)
    print("Example 6: Integration with your evaluation loop")
    print("=" * 70)
    print("""
    # In your evaluate_with_prompts function:
    action_pred = agent.get_action(obs_at_ts)  # (1, 30, 4)
    action_pred = action_pred.squeeze(0)  # (30, 4)
    
    # Option 1: Get as tensor (GPU-friendly, differentiable)
    ur5_actions_tensor = convert_maniskill_to_ur5(
        action_pred, 
        sequence_format=True, 
        return_tensor=True,
        device='cuda'  # or 'cpu'
    )
    # ur5_actions_tensor.shape = (30, 15)
    # Can compute gradients through this if needed
    
    # Option 2: Get as dict (human-readable)
    ur5_actions_dict = convert_maniskill_to_ur5(
        action_pred, 
        sequence_format=True, 
        return_dict=True
    )
    # List of 30 dicts with UR5 keys
    
    # Option 3: Get as numpy (fast batch processing)
    ur5_actions_array = convert_maniskill_to_ur5(
        action_pred, 
        sequence_format=True, 
        return_tensor=False
    )
    # numpy array of shape (30, 15)
    """)