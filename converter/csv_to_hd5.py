"""
Convert UR5 robot trajectory data (CSV + MP4) to ManiSkill-compatible H5 format.

This script:
1. Extracts robot state from CSV files
2. Extracts RGB frames from MP4 videos
3. Uses PyBullet to compute IK for UR5 → Panda joint mapping
4. Generates actions in delta_pos format
5. Appends trajectories to existing ManiSkill H5 file

Usage:
    python ur5_to_maniskill_converter.py \
        --input-dir /path/to/good_cycles \
        --original-h5 /path/to/trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5 \
        --output-h5 /path/to/robot_stack_cube.h5
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

import numpy as np
import pandas as pd
import cv2
import h5py
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYBULLET IK SOLVER FOR UR5 → PANDA MAPPING
# ============================================================================

class PyBulletIKSolver:
    """
    Use PyBullet to solve inverse kinematics for UR5 and Panda robots.
    Maps UR5 EEF poses to Panda joint configurations.
    """
    
    def __init__(self, use_gui: bool = False):
        """
        Initialize PyBullet physics engine.
        
        Args:
            use_gui: If True, enable GUI visualization (slow)
        """
        self.use_gui = use_gui
        
        # Connect to PyBullet
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load Panda robot
        # Panda URDF is typically available in pybullet_data
        try:
            self.panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0])
            logger.info("Successfully loaded Panda URDF from pybullet_data")
        except Exception as e:
            logger.warning(f"Could not load Panda URDF from pybullet_data: {e}")
            logger.warning("Will proceed with simple joint angle mapping instead of full IK")
            self.panda = None
        
        # Get Panda info
        if self.panda is not None:
            self.num_joints = p.getNumJoints(self.panda)
            self.end_effector_link_index = 11  # Typically panda_hand
            
            # Get joint info for movable joints (exclude fixed joints)
            self.movable_joints = []
            for i in range(self.num_joints):
                info = p.getJointInfo(self.panda, i)
                if info[2] != p.JOINT_FIXED:  # JOINT_FIXED = 4
                    self.movable_joints.append(i)
            
            logger.info(f"Panda has {len(self.movable_joints)} movable joints")
    
    def solve_ik_panda(
        self, 
        tcp_pos: np.ndarray, 
        tcp_quat: np.ndarray,
        tolerance: float = 1e-4,
        max_iterations: int = 100
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for Panda robot given TCP pose.
        
        Args:
            tcp_pos: Target position [x, y, z]
            tcp_quat: Target orientation as quaternion [w, x, y, z]
            tolerance: IK solution tolerance
            max_iterations: Max iterations for IK solver
        
        Returns:
            Joint angles [q1, q2, q3, q4, q5, q6, q7] or None if IK fails
        """
        if self.panda is None:
            return self._simple_joint_mapping(tcp_pos, tcp_quat)
        
        try:
            # PyBullet IK expects quat as [x, y, z, w]
            quat_xyzw = [tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]]
            
            joint_angles = p.calculateInverseKinematics(
                self.panda,
                self.end_effector_link_index,
                tcp_pos,
                quat_xyzw,
                maxNumIterations=max_iterations,
                residualThreshold=tolerance
            )
            
            # Return only arm joints (first 7, excluding gripper fingers)
            return np.array(joint_angles[:7], dtype=np.float32)
        
        except Exception as e:
            logger.debug(f"IK failed for pos={tcp_pos}, quat={tcp_quat}: {e}")
            return None
    
    def _simple_joint_mapping(
        self, 
        tcp_pos: np.ndarray, 
        tcp_quat: np.ndarray
    ) -> np.ndarray:
        """
        Fallback: Simple mapping of TCP pose to joint angles.
        Used when PyBullet IK is not available.
        
        This is a heuristic approach:
        - Map XYZ position to first 3 joints
        - Map quaternion to joints 4-7
        """
        qpos = np.zeros(7, dtype=np.float32)
        
        # Simple heuristic: map position to joint angles
        # Normalize and scale position
        qpos[0] = np.arctan2(tcp_pos[1], tcp_pos[0])  # Base rotation
        qpos[1] = -np.linalg.norm(tcp_pos) / 2.0  # Shoulder joint
        qpos[2] = tcp_pos[2] / 2.0  # Elbow joint
        
        # Map quaternion to wrist joints
        euler = R.from_quat([tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]]).as_euler('xyz')
        qpos[4] = euler[0]
        qpos[5] = euler[1]
        qpos[6] = euler[2]
        
        return qpos
    
    def close(self):
        """Close PyBullet connection."""
        if self.panda is not None:
            p.disconnect(self.client)


# ============================================================================
# VIDEO & DATA PROCESSING
# ============================================================================

class VideoFrameExtractor:
    """Extract and process frames from MP4 video files."""
    
    # Cropping coordinates for UR5 videos
    CROP_BOX = {
        'x1': 337,
        'y1': 54,
        'x2': 575,
        'y2': 473
    }
    
    @staticmethod
    def extract_frames(
        episode_dir: str,
        target_size: Tuple[int, int] = (128, 128),
        max_frames: Optional[int] = None,
        crop_box: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Extract all frames from video and resize to target size.
        
        Tries to load 'cropped.mp4' first, falls back to any other MP4 in directory.
        If fallback MP4 is used, applies cropping to all frames.
        
        Args:
            episode_dir: Directory containing MP4 file(s)
            target_size: Target (height, width) after resize
            max_frames: Maximum frames to extract (None = all)
            crop_box: Dict with 'x1', 'y1', 'x2', 'y2' for cropping (optional)
        
        Returns:
            Frames array of shape (N, H, W, 3) with dtype uint8
        
        Raises:
            RuntimeError if no video can be read or no MP4 files found
        """
        if crop_box is None:
            crop_box = VideoFrameExtractor.CROP_BOX
        
        episode_path = Path(episode_dir)
        
        # Find all MP4 files
        mp4_files = list(episode_path.glob('*.mp4'))
        
        if not mp4_files:
            raise RuntimeError(f"No MP4 files found in {episode_dir}")
        
        logger.info(f"Found {len(mp4_files)} MP4 file(s): {[f.name for f in mp4_files]}")
        
        # Try cropped.mp4 first
        cropped_mp4 = episode_path / 'cropped.mp4'
        use_cropping = False
        video_path = None
        
        if cropped_mp4.exists():
            video_path = cropped_mp4
            logger.info(f"Using cropped.mp4: {cropped_mp4}")
        else:
            # Fallback to first available MP4
            video_path = mp4_files[0]
            use_cropping = True
            logger.warning(f"cropped.mp4 not found, using fallback: {video_path.name}")
            logger.warning(f"Will apply cropping: x1={crop_box['x1']}, y1={crop_box['y1']}, "
                          f"x2={crop_box['x2']}, y2={crop_box['y2']}")
        
        frames = []
        frame_count = 0
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply cropping if needed
                if use_cropping:
                    frame = frame[
                        crop_box['y1']:crop_box['y2'],
                        crop_box['x1']:crop_box['x2']
                    ]
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                frame_resized = cv2.resize(
                    frame_rgb, 
                    target_size, 
                    interpolation=cv2.INTER_LINEAR
                )
                
                frames.append(frame_resized)
                frame_count += 1
                
                if max_frames is not None and frame_count >= max_frames:
                    break
        
        finally:
            cap.release()
        
        if not frames:
            raise RuntimeError(f"No frames extracted from {video_path}")
        
        frames_array = np.array(frames, dtype=np.uint8)
        logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
        logger.info(f"Frame shape: {frames_array.shape}")
        
        return frames_array


class CSVDataProcessor:
    """Process robot state data from CSV files."""
    
    @staticmethod
    def load_csv(csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with shape {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    @staticmethod
    def extract_tcp_pose(df: pd.DataFrame) -> np.ndarray:
        """
        Extract TCP (end-effector) pose from CSV data.
        
        Args:
            df: DataFrame with columns:
                - robot0_base_to_eef_pos_x/y/z
                - robot0_base_to_eef_quat_w/x/y/z
        
        Returns:
            TCP poses array of shape (N, 7) with dtype float32
            Format: [x, y, z, qw, qx, qy, qz]
        """
        pos_cols = ['robot0_base_to_eef_pos_x', 'robot0_base_to_eef_pos_y', 'robot0_base_to_eef_pos_z']
        quat_cols = ['robot0_base_to_eef_quat_w', 'robot0_base_to_eef_quat_x', 
                     'robot0_base_to_eef_quat_y', 'robot0_base_to_eef_quat_z']
        
        tcp_pos = df[pos_cols].values.astype(np.float32)
        tcp_quat = df[quat_cols].values.astype(np.float32)
        
        # Normalize quaternions
        tcp_quat = tcp_quat / np.linalg.norm(tcp_quat, axis=1, keepdims=True)
        
        tcp_pose = np.concatenate([tcp_pos, tcp_quat], axis=1)
        
        logger.info(f"Extracted TCP poses: shape {tcp_pose.shape}")
        
        return tcp_pose
    
    @staticmethod
    def extract_gripper_state(df: pd.DataFrame) -> np.ndarray:
        """
        Extract gripper joint position from CSV.
        
        Returns:
            Gripper states array of shape (N,) with dtype float32
        """
        gripper = df['robot0_gripper_qpos'].values.astype(np.float32)
        logger.info(f"Extracted gripper states: shape {gripper.shape}")
        return gripper


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

class TrajectoryGenerator:
    """Generate ManiSkill-compatible trajectory data."""
    
    def __init__(self, ik_solver: PyBulletIKSolver):
        """
        Args:
            ik_solver: PyBulletIKSolver instance for IK computation
        """
        self.ik_solver = ik_solver
    
    def generate_trajectory(
        self,
        tcp_poses: np.ndarray,
        gripper_states: np.ndarray,
        rgb_frames: np.ndarray,
        episode_name: str = "unknown",
        dt: float = 0.02
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete trajectory data for ManiSkill format.
        
        Args:
            tcp_poses: Shape (N, 7) - TCP poses [x,y,z,qw,qx,qy,qz]
            gripper_states: Shape (N,) - Gripper joint positions
            rgb_frames: Shape (N, H, W, 3) - RGB observations
            episode_name: Name of episode for logging
            dt: Time step for velocity computation
        
        Returns:
            Dictionary with all trajectory components:
                - actions: (L, 4) float32
                - qpos: (L+1, 9) float32
                - qvel: (L+1, 9) float32
                - tcp_pose: (L+1, 7) float32
                - rgb: (L+1, 128, 128, 3) uint8
                - (placeholder fields for compatibility)
        """
        n_frames = len(tcp_poses)
        n_actions = n_frames - 1
        
        logger.info(f"Generating trajectory for {episode_name}: {n_frames} frames")
        
        # =====================================================================
        # 1. Compute joint angles from TCP poses using IK
        # =====================================================================
        logger.info("Computing IK for all TCP poses...")
        qpos_arm = []  # 7D arm joint angles
        
        for i, tcp_pose in enumerate(tqdm(tcp_poses, desc="IK solving")):
            pos = tcp_pose[:3]
            quat = tcp_pose[3:]  # [qw, qx, qy, qz]
            
            joint_angles = self.ik_solver.solve_ik_panda(pos, quat)
            
            if joint_angles is None:
                logger.warning(f"IK failed at frame {i}, using previous solution or zero")
                if i > 0:
                    joint_angles = qpos_arm[-1].copy()
                else:
                    joint_angles = np.zeros(7, dtype=np.float32)
            
            qpos_arm.append(joint_angles)
        
        qpos_arm = np.array(qpos_arm, dtype=np.float32)  # (N, 7)
        
        # =====================================================================
        # 2. Convert gripper state to 2D (two fingers) for Panda
        # =====================================================================
        # UR5 has 1D gripper, Panda has 2D (two fingers)
        # Simple mapping: duplicate gripper state
        gripper_2d = np.zeros((n_frames, 2), dtype=np.float32)
        gripper_2d[:, 0] = gripper_states
        gripper_2d[:, 1] = gripper_states
        
        # =====================================================================
        # 3. Combine to 9D qpos (7 arm + 2 gripper)
        # =====================================================================
        qpos = np.concatenate([qpos_arm, gripper_2d], axis=1)  # (N, 9)
        assert qpos.shape == (n_frames, 9), f"qpos shape mismatch: {qpos.shape}"
        
        logger.info(f"qpos shape: {qpos.shape}")
        
        # =====================================================================
        # 4. Compute joint velocities from finite differences
        # =====================================================================
        qvel = np.zeros_like(qpos)  # (N, 9)
        qvel[:-1] = (qpos[1:] - qpos[:-1]) / dt
        qvel[-1] = qvel[-2]  # Replicate last velocity
        
        logger.info(f"qvel shape: {qvel.shape}")
        
        # =====================================================================
        # 5. Compute actions (delta_pos format)
        # =====================================================================
        # Actions are 4D: [delta_x, delta_y, delta_z, delta_gripper]
        actions = np.zeros((n_actions, 4), dtype=np.float32)
        
        # Delta position
        actions[:, :3] = tcp_poses[1:, :3] - tcp_poses[:-1, :3]
        
        # Delta gripper
        actions[:, 3] = gripper_states[1:] - gripper_states[:-1]
        
        logger.info(f"actions shape: {actions.shape}")
        logger.info(f"actions stats - pos: mean={actions[:, :3].mean(axis=0)}, "
                   f"std={actions[:, :3].std(axis=0)}")
        logger.info(f"actions stats - gripper: mean={actions[:, 3].mean()}, "
                   f"std={actions[:, 3].std()}")
        
        # =====================================================================
        # 6. Verify RGB frame count and resize if needed
        # =====================================================================
        if rgb_frames.shape[0] != n_frames:
            logger.warning(f"RGB frame count ({rgb_frames.shape[0]}) != "
                          f"TCP pose count ({n_frames}). Resampling...")
            rgb_frames = self._resample_frames(rgb_frames, n_frames)
        
        assert rgb_frames.shape == (n_frames, 128, 128, 3), \
            f"RGB shape mismatch: {rgb_frames.shape}"
        
        logger.info(f"RGB shape: {rgb_frames.shape}")
        
        # =====================================================================
        # 7. Create trajectory dictionary
        # =====================================================================
        trajectory = {
            'actions': actions,
            'qpos': qpos,
            'qvel': qvel,
            'tcp_pose': tcp_poses,
            'rgb': rgb_frames,
            'gripper_states': gripper_states,
        }
        
        logger.info(f"Trajectory generation complete for {episode_name}")
        
        return trajectory
    
    @staticmethod
    def _resample_frames(
        frames: np.ndarray, 
        target_count: int
    ) -> np.ndarray:
        """
        Resample frames to target count using linear interpolation.
        
        Args:
            frames: Original frames (N, H, W, 3)
            target_count: Target number of frames
        
        Returns:
            Resampled frames (target_count, H, W, 3)
        """
        n_frames, h, w, c = frames.shape
        
        # Use cv2.remap for fast frame interpolation
        resampled = []
        indices = np.linspace(0, n_frames - 1, target_count)
        
        for idx in indices:
            lower_idx = int(np.floor(idx))
            upper_idx = int(np.ceil(idx))
            alpha = idx - lower_idx
            
            if lower_idx == upper_idx:
                frame = frames[lower_idx]
            else:
                frame = (1 - alpha) * frames[lower_idx] + alpha * frames[upper_idx]
                frame = frame.astype(np.uint8)
            
            resampled.append(frame)
        
        return np.array(resampled, dtype=np.uint8)


# ============================================================================
# PLACEHOLDER DATA GENERATION
# ============================================================================

class PlaceholderGenerator:
    """Generate placeholder data for fields not available from UR5."""
    
    @staticmethod
    def create_placeholder_actor_states(n_frames: int) -> np.ndarray:
        """
        Create placeholder actor states (cubes).
        
        Returns:
            Shape (n_frames, 13) - [pos_x,y,z, quat_w,x,y,z, vel_x,y,z, ang_vel_z, ...]
        """
        # 13D: 3D position + 4D quat + 3D linear velocity + 3D angular velocity
        return np.zeros((n_frames, 13), dtype=np.float32)
    
    @staticmethod
    def create_placeholder_robot_state(n_frames: int) -> np.ndarray:
        """Create placeholder full robot state."""
        # 31D: Panda articulation state (joints + velocities + forces)
        return np.zeros((n_frames, 31), dtype=np.float32)
    
    @staticmethod
    def create_camera_intrinsics(
        h: int = 128,
        w: int = 128,
        fx: float = 500.0,
        fy: float = 500.0
    ) -> np.ndarray:
        """
        Create camera intrinsic matrix.
        
        Returns:
            Shape (3, 3) float32
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        cx = w / 2.0
        cy = h / 2.0
        
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsic
    
    @staticmethod
    def create_camera_extrinsic(n_frames: int) -> np.ndarray:
        """
        Create camera extrinsic matrix (3x4).
        
        Returns:
            Shape (n_frames, 3, 4) float32
            [R | t] format: first 3x3 is rotation, last 3x1 is translation
        """
        extrinsic = np.zeros((n_frames, 3, 4), dtype=np.float32)
        extrinsic[:, :3, :3] = np.eye(3)
        return extrinsic
    
    @staticmethod
    def create_camera_pose_matrix(n_frames: int) -> np.ndarray:
        """
        Create camera pose matrix (4x4 homogeneous transformation).
        
        Returns:
            Shape (n_frames, 4, 4) float32
        """
        pose = np.zeros((n_frames, 4, 4), dtype=np.float32)
        pose[:, :3, :3] = np.eye(3)
        pose[:, 3, 3] = 1.0
        return pose
    
    @staticmethod
    def create_trajectory_flags(n_actions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create trajectory terminal flags.
        
        Args:
            n_actions: Number of actions (n_frames - 1)
        
        Returns:
            (success, terminated, truncated) - all shape (n_actions,) bool
        """
        success = np.ones(n_actions, dtype=bool)
        terminated = np.zeros(n_actions, dtype=bool)
        truncated = np.zeros(n_actions, dtype=bool)
        
        return success, terminated, truncated


# ============================================================================
# H5 FILE OPERATIONS
# ============================================================================

class H5FileManager:
    """Manage H5 file operations: read, write, append."""
    
    @staticmethod
    def get_original_trajectory_count(h5_path: str) -> int:
        """Get number of trajectories in original H5 file."""
        try:
            with h5py.File(h5_path, 'r') as f:
                traj_ids = [int(k.replace('traj_', '')) 
                           for k in f.keys() if k.startswith('traj_')]
                if traj_ids:
                    return max(traj_ids) + 1
                return 0
        except Exception as e:
            logger.error(f"Error reading original H5: {e}")
            return 0
    
    @staticmethod
    def copy_h5_file(src: str, dst: str) -> None:
        """Copy H5 file from src to dst."""
        import shutil
        shutil.copy2(src, dst)
        logger.info(f"Copied H5 file: {src} → {dst}")
    
    @staticmethod
    def append_trajectory_to_h5(
        h5_path: str,
        traj_id: int,
        trajectory: Dict[str, np.ndarray],
        episode_name: str = "unknown",
        compression: bool = False
    ) -> None:
        """
        Append a new trajectory to H5 file.
        
        Args:
            h5_path: Path to H5 file (must exist)
            traj_id: Trajectory ID (e.g., 999 for traj_999)
            trajectory: Dictionary with trajectory data
            episode_name: Name for logging
            compression: If True, use gzip compression
        """
        traj_key = f'traj_{traj_id}'
        n_frames = trajectory['qpos'].shape[0]
        n_actions = trajectory['actions'].shape[0]
        
        logger.info(f"Writing trajectory {traj_key} ({episode_name}) to {h5_path}")
        
        kwargs = {'compression': 'gzip', 'compression_opts': 4} if compression else {}
        
        with h5py.File(h5_path, 'a') as f:
            # Create main trajectory group
            traj_group = f.create_group(traj_key)
            
            # ================================================================
            # Actions
            # ================================================================
            traj_group.create_dataset(
                'actions',
                data=trajectory['actions'],
                dtype=np.float32,
                **kwargs
            )
            
            # ================================================================
            # Observations
            # ================================================================
            obs_group = traj_group.create_group('obs')
            
            # Agent state
            agent_group = obs_group.create_group('agent')
            agent_group.create_dataset('qpos', data=trajectory['qpos'], dtype=np.float32, **kwargs)
            agent_group.create_dataset('qvel', data=trajectory['qvel'], dtype=np.float32, **kwargs)
            
            # Extra observations
            extra_group = obs_group.create_group('extra')
            extra_group.create_dataset('tcp_pose', data=trajectory['tcp_pose'], 
                                      dtype=np.float32, **kwargs)
            
            # Sensor data (RGB only, no depth as per requirements)
            sensor_data_group = obs_group.create_group('sensor_data')
            
            base_cam_group = sensor_data_group.create_group('base_camera')
            base_cam_group.create_dataset('rgb', data=trajectory['rgb'], dtype=np.uint8, **kwargs)
            
            hand_cam_group = sensor_data_group.create_group('hand_camera')
            hand_cam_group.create_dataset('rgb', data=trajectory['rgb'], dtype=np.uint8, **kwargs)
            
            # Sensor parameters
            sensor_param_group = obs_group.create_group('sensor_param')
            
            intrinsic = PlaceholderGenerator.create_camera_intrinsics()
            extrinsic = PlaceholderGenerator.create_camera_extrinsic(n_frames)
            pose_matrix = PlaceholderGenerator.create_camera_pose_matrix(n_frames)
            
            base_param = sensor_param_group.create_group('base_camera')
            base_param.create_dataset('intrinsic_cv', data=np.tile(intrinsic[None], (n_frames, 1, 1)),
                                     dtype=np.float32, **kwargs)
            base_param.create_dataset('extrinsic_cv', data=extrinsic, dtype=np.float32, **kwargs)
            base_param.create_dataset('cam2world_gl', data=pose_matrix, dtype=np.float32, **kwargs)
            
            hand_param = sensor_param_group.create_group('hand_camera')
            hand_param.create_dataset('intrinsic_cv', data=np.tile(intrinsic[None], (n_frames, 1, 1)),
                                     dtype=np.float32, **kwargs)
            hand_param.create_dataset('extrinsic_cv', data=extrinsic, dtype=np.float32, **kwargs)
            hand_param.create_dataset('cam2world_gl', data=pose_matrix, dtype=np.float32, **kwargs)
            
            # ================================================================
            # Environment states (placeholder for cube dynamics)
            # ================================================================
            env_group = traj_group.create_group('env_states')
            
            actors_group = env_group.create_group('actors')
            cubeA = PlaceholderGenerator.create_placeholder_actor_states(n_frames)
            cubeB = PlaceholderGenerator.create_placeholder_actor_states(n_frames)
            table = PlaceholderGenerator.create_placeholder_actor_states(n_frames)
            
            actors_group.create_dataset('cubeA', data=cubeA, dtype=np.float32, **kwargs)
            actors_group.create_dataset('cubeB', data=cubeB, dtype=np.float32, **kwargs)
            actors_group.create_dataset('table-workspace', data=table, dtype=np.float32, **kwargs)
            
            articulations_group = env_group.create_group('articulations')
            panda_state = PlaceholderGenerator.create_placeholder_robot_state(n_frames)
            articulations_group.create_dataset('panda_wristcam', data=panda_state, 
                                             dtype=np.float32, **kwargs)
            
            # ================================================================
            # Terminal flags
            # ================================================================
            success, terminated, truncated = PlaceholderGenerator.create_trajectory_flags(n_actions)
            
            traj_group.create_dataset('success', data=success, dtype=bool, **kwargs)
            traj_group.create_dataset('terminated', data=terminated, dtype=bool, **kwargs)
            traj_group.create_dataset('truncated', data=truncated, dtype=bool, **kwargs)
            
            logger.info(f"Successfully wrote {traj_key} to H5 file")
    
    @staticmethod
    def validate_trajectory_in_h5(h5_path: str, traj_id: int) -> bool:
        """
        Validate that a trajectory was correctly written.
        
        Returns:
            True if all expected keys and shapes are correct
        """
        traj_key = f'traj_{traj_id}'
        
        try:
            with h5py.File(h5_path, 'r') as f:
                if traj_key not in f:
                    logger.error(f"Trajectory {traj_key} not found in H5")
                    return False
                
                traj = f[traj_key]
                
                # Check required datasets
                required_keys = [
                    'actions',
                    'obs/agent/qpos',
                    'obs/agent/qvel',
                    'obs/extra/tcp_pose',
                    'obs/sensor_data/base_camera/rgb',
                    'obs/sensor_data/hand_camera/rgb',
                    'success',
                    'terminated',
                    'truncated'
                ]
                
                for key in required_keys:
                    if key not in traj:
                        logger.error(f"Missing key in {traj_key}: {key}")
                        return False
                
                # Check shapes
                n_actions = traj['actions'].shape[0]
                n_frames = n_actions + 1
                
                checks = {
                    'actions': (n_actions, 4),
                    'obs/agent/qpos': (n_frames, 9),
                    'obs/agent/qvel': (n_frames, 9),
                    'obs/extra/tcp_pose': (n_frames, 7),
                    'obs/sensor_data/base_camera/rgb': (n_frames, 128, 128, 3),
                    'obs/sensor_data/hand_camera/rgb': (n_frames, 128, 128, 3),
                    'success': (n_actions,),
                    'terminated': (n_actions,),
                    'truncated': (n_actions,),
                }
                
                for key, expected_shape in checks.items():
                    actual_shape = traj[key].shape
                    if actual_shape != expected_shape:
                        logger.error(f"Shape mismatch for {key}: "
                                   f"expected {expected_shape}, got {actual_shape}")
                        return False
                
                logger.info(f"Trajectory {traj_key} validation passed")
                return True
        
        except Exception as e:
            logger.error(f"Error validating trajectory: {e}")
            return False


# ============================================================================
# MAIN CONVERTER
# ============================================================================

class UR5ToManiSkillConverter:
    """Main converter orchestrator."""
    
    def __init__(self, input_dir: str, original_h5: str, output_h5: str):
        """
        Args:
            input_dir: Directory containing episode_* subdirectories
            original_h5: Path to original ManiSkill H5 file
            output_h5: Path to output H5 file
        """
        self.input_dir = Path(input_dir)
        self.original_h5 = Path(original_h5)
        self.output_h5 = Path(output_h5)
        
        # Validate input directory
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if not self.original_h5.exists():
            raise FileNotFoundError(f"Original H5 file not found: {original_h5}")
        
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Original H5: {self.original_h5}")
        logger.info(f"Output H5: {self.output_h5}")
    
    def find_episodes(self) -> List[Path]:
        """Find all episode directories."""
        episodes = sorted([
            d for d in self.input_dir.iterdir() 
            if d.is_dir() and d.name.startswith('episode_')
        ])
        
        logger.info(f"Found {len(episodes)} episodes")
        for ep in episodes:
            logger.info(f"  - {ep.name}")
        
        return episodes
    
    def process_episode(
        self,
        episode_dir: Path,
        ik_solver: PyBulletIKSolver,
        traj_generator: TrajectoryGenerator
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single episode directory.
        
        Returns:
            Trajectory dictionary or None if processing failed
        """
        episode_name = episode_dir.name
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {episode_name}")
        logger.info(f"{'='*70}")
        
        try:
            # Find CSV files
            csv_files = list(episode_dir.glob('*.csv'))
            
            if not csv_files:
                logger.error(f"No CSV files found in {episode_dir}")
                return None
            
            csv_path = csv_files[0]
            logger.info(f"CSV: {csv_path.name}")
            
            # Load CSV data
            df = CSVDataProcessor.load_csv(str(csv_path))
            tcp_poses = CSVDataProcessor.extract_tcp_pose(df)
            gripper_states = CSVDataProcessor.extract_gripper_state(df)
            
            # Extract video frames (handles MP4 loading with fallback and auto-cropping)
            rgb_frames = VideoFrameExtractor.extract_frames(
                str(episode_dir),
                target_size=(128, 128)
            )
            
            # Generate trajectory
            trajectory = traj_generator.generate_trajectory(
                tcp_poses=tcp_poses,
                gripper_states=gripper_states,
                rgb_frames=rgb_frames,
                episode_name=episode_name
            )
            
            logger.info(f"✓ Successfully processed {episode_name}")
            
            return trajectory
        
        except Exception as e:
            logger.error(f"✗ Error processing {episode_name}: {e}", exc_info=True)
            return None
    
    def convert(self, use_pybullet_gui: bool = False, compression: bool = False) -> None:
        """
        Execute the full conversion pipeline.
        
        Args:
            use_pybullet_gui: If True, enable PyBullet GUI visualization (slow)
            compression: If True, use gzip compression in H5 file
        """
        logger.info("\n" + "="*70)
        logger.info("UR5 TO MANISKILL CONVERTER")
        logger.info("="*70)
        
        # Initialize IK solver
        logger.info("Initializing PyBullet IK solver...")
        ik_solver = PyBulletIKSolver(use_gui=use_pybullet_gui)
        
        # Initialize trajectory generator
        traj_generator = TrajectoryGenerator(ik_solver)
        
        # Find episodes
        episodes = self.find_episodes()
        if not episodes:
            logger.error("No episodes found!")
            return
        
        # Get next trajectory ID from original H5
        start_traj_id = H5FileManager.get_original_trajectory_count(str(self.original_h5))
        logger.info(f"Starting trajectory ID: {start_traj_id}")
        
        # Copy original H5 to output location
        H5FileManager.copy_h5_file(str(self.original_h5), str(self.output_h5))
        
        # Process episodes
        successful_episodes = 0
        failed_episodes = []
        
        for i, episode_dir in enumerate(episodes):
            trajectory = self.process_episode(
                episode_dir,
                ik_solver,
                traj_generator
            )
            
            if trajectory is not None:
                traj_id = start_traj_id + successful_episodes
                
                # Append to H5
                H5FileManager.append_trajectory_to_h5(
                    str(self.output_h5),
                    traj_id,
                    trajectory,
                    episode_name=episode_dir.name,
                    compression=compression
                )
                
                # Validate
                if H5FileManager.validate_trajectory_in_h5(str(self.output_h5), traj_id):
                    successful_episodes += 1
                    logger.info(f"✓ Trajectory {traj_id} validated")
                else:
                    logger.error(f"✗ Trajectory {traj_id} validation failed")
                    failed_episodes.append(episode_dir.name)
            else:
                failed_episodes.append(episode_dir.name)
        
        # Close IK solver
        ik_solver.close()
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*70)
        logger.info(f"Successful conversions: {successful_episodes}/{len(episodes)}")
        logger.info(f"Failed conversions: {len(failed_episodes)}")
        
        if failed_episodes:
            logger.info("Failed episodes:")
            for ep in failed_episodes:
                logger.info(f"  - {ep}")
        
        logger.info(f"Output H5: {self.output_h5}")
        logger.info(f"Output file size: {self.output_h5.stat().st_size / 1e9:.2f} GB")
        logger.info("="*70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert UR5 robot trajectory data to ManiSkill H5 format"
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing episode_* subdirectories'
    )
    
    parser.add_argument(
        '--original-h5',
        type=str,
        required=True,
        help='Path to original ManiSkill H5 file'
    )
    
    parser.add_argument(
        '--output-h5',
        type=str,
        required=True,
        help='Path to output H5 file'
    )
    
    parser.add_argument(
        '--use-gui',
        action='store_true',
        help='Enable PyBullet GUI visualization (slow)'
    )
    
    parser.add_argument(
        '--compression',
        action='store_true',
        help='Use gzip compression in H5 file'
    )
    
    args = parser.parse_args()
    
    try:
        converter = UR5ToManiSkillConverter(
            input_dir=args.input_dir,
            original_h5=args.original_h5,
            output_h5=args.output_h5
        )
        
        converter.convert(
            use_pybullet_gui=args.use_gui,
            compression=args.compression
        )
        
        logger.info("✓ Conversion completed successfully!")
    
    except Exception as e:
        logger.error(f"✗ Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()