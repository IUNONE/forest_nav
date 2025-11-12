import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import h5py

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NavigationDataset(Dataset):
    
    def __init__(self, 
        data_config: Dict[str, Any], 
        model_config, 
        split: str
    ):
        self.split = split
        self.h5_file_path = data_config['h5_file_path']
               
        self.transform = data_config["tranform"]
        self.time_horizon = data_config["time_horizon"]

        self.len_traj_pred = model_config['len_traj_pred']

        # Load all group keys from h5 file as tokens
        with h5py.File(self.h5_file_path, 'r') as f:
            self.tokens = sorted(list(f.keys()))
        
        print(f"Loaded {len(self.tokens)} samples from {self.h5_file_path}")
        
    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors
                imgs (torch.Tensor): [T, 4, H, W] rgbd
                goal (torch.Tensor): [2] containing the goal position x,y in current frame local coordinate
                future_wpts (torch.Tensor): [n_future, 2] containing the waypoints in future
        """
        # Get current token and extract dataset_name
        current_token = self.tokens[i]
        dataset_name = current_token.rsplit('_', 1)[0]
        
        # Find all tokens from the same scenario
        scenario_indices = [idx for idx, t in enumerate(self.tokens) if t.startswith(dataset_name + '_')]
        current_idx_in_scenario = scenario_indices.index(i)
        
        # Get history frames within the same scenario
        start_idx_in_scenario = max(0, current_idx_in_scenario - (self.time_horizon - 1))
        history_scenario_indices = scenario_indices[start_idx_in_scenario:current_idx_in_scenario + 1]
        
        # Padding with first frame of the scenario if not enough
        if len(history_scenario_indices) < self.time_horizon:
            padding = [scenario_indices[0]] * (self.time_horizon - len(history_scenario_indices))
            history_scenario_indices = padding + history_scenario_indices
        
        imgs_list = []
        for idx in history_scenario_indices:
            token = self.tokens[idx]
            with h5py.File(self.h5_file_path, 'r', swmr=True) as f:
                image = f[token]['image'][:]
                
                if idx == i:
                    future_waypoint_local = f[token]['future_waypoint_local'][:]
                    goal_local = f[token]['goal_local'][:]
            
            depth_channel = image[:, :, 3]
            image[:, :, 3] = np.where(np.isinf(depth_channel), 50.0, depth_channel) / 50.0
            
            imgs = np.transpose(image, (2, 0, 1))
            imgs_list.append(torch.as_tensor(imgs, dtype=torch.float32))
        
        imgs_transformed = self._transform(imgs_list)
        imgs = torch.stack(imgs_transformed, dim=0)
        
        goal = goal_local[:2]
        future_wpts = future_waypoint_local[:self.len_traj_pred, :2]

        return (
            imgs,
            torch.as_tensor(goal, dtype=torch.float32),
            torch.as_tensor(future_wpts, dtype=torch.float32),    
        )
    
    def _transform(self, obs_image):
        transform_list = []
        
        # 1. Resize (if configured)
        if 'resize' in self.transform:
            w, h = self.transform['resize']['w_h']
            # transforms.Resize expects (height, width)
            transform_list.append(transforms.Resize((h, w)))
        
        # 2. Normalize (if configured)
        if 'norm' in self.transform:
            transform_list.append(
                transforms.Normalize(
                    self.transform['norm']['mean'], 
                    self.transform['norm']['std']
                )
            )
        
        if len(transform_list) > 0:
            transform = transforms.Compose(transform_list)
            # obs_images list of [C, H, W] * N
            return [transform(img) for img in obs_image]
        else:
            return obs_image
