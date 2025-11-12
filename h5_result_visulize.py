import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from argparse import ArgumentParser
from glob import glob
import yaml
import torch
from torchvision import transforms
import time

from navdiffusion.data.transform_utils import project_to_img
from navdiffusion import NavDiffsionLightning


class H5ResultVisualizer:
    
    def __init__(self, h5_path: str, config_path: str, ckpt_dir: str):
        self.h5_path = h5_path
        self.current_idx = 0
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.time_horizon = self.config['data_params']['time_horizon']
        self.transform_config = self.config['data_params']['tranform']
        self.len_traj_pred = self.config['model_params']['len_traj_pred']
        
        # Load h5 file and get all group keys
        with h5py.File(h5_path, 'r') as f:
            self.group_keys = sorted(list(f.keys()))
        
        if len(self.group_keys) == 0:
            raise ValueError(f"No groups found in {h5_path}")
        
        print(f"Loaded {len(self.group_keys)} groups from {h5_path}")
        
        # Load model checkpoint
        ckpt_files = glob(f"{ckpt_dir}/*.ckpt")
        if not ckpt_files:
            raise ValueError(f"No checkpoint files found in {ckpt_dir}")
        latest_ckpt = sorted(ckpt_files)[-1]
        print(f"Loading checkpoint: {latest_ckpt}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NavDiffsionLightning.load_from_checkpoint(latest_ckpt)
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title('H5 Result Visualizer')
        
        # Left subplot for image
        self.ax_img = plt.subplot(1, 2, 1)
        self.ax_img.set_title('RGB Image')
        self.ax_img.axis('off')
        
        # Right subplot for trajectory
        self.ax_traj = plt.subplot(1, 2, 2)
        self.ax_traj.set_title('Trajectory')
        self.ax_traj.set_xlabel('X (m)')
        self.ax_traj.set_ylabel('Y (m)')
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.set_aspect('equal')
        
        # Create buttons
        ax_prev = plt.axes([0.35, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.55, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)
        
        # Display first sample
        self.update_display()
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.show()
    
    def load_history_images(self, idx: int):
        """Load history images for temporal input following dataset.py logic"""
        start_idx = max(0, idx - (self.time_horizon - 1))
        history_indices = list(range(start_idx, idx + 1))
        
        if len(history_indices) < self.time_horizon:
            padding = [history_indices[0]] * (self.time_horizon - len(history_indices))
            history_indices = padding + history_indices
        
        imgs_list = []
        for i in history_indices:
            token = self.group_keys[i]
            with h5py.File(self.h5_path, 'r', swmr=True) as f:
                image = f[token]['image'][:]
            
            # Process depth channel
            depth_channel = image[:, :, 3]
            image[:, :, 3] = np.where(np.isinf(depth_channel), 50.0, depth_channel) / 50.0
            
            # Transpose to [C, H, W] and convert to tensor
            imgs = np.transpose(image, (2, 0, 1))
            imgs_list.append(torch.as_tensor(imgs, dtype=torch.float32))
        
        # Apply transforms
        imgs_transformed = self._transform(imgs_list)
        imgs = torch.stack(imgs_transformed, dim=0)  # [T, 4, H, W]
        
        return imgs
    
    def _transform(self, obs_image):
        """Apply transformations following dataset.py logic"""
        transform_list = []
        
        # Resize
        if 'resize' in self.transform_config:
            w, h = self.transform_config['resize']['w_h']
            transform_list.append(transforms.Resize((h, w)))
        
        # Normalize
        if 'norm' in self.transform_config:
            transform_list.append(
                transforms.Normalize(
                    self.transform_config['norm']['mean'], 
                    self.transform_config['norm']['std']
                )
            )
        
        if len(transform_list) > 0:
            transform = transforms.Compose(transform_list)
            return [transform(img) for img in obs_image]
        else:
            return obs_image
    
    def load_sample(self, idx: int):
        """Load data from h5 file for given index"""
        group_key = self.group_keys[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            group = f[group_key]
            image = group['image'][:]
            future_waypoint_local = group['future_waypoint_local'][:]
            goal_local = group['goal_local'][:]
        
        # Replace inf values in depth channel
        depth_channel = image[:, :, 3]
        inf_mask = np.isinf(depth_channel)
        if inf_mask.any():
            max_valid_depth = depth_channel[~inf_mask].max() if (~inf_mask).any() else 50.0
            image[:, :, 3][inf_mask] = max_valid_depth
        
        return group_key, image, future_waypoint_local, goal_local
    
    def predict_trajectory(self, idx: int):
        """Predict trajectory using the model"""
        start_time = time.time()
        
        # Load history images
        imgs = self.load_history_images(idx)  # [T, 4, H, W]
        
        # Load current goal
        group_key = self.group_keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            goal_local = f[group_key]['goal_local'][:]
        
        goal = torch.as_tensor(goal_local[:2], dtype=torch.float32)
        
        # Add batch dimension
        imgs = imgs.unsqueeze(0).to(self.device)  # [1, T, 4, H, W]
        goal = goal.unsqueeze(0).to(self.device)  # [1, 2]
        
        # Predict
        with torch.no_grad():
            pred_traj = self.model.predict((imgs, goal, None))
        
        inference_time = time.time() - start_time
        
        return pred_traj.cpu().numpy()[0], inference_time  # [len_traj_pred, 2], time
    
    def update_display(self):
        """Update the visualization"""
        # Load current sample
        group_key, image, future_waypoint, goal = self.load_sample(self.current_idx)
        
        # Predict trajectory
        print(f"\nPredicting trajectory for frame {self.current_idx + 1}/{len(self.group_keys)}...")
        pred_traj, inference_time = self.predict_trajectory(self.current_idx)
        print(f"Inference time: {inference_time*1000:.2f} ms")
        
        # Depth channel statistics
        depth_channel = image[:, :, 3]
        print(f"[Frame {self.current_idx + 1}/{len(self.group_keys)}] Depth stats:")
        print(f"  min: {depth_channel.min():.2f}m, max: {depth_channel.max():.2f}m")
        print(f"  mean: {depth_channel.mean():.2f}m, std: {depth_channel.std():.2f}m")
        
        # Clear axes
        self.ax_img.clear()
        self.ax_traj.clear()
        
        # Display RGB image
        rgb_image = image[:, :, :3]
        self.ax_img.imshow(rgb_image)
        
        # Project predicted waypoints to image
        pred_img_coords = project_to_img(pred_traj, debug=False)
        pred_valid_mask = ~np.isnan(pred_img_coords[:, 0])
        pred_valid_coords = pred_img_coords[pred_valid_mask]
        
        # Project ground truth waypoints to image
        gt_img_coords = project_to_img(future_waypoint[:, :2], debug=False)
        gt_valid_mask = ~np.isnan(gt_img_coords[:, 0])
        gt_valid_coords = gt_img_coords[gt_valid_mask]
        
        # Draw predicted waypoints on image (orange)
        if len(pred_valid_coords) > 0:
            self.ax_img.scatter(pred_valid_coords[:, 0], pred_valid_coords[:, 1], 
                              c='orange', s=50, marker='o', edgecolors='white', 
                              linewidths=1.5, zorder=6, alpha=0.8, label='Predicted')
        
        # Draw ground truth waypoints on image (lime)
        if len(gt_valid_coords) > 0:
            self.ax_img.scatter(gt_valid_coords[:, 0], gt_valid_coords[:, 1], 
                              c='lime', s=50, marker='o', edgecolors='white', 
                              linewidths=1.5, zorder=5, alpha=0.8, label='Ground Truth')
        
        self.ax_img.set_title('RGB Image with Projected Waypoints')
        self.ax_img.axis('off')
        self.ax_img.legend(loc='upper right')
        
        # Plot trajectory
        self.ax_traj.set_title('Trajectory Comparison (Local Frame)')
        self.ax_traj.set_xlabel('X (m)')
        self.ax_traj.set_ylabel('Y (m)')
        self.ax_traj.grid(True, alpha=0.3)
        
        # Current position at origin
        self.ax_traj.scatter(0, 0, c='blue', s=100, marker='o', label='Current', zorder=3)
        
        # Ground truth future waypoints (green)
        gt_x = future_waypoint[:, 0]
        gt_y = future_waypoint[:, 1]
        self.ax_traj.plot(gt_x, gt_y, 'g-', alpha=0.6, linewidth=2, label='GT Path')
        self.ax_traj.scatter(gt_x, gt_y, c='green', s=30, marker='o', zorder=2)
        
        # Predicted trajectory (orange)
        pred_x = pred_traj[:, 0]
        pred_y = pred_traj[:, 1]
        self.ax_traj.plot(pred_x, pred_y, 'orange', alpha=0.6, linewidth=2, 
                         linestyle='--', label='Predicted Path')
        self.ax_traj.scatter(pred_x, pred_y, c='orange', s=30, marker='^', zorder=2)
        
        # Goal position (star marker)
        self.ax_traj.scatter(goal[0], goal[1], c='red', s=300, marker='*', 
                            label='Goal', zorder=4, edgecolors='darkred', linewidths=1.5)
        
        # Set fixed axis limits
        self.ax_traj.set_xlim(0, 30)
        self.ax_traj.set_ylim(-15, 15)
        self.ax_traj.set_aspect('equal')
        self.ax_traj.legend(loc='upper right')
        
        # Update main title
        self.fig.suptitle(f'Group: {group_key}  [{self.current_idx + 1}/{len(self.group_keys)}]', 
                         fontsize=12, fontweight='bold')
        
        self.fig.canvas.draw()
    
    def next_sample(self, event):
        """Show next sample"""
        if self.current_idx < len(self.group_keys) - 1:
            self.current_idx += 1
            self.update_display()
    
    def prev_sample(self, event):
        """Show previous sample"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()


def main():
    parser = ArgumentParser(description='Visualize H5 navigation results with model predictions')
    parser.add_argument('--h5_path', type=str, required=True, 
                       help='Path to converted h5 file')
    parser.add_argument('--config_path', type=str, default='config/lightning.yaml',
                       help='Path to config file')
    parser.add_argument('--ckpt_dir', type=str, default='results',
                       help='Directory containing checkpoint files')
    
    args = parser.parse_args()
    
    visualizer = H5ResultVisualizer(args.h5_path, args.config_path, args.ckpt_dir)


if __name__ == "__main__":
    main()
