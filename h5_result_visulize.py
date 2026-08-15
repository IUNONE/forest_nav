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

from navdiffusion import NavDiffsionLightning


class H5ResultVisualizer:
    
    def __init__(self, h5_path: str, config_path: str, ckpt_dir: str):
        self.h5_path = h5_path
        self.current_idx = 0
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.transform_config = self.config['data_params'].get(
            'transform', self.config['data_params'].get('tranform', {})
        )
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
    
    def load_image(self, idx: int):
        """Load and normalize the current RGB observation."""
        token = self.group_keys[idx]
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            image = f[token]['image'][:].astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        image = torch.as_tensor(np.transpose(image, (2, 0, 1)))
        return self._transform(image)
    
    def _transform(self, image):
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
            return transform(image)
        return image
    
    def load_sample(self, idx: int):
        """Load data from h5 file for given index"""
        group_key = self.group_keys[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            group = f[group_key]
            image = group['image'][:]
            future_waypoint_local = group['future_waypoint_local'][:]
        return group_key, image, future_waypoint_local
    
    def predict_trajectory(self, idx: int):
        """Predict trajectory using the model"""
        start_time = time.time()
        
        image = self.load_image(idx).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_traj = self.model.predict((image, None))
        
        inference_time = time.time() - start_time
        
        return pred_traj.cpu().numpy()[0], inference_time  # [len_traj_pred, 2], time
    
    def update_display(self):
        """Update the visualization"""
        # Load current sample
        group_key, image, future_waypoint = self.load_sample(self.current_idx)
        
        # Predict trajectory
        print(f"\nPredicting trajectory for frame {self.current_idx + 1}/{len(self.group_keys)}...")
        pred_traj, inference_time = self.predict_trajectory(self.current_idx)
        print(f"Inference time: {inference_time*1000:.2f} ms")
        
        # Clear axes
        self.ax_img.clear()
        self.ax_traj.clear()
        
        # Display RGB image
        rgb_image = image[:, :, :3]
        if rgb_image.dtype != np.uint8 and rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0
        self.ax_img.imshow(rgb_image)
        
        # Projection is intentionally omitted: the bags do not contain the
        # body-to-camera extrinsic required to project body-frame trajectories.
        self.ax_img.set_title('RGB Image')
        self.ax_img.axis('off')
        
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
        
        all_x = np.concatenate(([0.0], gt_x, pred_x))
        all_y = np.concatenate(([0.0], gt_y, pred_y))
        margin = 0.5
        self.ax_traj.set_xlim(all_x.min() - margin, all_x.max() + margin)
        self.ax_traj.set_ylim(all_y.min() - margin, all_y.max() + margin)
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
    parser.add_argument('--ckpt_dir', type=str, default='results/resnet50_spatial',
                       help='Directory containing checkpoint files')
    
    args = parser.parse_args()
    
    visualizer = H5ResultVisualizer(args.h5_path, args.config_path, args.ckpt_dir)


if __name__ == "__main__":
    main()
