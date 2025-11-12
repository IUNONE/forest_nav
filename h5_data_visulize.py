import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from argparse import ArgumentParser

from navdiffusion.data.transform_utils import project_to_img


class H5DataVisualizer:
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.current_idx = 0
        
        # Load h5 file and get all group keys
        with h5py.File(h5_path, 'r') as f:
            self.group_keys = sorted(list(f.keys()))
        
        if len(self.group_keys) == 0:
            raise ValueError(f"No groups found in {h5_path}")
        
        print(f"Loaded {len(self.group_keys)} groups from {h5_path}")
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title('H5 Data Visualizer')
        
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
    
    def load_sample(self, idx: int):
        """Load data from h5 file for given index"""
        group_key = self.group_keys[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            group = f[group_key]
            image = group['image'][:]
            future_waypoint_local = group['future_waypoint_local'][:]
            goal_local = group['goal_local'][:]
        
        # Replace inf values in depth channel with max non-inf value
        depth_channel = image[:, :, 3]
        inf_mask = np.isinf(depth_channel)
        if inf_mask.any():
            max_valid_depth = depth_channel[~inf_mask].max() if (~inf_mask).any() else 50.0
            image[:, :, 3][inf_mask] = max_valid_depth
        
        return group_key, image, future_waypoint_local, goal_local
    
    def update_display(self):
        """Update the visualization"""
        # Load current sample
        group_key, image, future_waypoint, goal = self.load_sample(self.current_idx)
        
        # Depth channel statistics (output for every frame)
        depth_channel = image[:, :, 3]
        print(f"\n[Frame {self.current_idx + 1}/{len(self.group_keys)}] Depth stats:")
        print(f"  min: {depth_channel.min():.2f}m, max: {depth_channel.max():.2f}m")
        print(f"  mean: {depth_channel.mean():.2f}m, std: {depth_channel.std():.2f}m")
        
        # Clear axes
        self.ax_img.clear()
        self.ax_traj.clear()
        
        # Display RGB image (first 3 channels)
        rgb_image = image[:, :, :3]
        self.ax_img.imshow(rgb_image)
        
        # Project future waypoints to image
        img_coords = project_to_img(future_waypoint[:, :2], debug=(self.current_idx == 0))
        valid_mask = ~np.isnan(img_coords[:, 0])
        valid_coords = img_coords[valid_mask]
        
        # Debug output for first frame
        if self.current_idx == 0:
            print(f"\nDebug info for first frame:")
            print(f"Future waypoints shape: {future_waypoint.shape}")
            print(f"First 3 waypoints (local): {future_waypoint[:3, :2]}")
            print(f"Projected coords shape: {img_coords.shape}")
            print(f"Valid points: {valid_mask.sum()}/{len(valid_mask)}")
            if len(valid_coords) > 0:
                print(f"Valid coords range - u: [{valid_coords[:, 0].min():.1f}, {valid_coords[:, 0].max():.1f}]")
                print(f"Valid coords range - v: [{valid_coords[:, 1].min():.1f}, {valid_coords[:, 1].max():.1f}]")
        
        # Draw projected waypoints on image
        if len(valid_coords) > 0:
            self.ax_img.scatter(valid_coords[:, 0], valid_coords[:, 1], 
                              c='lime', s=50, marker='o', edgecolors='white', 
                              linewidths=1.5, zorder=5, alpha=0.8)
            if self.current_idx == 0:
                print(f"Drew {len(valid_coords)} waypoints on image")
        else:
            if self.current_idx == 0:
                print("No valid waypoints to draw!")
        
        self.ax_img.set_title('RGB Image with Projected Waypoints')
        self.ax_img.axis('off')
        
        # Plot trajectory
        self.ax_traj.set_title('Trajectory (Local Frame)')
        self.ax_traj.set_xlabel('X (m)')
        self.ax_traj.set_ylabel('Y (m)')
        self.ax_traj.grid(True, alpha=0.3)
        
        # Current position at origin
        self.ax_traj.scatter(0, 0, c='blue', s=100, marker='o', label='Current', zorder=3)
        
        # Future waypoints
        waypoint_x = future_waypoint[:, 0]
        waypoint_y = future_waypoint[:, 1]
        self.ax_traj.plot(waypoint_x, waypoint_y, 'g-', alpha=0.6, linewidth=2, label='Future Path')
        self.ax_traj.scatter(waypoint_x, waypoint_y, c='green', s=20, marker='o', zorder=2)
        
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
    parser = ArgumentParser(description='Visualize H5 navigation dataset')
    parser.add_argument('--h5_path', type=str, required=True, 
                       help='Path to converted h5 file')
    
    args = parser.parse_args()
    
    visualizer = H5DataVisualizer(args.h5_path)


if __name__ == "__main__":
    main()
