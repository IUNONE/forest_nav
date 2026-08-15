import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from argparse import ArgumentParser

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
        return group_key, image, future_waypoint_local
    
    def update_display(self):
        """Update the visualization"""
        # Load current sample
        group_key, image, future_waypoint = self.load_sample(self.current_idx)
        
        # Clear axes
        self.ax_img.clear()
        self.ax_traj.clear()
        
        # Display RGB image (first 3 channels)
        rgb_image = image[:, :, :3]
        if rgb_image.dtype != np.uint8 and rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0
        self.ax_img.imshow(rgb_image)
        
        # The bag does not contain the body-to-camera extrinsic, so projecting
        # body-frame waypoints onto the RGB image would be misleading.
        if self.current_idx == 0:
            print(f"\nDebug info for first frame:")
            print(f"Future waypoints shape: {future_waypoint.shape}")
            print(f"First 3 waypoints (local): {future_waypoint[:3, :2]}")

        self.ax_img.set_title('RGB Image')
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
        
        # Keep the limits useful for a 3.2 second trajectory while including
        # any lateral or reverse motion in the selected sample.
        all_x = np.concatenate(([0.0], waypoint_x))
        all_y = np.concatenate(([0.0], waypoint_y))
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
    parser = ArgumentParser(description='Visualize H5 navigation dataset')
    parser.add_argument('--h5_path', type=str, required=True, 
                       help='Path to converted h5 file')
    
    args = parser.parse_args()
    
    visualizer = H5DataVisualizer(args.h5_path)


if __name__ == "__main__":
    main()
