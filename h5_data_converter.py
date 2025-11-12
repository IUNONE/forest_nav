import h5py
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from navdiffusion.data.transform_utils import to_local_coords


def convert_h5_data(h5_path: str, dataset_name: str, n_future: int):
    """
    Convert h5 data to navigation task format
    
    Args:
        h5_path: input h5 file path
        dataset_name: dataset name for group key prefix
        n_future: number of future waypoints
    """
    input_path = Path(h5_path)
    output_path = input_path.parent / f"{input_path.stem}_converted.h5"
    
    # Check existing groups
    existing_keys = set()
    if output_path.exists():
        with h5py.File(output_path, 'r') as f:
            existing_keys = set(f.keys())
        print(f"Found existing file with {len(existing_keys)} groups")
    
    # Read input data
    with h5py.File(h5_path, 'r') as f:
        rgb_images = f['rgb_images'][:]
        depth_images = f['depth_images'][:]
        robot_positions = f['robot_positions'][:]
        robot_heading = f['robot_heading'][:]
        timestamps = f['timestamps'][:]
        steps = f['steps'][:]
    
    n_steps = len(steps)
    new_groups = 0
    skipped_groups = 0
    
    # Open output file in append mode
    with h5py.File(output_path, 'a') as out_f:
        for i in tqdm(range(n_steps), desc="Converting"):
            # Generate group key
            group_key = f"{dataset_name}_{timestamps[i]:.3f}"
            
            # Skip if already exists
            if group_key in existing_keys:
                skipped_groups += 1
                continue
            
            # Create group
            group = out_f.create_group(group_key)
            
            # 1. Process image (480, 640, 4)
            rgb_norm = rgb_images[i].astype(np.float32) / 255.0
            
            # Process depth: replace inf values with max non-inf value
            depth = depth_images[i].copy()
            inf_mask = np.isinf(depth)
            if inf_mask.any():
                max_valid_depth = depth[~inf_mask].max() if (~inf_mask).any() else 50.0
                depth[inf_mask] = max_valid_depth
            
            depth_expanded = depth[..., None]
            image = np.concatenate([rgb_norm, depth_expanded], axis=-1)
            
            # 2. Process future_waypoint_local (n_future, 3)
            end_idx = min(i + 1 + n_future, n_steps)
            future_positions = robot_positions[i+1:end_idx, :2]
            future_yaws = robot_heading[i+1:end_idx]
            
            # Padding with last point if not enough
            if len(future_positions) < n_future:
                last_pos = robot_positions[-1, :2]
                last_yaw = robot_heading[-1]
                padding_count = n_future - len(future_positions)
                future_positions = np.vstack([
                    future_positions, 
                    np.tile(last_pos, (padding_count, 1))
                ])
                future_yaws = np.concatenate([
                    future_yaws, 
                    np.full(padding_count, last_yaw)
                ])
            
            # Transform to local coordinates
            curr_pos = robot_positions[i, :2]
            curr_yaw = robot_heading[i]
            local_xy = to_local_coords(future_positions, curr_pos, curr_yaw)
            relative_yaws = future_yaws - curr_yaw
            future_waypoint_local = np.column_stack([local_xy, relative_yaws])
            
            # 3. Process goal_local (3,)
            goal_pos = robot_positions[-1, :2]
            goal_yaw = robot_heading[-1]
            goal_local_xy = to_local_coords(
                goal_pos.reshape(1, 2), curr_pos, curr_yaw
            )[0]
            goal_local = np.array([
                goal_local_xy[0], 
                goal_local_xy[1], 
                goal_yaw - curr_yaw
            ])
            
            # Save to group
            group.create_dataset('image', data=image, compression='gzip')
            group.create_dataset('future_waypoint_local', data=future_waypoint_local)
            group.create_dataset('goal_local', data=goal_local)
            
            new_groups += 1
    
    print(f"Conversion completed:")
    print(f"  - New groups added: {new_groups}")
    print(f"  - Skipped (existing): {skipped_groups}")
    print(f"  - Output file: {output_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Input h5 file path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for group key prefix")
    parser.add_argument("--n_future", type=int, required=True, help="Number of future waypoints")
    
    args = parser.parse_args()
    convert_h5_data(args.h5_path, args.dataset_name, args.n_future)


if __name__ == "__main__":
    main()
