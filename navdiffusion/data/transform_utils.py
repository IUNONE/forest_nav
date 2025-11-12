import numpy as np


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
        the yaw direction is the 
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def project_to_img(
    points_local: np.ndarray,
    camera_pitch: float = np.radians(-11.997),
    camera_height: float = 0.3,
    focal_length: float = 500.0,
    image_width: int = 640,
    image_height: int = 480,
    debug: bool = False
) -> np.ndarray:
    """
    Project local 3D points to image coordinates
    
    Args:
        points_local: (N, 2) or (N, 3) local coordinates [x, y] or [x, y, z]
        camera_pitch: camera pitch angle in radians (downward is positive)
        camera_height: camera height above ground in meters
        focal_length: camera focal length in pixels
        image_width: image width in pixels
        image_height: image height in pixels
        debug: print debug information
    
    Returns:
        np.ndarray: (N, 2) image coordinates [u, v], or None for points behind camera
    """
    # Convert 2D to 3D if needed (assume z=0 for ground points)
    if points_local.shape[-1] == 2:
        points_3d = np.column_stack([points_local, np.zeros(len(points_local))])
    else:
        points_3d = points_local.copy()
    
    if debug:
        print(f"Input points (robot frame): {points_3d[:3]}")
    
    # Camera is at origin looking forward, apply pitch rotation
    # Rotation matrix for pitch (around y-axis)
    cos_p = np.cos(camera_pitch)
    sin_p = np.sin(camera_pitch)
    pitch_rot = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ])
    
    # Transform points: translate to camera frame
    points_translated = points_3d - np.array([0, 0, camera_height])
    
    # Robot frame (x-forward, y-left, z-up) -> Camera frame (x-right, y-down, z-forward)
    # First rotate by pitch, then convert axes
    points_rotated = points_translated @ pitch_rot.T
    
    z_cam = points_rotated[:, 0]  # robot x (forward) -> camera z (forward)
    x_cam = -points_rotated[:, 1]  # robot y (left) -> camera x (right)
    y_cam = -points_rotated[:, 2]  # robot z (up) -> camera y (down)
    
    if debug:
        print(f"Camera frame coords - x: {x_cam[:3]}, y: {y_cam[:3]}, z: {z_cam[:3]}")
    
    # Filter points behind camera
    valid_mask = z_cam > 0
    
    # Project to image plane
    cx = image_width / 2
    cy = image_height / 2
    
    u = focal_length * x_cam / z_cam + cx
    v = focal_length * y_cam / z_cam + cy
    
    if debug:
        print(f"Projected coords - u: {u[:3]}, v: {v[:3]}")
        print(f"Valid mask: {valid_mask[:10]}")
    
    # Stack and filter
    img_coords = np.column_stack([u, v])
    img_coords[~valid_mask] = np.nan
    
    return img_coords
