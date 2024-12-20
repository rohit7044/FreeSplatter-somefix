import torch
import numpy as np


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def blender_to_opencv(camera_matrix: torch.Tensor):
    """
    Convert Blender World-to-Camera matrix into OpenCV space by flipping y and z axes
    Blender camera system: x-right, y-up, z-backward
    OpenCV camera system: x-right, y-down, z-forward
    """
    flip_yz = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if camera_matrix.ndim == 3:
        flip_yz = flip_yz.unsqueeze(0)
    camera_matrix_opencv = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_opencv


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def create_camera_to_world(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None, camera_system: str = 'opencv'):
    """
    Create OpenCV or OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    assert camera_system in ['opencv', 'opengl']
    if camera_system == 'opencv':
        # OpenCV camera: z-forward, x-right, y-down
        z_axis = look_at - camera_position
        z_axis = normalize_vecs(z_axis).float()
        x_axis = torch.cross(z_axis, up_world)
        x_axis = normalize_vecs(x_axis).float()
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = normalize_vecs(y_axis).float()
    else:
        # OpenGL camera: z-backward, x-right, y-up
        z_axis = camera_position - look_at
        z_axis = normalize_vecs(z_axis).float()
        x_axis = torch.cross(up_world, z_axis)
        x_axis = normalize_vecs(x_axis).float()
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = normalize_vecs(y_axis).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def normalize_cameras(extrinsics, camera_position: torch.Tensor = None, camera_system: str = 'opencv', canonical_index=0):
    """
    Normalize the first camera to the canonical camera position, and transform other cameras accordingly.

    extrinsics: (N, 4, 4)
    """
    if camera_position is None:
        camera_position = torch.tensor([[0, -2, 0]]).float()
    assert camera_system in ['opencv', 'opengl']

    canonical_distance = camera_position.norm()

    # compute conditional camera distances
    cond_extrinsic = extrinsics[canonical_index]
    # cond_extrinsic = extrinsics[0]
    cond_camera_distance = cond_extrinsic[:3, 3].norm(dim=-1, keepdim=False)

    # scale camera distances
    scale = canonical_distance / cond_camera_distance
    extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * scale
    
    # rotate all cameras
    canonical_extrinsic = create_camera_to_world(camera_position, camera_system=camera_system).to(extrinsics)
    # transform_matrix = torch.matmul(canonical_extrinsic, torch.linalg.inv(extrinsics[0:1]))
    transform_matrix = torch.matmul(canonical_extrinsic, torch.linalg.inv(extrinsics[canonical_index:canonical_index+1]))
    normalized_extrinsics = torch.matmul(transform_matrix, extrinsics)

    return normalized_extrinsics, scale