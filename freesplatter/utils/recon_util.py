import cv2
import math
import scipy
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

from .camera_util import create_camera_to_world


###############################################################################
# Camera Trajectory
###############################################################################

def fibonacci_sampling_on_sphere(num_samples=1):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def get_fibonacci_cameras(N=20, radius=2.0, device='cuda'):
    def normalize_vecs(vectors): 
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

    t = torch.linspace(0, 1, N).reshape(-1, 1)

    cam_pos = fibonacci_sampling_on_sphere(N)
    cam_pos = torch.from_numpy(cam_pos).float().to(device)
    cam_pos = cam_pos * radius

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 0, 1], dtype=torch.float,
                                        device=device).reshape(-1).expand_as(forward_vector)
    right_vector = normalize_vecs(torch.cross(forward_vector, up_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(right_vector, forward_vector, dim=-1))
    rotate = torch.stack(
                    (right_vector, -up_vector, forward_vector), dim=-1)

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = rotate

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos
    cam2world = translation_matrix @ rotation_matrix
    return cam2world


def get_circular_cameras(N=120, elevation=0, radius=2.0, normalize=True, device='cuda'):
    camera_positions = []
    for i in range(N):
        azimuth = 2 * np.pi * i / N - np.pi / 2
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    c2ws = create_camera_to_world(camera_positions, camera_system='opencv')

    if normalize:
        c2ws_first = create_camera_to_world(torch.tensor([0, -2, 0]), camera_system='opencv').unsqueeze(0)
        c2ws = torch.linalg.inv(c2ws_first) @ c2ws

    return c2ws

###############################################################################
# TSDF Fusion
###############################################################################

def rgbd_to_mesh(images, depths, c2ws, fov, mesh_path, cam_elev_thr=0):

    voxel_length = 2 * 2.0 / 512.0
    sdf_trunc = 2 * 0.02
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_type,
    )

    for i in tqdm(range(c2ws.shape[0])):
        camera_to_world = c2ws[i]
        world_to_camera = np.linalg.inv(camera_to_world)
        camera_position = camera_to_world[:3, 3]
        # camera_elevation = np.rad2deg(np.arcsin(camera_position[2]))
        camera_elevation = np.rad2deg(np.arcsin(camera_position[2] / np.linalg.norm(camera_position)))
        if camera_elevation < cam_elev_thr:
            continue
        color_image = o3d.geometry.Image(np.ascontiguousarray(images[i]))
        depth_image = o3d.geometry.Image(np.ascontiguousarray(depths[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
        )
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()

        fx = fy =  images[i].shape[1] / 2. / np.tan(np.deg2rad(fov / 2.0))
        cx = cy = images[i].shape[1] / 2.
        h =  images[i].shape[0]
        w =  images[i].shape[1]
        camera_intrinsics.set_intrinsics(
            w, h, fx, fy, cx, cy
        )
        volume.integrate(
            rgbd_image,
            camera_intrinsics,
            world_to_camera,
        )

    fused_mesh = volume.extract_triangle_mesh()

    triangle_clusters, cluster_n_triangles, cluster_area = (
            fused_mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 500
    fused_mesh.remove_triangles_by_mask(triangles_to_remove)
    fused_mesh.remove_unreferenced_vertices()

    fused_mesh = fused_mesh.filter_smooth_simple(number_of_iterations=2)
    fused_mesh = fused_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(mesh_path, fused_mesh)

###############################################################################
# Visualization
###############################################################################

def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)

###############################################################################
# Camera Estimation
###############################################################################

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def estimate_focal(pts3d, pp=None, mask=None, min_focal=0., max_focal=np.inf):
    """ 
    Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    H, W, THREE = pts3d.shape
    assert THREE == 3

    if pp is None:
        pp = torch.tensor([W/2, H/2]).to(pts3d)

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(-1, 2) - pp.view(1, 2)  # (HW, 2)
    pts3d = pts3d.view(H*W, 3).contiguous()  # (HW, 3)

    # mask points if provided
    if mask is not None:
        mask = mask.to(pts3d.device).ravel().bool()
        assert len(mask) == pts3d.shape[0]
        pts3d = pts3d[mask]
        pixels = pixels[mask]
    
    # weiszfeld
    # init focal with l2 closed form
    # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
    xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

    dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
    dot_xy_xy = xy_over_z.square().sum(dim=-1)

    focal = dot_xy_px.mean(dim=0) / dot_xy_xy.mean(dim=0)

    # iterative re-weighted least-squares
    for iter in range(10):
        # re-weighting by inverse of distance
        dis = (pixels - focal.view(-1, 1) * xy_over_z).norm(dim=-1)
        # print(dis.nanmean(-1))
        w = dis.clip(min=1e-8).reciprocal()
        # update the scaling with the new weights
        focal = (w * dot_xy_px).mean(dim=0) / (w * dot_xy_xy).mean(dim=0)

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    return focal.ravel()


def fast_pnp(pts3d, mask, focal=None, pp=None, niter_PnP=10):
    """
    Estimate camera poses and focals with RANSAC-PnP.

    Inputs:
        pts3d:  H x W x 3
        focal:  1
        mask:   H x W
        pp
    """
    H, W, _ = pts3d.shape
    pixels = np.mgrid[:W, :H].T.astype(float)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S/2, S*3, 21)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W/2, H/2)

    best = 0,
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                                                    iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    world2cam = np.eye(4).astype(float)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = T.reshape(3)
    cam2world = np.linalg.inv(world2cam)

    return best_focal, cam2world
