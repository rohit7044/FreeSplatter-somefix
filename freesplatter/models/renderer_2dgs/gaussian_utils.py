"""
Gaussian Splatting.
Partially borrowed from https://github.com/graphdeco-inria/gaussian-splatting.
"""


import os
import torch
from torch import nn
import numpy as np
from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.w, view.h
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


class Camera(nn.Module):
    def __init__(self, C2W, fxfycxcy, h, w):
        """
        C2W: 4x4 camera-to-world matrix; opencv convention
        fxfycxcy: 4
        """
        super().__init__()
        self.C2W = C2W.float()
        self.W2C = self.C2W.inverse()

        self.znear = 0.01
        self.zfar = 100.0
        self.h = h
        self.w = w

        fx, fy, cx, cy = fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]
        self.tanfovX = 1 / (2 * fx)
        self.tanfovY = 1 / (2 * fy)
        self.fovX = 2 * torch.atan(self.tanfovX)
        self.fovY = 2 * torch.atan(self.tanfovY)
        self.shiftX = 2 * cx - 1
        self.shiftY = 2 * cy - 1

        def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX, shiftY):
            tanHalfFovY = torch.tan((fovY / 2))
            tanHalfFovX = torch.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4, dtype=torch.float32, device=fovX.device)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left) + shiftX
            P[1, 2] = (top + bottom) / (top - bottom) + shiftY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.fovX, fovY=self.fovY, shiftX=self.shiftX, shiftY=self.shiftY
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.C2W[:3, 3]


class GaussianModel:
    def setup_functions(self, scaling_activation_type='sigmoid', scale_min_act=0.001, scale_max_act=0.3, scale_multi_act=0.1):

        if scaling_activation_type == 'exp':
            self.scaling_activation = torch.exp
        elif scaling_activation_type == 'softplus':
            self.scaling_activation = torch.nn.functional.softplus
            self.scale_multi_act = scale_multi_act
        elif scaling_activation_type == 'sigmoid':
            self.scale_min_act = scale_min_act
            self.scale_max_act = scale_max_act
            self.scaling_activation = torch.sigmoid
        else:
            raise NotImplementedError
        self.scaling_activation_type = scaling_activation_type

        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid
        self.feature_activation = torch.sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    def __init__(self, sh_degree: int, scaling_activation_type='exp', scale_min_act=0.001, scale_max_act=0.3, scale_multi_act=0.1):
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        if self.sh_degree > 0:
            self._features_rest = torch.empty(0)
        else:
            self._features_rest = None
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions(scaling_activation_type=scaling_activation_type, scale_min_act=scale_min_act, scale_max_act=scale_max_act, scale_multi_act=scale_multi_act)

    def set_data(self, xyz, features, scaling, rotation, opacity):
        self._xyz = xyz
        self._features_dc = features[:, 0, :].contiguous() if self.sh_degree == 0 else features[:, 0:1, :].contiguous()
        if self.sh_degree > 0:
            self._features_rest = features[:, 1:, :].contiguous()
        else:
            self._features_rest = None
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        return self

    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        if self.sh_degree > 0:
            self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        return self

    @property
    def get_scaling(self):
        if self.scaling_activation_type == 'exp':
            scales = self.scaling_activation(self._scaling)
        elif self.scaling_activation_type == 'softplus':
            scales = self.scaling_activation(self._scaling) * self.scale_multi_act
        elif self.scaling_activation_type == 'sigmoid':
            scales = self.scale_min_act + (self.scale_max_act - self.scale_min_act) * self.scaling_activation(self._scaling)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self.sh_degree > 0:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return self.feature_activation(self._features_dc)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def construct_list_of_attributes(self, num_rest=0):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(num_rest):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply_vis(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyzs = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()

        scales = torch.log(self.get_scaling)
        scales = scales.detach().cpu().numpy()

        rot_mat_vis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        xyzs = xyzs @ rot_mat_vis.T
        rotations = self._rotation.detach().cpu().numpy()
        rotations = R.from_quat(rotations[:, [1,2,3,0]]).as_matrix()
        rotations = rot_mat_vis @ rotations
        rotations = R.from_matrix(rotations).as_quat()[:, [3,0,1,2]]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(0)]
        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyzs = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        if self.sh_degree > 0:
            f_rest = self._features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            f_rest = np.zeros((f_dc.shape[0], 0), dtype=f_dc.dtype)
        opacities = self._opacity.detach().cpu().numpy()

        scales = torch.log(self.get_scaling)
        scales = scales.detach().cpu().numpy()

        rotations = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(f_rest.shape[-1])]
        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, f_rest, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    # def load_ply(self, path):
    #     plydata = PlyData.read(path)

    #     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                     np.asarray(plydata.elements[0]["y"]),
    #                     np.asarray(plydata.elements[0]["z"])),  axis=1)
    #     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    #     features_dc = np.zeros((xyz.shape[0], 3, 1))
    #     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    #     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    #     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    #     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    #     scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    #     scales = np.zeros((xyz.shape[0], len(scale_names)))
    #     for idx, attr_name in enumerate(scale_names):
    #         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    #     rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    #     rots = np.zeros((xyz.shape[0], len(rot_names)))
    #     for idx, attr_name in enumerate(rot_names):
    #         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     self._xyz = torch.from_numpy(xyz.astype(np.float32))
    #     self._features_dc = torch.from_numpy(features_dc.astype(np.float32)).transpose(1, 2).contiguous()
    #     self._opacity = torch.from_numpy(opacities.astype(np.float32)).contiguous()
    #     self._scaling = torch.from_numpy(scales.astype(np.float32)).contiguous()
    #     self._rotation = torch.from_numpy(rots.astype(np.float32)).contiguous()


def render(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    bg_color=(1.0, 1.0, 1.0),
    scaling_modifier=1.0,
):
    """
    Render the scene.
    """
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    viewpoint_camera = Camera(C2W=C2W, fxfycxcy=fxfycxcy, h=height, w=width)

    bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.h),
        image_width=int(viewpoint_camera.w),
        tanfovx=viewpoint_camera.tanfovX,
        tanfovy=viewpoint_camera.tanfovY,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    rendered_image, _, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None if pc.sh_degree == 0 else shs,
        colors_precomp=shs if pc.sh_degree == 0 else None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    depth_ratio = 0.0
    surf_depth = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    return {
        "render": rendered_image,
        "depth": surf_depth,
        "alpha": render_alpha,
        'surf_normal': surf_normal,
        'rend_normal': render_normal,
        'dist': render_dist,
    }
