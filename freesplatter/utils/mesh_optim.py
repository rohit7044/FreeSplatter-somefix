from typing import *
import numpy as np
import torch
import utils3d
import nvdiffrast.torch as dr
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import cv2
from PIL import Image
import fast_simplification


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.
    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
):
    """
    Bake texture to a mesh from multiple observations.
    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """
    vertices = torch.tensor(vertices).float().cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).float().cuda()
    observations = [torch.tensor(obs).float().cuda() for obs in observations]
    masks = [torch.tensor(m>1e-2).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).float().cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).float().cuda(), near, far) for intr in intrinsics]

    if mode == 'fast':
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend='cuda')
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)
                mask = rast['mask'][0].detach().bool() & masks[0]
            
            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # inpaint
        mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == 'opt':
        rastctx = utils3d.torch.RastContext(backend='cuda')
        observations = [observations.flip(0) for observations in observations]
        masks = [m.flip(0) for m in masks]
        _uv = []
        _uv_dr = []
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                _uv.append(rast['uv'].detach())
                _uv_dr.append(rast['uv_dr'].detach())

        texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
        
        def tv_loss(texture):
            return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
                   torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])
    
        total_steps = 2500
        with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views))
                uv, uv_dr, observation, mask = _uv[selected], _uv_dr[selected], observations[selected], masks[selected]
                render = dr.texture(texture, uv, uv_dr)[0]
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()
                # annealing
                optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )['mask'][0].detach().cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return texture


def optimize_mesh(
    mesh,
    images: torch.Tensor,
    masks: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    simplify: float = 0.95,
    texture_size: int = 1024,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.
    Args:
        mesh: Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        texture_size (int): Size of the texture.
        verbose (bool): Whether to print progress.
    """
    vertices = np.array(mesh.vertices).astype(float)
    faces = np.array(mesh.faces).astype(int)

    # mesh simplification
    max_faces = 50000
    mesh_reduction = max(1 - max_faces / faces.shape[0], simplify)
    vertices, faces = fast_simplification.simplify(
        vertices, faces, target_reduction=mesh_reduction)
    
    # parametrize mesh
    vertices, faces, uvs = parametrize_mesh(vertices, faces)

    # bake texture
    images = [images[i].cpu().numpy() for i in range(len(images))]
    masks = [masks[i].cpu().numpy() for i in range(len(masks))]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    texture = bake_texture(
        vertices.astype(float), faces.astype(float), uvs,
        images, masks, extrinsics, intrinsics,
        texture_size=texture_size, 
        mode='opt',
        lambda_tv=0.01,
        verbose=verbose
    )
    texture = Image.fromarray(texture)

    # rotate mesh
    vertices = vertices.astype(float) @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]).astype(float)
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, image=texture))
    return mesh
