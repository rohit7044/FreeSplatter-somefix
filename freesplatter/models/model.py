import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from einops import rearrange

from freesplatter.models.transformer import Transformer
from freesplatter.utils.infer_util import instantiate_from_config
from freesplatter.utils.recon_util import estimate_focal, fast_pnp


C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0


class FreeSplatterModel(nn.Module):
    def __init__(
        self, 
        transformer_config=None,
        renderer_config=None,
        use_2dgs=False,
        sh_residual=False,
    ):
        super().__init__()

        self.sh_dim = (renderer_config.sh_degree + 1) ** 2 * 3
        self.sh_residual = sh_residual
        self.use_2dgs = use_2dgs
        
        self.transformer = instantiate_from_config(transformer_config)

        if not use_2dgs:
            from .renderer.gaussian_renderer import GaussianRenderer
        else:
            from .renderer_2dgs.gaussian_renderer import GaussianRenderer
        self.gs_renderer = GaussianRenderer(renderer_config=renderer_config)

        self.register_buffer('pp', torch.tensor([256, 256], dtype=torch.float32), persistent=False)

    def forward_gaussians(self, images, **kwargs):
        """
        images: B x N x 3 x H x W
        """
        gaussians = self.transformer(images)    # B x N x H x W x C
        if self.sh_residual:
            residual = torch.zeros_like(gaussians)
            sh = RGB2SH(rearrange(images, 'b n c h w -> b n h w c'))
            residual[..., 3:6] = sh
            gaussians = gaussians + residual

        gaussians = rearrange(gaussians, 'b n h w c -> b (n h w) c')

        return gaussians
    
    def forward_renderer(self, gaussians, c2ws, fxfycxcy, **kwargs):
        """
        gaussians: B x K x 14
        c2ws: B x N x 4 x 4
        fxfycxcy: B x N x 4
        """
        render_results = self.gs_renderer.render(gaussians, fxfycxcy, c2ws, **kwargs)

        return render_results
    
    @torch.inference_mode()
    def estimate_focals(
        self, 
        images, 
        masks=None,
        use_first_focal=False,
    ):
        """
        Estimate the focal lengths of N input images.

        images: N x 3 x H x W
        masks: N x 1 x H x W
        """
        assert images.ndim == 4
        N, _, H, W = images.shape
        assert H == W, "Non-square images are not supported."

        pp = self.pp.to(images)
        # pp = torch.tensor([W/2, H/2]).to(images)

        focals = []
        for i in range(N):
            if use_first_focal and i > 0:
                break
            images_input = torch.cat([images[i:], images[:i]], dim=0)
            gaussians = self.forward_gaussians(images_input.unsqueeze(0))     # 1 x (N x H x W) x 14
            points = rearrange(gaussians[0, :H*W, :3], '(h w) c -> h w c', h=H, w=W)
            mask = masks[i] if masks is not None else None
            focal = estimate_focal(points, pp=pp, mask=mask)
            focals.append(focal)
        
        focals = torch.stack(focals).to(images)
        focals = focals.mean().reshape(1).repeat(N)
        return focals
    
    @torch.inference_mode()
    def estimate_poses(
        self, 
        images, 
        gaussians=None, 
        masks=None,
        focals=None,
        use_first_focal=True,
        opacity_threshold=5e-2, 
        pnp_iter=20,
    ):
        """
        Estimate the camera poses of N input images.

        images: N x 3 x h x W
        gaussians: K x 14 or 1 x K x 14
        masks: N x 1 x H x W
        focals: N
        """
        assert images.ndim == 4
        N, _, H, W = images.shape
        assert H == W, "Non-square images are not supported."

        # predict gaussians from images
        if gaussians is None:
            gaussians = self.forward_gaussians(images.unsqueeze(0))     # 1 x (N x H x W) x 14
        else:
            if gaussians.ndim == 2:
                gaussians = gaussians.unsqueeze(0)
            assert gaussians.shape[1] == N * H * W

        points = gaussians[..., :3].reshape(1, N, H, W, 3).squeeze(0)   # N x H x W x 3
        opacities = gaussians[..., 3+self.sh_dim].reshape(1, N, H, W).squeeze(0)
        opacities = torch.sigmoid(opacities)    # N x H x W

        # estimate focals if not provided
        if focals is None:
            focals = self.estimate_focals(images, masks=masks, use_first_focal=use_first_focal)

        # run PnP
        c2ws = []
        for i in range(N):
            pts3d = points[i].float().detach().cpu().numpy()
            # If masks are not provided, we use Gaussian opacities
            if masks is None:
                mask = (opacities[i] > opacity_threshold).detach().cpu().numpy()
            else:
                mask = masks[i].reshape(H, W).bool().detach().cpu().numpy()

            focal = focals[i].item()
            _, c2w = fast_pnp(pts3d, mask, focal=focal, niter_PnP=pnp_iter)

            c2ws.append(torch.from_numpy(c2w))
        
        c2ws = torch.stack(c2ws, dim=0).to(images)
        return c2ws, focals