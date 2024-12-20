import torch

from .gaussian_utils import render, GaussianModel


class GaussianRenderer:
    def __init__(self, renderer_config=None):
        if 'scaling_activation_type' not in renderer_config:
            renderer_config['scaling_activation_type'] = 'exp'
        if 'scale_min_act' not in renderer_config:
            renderer_config['scale_min_act'] = 1 
            renderer_config['scale_max_act'] = 1 
            renderer_config['scale_multi_act'] = 0.1 

        self.gaussian_model = GaussianModel(sh_degree=renderer_config.sh_degree, 
                                            scaling_activation_type=renderer_config.scaling_activation_type, 
                                            scale_min_act=renderer_config.scale_min_act, 
                                            scale_max_act=renderer_config.scale_max_act, 
                                            scale_multi_act=renderer_config.scale_multi_act)
        self.img_height = renderer_config.img_height
        self.img_width = renderer_config.img_width
        self.bg_color = renderer_config.bg_color if 'bg_color' in renderer_config else (1.0, 1.0, 1.0)

    def render(self, latent, output_fxfycxcy, output_c2ws, rescale=None, render_size=None):
        if render_size is None:
            img_height, img_width = self.img_height, self.img_width
        else:
            img_height, img_width = render_size
        if rescale is None:
            rescale = torch.ones(latent.shape[0]).to(latent)
        
        shs_dim = (self.gaussian_model.sh_degree + 1) ** 2 * 3
        xyz, features, opacity, scaling, rotation = latent.split([3, shs_dim, 1, 3, 4], dim=-1)

        features = features.reshape(features.shape[0], -1, shs_dim//3, 3)
 
        bs, vs = output_fxfycxcy.shape[:2] 
        images = torch.zeros(bs, vs, 3, img_height, img_width, dtype=torch.float32, device=output_c2ws.device)
        alphas = torch.zeros(bs, vs, 1, img_height, img_width, dtype=torch.float32, device=output_c2ws.device)
        depths = torch.zeros(bs, vs, 1, img_height, img_width, dtype=torch.float32, device=output_c2ws.device)

        for idx in range(bs):
            pc = self.gaussian_model.set_data(xyz[idx], features[idx], scaling[idx], rotation[idx], opacity[idx], rescale[idx])
            for vidx in range(vs):
                render_results = render(pc, img_height, img_width, output_c2ws[idx, vidx], output_fxfycxcy[idx, vidx], self.bg_color)
                image = render_results['render']
                alpha = render_results['alpha']
                depth = render_results['depth']
                images[idx, vidx] =  image
                alphas[idx, vidx] =  alpha
                depths[idx, vidx] =  depth
        results = {'image': images, 'alpha': alphas, 'depth': depths}
        return results
