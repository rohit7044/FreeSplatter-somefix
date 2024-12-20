import os
import importlib
import imageio
import torch
import rembg
import numpy as np
import PIL.Image
from PIL import Image
from typing import Any
from torchvision import transforms


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def resize_without_crop(pil_image, target_width, target_height):
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)[:, :, :3]


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 255.0 * 2.0 - 1.0
    h = h.movedim(-1, 1)
    return h


@torch.inference_mode()
def remove_background(
    image: PIL.Image.Image,
    rembg: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        W, H = image.size
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(image, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = numpy2pytorch([feed]).to(device=rembg.device, dtype=torch.float32)
        alpha = rembg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.squeeze().clamp(0, 1)
        alpha = (alpha * 255).cpu().data.numpy().astype(np.uint8)
        alpha = Image.fromarray(alpha)

        no_bg_image = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
        no_bg_image.paste(image, mask=alpha)
        image = no_bg_image
    return image


@torch.inference_mode()
def remove_background(
    image: PIL.Image.Image,
    rembg: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = image.convert('RGB')
        input_images = transform_image(image).unsqueeze(0).to(rembg.device)
        with torch.no_grad():
            preds = rembg(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
    return image


# def remove_background(
#     image: PIL.Image.Image,
#     rembg_session: Any = None,
#     force: bool = False,
#     **rembg_kwargs,
# ) -> PIL.Image.Image:
#     do_remove = True
#     if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
#         do_remove = False
#     do_remove = do_remove or force
#     if do_remove:
#         image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
#     return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = Image.fromarray(new_image)
    return new_image


def rgba_to_white_background(image: PIL.Image.Image) -> torch.Tensor:
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).movedim(2, 0).float()
    image, alpha = image.split([3, 1], dim=0)
    image = image * alpha + torch.ones_like(image) * (1 - alpha)
    return image, alpha


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 30,
) -> None:
    # images: (N, C, H, W)
    frames = [(frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in frames]
    writer = imageio.get_writer(output_path, mode='I', fps=fps, codec='libx264')
    for frame in frames:
        writer.append_data(frame)
    writer.close()