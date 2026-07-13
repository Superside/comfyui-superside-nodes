import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

RESAMPLE_FILTERS = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


class SupersideResizeLongSideNode:
    """
    Resize (Long Side) Node: scale an image so its longest side equals (or is
    capped at) a target size, preserving aspect ratio. Handy for keeping the
    biggest dimension of catalogue images under a limit before further
    processing, without distorting them.
    """

    CATEGORY = "Superside"

    RESAMPLE_OPTIONS = ["lanczos", "bicubic", "bilinear", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_long_side": (
                    "INT",
                    {"default": 2048, "min": 16, "max": 16384, "step": 8,
                     "tooltip": "Target size, in pixels, for the image's longest side. The shorter side scales proportionally."},
                ),
            },
            "optional": {
                "only_downscale": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "ON: only shrink images whose long side is bigger than the target; smaller images are left untouched (never upscaled). OFF: always set the long side to exactly the target, scaling up small images too."},
                ),
                "resample": (
                    cls.RESAMPLE_OPTIONS,
                    {"default": "lanczos",
                     "tooltip": "Resampling filter. Lanczos gives the sharpest results for photos."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    DESCRIPTION = (
        "Scale an image so its longest side hits a target size, keeping "
        "aspect ratio. Optionally only downscale (never enlarge). Also "
        "outputs the resulting width and height."
    )

    def resize(self, image, max_long_side=2048, only_downscale=True, resample="lanczos"):
        try:
            resample_filter = RESAMPLE_FILTERS.get(resample, Image.LANCZOS)

            image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
            if image_np.ndim == 3:
                image_np = image_np[None, ...]

            # All frames in a ComfyUI IMAGE batch share dimensions, so the
            # target size is computed once from the first frame.
            h, w = image_np.shape[1], image_np.shape[2]
            long_side = max(w, h)

            if only_downscale and long_side <= max_long_side:
                new_w, new_h = w, h
            else:
                scale = max_long_side / float(long_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))

            if (new_w, new_h) == (w, h):
                out_tensor = image if isinstance(image, torch.Tensor) else torch.from_numpy(image_np)
                return (out_tensor, w, h)

            frames = []
            for frame in image_np:
                if frame.dtype != np.uint8:
                    frame_u8 = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                else:
                    frame_u8 = frame
                pil = Image.fromarray(frame_u8)
                resized = pil.resize((new_w, new_h), resample_filter)
                frames.append(np.array(resized).astype(np.float32) / 255.0)

            out_tensor = torch.from_numpy(np.stack(frames, axis=0))
            logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h} (long side -> {max_long_side})")
            return (out_tensor, new_w, new_h)

        except Exception as e:
            logger.error(f"Resize long side failed: {str(e)}")
            raise RuntimeError(f"Resize long side failed: {str(e)}") from e
