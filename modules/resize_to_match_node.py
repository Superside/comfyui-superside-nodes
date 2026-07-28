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


class SupersideResizeToMatchNode:
    """
    Resize (To Match) Node: scale `image` (and optionally `mask`) so it lands
    at exactly `reference_image`'s width/height - a plain full-frame resize,
    never a crop or a reposition.

    Built for a "process at a smaller working resolution, then bring the
    result back up to the original size" pattern: run detection/generation
    on a downscaled copy (fast), then use this node to stretch the result
    back to match the true original before a final composite/save - so the
    delivered image is always the same size as what came in, regardless of
    what resolution it was actually generated at.

    Unlike a crop+stitch pair, there is no position bookkeeping here (no
    crop_x/crop_y/crop_w/crop_h) - the whole frame is resized, so there is
    no coordinate math that can drift or misalign.
    """

    CATEGORY = "Superside"

    RESAMPLE_OPTIONS = ["lanczos", "bicubic", "bilinear", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "image is resized to exactly match this image's width/height."},
                ),
            },
            "optional": {
                "mask": ("MASK",),
                "upscale_method": (
                    cls.RESAMPLE_OPTIONS,
                    {"default": "lanczos", "tooltip": "Resampling filter for the image. The mask always uses bilinear."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "resize_to_match"
    DESCRIPTION = (
        "Resize image (and optionally mask) to exactly match reference_image's "
        "width/height - a full-frame resize with no cropping or repositioning. "
        "Use to bring a result generated at a smaller working resolution back "
        "up to the original image's size before a final composite."
    )

    def _frame_to_uint8(self, frame_np):
        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                return np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
            return np.clip(frame_np, 0, 255).astype(np.uint8)
        return frame_np

    def resize_to_match(self, image, reference_image, mask=None, upscale_method="lanczos"):
        try:
            resample_filter = RESAMPLE_FILTERS.get(upscale_method, Image.LANCZOS)

            ref_np = reference_image.detach().cpu().numpy() if isinstance(reference_image, torch.Tensor) else np.asarray(reference_image)
            target_h, target_w = ref_np.shape[1], ref_np.shape[2]

            image_np = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
            src_h, src_w = image_np.shape[1], image_np.shape[2]

            if (src_w, src_h) == (target_w, target_h):
                image_out = image if isinstance(image, torch.Tensor) else torch.from_numpy(image_np)
            else:
                frames = []
                for frame in image_np:
                    frame_u8 = self._frame_to_uint8(frame)
                    pil = Image.fromarray(frame_u8)
                    resized = pil.resize((target_w, target_h), resample_filter)
                    frames.append(np.array(resized).astype(np.float32) / 255.0)
                image_out = torch.from_numpy(np.stack(frames, axis=0))
                logger.info(f"Resized image from {src_w}x{src_h} to {target_w}x{target_h} to match reference_image.")

            if mask is None:
                mask_out = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            else:
                mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
                mask_h, mask_w = mask_np.shape[1], mask_np.shape[2] if mask_np.ndim == 3 else mask_np.shape[:2]
                if mask_np.ndim == 2:
                    mask_np = mask_np[None, ...]
                    mask_h, mask_w = mask_np.shape[1], mask_np.shape[2]
                if (mask_w, mask_h) == (target_w, target_h):
                    mask_out = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask_np)
                else:
                    mask_frames = []
                    for frame in mask_np:
                        frame_u8 = np.clip(frame, 0.0, 1.0)
                        frame_u8 = (frame_u8 * 255.0).astype(np.uint8)
                        pil_mask = Image.fromarray(frame_u8, mode="L")
                        resized_mask = pil_mask.resize((target_w, target_h), Image.BILINEAR)
                        mask_frames.append(np.array(resized_mask).astype(np.float32) / 255.0)
                    mask_out = torch.from_numpy(np.stack(mask_frames, axis=0))

            return (image_out, mask_out)

        except Exception as e:
            logger.error(f"Resize to match failed: {str(e)}")
            raise RuntimeError(f"Resize to match failed: {str(e)}") from e
