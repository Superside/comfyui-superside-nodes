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


class SupersideCropToSizeNode:
    """
    Crop to Size Node: force an image to exact pixel dimensions, choosing
    which part of it survives, without ever distorting it.

    The reason this exists: image models only accept a fixed menu of aspect
    ratios (Grok Imagine has no 4:5, for instance), so hitting a format the
    model cannot generate means taking the nearest ratio it does support and
    cropping the difference away. Scaling the frame to the target instead
    would squash faces and product edges, so this node only ever scales by a
    single factor and then cuts.

    Anchor picks the side that stays put: cropping a 3:4 portrait to 4:5
    removes 6% of the height, and `top` keeps the head while `center` shaves
    the crown and the chin equally.
    """

    CATEGORY = "Superside"

    # The full 3x3 grid, named so there is nothing to guess: "top-center"
    # rather than a bare "top" that could be read as "the whole top edge".
    # The bare forms are still accepted for anyone who saved them.
    ANCHOR_OPTIONS = [
        "center",
        "top-center",
        "bottom-center",
        "middle-left",
        "middle-right",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ]

    FIT_OPTIONS = [
        "cover - scale to fill, then crop",
        "crop only - no scaling",
    ]

    RESAMPLE_OPTIONS = ["lanczos", "bicubic", "bilinear", "nearest"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": (
                    "INT",
                    {"default": 1536, "min": 1, "max": 16384, "step": 8,
                     "tooltip": "Output width in pixels."},
                ),
                "target_height": (
                    "INT",
                    {"default": 1920, "min": 1, "max": 16384, "step": 8,
                     "tooltip": "Output height in pixels. 1536x1920 is 4:5, the format Grok Imagine cannot generate natively."},
                ),
                "anchor": (
                    cls.ANCHOR_OPTIONS,
                    {"default": "center",
                     "tooltip": "Which part of the image is kept. 'top' keeps the head in a portrait; 'center' cuts equally from both sides."},
                ),
            },
            "optional": {
                "fit": (
                    cls.FIT_OPTIONS,
                    {"default": "cover - scale to fill, then crop",
                     "tooltip": "'cover' scales by a single factor until the image covers the target, then crops - the output is always exactly the target size and never distorted. 'crop only' cuts the pixel rectangle as-is and never scales, so a source smaller than the target comes out smaller."},
                ),
                "mask": (
                    "MASK",
                    {"tooltip": "Optional mask cropped with the exact same geometry, so it stays aligned with the image."},
                ),
                "resample": (
                    cls.RESAMPLE_OPTIONS,
                    {"default": "lanczos",
                     "tooltip": "Resampling filter used by 'cover'. Lanczos is sharpest for photos."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "crop_to_size"
    DESCRIPTION = (
        "Crop an image to exact pixel dimensions with a choice of anchor "
        "(center / top / bottom / left / right / corners), scaling by one "
        "factor first so the output always fills the target and is never "
        "distorted. Use it to reach a format the model cannot generate - "
        "e.g. 4:5 from Grok Imagine's 3:4. An optional mask is cropped with "
        "the same geometry."
    )

    @staticmethod
    def _anchor_offset(edge, source, target):
        """Offset of the crop window along one axis for the chosen edge."""
        slack = max(0, source - target)
        if edge in ("left", "top"):
            return 0
        if edge in ("right", "bottom"):
            return slack
        return slack // 2  # center / middle

    @classmethod
    def _crop_window(cls, anchor, source_w, source_h, target_w, target_h):
        """
        Resolve an anchor name into the crop window's top-left corner.

        Accepts the explicit grid names ("top-center", "middle-left", the four
        corners, "center") and the bare edge names ("top", "left", ...) so a
        workflow saved with either keeps working.
        """
        parts = [p for p in anchor.split("-") if p]
        vertical = next((p for p in parts if p in ("top", "bottom")), "middle")
        horizontal = next((p for p in parts if p in ("left", "right")), "center")

        left = cls._anchor_offset(horizontal, source_w, target_w)
        top = cls._anchor_offset(vertical, source_h, target_h)
        return left, top

    @staticmethod
    def _to_uint8(frame):
        if frame.dtype == np.uint8:
            return frame
        if frame.max() <= 1.0:
            return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def crop_to_size(
        self,
        image,
        target_width=1536,
        target_height=1920,
        anchor="center",
        fit="cover - scale to fill, then crop",
        mask=None,
        resample="lanczos",
    ):
        try:
            resample_filter = RESAMPLE_FILTERS.get(resample, Image.LANCZOS)
            cover = fit.startswith("cover")

            image_np = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
            if image_np.ndim == 3:
                image_np = image_np[None, ...]
            source_h, source_w = int(image_np.shape[1]), int(image_np.shape[2])

            if cover:
                scale = max(target_width / source_w, target_height / source_h)
                scaled_w = max(target_width, int(round(source_w * scale)))
                scaled_h = max(target_height, int(round(source_h * scale)))
                out_w, out_h = target_width, target_height
            else:
                # Never invent pixels: a source smaller than the target simply
                # comes out at its own size rather than being padded.
                scaled_w, scaled_h = source_w, source_h
                out_w = min(target_width, source_w)
                out_h = min(target_height, source_h)
                if (out_w, out_h) != (target_width, target_height):
                    logger.warning(
                        "Crop to Size: source is %dx%d, smaller than the %dx%d target on at "
                        "least one axis - 'crop only' does not scale, so the output is %dx%d. "
                        "Switch fit to 'cover' for an exact %dx%d result.",
                        source_w, source_h, target_width, target_height,
                        out_w, out_h, target_width, target_height,
                    )

            left, top = self._crop_window(anchor, scaled_w, scaled_h, out_w, out_h)
            box = (left, top, left + out_w, top + out_h)

            frames = []
            for frame in image_np:
                pil = Image.fromarray(self._to_uint8(frame))
                if (scaled_w, scaled_h) != (source_w, source_h):
                    pil = pil.resize((scaled_w, scaled_h), resample_filter)
                frames.append(np.array(pil.crop(box)).astype(np.float32) / 255.0)
            image_out = torch.from_numpy(np.stack(frames, axis=0))

            if mask is None:
                mask_out = torch.zeros((1, out_h, out_w), dtype=torch.float32)
            else:
                mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
                if mask_np.ndim == 2:
                    mask_np = mask_np[None, ...]
                mask_frames = []
                for frame in mask_np:
                    pil = Image.fromarray(
                        (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
                    )
                    # The mask is scaled to the same grid as the image before
                    # cropping, so it cannot drift out of alignment.
                    if (pil.width, pil.height) != (scaled_w, scaled_h):
                        pil = pil.resize((scaled_w, scaled_h), Image.BILINEAR)
                    mask_frames.append(np.array(pil.crop(box)).astype(np.float32) / 255.0)
                mask_out = torch.from_numpy(np.stack(mask_frames, axis=0))

            dropped = 1.0 - (out_w * out_h) / float(scaled_w * scaled_h)
            logger.info(
                "Crop to Size: %dx%d -> %dx%d, anchor %s, %s (%.1f%% of the frame cropped away)",
                source_w, source_h, out_w, out_h, anchor,
                "cover" if cover else "crop only", dropped * 100.0,
            )

            return (image_out, mask_out, out_w, out_h)

        except Exception as e:
            logger.error(f"Crop to Size failed: {str(e)}")
            raise RuntimeError(f"Crop to Size failed: {str(e)}") from e
