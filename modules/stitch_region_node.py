import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import grey_dilation


class SupersideStitchRegionNode:
    """
    Paste a processed crop back into the full-resolution original image at
    an exact position (crop_x, crop_y, crop_w, crop_h) - the counterpart to
    SupersideCropByRegionNode.

    The crop is resized to exactly crop_w x crop_h before pasting, so even
    if the generator/API returned a slightly different resolution, the
    stitch still lands pixel-perfect at the recorded position. The paste
    mask is feathered (Gaussian blur) so the seam blends smoothly instead of
    showing a hard rectangle edge.
    """

    # Below DECONTAMINATE_FLOOR the seam takes the destination's colour
    # outright; by FLOOR + RANGE it is the source's. Same curve the retired
    # comfyui-inpaint-cropstitch-nb2 stitch used, which is where this
    # behaviour comes from.
    DECONTAMINATE_FLOOR = 0.12
    DECONTAMINATE_RANGE = 0.55

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "crop_x": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
                "crop_y": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Full-resolution mask (the same one fed into SupersideCropByRegionNode). If omitted, the whole crop rectangle is pasted."
                    },
                ),
                "feather_pixels": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Gaussian blur radius applied to the paste mask edge, for a seamless blend.",
                    },
                ),
                "mask_expand_pixels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Dilate the mask before feathering, so a slightly-too-tight segmentation still covers the whole edited object. A segmentation of a spectacle frame usually needs some of this, or the frame's outer edge keeps the old pixels.",
                    },
                ),
                "decontaminate_edge": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Where the feathered mask is weak, take the destination's own colour instead of a mix with the source. Image editors often return the edited object over a white or blank background, and blending that through a soft matte leaks a bright halo along the edge. Turn it off only if you want the raw blend.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "Superside"
    DESCRIPTION = (
        "Paste a processed crop back into the original image at crop_x/"
        "crop_y, resizing it to crop_w x crop_h so the stitch lands exactly "
        "in place, and feathering the mask edge. Pair with "
        "SupersideCropByRegionNode."
    )

    def stitch(
        self,
        destination,
        source,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        mask=None,
        feather_pixels=24,
        mask_expand_pixels=0,
        decontaminate_edge=True,
    ):
        if not isinstance(destination, torch.Tensor) or destination.ndim != 4:
            raise ValueError("destination must be a ComfyUI IMAGE tensor [B,H,W,C].")
        if not isinstance(source, torch.Tensor) or source.ndim != 4:
            raise ValueError("source must be a ComfyUI IMAGE tensor [B,H,W,C].")
        if destination.shape[0] != 1 or source.shape[0] != 1:
            raise ValueError("SupersideStitchRegionNode currently supports batch size 1 only.")

        dest_np = destination[0].detach().cpu().numpy().astype(np.float32)
        full_h, full_w = dest_np.shape[:2]
        if dest_np.shape[-1] < 3:
            raise ValueError("destination image must have at least 3 channels (RGB).")

        crop_x = max(0, min(int(crop_x), full_w - 1))
        crop_y = max(0, min(int(crop_y), full_h - 1))
        crop_w = max(1, min(int(crop_w), full_w - crop_x))
        crop_h = max(1, min(int(crop_h), full_h - crop_y))

        src_np = source[0].detach().cpu().numpy().astype(np.float32)
        if (src_np.shape[0], src_np.shape[1]) != (crop_h, crop_w):
            pil_src = Image.fromarray(np.clip(src_np * 255.0, 0.0, 255.0).astype(np.uint8))
            pil_src = pil_src.resize((crop_w, crop_h), Image.LANCZOS)
            src_np = np.asarray(pil_src, dtype=np.float32) / 255.0

        if mask is not None:
            mask_np = mask[0].detach().cpu().numpy().astype(np.float32) if isinstance(mask, torch.Tensor) else np.asarray(mask[0], dtype=np.float32)
            if mask_np.shape[:2] != (full_h, full_w):
                pil_mask = Image.fromarray((np.clip(mask_np, 0.0, 1.0) * 255.0).astype(np.uint8))
                pil_mask = pil_mask.resize((full_w, full_h), Image.BILINEAR)
                mask_np = np.asarray(pil_mask, dtype=np.float32) / 255.0
            region_mask = mask_np[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        else:
            region_mask = np.ones((crop_h, crop_w), dtype=np.float32)

        # Dilate before feathering: growing a blurred mask just moves the
        # gradient, growing a hard one actually covers more of the object.
        if mask_expand_pixels and int(mask_expand_pixels) > 0:
            radius = int(mask_expand_pixels)
            footprint = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
            region_mask = grey_dilation(region_mask, footprint=footprint)

        if feather_pixels and int(feather_pixels) > 0:
            pil_mask = Image.fromarray((np.clip(region_mask, 0.0, 1.0) * 255.0).astype(np.uint8))
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=float(feather_pixels)))
            region_mask = np.asarray(pil_mask, dtype=np.float32) / 255.0

        region_mask_3 = region_mask[:, :, None]

        result = dest_np.copy()
        dest_region = result[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :3]
        src_rgb = src_np[:, :, :3]

        if decontaminate_edge:
            # An editor that returns the object over a white/blank background
            # leaks that background into the seam once it is blended through a
            # soft matte. Replace the source colour with the destination's own
            # wherever the matte is weak, before the blend, so the halo has
            # nothing to come from. Fully inside the mask nothing changes.
            cleanup = np.clip((region_mask - self.DECONTAMINATE_FLOOR)
                              / self.DECONTAMINATE_RANGE, 0.0, 1.0)[:, :, None]
            src_rgb = cleanup * src_rgb + (1.0 - cleanup) * dest_region

        blended = dest_region * (1.0 - region_mask_3) + src_rgb * region_mask_3
        result[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :3] = blended

        out = torch.from_numpy(np.clip(result, 0.0, 1.0).astype(np.float32)).unsqueeze(0)
        return (out,)
