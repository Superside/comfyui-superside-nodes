import numpy as np
import torch
from PIL import Image


class SupersideCropByRegionNode:
    """
    Crop an image + mask around a selected region instead of sending the
    whole image (and a huge mask) to an edit/inpaint call.

    Designed to consume the center_x / center_y / crop_width / crop_height
    outputs of SupersideSAM3RegionSelectorNode or
    SupersideFlorence2RegionSelectorNode. Pair with
    SupersideStitchRegionNode, which pastes the processed crop back into the
    full-resolution original at the exact same position - the original
    image is never resized, only a small region around it is cropped out,
    processed, and pasted back.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "center_x": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
                "center_y": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
                "crop_width": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
                "crop_height": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            },
            "optional": {
                "padding_percent": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 0.0,
                        "max": 200.0,
                        "step": 1.0,
                        "tooltip": "Extra context margin added around the selected region before cropping.",
                    },
                ),
                "multiple_of": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Round the final crop width/height up to a multiple of this (diffusion-friendly).",
                    },
                ),
                "min_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Minimum crop width/height, in case the selected region is tiny.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "crop_x", "crop_y", "crop_w", "crop_h")
    FUNCTION = "crop"
    CATEGORY = "Superside"
    DESCRIPTION = (
        "Crop an image+mask around a selected region (from a SAM3/Florence "
        "region selector) instead of processing the whole image with a huge "
        "mask. Outputs the exact crop_x/crop_y/crop_w/crop_h used, so "
        "SupersideStitchRegionNode can paste the result back at the same "
        "spot without ever resizing the original image."
    )

    def crop(
        self,
        image,
        mask,
        center_x,
        center_y,
        crop_width,
        crop_height,
        padding_percent=25.0,
        multiple_of=64,
        min_size=512,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("image must be a ComfyUI IMAGE tensor [B,H,W,C].")
        if image.shape[0] != 1:
            raise ValueError("SupersideCropByRegionNode currently supports batch size 1 only.")

        img_np = image[0].detach().cpu().numpy().astype(np.float32)
        full_h, full_w = img_np.shape[:2]

        base_w = max(1, int(crop_width))
        base_h = max(1, int(crop_height))
        pad = max(0.0, float(padding_percent)) / 100.0
        want_w = base_w * (1.0 + pad)
        want_h = base_h * (1.0 + pad)

        want_w = max(want_w, float(min_size))
        want_h = max(want_h, float(min_size))

        m = max(1, int(multiple_of))
        final_w = int(np.ceil(want_w / m) * m)
        final_h = int(np.ceil(want_h / m) * m)
        final_w = max(1, min(final_w, full_w))
        final_h = max(1, min(final_h, full_h))

        x1 = int(round(int(center_x) - final_w / 2.0))
        y1 = int(round(int(center_y) - final_h / 2.0))
        x1 = max(0, min(x1, full_w - final_w))
        y1 = max(0, min(y1, full_h - final_h))
        x2 = x1 + final_w
        y2 = y1 + final_h

        cropped_img = img_np[y1:y2, x1:x2, :].copy()

        mask_np = mask[0].detach().cpu().numpy().astype(np.float32) if isinstance(mask, torch.Tensor) else np.asarray(mask[0], dtype=np.float32)
        if mask_np.shape[:2] != (full_h, full_w):
            pil_mask = Image.fromarray((np.clip(mask_np, 0.0, 1.0) * 255.0).astype(np.uint8))
            pil_mask = pil_mask.resize((full_w, full_h), Image.BILINEAR)
            mask_np = np.asarray(pil_mask, dtype=np.float32) / 255.0
        cropped_mask = mask_np[y1:y2, x1:x2].copy()

        image_out = torch.from_numpy(cropped_img).unsqueeze(0)
        mask_out = torch.from_numpy(cropped_mask).unsqueeze(0)

        return (image_out, mask_out, int(x1), int(y1), int(final_w), int(final_h))
