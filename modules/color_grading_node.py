import logging

import numpy as np
import torch
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


class SupersideColorGradingNode:
    """
    Color Grading Node: apply brightness, contrast, saturation and per-channel
    R/G/B offset adjustments to an image. Purely local (no API, no key).

    Behaviour matches the common "Color Grading" utility:
      - brightness / contrast / saturation are multiplicative factors
        (1.0 = no change) applied via PIL's ImageEnhance.
      - R / G / B are additive offsets (-255..255) added to each colour
        channel and clamped to 0-255.
    Alpha is preserved if the input has it.
    """

    CATEGORY = "Superside"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                                         "tooltip": "Brightness factor. 1.0 = no change, <1 darker, >1 brighter."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                                       "tooltip": "Contrast factor. 1.0 = no change."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                                         "tooltip": "Saturation factor. 1.0 = no change, 0 = grayscale."}),
                "R": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1,
                              "tooltip": "Red channel offset added to every pixel (clamped 0-255)."}),
                "G": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1,
                              "tooltip": "Green channel offset added to every pixel (clamped 0-255)."}),
                "B": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1,
                              "tooltip": "Blue channel offset added to every pixel (clamped 0-255)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_grade"
    DESCRIPTION = (
        "Apply brightness, contrast, saturation and per-channel R/G/B offsets "
        "to an image. Local color grading - no API key, no model call."
    )

    def color_grade(self, image, brightness, contrast, saturation, R, G, B):
        out_frames = []

        # A ComfyUI IMAGE batch is (B, H, W, C) float 0-1.
        for frame in image:
            frame_np = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
            has_alpha = frame_np.ndim == 3 and frame_np.shape[2] == 4

            rgb_u8 = (np.clip(frame_np[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
            pil = Image.fromarray(rgb_u8, mode="RGB")

            if brightness != 1.0:
                pil = ImageEnhance.Brightness(pil).enhance(brightness)
            if contrast != 1.0:
                pil = ImageEnhance.Contrast(pil).enhance(contrast)
            if saturation != 1.0:
                pil = ImageEnhance.Color(pil).enhance(saturation)

            arr = np.array(pil).astype(np.int16)
            if R != 0:
                arr[..., 0] = np.clip(arr[..., 0] + R, 0, 255)
            if G != 0:
                arr[..., 1] = np.clip(arr[..., 1] + G, 0, 255)
            if B != 0:
                arr[..., 2] = np.clip(arr[..., 2] + B, 0, 255)
            arr = arr.astype(np.float32) / 255.0

            if has_alpha:
                alpha = np.clip(frame_np[..., 3:4], 0.0, 1.0).astype(np.float32)
                arr = np.concatenate([arr, alpha], axis=-1)

            out_frames.append(torch.from_numpy(arr).unsqueeze(0))

        return (torch.cat(out_frames, dim=0),)
