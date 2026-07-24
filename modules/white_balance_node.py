import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SupersideWhiteBalanceNode:
    """
    White Balance Node: neutralize a color cast by calibrating RGB from a
    neutral/white reference, so a "white" area reads as truly white again.

    Built for iterative-editing color drift: when you run an image through a
    generative editor repeatedly (e.g. Nano Banana Pro / V2), each pass adds
    a tiny color bias that accumulates - usually a warm/red/magenta cast.
    This node measures a neutral reference and rescales each channel to
    remove that cast while preserving overall brightness. Purely local, no
    API, no model call.

    Modes:
      - manual_sample: you point at a region you KNOW should be white/neutral
        (e.g. the catalogue's white background) via normalized coordinates;
        that patch is used as the reference. Most reliable.
      - auto_white_patch: uses the brightest near-neutral pixels (top
        `auto_percentile`% by luminance) as the white reference.
      - gray_world: assumes the whole image should average to neutral gray.
    """

    CATEGORY = "Superside"

    MODE_OPTIONS = ["manual_sample", "auto_white_patch", "gray_world"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODE_OPTIONS, {"default": "manual_sample"}),
            },
            "optional": {
                "sample_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "manual_sample: horizontal center of the white reference patch (0 = left, 1 = right)."}),
                "sample_y": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "manual_sample: vertical center of the white reference patch (0 = top, 1 = bottom). A corner/background is usually safest."}),
                "sample_size": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                                          "tooltip": "manual_sample: size of the square reference patch, as a fraction of the shorter side."}),
                "auto_percentile": ("FLOAT", {"default": 95.0, "min": 50.0, "max": 100.0, "step": 0.5,
                                              "tooltip": "auto_white_patch: pixels brighter than this luminance percentile are treated as the white reference."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "How much of the correction to apply. 1.0 = full neutralization, 0.5 = halfway."}),
                "preserve_luminance": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Scale the correction so overall brightness stays the same (only the color cast changes)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "white_balance"
    DESCRIPTION = (
        "Neutralize a color cast (e.g. the red/warm drift from repeated "
        "generative edits) by calibrating RGB from a white/neutral reference "
        "- sampled manually, auto-detected from the brightest pixels, or via "
        "gray-world. Local, no API key."
    )

    def _reference_rgb(self, rgb, mode, sample_x, sample_y, sample_size, auto_percentile):
        """Return the mean (r,g,b) of the chosen neutral reference, in 0-1."""
        h, w = rgb.shape[:2]

        if mode == "gray_world":
            return rgb.reshape(-1, 3).mean(axis=0)

        if mode == "auto_white_patch":
            lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            thresh = np.percentile(lum, auto_percentile)
            mask = lum >= thresh
            if mask.sum() < 16:  # too few - fall back to top pixels
                idx = np.argsort(lum.ravel())[-max(16, lum.size // 100):]
                sel = rgb.reshape(-1, 3)[idx]
                return sel.mean(axis=0)
            return rgb[mask].mean(axis=0)

        # manual_sample
        side = max(2, int(round(sample_size * min(w, h))))
        half = side // 2
        cx = int(round(sample_x * w))
        cy = int(round(sample_y * h))
        x1 = max(0, min(cx - half, w - side))
        y1 = max(0, min(cy - half, h - side))
        patch = rgb[y1:y1 + side, x1:x1 + side].reshape(-1, 3)
        return patch.mean(axis=0)

    def white_balance(self, image, mode="manual_sample", sample_x=0.5, sample_y=0.08,
                      sample_size=0.1, auto_percentile=95.0, strength=1.0, preserve_luminance=True):
        out_frames = []

        for frame in image:
            frame_np = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
            has_alpha = frame_np.ndim == 3 and frame_np.shape[2] == 4
            rgb = np.clip(frame_np[..., :3].astype(np.float32), 0.0, 1.0)

            ref = self._reference_rgb(rgb, mode, sample_x, sample_y, sample_size, auto_percentile)
            ref = np.maximum(ref, 1e-4)  # avoid divide-by-zero on a black reference

            # Per-channel gain that makes the reference neutral. Target = the
            # reference's own mean, so brightness of that patch is preserved
            # and only the color balance shifts.
            target = float(ref.mean())
            gains = target / ref

            if not preserve_luminance:
                # Normalize the reference to white (1.0) instead of its own gray.
                gains = 1.0 / ref

            # Blend by strength: 1.0 = full correction, 0.0 = no change.
            gains = 1.0 + (gains - 1.0) * float(strength)

            corrected = np.clip(rgb * gains[None, None, :], 0.0, 1.0)

            if has_alpha:
                alpha = np.clip(frame_np[..., 3:4].astype(np.float32), 0.0, 1.0)
                corrected = np.concatenate([corrected, alpha], axis=-1)

            out_frames.append(torch.from_numpy(corrected).unsqueeze(0))

        return (torch.cat(out_frames, dim=0),)
