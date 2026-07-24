import logging

import numpy as np
import torch
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


# ---- self-contained sRGB <-> CIE LAB conversion (numpy only) ----
# D65 white point.
_XYZ_FROM_RGB = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def _f(t):
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)


def _f_inv(t):
    d = 6.0 / 29.0
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))


def rgb_to_lab(rgb):
    lin = _srgb_to_linear(rgb.astype(np.float64))
    xyz = lin @ _XYZ_FROM_RGB.T
    xyz = xyz / _WHITE
    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1) * _WHITE
    lin = xyz @ _RGB_FROM_XYZ.T
    return _linear_to_srgb(lin)


class SupersideColorMatchNode:
    """
    Color Match Node: transfer the color character of a reference image onto
    another image, so the two share the same tonality. Purely local (no API).

    Built for generative color drift: when an editor (e.g. Nano Banana
    Pro/V2) re-renders a subject warmer/redder than the source, feed the
    editor's output as `image` and the clean original as `reference`; this
    matches the output's per-channel color statistics to the original,
    pulling skin/hair/lighting back toward the source tonality. It works
    even when the pose/framing changed, because it matches global
    distribution statistics rather than aligning pixels.

    Method: Reinhard color transfer - for each channel, rescale the image so
    its mean and standard deviation match the reference's. Done in CIE LAB
    (perceptual) by default for natural skin results, or in RGB.
    """

    CATEGORY = "Superside"

    METHOD_OPTIONS = ["LAB (Reinhard)", "RGB"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "How strongly to pull the image toward the reference's color. 1.0 = full match, 0.5 = halfway."}),
                "method": (cls.METHOD_OPTIONS, {"default": "LAB (Reinhard)",
                                                "tooltip": "LAB is perceptual and usually best for skin. RGB is a simpler per-channel match."}),
                "match_luminance": ("BOOLEAN", {"default": True,
                                                "tooltip": "ON also matches brightness/contrast to the reference. OFF keeps the image's own luminance and only corrects color (a/b), useful if you like the generated exposure."}),
                "ignore_background": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Exclude near-white background pixels when measuring color, so the SUBJECT (skin/hair) drives the match instead of the large white backdrop. Ideal for catalogue shots on white."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_match"
    DESCRIPTION = (
        "Match an image's color to a reference image (Reinhard transfer in "
        "LAB or RGB). Use it to pull a generative edit's warm/red drift back "
        "toward the original: image = the edit, reference = the original. "
        "Local, no API key."
    )

    @staticmethod
    def _subject_mask(rgb):
        """
        Boolean mask of non-background pixels: everything that ISN'T a bright,
        low-saturation (near-white) pixel. Used so a large white backdrop
        doesn't dominate the color statistics.
        """
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=rgb.dtype)
        spread = rgb.max(axis=-1) - rgb.min(axis=-1)
        is_bg = (lum > 0.82) & (spread < 0.10)
        return ~is_bg

    @staticmethod
    def _stats(arr, mask=None):
        flat = arr.reshape(-1, arr.shape[-1])
        if mask is not None:
            m = mask.reshape(-1)
            if m.sum() >= 64:  # enough subject pixels to be meaningful
                flat = flat[m]
        return flat.mean(axis=0), flat.std(axis=0)

    def _match_channels(self, src, ref, match_luminance, lab, src_mask, ref_mask):
        """src/ref are HxWx3 arrays in the working space. Returns matched src."""
        s_mean, s_std = self._stats(src, src_mask)
        r_mean, r_std = self._stats(ref, ref_mask)
        out = src.copy()
        n_channels = 3
        for c in range(n_channels):
            if lab and c == 0 and not match_luminance:
                continue  # keep source luminance (L)
            ss = s_std[c] if s_std[c] > 1e-5 else 1e-5
            out[..., c] = (src[..., c] - s_mean[c]) * (r_std[c] / ss) + r_mean[c]
        return out

    def _to_working(self, rgb, lab):
        return rgb_to_lab(rgb) if lab else rgb.astype(np.float64)

    def _from_working(self, work, lab):
        return lab_to_rgb(work) if lab else np.clip(work, 0.0, 1.0)

    def color_match(self, image, reference, strength=1.0, method="LAB (Reinhard)",
                    match_luminance=True, ignore_background=True):
        lab = method.startswith("LAB")

        ref_np = reference.cpu().numpy() if isinstance(reference, torch.Tensor) else np.asarray(reference)
        if ref_np.ndim == 4:
            ref_np = ref_np[0]
        ref_rgb = np.clip(ref_np[..., :3].astype(np.float32), 0.0, 1.0)
        ref_work = self._to_working(ref_rgb, lab)
        ref_mask = self._subject_mask(ref_rgb) if ignore_background else None

        out_frames = []
        for frame in image:
            frame_np = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
            has_alpha = frame_np.ndim == 3 and frame_np.shape[2] == 4
            src_rgb = np.clip(frame_np[..., :3].astype(np.float32), 0.0, 1.0)

            src_work = self._to_working(src_rgb, lab)
            src_mask = self._subject_mask(src_rgb) if ignore_background else None
            matched_work = self._match_channels(src_work, ref_work, match_luminance, lab, src_mask, ref_mask)
            matched_rgb = np.clip(self._from_working(matched_work, lab), 0.0, 1.0).astype(np.float32)

            s = float(strength)
            result = src_rgb * (1.0 - s) + matched_rgb * s

            # When we measured on the subject only, apply the correction to the
            # subject only too - otherwise the subject-derived gains tint the
            # white backdrop (e.g. it goes cyan). Keep background pixels as the
            # original, with a feathered edge so there's no halo.
            if ignore_background and src_mask is not None and src_mask.sum() >= 64:
                mimg = Image.fromarray((src_mask * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(3))
                mw = (np.asarray(mimg).astype(np.float32) / 255.0)[..., None]
                result = result * mw + src_rgb * (1.0 - mw)

            if has_alpha:
                alpha = np.clip(frame_np[..., 3:4].astype(np.float32), 0.0, 1.0)
                result = np.concatenate([result, alpha], axis=-1)

            out_frames.append(torch.from_numpy(result.astype(np.float32)).unsqueeze(0))

        return (torch.cat(out_frames, dim=0),)
