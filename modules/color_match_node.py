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

    # Study-derived per-pass NB2 drift (subject region, identity-prompt
    # condition, averaged across 3 test images). Drift direction: hue rotates
    # toward red, saturation creeps up, brightness drops - so the correction
    # counter-rotates hue, trims saturation, lifts brightness, and nudges the
    # RGB curve (Hue/Sat/Value alone under-corrects the darkening).
    # See nb2_color_drift_study/nb2_hue_saturation_correction_pattern.md.
    _HUE_DEG_PER_PASS = 0.9
    _SAT_PP_PER_PASS = 0.5
    _VALUE_PP_PER_PASS = 1.1
    _RGB_NUDGE_PER_PASS = (2.8, 3.3, 3.0)  # 0-255 scale

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
                "nb2_passes_since_reference": ("INT", {
                    "default": 0, "min": 0, "max": 50, "step": 1,
                    "tooltip": "How many NB2 (Nano Banana 2) edit passes separate `image` from a TRUE clean original "
                               "(0 if `reference` already IS that clean original - leave at 0 for normal use). When > 0, "
                               "applies a small study-derived directional pre-correction (hue rotated back from red, "
                               "saturation trimmed, brightness lifted, RGB curve nudged) BEFORE the Reinhard match, so "
                               "the match has less residual drift to compensate for. Useful when `reference` isn't the "
                               "perfect original, or the subject mask is small/noisy and the Reinhard stats alone are unreliable."}),
                "nb2_bias_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales the nb2_passes_since_reference pre-correction. 1.0 = the study's measured average. "
                               "Only has any effect when nb2_passes_since_reference > 0."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_match"
    DESCRIPTION = (
        "Match an image's color to a reference image (Reinhard transfer in "
        "LAB or RGB). Use it to pull a generative edit's warm/red drift back "
        "toward the original: image = the edit, reference = the original. "
        "Local, no API key. For deep NB2 pipelines where `reference` isn't a "
        "perfect clean original (or the subject mask is small/noisy), set "
        "`nb2_passes_since_reference` > 0 to apply a study-derived directional "
        "pre-correction before the match; leave it at 0 for the normal case "
        "(a true clean reference)."
    )

    @staticmethod
    def _rgb_to_hsv_np(rgb):
        """Vectorized RGB->HSV, rgb in [0,1], HxWx3. Verified against colorsys
        (round-trip error ~1e-16)."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = rgb.max(axis=-1)
        minc = rgb.min(axis=-1)
        v = maxc
        delta = maxc - minc
        s = np.where(maxc > 0, delta / np.where(maxc == 0, 1, maxc), 0.0)

        h = np.zeros_like(maxc)
        mask = delta > 1e-8
        safe_delta = np.where(mask, delta, 1.0)
        rc = (maxc - r) / safe_delta
        gc = (maxc - g) / safe_delta
        bc = (maxc - b) / safe_delta

        h = np.where(mask & (maxc == r), (bc - gc), h)
        h = np.where(mask & (maxc == g) & ~(maxc == r), 2.0 + rc - bc, h)
        h = np.where(mask & (maxc == b) & ~(maxc == r) & ~(maxc == g), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        h = np.where(mask, h, 0.0)
        return np.stack([h, s, v], axis=-1)

    @staticmethod
    def _hsv_to_rgb_np(hsv):
        """Vectorized HSV->RGB inverse of _rgb_to_hsv_np."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i.astype(np.int32) % 6

        r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
        g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
        b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
        return np.stack([r, g, b], axis=-1)

    def _apply_nb2_bias_correction(self, rgb, mask, passes, bias_strength):
        """Pre-correction toward the study's measured NB2 drift direction,
        applied (subject-masked) BEFORE the reference-based Reinhard match.
        No-op when passes <= 0, so default behavior is byte-for-byte unchanged."""
        k = float(passes) * float(bias_strength)
        if k <= 0:
            return rgb

        hsv = self._rgb_to_hsv_np(rgb)
        hsv[..., 0] = (hsv[..., 0] + self._HUE_DEG_PER_PASS * k / 360.0) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] - self._SAT_PP_PER_PASS * k / 100.0, 0.0, 1.0)
        hsv[..., 2] = np.clip(hsv[..., 2] + self._VALUE_PP_PER_PASS * k / 100.0, 0.0, 1.0)
        corrected = self._hsv_to_rgb_np(hsv)

        nudge = np.array(self._RGB_NUDGE_PER_PASS, dtype=np.float32) * k / 255.0
        corrected = np.clip(corrected + nudge, 0.0, 1.0)

        if mask is not None:
            return np.where(mask[..., None], corrected, rgb).astype(np.float32)
        return corrected.astype(np.float32)

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
                    match_luminance=True, ignore_background=True,
                    nb2_passes_since_reference=0, nb2_bias_strength=1.0):
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

            src_mask = self._subject_mask(src_rgb) if ignore_background else None

            # Optional NB2 drift pre-correction, applied to the subject before
            # the reference-based Reinhard match. No-op when passes = 0, so the
            # default path is unchanged. Correction is masked to the subject so
            # the background stays as the original.
            src_rgb = self._apply_nb2_bias_correction(
                src_rgb, src_mask, nb2_passes_since_reference, nb2_bias_strength
            )

            src_work = self._to_working(src_rgb, lab)
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
