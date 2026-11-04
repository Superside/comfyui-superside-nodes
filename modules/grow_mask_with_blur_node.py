import io

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import (binary_fill_holes, grey_dilation, grey_erosion, maximum_filter,
                           minimum_filter)

MAX_RESOLUTION = 16384

# 3x3 structuring elements matching kjnodes' GrowMaskWithBlur kernels:
# tapered_corners=True -> cross (corners excluded), False -> full square.
_CROSS_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_SQUARE_KERNEL = np.ones((3, 3), dtype=bool)

# Up to this many steps the mask is grown with the step-by-step loop; above it,
# one filter does all the steps at once (same result, see _grow).
_LOOP_MAX_STEPS = 8


def _grow(mask, steps, dilate, tapered):
    """`steps` repeats of a 3x3 grey dilation (or erosion) with a zero border,
    done as one operation when `steps` is large.

    Repeating the 3x3 square `steps` times is one (2*steps+1) square; repeating
    the cross is a diamond, |dx| + |dy| <= steps. Max and min only pick values,
    so the result is identical to the loop - but its time no longer grows with
    `steps` (4096 steps on a 2048 px mask took ~2 min in the loop, now < 1 s).
    """
    if steps <= _LOOP_MAX_STEPS:
        op = grey_dilation if dilate else grey_erosion
        kernel = _CROSS_KERNEL if tapered else _SQUARE_KERNEL
        for _ in range(steps):
            mask = op(mask, footprint=kernel, mode="constant", cval=0.0)
        return mask
    filt = maximum_filter if dilate else minimum_filter
    size = 2 * steps + 1
    if not tapered:
        return filt(mask, size=size, mode="constant", cval=0.0)
    # The diamond is a square once the grid is turned 45 degrees: u = x + y,
    # v = x - y (+ H - 1). Cells of the turned grid that are not pixels (u and v
    # of the wrong parity) must never win; pixels outside the image are the zero
    # border, as in the loop.
    H, W = mask.shape
    n = H + W - 1
    rot = np.full((n, n), -np.inf if dilate else np.inf, dtype=np.float32)
    p = (H - 1) % 2
    rot[0::2, p::2] = 0.0
    rot[1::2, 1 - p::2] = 0.0
    u = np.add.outer(np.arange(H), np.arange(W))
    v = np.subtract.outer(np.arange(W), np.arange(H)).T + (H - 1)
    rot[u, v] = mask
    return filt(rot, size=size, mode="constant", cval=0.0)[u, v]


class SupersideGrowMaskWithBlurNode:
    """
    Superside Grow Mask With Blur: in-house replacement for
    comfyui-kjnodes' GrowMaskWithBlur, with the same algorithm (dilate/erode
    per batch frame with a running expand rate, optional hole-filling,
    optional temporal lerp/decay against the previous frame, then a
    per-frame Gaussian blur) but using scipy instead of kornia.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "mask_inverted")
    FUNCTION = "expand_mask"
    DESCRIPTION = "Grow/shrink and blur a MASK - in-house equivalent of comfyui-kjnodes' GrowMaskWithBlur."

    def expand_mask(self, mask, expand, incremental_expandrate, tapered_corners, flip_input,
                     blur_radius, lerp_alpha, decay_factor, fill_holes=False):
        if flip_input:
            mask = 1.0 - mask

        H, W = mask.shape[-2], mask.shape[-1]
        growmask = mask.reshape((-1, H, W))

        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            n_iter = int(abs(round(current_expand)))
            if n_iter > 0 and output.max() > 0:
                # Grayscale morphology so partial (gray) mask values survive the
                # grow/shrink - e.g. a section written at 0.5 opacity upstream.
                # On a binary (0/1) mask this is identical to the old
                # binary_dilation/erosion, so existing behavior is unchanged.
                output = _grow(output, n_iter, current_expand >= 0, tapered_corners)

            current_expand += abs(incremental_expandrate) if current_expand >= 0 else -abs(incremental_expandrate)

            if fill_holes:
                binary = output > 0
                binary = binary_fill_holes(binary)
                output = binary.astype(np.float32)

            output_t = torch.from_numpy(output)
            if previous_output is not None and lerp_alpha < 1.0:
                output_t = lerp_alpha * output_t + (1.0 - lerp_alpha) * previous_output
            if previous_output is not None and decay_factor < 1.0:
                output_t = output_t + decay_factor * previous_output
                max_val = output_t.max()
                if max_val > 0:
                    output_t = output_t / max_val

            previous_output = output_t
            out.append(output_t.cpu())

        if blur_radius != 0:
            blurred = []
            for m in out:
                img = Image.fromarray((m.numpy() * 255.0).clip(0, 255).astype(np.uint8), mode="L")
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                arr = np.array(img).astype(np.float32) / 255.0
                blurred.append(torch.from_numpy(arr).unsqueeze(0))
            result = torch.cat(blurred, dim=0)
        else:
            result = torch.stack(out, dim=0)

        return (result, 1.0 - result)
