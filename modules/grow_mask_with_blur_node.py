import io

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import grey_dilation, grey_erosion, binary_fill_holes

MAX_RESOLUTION = 16384

# 3x3 structuring elements matching kjnodes' GrowMaskWithBlur kernels:
# tapered_corners=True -> cross (corners excluded), False -> full square.
_CROSS_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_SQUARE_KERNEL = np.ones((3, 3), dtype=bool)


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
        kernel = _CROSS_KERNEL if tapered_corners else _SQUARE_KERNEL

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
                op = grey_dilation if current_expand >= 0 else grey_erosion
                for _ in range(n_iter):
                    output = op(output, footprint=kernel, mode="constant", cval=0.0)

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
