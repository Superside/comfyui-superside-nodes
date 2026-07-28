import torch
import torch.nn.functional as F

import comfy.utils
import node_helpers


def _composite(destination, source, x, y, mask=None, multiplier=1, resize_source=False):
    """Same paste-with-mask algorithm as core comfy_extras.nodes_mask.composite()
    (multiplier fixed to 1 here since this node always operates in pixel space,
    never latent space)."""
    source = source.to(destination.device)
    if resize_source:
        source = F.interpolate(source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear")
    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))
    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[-1], top + source.shape[-2])

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = F.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    visible_width = destination.shape[-1] - left + min(0, x)
    visible_height = destination.shape[-2] - top + min(0, y)
    mask = mask[:, :, :visible_height, :visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[..., :visible_height, :visible_width]
    destination_portion = inverse_mask * destination[..., top:bottom, left:right]
    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination


class SupersideImageCompositeMaskedNode:
    """
    Superside Image Composite Masked: in-house replacement for core
    ImageCompositeMasked. Pastes `source` onto `destination` at (x, y),
    blended by `mask` (1 = source visible, 0 = destination visible - not a
    hard cutout, since the mask is bilinearly resized and can carry
    intermediate values). `resize_source` stretches source to fill the
    whole destination canvas before the x/y offset is applied.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    DESCRIPTION = "Paste one image onto another using a mask - in-house equivalent of core ImageCompositeMasked."

    def composite(self, destination, source, x, y, resize_source, mask=None):
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        output = _composite(
            destination, source.movedim(-1, 1), x, y, mask, 1, resize_source
        ).movedim(1, -1)
        return (output,)
