import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

VERY_BIG_SIZE = 1024 * 1024


def _tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    if t.shape[-1] == 4:
        return t
    if t.shape[-1] == 3:
        alpha = torch.ones((*t.shape[:-1], 1), device=t.device)
        return torch.cat((t, alpha), dim=-1)
    # single-channel (mask-shaped) image -> broadcast to RGBA
    size = t.shape[-1] if t.dim() == 4 else 1
    if t.dim() == 3:
        t = t.unsqueeze(-1)
    return t.expand(*t.shape[:-1], 4)


def _tensor2mask(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4:
        if t.shape[-1] == 1:
            return t[:, :, :, 0]
        if t.shape[-1] == 4:
            # alpha channel as mask
            return t[:, :, :, 3]
        return t[:, :, :, 0]
    return t


class SupersideCutByMaskNode:
    """
    Superside Cut By Mask: in-house replacement for masquerade-nodes'
    "Cut By Mask". Crops each image in the batch to its mask's bounding
    box, then resizes every crop to a common size (the largest box in the
    batch, or a fixed force_resize_width/height when > 0) with bicubic
    interpolation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "force_resize_width": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "force_resize_height": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
            }
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cut"
    DESCRIPTION = "Crop an image to its mask's bounding box - in-house equivalent of Masquerade's Cut By Mask."

    def cut(self, image, mask, force_resize_width=0, force_resize_height=0):
        image = _tensor2rgba(image)
        mask = _tensor2mask(mask)

        if mask.shape[-2:] != image.shape[-3:-1]:
            mask = F.interpolate(
                mask.unsqueeze(1), size=(image.shape[-3], image.shape[-2]), mode="nearest"
            ).squeeze(1)

        B, MB = image.shape[0], mask.shape[0]
        if MB < B:
            if B % MB != 0:
                raise ValueError("Cannot match image batch size to mask batch size")
            mask = mask.repeat(B // MB, 1, 1)

        is_empty = mask.reshape(mask.shape[0], -1).max(dim=1).values <= 0
        safe_mask = mask.clone()
        if is_empty.any():
            safe_mask[is_empty, 0, 0] = 1.0
        boxes = masks_to_boxes(safe_mask)  # (N, 4) xyxy

        widths = (boxes[:, 2] - boxes[:, 0] + 1).clamp(min=1)
        heights = (boxes[:, 3] - boxes[:, 1] + 1).clamp(min=1)
        use_width = int(force_resize_width) if force_resize_width > 0 else int(widths.max().item())
        use_height = int(force_resize_height) if force_resize_height > 0 else int(heights.max().item())

        alpha_mask = torch.ones_like(image)
        alpha_mask[:, :, :, 3] = mask
        image = image * alpha_mask

        results = []
        for i in range(image.shape[0]):
            if is_empty[i]:
                results.append(torch.zeros((use_height, use_width, image.shape[-1]), device=image.device))
                continue
            x1, y1, x2, y2 = (int(v.item()) for v in boxes[i])
            crop = image[i, y1:y2 + 1, x1:x2 + 1, :]
            crop = crop.movedim(-1, 0).unsqueeze(0)
            crop = F.interpolate(crop, size=(use_height, use_width), mode="bicubic")
            crop = crop.squeeze(0).movedim(0, -1)
            results.append(crop)

        result = torch.stack(results, dim=0)

        channels = image.shape[-1]
        if channels == 4 and torch.min(result[:, :, :, 3]).item() >= 1.0:
            result = result[:, :, :, :3]
        return (result,)
