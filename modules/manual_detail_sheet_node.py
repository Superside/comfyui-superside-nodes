import json
import logging
import os
import random

import numpy as np
import torch

from .base_node import DetailSheetCompositionMixin

logger = logging.getLogger(__name__)


def _save_temp_preview(pil_image, prefix):
    """
    Save a PIL image into ComfyUI's temp dir and return the {filename,
    subfolder, type} descriptor the frontend needs to fetch it via /view.
    Used so the node's in-canvas widget can show the exact image it received
    (e.g. an upstream-normalized photo) after execution, since the widget
    otherwise can't read an image out of a non-preview upstream node.
    """
    try:
        import folder_paths
        out_dir = folder_paths.get_temp_directory()
    except Exception:
        return None
    os.makedirs(out_dir, exist_ok=True)
    suffix = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8))
    filename = f"{prefix}_{suffix}.png"
    pil_image.save(os.path.join(out_dir, filename))
    return {"filename": filename, "subfolder": "", "type": "temp"}


class SupersideManualDetailSheetNode(DetailSheetCompositionMixin):
    """
    Manual Detail Sheet Node: like Smart Detail Sheet, but you draw the
    detail boxes yourself directly on the image preview in the node (drag
    to move, scroll to resize, click the corner dot to turn a box on/off)
    instead of an AI model choosing them. No API key, no API call - this
    node never leaves the machine.
    """

    CATEGORY = "Superside"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "crop_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.5,
                        "tooltip": "How much to enlarge each selected box (Lanczos resize, no AI upscaling).",
                    },
                ),
                "boxes": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Internal: box positions set by dragging on the image preview above. Not meant to be edited by hand.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Draw up to 6 boxes directly on the image preview (drag to move, "
        "scroll to resize, click the corner dot to turn a box on/off) to "
        "manually pick which details get cropped, upscaled, and composited "
        "alongside the original into a product-spec-sheet-style image - no "
        "AI detection, no API key needed."
    )

    def _parse_boxes(self, boxes_json, image_width, image_height):
        if not boxes_json or not boxes_json.strip():
            raise ValueError(
                "No boxes have been drawn yet. Drag on the image preview in "
                "this node to position the detail boxes, then run the workflow again."
            )

        try:
            data = json.loads(boxes_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse box data: {e}") from e

        raw_boxes = data.get("boxes", []) if isinstance(data, dict) else data
        if not isinstance(raw_boxes, list):
            raise ValueError("Box data is not a list.")

        active_boxes = []
        for i, b in enumerate(raw_boxes):
            if not isinstance(b, dict) or not b.get("active", True):
                continue
            # The widget stores box coordinates as 0-1 fractions of the
            # image, so they stay correct regardless of how the preview
            # canvas happened to be scaled/resized in the browser.
            x1 = float(b.get("x1", 0)) * image_width
            y1 = float(b.get("y1", 0)) * image_height
            x2 = float(b.get("x2", 0)) * image_width
            y2 = float(b.get("y2", 0)) * image_height
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            # Force a true 1:1 square crop in pixels, centered on the box.
            # The widget already draws squares, but fraction rounding can
            # leave the pixel rect a pixel or two off - square it here so the
            # output crop is guaranteed 1:1. The side is capped to the image
            # and the square is shifted (not clipped) to stay fully inside.
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            side = min(max(x2 - x1, y2 - y1), float(image_width), float(image_height))
            half = side / 2.0
            sx1, sy1, sx2, sy2 = cx - half, cy - half, cx + half, cy + half
            if sx1 < 0:
                sx2 -= sx1
                sx1 = 0.0
            if sy1 < 0:
                sy2 -= sy1
                sy1 = 0.0
            if sx2 > image_width:
                sx1 -= (sx2 - image_width)
                sx2 = float(image_width)
            if sy2 > image_height:
                sy1 -= (sy2 - image_height)
                sy2 = float(image_height)

            active_boxes.append({"label": f"detail {i + 1}", "x1": sx1, "y1": sy1, "x2": sx2, "y2": sy2})

        if not active_boxes:
            raise ValueError(
                "No active boxes to crop. Turn on at least one box (click "
                "its corner dot) in the image preview above."
            )

        return active_boxes

    def generate(self, image, crop_scale=2.0, boxes=""):
        try:
            source_img = self._tensor_to_pil(image)

            # Send the received image back to the node's widget so it can be
            # drawn on / boxed - this is how the widget sees an upstream-
            # processed image (e.g. from Normalize Product) that has no
            # preview thumbnail of its own until it runs.
            source_preview = _save_temp_preview(source_img, "superside_mds_src")

            active_boxes = self._parse_boxes(boxes, source_img.width, source_img.height)

            kept_details = []
            detail_crops = []
            for b in active_boxes:
                crop = self._crop_rect_scale_check_blank(
                    source_img, b["x1"], b["y1"], b["x2"], b["y2"], crop_scale, label=b["label"]
                )
                if crop is not None:
                    kept_details.append(b)
                    detail_crops.append(crop)

            if not detail_crops:
                raise RuntimeError(
                    "All selected boxes were discarded as blank/empty - "
                    "reposition them over a part of the image with visible detail."
                )

            if len(detail_crops) < len(active_boxes):
                logger.warning(
                    f"Discarded {len(active_boxes) - len(detail_crops)} blank box(es); "
                    f"kept {len(detail_crops)} of {len(active_boxes)} selected."
                )

            composed = self._compose_detail_sheet(source_img, detail_crops)

            composed_np = np.array(composed).astype(np.float32) / 255.0
            composed_tensor = torch.from_numpy(composed_np).unsqueeze(0)

            info = json.dumps({
                "details": kept_details,
                "discarded_count": len(active_boxes) - len(detail_crops),
                "crop_scale": crop_scale,
            })

            result = (composed_tensor, info)
            if source_preview is not None:
                return {"ui": {"superside_src": [source_preview]}, "result": result}
            return result

        except Exception as e:
            logger.error(f"Manual detail sheet failed: {str(e)}")
            raise RuntimeError(f"Manual detail sheet failed: {str(e)}") from e
