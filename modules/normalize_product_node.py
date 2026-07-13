import json
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _hex_to_rgb(hex_str):
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


class SupersideNormalizeProductNode:
    """
    Normalize Product Node: places a catalogue product photo into a
    consistent, standardized frame so a fixed set of detail crops lands on
    the same spot across every SKU.

    It detects the product against the light catalogue background, then
    centers and scales it into a fixed-size canvas with a fixed margin. The
    product ends up occupying the same relative area of the output every
    time, so downstream fractional crop boxes (e.g. from the Manual Detail
    Sheet node, pre-positioned once per profile) stay aligned across the
    whole catalogue. No AI, no API key - purely local image processing.
    """

    CATEGORY = "Superside"

    FIT_OPTIONS = ["contain", "width", "height"]
    MODE_OPTIONS = ["keep resolution (pad)", "fixed canvas (scale)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mode": (
                    cls.MODE_OPTIONS,
                    {"default": "keep resolution (pad)",
                     "tooltip": "keep resolution (pad): crop to the product and pad with the margin at NATIVE resolution - no downscaling, no quality loss (output size varies per photo). fixed canvas (scale): scale the product into a fixed output_width x output_height canvas (may downscale)."},
                ),
                "margin_percent": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 45.0, "step": 0.5,
                     "tooltip": "Empty margin kept around the product, as a percent of the frame. The product always ends up spanning the same centered region, so crops stay aligned across SKUs."},
                ),
                "output_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8,
                     "tooltip": "Only used in 'fixed canvas' mode - width of the output canvas."},
                ),
                "output_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8,
                     "tooltip": "Only used in 'fixed canvas' mode - height of the output canvas."},
                ),
                "fit": (
                    cls.FIT_OPTIONS,
                    {"default": "contain",
                     "tooltip": "Only used in 'fixed canvas' mode. contain: fit the whole product. width: scale so the product's width fills the frame. height: scale to fill height."},
                ),
                "background_hex": (
                    "STRING",
                    {"default": "",
                     "placeholder": "#F5F5F5 - leave empty to auto-match the photo's background",
                     "tooltip": "Fill color for the output canvas. Leave empty to auto-sample the source photo's corner background."},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 12.0, "min": 1.0, "max": 128.0, "step": 1.0,
                     "tooltip": "How different from the background a pixel must be (0-255) to count as part of the product. Lower catches fainter details (thin metal), higher ignores soft shadows."},
                ),
                "detect_pad_percent": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 25.0, "step": 0.5,
                     "tooltip": "Extra padding added around the detected product before placing it, as a percent of the detected size."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "normalize"
    DESCRIPTION = (
        "Centers and scales a catalogue product into a fixed-size, "
        "fixed-margin frame so it occupies the same relative area every "
        "time - making a pre-set set of detail crops reusable across a whole "
        "catalogue. Detects the product against the light background locally; "
        "no AI, no API key."
    )

    @staticmethod
    def _tensor_to_pil(image):
        image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        if image_np.ndim == 4:
            image_np = image_np[0]
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        return Image.fromarray(image_np).convert("RGB")

    def _sample_background(self, arr):
        """Median color of the four corner patches - the catalogue backdrop."""
        h, w = arr.shape[:2]
        p = max(4, min(h, w) // 40)
        corners = np.concatenate([
            arr[:p, :p].reshape(-1, 3),
            arr[:p, -p:].reshape(-1, 3),
            arr[-p:, :p].reshape(-1, 3),
            arr[-p:, -p:].reshape(-1, 3),
        ], axis=0)
        return np.median(corners, axis=0)

    def _detect_bbox(self, arr, bg, threshold, detect_pad_percent):
        """Bounding box of pixels that differ from the background color."""
        diff = np.abs(arr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
        mask = diff > threshold
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

        pad_x = int((x2 - x1) * detect_pad_percent / 100.0)
        pad_y = int((y2 - y1) * detect_pad_percent / 100.0)
        h, w = arr.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return (x1, y1, x2, y2)

    def normalize(self, image, mode="keep resolution (pad)", output_width=1024, output_height=1024,
                  margin_percent=8.0, fit="contain", background_hex="", threshold=12.0,
                  detect_pad_percent=2.0):
        try:
            source_img = self._tensor_to_pil(image)
            arr = np.array(source_img)

            bg_sampled = self._sample_background(arr)
            fill_rgb = _hex_to_rgb(background_hex)
            if fill_rgb is None:
                fill_rgb = tuple(int(round(c)) for c in bg_sampled)

            bbox = self._detect_bbox(arr, bg_sampled, threshold, detect_pad_percent)
            if bbox is None:
                raise RuntimeError(
                    "Could not detect a product against the background. Try lowering "
                    "the threshold, or set background_hex to the photo's backdrop color."
                )

            x1, y1, x2, y2 = bbox
            product = source_img.crop((x1, y1, x2, y2))
            pw, ph = product.size
            margin = margin_percent / 100.0

            if mode == "fixed canvas (scale)":
                # Scale the product into a fixed W x H canvas. May downscale.
                inner_w = max(1.0, output_width * (1.0 - 2.0 * margin))
                inner_h = max(1.0, output_height * (1.0 - 2.0 * margin))
                if fit == "width":
                    scale = inner_w / pw
                elif fit == "height":
                    scale = inner_h / ph
                else:
                    scale = min(inner_w / pw, inner_h / ph)
                new_w = max(1, int(round(pw * scale)))
                new_h = max(1, int(round(ph * scale)))
                placed = product.resize((new_w, new_h), Image.LANCZOS)
                canvas = Image.new("RGB", (output_width, output_height), fill_rgb)
                canvas.paste(placed, ((output_width - new_w) // 2, (output_height - new_h) // 2))
                placed_size = [new_w, new_h]
                canvas_size = [output_width, output_height]
            else:
                # Keep native resolution: never resize the product, just pad
                # it with the margin so it spans the same centered [m, 1-m]
                # region. Output size follows the product, so no quality loss.
                denom = max(1e-6, 1.0 - 2.0 * margin)
                pad_x = int(round(pw * margin / denom))
                pad_y = int(round(ph * margin / denom))
                canvas_w = pw + 2 * pad_x
                canvas_h = ph + 2 * pad_y
                canvas = Image.new("RGB", (canvas_w, canvas_h), fill_rgb)
                canvas.paste(product, (pad_x, pad_y))
                placed_size = [pw, ph]
                canvas_size = [canvas_w, canvas_h]

            out_np = np.array(canvas).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_np).unsqueeze(0)

            info = json.dumps({
                "mode": mode,
                "detected_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "product_size": [pw, ph],
                "placed_size": placed_size,
                "output_size": canvas_size,
                "margin_percent": margin_percent,
                "background_rgb": list(fill_rgb),
            })

            return (out_tensor, info)

        except Exception as e:
            logger.error(f"Normalize product failed: {str(e)}")
            raise RuntimeError(f"Normalize product failed: {str(e)}") from e
