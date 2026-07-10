import json
import logging
import re

import numpy as np
import torch
from PIL import Image

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)

VISION_MODEL_OPTIONS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.6",
]


class SupersideSmartDetailSheetNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Smart Detail Sheet Node: uses a vision LLM to find the most visually
    interesting close-up details in a product photo (textures, logos,
    hinges, pads, seams, materials, etc), crops each one at native
    resolution, upscales the crops locally, and composites everything into
    one final image: the original photo on top, with a row of enlarged
    detail callouts underneath - like a product spec sheet.

    No detail-cropping model call is made beyond the single vision LLM
    request; cropping/upscaling/compositing all happen locally so the
    crops are exact 1:1 pixel regions of the source image, just scaled up
    uniformly (never stretched/distorted).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "num_details": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 6,
                        "tooltip": "How many detail close-ups to find and crop.",
                    },
                ),
                "detail_hint": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional: guide the model, e.g. \"focus on the nose pads, hinge, and logo\"",
                    },
                ),
                "crop_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.5,
                        "tooltip": "How much to enlarge each detail crop (Lanczos resize, no AI upscaling).",
                    },
                ),
                "model": (VISION_MODEL_OPTIONS, {"default": "openai/gpt-4o"}),
                "padding_percent": (
                    "FLOAT",
                    {
                        "default": 200.0,
                        "min": 0.0,
                        "max": 300.0,
                        "tooltip": "Extra margin added around each detected detail before cropping, as a percent of the detail's own size. Compensates for imprecise bounding boxes and gives enough surrounding context that the detail reads as part of the product, not an abstract texture patch.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Finds interesting product details with a vision LLM, crops them at "
        "native resolution, upscales the crops locally, and composites the "
        "original photo plus a row of enlarged detail callouts into one "
        "final image - like a product spec sheet."
    )

    ENDPOINT_CANDIDATES = [
        "openrouter/router/vision",
        "fal-ai/any-llm/vision",
    ]

    def _detect_details(self, client, model, image_url, num_details, detail_hint):
        hint_line = f"\n\nAdditional guidance: {detail_hint.strip()}" if detail_hint and detail_hint.strip() else ""
        prompt = (
            f"Analyze this product photo and identify exactly {num_details} of the most "
            "visually interesting close-up details worth showcasing to a buyer.\n\n"
            "First silently identify what kind of product this is, then pick the "
            "details that matter most for that specific category. For example: on "
            "eyewear, prioritize the nose pads, hinges, temple tips, the bridge, and "
            "any screws/rivets; on footwear, prioritize stitching, sole texture, "
            "laces, and logos; on bags, prioritize hardware, zippers, and seams; on "
            "apparel, prioritize fabric weave, seams, buttons, and zippers. More "
            "generally, look for textures, materials, logos, hinges, pads, seams, "
            f"stitching, or other unique design elements.{hint_line}\n\n"
            "Return ONLY a JSON array, no other text, no markdown code fences, in "
            "this exact format:\n"
            '[{"label": "short description", "x1": 0, "y1": 0, "x2": 0, "y2": 0}, ...]\n\n'
            "Coordinates are normalized to a 0-1000 scale where (0,0) is the "
            "top-left corner and (1000,1000) is the bottom-right corner of the "
            "image. Each box should tightly frame just that one detail - padding "
            f"for context is added separately afterwards. Return exactly {num_details} entries."
        )

        arguments = {
            "prompt": prompt,
            "image_urls": [image_url],
            "model": model,
        }

        result = None
        last_error = None
        for endpoint in self.ENDPOINT_CANDIDATES:
            try:
                logger.info(f"Trying vision endpoint: {endpoint}")
                result = self.call_api(client, endpoint, arguments)
                break
            except Exception as endpoint_error:
                last_error = endpoint_error
                logger.warning(f"Endpoint failed: {endpoint} - {str(endpoint_error)}")

        if result is None:
            raise RuntimeError(f"All vision endpoints failed. Last error: {last_error}")

        output_text = result.get("output", "")
        logger.debug(f"Vision model raw output: {output_text}")

        match = re.search(r"\[.*\]", output_text, re.DOTALL)
        if not match:
            raise RuntimeError(
                f"Vision model did not return a parseable JSON array. Raw output: {output_text[:500]}"
            )

        try:
            details = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse detail JSON from vision model: {e}") from e

        if not isinstance(details, list) or not details:
            raise RuntimeError("Vision model returned no usable details.")

        return details[:num_details]

    # A crop whose pixel std-dev is below this is treated as a flat/blank
    # region (the model's bounding box missed the actual detail) and is
    # dropped from the final sheet instead of showing an empty patch. Kept
    # low because legitimate details can be subtle (e.g. silver hardware on
    # a white background for rimless eyewear) - a truly blank/background
    # crop is much closer to 0.
    BLANK_CROP_STD_THRESHOLD = 4.0

    def _crop_and_scale_detail(self, source_img, detail, crop_scale, padding_percent):
        width, height = source_img.size

        x1 = float(detail.get("x1", 0)) / 1000.0 * width
        y1 = float(detail.get("y1", 0)) / 1000.0 * height
        x2 = float(detail.get("x2", 1000)) / 1000.0 * width
        y2 = float(detail.get("y2", 1000)) / 1000.0 * height

        if x2 <= x1:
            x1, x2 = x2, x1
        if y2 <= y1:
            y1, y2 = y2, y1

        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        pad_x = box_w * (padding_percent / 100.0)
        pad_y = box_h * (padding_percent / 100.0)

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop region for detail '{detail.get('label', '?')}' - skipping")
            return None

        cropped = source_img.crop((x1, y1, x2, y2))

        # Discard crops that landed on a flat/empty region (background) instead
        # of the intended detail - the model's bounding box missed the target.
        crop_std = float(np.array(cropped.convert("L")).std())
        if crop_std < self.BLANK_CROP_STD_THRESHOLD:
            logger.warning(
                f"Detail '{detail.get('label', '?')}' crop looks blank "
                f"(std={crop_std:.1f}) - discarding"
            )
            return None

        # Uniform scale only - never distorts aspect ratio ("1:1" crop, just enlarged).
        new_size = (max(1, int(cropped.width * crop_scale)), max(1, int(cropped.height * crop_scale)))
        return cropped.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _rescale_crops(crops, scale):
        """Uniformly rescale every crop by the same factor (never distorts)."""
        if scale == 1.0:
            return crops
        rescaled = []
        for c in crops:
            new_size = (max(1, int(c.width * scale)), max(1, int(c.height * scale)))
            rescaled.append(c.resize(new_size, Image.LANCZOS))
        return rescaled

    # The detail block's corresponding dimension (column height for the side
    # layout, row width for the below layout) is kept within this fraction
    # range of the original's own dimension: never below the floor (so
    # details stay legible) and never above the ceiling of 1.0 (so the
    # original photo stays the dominant element). Floor is always well below
    # the ceiling, so applying it can never overshoot the cap.
    MIN_DETAIL_BLOCK_RATIO = 0.35
    MAX_DETAIL_BLOCK_RATIO = 1.0

    def _fit_block_to_original(self, resized_crops, natural_size, original_size):
        """
        Uniformly rescale a column/row of crops so its overall size sits
        within [MIN_DETAIL_BLOCK_RATIO, MAX_DETAIL_BLOCK_RATIO] of the
        original's corresponding dimension. Scales up if the details would
        otherwise be too small to read, scales down if they'd overwhelm the
        original - same factor applied to every crop, so proportions never
        distort.
        """
        floor = original_size * self.MIN_DETAIL_BLOCK_RATIO
        ceiling = original_size * self.MAX_DETAIL_BLOCK_RATIO

        if natural_size < floor:
            fit_scale = floor / natural_size
        elif natural_size > ceiling:
            fit_scale = ceiling / natural_size
        else:
            return resized_crops

        return self._rescale_crops(resized_crops, fit_scale)

    def _compose_detail_sheet(self, original_img, detail_crops):
        """
        Portrait/tall originals (width < height, e.g. 4:5) get their detail
        crops arranged in a column beside the image, so the canvas doesn't
        keep growing taller. Landscape/square originals (width >= height,
        e.g. 16:9) get their crops in a row underneath instead.

        The original image must stay the dominant element, but the details
        must also stay legible: the detail column/row as a whole is kept
        within MIN_DETAIL_BLOCK_RATIO-MAX_DETAIL_BLOCK_RATIO of the original's
        corresponding dimension (height for the side layout, width for the
        below layout). The whole block of crops is scaled uniformly (same
        factor for every crop, so their individual proportions stay
        untouched) to land inside that range.
        """
        is_portrait = original_img.width < original_img.height
        gap = 24

        if is_portrait:
            # Column of details to the right, each normalized to the same width.
            col_width = max(c.width for c in detail_crops)
            resized_crops = []
            for c in detail_crops:
                if c.width != col_width:
                    scale = col_width / c.width
                    c = c.resize((col_width, max(1, int(c.height * scale))), Image.LANCZOS)
                resized_crops.append(c)

            col_height = sum(c.height for c in resized_crops) + gap * (len(resized_crops) - 1)

            resized_crops = self._fit_block_to_original(resized_crops, col_height, original_img.height)
            col_width = max(c.width for c in resized_crops)
            col_height = sum(c.height for c in resized_crops) + gap * (len(resized_crops) - 1)

            canvas_width = original_img.width + gap + col_width
            canvas_height = max(original_img.height, col_height)

            canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
            canvas.paste(original_img, (0, (canvas_height - original_img.height) // 2))

            x = original_img.width + gap
            y = (canvas_height - col_height) // 2
            for c in resized_crops:
                canvas.paste(c, (x, y))
                y += c.height + gap

            return canvas

        # Landscape/square: row of details below, each normalized to the same height.
        row_height = max(c.height for c in detail_crops)
        resized_crops = []
        for c in detail_crops:
            if c.height != row_height:
                scale = row_height / c.height
                c = c.resize((max(1, int(c.width * scale)), row_height), Image.LANCZOS)
            resized_crops.append(c)

        row_width = sum(c.width for c in resized_crops) + gap * (len(resized_crops) - 1)

        resized_crops = self._fit_block_to_original(resized_crops, row_width, original_img.width)
        row_height = max(c.height for c in resized_crops)
        row_width = sum(c.width for c in resized_crops) + gap * (len(resized_crops) - 1)

        canvas_width = max(original_img.width, row_width)
        canvas_height = original_img.height + gap + row_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

        orig_x = (canvas_width - original_img.width) // 2
        canvas.paste(original_img, (orig_x, 0))

        row_x = (canvas_width - row_width) // 2
        y = original_img.height + gap
        for c in resized_crops:
            canvas.paste(c, (row_x, y))
            row_x += c.width + gap

        return canvas

    def generate(self, image, api_key, **kwargs):
        try:
            client = self.get_client(api_key)

            num_details = kwargs.get("num_details", 3)
            detail_hint = kwargs.get("detail_hint", "")
            crop_scale = kwargs.get("crop_scale", 2.0)
            model = kwargs.get("model", "openai/gpt-4o")
            padding_percent = kwargs.get("padding_percent", 200.0)

            image_url = self.upload_image(client, image)
            logger.info(f"Uploaded source image: {image_url}")

            details = self._detect_details(client, model, image_url, num_details, detail_hint)
            logger.info(f"Detected {len(details)} details: {[d.get('label') for d in details]}")

            # Convert the source IMAGE tensor to a PIL image for local cropping.
            image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
            if image_np.ndim == 4:
                image_np = image_np[0]
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
            source_img = Image.fromarray(image_np).convert("RGB")

            kept_details = []
            detail_crops = []
            for d in details:
                crop = self._crop_and_scale_detail(source_img, d, crop_scale, padding_percent)
                if crop is not None:
                    kept_details.append(d)
                    detail_crops.append(crop)

            if not detail_crops:
                raise RuntimeError(
                    "All detected detail crops were discarded as blank/empty. "
                    "Try a more precise model (gemini-2.5-pro, gpt-4o) or a "
                    "higher padding_percent."
                )

            if len(detail_crops) < len(details):
                logger.warning(
                    f"Discarded {len(details) - len(detail_crops)} blank crop(s); "
                    f"kept {len(detail_crops)} of {len(details)} requested details."
                )

            composed = self._compose_detail_sheet(source_img, detail_crops)

            composed_np = np.array(composed).astype(np.float32) / 255.0
            composed_tensor = torch.from_numpy(composed_np).unsqueeze(0)

            info = json.dumps({
                "details": kept_details,
                "discarded_count": len(details) - len(detail_crops),
                "crop_scale": crop_scale,
                "model": model,
            })

            return (composed_tensor, info)

        except Exception as e:
            logger.error(f"Smart detail sheet failed: {str(e)}")
            raise RuntimeError(f"Smart detail sheet failed: {str(e)}") from e
