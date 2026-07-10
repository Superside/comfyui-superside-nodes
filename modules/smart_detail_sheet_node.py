import json
import logging

import numpy as np
import torch
from PIL import Image

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    DetailSheetCompositionMixin,
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
    SupersideFalNode, ImageProcessingMixin, APIClientMixin, DetailSheetCompositionMixin
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
                "product_category": (
                    ["auto", "eyewear"],
                    {
                        "default": "auto",
                        "tooltip": "\"auto\" lets the model freely pick whichever details look most interesting. \"eyewear\" instead forces exactly 3 fixed, reliable zones every time: the nose pad + its mounting clip, the bridge/hinge assembly (with any decorative hardware), and a temple tip - overriding num_details.",
                    },
                ),
                "num_details": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 6,
                        "tooltip": "How many detail close-ups to find and crop. Ignored when product_category is \"eyewear\" (always 3 fixed zones).",
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
                "crop_size_percent": (
                    "FLOAT",
                    {
                        "default": 35.0,
                        "min": 5.0,
                        "max": 100.0,
                        "tooltip": "Size of each detail crop as a percent of the original photo's shorter side, always a square centered on the detected detail. A fixed size (instead of expanding the model's own bounding box) keeps crops consistent and robust to imprecise/oddly-shaped boxes.",
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

    # Fixed, reliable zones for eyewear - located (in this order) via
    # Florence-2's grounding endpoint when product_category="eyewear",
    # instead of asking a general vision LLM to guess coordinates from text.
    # In testing, the LLM approach was unreliable for this specific task -
    # even with one isolated call per zone, it tended to collapse "nose pad"
    # and "bridge/hinge" onto the same spot. Florence-2 is a dedicated
    # grounding model (already used by SupersideFlorence2RegionSelectorNode)
    # and located these short, literal queries accurately and distinctly.
    # (query, display_label) - query is what's sent to Florence, label is
    # what shows up in the node's output/info.
    EYEWEAR_FIXED_ZONES = [
        ("nose pad", "nose pad and mounting clip"),
        ("hinge screw", "hinge screw joint"),
        ("temple tip", "temple tip"),
    ]

    # Florence-2 grounding often returns one or two near-whole-image boxes
    # as a weak first guess alongside genuinely specific ones. A candidate
    # box covering more than this fraction of the image's area is treated
    # as one of those broad guesses and skipped in favor of tighter boxes.
    FLORENCE_MAX_BOX_AREA_RATIO = 0.15

    # A returned box whose x-span or y-span (on the 0-1000 scale) is smaller
    # than this is treated as degenerate (effectively a single point, giving
    # the crop's center no real signal) and triggers a retry rather than
    # being used as-is.
    MIN_BOX_SPAN = 5.0

    # Two boxes whose centers (on the 0-1000 scale) are closer than this are
    # treated as the same real-world spot - their crops would otherwise be
    # near-duplicates of one area instead of two distinct details.
    MIN_CENTER_SEPARATION = 80.0

    def _request_details_json(self, client, model, image_url, prompt, expected_count, check_collisions):
        """
        Call the vision model with `prompt` and parse a JSON array of exactly
        `expected_count` detail boxes out of it, retrying with a sharper
        reminder if the model returns prose instead of JSON, a degenerate
        (zero-size) box, or - when check_collisions is True - two boxes
        whose centers land on nearly the same spot.
        """
        arguments = {
            "prompt": prompt,
            "image_urls": [image_url],
            "model": model,
            # Some models (e.g. gemini-2.5-pro) reject the primary endpoint
            # outright with "Reasoning is mandatory" unless this is set,
            # which otherwise forces a fallback to a less obedient endpoint.
            "reasoning": True,
        }

        MAX_ATTEMPTS = 2
        last_parse_error = None
        for attempt in range(MAX_ATTEMPTS):
            call_arguments = dict(arguments)
            if attempt > 0:
                call_arguments["prompt"] = (
                    arguments["prompt"] + "\n\nIMPORTANT: your previous response was rejected "
                    f"({last_parse_error}). Respond with NOTHING except the raw JSON array - "
                    "no headings, no bullet points, no explanation. Every box must have real "
                    "width and height, and every zone must be centered on a clearly different "
                    "spot on the frame - never return the same location twice."
                )

            result = None
            last_error = None
            for endpoint in self.ENDPOINT_CANDIDATES:
                try:
                    logger.info(f"Trying vision endpoint: {endpoint} (attempt {attempt + 1})")
                    result = self.call_api(client, endpoint, call_arguments)
                    break
                except Exception as endpoint_error:
                    last_error = endpoint_error
                    logger.warning(f"Endpoint failed: {endpoint} - {str(endpoint_error)}")

            if result is None:
                raise RuntimeError(f"All vision endpoints failed. Last error: {last_error}")

            output_text = result.get("output", "")
            logger.debug(f"Vision model raw output: {output_text}")

            # Some models append reasoning/notes after the JSON array. A
            # greedy regex from the first "[" to the last "]" would swallow
            # that trailing text and fail to parse, so instead decode just
            # the JSON value starting at the first "[" and ignore whatever
            # follows it.
            start = output_text.find("[")
            if start == -1:
                last_parse_error = RuntimeError(
                    f"Vision model did not return a parseable JSON array. Raw output: {output_text[:500]}"
                )
                logger.warning(f"Attempt {attempt + 1}: {last_parse_error}")
                continue

            try:
                details, _ = json.JSONDecoder().raw_decode(output_text, start)
            except json.JSONDecodeError as e:
                last_parse_error = RuntimeError(f"Failed to parse detail JSON from vision model: {e}")
                logger.warning(f"Attempt {attempt + 1}: {last_parse_error}")
                continue

            degenerate = [
                d for d in details
                if isinstance(d, dict)
                and (abs(float(d.get("x2", 0)) - float(d.get("x1", 0))) < self.MIN_BOX_SPAN
                     or abs(float(d.get("y2", 0)) - float(d.get("y1", 0))) < self.MIN_BOX_SPAN)
            ]
            if degenerate:
                last_parse_error = RuntimeError(
                    f"Vision model returned {len(degenerate)} zero-size/degenerate box(es): "
                    f"{[d.get('label') for d in degenerate]}"
                )
                logger.warning(f"Attempt {attempt + 1}: {last_parse_error}")
                continue

            if check_collisions:
                # Distinct zones (e.g. eyewear's nose pad vs. bridge/hinge)
                # should land on clearly different spots. If two boxes'
                # centers collapse onto nearly the same point, their crops
                # end up as near-duplicates of the same real-world area -
                # treat that as a failure worth retrying rather than
                # silently returning two copies of one detail.
                centers = [
                    ((float(d.get("x1", 0)) + float(d.get("x2", 0))) / 2.0,
                     (float(d.get("y1", 0)) + float(d.get("y2", 0))) / 2.0)
                    for d in details if isinstance(d, dict)
                ]
                collided = False
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        if (dx * dx + dy * dy) ** 0.5 < self.MIN_CENTER_SEPARATION:
                            collided = True
                            break
                    if collided:
                        break

                if collided:
                    last_parse_error = RuntimeError(
                        f"Vision model returned overlapping/duplicate zones: {[d.get('label') for d in details]}"
                    )
                    logger.warning(f"Attempt {attempt + 1}: {last_parse_error}")
                    continue

            break
        else:
            raise last_parse_error

        if not isinstance(details, list) or not details:
            raise RuntimeError("Vision model returned no usable details.")

        return details[:expected_count]

    def _detect_zone_via_florence(self, client, image_url, image_width, image_height, query, label):
        """
        Locate one short text query (e.g. "nose pad") with Florence-2's
        grounding endpoint and return it as a detail dict in this node's
        usual 0-1000 normalized x1/y1/x2/y2 format.
        """
        result = self.call_api(
            client,
            "fal-ai/florence-2-large/caption-to-phrase-grounding",
            {"image_url": image_url, "text_input": query},
        )
        bboxes = result.get("results", {}).get("bboxes", [])
        if not bboxes:
            raise RuntimeError(f"Florence-2 returned no bounding box for '{query}'.")

        image_area = image_width * image_height
        candidates = []
        for b in bboxes:
            if not isinstance(b, dict) or not all(k in b for k in ("x", "y", "w", "h")):
                continue
            area = float(b["w"]) * float(b["h"])
            candidates.append((area, b))

        if not candidates:
            raise RuntimeError(f"Florence-2 returned no usable bounding box for '{query}'.")

        # Florence often returns one or two broad, near-whole-image boxes as
        # a weak first guess alongside genuinely specific ones - prefer the
        # tightest box under the area threshold; only fall back to the
        # smallest overall if every candidate happens to be that broad.
        specific = [c for c in candidates if c[0] < image_area * self.FLORENCE_MAX_BOX_AREA_RATIO]
        area, best = min(specific or candidates, key=lambda c: c[0])

        x1 = float(best["x"])
        y1 = float(best["y"])
        x2 = x1 + float(best["w"])
        y2 = y1 + float(best["h"])

        return {
            "label": label,
            "x1": x1 / image_width * 1000.0,
            "y1": y1 / image_height * 1000.0,
            "x2": x2 / image_width * 1000.0,
            "y2": y2 / image_height * 1000.0,
        }

    def _detect_details(
        self, client, model, image_url, num_details, detail_hint, product_category,
        image_width=None, image_height=None,
    ):
        if product_category == "eyewear":
            return [
                self._detect_zone_via_florence(client, image_url, image_width, image_height, query, label)
                for query, label in self.EYEWEAR_FIXED_ZONES
            ]

        hint_line = f"\n\nAdditional guidance: {detail_hint.strip()}" if detail_hint and detail_hint.strip() else ""
        prompt = (
            f"Analyze this product photo and identify exactly {num_details} of the most "
            "visually interesting close-up details worth showcasing to a buyer.\n\n"
            "First silently identify what kind of product this is, then pick the "
            "details that matter most for that specific category. For example: on "
            "eyewear, prioritize: the joints/unions where parts connect (e.g. where "
            "the temple meets the front); the bridge together with its nose pads "
            "(the piece over the nose - always include the nose pads when framing "
            "the bridge); the nose pad's own mounting clip/arm where it attaches to "
            "the lens or frame; the temple arms (the full pieces extending back to "
            "the ears, not just their tips); the points where the lens itself "
            "mounts directly to the metal frame via small screws (common on "
            "rimless/drill-mount designs); and any decorative or branded hardware "
            "such as ornamental rivets, shaped screws (hearts, logos, engravings) "
            "at the hinge or bridge. On footwear, prioritize stitching, sole "
            "texture, laces, and logos; on bags, prioritize hardware, zippers, and "
            "seams; on apparel, prioritize fabric weave, seams, buttons, and "
            "zippers. More generally, look for textures, materials, logos, joints, "
            f"pads, seams, stitching, or other unique design elements.{hint_line}\n\n"
            "Return ONLY a JSON array, no other text, no markdown code fences, in "
            "this exact format:\n"
            '[{"label": "short description", "x1": 0, "y1": 0, "x2": 0, "y2": 0}, ...]\n\n'
            "Coordinates are normalized to a 0-1000 scale where (0,0) is the "
            "top-left corner and (1000,1000) is the bottom-right corner of the "
            "image. Each box should tightly frame just that one detail - padding "
            "for context is added separately afterwards. Each box MUST have "
            "real width and height - x2 noticeably greater than x1, and y2 "
            "noticeably greater than y1 (at least 40 units apart on the "
            "0-1000 scale). Never return a single point or a zero-size box. "
            f"Return exactly {num_details} entries."
        )
        return self._request_details_json(
            client, model, image_url, prompt, expected_count=num_details, check_collisions=True
        )

    def _crop_and_scale_detail(self, source_img, detail, crop_scale, crop_size_percent):
        width, height = source_img.size

        x1_raw = float(detail.get("x1", 0))
        y1_raw = float(detail.get("y1", 0))
        x2_raw = float(detail.get("x2", 1000))
        y2_raw = float(detail.get("y2", 1000))
        if x2_raw <= x1_raw:
            x1_raw, x2_raw = x2_raw, x1_raw
        if y2_raw <= y1_raw:
            y1_raw, y2_raw = y2_raw, y1_raw

        # Center point of the detected box, in pixel coordinates - not the
        # box's own width/height. This makes the crop robust to imprecise or
        # oddly-shaped bounding boxes: even when the model's box is too
        # tight, too wide, or lopsided, the crop still expands symmetrically
        # around roughly the right spot instead of inheriting the box's
        # (possibly wrong) shape.
        cx = (x1_raw + x2_raw) / 2.0 / 1000.0 * width
        cy = (y1_raw + y2_raw) / 2.0 / 1000.0 * height

        # Fixed square size, as a percent of the image's shorter side, so
        # every detail crop is a consistent, predictable size regardless of
        # how large or small the detected box happened to be.
        size = max(2.0, crop_size_percent / 100.0 * min(width, height))
        half = size / 2.0

        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half

        # Keep the crop fully inside the image while staying square - shift
        # it instead of clipping, so it never gets squashed into a thin
        # rectangle when the center point is near an edge.
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if x2 > width:
            x1 -= (x2 - width)
            x2 = float(width)
        if y2 > height:
            y1 -= (y2 - height)
            y2 = float(height)

        return self._crop_rect_scale_check_blank(
            source_img, x1, y1, x2, y2, crop_scale, label=detail.get("label", "?")
        )

    def generate(self, image, api_key, **kwargs):
        try:
            client = self.get_client(api_key)

            product_category = kwargs.get("product_category", "auto")
            num_details = kwargs.get("num_details", 3)
            detail_hint = kwargs.get("detail_hint", "")
            crop_scale = kwargs.get("crop_scale", 2.0)
            model = kwargs.get("model", "openai/gpt-4o")
            crop_size_percent = kwargs.get("crop_size_percent", 35.0)

            image_url = self.upload_image(client, image)
            logger.info(f"Uploaded source image: {image_url}")

            # Convert the source IMAGE tensor to a PIL image for local cropping
            # (and for eyewear-mode Florence-2 detection, which needs pixel
            # dimensions to normalize its boxes).
            source_img = self._tensor_to_pil(image)

            details = self._detect_details(
                client, model, image_url, num_details, detail_hint, product_category,
                source_img.width, source_img.height,
            )
            logger.info(f"Detected {len(details)} details: {[d.get('label') for d in details]}")

            kept_details = []
            detail_crops = []
            for d in details:
                crop = self._crop_and_scale_detail(source_img, d, crop_scale, crop_size_percent)
                if crop is not None:
                    kept_details.append(d)
                    detail_crops.append(crop)

            if not detail_crops:
                raise RuntimeError(
                    "All detected detail crops were discarded as blank/empty. "
                    "Try a more precise model (gemini-2.5-pro, gpt-4o) or a "
                    "different crop_size_percent."
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
