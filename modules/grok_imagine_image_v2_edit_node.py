import logging

import numpy as np
import torch
from PIL import Image

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SupersideGrokImagineImageV2EditNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Grok Imagine Image v2 Edit Node: Edit images using xai/grok-imagine-image/v2.0/edit.

    Same shape as the quality/edit endpoint plus a `quality` level (low/medium).
    Supports up to 3 reference images and returns edited images with a revised prompt.
    """

    ENDPOINT = "xai/grok-imagine-image/v2.0/edit"

    # Grok only accepts aspect_ratio + resolution - unlike GPT Image 2 there is
    # no width/height, so it cannot be asked for the input's exact pixel size.
    # Crop-stitch inpainting needs exactly that: the stitch node rescales the
    # edit straight onto the crop rectangle, so an edit with a different aspect
    # ratio comes back visibly stretched. MATCH_INPUT restores the contract by
    # fitting the result to image_1 locally.
    MATCH_INPUT = "match input image_1 (crop-stitch safe)"
    FAL_NATIVE = "fal native (aspect_ratio + resolution)"
    OUTPUT_SIZE_OPTIONS = [MATCH_INPUT, FAL_NATIVE]

    ASPECT_RATIO_OPTIONS = [
        "auto",
        "2:1",
        "20:9",
        "19.5:9",
        "16:9",
        "4:3",
        "3:2",
        "1:1",
        "2:3",
        "3:4",
        "9:16",
        "9:19.5",
        "9:20",
        "1:2",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Describe the edit you want to apply",
                    },
                ),
                "image_1": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "aspect_ratio": (cls.ASPECT_RATIO_OPTIONS, {"default": "auto"}),
                "resolution": (["1k", "2k"], {"default": "1k"}),
                "quality": (["low", "medium"], {"default": "medium"}),
                "output_format": (["jpeg", "png", "webp"], {"default": "jpeg"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "output_size": (
                    cls.OUTPUT_SIZE_OPTIONS,
                    {
                        "default": cls.MATCH_INPUT,
                        "tooltip": (
                            "'match input image_1' returns the edit at image_1's exact "
                            "pixel size, which is what crop-stitch inpainting needs - the "
                            "stitch node rescales the edit straight onto the crop rectangle, "
                            "so any other shape gets stretched. Scaled to cover and "
                            "centre-cropped, never squashed. Keep aspect_ratio on 'auto' so "
                            "there is almost nothing to crop. 'fal native' returns whatever "
                            "aspect_ratio + resolution produced."
                        ),
                    },
                ),
                "reference_max_dimension": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "tooltip": (
                            "Cap the long side of the REFERENCE images (image_2, image_3) "
                            "before upload; image_1 always goes at full size. Grok's API "
                            "takes one flat list of 'images to edit' with no designated "
                            "base, so a reference far larger and crisper than image_1 can "
                            "end up driving the output - a product reference at 4700px "
                            "against a 2700px crop invites Grok to return the product "
                            "rather than the edited crop. 0 disables the cap."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "revised_prompt")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Edit images using xAI Grok Imagine v2.0 on fal.ai. "
        "Supports up to 3 reference images, aspect ratio control, 1K/2K output "
        "resolution and a low/medium quality level. "
        "Drop-in for crop-stitch inpainting: output_size defaults to matching "
        "image_1's exact pixel size (aspect preserved), which is what the stitch "
        "node needs to paste the edit back without stretching it."
    )

    def prepare_image_urls(self, client, **kwargs):
        # image_1 is the image being edited and always goes at full size; the
        # references can be capped so they do not out-resolve it.
        reference_cap = int(kwargs.get("reference_max_dimension") or 0) or None

        image_urls = []
        for i in range(1, 4):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                cap = None if i == 1 else reference_cap
                try:
                    url = self.upload_image(client, kwargs[image_key], max_dimension=cap)
                    image_urls.append(url)
                    if cap:
                        logger.info("Uploaded %s capped at %dpx: %s", image_key, cap, url)
                    else:
                        logger.info(f"Uploaded {image_key}: {url}")
                except Exception as e:
                    logger.warning(f"Failed to upload {image_key}: {str(e)}")
        return image_urls

    @staticmethod
    def _cover_center_crop(frame_np, target_w, target_h):
        """
        Fit one HWC float frame to target_w x target_h without distorting it.

        Scales by a single factor so the frame covers the target, then takes the
        centre. Squashing the frame to the target instead would deform faces and
        product edges, which is exactly the artefact this option exists to stop.
        """
        source_h, source_w = frame_np.shape[0], frame_np.shape[1]
        pil = Image.fromarray((np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8))

        scale = max(target_w / source_w, target_h / source_h)
        scaled_w = max(target_w, int(round(source_w * scale)))
        scaled_h = max(target_h, int(round(source_h * scale)))
        pil = pil.resize((scaled_w, scaled_h), Image.LANCZOS)

        left = (scaled_w - target_w) // 2
        top = (scaled_h - target_h) // 2
        pil = pil.crop((left, top, left + target_w, top + target_h))

        cropped_fraction = 1.0 - (target_w * target_h) / float(scaled_w * scaled_h)
        return np.array(pil).astype(np.float32) / 255.0, cropped_fraction

    def match_input_size(self, images, reference):
        """
        Return `images` at `reference`'s exact pixel size (aspect preserved).

        `images` is the (tensor,) tuple from process_images; `reference` is the
        IMAGE tensor wired into image_1.
        """
        target_h, target_w = int(reference.shape[1]), int(reference.shape[2])
        tensor = images[0]
        source_h, source_w = int(tensor.shape[1]), int(tensor.shape[2])

        if (source_w, source_h) == (target_w, target_h):
            logger.info(
                "Grok output already matches image_1 (%dx%d) - no resize needed",
                target_w, target_h,
            )
            return images

        frames = []
        cropped_fraction = 0.0
        for frame in tensor.detach().cpu().numpy():
            fitted, fraction = self._cover_center_crop(frame, target_w, target_h)
            frames.append(fitted)
            cropped_fraction = max(cropped_fraction, fraction)

        logger.info(
            "Matched Grok output %dx%d to image_1 %dx%d (aspect preserved, %.1f%% "
            "of the generated area cropped away)",
            source_w, source_h, target_w, target_h, cropped_fraction * 100.0,
        )
        if cropped_fraction > 0.08:
            logger.warning(
                "Grok returned a noticeably different aspect ratio than image_1, so "
                "%.1f%% of the edit was cropped to avoid stretching it. Set "
                "aspect_ratio to 'auto' to have Grok follow image_1's own ratio.",
                cropped_fraction * 100.0,
            )

        return (torch.from_numpy(np.stack(frames, axis=0)),)

    def prepare_arguments(self, client, prompt, **kwargs):
        image_urls = self.prepare_image_urls(client, **kwargs)
        if not image_urls:
            raise ValueError("At least one image is required for Grok Imagine editing")

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
        }

        # output_size is applied locally to the response, not sent to fal.
        for key in (
            "aspect_ratio",
            "resolution",
            "quality",
            "output_format",
            "num_images",
            "sync_mode",
        ):
            if kwargs.get(key) is not None:
                arguments[key] = kwargs[key]

        return arguments

    def generate(self, prompt, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, self.ENDPOINT, arguments)

            images = self.process_images(result)

            output_size = kwargs.get("output_size") or self.MATCH_INPUT
            if output_size == self.MATCH_INPUT and kwargs.get("image_1") is not None:
                images = self.match_input_size(images, kwargs["image_1"])

            revised_prompt = result.get("revised_prompt") or ""

            if unique_id is not None and extra_pnginfo is not None:
                if (
                    isinstance(extra_pnginfo, list)
                    and isinstance(extra_pnginfo[0], dict)
                    and "workflow" in extra_pnginfo[0]
                ):
                    workflow = extra_pnginfo[0]["workflow"]
                    node = next(
                        (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id)),
                        None,
                    )
                    if node:
                        node["widgets_values"] = [revised_prompt]

            return {"ui": {"text": [revised_prompt]}, "result": (images[0], revised_prompt)}
        except Exception as e:
            logger.error(f"Grok Imagine v2 edit failed: {str(e)}")
            raise RuntimeError(f"Grok Imagine v2 edit failed: {str(e)}") from e
