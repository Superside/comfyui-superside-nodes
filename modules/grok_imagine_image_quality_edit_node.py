import logging

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GrokImagineImageQualityEditNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Grok Imagine Image Quality Edit Node: Edit images using xai/grok-imagine-image/quality/edit.

    Supports up to 3 reference images and returns edited images with a revised prompt.
    """

    ASPECT_RATIO_OPTIONS = [
        "auto",
        "1:1",
        "2:1",
        "20:9",
        "19.5:9",
        "16:9",
        "4:3",
        "3:2",
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
                "output_format": (["jpeg", "png", "webp"], {"default": "jpeg"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "sync_mode": ("BOOLEAN", {"default": False}),
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
        "Edit images using xAI Grok Imagine (quality) on fal.ai. "
        "Supports up to 3 reference images, aspect ratio control, and 1K/2K output resolution."
    )

    def prepare_image_urls(self, client, **kwargs):
        image_urls = []
        for i in range(1, 4):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                try:
                    url = self.upload_image(client, kwargs[image_key])
                    image_urls.append(url)
                    logger.info(f"Uploaded {image_key}: {url}")
                except Exception as e:
                    logger.warning(f"Failed to upload {image_key}: {str(e)}")
        return image_urls

    def prepare_arguments(self, client, prompt, **kwargs):
        image_urls = self.prepare_image_urls(client, **kwargs)
        if not image_urls:
            raise ValueError("At least one image is required for Grok Imagine editing")

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
        }

        if kwargs.get("aspect_ratio") is not None:
            arguments["aspect_ratio"] = kwargs["aspect_ratio"]

        if kwargs.get("resolution") is not None:
            arguments["resolution"] = kwargs["resolution"]

        if kwargs.get("output_format") is not None:
            arguments["output_format"] = kwargs["output_format"]

        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, prompt, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "xai/grok-imagine-image/quality/edit", arguments)

            images = self.process_images(result)
            revised_prompt = result.get("revised_prompt", "")

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
            logger.error(f"Grok Imagine edit failed: {str(e)}")
            raise RuntimeError(f"Grok Imagine edit failed: {str(e)}") from e
