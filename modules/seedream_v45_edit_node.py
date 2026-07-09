import logging

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SupersideSeedreamV45EditNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Seedream V4.5 Edit Node: Edit images using fal-ai/bytedance/seedream/v4.5/edit.

    Supports up to 10 reference images and preset or custom output sizes.
    """

    IMAGE_SIZE_OPTIONS = [
        "auto_2K",
        "auto_4K",
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
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
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "size_mode": (
                    ["preset", "custom"],
                    {"default": "preset"},
                ),
                "image_size": (cls.IMAGE_SIZE_OPTIONS, {"default": "auto_2K"}),
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 64,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Used when size_mode is custom. API requires min 1920.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 64,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Used when size_mode is custom. API requires min 1920.",
                    },
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "-1 = random"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Edit images using ByteDance Seedream V4.5 on fal.ai. "
        "Supports up to 10 reference images, preset and custom output sizes."
    )

    def prepare_image_urls(self, client, **kwargs):
        image_urls = []
        for i in range(1, 11):
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
            raise ValueError("At least one image is required for Seedream V4.5 editing")

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
        }

        size_mode = kwargs.get("size_mode", "preset")
        if size_mode == "custom":
            arguments["image_size"] = {
                "width": kwargs.get("width", 1920),
                "height": kwargs.get("height", 1080),
            }
        else:
            arguments["image_size"] = kwargs.get("image_size", "auto_2K")

        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("max_images") is not None:
            arguments["max_images"] = kwargs["max_images"]

        seed = kwargs.get("seed", -1)
        if seed is not None and seed != -1:
            arguments["seed"] = seed

        if kwargs.get("enable_safety_checker") is not None:
            arguments["enable_safety_checker"] = kwargs["enable_safety_checker"]

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, prompt, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "fal-ai/bytedance/seedream/v4.5/edit", arguments)

            images = self.process_images(result)
            info = ""
            if result.get("images") and isinstance(result["images"], list):
                info = result["images"][0].get("url", "")

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
                        node["widgets_values"] = [info]

            return {"ui": {"text": [info]}, "result": (images[0], info)}
        except Exception as e:
            logger.error(f"Seedream V4.5 edit failed: {str(e)}")
            raise RuntimeError(f"Seedream V4.5 edit failed: {str(e)}") from e
