import logging

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SeedreamV5ProEditNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Seedream V5 Pro Edit Node: Edit images using bytedance/seedream/v5/pro/edit.

    Grounded, region-precise image editing that changes one element while
    keeping the rest of the frame intact. Supports up to 10 reference images.
    """

    IMAGE_SIZE_OPTIONS = [
        "auto_1K",
        "auto_2K",
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
                        "default": 1024,
                        "min": 64,
                        "max": 2048,
                        "step": 16,
                        "tooltip": "Used when size_mode is custom. Total pixels (width x height) must stay between 1024x1024 and 2048x2048, with aspect ratio between 1/16 and 16.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 2048,
                        "step": 16,
                        "tooltip": "Used when size_mode is custom. Total pixels (width x height) must stay between 1024x1024 and 2048x2048, with aspect ratio between 1/16 and 16.",
                    },
                ),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
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
        "Edit images using ByteDance Seedream V5 Pro on fal.ai. "
        "Grounded, region-precise editing with up to 10 reference images."
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
            raise ValueError("At least one image is required for Seedream V5 Pro editing")

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
        }

        size_mode = kwargs.get("size_mode", "preset")
        if size_mode == "custom":
            arguments["image_size"] = {
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
            }
        else:
            arguments["image_size"] = kwargs.get("image_size", "auto_2K")

        if kwargs.get("output_format"):
            arguments["output_format"] = kwargs["output_format"]

        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("enable_safety_checker") is not None:
            arguments["enable_safety_checker"] = kwargs["enable_safety_checker"]

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, prompt, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "bytedance/seedream/v5/pro/edit", arguments)

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
            logger.error(f"Seedream V5 Pro edit failed: {str(e)}")
            raise RuntimeError(f"Seedream V5 Pro edit failed: {str(e)}") from e
