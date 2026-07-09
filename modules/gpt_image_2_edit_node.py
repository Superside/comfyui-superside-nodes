import logging

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GPTImage2EditNode(SupersideFalNode, ImageProcessingMixin, APIClientMixin):
    """
    GPT Image 2 Edit Node: Edit images using openai/gpt-image-2/edit.

    Supports one or more reference images plus an optional mask image for more
    precise inpainting-style edits.
    """

    IMAGE_SIZE_OPTIONS = [
        "auto",
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
    ]

    ASPECT_RATIO_OPTIONS = [
        "1:1",
        "4:5",
        "5:4",
        "4:3",
        "3:4",
        "16:9",
        "9:16",
        "3:2",
        "2:3",
    ]

    RESOLUTION_OPTIONS = ["1K", "2K", "4K"]

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
                "mask_image": ("IMAGE",),
                "size_mode": (["preset", "aspect_ratio", "custom"], {"default": "preset"}),
                "image_size": (cls.IMAGE_SIZE_OPTIONS, {"default": "auto"}),
                "aspect_ratio": (cls.ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "resolution": (cls.RESOLUTION_OPTIONS, {"default": "1K"}),
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 16,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Used when size_mode is custom. Must be a multiple of 16.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 16,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Used when size_mode is custom. Must be a multiple of 16.",
                    },
                ),
                "quality": (["low", "medium", "high"], {"default": "high"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Edit images using OpenAI GPT Image 2 on fal. "
        "Supports multi-image references, mask-based editing, preset sizes, "
        "aspect-ratio + resolution sizing, and custom output sizes."
    )

    def _round_to_multiple_of_16(self, value):
        """Round a numeric dimension to the nearest lower multiple of 16."""
        return max(16, int(value) // 16 * 16)

    def _calculate_image_size_from_aspect_ratio(self, aspect_ratio, resolution):
        """Convert aspect ratio + resolution preset to custom dimensions."""
        long_edge_map = {
            "1K": 1024,
            "2K": 2048,
            "4K": 4096,
        }
        long_edge = long_edge_map[resolution]

        width_ratio, height_ratio = aspect_ratio.split(":")
        width_ratio = int(width_ratio)
        height_ratio = int(height_ratio)

        if width_ratio >= height_ratio:
            width = long_edge
            height = long_edge * height_ratio / width_ratio
        else:
            height = long_edge
            width = long_edge * width_ratio / height_ratio

        width = self._round_to_multiple_of_16(width)
        height = self._round_to_multiple_of_16(height)

        return {
            "width": width,
            "height": height,
        }

    def prepare_image_urls(self, client, **kwargs):
        """Prepare list of image URLs from input images."""
        image_urls = []

        for i in range(1, 7):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                try:
                    url = self.upload_image(client, kwargs[image_key])
                    image_urls.append(url)
                    logger.info(f"Uploaded {image_key}: {url}")
                except Exception as e:
                    logger.warning(f"Failed to upload {image_key}: {str(e)}")
                    continue

        return image_urls

    def prepare_arguments(self, client, prompt, **kwargs):
        """Prepare arguments for the API call."""
        image_urls = self.prepare_image_urls(client, **kwargs)
        if not image_urls:
            raise ValueError("At least one image is required for GPT Image 2 editing")

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
        }

        if kwargs.get("mask_image") is not None:
            arguments["mask_image_url"] = self.upload_image(client, kwargs["mask_image"])

        size_mode = kwargs.get("size_mode", "preset")
        if size_mode == "aspect_ratio":
            arguments["image_size"] = self._calculate_image_size_from_aspect_ratio(
                kwargs.get("aspect_ratio", "16:9"),
                kwargs.get("resolution", "1K"),
            )
        elif size_mode == "custom":
            width = kwargs.get("width", 1920)
            height = kwargs.get("height", 1080)
            if width % 16 != 0 or height % 16 != 0:
                raise ValueError("Custom width and height must both be multiples of 16")
            arguments["image_size"] = {
                "width": width,
                "height": height,
            }
        elif kwargs.get("image_size") is not None:
            arguments["image_size"] = kwargs["image_size"]

        if kwargs.get("quality") is not None:
            arguments["quality"] = kwargs["quality"]

        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("output_format") is not None:
            arguments["output_format"] = kwargs["output_format"]

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, prompt, api_key, **kwargs):
        """Main image editing function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "openai/gpt-image-2/edit", arguments)

            images = self.process_images(result)
            info = ""
            if result.get("images") and isinstance(result["images"], list):
                first_url = result["images"][0].get("url", "")
                if isinstance(first_url, str) and first_url.startswith("data:"):
                    info = "data-uri image returned"
                else:
                    info = first_url

            return (images[0], info)
        except Exception as e:
            logger.error(f"GPT Image 2 edit failed: {str(e)}")
            raise RuntimeError(f"GPT Image 2 edit failed: {str(e)}") from e
