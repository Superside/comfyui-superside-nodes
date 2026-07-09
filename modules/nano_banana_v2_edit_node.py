import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NanoBananaV2EditNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Nano Banana V2 Edit Node: Edit images using nano-banana-2/edit
    with multiple image inputs for context-aware image editing.
    This node allows up to 6 image inputs for advanced image editing
    using fal.ai API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Enter your editing prompt here",
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
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": None, "min": 0, "max": 2147483647}),
                "aspect_ratio": (
                    [
                        "21:9",
                        "16:9",
                        "3:2",
                        "4:3",
                        "5:4",
                        "1:1",
                        "4:5",
                        "3:4",
                        "2:3",
                        "9:16",
                        "4:1",
                        "1:4",
                        "8:1",
                        "1:8",
                        "auto",
                    ],
                    {"default": "auto"},
                ),
                "output_format": (["jpeg", "png", "webp"], {"default": "png"}),
                "safety_tolerance": ("INT", {"default": 4, "min": 1, "max": 6}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "limit_generations": ("BOOLEAN", {"default": True}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "thinking_level": (["none", "minimal", "high"], {"default": "none"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Edit images using nano-banana-2/edit with up to 6 image "
        "inputs for context-aware image editing"
    )

    def prepare_image_urls(self, client, **kwargs):
        """Prepare list of image URLs from input images."""
        image_urls = []

        # Check for up to 6 images
        for i in range(1, 7):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                try:
                    url = self.upload_image(client, kwargs[image_key])
                    image_urls.append(url)
                except Exception as e:
                    logger.warning(f"Failed to upload {image_key}: {str(e)}")
                    continue

        return image_urls

    def prepare_arguments(self, client, prompt, **kwargs):
        """Prepare arguments for the API call."""
        # Get image URLs
        image_urls = self.prepare_image_urls(client, **kwargs)

        if not image_urls:
            raise ValueError("At least one image is required for Nano Banana 2 editing")

        arguments = {"prompt": prompt, "image_urls": image_urls}

        # Add optional parameters if provided
        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("seed") is not None:
            arguments["seed"] = kwargs["seed"]

        if kwargs.get("aspect_ratio") is not None:
            arguments["aspect_ratio"] = kwargs["aspect_ratio"]

        if kwargs.get("output_format"):
            arguments["output_format"] = kwargs["output_format"]

        if kwargs.get("safety_tolerance") is not None:
            arguments["safety_tolerance"] = kwargs["safety_tolerance"]

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        if kwargs.get("resolution") is not None:
            arguments["resolution"] = kwargs["resolution"]

        if kwargs.get("limit_generations") is not None:
            arguments["limit_generations"] = kwargs["limit_generations"]

        if kwargs.get("enable_web_search") is not None:
            arguments["enable_web_search"] = kwargs["enable_web_search"]

        if kwargs.get("thinking_level") != "none":
            arguments["thinking_level"] = kwargs["thinking_level"]

        return arguments

    def generate(self, prompt, api_key, **kwargs):
        """Main generation function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "fal-ai/nano-banana-2/edit", arguments)

            # Process images
            images = self.process_images(result)

            # Extract description from result
            description = result.get("description", "")

            return (images[0], description)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
