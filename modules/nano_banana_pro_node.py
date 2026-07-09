import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NanoBananaProEditNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Nano Banana Pro Edit Node: Edit images using nano-banana-pro/edit
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
                "aspect_ratio": (
                    [
                        "auto",
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
                    ],
                    {"default": "auto"},
                ),
                "output_format": (["jpeg", "png", "webp"], {"default": "png"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Edit images using nano-banana-pro/edit with up to 6 image "
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
            raise ValueError(
                "At least one image is required for Nano Banana Pro editing"
            )

        arguments = {"prompt": prompt, "image_urls": image_urls}

        # Add optional parameters if provided
        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("aspect_ratio"):
            arguments["aspect_ratio"] = kwargs["aspect_ratio"]

        if kwargs.get("output_format"):
            arguments["output_format"] = kwargs["output_format"]

        if kwargs.get("resolution"):
            arguments["resolution"] = kwargs["resolution"]

        if kwargs.get("sync_mode"):
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, prompt, api_key, **kwargs):
        """Main generation function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "fal-ai/nano-banana-pro/edit", arguments)

            # Process images
            images = self.process_images(result)

            # Extract description from result
            description = result.get("description", "")

            return (images[0], description)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
