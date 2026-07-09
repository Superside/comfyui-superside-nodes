import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FluxKontextMaxMultiImageNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Flux Kontext Multi-Image Node: Generate images using FLUX.1 Kontext [Max]
    with multiple image inputs for context-aware generation.
    This node allows up to 4 image inputs for advanced context-aware image
    generation using fal.ai API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Put the little duckling on top of the "
                        "woman's t-shirt.",
                        "placeholder": "Enter your prompt here",
                    },
                ),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "seed": ("INT", {"min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 1, "max": 40.0, "step": 0.5},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
                "aspect_ratio": (
                    ["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"],
                    {"default": "3:4"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    DESCRIPTION = (
        "Generate images using FLUX.1 Kontext [Max] with multiple image "
        "inputs for context-aware generation"
    )

    def prepare_image_urls(self, client, **kwargs):
        """Prepare list of image URLs from input images."""
        image_urls = []

        # Check for up to 4 images
        for i in range(1, 5):
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
                "At least one image is required for Flux Kontext generation"
            )

        arguments = {"prompt": prompt, "image_urls": image_urls}

        # Add optional parameters if provided
        if kwargs.get("seed") and kwargs["seed"] != 0:
            arguments["seed"] = kwargs["seed"]

        if kwargs.get("guidance_scale"):
            arguments["guidance_scale"] = kwargs["guidance_scale"]

        if kwargs.get("num_images"):
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("safety_tolerance"):
            arguments["safety_tolerance"] = kwargs["safety_tolerance"]

        if kwargs.get("output_format"):
            arguments["output_format"] = kwargs["output_format"]

        if kwargs.get("aspect_ratio"):
            arguments["aspect_ratio"] = kwargs["aspect_ratio"]

        return arguments

    def generate(self, prompt, api_key, **kwargs):
        """Main generation function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "fal-ai/flux-pro/kontext/max/multi", arguments)
            return self.process_images(result)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
