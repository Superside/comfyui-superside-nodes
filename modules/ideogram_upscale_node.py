import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class IdeogramUpscaleNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Ideogram Upscale Node: Upscale images using Ideogram's state-of-the-art
    image upscaling model via fal.ai API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional prompt to guide the upscaling",
                    },
                ),
                "resemblance": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                "detail": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                "expand_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "seed": ("INT", {"min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    DESCRIPTION = (
        "Upscale images using Ideogram's state-of-the-art image upscaling model"
    )

    def prepare_arguments(self, client, image, **kwargs):
        """Prepare arguments for the API call."""
        # Upload image and get URL
        image_url = self.upload_image(client, image)

        arguments = {
            "image_url": image_url,
            "resemblance": kwargs.get("resemblance", 50),
            "detail": kwargs.get("detail", 50),
        }

        # Add optional parameters if provided
        if kwargs.get("prompt"):
            arguments["prompt"] = kwargs["prompt"]

        if kwargs.get("expand_prompt") is not None:
            arguments["expand_prompt"] = kwargs["expand_prompt"]

        if kwargs.get("seed") and kwargs["seed"] != 0:
            arguments["seed"] = kwargs["seed"]

        return arguments

    def upscale(self, image, api_key, **kwargs):
        """Main upscaling function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, image, **kwargs)
            result = self.call_api(client, "fal-ai/ideogram/upscale", arguments)
            return self.process_images(result)
        except Exception as e:
            logger.error(f"Upscaling failed: {str(e)}")
            raise
