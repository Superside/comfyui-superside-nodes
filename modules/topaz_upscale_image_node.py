import logging

from .base_node import (
    APIClientMixin,
    SupersideFalNode,
    ImageProcessingMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class TopazUpscaleImageNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Topaz Upscale Image Node: Upscale or enhance images with Topaz models
    via fal.ai API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "model": (
                    [
                        "Low Resolution V2",
                        "Standard V2",
                        "CGI",
                        "High Fidelity V2",
                        "Text Refine",
                        "Recovery",
                        "Redefine",
                        "Recovery V2",
                        "Standard MAX",
                        "Wonder",
                    ],
                    {"default": "Standard V2"},
                ),
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "crop_to_fill": ("BOOLEAN", {"default": False}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "subject_detection": (
                    ["All", "Foreground", "Background"],
                    {"default": "All"},
                ),
                "face_enhancement": ("BOOLEAN", {"default": True}),
                "face_enhancement_creativity": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "face_enhancement_strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "sharpen": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "fix_compression": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.99,
                        "display": "slider",
                    },
                ),
                "creativity": ("INT", {"default": 1, "min": 1, "max": 6}),
                "texture": ("INT", {"default": 5, "min": 1, "max": 5}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Optional prompt for Redefine model edits",
                    },
                ),
                "autoprompt": ("BOOLEAN", {"default": True}),
                "detail": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    DISPLAY_NAME = "Topaz Upscale Image"
    DESCRIPTION = "Upscale images using Topaz enhancement models"

    def prepare_arguments(self, client, image, **kwargs):
        """Prepare arguments for the API call."""
        arguments = {
            "image_url": self.upload_image(client, image),
        }

        for key in (
            "model",
            "upscale_factor",
            "crop_to_fill",
            "output_format",
            "subject_detection",
            "face_enhancement",
            "face_enhancement_creativity",
            "face_enhancement_strength",
            "sharpen",
            "denoise",
            "fix_compression",
            "strength",
            "creativity",
            "texture",
            "autoprompt",
            "detail",
        ):
            if kwargs.get(key) is not None:
                arguments[key] = kwargs[key]

        if kwargs.get("prompt"):
            arguments["prompt"] = kwargs["prompt"]

        return arguments

    def process_result(self, result):
        """Process Topaz's single-image response format."""
        if "image" not in result or not result["image"]:
            raise RuntimeError("No image was generated by the API.")

        img_info = result["image"]
        logger.debug(f"Processing Topaz image: {img_info}")

        if not isinstance(img_info, dict) or "url" not in img_info:
            raise RuntimeError("Invalid image format in response")

        return self.process_images({"images": [img_info]})

    def upscale(self, image, api_key, **kwargs):
        """Main upscaling function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, image, **kwargs)
            logger.info(f"Calling Topaz Upscale API with arguments: {arguments}")

            result = self.call_api(client, "fal-ai/topaz/upscale/image", arguments)
            return self.process_result(result)
        except Exception as e:
            logger.error(f"Topaz upscaling failed: {str(e)}")
            raise
