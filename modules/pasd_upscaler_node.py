import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SupersidePASDUpscalerNode(SupersideFalNode, ImageProcessingMixin, APIClientMixin):
    """
    PASD Upscaler Node: Perform high-quality image super-resolution using PASD-SDXL.

    The model applies AI-enhanced super-resolution with:
    - Pixel-aware stable diffusion
    - ControlNet guidance for structure preservation
    - Wavelet color correction for natural results
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "scale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "steps": ("INT", {"default": 25, "min": 10, "max": 50, "step": 1}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "conditioning_scale": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Additional prompt to guide "
                        "super-resolution (optional)",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry, dirty, messy, frames, deformed, dotted, "
                        "noise, raster lines, unclear, lowres, over-smoothed, "
                        "painting, ai generated",
                        "placeholder": "Negative prompt to avoid unwanted artifacts",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    DESCRIPTION = (
        "Perform high-quality image super-resolution using PASD-SDXL with "
        "pixel-aware stable diffusion and ControlNet guidance"
    )

    def prepare_arguments(self, client, image, **kwargs):
        """Prepare arguments for the API call."""
        # Upload the input image
        image_url = self.upload_image(client, image)

        arguments = {"image_url": image_url}

        # Add optional parameters if provided
        if kwargs.get("scale"):
            arguments["scale"] = kwargs["scale"]

        if kwargs.get("steps"):
            arguments["steps"] = kwargs["steps"]

        if kwargs.get("guidance_scale"):
            arguments["guidance_scale"] = kwargs["guidance_scale"]

        if kwargs.get("conditioning_scale"):
            arguments["conditioning_scale"] = kwargs["conditioning_scale"]

        if kwargs.get("prompt") and kwargs["prompt"].strip():
            arguments["prompt"] = kwargs["prompt"].strip()

        if kwargs.get("negative_prompt") and kwargs["negative_prompt"].strip():
            arguments["negative_prompt"] = kwargs["negative_prompt"].strip()

        return arguments

    def upscale(self, image, api_key, **kwargs):
        """Main upscaling function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, image, **kwargs)
            result = self.call_api(client, "fal-ai/pasd", arguments)
            return self.process_images(result)
        except Exception as e:
            logger.error(f"Upscaling failed: {str(e)}")
            raise
