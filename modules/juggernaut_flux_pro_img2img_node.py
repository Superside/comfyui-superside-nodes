import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SupersideJuggernautFluxProImg2ImgNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Juggernaut Flux Pro Image-to-Image Node: Transform images using
    Juggernaut Pro Flux model via fal.ai API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Enter your prompt here",
                    },
                ),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 40,
                        "min": 10,
                        "max": 50,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                "seed": ("INT", {"min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 3.5,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.5,
                        "display": "slider",
                    },
                ),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                    },
                ),
                "enable_safety_checker": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    DESCRIPTION = "Transform images using Juggernaut Pro Flux model with high realism"

    def prepare_arguments(self, client, image, prompt, **kwargs):
        """Prepare arguments for the API call."""
        # Upload image and get URL
        image_url = self.upload_image(client, image)

        arguments = {
            "image_url": image_url,
            "prompt": prompt,
        }

        # Add optional parameters if provided
        if "strength" in kwargs and kwargs["strength"] is not None:
            arguments["strength"] = kwargs["strength"]

        if (
            "num_inference_steps" in kwargs
            and kwargs["num_inference_steps"] is not None
        ):
            arguments["num_inference_steps"] = kwargs["num_inference_steps"]

        if kwargs.get("seed") and kwargs["seed"] != 0:
            arguments["seed"] = kwargs["seed"]

        if "guidance_scale" in kwargs and kwargs["guidance_scale"] is not None:
            arguments["guidance_scale"] = kwargs["guidance_scale"]

        if "num_images" in kwargs and kwargs["num_images"] is not None:
            arguments["num_images"] = kwargs["num_images"]

        if (
            "enable_safety_checker" in kwargs
            and kwargs["enable_safety_checker"] is not None
        ):
            arguments["enable_safety_checker"] = kwargs["enable_safety_checker"]

        # Don't set sync_mode, let it default to False for URL responses

        return arguments

    def generate(self, image, prompt, api_key, **kwargs):
        """Main generation function."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, image, prompt, **kwargs)
            logger.info(f"Calling Juggernaut API with arguments: {arguments}")

            result = self.call_api(
                client, "rundiffusion-fal/juggernaut-flux/pro/image-to-image", arguments
            )

            logger.debug(f"API Response: {result}")

            # Process the images using the parent class method
            return self.process_images(result)

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
