import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideWan25ImageToImageNode(
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
):
    """
    Wan 2.5 Image-to-Image Node: Edit and transform images using Wan 2.5 model
    via fal.ai API.

    This node provides advanced image editing capabilities with support for
    single or multi-reference image editing, custom image sizes, and multiple outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Describe how to edit the image"
                        " (max 2000 characters)",
                    },
                ),
                "image_1": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "image_2": ("IMAGE",),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Content to avoid (max 500 characters)",
                    },
                ),
                "image_size": (
                    ["square", "landscape_16_9", "portrait_16_9"],
                    {"default": "square"},
                ),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of images to generate (1-4)",
                    },
                ),
                "seed": ("INT", {"min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    DESCRIPTION = (
        "Edit and transform images using Wan 2.5 model. "
        "Supports single or multi-reference editing, custom "
        "sizes, and batch generation."
    )

    def prepare_image_urls(self, client, **kwargs):
        """Prepare list of image URLs from input images."""
        image_urls = []

        for i in range(1, 3):
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
        """Prepare arguments for the Wan 2.5 Image-to-Image API call."""
        image_urls = self.prepare_image_urls(client, **kwargs)

        if not image_urls:
            raise ValueError(
                "At least one image is required for Wan 2.5 image-to-image generation"
            )

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
        }

        if "negative_prompt" in kwargs and kwargs["negative_prompt"]:
            arguments["negative_prompt"] = kwargs["negative_prompt"]

        if "image_size" in kwargs and kwargs["image_size"] is not None:
            arguments["image_size"] = kwargs["image_size"]

        if "num_images" in kwargs and kwargs["num_images"] is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("seed") and kwargs["seed"] != 0:
            arguments["seed"] = kwargs["seed"]

        logger.debug(f"Prepared API arguments: {arguments}")
        return arguments

    def generate(self, prompt, api_key, **kwargs):
        """Main generation function for Wan 2.5 image-to-image editing."""
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            logger.info(
                f"Starting Wan 2.5 image-to-image generation with prompt: '{prompt}'"
            )

            result = self.call_api(client, "fal-ai/wan-25-preview/image-to-image", arguments)

            logger.debug(f"Wan 2.5 API response: {result}")

            return self.process_images(result)

        except Exception as e:
            logger.error(f"Wan 2.5 image-to-image generation failed: {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}") from e
