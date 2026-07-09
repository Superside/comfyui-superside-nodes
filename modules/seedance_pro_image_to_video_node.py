import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SupersideSeedanceProImageToVideoNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Seedance Pro Image-to-Video Node: Generate high-quality videos from images
    using Seedance 1.0 Pro model via fal.ai API.

    This node provides advanced video generation capabilities with support for
    aspect ratio control, resolution settings, duration options, and camera controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A skier glides over fresh snow, joyously smiling"
                        " while kicking up large clouds of snow as he turns.",
                        "placeholder": "Enter your video prompt here",
                    },
                ),
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "end_image": ("IMAGE",),
                "aspect_ratio": (
                    ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
                    {"default": "auto"},
                ),
                "resolution": (
                    ["480p", "720p", "1080p"],
                    {"default": "1080p"},
                ),
                "duration": (
                    ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    {"default": "5"},
                ),
                "camera_fixed": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to fix the camera position",
                    },
                ),
                "seed": ("INT", {"min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "enable_safety_checker": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable safety checker to filter "
                        "inappropriate content",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    DESCRIPTION = (
        "Generate high-quality videos from images using Seedance 1.0 Pro model. "
        "Supports multiple resolutions, aspect ratios, and advanced camera controls."
    )

    def prepare_image_urls(self, client, image, end_image=None):
        """Prepare image URLs for the API call."""
        try:
            # Upload the main image
            image_url = self.upload_image(client, image)
            logger.info(f"Main image uploaded successfully: {image_url}")

            urls = {"image_url": image_url}

            # Upload end image if provided
            if end_image is not None:
                try:
                    end_image_url = self.upload_image(client, end_image)
                    urls["end_image_url"] = end_image_url
                    logger.info(f"End image uploaded successfully: {end_image_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload end image: {str(e)}")
                    # Continue without end image rather than failing completely

            return urls

        except Exception as e:
            logger.error(f"Failed to upload main image: {str(e)}")
            raise ValueError(f"Image upload failed: {str(e)}") from e

    def prepare_arguments(self, client, prompt, **kwargs):
        """Prepare arguments for the Seedance Pro API call."""
        # Get image URLs
        image_urls = self.prepare_image_urls(
            client, kwargs.get("image"), kwargs.get("end_image")
        )

        # Build base arguments
        arguments = {
            "prompt": prompt,
            **image_urls,
        }

        # Add optional parameters with validation
        optional_params = {
            "aspect_ratio": kwargs.get("aspect_ratio"),
            "resolution": kwargs.get("resolution"),
            "duration": kwargs.get("duration"),
            "camera_fixed": kwargs.get("camera_fixed"),
            "enable_safety_checker": kwargs.get("enable_safety_checker"),
        }

        # Only add non-None optional parameters
        for key, value in optional_params.items():
            if value is not None:
                arguments[key] = value

        # Handle seed parameter (exclude -1 values)
        seed = kwargs.get("seed")
        if seed is not None and seed != -1:
            arguments["seed"] = seed

        logger.debug(f"Prepared API arguments: {arguments}")
        return arguments

    def process_video_result(self, result):
        """Process the video result from the Seedance Pro API."""
        if "video" not in result or not result["video"]:
            raise RuntimeError("No video was generated by the API.")

        try:
            video_info = result["video"]
            logger.debug(f"Processing video response: {video_info}")

            if not isinstance(video_info, dict):
                raise RuntimeError("Invalid video format in API response")

            video_url = video_info.get("url")
            if not video_url:
                raise RuntimeError("No video URL found in API response")

            # Log additional metadata if available
            if "seed" in result:
                logger.info(f"Video generated with seed: {result['seed']}")

            logger.info(f"Video generation successful: {video_url}")
            return video_url

        except Exception as e:
            logger.error(f"Failed to process video result: {str(e)}")
            raise RuntimeError(f"Video processing failed: {str(e)}") from e

    def generate(self, prompt, api_key, **kwargs):
        """Main generation function for Seedance Pro video generation."""
        try:
            client = self.get_client(api_key)

            # Prepare API arguments
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            logger.info(
                f"Starting Seedance Pro video generation with prompt: '{prompt}'"
            )

            # Call the Seedance Pro API
            result = self.call_api(
                client, "fal-ai/bytedance/seedance/v1/pro/image-to-video", arguments
            )

            logger.debug(f"Seedance Pro API response: {result}")

            # Process and return the video result
            video_url = self.process_video_result(result)
            return (video_url,)

        except Exception as e:
            logger.error(f"Seedance Pro video generation failed: {str(e)}")
            raise RuntimeError(f"Video generation failed: {str(e)}") from e
