import logging

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideCrystalUpscalerNode(SupersideFalNode, ImageProcessingMixin, APIClientMixin):
    """
    Crystal Upscaler Node: portrait/facial-detail-specialized upscaler
    (fal endpoint "fal-ai/crystal-upscaler", Clarity AI's upscaling tech).

    Why this node exists: the Z-Image Turbo inpaint endpoint has a real,
    empirically-confirmed ceiling around ~2048px on its longest edge -
    requesting a custom image_size above that doesn't give more detail, it
    silently falls back to a default square size instead (see
    z_image_inpaint_lora_node.py's MAX_GENERATION_LONG_EDGE clamp). So once
    generation is already at that ceiling, "more resolution" can't come from
    asking the generator for a bigger image - it has to come from a
    dedicated upscaler pass on the result instead.

    This is meant to sit between the raw Z-Image Turbo output and
    SupersideResizeToMatchNode's final resize-to-original-size: instead of
    stretching a ~1536x2048 result up to the true original (e.g. 3584x4800)
    with a single plain Lanczos resize (a big ~2.3x stretch that looks
    soft), this does a smaller, AI-upscaled jump first (2x by default,
    portrait/face-aware), then Lanczos only has to cover the remaining
    smaller gap - net sharper result without exceeding the generator's own
    resolution ceiling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "How much to upscale before the final resize-to-original-size step.",
                    },
                ),
                "creativity": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "How much the upscaler is allowed to invent/enhance detail vs. stay literal.",
                    },
                ),
                "output_format": (["png", "jpg"], {"default": "png"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "upscale"
    DESCRIPTION = (
        "Portrait/facial-detail-specialized upscaler (fal-ai/crystal-upscaler). "
        "Use after generation, before resize-to-original-size, to add real "
        "detail instead of relying on plain interpolation for a big stretch."
    )

    def upscale(self, image, api_key, scale_factor=2.0, creativity=0.0, output_format="png"):
        try:
            client = self.get_client(api_key)

            image_url = self.upload_image(client, image)

            arguments = {
                "image_url": image_url,
                "scale_factor": float(scale_factor),
                "output_format": output_format,
            }
            if creativity is not None and float(creativity) > 0:
                arguments["creativity"] = float(creativity)

            result = self.call_api(client, "fal-ai/crystal-upscaler", arguments)

            images = self.process_images(result)

            first_url = ""
            if isinstance(result.get("images"), list) and result["images"]:
                first_url = result["images"][0].get("url", "")

            return (images[0], first_url)

        except Exception as e:
            logger.error(f"Crystal Upscaler failed: {str(e)}")
            raise RuntimeError(f"Crystal Upscaler failed: {str(e)}") from e
