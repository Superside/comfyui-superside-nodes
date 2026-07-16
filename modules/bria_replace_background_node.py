import logging

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideBriaReplaceBackgroundNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Bria Background Replace Node: Replace an image's background using Bria's
    newer Background Replace model on fal.ai (endpoint
    "fal-ai/bria/background/replace").

    This is the richer, current background-replacement endpoint - NOT the
    older basic "bria/replace-background" (which only takes a prompt +
    steps_num). It supports guiding the new background with a reference
    image, prompt refinement, a fast/quality toggle, and generating several
    variations at once. Trained on licensed data for commercial use.
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
                        "placeholder": "Describe the new background scene",
                    },
                ),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "ref_image": (
                    "IMAGE",
                    {"tooltip": "Optional reference image to guide the new background's look, instead of (or alongside) the text prompt."},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Describe what to avoid in the background",
                    },
                ),
                "num_images": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "tooltip": "How many background variations to generate."},
                ),
                "refine_prompt": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Let Bria refine/expand your prompt for better results."},
                ),
                "fast": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "ON = faster model; OFF = higher-quality (slower) model."},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "-1 = random"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Replace an image's background using Bria Background Replace on fal.ai "
        "(endpoint fal-ai/bria/background/replace - the richer current model, "
        "not the basic bria/replace-background). Guide it with a text prompt "
        "and/or a reference image; supports multiple variations."
    )

    def prepare_arguments(self, client, image, prompt, **kwargs):
        image_url = self.upload_image(client, image)

        arguments = {"image_url": image_url}

        if prompt and prompt.strip():
            arguments["prompt"] = prompt.strip()

        ref_image = kwargs.get("ref_image")
        if ref_image is not None:
            arguments["ref_image_url"] = self.upload_image(client, ref_image)

        if kwargs.get("negative_prompt"):
            arguments["negative_prompt"] = kwargs["negative_prompt"]

        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("refine_prompt") is not None:
            arguments["refine_prompt"] = kwargs["refine_prompt"]

        if kwargs.get("fast") is not None:
            arguments["fast"] = kwargs["fast"]

        seed = kwargs.get("seed", -1)
        if seed is not None and seed != -1:
            arguments["seed"] = seed

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, image, prompt, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, image, prompt, **kwargs)
            result = self.call_api(client, "fal-ai/bria/background/replace", arguments)

            # This endpoint returns an "images" list.
            images = self.process_images(result)

            first_url = ""
            if isinstance(result.get("images"), list) and result["images"]:
                first_url = result["images"][0].get("url", "")
            info = first_url

            if unique_id is not None and extra_pnginfo is not None:
                if (
                    isinstance(extra_pnginfo, list)
                    and isinstance(extra_pnginfo[0], dict)
                    and "workflow" in extra_pnginfo[0]
                ):
                    workflow = extra_pnginfo[0]["workflow"]
                    node = next(
                        (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id)),
                        None,
                    )
                    if node:
                        node["widgets_values"] = [info]

            return {"ui": {"text": [info]}, "result": (images[0], info)}
        except Exception as e:
            logger.error(f"Bria background replace failed: {str(e)}")
            raise RuntimeError(f"Bria background replace failed: {str(e)}") from e
