import logging

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideImageRetouchNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Image Retouch Node: retouch/clean up an image (e.g. skin, blemishes,
    imperfections) using fal.ai's image-editing retouch model
    (endpoint "fal-ai/image-editing/retouch").

    Takes a single image and returns a retouched version. No prompt needed -
    the model applies its retouching pass automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                     "tooltip": "CFG scale - how strongly the model follows its retouch objective. Higher = stronger effect."},
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 30, "min": 1, "max": 100,
                     "tooltip": "Number of sampling steps. More steps = potentially cleaner result, slower."},
                ),
                "lora_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": "Strength of the retouch LoRA. Lower = subtler retouch, higher = stronger."},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "-1 = random"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
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
        "Retouch/clean up an image (skin, blemishes, imperfections) using "
        "fal.ai's image-editing retouch model (fal-ai/image-editing/retouch). "
        "No prompt needed - just connect an image."
    )

    def prepare_arguments(self, client, image, **kwargs):
        image_url = self.upload_image(client, image)
        arguments = {"image_url": image_url}

        if kwargs.get("guidance_scale") is not None:
            arguments["guidance_scale"] = kwargs["guidance_scale"]

        if kwargs.get("num_inference_steps") is not None:
            arguments["num_inference_steps"] = kwargs["num_inference_steps"]

        if kwargs.get("lora_scale") is not None:
            arguments["lora_scale"] = kwargs["lora_scale"]

        seed = kwargs.get("seed", -1)
        if seed is not None and seed != -1:
            arguments["seed"] = seed

        if kwargs.get("enable_safety_checker") is not None:
            arguments["enable_safety_checker"] = kwargs["enable_safety_checker"]

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, image, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, image, **kwargs)
            result = self.call_api(client, "fal-ai/image-editing/retouch", arguments)

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
            logger.error(f"Image retouch failed: {str(e)}")
            raise RuntimeError(f"Image retouch failed: {str(e)}") from e
