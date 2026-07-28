import logging

import numpy as np
import torch

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideFluxProFillNode(SupersideFalNode, ImageProcessingMixin, APIClientMixin):
    """
    FLUX.1 [pro] Fill Node: dedicated inpainting/outpainting model (fal
    endpoint "fal-ai/flux-pro/v1/fill").

    Unlike Superside Z-Image Turbo Inpaint+LoRA (a "masked image-to-image"
    call - image + mask + strength, applied as an img2img noise mix over the
    whole frame), FLUX.1 Fill is architected end-to-end for inpainting: the
    masked image and mask are fed to the model as explicit conditioning
    channels, not blended in afterward. There is no "strength" knob here -
    the model always fully regenerates the masked region using the rest of
    the image as context, which in practice tends to hold the unmasked
    region much closer to pixel-identical.

    Test node for the A/B comparison against the Z-Image Turbo path - no
    LoRA support here (this is the base "pro" endpoint; a LoRA variant
    exists as "fal-ai/flux-lora/inpainting" but would need our skin LoRA
    retrained against FLUX's base weights, since LoRAs aren't portable
    across model families).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "placeholder": "Enter your prompt here"},
                ),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "-1 = random"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "safety_tolerance": (
                    ["1", "2", "3", "4", "5", "6"],
                    {"default": "2", "tooltip": "1 = most strict, 6 = most permissive."},
                ),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Dedicated inpainting/outpainting model (fal-ai/flux-pro/v1/fill). "
        "No strength/LoRA knobs - the mask and masked image are direct model "
        "conditioning, not a post-hoc blend, for tighter preservation of "
        "everything outside the mask."
    )

    def _mask_to_image_tensor(self, mask):
        """Convert a ComfyUI MASK tensor ([B,H,W], 0-1) into a grayscale IMAGE
        tensor ([1,H,W,3]) suitable for upload_image - same white=selected
        convention used elsewhere in this package (e.g. MaskToImage)."""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        mask_np = np.clip(mask_np, 0.0, 1.0).astype(np.float32)
        rgb = np.stack([mask_np] * 3, axis=-1)
        return torch.from_numpy(rgb).unsqueeze(0)

    def generate(
        self,
        image,
        mask,
        prompt,
        api_key,
        seed=-1,
        num_images=1,
        output_format="png",
        safety_tolerance="2",
        enhance_prompt=False,
    ):
        try:
            client = self.get_client(api_key)

            image_url = self.upload_image(client, image)
            mask_image_tensor = self._mask_to_image_tensor(mask)
            mask_url = self.upload_image(client, mask_image_tensor)

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "mask_url": mask_url,
                "num_images": int(num_images),
                "output_format": output_format,
                "safety_tolerance": str(safety_tolerance),
                "enhance_prompt": bool(enhance_prompt),
            }
            if seed is not None and int(seed) != -1:
                arguments["seed"] = int(seed)

            result = self.call_api(client, "fal-ai/flux-pro/v1/fill", arguments)

            images = self.process_images(result)

            first_url = ""
            if isinstance(result.get("images"), list) and result["images"]:
                first_url = result["images"][0].get("url", "")

            return (images[0], first_url)

        except Exception as e:
            logger.error(f"FLUX.1 Fill failed: {str(e)}")
            raise RuntimeError(f"FLUX.1 Fill failed: {str(e)}") from e
