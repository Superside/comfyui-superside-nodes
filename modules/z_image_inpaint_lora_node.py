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

# fal's custom image_size for this endpoint has an undocumented max accepted
# resolution. Requesting above it does NOT error - it silently falls back to
# a default square size (observed: a 3136x2336-ish request came back as a
# flat 2048x2048), discarding the input's aspect ratio entirely. Since our
# working resolution (node168 upstream) can be pushed arbitrarily high for
# mask-detection purposes, match_input_resolution must clamp its own
# generation request independently, so raising the working resolution can
# never again silently break the aspect ratio of the final image.
MAX_GENERATION_LONG_EDGE = 2048


class SupersideZImageInpaintLoraNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Z-Image Turbo Inpaint + LoRA Node: masked image-to-image (inpainting)
    with Tongyi-MAI's Z-Image Turbo model, with an optional custom LoRA
    (fal endpoint "fal-ai/z-image/turbo/inpaint/lora").

    This is the img2img/inpainting counterpart to a Z-Image Turbo LoRA
    trained with Superside Z-Image LoRA Trainer - the two use the same base
    model, so a LoRA trained there is guaranteed compatible here (unlike
    mixing LoRAs/checkpoints across unrelated model families).

    Takes a native ComfyUI IMAGE + MASK (white = area to regenerate, same
    convention as SetLatentNoiseMask/GrowMaskWithBlur) and a text prompt.
    `strength` plays the same role as a KSampler's denoise: 1.0 fully
    regenerates the masked area, 0.0 leaves it untouched.
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
                # Three generic, stackable LoRA slots (fal's LoRAInput list allows
                # up to 3 in one call). Any LoRA in any slot - they're merged
                # together in the single inpaint pass, each with its own scale.
                "lora_1_url": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "LoRA #1 URL - HuggingFace /resolve/ raw .safetensors (NOT /blob/). Empty = base model.",
                        "tooltip": "diffusers_lora_file URL (e.g. from Superside Z-Image LoRA Trainer, or a HuggingFace '/resolve/main/<file>.safetensors' raw URL - a '/blob/' URL is an HTML page and won't download). Slot 1 of 3.",
                    },
                ),
                "lora_1_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                     "tooltip": "Strength of lora_1_url. Ignored if lora_1_url is empty."},
                ),
                "lora_2_url": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "LoRA #2 URL (optional) - use /resolve/ not /blob/.",
                        "tooltip": "Slot 2 of 3. Any LoRA, stacked on top of slot 1 in the same call.",
                    },
                ),
                "lora_2_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                     "tooltip": "Strength of lora_2_url. Ignored if lora_2_url is empty."},
                ),
                "lora_3_url": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "LoRA #3 URL (optional) - use /resolve/ not /blob/.",
                        "tooltip": "Slot 3 of 3 (fal's max). Any LoRA, stacked with slots 1 and 2.",
                    },
                ),
                "lora_3_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                     "tooltip": "Strength of lora_3_url. Ignored if lora_3_url is empty."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Inpaint strength - same role as a KSampler's denoise. 1.0 = fully regenerate the masked area, 0.0 = keep it untouched.",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 8, "min": 1, "max": 8, "tooltip": "Z-Image Turbo is a distilled few-step model; 8 is its max."},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "-1 = random"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image_size": (
                    ["auto", "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"],
                    {"default": "auto", "tooltip": "\"auto\" keeps the input image's own size/aspect ratio."},
                ),
                "control_scale": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the structural conditioning taken from the input image."},
                ),
                "control_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "match_input_resolution": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Generate at the input image's own resolution (rounded to a "
                            "multiple of 16) instead of fal's image_size preset. fal's "
                            "'auto'/enum presets default to a small size (~512px short "
                            "side) which looks flat/soft on a cropped high-res photo - "
                            "enable this to keep full detail. Disable to use the "
                            "image_size dropdown above instead."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Masked image-to-image (inpainting) with Z-Image Turbo plus an "
        "optional custom LoRA (fal-ai/z-image/turbo/inpaint/lora). Drop-in "
        "replacement for a VAEEncode -> SetLatentNoiseMask -> LoRA loader -> "
        "KSampler -> VAEDecode chain, but as a single fal.ai call."
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
        lora_1_url="",
        lora_1_scale=1.0,
        lora_2_url="",
        lora_2_scale=1.0,
        lora_3_url="",
        lora_3_scale=1.0,
        strength=0.4,
        num_inference_steps=8,
        seed=-1,
        num_images=1,
        image_size="auto",
        control_scale=0.75,
        control_start=0.0,
        control_end=0.8,
        enable_prompt_expansion=False,
        enable_safety_checker=True,
        output_format="png",
        acceleration="regular",
        match_input_resolution=True,
        **kwargs,
    ):
        try:
            client = self.get_client(api_key)

            image_url = self.upload_image(client, image)
            mask_image_tensor = self._mask_to_image_tensor(mask)
            mask_url = self.upload_image(client, mask_image_tensor)

            # Backward compatibility: older graphs / the skin subclass pass
            # lora_url / skin_detail_lora_url. Map them onto slots 1 and 2 when
            # the new slot is empty, so nothing breaks after the rename.
            if not lora_1_url and kwargs.get("lora_url"):
                lora_1_url = kwargs["lora_url"]
                lora_1_scale = kwargs.get("lora_scale", lora_1_scale)
            if not lora_2_url and kwargs.get("skin_detail_lora_url"):
                lora_2_url = kwargs["skin_detail_lora_url"]
                lora_2_scale = kwargs.get("skin_detail_lora_scale", lora_2_scale)

            # fal's LoRAInput list supports stacking up to 3 LoRAs in one call;
            # any LoRA in any slot, all merged together in this single pass.
            loras = []
            for url, scale in ((lora_1_url, lora_1_scale), (lora_2_url, lora_2_scale), (lora_3_url, lora_3_scale)):
                if url and url.strip():
                    loras.append({"path": url.strip(), "scale": float(scale)})

            if match_input_resolution:
                in_h, in_w = int(image.shape[1]), int(image.shape[2])
                # Round to a multiple of 16 (diffusion-friendly) while
                # preserving the exact input aspect ratio - rounding width
                # and height independently can drift the ratio slightly,
                # which some backends handle by letterboxing/cropping
                # internally, shifting content within the frame and causing
                # a misaligned stitch when pasted back at the original crop
                # position. A single shared scale factor avoids that.
                longest = max(in_w, in_h)
                rounded_longest = max(16, int(round(longest / 16.0)) * 16)
                # Clamp to fal's real accepted max BEFORE rounding to 16, so
                # a high working resolution (node168) never causes this
                # request to exceed what the endpoint will actually honor -
                # see MAX_GENERATION_LONG_EDGE note above. The upstream
                # SupersideResizeToMatchNode always upscales the result back
                # to the true original size afterward, so this clamp only
                # affects the generation call, not the final output size.
                if rounded_longest > MAX_GENERATION_LONG_EDGE:
                    rounded_longest = max(
                        16, int(round(MAX_GENERATION_LONG_EDGE / 16.0)) * 16
                    )
                scale = rounded_longest / float(longest)
                round_w = max(16, int(round((in_w * scale) / 16.0)) * 16)
                round_h = max(16, int(round((in_h * scale) / 16.0)) * 16)
                resolved_image_size = {"width": round_w, "height": round_h}
            else:
                resolved_image_size = image_size

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "mask_image_url": mask_url,
                "strength": float(strength),
                "loras": loras,
                "num_inference_steps": int(num_inference_steps),
                "num_images": int(num_images),
                "image_size": resolved_image_size,
                "control_scale": float(control_scale),
                "control_start": float(control_start),
                "control_end": float(control_end),
                "enable_prompt_expansion": bool(enable_prompt_expansion),
                "enable_safety_checker": bool(enable_safety_checker),
                "output_format": output_format,
                "acceleration": acceleration,
            }
            if seed is not None and int(seed) != -1:
                arguments["seed"] = int(seed)

            result = self.call_api(client, "fal-ai/z-image/turbo/inpaint/lora", arguments)

            images = self.process_images(result)

            first_url = ""
            if isinstance(result.get("images"), list) and result["images"]:
                first_url = result["images"][0].get("url", "")

            return (images[0], first_url)

        except Exception as e:
            logger.error(f"Z-Image inpaint+LoRA failed: {str(e)}")
            raise RuntimeError(f"Z-Image inpaint+LoRA failed: {str(e)}") from e
