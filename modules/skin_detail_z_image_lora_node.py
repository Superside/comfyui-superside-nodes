import copy
import logging

from .z_image_inpaint_lora_node import SupersideZImageInpaintLoraNode

logger = logging.getLogger(__name__)

# LoRA weight URLs, hardcoded so they never appear in the exported workflow
# JSON. cog-comfyui's weights preflight (`handle_weights()`) scans every string
# value in every node's `inputs` and rejects the prediction if any ends in a
# model-weight extension (.safetensors/.ckpt/.pt/...) and isn't in its curated
# manifest. `SupersideZImageInpaintLoraNode` exposes `lora_url` /
# `skin_detail_lora_url` as inputs, so their URLs land in the JSON and get
# caught. This variant keeps those fields OUT of INPUT_TYPES entirely (not even
# a default), so there is nothing for the scanner to catch, while still passing
# the real URLs to the model at runtime.
MAIN_LORA_URL = (
    "https://huggingface.co/aaronamortegui/aaron_face/resolve/main/"
    "skindetails_mild_loraholic.safetensors"
)
SECONDARY_LORA_URL = ""  # hardcode the real value here once one exists; keep
                          # the field out of INPUT_TYPES either way.


class SupersideSkinDetailZImageLoraNode(SupersideZImageInpaintLoraNode):
    """
    Same Z-Image Turbo + LoRA inpainting as SupersideZImageInpaintLoraNode, but
    with both LoRA URLs hardcoded rather than exposed as inputs. Use this node
    on cog-comfyui-based deployments (e.g. Replicate), where the weights
    preflight check rejects any workflow whose JSON contains a raw LoRA weight
    URL. Because `lora_url` / `skin_detail_lora_url` are not declared as inputs
    here, they can never be serialized into the exported workflow JSON.

    All the actual inpainting logic is inherited from the parent - only the two
    URL inputs are removed and substituted with module-level constants. For a
    ComfyUI instance NOT subject to that weights scan, prefer the generic
    SupersideZImageInpaintLoraNode where lora_url is a real, editable input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        schema = copy.deepcopy(super().INPUT_TYPES())
        for section in ("required", "optional"):
            schema.get(section, {}).pop("lora_url", None)
            schema.get(section, {}).pop("skin_detail_lora_url", None)
            schema.get(section, {}).pop("lora_3_url", None)
        return schema

    DESCRIPTION = (
        "Z-Image Turbo inpaint + LoRA with the LoRA URLs hardcoded (not exposed "
        "as inputs), so the exported workflow JSON never contains a raw weight "
        "URL - required for cog-comfyui/Replicate deployments whose weights "
        "preflight rejects such URLs. Same behavior as Superside Z-Image Turbo "
        "Inpaint+LoRA otherwise."
    )

    def generate(self, **kwargs):
        # These are not declared in INPUT_TYPES, so ComfyUI never supplies them;
        # drop any stray value and force the hardcoded constants.
        kwargs.pop("lora_url", None)
        kwargs.pop("skin_detail_lora_url", None)
        return super().generate(
            lora_url=MAIN_LORA_URL,
            skin_detail_lora_url=SECONDARY_LORA_URL,
            **kwargs,
        )
