import logging

from .gpt_image_2_edit_node import SupersideGPTImage2EditNode

logger = logging.getLogger(__name__)


class SupersideGPTImage25SunburstEditNode(SupersideGPTImage2EditNode):
    """
    GPT Image 2.5 Sunburst Edit: openai/gpt-image-2.5/sunburst/edit.

    Same input shape as GPT Image 2 Edit - multi-image references, a real
    mask_url, and the image_size / quality / output_format controls - so this
    subclasses that node and only declares what differs:

      * a wider quality enum ("xhigh" and "max" on top of the usual four)
      * a background control (auto / transparent / opaque)
      * an optional output_compression for jpeg and webp

    Pricing is per token, like the rest of the gpt-image family, so the cost
    ledger records the call without a dollar estimate.
    """

    ENDPOINT = "openai/gpt-image-2.5/sunburst/edit"

    QUALITY_OPTIONS = ["auto", "low", "medium", "high", "xhigh", "max"]
    BACKGROUND_OPTIONS = ["auto", "transparent", "opaque"]

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Edit images using OpenAI GPT Image 2.5 (Sunburst) on fal. Multi-image "
        "references, mask-based editing, preset or custom sizes, and quality up "
        "to 'max'. Cost is per token and rises steeply with quality - 'high' is "
        "the default, 'xhigh' and 'max' are considerably more expensive."
    )

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        opt = spec["optional"]
        opt["background"] = (
            cls.BACKGROUND_OPTIONS,
            {
                "default": "auto",
                "tooltip": (
                    "Background of the generated image. 'transparent' needs "
                    "output_format png or webp; with jpeg the model falls back "
                    "to opaque."
                ),
            },
        )
        opt["output_compression"] = (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 100,
                "step": 1,
                "tooltip": (
                    "Compression level for jpeg and webp output, 0 to leave it "
                    "to the API. Ignored for png."
                ),
            },
        )
        return spec

    def prepare_arguments(self, client, prompt, **kwargs):
        arguments = super().prepare_arguments(client, prompt, **kwargs)

        background = kwargs.get("background")
        if background and background != "auto":
            arguments["background"] = background

        # 0 means "don't send it" - the API's own default then applies.
        compression = kwargs.get("output_compression")
        if compression:
            arguments["output_compression"] = int(compression)

        return arguments
