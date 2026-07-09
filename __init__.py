try:
    # Normal path when ComfyUI loads this as a proper package.
    from .modules.any_llm_text_node import AnyLLMTextNode
    from .modules.any_llm_vision_node import AnyLLMVisionNode
    from .modules.bria_background_standardizer_node import BriaBackgroundStandardizerNode
    from .modules.bria_replace_background_node import BriaReplaceBackgroundNode
    from .modules.florence_2_caption_node import Florence2CaptionNode
    from .modules.florence_2_region_selector_node import Florence2RegionSelectorNode
    from .modules.flux_kontext_max_multi_node import FluxKontextMaxMultiImageNode
    from .modules.gpt_image_2_edit_node import GPTImage2EditNode
    from .modules.grok_imagine_image_quality_edit_node import GrokImagineImageQualityEditNode
    from .modules.ideogram_upscale_node import IdeogramUpscaleNode
    from .modules.juggernaut_flux_pro_img2img_node import JuggernautFluxProImg2ImgNode
    from .modules.kling_21_image_to_video_node import Kling21ImageToVideoNode
    from .modules.kling_25_turbo_pro_image_to_video_node import Kling25TurboProImageToVideoNode
    from .modules.nano_banana_pro_node import NanoBananaProEditNode
    from .modules.nano_banana_v2_edit_node import NanoBananaV2EditNode
    from .modules.pasd_upscaler_node import PASDUpscalerNode
    from .modules.prompt_box_node import PromptBoxNode
    from .modules.prompt_splitter_node import PromptSplitterNode
    from .modules.sam_3_region_selector_node import SAM3RegionSelectorNode
    from .modules.seedance_lite_image_to_video_node import SeedanceLiteImageToVideoNode
    from .modules.seedance_pro_image_to_video_node import SeedanceProImageToVideoNode
    from .modules.seedream_v45_edit_node import SeedreamV45EditNode
    from .modules.seedream_v5_pro_edit_node import SeedreamV5ProEditNode
    from .modules.seedvr_2_upscale_image_node import SeedVR2UpscaleImageNode
    from .modules.seedvr_upscale_video_node import SeedVRUpscaleVideoNode
    from .modules.topaz_upscale_image_node import TopazUpscaleImageNode
    from .modules.wan_25_image_to_image_node import Wan25ImageToImageNode
    from .modules.wan_25_image_to_video_node import Wan25ImageToVideoNode

except ImportError:
    # Fallback for environments where this hyphenated folder name doesn't get
    # a normal package import context (e.g. running files directly).
    import os
    import sys

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    from modules.any_llm_text_node import AnyLLMTextNode
    from modules.any_llm_vision_node import AnyLLMVisionNode
    from modules.bria_background_standardizer_node import BriaBackgroundStandardizerNode
    from modules.bria_replace_background_node import BriaReplaceBackgroundNode
    from modules.florence_2_caption_node import Florence2CaptionNode
    from modules.florence_2_region_selector_node import Florence2RegionSelectorNode
    from modules.flux_kontext_max_multi_node import FluxKontextMaxMultiImageNode
    from modules.gpt_image_2_edit_node import GPTImage2EditNode
    from modules.grok_imagine_image_quality_edit_node import GrokImagineImageQualityEditNode
    from modules.ideogram_upscale_node import IdeogramUpscaleNode
    from modules.juggernaut_flux_pro_img2img_node import JuggernautFluxProImg2ImgNode
    from modules.kling_21_image_to_video_node import Kling21ImageToVideoNode
    from modules.kling_25_turbo_pro_image_to_video_node import Kling25TurboProImageToVideoNode
    from modules.nano_banana_pro_node import NanoBananaProEditNode
    from modules.nano_banana_v2_edit_node import NanoBananaV2EditNode
    from modules.pasd_upscaler_node import PASDUpscalerNode
    from modules.prompt_box_node import PromptBoxNode
    from modules.prompt_splitter_node import PromptSplitterNode
    from modules.sam_3_region_selector_node import SAM3RegionSelectorNode
    from modules.seedance_lite_image_to_video_node import SeedanceLiteImageToVideoNode
    from modules.seedance_pro_image_to_video_node import SeedanceProImageToVideoNode
    from modules.seedream_v45_edit_node import SeedreamV45EditNode
    from modules.seedream_v5_pro_edit_node import SeedreamV5ProEditNode
    from modules.seedvr_2_upscale_image_node import SeedVR2UpscaleImageNode
    from modules.seedvr_upscale_video_node import SeedVRUpscaleVideoNode
    from modules.topaz_upscale_image_node import TopazUpscaleImageNode
    from modules.wan_25_image_to_image_node import Wan25ImageToImageNode
    from modules.wan_25_image_to_video_node import Wan25ImageToVideoNode

NODE_CLASS_MAPPINGS = {
    "AnyLLMTextNode": AnyLLMTextNode,
    "AnyLLMVisionNode": AnyLLMVisionNode,
    "BriaBackgroundStandardizerNode": BriaBackgroundStandardizerNode,
    "BriaReplaceBackgroundNode": BriaReplaceBackgroundNode,
    "Florence2CaptionNode": Florence2CaptionNode,
    "Florence2RegionSelectorNode": Florence2RegionSelectorNode,
    "FluxKontextMaxMultiImageNode": FluxKontextMaxMultiImageNode,
    "GPTImage2EditNode": GPTImage2EditNode,
    "GrokImagineImageQualityEditNode": GrokImagineImageQualityEditNode,
    "IdeogramUpscaleNode": IdeogramUpscaleNode,
    "JuggernautFluxProImg2ImgNode": JuggernautFluxProImg2ImgNode,
    "Kling21ImageToVideoNode": Kling21ImageToVideoNode,
    "Kling25TurboProImageToVideoNode": Kling25TurboProImageToVideoNode,
    "NanoBananaProEditNode": NanoBananaProEditNode,
    "NanoBananaV2EditNode": NanoBananaV2EditNode,
    "PASDUpscalerNode": PASDUpscalerNode,
    "PromptBoxNode": PromptBoxNode,
    "PromptSplitterNode": PromptSplitterNode,
    "SAM3RegionSelectorNode": SAM3RegionSelectorNode,
    "SeedanceLiteImageToVideoNode": SeedanceLiteImageToVideoNode,
    "SeedanceProImageToVideoNode": SeedanceProImageToVideoNode,
    "SeedreamV45EditNode": SeedreamV45EditNode,
    "SeedreamV5ProEditNode": SeedreamV5ProEditNode,
    "SeedVR2UpscaleImageNode": SeedVR2UpscaleImageNode,
    "SeedVRUpscaleVideoNode": SeedVRUpscaleVideoNode,
    "TopazUpscaleImageNode": TopazUpscaleImageNode,
    "Wan25ImageToImageNode": Wan25ImageToImageNode,
    "Wan25ImageToVideoNode": Wan25ImageToVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyLLMTextNode": "Superside Any LLM Text",
    "AnyLLMVisionNode": "Superside Any LLM Vision",
    "BriaBackgroundStandardizerNode": "Superside Bria Background Standardizer (Hex Color)",
    "BriaReplaceBackgroundNode": "Superside Bria Replace Background",
    "Florence2CaptionNode": "Superside Florence-2 Detailed Caption",
    "Florence2RegionSelectorNode": "Superside Florence-2 Smart Region Selector",
    "FluxKontextMaxMultiImageNode": "Superside Flux Kontext Max Multi-Image Node",
    "GPTImage2EditNode": "Superside GPT Image 2 Edit",
    "GrokImagineImageQualityEditNode": "Superside Grok Imagine Image Quality Edit",
    "IdeogramUpscaleNode": "Superside Ideogram Upscale",
    "JuggernautFluxProImg2ImgNode": "Superside Juggernaut Flux Pro Image-to-Image",
    "Kling21ImageToVideoNode": "Superside Kling 2.1 Image-to-Video",
    "Kling25TurboProImageToVideoNode": "Superside Kling 2.5 Turbo Pro Image-to-Video",
    "NanoBananaProEditNode": "Superside Nano Banana Pro Edit Node",
    "NanoBananaV2EditNode": "Superside Nano Banana V2 Edit Node",
    "PASDUpscalerNode": "Superside PASD Upscaler Node",
    "PromptBoxNode": "Superside Prompt Box",
    "PromptSplitterNode": "Superside Prompt Splitter",
    "SAM3RegionSelectorNode": "Superside SAM 3 Smart Region Selector",
    "SeedanceLiteImageToVideoNode": "Superside Seedance Lite Image-to-Video",
    "SeedanceProImageToVideoNode": "Superside Seedance Pro Image-to-Video",
    "SeedreamV45EditNode": "Superside Seedream V4.5 Edit",
    "SeedreamV5ProEditNode": "Superside Seedream V5 Pro Edit",
    "SeedVR2UpscaleImageNode": "Superside SeedVR2 Upscale Image",
    "SeedVRUpscaleVideoNode": "Superside SeedVR Upscale Video",
    "TopazUpscaleImageNode": "Superside Topaz Upscale Image",
    "Wan25ImageToImageNode": "Superside Wan 2.5 Image-to-Image",
    "Wan25ImageToVideoNode": "Superside Wan 2.5 Image-to-Video",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
