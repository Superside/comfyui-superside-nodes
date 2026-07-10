try:
    # Normal path when ComfyUI loads this as a proper package.
    from .modules.any_llm_text_node import SupersideAnyLLMTextNode
    from .modules.any_llm_vision_node import SupersideAnyLLMVisionNode
    from .modules.bria_background_standardizer_node import SupersideBriaBackgroundStandardizerNode
    from .modules.bria_replace_background_node import SupersideBriaReplaceBackgroundNode
    from .modules.florence_2_caption_node import SupersideFlorence2CaptionNode
    from .modules.florence_2_region_selector_node import SupersideFlorence2RegionSelectorNode
    from .modules.flux_kontext_max_multi_node import SupersideFluxKontextMaxMultiImageNode
    from .modules.gemini_omni_flash_edit_node import SupersideGeminiOmniFlashEditNode
    from .modules.gpt_image_2_edit_node import SupersideGPTImage2EditNode
    from .modules.grok_imagine_image_quality_edit_node import SupersideGrokImagineImageQualityEditNode
    from .modules.ideogram_upscale_node import SupersideIdeogramUpscaleNode
    from .modules.juggernaut_flux_pro_img2img_node import SupersideJuggernautFluxProImg2ImgNode
    from .modules.kling_21_image_to_video_node import SupersideKling21ImageToVideoNode
    from .modules.kling_25_turbo_pro_image_to_video_node import SupersideKling25TurboProImageToVideoNode
    from .modules.nano_banana_pro_node import SupersideNanoBananaProEditNode
    from .modules.nano_banana_v2_edit_node import SupersideNanoBananaV2EditNode
    from .modules.pasd_upscaler_node import SupersidePASDUpscalerNode
    from .modules.prompt_box_node import SupersidePromptBoxNode
    from .modules.prompt_splitter_node import SupersidePromptSplitterNode
    from .modules.sam_3_region_selector_node import SupersideSAM3RegionSelectorNode
    from .modules.seedance_lite_image_to_video_node import SupersideSeedanceLiteImageToVideoNode
    from .modules.seedance_pro_image_to_video_node import SupersideSeedanceProImageToVideoNode
    from .modules.seedream_v45_edit_node import SupersideSeedreamV45EditNode
    from .modules.seedream_v5_pro_edit_node import SupersideSeedreamV5ProEditNode
    from .modules.seedvr_2_upscale_image_node import SupersideSeedVR2UpscaleImageNode
    from .modules.seedvr_upscale_video_node import SupersideSeedVRUpscaleVideoNode
    from .modules.smart_detail_sheet_node import SupersideSmartDetailSheetNode
    from .modules.topaz_upscale_image_node import SupersideTopazUpscaleImageNode
    from .modules.wan_25_image_to_image_node import SupersideWan25ImageToImageNode
    from .modules.wan_25_image_to_video_node import SupersideWan25ImageToVideoNode

except ImportError:
    # Fallback for environments where this hyphenated folder name doesn't get
    # a normal package import context (e.g. running files directly).
    import os
    import sys

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    from modules.any_llm_text_node import SupersideAnyLLMTextNode
    from modules.any_llm_vision_node import SupersideAnyLLMVisionNode
    from modules.bria_background_standardizer_node import SupersideBriaBackgroundStandardizerNode
    from modules.bria_replace_background_node import SupersideBriaReplaceBackgroundNode
    from modules.florence_2_caption_node import SupersideFlorence2CaptionNode
    from modules.florence_2_region_selector_node import SupersideFlorence2RegionSelectorNode
    from modules.flux_kontext_max_multi_node import SupersideFluxKontextMaxMultiImageNode
    from modules.gemini_omni_flash_edit_node import SupersideGeminiOmniFlashEditNode
    from modules.gpt_image_2_edit_node import SupersideGPTImage2EditNode
    from modules.grok_imagine_image_quality_edit_node import SupersideGrokImagineImageQualityEditNode
    from modules.ideogram_upscale_node import SupersideIdeogramUpscaleNode
    from modules.juggernaut_flux_pro_img2img_node import SupersideJuggernautFluxProImg2ImgNode
    from modules.kling_21_image_to_video_node import SupersideKling21ImageToVideoNode
    from modules.kling_25_turbo_pro_image_to_video_node import SupersideKling25TurboProImageToVideoNode
    from modules.nano_banana_pro_node import SupersideNanoBananaProEditNode
    from modules.nano_banana_v2_edit_node import SupersideNanoBananaV2EditNode
    from modules.pasd_upscaler_node import SupersidePASDUpscalerNode
    from modules.prompt_box_node import SupersidePromptBoxNode
    from modules.prompt_splitter_node import SupersidePromptSplitterNode
    from modules.sam_3_region_selector_node import SupersideSAM3RegionSelectorNode
    from modules.seedance_lite_image_to_video_node import SupersideSeedanceLiteImageToVideoNode
    from modules.seedance_pro_image_to_video_node import SupersideSeedanceProImageToVideoNode
    from modules.seedream_v45_edit_node import SupersideSeedreamV45EditNode
    from modules.seedream_v5_pro_edit_node import SupersideSeedreamV5ProEditNode
    from modules.seedvr_2_upscale_image_node import SupersideSeedVR2UpscaleImageNode
    from modules.seedvr_upscale_video_node import SupersideSeedVRUpscaleVideoNode
    from modules.smart_detail_sheet_node import SupersideSmartDetailSheetNode
    from modules.topaz_upscale_image_node import SupersideTopazUpscaleImageNode
    from modules.wan_25_image_to_image_node import SupersideWan25ImageToImageNode
    from modules.wan_25_image_to_video_node import SupersideWan25ImageToVideoNode

NODE_CLASS_MAPPINGS = {
    "SupersideAnyLLMTextNode": SupersideAnyLLMTextNode,
    "SupersideAnyLLMVisionNode": SupersideAnyLLMVisionNode,
    "SupersideBriaBackgroundStandardizerNode": SupersideBriaBackgroundStandardizerNode,
    "SupersideBriaReplaceBackgroundNode": SupersideBriaReplaceBackgroundNode,
    "SupersideFlorence2CaptionNode": SupersideFlorence2CaptionNode,
    "SupersideFlorence2RegionSelectorNode": SupersideFlorence2RegionSelectorNode,
    "SupersideFluxKontextMaxMultiImageNode": SupersideFluxKontextMaxMultiImageNode,
    "SupersideGeminiOmniFlashEditNode": SupersideGeminiOmniFlashEditNode,
    "SupersideGPTImage2EditNode": SupersideGPTImage2EditNode,
    "SupersideGrokImagineImageQualityEditNode": SupersideGrokImagineImageQualityEditNode,
    "SupersideIdeogramUpscaleNode": SupersideIdeogramUpscaleNode,
    "SupersideJuggernautFluxProImg2ImgNode": SupersideJuggernautFluxProImg2ImgNode,
    "SupersideKling21ImageToVideoNode": SupersideKling21ImageToVideoNode,
    "SupersideKling25TurboProImageToVideoNode": SupersideKling25TurboProImageToVideoNode,
    "SupersideNanoBananaProEditNode": SupersideNanoBananaProEditNode,
    "SupersideNanoBananaV2EditNode": SupersideNanoBananaV2EditNode,
    "SupersidePASDUpscalerNode": SupersidePASDUpscalerNode,
    "SupersidePromptBoxNode": SupersidePromptBoxNode,
    "SupersidePromptSplitterNode": SupersidePromptSplitterNode,
    "SupersideSAM3RegionSelectorNode": SupersideSAM3RegionSelectorNode,
    "SupersideSeedanceLiteImageToVideoNode": SupersideSeedanceLiteImageToVideoNode,
    "SupersideSeedanceProImageToVideoNode": SupersideSeedanceProImageToVideoNode,
    "SupersideSeedreamV45EditNode": SupersideSeedreamV45EditNode,
    "SupersideSeedreamV5ProEditNode": SupersideSeedreamV5ProEditNode,
    "SupersideSeedVR2UpscaleImageNode": SupersideSeedVR2UpscaleImageNode,
    "SupersideSeedVRUpscaleVideoNode": SupersideSeedVRUpscaleVideoNode,
    "SupersideSmartDetailSheetNode": SupersideSmartDetailSheetNode,
    "SupersideTopazUpscaleImageNode": SupersideTopazUpscaleImageNode,
    "SupersideWan25ImageToImageNode": SupersideWan25ImageToImageNode,
    "SupersideWan25ImageToVideoNode": SupersideWan25ImageToVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SupersideAnyLLMTextNode": "Superside Any LLM Text",
    "SupersideAnyLLMVisionNode": "Superside Any LLM Vision",
    "SupersideBriaBackgroundStandardizerNode": "Superside Bria Background Standardizer (Hex Color)",
    "SupersideBriaReplaceBackgroundNode": "Superside Bria Replace Background",
    "SupersideFlorence2CaptionNode": "Superside Florence-2 Detailed Caption",
    "SupersideFlorence2RegionSelectorNode": "Superside Florence-2 Smart Region Selector",
    "SupersideFluxKontextMaxMultiImageNode": "Superside Flux Kontext Max Multi-Image Node",
    "SupersideGeminiOmniFlashEditNode": "Superside Gemini Omni Flash Edit",
    "SupersideGPTImage2EditNode": "Superside GPT Image 2 Edit",
    "SupersideGrokImagineImageQualityEditNode": "Superside Grok Imagine Image Quality Edit",
    "SupersideIdeogramUpscaleNode": "Superside Ideogram Upscale",
    "SupersideJuggernautFluxProImg2ImgNode": "Superside Juggernaut Flux Pro Image-to-Image",
    "SupersideKling21ImageToVideoNode": "Superside Kling 2.1 Image-to-Video",
    "SupersideKling25TurboProImageToVideoNode": "Superside Kling 2.5 Turbo Pro Image-to-Video",
    "SupersideNanoBananaProEditNode": "Superside Nano Banana Pro Edit Node",
    "SupersideNanoBananaV2EditNode": "Superside Nano Banana V2 Edit Node",
    "SupersidePASDUpscalerNode": "Superside PASD Upscaler Node",
    "SupersidePromptBoxNode": "Superside Prompt Box",
    "SupersidePromptSplitterNode": "Superside Prompt Splitter",
    "SupersideSAM3RegionSelectorNode": "Superside SAM 3 Smart Region Selector",
    "SupersideSeedanceLiteImageToVideoNode": "Superside Seedance Lite Image-to-Video",
    "SupersideSeedanceProImageToVideoNode": "Superside Seedance Pro Image-to-Video",
    "SupersideSeedreamV45EditNode": "Superside Seedream V4.5 Edit",
    "SupersideSeedreamV5ProEditNode": "Superside Seedream V5 Pro Edit",
    "SupersideSeedVR2UpscaleImageNode": "Superside SeedVR2 Upscale Image",
    "SupersideSeedVRUpscaleVideoNode": "Superside SeedVR Upscale Video",
    "SupersideSmartDetailSheetNode": "Superside Smart Detail Sheet",
    "SupersideTopazUpscaleImageNode": "Superside Topaz Upscale Image",
    "SupersideWan25ImageToImageNode": "Superside Wan 2.5 Image-to-Image",
    "SupersideWan25ImageToVideoNode": "Superside Wan 2.5 Image-to-Video",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
