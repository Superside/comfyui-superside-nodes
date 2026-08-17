try:
    # Normal path when ComfyUI loads this as a proper package.
    from .modules.any_llm_text_node import SupersideAnyLLMTextNode
    from .modules.any_llm_vision_node import SupersideAnyLLMVisionNode
    from .modules.bria_background_replace_node import SupersideBriaBackgroundReplaceNode
    from .modules.bria_background_standardizer_node import SupersideBriaBackgroundStandardizerNode
    from .modules.bria_replace_background_node import SupersideBriaReplaceBackgroundNode
    from .modules.color_grading_node import SupersideColorGradingNode
    from .modules.color_match_node import SupersideColorMatchNode
    from .modules.crop_by_region_node import SupersideCropByRegionNode
    from .modules.crystal_upscaler_node import SupersideCrystalUpscalerNode
    from .modules.stitch_region_node import SupersideStitchRegionNode
    from .modules.skin_intensity_prompt_node import SupersideSkinIntensityPromptNode
    from .modules.florence_2_caption_node import SupersideFlorence2CaptionNode
    from .modules.florence_2_region_selector_node import SupersideFlorence2RegionSelectorNode
    from .modules.flux_kontext_max_multi_node import SupersideFluxKontextMaxMultiImageNode
    from .modules.flux_pro_fill_node import SupersideFluxProFillNode
    from .modules.gemini_omni_flash_edit_node import SupersideGeminiOmniFlashEditNode
    from .modules.gpt_image_2_edit_node import SupersideGPTImage2EditNode
    from .modules.grok_imagine_image_quality_edit_node import SupersideGrokImagineImageQualityEditNode
    from .modules.ideogram_upscale_node import SupersideIdeogramUpscaleNode
    from .modules.image_retouch_node import SupersideImageRetouchNode
    from .modules.juggernaut_flux_pro_img2img_node import SupersideJuggernautFluxProImg2ImgNode
    from .modules.kling_21_image_to_video_node import SupersideKling21ImageToVideoNode
    from .modules.kling_25_turbo_pro_image_to_video_node import SupersideKling25TurboProImageToVideoNode
    from .modules.manual_detail_sheet_node import SupersideManualDetailSheetNode
    from .modules.nano_banana_pro_node import SupersideNanoBananaProEditNode
    from .modules.nano_banana_v2_edit_node import SupersideNanoBananaV2EditNode
    from .modules.normalize_product_node import SupersideNormalizeProductNode
    from .modules.pasd_upscaler_node import SupersidePASDUpscalerNode
    from .modules.prompt_box_node import SupersidePromptBoxNode
    from .modules.prompt_splitter_node import SupersidePromptSplitterNode
    from .modules.resize_long_side_node import SupersideResizeLongSideNode
    from .modules.resize_to_match_node import SupersideResizeToMatchNode
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
    from .modules.white_balance_node import SupersideWhiteBalanceNode
    from .modules.z_image_inpaint_lora_node import SupersideZImageInpaintLoraNode
    from .modules.skin_detail_z_image_lora_node import SupersideSkinDetailZImageLoraNode
    from .modules.z_image_lora_trainer_node import SupersideZImageLoraTrainerNode
    from .modules.load_image_node import SupersideLoadImageNode
    from .modules.save_image_node import SupersideSaveImageNode, SupersidePreviewImageNode
    from .modules.image_scale_to_total_pixels_node import SupersideImageScaleToTotalPixelsNode
    from .modules.image_composite_masked_node import SupersideImageCompositeMaskedNode
    from .modules.mask_to_image_node import SupersideMaskToImageNode, SupersideMaskPreviewNode
    from .modules.grow_mask_with_blur_node import SupersideGrowMaskWithBlurNode
    from .modules.cut_by_mask_node import SupersideCutByMaskNode
    from .modules.combine_prompt_node import SupersideCombinePromptNode
    from .modules.image_compare_node import SupersideImageCompareNode, SupersideImageComparerNode
    from .modules.load_images_from_folder_node import SupersideLoadImagesFromFolderNode
    from .modules.text_preview_node import SupersideTextPreviewNode
    from .modules.portrait_sections_node import SupersidePortraitSectionsNode
    from .modules.scene_exclusion_mask_node import SupersideSceneExclusionMaskNode
    from .modules.scene_realism_prompt_node import SupersideSceneRealismPromptNode

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
    from modules.bria_background_replace_node import SupersideBriaBackgroundReplaceNode
    from modules.bria_background_standardizer_node import SupersideBriaBackgroundStandardizerNode
    from modules.bria_replace_background_node import SupersideBriaReplaceBackgroundNode
    from modules.color_grading_node import SupersideColorGradingNode
    from modules.color_match_node import SupersideColorMatchNode
    from modules.crop_by_region_node import SupersideCropByRegionNode
    from modules.crystal_upscaler_node import SupersideCrystalUpscalerNode
    from modules.stitch_region_node import SupersideStitchRegionNode
    from modules.skin_intensity_prompt_node import SupersideSkinIntensityPromptNode
    from modules.florence_2_caption_node import SupersideFlorence2CaptionNode
    from modules.florence_2_region_selector_node import SupersideFlorence2RegionSelectorNode
    from modules.flux_kontext_max_multi_node import SupersideFluxKontextMaxMultiImageNode
    from modules.flux_pro_fill_node import SupersideFluxProFillNode
    from modules.gemini_omni_flash_edit_node import SupersideGeminiOmniFlashEditNode
    from modules.gpt_image_2_edit_node import SupersideGPTImage2EditNode
    from modules.grok_imagine_image_quality_edit_node import SupersideGrokImagineImageQualityEditNode
    from modules.ideogram_upscale_node import SupersideIdeogramUpscaleNode
    from modules.image_retouch_node import SupersideImageRetouchNode
    from modules.juggernaut_flux_pro_img2img_node import SupersideJuggernautFluxProImg2ImgNode
    from modules.kling_21_image_to_video_node import SupersideKling21ImageToVideoNode
    from modules.kling_25_turbo_pro_image_to_video_node import SupersideKling25TurboProImageToVideoNode
    from modules.manual_detail_sheet_node import SupersideManualDetailSheetNode
    from modules.nano_banana_pro_node import SupersideNanoBananaProEditNode
    from modules.nano_banana_v2_edit_node import SupersideNanoBananaV2EditNode
    from modules.normalize_product_node import SupersideNormalizeProductNode
    from modules.pasd_upscaler_node import SupersidePASDUpscalerNode
    from modules.prompt_box_node import SupersidePromptBoxNode
    from modules.prompt_splitter_node import SupersidePromptSplitterNode
    from modules.resize_long_side_node import SupersideResizeLongSideNode
    from modules.resize_to_match_node import SupersideResizeToMatchNode
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
    from modules.white_balance_node import SupersideWhiteBalanceNode
    from modules.z_image_inpaint_lora_node import SupersideZImageInpaintLoraNode
    from modules.skin_detail_z_image_lora_node import SupersideSkinDetailZImageLoraNode
    from modules.z_image_lora_trainer_node import SupersideZImageLoraTrainerNode
    from modules.load_image_node import SupersideLoadImageNode
    from modules.save_image_node import SupersideSaveImageNode, SupersidePreviewImageNode
    from modules.image_scale_to_total_pixels_node import SupersideImageScaleToTotalPixelsNode
    from modules.image_composite_masked_node import SupersideImageCompositeMaskedNode
    from modules.mask_to_image_node import SupersideMaskToImageNode, SupersideMaskPreviewNode
    from modules.grow_mask_with_blur_node import SupersideGrowMaskWithBlurNode
    from modules.cut_by_mask_node import SupersideCutByMaskNode
    from modules.combine_prompt_node import SupersideCombinePromptNode
    from modules.image_compare_node import SupersideImageCompareNode, SupersideImageComparerNode
    from modules.load_images_from_folder_node import SupersideLoadImagesFromFolderNode
    from modules.text_preview_node import SupersideTextPreviewNode
    from modules.portrait_sections_node import SupersidePortraitSectionsNode
    from modules.scene_exclusion_mask_node import SupersideSceneExclusionMaskNode
    from modules.scene_realism_prompt_node import SupersideSceneRealismPromptNode

NODE_CLASS_MAPPINGS = {
    "SupersideAnyLLMTextNode": SupersideAnyLLMTextNode,
    "SupersideAnyLLMVisionNode": SupersideAnyLLMVisionNode,
    "SupersideBriaBackgroundReplaceNode": SupersideBriaBackgroundReplaceNode,
    "SupersideBriaBackgroundStandardizerNode": SupersideBriaBackgroundStandardizerNode,
    "SupersideBriaReplaceBackgroundNode": SupersideBriaReplaceBackgroundNode,
    "SupersideColorGradingNode": SupersideColorGradingNode,
    "SupersideColorMatchNode": SupersideColorMatchNode,
    "SupersideCropByRegionNode": SupersideCropByRegionNode,
    "SupersideCrystalUpscalerNode": SupersideCrystalUpscalerNode,
    "SupersideStitchRegionNode": SupersideStitchRegionNode,
    "SupersideSkinIntensityPromptNode": SupersideSkinIntensityPromptNode,
    "SupersideFlorence2CaptionNode": SupersideFlorence2CaptionNode,
    "SupersideFlorence2RegionSelectorNode": SupersideFlorence2RegionSelectorNode,
    "SupersideFluxKontextMaxMultiImageNode": SupersideFluxKontextMaxMultiImageNode,
    "SupersideFluxProFillNode": SupersideFluxProFillNode,
    "SupersideGeminiOmniFlashEditNode": SupersideGeminiOmniFlashEditNode,
    "SupersideGPTImage2EditNode": SupersideGPTImage2EditNode,
    "SupersideGrokImagineImageQualityEditNode": SupersideGrokImagineImageQualityEditNode,
    "SupersideIdeogramUpscaleNode": SupersideIdeogramUpscaleNode,
    "SupersideImageRetouchNode": SupersideImageRetouchNode,
    "SupersideJuggernautFluxProImg2ImgNode": SupersideJuggernautFluxProImg2ImgNode,
    "SupersideKling21ImageToVideoNode": SupersideKling21ImageToVideoNode,
    "SupersideKling25TurboProImageToVideoNode": SupersideKling25TurboProImageToVideoNode,
    "SupersideManualDetailSheetNode": SupersideManualDetailSheetNode,
    "SupersideNanoBananaProEditNode": SupersideNanoBananaProEditNode,
    "SupersideNanoBananaV2EditNode": SupersideNanoBananaV2EditNode,
    "SupersideNormalizeProductNode": SupersideNormalizeProductNode,
    "SupersidePASDUpscalerNode": SupersidePASDUpscalerNode,
    "SupersidePromptBoxNode": SupersidePromptBoxNode,
    "SupersidePromptSplitterNode": SupersidePromptSplitterNode,
    "SupersideResizeLongSideNode": SupersideResizeLongSideNode,
    "SupersideResizeToMatchNode": SupersideResizeToMatchNode,
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
    "SupersideWhiteBalanceNode": SupersideWhiteBalanceNode,
    "SupersideZImageInpaintLoraNode": SupersideZImageInpaintLoraNode,
    "SupersideSkinDetailZImageLoraNode": SupersideSkinDetailZImageLoraNode,
    "SupersideZImageLoraTrainerNode": SupersideZImageLoraTrainerNode,
    "SupersideLoadImageNode": SupersideLoadImageNode,
    "SupersideSaveImageNode": SupersideSaveImageNode,
    "SupersidePreviewImageNode": SupersidePreviewImageNode,
    "SupersideImageScaleToTotalPixelsNode": SupersideImageScaleToTotalPixelsNode,
    "SupersideImageCompositeMaskedNode": SupersideImageCompositeMaskedNode,
    "SupersideMaskToImageNode": SupersideMaskToImageNode,
    "SupersideMaskPreviewNode": SupersideMaskPreviewNode,
    "SupersideGrowMaskWithBlurNode": SupersideGrowMaskWithBlurNode,
    "SupersideCutByMaskNode": SupersideCutByMaskNode,
    "SupersideCombinePromptNode": SupersideCombinePromptNode,
    "SupersideImageCompareNode": SupersideImageCompareNode,
    "SupersideImageComparerNode": SupersideImageComparerNode,
    "SupersideLoadImagesFromFolderNode": SupersideLoadImagesFromFolderNode,
    "SupersideTextPreviewNode": SupersideTextPreviewNode,
    "SupersidePortraitSectionsNode": SupersidePortraitSectionsNode,
    "SupersideSceneExclusionMaskNode": SupersideSceneExclusionMaskNode,
    "SupersideSceneRealismPromptNode": SupersideSceneRealismPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SupersideAnyLLMTextNode": "Superside Any LLM Text",
    "SupersideAnyLLMVisionNode": "Superside Any LLM Vision",
    "SupersideBriaBackgroundReplaceNode": "Superside Bria Background Replace",
    "SupersideBriaBackgroundStandardizerNode": "Superside Bria Background Standardizer (Hex Color)",
    "SupersideBriaReplaceBackgroundNode": "Superside Bria Replace Background V2",
    "SupersideColorGradingNode": "Superside Color Grading",
    "SupersideColorMatchNode": "Superside Color Match",
    "SupersideCropByRegionNode": "Superside Crop By Region",
    "SupersideCrystalUpscalerNode": "Superside Crystal Upscaler (portrait detail)",
    "SupersideStitchRegionNode": "Superside Stitch Region",
    "SupersideSkinIntensityPromptNode": "Superside Skin Intensity Dial",
    "SupersideFlorence2CaptionNode": "Superside Florence-2 Detailed Caption",
    "SupersideFlorence2RegionSelectorNode": "Superside Florence-2 Smart Region Selector",
    "SupersideFluxKontextMaxMultiImageNode": "Superside Flux Kontext Max Multi-Image Node",
    "SupersideFluxProFillNode": "Superside FLUX.1 Pro Fill (dedicated inpaint)",
    "SupersideGeminiOmniFlashEditNode": "Superside Gemini Omni Flash Edit",
    "SupersideGPTImage2EditNode": "Superside GPT Image 2 Edit",
    "SupersideGrokImagineImageQualityEditNode": "Superside Grok Imagine Image Quality Edit",
    "SupersideIdeogramUpscaleNode": "Superside Ideogram Upscale",
    "SupersideImageRetouchNode": "Superside Image Retouch",
    "SupersideJuggernautFluxProImg2ImgNode": "Superside Juggernaut Flux Pro Image-to-Image",
    "SupersideKling21ImageToVideoNode": "Superside Kling 2.1 Image-to-Video",
    "SupersideKling25TurboProImageToVideoNode": "Superside Kling 2.5 Turbo Pro Image-to-Video",
    "SupersideManualDetailSheetNode": "Superside Manual Detail Sheet",
    "SupersideNanoBananaProEditNode": "Superside Nano Banana Pro Edit Node",
    "SupersideNanoBananaV2EditNode": "Superside Nano Banana V2 Edit Node",
    "SupersideNormalizeProductNode": "Superside Normalize Product",
    "SupersidePASDUpscalerNode": "Superside PASD Upscaler Node",
    "SupersidePromptBoxNode": "Superside Prompt Box",
    "SupersidePromptSplitterNode": "Superside Prompt Splitter",
    "SupersideResizeLongSideNode": "Superside Resize (Long Side)",
    "SupersideResizeToMatchNode": "Superside Resize To Match",
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
    "SupersideWhiteBalanceNode": "Superside White Balance",
    "SupersideZImageInpaintLoraNode": "Superside Z-Image Turbo Inpaint+LoRA",
    "SupersideSkinDetailZImageLoraNode": "Superside Z-Image Skin-Detail Inpaint (fixed LoRA)",
    "SupersideZImageLoraTrainerNode": "Superside Z-Image LoRA Trainer",
    "SupersideLoadImageNode": "Superside Load Image",
    "SupersideSaveImageNode": "Superside Save Image",
    "SupersidePreviewImageNode": "Superside Preview Image",
    "SupersideImageScaleToTotalPixelsNode": "Superside Scale Image to Total Pixels",
    "SupersideImageCompositeMaskedNode": "Superside Image Composite Masked",
    "SupersideMaskToImageNode": "Superside Mask To Image",
    "SupersideMaskPreviewNode": "Superside Mask Preview",
    "SupersideGrowMaskWithBlurNode": "Superside Grow Mask With Blur",
    "SupersideCutByMaskNode": "Superside Cut By Mask",
    "SupersideCombinePromptNode": "Superside Combine Prompt",
    "SupersideImageCompareNode": "Superside Image Compare",
    "SupersideImageComparerNode": "Superside Image Comparer",
    "SupersideLoadImagesFromFolderNode": "Superside Load Images From Folder",
    "SupersideTextPreviewNode": "Superside Text Preview",
    "SupersidePortraitSectionsNode": "Superside Portrait Sections",
    "SupersideSceneExclusionMaskNode": "Superside Scene Exclusion Mask (generic)",
    "SupersideSceneRealismPromptNode": "Superside Scene Realism Dial",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
