class SupersideMaskToImageNode:
    """
    Superside Mask To Image: in-house replacement for core MaskToImage.
    Flattens any batch shape to (B,1,H,W), moves the channel to last
    position, and broadcasts to 3 channels (R=G=B=mask value).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    DESCRIPTION = "Convert a MASK into a grayscale IMAGE - in-house equivalent of core MaskToImage."

    def convert(self, mask):
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return (result,)


class SupersideMaskPreviewNode(SupersideMaskToImageNode):
    """
    Superside Mask Preview: in-house replacement for comfyui_essentials'
    MaskPreview+. Converts the mask to a grayscale image (same routine as
    Mask To Image) and saves it to the temp folder as a UI preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mask": ("MASK",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    DESCRIPTION = "Preview a MASK as a grayscale image - in-house equivalent of comfyui_essentials' MaskPreview+."

    def preview(self, mask, prompt=None, extra_pnginfo=None):
        (preview_image,) = self.convert(mask)
        # Reuse Superside's own PreviewImage node so both nodes share one
        # implementation of the temp-save/UI-preview plumbing.
        from .save_image_node import SupersidePreviewImageNode

        return SupersidePreviewImageNode().save_images(preview_image, prompt=prompt, extra_pnginfo=extra_pnginfo)
