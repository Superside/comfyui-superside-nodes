import math

import comfy.utils


class SupersideImageScaleToTotalPixelsNode:
    """
    Superside Scale Image to Total Pixels: in-house replacement for core
    ImageScaleToTotalPixels. Same exact formula: scale_by = sqrt(target_px /
    (W*H)), applied identically to width and height (preserves aspect
    ratio), then rounded to a multiple of resolution_steps.
    """

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods,),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 1, "min": 1, "max": 256}),
            }
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    DESCRIPTION = "Resize an image to hit a target megapixel count, preserving aspect ratio - in-house equivalent of core ImageScaleToTotalPixels."

    def upscale(self, image, upscale_method, megapixels, resolution_steps=1):
        samples = image.movedim(-1, 1)
        total = megapixels * 1024 * 1024
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps

        s = comfy.utils.common_upscale(samples, int(width), int(height), upscale_method, "disabled")
        s = s.movedim(1, -1)
        return (s,)
