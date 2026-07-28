import json
import os
import random

import folder_paths
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    from comfy.cli_args import args as _comfy_args
    _DISABLE_METADATA = getattr(_comfy_args, "disable_metadata", False)
except Exception:
    _DISABLE_METADATA = False


class SupersideSaveImageNode:
    """
    Superside Save Image: in-house replacement for core SaveImage. Same
    filename-counter convention ("{prefix}_{counter:05}_.png") so existing
    output folders/scripts keep working.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    DESCRIPTION = "Save images to the output folder - in-house equivalent of core SaveImage."

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not _DISABLE_METADATA:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for k, v in extra_pnginfo.items():
                        metadata.add_text(k, json.dumps(v))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}, "result": (images,)}


class SupersidePreviewImageNode(SupersideSaveImageNode):
    """
    Superside Preview Image: in-house replacement for core PreviewImage -
    saves to the temp folder with a random prefix instead of accumulating
    permanent output files.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)
        )
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "Superside"
    DESCRIPTION = "Preview images in the node graph - in-house equivalent of core PreviewImage."
