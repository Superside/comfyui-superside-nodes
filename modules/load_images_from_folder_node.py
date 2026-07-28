import os

import numpy as np
import torch
from PIL import Image, ImageOps

_VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tga")


class SupersideLoadImagesFromFolderNode:
    """
    Superside Load Images From Folder: in-house replacement for
    comfyui-kjnodes' "Load Images From Folder (KJ)". Loads every image in a
    folder into a single IMAGE batch (resized to a common size), with an
    optional load cap and start index - handy for feeding a batch loader
    into a LoRA trainer node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 1024, "min": 64, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "step": 8}),
                "keep_aspect_ratio": (["crop", "pad", "stretch"],),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "count", "image_path")
    FUNCTION = "load_images"
    DESCRIPTION = "Load every image in a folder into one IMAGE batch - in-house equivalent of comfyui-kjnodes' Load Images From Folder (KJ)."

    def _fit(self, img, width, height, keep_aspect_ratio):
        if keep_aspect_ratio == "stretch":
            return img.resize((width, height), Image.LANCZOS)
        if keep_aspect_ratio == "pad":
            return ImageOps.pad(img, (width, height), method=Image.LANCZOS, color=(0, 0, 0))
        # "crop": scale to cover, then center-crop
        return ImageOps.fit(img, (width, height), method=Image.LANCZOS)

    def load_images(self, folder, width, height, keep_aspect_ratio,
                     image_load_cap=0, start_index=0, include_subfolders=False):
        if not folder or not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")

        image_paths = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(_VALID_EXTENSIONS):
                        image_paths.append(os.path.join(root, f))
        else:
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(_VALID_EXTENSIONS):
                    image_paths.append(os.path.join(folder, f))

        image_paths = sorted(image_paths)
        if not image_paths:
            raise FileNotFoundError(f"No image files found in '{folder}'.")

        image_paths = image_paths[start_index:]
        if image_load_cap > 0:
            image_paths = image_paths[:image_load_cap]

        tensors = []
        for path in image_paths:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            img = self._fit(img, width, height, keep_aspect_ratio)
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).unsqueeze(0))

        batch = torch.cat(tensors, dim=0)
        return (batch, len(image_paths), image_paths[0] if image_paths else "")
