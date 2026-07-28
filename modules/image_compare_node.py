import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import folder_paths
    _FONT_DIRS = [os.path.join(folder_paths.base_path, "fonts")]
except Exception:
    _FONT_DIRS = []

# A couple of common system font locations as a fallback when no bundled
# .ttf is found, so the node still renders labels instead of erroring.
_FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


def _tensor_to_pil(image_tensor):
    arr = image_tensor.detach().cpu().numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor
    if arr.ndim == 4:
        arr = arr[0]
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_to_tensor(img):
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_font(font_size):
    for path in _FONT_DIRS:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith(".ttf"):
                    try:
                        return ImageFont.truetype(os.path.join(path, f), font_size)
                    except Exception:
                        continue
    for path in _FALLBACK_FONTS:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_panel(width, height, text, font_size, font_color, bg_color):
    panel = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(panel)
    font = _load_font(font_size)
    lines = text.split("\n")
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights)
    y = max(0, (height - total_h) // 2)
    for line, lh in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = max(0, (width - lw) // 2)
        draw.text((x, y), line, font=font, fill=font_color)
        y += lh
    return panel


def _resize_to_match(img, width, height):
    if img.size != (width, height):
        return img.resize((width, height), Image.BICUBIC)
    return img


class SupersideImageCompareNode:
    """
    Superside Image Compare: in-house replacement for Comfyroll's
    "CR Simple Image Compare". Stacks each image with a labeled text
    footer, then places both panels side by side with a border - same
    layout/labels as the original (text1/text2 are free labels, not
    hardcoded "BEFORE"/"AFTER").
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"multiline": True, "default": "BEFORE"}),
                "text2": ("STRING", {"multiline": True, "default": "AFTER"}),
                "footer_height": ("INT", {"default": 100, "min": 0, "max": 1024}),
                "font_size": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "mode": (["normal", "dark"],),
                "border_thickness": ("INT", {"default": 20, "min": 0, "max": 1024}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "layout"
    DESCRIPTION = "Side-by-side labeled before/after comparison - in-house equivalent of Comfyroll's CR Simple Image Compare."

    def layout(self, text1, text2, footer_height, font_size, mode, border_thickness,
               image1=None, image2=None):
        font_color = "white" if mode == "dark" else "black"
        bg_color = "black" if mode == "dark" else "white"
        outline_thickness = border_thickness // 2
        border_thickness = border_thickness // 2

        if image1 is not None and image2 is not None:
            img1 = _tensor_to_pil(image1)
            img2 = _tensor_to_pil(image2)
            image_width, image_height = img1.size
            img2 = _resize_to_match(img2, image_width, image_height)

            halves = []
            for img, text in ((img1, text1), (img2, text2)):
                if footer_height > 0:
                    panel = _text_panel(image_width, footer_height, text, font_size, font_color, bg_color)
                    combined = Image.new("RGB", (image_width, image_height + footer_height), bg_color)
                    combined.paste(img, (0, 0))
                    combined.paste(panel, (0, image_height))
                else:
                    combined = img
                if outline_thickness > 0:
                    combined = ImageOps.expand(combined, outline_thickness, fill=bg_color)
                halves.append(combined)

            h = max(h.size[1] for h in halves)
            total_w = sum(h.size[0] for h in halves)
            result_img = Image.new("RGB", (total_w, h), bg_color)
            x = 0
            for half in halves:
                result_img.paste(half, (x, 0))
                x += half.size[0]
        else:
            result_img = Image.new("RGB", (512, 512), bg_color)

        if border_thickness > 0:
            result_img = ImageOps.expand(result_img, border_thickness, bg_color)

        return (_pil_to_tensor(result_img),)


class SupersideImageComparerNode:
    """
    Superside Image Comparer: simplified in-house replacement for
    rgthree's "Image Comparer". rgthree's node is a custom interactive
    slide/click widget implemented in frontend JS (web/comparer.js) - that
    interactivity can't be replicated by a python-only node. This node
    instead produces a static side-by-side image (same visual comparison,
    no in-canvas slider), which is enough for downstream use in a pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "optional": {
                "gap": ("INT", {"default": 8, "min": 0, "max": 256}),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "compare"
    DESCRIPTION = (
        "Static side-by-side comparison of two images - simplified in-house "
        "equivalent of rgthree's Image Comparer (no interactive slider; that "
        "requires the original package's frontend JS widget)."
    )

    def compare(self, image_a, image_b, gap=8):
        img_a = _tensor_to_pil(image_a)
        img_b = _tensor_to_pil(image_b)
        img_b = _resize_to_match(img_b, img_a.size[0], img_a.size[1])

        total_w = img_a.size[0] + gap + img_b.size[0]
        h = max(img_a.size[1], img_b.size[1])
        canvas = Image.new("RGB", (total_w, h), "black")
        canvas.paste(img_a, (0, 0))
        canvas.paste(img_b, (img_a.size[0] + gap, 0))
        return (_pil_to_tensor(canvas),)
