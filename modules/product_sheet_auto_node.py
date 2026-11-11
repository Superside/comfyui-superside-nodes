"""Product Sheet (one step): photo in, product sheet out.

Product Research and Product Sheet in a single node, for when nobody needs to look at the folder in between.
It runs the research only when the folder has not been researched yet (or `redo_research` is on), so pressing
Run again rebuilds the sheet without searching and downloading a second time.
"""

import os

from .base_node import API_KEY_INPUT_SPEC
from .product_research_node import SupersideProductResearchNode, _output_root
from .product_sheet_node import SupersideProductSheetNode
from . import product_sheet_core as core


class SupersideProductSheetAutoNode:
    """Identify, research, and build the sheet in one go."""

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("sheet", "product_info", "product_dir", "report", "scene_prompt")
    FUNCTION = "run"
    DISPLAY_NAME = "Product Sheet (one step)"
    DESCRIPTION = (
        "One node for the whole product sheet: connect a photo of the product (and write the brand and model if "
        "you know them), press Run, get the 5000 x 5000 sheet and the PRODUCT INFO text. It searches the web, "
        "downloads and tags the photos, then builds the sheet. Everything is saved in one folder (output_dir, or "
        "output/product_sheets/<brand_model>); a second Run reuses that folder instead of searching again."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"api_key": API_KEY_INPUT_SPEC},
            "optional": {
                "image": ("IMAGE", {"tooltip": "A photo of the product."}),
                "product_hint": ("STRING", {"default": "", "multiline": False, "tooltip": "Brand and model, e.g. 'Acme AB-1200X'. Strongly recommended."}),
                "source_urls": ("STRING", {"default": "", "multiline": True, "tooltip": "Product pages to read first, one per line."}),
                "output_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "Folder for this product. Empty = output/product_sheets/<brand_model>."}),
                "mode": (["mixed", "real", "generated"], {"default": "mixed"}),
                "max_generated": ("INT", {"default": 5, "min": 0, "max": 12}),
                "redo_research": ("BOOLEAN", {"default": False, "tooltip": "Search and download again even if the folder is already researched."}),
                "strict": ("BOOLEAN", {"default": True, "tooltip": "Stop instead of returning a sheet that fails a hard gate."}),
            },
        }

    def run(self, api_key, image=None, product_hint="", source_urls="", output_dir="", mode="mixed", max_generated=5,
            redo_research=False, strict=True):
        folder = output_dir.strip().strip('"')
        if not folder and product_hint.strip():
            folder = os.path.join(_output_root(), core.slug(product_hint.strip()))      # stable folder, so a second Run finds it
        researched = bool(folder) and os.path.isfile(os.path.join(folder, "image_tags.json"))
        report = f"reused the researched folder {folder}"
        if redo_research or not researched:
            folder, _info, _contact, report, _scene = SupersideProductResearchNode().research(
                api_key, image=image, product_hint=product_hint, source_urls=source_urls, output_dir=folder)
        sheet, info, meta, scene = SupersideProductSheetNode().build(folder, api_key, mode=mode, max_generated=max_generated, strict=strict)
        return (sheet, info, folder, report + "\n\n" + meta, scene)
