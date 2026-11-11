"""Product Research: from one photo (or a name) to a folder that describes the product.

Step 1 of 2 (step 2 is Product Sheet). It identifies the product, searches the web for pages that show it,
downloads the photos named after its model, tags every photo, and writes the product info text. Everything
is saved in one folder so the slow, paid part runs once per product and the sheet can be rebuilt freely.

Nothing is discarded: every downloaded photo stays in the folder with its tags; photos the tagger marks
"skip" are simply not offered to the sheet.
"""

import json
import logging
import os

import numpy as np
import torch
from PIL import Image

from . import product_sheet_core as core
from .base_node import API_KEY_INPUT_SPEC, APIClientMixin, SupersideFalNode

logger = logging.getLogger(__name__)


def _output_root():
    try:
        import folder_paths
        return os.path.join(folder_paths.get_output_directory(), "product_sheets")
    except Exception:
        return os.path.join(os.getcwd(), "product_sheets")


class SupersideProductResearchNode(SupersideFalNode, APIClientMixin):
    """Identify a product, collect its photos from the web, tag them and write its info text."""

    CATEGORY = "Superside"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("product_dir", "product_info", "contact_sheet", "report")
    FUNCTION = "research"
    DISPLAY_NAME = "Product Research"
    DESCRIPTION = (
        "Step 1 of the product sheet: give it a photo of a product (and, if you know them, the brand and "
        "model). It identifies the product, finds web pages that show it, downloads the photos, tags each one "
        "(view, state, parts, clean or not) and writes a PRODUCT INFO text you can paste into prompts. "
        "Everything is saved in product_dir; connect product_dir to Product Sheet. Run it once per product."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"api_key": API_KEY_INPUT_SPEC},
            "optional": {
                "image": ("IMAGE", {"tooltip": "A photo of the product. Optional if product_hint names the brand and model."}),
                "product_hint": ("STRING", {"default": "", "multiline": False,
                                            "tooltip": "Brand and model if you know them, e.g. 'Acme AB-1200X'. Makes identification certain."}),
                "source_urls": ("STRING", {"default": "", "multiline": True,
                                           "tooltip": "Product pages to read first, one per line. Pages that refuse automated access are skipped."}),
                "output_dir": ("STRING", {"default": "", "multiline": False,
                                          "tooltip": "Folder for this product. Empty = ComfyUI output/product_sheets/<brand_model>."}),
                "search_pages": ("INT", {"default": 8, "min": 0, "max": 20, "tooltip": "How many pages to look for with web search. 0 = only source_urls."}),
                "max_images": ("INT", {"default": 40, "min": 1, "max": 200}),
                "min_image_side": ("INT", {"default": 700, "min": 200, "max": 4000, "tooltip": "Smaller photos are not kept."}),
                "llm_model": ("STRING", {"default": "anthropic/claude-sonnet-4.6", "tooltip": "Identifies the product and writes the info text."}),
                "tag_model": ("STRING", {"default": "google/gemini-2.5-flash", "tooltip": "Tags every photo. A cheap vision model is enough."}),
            },
        }

    def research(self, api_key, image=None, product_hint="", source_urls="", output_dir="", search_pages=8,
                 max_images=40, min_image_side=700, llm_model="anthropic/claude-sonnet-4.6", tag_model="google/gemini-2.5-flash"):
        client = self.get_client(api_key)
        call = lambda endpoint, arguments: self.call_api(client, endpoint, arguments)   # noqa: E731
        pil = None
        if image is not None:
            pil = Image.fromarray((image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
        if pil is None and not product_hint.strip():
            raise ValueError("Connect an image or write the brand and model in product_hint.")

        report = []
        if pil is not None:
            ident = core.identify(call, pil, product_hint.strip(), llm_model)
        else:
            ident = core.parse_json(core.llm(call, llm_model, core.IDENTIFY.format(hint=f" named: {product_hint.strip()}"), max_tokens=600))
        product = {"brand": ident.get("brand", ""), "model": ident.get("model", ""), "name": ident.get("name", ""),
                   "colour": ident.get("colour", ""),
                   "category": ident.get("category", ""), "identified_with": ident.get("confidence", ""), "hint": product_hint.strip()}
        report.append(f'identified: {product["brand"]} {product["model"]} - {product["name"]} ({product["identified_with"]} confidence)')
        if ident.get("alternatives"):
            report.append("could also be: " + ", ".join(str(a) for a in ident["alternatives"]) + " - write the model in product_hint to be sure")

        folder = output_dir.strip() or os.path.join(_output_root(), core.slug(f'{product["brand"]}_{product["model"] or product["name"]}'))
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        if pil is not None:
            pil.save(os.path.join(folder, "images", "input_photo.jpg"), "JPEG", quality=95)

        pages = [u.strip() for u in source_urls.splitlines() if u.strip().startswith("http")]
        if search_pages:
            query = ident.get("search") or f'{product["brand"]} {product["model"]} {product["name"]}'
            pages += [u for u in core.search_pages(call, query, search_pages) if u not in pages]
        tokens = core.model_tokens(product["model"])
        notes, saved, used_pages = [], [], []
        for url in pages:
            if len(saved) >= max_images or not tokens:
                break
            urls, text = core.page_images(url, tokens)
            if not urls:
                report.append(f"no usable photos: {url}")
                continue
            got = core.download_images(urls, os.path.join(folder, "images"), max_images - len(saved), min_image_side, referer=url)
            saved += got
            if got:
                used_pages.append(url)
                notes.append(text[:2500])
            report.append(f"{len(got)} photos: {url}")
        if not tokens:
            report.append("no model number was identified: photos cannot be matched to the product, only the input photo is kept")
        product["sources"] = used_pages
        json.dump(product, open(os.path.join(folder, "product.json"), "w", encoding="utf-8"), indent=1)

        tags = core.tag_images(call, folder, product, tag_model)
        uses = {}
        other = sum(1 for t in tags.values() if t.get("same_product") is False)
        for t in tags.values():
            uses[t.get("use", "?")] = uses.get(t.get("use", "?"), 0) + 1
        if other:
            report.append(f"{other} photos show another colour or model and are kept but not used")
        report.append(f"{len(tags)} photos tagged: " + ", ".join(f"{k} {v}" for k, v in sorted(uses.items())))
        info = core.write_info(call, folder, product, tags, "\n".join(notes)[:6000], llm_model) if tags else ""

        cands = core.candidates(tags)                       # what the sheet may use, best first ...
        cands += [k for k in tags if k not in cands]          # ... then everything else that was found, so nothing is hidden
        cands = cands or [f"images/{f}" for f in core.list_images(os.path.join(folder, "images"))]
        sheet = core.contact_sheet(folder, cands) if cands else Image.new("RGB", (512, 512), (24, 24, 24))
        tensor = torch.from_numpy(np.array(sheet).astype(np.float32) / 255.0).unsqueeze(0)
        text = "\n".join(report)
        open(os.path.join(folder, "research_report.txt"), "w", encoding="utf-8").write(text + "\n")
        return (folder, info, tensor, text)
