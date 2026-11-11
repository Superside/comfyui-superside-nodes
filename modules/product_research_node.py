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
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("product_dir", "product_info", "contact_sheet", "report", "scene_prompt")
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
                "min_images": ("INT", {"default": 40, "min": 1, "max": 150,
                                       "tooltip": "Keep searching for more pages (up to 3 rounds) until this many photos are saved."}),
                "max_images": ("INT", {"default": 80, "min": 1, "max": 200}),
                "min_image_side": ("INT", {"default": 700, "min": 200, "max": 4000, "tooltip": "Smaller photos are not kept."}),
                "llm_model": ("STRING", {"default": "anthropic/claude-sonnet-4.6", "tooltip": "Identifies the product and writes the info text."}),
                "tag_model": ("STRING", {"default": "google/gemini-2.5-flash", "tooltip": "Tags every photo. A cheap vision model is enough."}),
                "search_model": ("STRING", {"default": "anthropic/claude-sonnet-4.6:online",
                                            "tooltip": "Model with web search that finds the pages. Measured on one product: Claude found 10 pages "
                                                       "(5 readable) for about $0.27 a search; google/gemini-2.5-flash:online found 5 (2 readable) for "
                                                       "under a cent."}),
                "image_search": ("BOOLEAN", {"default": True,
                                             "tooltip": "Also look for photos with an image search (open-source ddgs, no key, no cost). "
                                                        "Finds many more photos; the tagger then drops the ones that are not the product."}),
            },
        }

    def research(self, api_key, image=None, product_hint="", source_urls="", output_dir="", search_pages=8,
                 min_images=40, max_images=80, min_image_side=700, llm_model="anthropic/claude-sonnet-4.6", tag_model="google/gemini-2.5-flash",
                 search_model="anthropic/claude-sonnet-4.6:online", image_search=True):
        core.SEARCH_MODEL = (search_model or "").strip() or core.SEARCH_MODEL
        client = self.get_client(api_key)
        call = lambda endpoint, arguments: self.call_api(client, endpoint, arguments)   # noqa: E731
        pil = None
        if image is not None:
            pil = Image.fromarray((image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")

        report = []
        # what the user gave besides the photo: a name, and/or links. A link names the product in its address and its
        # page title, so it identifies the product even when nothing is typed in product_hint.
        given_urls = [u.strip() for u in source_urls.splitlines() if u.strip().startswith("http")]
        clues = [product_hint.strip()] if product_hint.strip() else []
        for u in given_urls[:3]:
            clues.append("link given: " + u[:300])
            try:
                import re as _re
                page = core.fetch(u, timeout=25).decode("utf-8", "ignore")
                title = _re.search(r"<title[^>]*>(.*?)</title>", page, _re.S | _re.I)
                if title and not _re.search(r"denied|captcha|robot|access|^\s*amazon\.[a-z.]+\s*$", title.group(1), _re.I):
                    clues.append("that page is titled: " + " ".join(title.group(1).split())[:200])
                    continue
            except Exception:
                pass
            found = core.lookup_link(call, u)                  # the page could not be read: ask a model with web search
            if found and "unknown" not in found.lower():
                clues.append("a web lookup of that link says it sells: " + found)
                report.append(f"link not readable by script, identified by web lookup: {found}")
        hint = "; ".join(clues)
        if pil is None and not hint:
            raise ValueError("Connect an image, write the brand and model in product_hint, or give a product link in source_urls.")
        if pil is not None:
            ident = core.identify(call, pil, hint, llm_model)
            if not hint and (not ident.get("model") or ident.get("confidence") == "low"):
                found = core.lookup_photo(call, pil)           # no name on the product: try to find it on the web by its look
                if found and "unknown" not in found.lower():
                    report.append(f"no name visible in the photo; a web lookup suggests: {found}")
                    ident = core.identify(call, pil, "a web lookup by image suggests: " + found, llm_model)
            if ident.get("photo_matches") is False:
                report.append("the photo shows a different product than the link: the link was followed, the photo was ignored")
                pil = None
        else:
            ident = core.parse_json(core.llm(call, llm_model, core.IDENTIFY.format(hint=f" described by: {hint}"), max_tokens=600))
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

        import urllib.parse
        host = lambda u: urllib.parse.urlparse(u).netloc.replace("www.", "")       # noqa: E731
        pages = [u.strip() for u in source_urls.splitlines() if u.strip().startswith("http")]
        query = ident.get("search") or f'{product["brand"]} {product["model"]} {product["name"]}'
        tokens = core.model_tokens(product["model"])
        notes, saved, used_pages, tried, seen = [], [], [], set(), set()
        images_dir = os.path.join(folder, "images")
        for round_no in range(4):                              # round 0 = the given links, then up to 3 page searches
            if round_no == 2 and image_search and len(saved) < max_images:
                # after the links and the first page search (the maker's sites, which also give the product text):
                # an image search, which finds photos directly instead of pages
                results, note = core.search_images(query, min_side=min_image_side)
                before = len(saved)
                for r in results:
                    if len(saved) >= max_images:
                        break
                    saved += core.download_images([r["image"]], images_dir, 1, min_image_side, referer=r["page"] or None, seen=seen, timeout=20)
                report.append(f"{note}; {len(saved) - before} new photos saved")
            if round_no and not search_pages:
                continue
            if round_no and len(saved) >= min_images and (notes or round_no > 1):
                continue
            if round_no:
                found, pages = core.search_pages(call, query, max(search_pages, 10), tried, round_no), []
                for u in found:                                 # one page per site, and no site twice
                    if host(u) not in tried and host(u) not in {host(x) for x in pages}:
                        pages.append(u)
                report.append(f"search {round_no}: {len(pages)} new pages")
            for url in pages:
                if len(saved) >= max_images:
                    break
                tried.add(host(url))
                urls, text = core.page_images(url, tokens)
                if not urls:
                    report.append(f"no usable photos: {url}")
                    continue
                got = core.download_images(urls, images_dir, max_images - len(saved), min_image_side, referer=url, seen=seen)
                saved += got
                if got:
                    used_pages.append(url)
                    notes.append(text[:2500])
                report.append(f"{len(got)} photos: {url}")
        if len(saved) < min_images:
            report.append(f"only {len(saved)} photos found (wanted {min_images}): add product pages in source_urls, or lower min_image_side")
        if not tokens:
            report.append("no model number was identified: photos were taken from the pages as a whole and checked one by one "
                          "against the input photo - write the model in product_hint for a sharper search")
        product["sources"] = used_pages
        json.dump(product, open(os.path.join(folder, "product.json"), "w", encoding="utf-8"), indent=1)

        def tag_and_sort():
            """Tag what is new; photos that turned out not to show the product go to their own folder, out of the tags."""
            tags = core.tag_images(call, folder, product, tag_model)
            for note in core.lock_variant(call, folder, product, tags, llm_model, had_photo=pil is not None):
                if note not in report:
                    report.append(note)
            strangers = [k for k, t in tags.items() if t.get("view") == "not the product"]
            if strangers:
                os.makedirs(os.path.join(folder, "not_product"), exist_ok=True)
                for k in strangers:
                    os.replace(os.path.join(folder, k), os.path.join(folder, "not_product", os.path.basename(k)))
                    tags.pop(k)
                json.dump(tags, open(os.path.join(folder, "image_tags.json"), "w", encoding="utf-8"), indent=1)
                report.append(f"{len(strangers)} downloaded photos do not show the product: moved to not_product/")
            return tags

        tags = tag_and_sort()
        uses = {}
        # A sheet is built from categories. The ones it cannot do without and that came back empty are searched for by
        # name, once: a close-up of the controls, of the logo, the product in a hand, its accessories, the product alone.
        cats = core.categorize(folder, tags)
        profile = core.product_profile(call, folder, product, "", llm_model)        # which conditional rules apply to it
        wanted = [(label, q) for label, _views, q in core.CATEGORIES if q and not cats[label] and core.category_needed(label, profile)]
        if wanted and image_search:
            for label, q in wanted:
                results, note = core.search_images(query, per_query=40, min_side=min(min_image_side, 500), patterns=(q,))
                got = 0
                for r in results:
                    if got >= 10:
                        break
                    new = core.download_images([r["image"]], images_dir, 1, min(min_image_side, 500), referer=r["page"] or None, seen=seen, timeout=20)
                    got += len(new)
                    saved += new
                report.append(f"no photo in category {label}: searched for it, {got} more photos saved")
            tags = tag_and_sort()
            cats = core.categorize(folder, tags)
        json.dump(cats, open(os.path.join(folder, "categories.json"), "w", encoding="utf-8"), indent=1)
        report.append("by category: " + ", ".join(f"{label} {len(keys)}" for label, keys in cats.items()))
        empty = [label for label, _v, q in core.CATEGORIES if q and not cats[label] and core.category_needed(label, profile)]
        if empty:
            report.append("still no photo for: " + ", ".join(empty) + " - the sheet will cut these from a whole-product photo")
        other = sum(1 for t in tags.values() if t.get("same_product") is False)
        for t in tags.values():
            uses[t.get("use", "?")] = uses.get(t.get("use", "?"), 0) + 1
        if other:
            report.append(f"{other} photos show another colour or model and are kept but not used")
        report.append(f"{len(tags)} photos tagged: " + ", ".join(f"{k} {v}" for k, v in sorted(uses.items())))
        info = core.write_info(call, folder, product, tags, "\n".join(notes)[:6000], llm_model) if tags else ""
        report.append("profile: " + core.profile_text(profile))
        # what the product is made of, counted (4 dials, 1 door ...): saved, and put first in the product info so every
        # prompt that uses the info carries the counts
        spec = core.product_spec(call, folder, product, info, tags, llm_model, notes="\n".join(notes)[:6000]) if tags else None
        if spec and spec.get("summary"):
            info = "COMPOSITION " + str(spec["summary"]).strip() + "\n" + info
            open(os.path.join(folder, "product_info.txt"), "w", encoding="utf-8").write(info + "\n")
            report.append("composition: " + str(spec["summary"])[:220])
        scene = core.write_scene(call, folder, product, info, llm_model)

        cands = core.candidates(tags)                       # what the sheet may use, best first ...
        cands += [k for k in tags if k not in cands]          # ... then everything else that was found, so nothing is hidden
        cands = cands or [f"images/{f}" for f in core.list_images(os.path.join(folder, "images"))]
        sheet = core.category_sheet(folder, cats) if any(cats.values()) else (
            core.contact_sheet(folder, cands) if cands else Image.new("RGB", (512, 512), (24, 24, 24)))
        sheet.convert("RGB").save(os.path.join(folder, "contact_sheet_by_category.jpg"), quality=85)
        tensor = torch.from_numpy(np.array(sheet).astype(np.float32) / 255.0).unsqueeze(0)
        text = "\n".join(report)
        open(os.path.join(folder, "research_report.txt"), "w", encoding="utf-8").write(text + "\n")
        return (folder, info, tensor, text, scene)
