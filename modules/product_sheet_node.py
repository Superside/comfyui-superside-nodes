"""Product Sheet: one 5000 x 5000 reference sheet from a researched product folder.

Step 2 of 2 (step 1 is Product Research). A vision LLM fills a fixed list of slots (whole views, controls,
states, parts, details) with the best photo for each; code frames every picture, lays the sheet out and
checks it against hard gates before delivering it. In `mixed` mode (the default, chosen on the kitchen-appliance
POC) real photos carry the whole views and every label close-up, and a clean view is generated only for
a state or part that no clean photo shows.

The gates exist because the first automatic sheet showed a second colour of the product, panels whose
picture did not match the caption, parts cut in half, marketing text and food. A missing panel is
acceptable; a wrong one is not - so a sheet that fails a gate is not returned.
"""

import json
import logging
import os
import re

import numpy as np
import torch
from PIL import Image

from . import product_sheet_core as core
from .base_node import API_KEY_INPUT_SPEC, APIClientMixin, SupersideFalNode

logger = logging.getLogger(__name__)


class SupersideProductSheetNode(SupersideFalNode, APIClientMixin):
    """Fill the slots, clean, generate what is missing, compose and gate the product sheet."""

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("sheet", "product_info", "info", "scene_prompt")
    FUNCTION = "build"
    DISPLAY_NAME = "Product Sheet"
    DESCRIPTION = (
        "Step 2 of the product sheet: builds one square reference sheet from the folder made by Product "
        "Research. The whole product gets the largest panels; controls, open states and parts follow. "
        "mode = real uses only photos; mixed (default) also generates clean views of states no photo shows, "
        "captioned (GENERATED); generated makes every panel. The sheet is checked before it is returned: "
        "if the product is not the protagonist or a required view is missing, the node stops and says why. "
        "The sheet is also saved as sheet_<mode>.png in product_dir."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "product_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "The folder from Product Research."}),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "mode": (["mixed", "real", "generated"], {"default": "mixed"}),
                "max_generated": ("INT", {"default": 6, "min": 0, "max": 12,
                                          "tooltip": "Most NEW generated views (each is one GPT Image 2.5 Sunburst call at quality high)."}),
                "isolate_busy_photos": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Cut whole-product photos out of kitchens and studios (background removal, real pixels)."}),
                "size": ("INT", {"default": 5000, "min": 2000, "max": 8000, "step": 100}),
                "llm_model": ("STRING", {"default": "anthropic/claude-sonnet-4.6", "tooltip": "Fills the slots."}),
                "check_model": ("STRING", {"default": "google/gemini-2.5-flash",
                                           "tooltip": "Tags generated views and checks every finished panel against its caption."}),
                "reuse_plan": ("BOOLEAN", {"default": True, "tooltip": "Reuse the plan and the generated views saved by an earlier run of the same mode."}),
                "strict": ("BOOLEAN", {"default": True, "tooltip": "Stop instead of returning a sheet that fails a hard gate."}),
                "include_parts": ("BOOLEAN", {"default": False, "tooltip": "Also show every removable part alone. Off: fewer, larger panels."}),
                "format": (["1:1", "16:9"], {"default": "1:1", "tooltip": "Shape of the sheet. 16:9 keeps `size` as the width."}),
                "variant": (["all in one", "product", "details", "in use"], {"default": "all in one",
                            "tooltip": "Which sheet to build. product: the whole product, its counted parts and sizes. details: close-ups "
                                       "of controls, logo, texture and accessories. in use: the product being used. Connect this node three "
                                       "times to the same product_dir for the full set; the plan and generated views are shared."}),
                "spec_panel": ("BOOLEAN", {"default": True, "tooltip": "Add a panel of text with the product's counted parts, what its controls read, "
                                                                        "materials and stated sizes, and a panel with dimension lines when sizes are known."}),
            },
        }

    def build(self, product_dir, api_key, mode="mixed", max_generated=5, isolate_busy_photos=True, size=5000,
              llm_model="anthropic/claude-sonnet-4.6", check_model="google/gemini-2.5-flash", reuse_plan=True, strict=True,
              include_parts=False, format="1:1", variant="all in one", spec_panel=True):
        folder = product_dir.strip().strip('"')
        sheet_h = round(size * 9 / 16) if format == "16:9" else size
        suffix = "" if variant == "all in one" else "_" + variant.replace(" ", "_")
        if not os.path.isfile(os.path.join(folder, "image_tags.json")):
            raise ValueError(f"'{folder}' is not a researched product folder (image_tags.json is missing). Run Product Research first.")
        client = self.get_client(api_key)
        call = lambda endpoint, arguments: self.call_api(client, endpoint, arguments)   # noqa: E731
        product = json.load(open(os.path.join(folder, "product.json"), encoding="utf-8"))
        info_path = os.path.join(folder, "product_info.txt")
        info = open(info_path, encoding="utf-8").read().strip() if os.path.exists(info_path) else ""
        # what kind of product this is: read by the planner, the gates and the final QA (decided now if research did not)
        profile = core.product_profile(call, folder, product, info, llm_model)
        os.makedirs(os.path.join(folder, "generated"), exist_ok=True)

        # views generated by earlier runs are tagged once and offered to the planner, so they are not paid for twice
        subfolders = tuple(s for s in ("images", "images_new", "generated") if os.path.isdir(os.path.join(folder, s)))
        tags = core.tag_images(call, folder, product, check_model.strip() or llm_model, subfolders=subfolders)
        for key, t in tags.items():
            if key.startswith("generated/"):
                t["generated"] = True
        json.dump(tags, open(os.path.join(folder, "image_tags.json"), "w", encoding="utf-8"), indent=1)

        # what the product is made of, counted: shown on the sheet as text and given to the final QA
        spec = core.product_spec(call, folder, product, info, tags, llm_model)
        plan_path = os.path.join(folder, f"sheet_plan_{mode}.json")
        saved = json.load(open(plan_path, encoding="utf-8")) if reuse_plan and os.path.exists(plan_path) else {}
        if saved.get("slots"):
            slots, notes = saved["slots"], []
            for sl in slots:
                sl.pop("dropped", None)
        else:
            # a plan with no slot is a failed answer, not a plan: asked again once, and never saved
            for _try in range(2):
                slots, notes = core.plan_sheet(call, folder, product, tags, info, mode, 0 if mode == "real" else max_generated, llm_model,
                                               include_parts=include_parts, profile=profile)
                if slots:
                    break
            if not slots:
                raise ValueError("The planner returned no panels twice for this product. Run the node again; if it repeats, "
                                 "check that the folder has photos of the product alone.")
            json.dump({"slots": [x for x in slots if not x.get("transient")], "notes": notes}, open(plan_path, "w", encoding="utf-8"), indent=1)   # paid for: keep it

        built = []
        for i, sl in enumerate(slots):
            if sl.get("missing") or not (sl.get("image") or sl.get("generate") or sl.get("file")):
                continue
            if sl.get("image"):
                if sl["kind"] == "controls" and sl.get("crop") and not sl.get("crop_fitted") and check_model.strip():
                    sl["crop"] = core.refine_controls(call, folder, sl, f'{product.get("brand", "")} {product.get("model", "")}', check_model.strip())
                    sl["crop_fitted"] = True
                im = core.prepare_panel(call, folder, sl, tags, isolate_busy_photos)
                src = Image.open(os.path.join(folder, sl["image"]))
                if sl["kind"] in ("whole", "part") and not sl.get("crop") and tags.get(sl["image"], {}).get("background") == "white" \
                        and core.touches_edge(src):
                    sl["dropped"] = "G2: the subject is cut off in the source photo"
                    notes.append(f'{sl["caption"]}: {sl["dropped"]}')
                    continue
            else:
                sl.setdefault("file", f"generated/{mode}_{i + 1:02d}_{core.slug(sl['caption'])[:30].lower()}.png")
                path = os.path.join(folder, sl["file"])
                if os.path.exists(path):
                    im = Image.open(path).convert("RGB")
                else:
                    im = core.generate_view(call, folder, sl["generate"]["what"], sl["generate"]["refs"], sl["generate"].get("wide", False),
                                            kind=sl["kind"])
                    im.save(path)
                im = core.trim_plain(im)
            caption = sl["caption"] + (" (GENERATED)" if sl.get("generated") and mode == "mixed" else "")
            built.append({"im": im, "role": sl["role"], "kind": sl["kind"], "caption": caption, "slot": i,
                          "source": sl.get("image") or sl.get("file"), "generated": bool(sl.get("generated")),
                          **({"max_up": float(sl["max_up"])} if sl.get("max_up") else {"max_up": 2.6 if sl["kind"] in ("whole", "state", "controls", "use") else 2.0})})
        if not built:
            raise ValueError("No slot could be filled. Check that Product Research found usable photos of this product.")

        def cut_closeup(i, sl, force_cut=False):
            """A close-up that was refused or could not be filled is not generated: it is found in the largest real
            photo of the whole product, cut out and checked. Returns the panel, or None."""
            name = f'{product.get("brand", "")} {product.get("model", "")}'
            photos = core.real_whole_photos(folder, tags, [k for k in tags if not k.startswith("generated/")])[:4]
            what = sl["caption"].lower() + (": every button, switch, dial, display and indicator, with labels" if sl["kind"] == "controls" else "")
            logo = sl["caption"].startswith("LOGO")            # a logo is small: a smaller cut is accepted, with some of the body around it
            texture = sl["caption"].startswith("TEXTURE")
            if logo:
                sl["caption"] = ("LOGO - " + str(product.get("brand", "")).upper()).strip(" -")
            used = {b["source"] for b in built}
            own = core.best_of(folder, tags, "logo close-up" if logo else "controls close-up", used) \
                if (logo or sl["kind"] == "controls") and not force_cut else None
            if own:                                             # the slot's own category first: a real close-up photograph
                trial = dict(sl, image=own, crop=None)
                cand = {"im": core.prepare_panel(call, folder, trial, tags, isolate_busy_photos), "role": sl["role"], "kind": sl["kind"],
                        "slot": i, "source": own, "generated": False, "caption": sl["caption"], "max_up": 2.6}
                for k2 in ("dropped", "missing", "why", "generate", "file", "crop"):
                    sl.pop(k2, None)
                sl.update(image=own, crop=None, generated=False)
                notes.append(f'{sl["caption"]}: taken from the close-up photo {os.path.basename(own)}')
                return cand
            if logo:
                what = "the brand logo or name printed on the product itself (and the model name if it is next to it)"
            if texture:
                return find_texture(i, sl, photos, name)
            # Photo by photo: a cut the check refuses in one photo (the logo tiny in a photo of the whole set) may be right
            # in the next. A logo or detail is looked for first in photos of single pieces and other close-ups.
            pool = list(photos)
            if logo or sl["kind"] == "detail":
                near = [k for k, t2 in tags.items() if t2.get("view") in ("logo close-up", "detail close-up", "part alone")
                        and t2.get("same_product") is not False and not k.startswith("generated/") and k not in used]
                near.sort(key=lambda k: -min(Image.open(os.path.join(folder, k)).size))
                pool = near[:3] + pool
            why = ""
            for photo in pool[:5]:
                key, boxes = core.locate_in_photo(call, folder, [photo], name, what, llm_model.strip() or check_model.strip(),
                                                  min_px=100 if logo else 130, grow=0.45 if logo else 0.15,
                                                  n=1 if logo else 2,
                                                  max_share=0.2 if (logo or sl["kind"] != "controls") else 0.4,
                                                  max_cells=6 if (logo or sl["kind"] != "controls") else 14)
                if not key:
                    continue
                trial = dict(sl, image=key, crop=boxes[0], crop_fitted=True)
                trial.pop("crops", None)
                if len(boxes) > 1:
                    trial["crops"] = boxes
                cand = {"im": core.prepare_panel(call, folder, trial, tags, isolate_busy_photos), "role": sl["role"], "kind": sl["kind"],
                        "slot": i, "source": key, "generated": False, "caption": sl["caption"], "max_up": 2.6}
                ok = [cand]
                if check_model.strip():
                    ok, _bad = core.verify_panels(call, name, [cand], check_model.strip(), keep_min=0)
                if not ok:
                    why = cand.get("why", "")[:120]
                    continue
                for k2 in ("dropped", "missing", "why", "generate", "file"):
                    sl.pop(k2, None)
                sl.pop("crops", None)
                sl.update(image=key, crop=trial["crop"], crop_fitted=True, generated=False, **({"crops": trial["crops"]} if trial.get("crops") else {}))
                notes.append(f'{sl["caption"]}: cut from the real photo {os.path.basename(key)}')
                return cand
            notes.append(f'{sl["caption"]}: no cut from {len(pool[:5])} real photos passed the check' + (f" (last: {why})" if why else ""))
            if not sl.get("image") and not sl.get("generate"):
                sl["missing"] = True
            return None

        def find_texture(i, sl, photos, name):
            """A square of one bare material. Where the material is comes from a vision model (grid cells), the square is
            measured inside that area only, its colour must fit the material, and a check model must see only that
            material in it (owner, 2026-11-11: textures from several parts of the product, one per material)."""
            material = sl["caption"].split("-", 1)[-1].strip()
            where = f' on the {sl["where"].lower()}' if sl.get("where") else ""
            what = (f"the largest flat, plain area of bare {material.lower()}{where}: only the material, with no button, knob, "
                    "text, logo, edge, seam or shadow in it")
            near = [k for k, t2 in tags.items() if t2.get("view") in ("detail close-up", "interior", "part alone")
                    and t2.get("same_product") is not False and not k.startswith("generated/")]
            near.sort(key=lambda k: -min(Image.open(os.path.join(folder, k)).size))
            why, good = "", []
            for photo in (list(photos[:3]) + near[:3])[:5]:
                if len(good) >= 3:
                    break
                key, boxes = core.locate_in_photo(call, folder, [photo], name, what, llm_model.strip() or check_model.strip(),
                                                  min_px=60, grow=0.0, n=1, max_share=0.45, max_cells=6)
                if not key:
                    continue
                im = Image.open(os.path.join(folder, key)).convert("RGB")
                for sq in core.texture_squares(im, boxes[0], share=0.34, k=4, strict=True):   # the largest square first
                    if len(good) >= 3:
                        break
                    patch = im.crop((int(sq[0] * im.width), int(sq[1] * im.height), int(sq[2] * im.width), int(sq[3] * im.height)))
                    q_ok, side_px, sharp = core.texture_quality(patch)
                    if not q_ok:                                  # too few real pixels, or a blurred surface: not a close-up
                        why = f"{side_px} px, sharpness {sharp}: too small or too soft for a texture close-up"
                        continue
                    if not core.colour_fits(patch, material):
                        why = "colour does not fit the material"
                        continue
                    if check_model.strip():
                        ok, why = core.texture_ok(call, check_model.strip(), patch, material, name)
                        if not ok:
                            continue
                    good.append((key, sq, patch))
            if not good:
                notes.append(f'{sl["caption"]}: no patch of only this material, sharp and large enough, was found' + (f" (last: {why})" if why else ""))
                sl["missing"] = True
                return None
            # several crops of the same material side by side (owner, 2026-11-11), at their own resolution
            for k2 in ("dropped", "missing", "why", "generate", "file", "crops"):
                sl.pop(k2, None)
            sl.update(image=good[0][0], crop=good[0][1], crop_fitted=True, generated=False, transient=True)
            notes.append(f'{sl["caption"]}: {len(good)} crops measured in real photos and checked')
            return {"im": core.texture_tile([g[2] for g in good]), "role": sl["role"], "kind": sl["kind"], "slot": i,
                    "source": good[0][0], "generated": False, "caption": sl["caption"], "max_up": 3.0}

        # G5: a second model looks at every finished panel; one that does not show what its caption says is removed
        if check_model.strip():
            kept, dropped = core.verify_panels(call, f'{product.get("brand", "")} {product.get("model", "")}', built, check_model.strip())
            kept_ids = {id(b) for b in kept}
            for b in built:
                if id(b) not in kept_ids:
                    slots[b["slot"]]["dropped"] = "G5: the picture does not match the caption"
                    slots[b["slot"]]["why"] = b.get("why", "")
            notes += [f"removed by the panel check - {d}" for d in dropped]
            built = kept

            # QA retry: a panel the check removed is not simply left out. It is made again once, from the product's own
            # whole views, with the reason it was refused, and checked again. Same for a required panel nobody could fill.
            name = f'{product.get("brand", "")} {product.get("model", "")}'
            whole_refs = [b["source"] for b in built if b["kind"] == "whole" and not b["generated"]][:2]
            retried = 0
            for i, sl in enumerate(slots):
                again = str(sl.get("dropped", "")).startswith("G5") or (sl.get("missing") and sl["kind"] in ("controls", "use", "detail"))
                if not again or retried >= 4 or mode == "real" or not whole_refs:
                    continue
                if sl["kind"] in ("controls", "detail"):
                    retried += 1
                    cand = cut_closeup(i, sl)
                    if cand:
                        built.append(cand)
                    continue
                gen = sl.get("generate") or {}
                refs = list(dict.fromkeys(list(gen.get("refs", [])) + whole_refs))[:3]
                what = gen.get("what") or (f'{sl["caption"].lower()} - one hand holding the product in use' if sl["kind"] == "use"
                                           else f'{sl["caption"].lower()}, alone and complete')
                file = f"generated/{mode}_{i + 1:02d}_{core.slug(sl['caption'])[:30].lower()}_retry.png"
                path = os.path.join(folder, file)
                if os.path.exists(path):
                    im = Image.open(path).convert("RGB")
                else:
                    try:
                        im = core.generate_view(call, folder, what, refs, False, kind=sl["kind"], feedback=sl.get("why", ""))
                    except Exception as e:
                        notes.append(f'{sl["caption"]}: second attempt failed to generate ({str(e)[:80]})')
                        continue
                    im.save(path)
                retried += 1
                cand = {"im": core.trim_plain(im), "role": sl["role"], "kind": sl["kind"], "slot": i, "source": file, "generated": True,
                        "caption": sl["caption"] + (" (GENERATED)" if mode == "mixed" else ""),
                        "max_up": 2.6 if sl["kind"] in ("whole", "state", "controls", "use") else 2.0}
                ok, _bad = core.verify_panels(call, name, [cand], check_model.strip(), keep_min=0)
                if ok:
                    for key in ("dropped", "missing", "image", "crop", "why"):
                        sl.pop(key, None)
                    sl.update(generated=True, file=file, generate={"what": what, "refs": refs, "wide": False})
                    built.append(cand)
                    notes.append(f'{sl["caption"]}: made again after the check refused the first picture')
                else:
                    notes.append(f'{sl["caption"]}: second attempt also refused ({cand.get("why", "")[:120]})')

        # G7: the first control close-up gets a row of its own, so its labels are large

        # Two close-ups are part of every sheet whatever the planner did: the controls and the logo. If none was
        # planned or none survived, they are looked for in the largest real photo of the whole product and cut from it.
        if built:
            for kind, caption, has in (("controls", "CONTROLS - BUTTONS & SWITCHES",
                                        any(b["kind"] == "controls" for b in built) or not core.applies(profile, "has_controls")),
                                       ("detail", "LOGO & MODEL NAME", any(re.search(r"LOGO|BADGE|BRAND", b["caption"]) for b in built))):
                if has:
                    continue
                sl = next((s for s in slots if (s["kind"] == "controls" if kind == "controls" else s["caption"].startswith("LOGO"))), None)
                if sl is None:
                    sl = {"kind": kind, "caption": caption, "role": core.KIND_ROLE[kind], "missing": True}
                    slots.append(sl)
                cand = cut_closeup(slots.index(sl), sl)
                if cand:
                    built.append(cand)
                elif not sl.get("dropped"):
                    sl["missing"] = True

        # Every panel shows something no other panel shows (owner's rule, 2026-11-11).
        # 1. The product alone is shown once per angle.
        # 2. Accessories get a panel of their own, WITHOUT the product in it again: a photo of the product with its
        #    accessories is dropped when the accessories already have a panel; otherwise the accessories are cut out of
        #    it. If they cannot be cut apart, the photo stays and the product-alone view from the same angle goes.
        # Not for a product that IS a set (no view of one piece alone exists): there the set is the product.
        def angle(b):
            t = tags.get(b["source"], {})
            return t.get("angle") if t.get("angle") not in (None, "", "none") else (t.get("view") if t.get("view") in core.WHOLE_VIEWS else "")

        def is_set(b):
            return b["kind"] in ("whole", "part") and not b["generated"] and tags.get(b["source"], {}).get("view") == "set together" \
                and not slots[b["slot"]].get("crop")

        def drop(b, why):
            slots[b["slot"]]["dropped"] = "REPEAT: " + why
            notes.append(f'{b["caption"]}: dropped - {why}')
            built.remove(b)

        seen_angles = {}
        for b in [x for x in built if x["kind"] == "whole" and not is_set(x)]:
            if angle(b) and angle(b) in seen_angles:
                drop(b, f'same angle ({angle(b)}) as "{seen_angles[angle(b)]}"')
            elif angle(b):
                seen_angles[angle(b)] = b["caption"]
        for b in [x for x in built if is_set(x)]:
            alone = [x for x in built if x["kind"] == "whole" and not is_set(x)]
            if not alone:
                break
            if any(x["kind"] == "part" and x is not b and not is_set(x) for x in built):
                drop(b, "the product is already shown alone and the accessories have their own panel")
                continue
            key, boxes = core.locate_in_photo(call, folder, [b["source"]], f'{product.get("brand", "")} {product.get("model", "")}',
                                              "all the accessories and removable parts lying next to the main product - NOT the main product itself",
                                              llm_model.strip() or check_model.strip(), min_px=300, grow=0.12, max_share=0.75, n=1, max_cells=30)
            if key:
                sl = {"kind": "part", "caption": "ACCESSORIES", "role": core.KIND_ROLE["part"], "image": key, "crop": boxes[0],
                      "crop_fitted": True, "generated": False}
                slots.append(sl)
                built.append({"im": core.trim_plain(core.prepare_panel(call, folder, sl, tags, isolate_busy_photos)), "role": sl["role"],
                              "kind": "part", "slot": len(slots) - 1, "source": key, "generated": False, "caption": "ACCESSORIES", "max_up": 2.0})
                drop(b, "the product is already shown alone; its accessories were cut from this photo into their own panel")
            else:
                same = [x for x in alone if angle(x) and angle(x) == angle(b)]
                if same and len(alone) > 1:
                    drop(same[-1], f'the product is seen from the same angle in "{b["caption"]}", which also shows the accessories')

        # Colour layer (owner, 2026-11-11: "one of them is a different colour, we need a colour layer"). Every whole view
        # is measured once it sits on white. One whose finish is far from the product's finish - a black-stainless oven
        # next to a brushed-steel one - is replaced by another real photo of the right finish, or dropped.
        ref_tone = core.reference_tone(folder, tags)
        wholes = [b for b in built if b["kind"] == "whole" and not b["generated"]]
        if ref_tone is None and wholes:
            ref_tone = core.tone_of(wholes[0]["im"])
        if ref_tone is not None:
            used = {b["source"] for b in built}
            for b in wholes:
                tone = core.tone_of(b["im"])
                # tone None = the panel still has no white background after preparation (a dark product cut badly from a
                # dark room): it breaks the always-on-white rule too, so it is treated the same way
                if tone is not None and abs(tone - ref_tone) < 34:
                    continue
                kept_angles = {angle(x) for x in built if x["kind"] == "whole" and x is not b}
                swap = None
                for k in core.real_whole_photos(folder, tags, [k for k in tags if k not in used]):
                    t = tags[k]
                    a = t.get("angle") if t.get("angle") not in (None, "", "none") else t.get("view")
                    if t.get("view") == "set together" or a in kept_angles:
                        continue
                    trial = {"kind": "whole", "image": k, "caption": b["caption"]}
                    im = core.prepare_panel(call, folder, trial, tags, isolate_busy_photos)
                    tt = core.tone_of(im)
                    if tt is not None and abs(tt - ref_tone) < 34:
                        swap = (k, im)
                        break
                sl = slots[b["slot"]]
                if swap:
                    sl.update(image=swap[0], crop=None, generated=False)
                    b.update(im=swap[1], source=swap[0])
                    used.add(swap[0])
                    notes.append(f'{b["caption"]}: the photo was another finish of the product, or could not be put on white; replaced by {os.path.basename(swap[0])}')
                elif len([x for x in built if x["kind"] == "whole"]) > 1:
                    sl["dropped"] = "COLOUR: another finish of the product"
                    notes.append(f'{b["caption"]}: dropped - the photo is another finish of the product and no photo of the right finish from another angle was found')
                    built.remove(b)

        # No view twice: a panel that repeats one already on the sheet is dropped (the product whole is kept, the loose
        # cut of it goes). A required close-up lost this way is cut again, tight, before the sheet is composed.
        for _round in range(2):
            built, gone = core.drop_repeats(built, slots)
            for b, why in gone:
                slots[b["slot"]]["dropped"] = "REPEAT: " + why
                notes.append(f'{b["caption"]}: dropped - {why}')
            again = [b for b, _why in gone if b["kind"] == "controls" or b["caption"].startswith("LOGO")]
            if _round or not again:
                break
            for b in again:
                if not any(k["kind"] == b["kind"] and k["caption"].startswith("LOGO") == b["caption"].startswith("LOGO") for k in built):
                    cand = cut_closeup(b["slot"], slots[b["slot"]], force_cut=True)
                    if cand:
                        built.append(cand)

        # Which sheet: all in one, or one of three - the product, its details, the product in use. All three are built from
        # the same plan and the same panels, so they agree with each other; each only draws its own part, larger.
        if variant == "details" and spec and (spec.get("materials") or []):
            # one swatch per material the product is made of (brushed steel, black plastic, glass ...), up to three: a
            # model needs each surface, not only the main one. Each is tried once per plan.
            seen_m = set()
            for entry in spec["materials"]:
                material, part = core.material_name(entry)
                key_m = frozenset(w for w in material.lower().split() if len(w) > 3)
                if not material or any(len(key_m & k) >= max(1, min(len(key_m), len(k)) - 1) for k in seen_m):
                    continue
                seen_m.add(key_m)
                if len(seen_m) > 3:
                    break
                cap = "TEXTURE - " + material
                if any(b["caption"] == cap for b in built) or any(str(x.get("caption", "")) == cap for x in slots):
                    continue
                tex = {"kind": "detail", "caption": cap, "role": "detail", "where": part[:60]}
                slots.append(tex)
                cand = cut_closeup(len(slots) - 1, tex, force_cut=True)
                if cand:
                    built.append(cand)
        for b in built:                                  # texture crops are shown at their own resolution, at most 2x
            if b["caption"].startswith("TEXTURE"):
                b["flex"], b["max_up"] = False, 3.0
        if variant == "product":
            built = [b for b in built if b["kind"] in ("whole", "state")]
        elif variant == "details":
            built = [b for b in built if b["kind"] in ("controls", "detail", "part")]
        elif variant == "in use":
            # in-use photos keep their scene: the room, the hands and the food are what this sheet is for. Only photos
            # of this product (its own model and finish) that carry no added text.
            used = {b["source"] for b in built}
            more = [k for k, t in tags.items() if t.get("view") in ("in hand", "in use") and t.get("same_product") is not False
                    and t.get("banner", "none") in ("none", None) and t.get("background") not in ("graphic", "white") and k not in used
                    and not k.startswith("generated/")]
            more.sort(key=lambda k: -min(Image.open(os.path.join(folder, k)).size))
            built = [b for b in built if b["kind"] == "use"]
            for k in more[:max(0, 4 - len(built))]:
                try:
                    im = Image.open(os.path.join(folder, k)).convert("RGB")
                except Exception:
                    continue
                built.append({"im": im, "role": "view", "kind": "use", "slot": -1, "source": k, "generated": False, "flex": True,
                              "flex_crop": 0.3, "caption": core.short_caption(tags[k].get("desc", "")), "max_up": 3.0})
            # owner, 2026-11-11: at least three ways of using it. What the photos do not show is generated from the
            # product's own whole views, one distinct use each, and checked like every generated panel.
            refs = core.real_whole_photos(folder, tags, [k for k in tags if not k.startswith("generated/")])[:2]
            short = core.MIN_USES - len(built)
            if short > 0 and mode != "real" and refs:
                ideas_path = os.path.join(folder, "use_cases.json")
                ideas = json.load(open(ideas_path, encoding="utf-8")) if os.path.exists(ideas_path) else []
                if len(ideas) < short:
                    try:
                        ideas += core.use_ideas(call, llm_model.strip() or check_model.strip(), product, info,
                                                [b["caption"] for b in built] + [u.get("caption", "") for u in ideas], short - len(ideas))
                        json.dump(ideas, open(ideas_path, "w", encoding="utf-8"), indent=1)
                    except Exception as e:
                        notes.append(f"in use: could not list more uses ({str(e)[:80]})")
                mats = "; ".join(str(m) for m in ((spec or {}).get("materials") or [])[:5])
                finish = (" The product is exactly the one in the reference photos, same materials, colours and finish inside and"
                          " out" + (f" ({mats})" if mats else "") + ": nothing is recoloured or given a coating it does not have.")
                name = f'{product.get("brand", "")} {product.get("model", "")}' + (f" - made of {mats}" if mats else "")
                os.makedirs(os.path.join(folder, "generated"), exist_ok=True)
                for u in ideas:
                    if len(built) >= core.MIN_USES:
                        break
                    cap = str(u.get("caption") or "IN USE").upper()[:30]
                    file = f"generated/use_{core.slug(cap)[:30].lower()}.png"
                    path = os.path.join(folder, file)
                    try:
                        im = Image.open(path).convert("RGB") if os.path.exists(path) else core.generate_view(
                            call, folder, u["what"] + finish, refs, True, kind="use")
                    except Exception as e:
                        notes.append(f"in use: {cap} failed to generate ({str(e)[:80]})")
                        continue
                    im.save(path)
                    cand = {"im": im, "role": "view", "kind": "use", "slot": -1, "source": file, "generated": True, "flex": True,
                            "flex_crop": 0.3, "caption": cap + (" (GENERATED)" if mode == "mixed" else ""), "max_up": 3.0}
                    if check_model.strip():
                        ok, _bad = core.verify_panels(call, name, [cand], check_model.strip(), keep_min=0)
                        if not ok:
                            notes.append(f"in use: {cap} was generated but refused by the check")
                            os.replace(path, path[:-4] + "_refused.png")
                            continue
                    built.append(cand)
        if not built:
            raise ValueError(f'Nothing to show on the "{variant}" sheet: Product Research found no photos for it.')

        title = f'{product.get("brand", "")} {product.get("model", "")} - {product.get("name", "")}'.strip(" -").upper()
        if variant != "all in one":
            title += " - " + variant.upper()
        size_line = next(((ln.split(":", 1)[1] if ":" in ln else ln[4:]).strip() for ln in info.splitlines() if ln.upper().startswith("SIZE")), "")
        subtitle = (size_line + " " if size_line else "") + "Close-ups are enlarged crops of the same product: use them for detail, not for size."
        mf = core.min_font_for(size, sheet_h)
        text_done = {}

        def compose_best(panels):
            """The collage: rows, or the first panel large with the others stacked beside it - each is drawn and the one
            with the most picture (least white) that keeps the whole views large enough is kept (owner's rule: no wasted
            white). Then the text is written into the white that is left, never in a panel of its own."""
            best = None
            for arr in core.arrangements(panels):
                for hw in (None, 3.4, 4.4, 6.0, 9.0, 14.0):            # G1: give the whole views more height until they own 40 % of the sheet
                    trial, trows = core.compose(arr, title, subtitle[:230], size=size, hero_weight=hw, height=sheet_h)
                    trep = core.gate_report(slots, panels, trows, size, profile, height=sheet_h, variant=variant)
                    if not any(f.startswith("G1: whole-product panels") for f in trep["fatal"]):
                        break
                score = core.sheet_fill(trial) - (0.5 if any(f.startswith("G1:") for f in trep["fatal"]) else 0)
                if best is None or score > best[0]:
                    best = (score, trial, trows, trep)
            _s, sh, rw, rp = best
            if spec_panel and spec:
                text_done.update(core.place_text(sh, rw, spec, variant, mf))
            return sh, rw, rp

        sheet, rows, report = compose_best(built)
        # final QA: one model looks at the finished sheet as a whole against the hierarchy. What it can fix by removing
        # a bad secondary panel is fixed and the sheet is judged once more; what remains is reported and blocks a strict run.
        qa, rounds = {"pass": True, "issues": [], "not_run": "no check model"}, []
        if llm_model.strip():
            name = f'{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")}'
            for qa_round in range(2):
                qa = core.qa_sheet(call, name, sheet, [r["caption"] for r in rows], llm_model.strip(), profile,
                                   composition=str((spec or {}).get("summary") or ""), variant=variant)
                rounds.append(qa)
                removable = {"artefact", "marketing_text", "wrong_caption", "duplicate", "product_cut"}
                first_whole = next((b for b in built if b["kind"] == "whole"), None)
                # the final QA deletes generated pictures only. A real photograph or a cut from one is reported, never
                # deleted: two checks that disagree from run to run must not decide what the sheet contains
                # a controls or logo panel that the QA says is not really a close-up (the whole product again, the wrong
                # thing) is cut again from a real photo of the product alone - reporting it is not enough for a required panel
                recut = [b for b in built if (b["kind"] == "controls" or re.search(r"LOGO|BADGE|BRAND", b["caption"])) and not b.get("recut")
                         and any(str(i.get("panel", "")).upper()[:24] == b["caption"].upper()[:24]
                                 and i.get("code") in ("duplicate", "wrong_caption", "product_unclear", "no_controls_closeup") for i in qa["issues"])]
                if not qa_round and recut:
                    changed = False
                    for b in recut:
                        cand = cut_closeup(b["slot"], slots[b["slot"]], force_cut=True)
                        if cand:
                            cand["recut"] = True
                            built[built.index(b)] = cand
                            notes.append(f'{b["caption"]}: the final QA said it was not a close-up; cut again from a real photo')
                            changed = True
                    if changed:
                        built, gone = core.drop_repeats(built, slots)          # the new cut may repeat a panel already there
                        for g, why in gone:
                            slots[g["slot"]]["dropped"] = "REPEAT: " + why
                            notes.append(f'{g["caption"]}: dropped - {why}')
                        sheet, rows, report = compose_best(built)
                        continue
                bad = [b for b in built if b is not first_whole and b["generated"]
                       and sum(1 for x in built if x["kind"] == b["kind"]) > 1 and any(
                    str(i.get("panel", "")).upper()[:24] == b["caption"].upper()[:24] and i.get("code") in removable for i in qa["issues"])]
                if qa_round or not bad or len(built) - len(bad) < 3:
                    break
                for b in bad:
                    slots[b["slot"]]["dropped"] = "QA: removed by the final sheet check"
                    notes.append(f'removed by the final QA - {b["caption"]}')
                built = [b for b in built if b not in bad]
                for b in bad:                               # a refused generated close-up is replaced by a cut from a real photo
                    if b["kind"] in ("controls", "detail") and b["generated"]:
                        cand = cut_closeup(b["slot"], slots[b["slot"]])
                        if cand:
                            built.append(cand)
                sheet, rows, report = compose_best(built)
            for i in qa["issues"]:
                report["fatal"].append(f'QA {i.get("code")}: {i.get("panel", "SHEET")} - {str(i.get("why", ""))[:140]}')
        report["qa"] = qa

        # Legibility: what an image model sees is this sheet scaled to about 1248 px (1664 x 928 in 16:9). The texts we
        # draw are sized for that by construction; this checks the texts printed on the product. When the controls are
        # not readable, the controls panel is given more room once and the sheet is checked again.
        legible = {"not_run": "no check model"}
        if check_model.strip() and spec:
            caps = [r["caption"] for r in rows]
            legible = core.legibility(call, sheet, spec, check_model.strip(), caps)
            labels = legible.get("labels", [])
            ok = sum(1 for x in labels if x["ok"])
            controls = [b for b in built if b["kind"] == "controls"]
            if labels and ok < 0.6 * len(labels) and controls:
                for b in controls:
                    b["weight"] = 5.0
                sheet, rows, report = compose_best(built)
                report["qa"] = qa
                again = core.legibility(call, sheet, spec, check_model.strip(), [r["caption"] for r in rows])
                again["first_try"] = f"{ok}/{len(labels)} labels readable; the control panel was enlarged"
                legible = again
                labels = legible.get("labels", [])
                ok = sum(1 for x in labels if x["ok"])
            if labels:
                legible["labels_readable"] = f"{ok}/{len(labels)}"
                if ok < 0.6 * len(labels) and controls and variant in ("all in one", "details"):
                    report["fatal"].append(f"L2: at the size an image model sees ({legible['model_view'][0]} px), only {ok} of "
                                           f"{len(labels)} labels printed on the product can be read")
            bad_caps = [x["name"] for x in legible.get("captions", []) if not x["ok"]]
            if bad_caps:
                legible["captions_unreadable"] = bad_caps
        report["legibility"] = legible
        # how much of the sheet is picture, not white (owner's rule: organise the collage, do not leave white)
        report["fill"] = core.sheet_fill(sheet)
        report["text"] = dict(text_done)
        if variant == "in use" and sum(1 for b in built if b["kind"] == "use") < core.MIN_USES:
            report["fatal"].append(f'U1: the in-use sheet shows {sum(1 for b in built if b["kind"] == "use")} uses; at least {core.MIN_USES} are required')
        if report["fill"] < 0.40:
            report["fatal"].append(f'G2: only {int(report["fill"] * 100)} % of the sheet is picture (the hand-approved sheets are 55-58 %)')
        report["qa_first_round"] = rounds[0]["issues"] if len(rounds) > 1 else []
        json.dump({"slots": [x for x in slots if not x.get("transient")], "notes": notes}, open(plan_path, "w", encoding="utf-8"), indent=1)
        out = {"mode": mode, "variant": variant, "size": [size, sheet_h], "profile": profile, "gates": report, "notes": notes,
               "panels": rows, "saved": os.path.join(folder, f"sheet_{mode}{suffix}.png")}
        json.dump(out, open(os.path.join(folder, f"sheet_{mode}{suffix}_info.json"), "w", encoding="utf-8"), indent=1)
        if report["fatal"] and strict:
            sheet.save(os.path.join(folder, f"sheet_{mode}{suffix}_REJECTED.png"))
            raise ValueError("The sheet failed a hard gate and was not delivered: " + "; ".join(report["fatal"])
                             + f". A copy for inspection is in sheet_{mode}{suffix}_REJECTED.png. Notes: " + " | ".join(notes[:6]))
        sheet.save(os.path.join(folder, f"sheet_{mode}{suffix}.png"))
        tensor = torch.from_numpy(np.array(sheet).astype(np.float32) / 255.0).unsqueeze(0)
        scene = core.read_scene(call, folder, product, info, llm_model)        # written by Product Research; made here if missing
        return (tensor, info, json.dumps(out, indent=1), scene)
