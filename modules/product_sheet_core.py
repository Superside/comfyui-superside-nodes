"""Shared logic for Product Research and Product Sheet.

Two steps, one folder between them:

  Product Research  ->  <product_dir>/  ->  Product Sheet
     identify, search, download,            pick panels, isolate, generate the
     tag, write the info text               missing views, compose 5000 x 5000

Everything here is plain Python + PIL. fal is reached only through `call`, a function
`call(endpoint, arguments) -> dict` that the node supplies (so retries and the cost ledger stay in one
place). What the tests behind these rules showed (kitchen-appliance POC, 2026-11):

- A vision LLM chooses WHAT to show well and crops badly (5-10 % off on a tenths grid, with or without a
  second review pass, Claude or Gemini). So the LLM names images and roles; boxes come from code: plain
  backgrounds are trimmed to the product, busy ones are cut out with background removal.
- Row layout by the LLM wasted half the sheet. Layout is computed here: whole views get the tallest rows.
- A generated view keeps shape and parts and redraws small printed labels wrong (oven dials). Generated
  panels are captioned "(GENERATED)" and a label close-up always comes from a real photo.
- Some shops refuse automated clients. A page that answers with an error or a challenge is skipped, never
  worked around.
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import urllib.parse
import urllib.request

from PIL import Image, ImageChops, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

LLM_TEXT = "openrouter/router"
LLM_VISION = "openrouter/router/vision"
BG_REMOVE = "fal-ai/bria/background/remove"
IMAGE_EDIT = "openai/gpt-image-2.5/sunburst/edit"
SEARCH_MODEL = "google/gemini-2.5-flash:online"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
ROLES = ("hero", "view", "detail")
ROLE_WEIGHT = {"hero": 2.6, "view": 1.5, "detail": 1.0}


# ----------------------------------------------------------------------------- small helpers

def data_url(im, side=None, quality=85):
    im = im.convert("RGB")
    if side:
        im = im.copy()
        im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json(text):
    m = re.search(r"\{.*\}|\[.*\]", text or "", re.S)
    if not m:
        raise ValueError("the model did not answer with JSON")
    return json.loads(m.group(0))


def llm(call, model, prompt, images=None, system=None, max_tokens=4000):
    args = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.1}
    if system:
        args["system_prompt"] = system
    if "gemini-2.5-pro" in model:                 # this model refuses calls without reasoning
        args.update(reasoning=True, max_tokens=max(max_tokens, 12000))
    if images:
        args["image_urls"] = images
    return (call(LLM_VISION if images else LLM_TEXT, args) or {}).get("output", "")


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_") or "product"


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)) if os.path.isdir(folder) else []


def fetch(url, timeout=40, referer=None):
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en"}
    if referer:
        headers["Referer"] = referer
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout).read()


def font(size, bold=True):
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ()) + ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ----------------------------------------------------------------------------- research

IDENTIFY = """Identify the product in the photo{hint}.
Answer one JSON object:
{{"brand": "", "model": "exact model / reference number if you can read or recognise it, else \\"\\"",
  "name": "what the maker calls it", "colour": "colour / finish of this unit, e.g. brushed stainless steel", "category": "short product category, e.g. countertop oven, drip coffee maker, cookware set",
  "confidence": "high | medium | low", "alternatives": ["other models of the same maker that look the same in this photo"],
  "search": "the best web search query to find this exact product"}}
Makers sell near-identical models under different numbers. Say "high" only if the model number is printed in the photo or
given by the user; if you recognise the product line but cannot tell the models apart, say "low" and list them."""

SEARCH = """Find up to {n} web pages that show product photos of this exact product: {query}.
Prefer the maker's regional sites, large retailers and hands-on reviews with their own photos. One page per site.
Answer with a JSON list of URLs only."""

TAG = """PRODUCT: {name} ({category}), colour / finish: {colour}
You get {n} photos collected for this product, in this order: {names}
For EACH photo answer one object:
- "image": the file name exactly as given
- "view": "front", "3/4 left", "3/4 right", "side", "top", "interior", "part alone", "detail close-up", "in use",
  "set together", "infographic" or "not the product"
- "state": what is open, pulled out, removed or switched on, in a few words; "" if nothing
- "parts": the parts clearly visible, in plain words
- "desc": one plain sentence, at most 14 words
- "tags": 5 to 8 short lowercase tags (view, state, parts, "knobs", "labels readable", ...)
- "background": "white", "studio", "kitchen", "graphic" or "other"
- "banner": where added marketing text sits: "none", "bottom", "top", "left", "right" or "all over"
- "clean": true only if there is no food, no props, no hands, no added text or badges
- "same_product": false if the photo shows another colour or finish than the one named above, another model, or only
  an accessory sold separately; true otherwise
- "use": "hero" (clean whole product), "view" (whole product, other angle or open state), "detail" (part or close-up),
  "reference" (shows how it works, not clean) or "skip" (food only, people, charts, other products, other colours)
Say only what is clearly visible. Answer with one JSON object {{"images": [...]}} and nothing else."""

INFO = """Write the PRODUCT INFO text for {name}. It will be pasted into image and video prompts, so it must let a
model draw the product correctly without seeing it. Use the photos (they are the truth) and these notes:
{notes}

Format: plain text, one line per topic, each starting with a label in capitals, in this order when they apply:
PRODUCT (what it is, materials, colours) / SIZE (only if the notes give it) / FRONT (controls left to right, every
readable label spelled exactly) / one line per dial or button group with its labels / SIDES, TOP / INSIDE /
REMOVABLE PARTS / HOW PARTS SIT / KEEP EXACT (what must not change, and what the product does NOT have).
Write only what you can see or the notes say. If a label cannot be read, say "label not readable" - never guess.
Describe this unit only ({colour}); ignore other colours or models that appear in the notes.
No markdown, no bullets, at most 2200 characters."""


def identify(call, image, hint, model):
    hint_txt = f" (the user says: {hint})" if hint else ""
    return parse_json(llm(call, model, IDENTIFY.format(hint=hint_txt), [data_url(image, 1024)], max_tokens=600))


def search_pages(call, query, n=8):
    try:
        urls = parse_json(llm(call, SEARCH_MODEL, SEARCH.format(n=n, query=query), max_tokens=1500))
    except Exception as e:
        logger.warning("search failed: %s", e)
        return []
    out = [u for u in urls if isinstance(u, str) and u.startswith("http")]
    return out[:n]


def model_tokens(model):
    """'TOA-70NAS' -> {'toa70nas', 'toa-70nas', 'toa70', 'toa-70'}: what a file of this product is named after."""
    m = (model or "").lower()
    core = re.match(r"[a-z]*-?\d+", m)
    toks = {m, m.replace("-", ""), m.replace("-", "_")}
    if core:
        toks |= {core.group(0), core.group(0).replace("-", "")}
    return {t for t in toks if len(t) >= 4}


def page_images(url, tokens):
    """Image URLs on a page that are named after the product. Blocked or failing pages give []."""
    try:
        html = fetch(url).decode("utf-8", "ignore")
    except Exception as e:
        logger.info("skipped %s (%s)", url, str(e)[:80])
        return [], ""
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    refused = re.search(r"access (to this page )?(has been )?(denied|restricted)|unauthorized ai agent|are you a robot|verify you are human",
                        (title.group(1) if title else "") + html[:3000], re.I)
    if refused and len(html) < 400000 and not re.search(r"add to cart|add-to-cart", html, re.I):
        logger.info("skipped %s (the site refuses automated clients)", url)
        return [], ""
    html = html.replace("\\/", "/").replace("&amp;", "&")
    found = re.findall(r"https?://[^\"'\s\\)<>]+?\.(?:jpe?g|png|webp)(?:\?[^\"'\s\\)<>]*)?", html, re.I)
    keep = []
    for u in found:
        low = urllib.parse.unquote(u).lower()
        if any(t in low for t in tokens) and not re.search(r"logo|icon|sprite|swatch|badge|thumb", low):
            keep.append(u)
    text = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html, flags=re.S)
    return list(dict.fromkeys(keep)), re.sub(r"\s+", " ", text)[:6000]


def best_variant(url):
    """The largest version of an image URL: drop resize parameters and CDN resize prefixes."""
    u = url.split("?")[0] if re.search(r"[?&](sw|sh|w|h|width|height|wid|hei)=", url) else url
    return re.sub(r"/dw/image/v2/[A-Z_]+/on/", "/on/", u)


def download_images(urls, out_dir, max_images, min_side=700, referer=None):
    os.makedirs(out_dir, exist_ok=True)
    seen, saved = set(), []
    for u in urls:
        if len(saved) >= max_images:
            break
        for cand in dict.fromkeys((best_variant(u), u)):
            try:
                raw = fetch(cand, timeout=60, referer=referer)
                im = Image.open(io.BytesIO(raw))
                im.load()
            except Exception:
                continue
            h = hashlib.md5(raw).hexdigest()
            look = hashlib.md5(im.convert("L").resize((24, 24)).point(lambda v: v // 24).tobytes()).hexdigest()   # same picture, any size
            if h in seen or look in seen or max(im.size) < min_side:
                break
            seen.update((h, look))
            host = re.sub(r"^www\.", "", urllib.parse.urlparse(cand).netloc).split(".")[0]
            parsed = urllib.parse.urlparse(cand)
            inner = urllib.parse.parse_qs(parsed.query).get("image", [parsed.path])[0]
            name = f"{host}_{slug(os.path.splitext(os.path.basename(inner))[0])[:60]}_{h[:6]}.jpg"
            im.convert("RGB").save(os.path.join(out_dir, name), "JPEG", quality=95)
            saved.append(name)
            break
    return saved


def tag_images(call, folder, product, model, subfolders=("images",), batch=10):
    path = os.path.join(folder, "image_tags.json")
    tags = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    todo = [f"{s}/{f}" for s in subfolders for f in list_images(os.path.join(folder, s)) if f"{s}/{f}" not in tags]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        names = ", ".join(f"{k + 1}) {os.path.basename(b)}" for k, b in enumerate(part))
        try:
            got = parse_json(llm(call, model, TAG.format(name=product.get("name", ""), category=product.get("category", ""),
                                                           colour=product.get("colour") or "as in most photos", n=len(part), names=names),
                                 [data_url(Image.open(os.path.join(folder, b)), 768, 80) for b in part], max_tokens=6000))["images"]
        except Exception as e:
            logger.warning("tagging batch failed: %s", e)
            continue
        by_name = {g.get("image"): g for g in got}
        for k, b in enumerate(part):
            g = by_name.get(os.path.basename(b)) or (got[k] if k < len(got) else None)
            if g:
                g.pop("image", None)
                if g.get("same_product") is False:
                    g["use"] = "skip"
                tags[b] = g
        json.dump(tags, open(path, "w", encoding="utf-8"), indent=1)
    return tags


def write_info(call, folder, product, tags, notes, model):
    order = {"hero": 0, "detail": 1, "view": 2, "reference": 3}
    readable = lambda k: not any("label" in t or "readable" in t for t in tags[k].get("tags", []))   # noqa: E731
    picks = sorted((k for k, v in tags.items() if v.get("use") in order), key=lambda k: (readable(k), order[tags[k]["use"]], not tags[k].get("clean")))[:8]
    images = [data_url(Image.open(os.path.join(folder, k)), 1536, 88) for k in picks]
    text = llm(call, model, INFO.format(name=f'{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")}'.strip(),
                                        notes=notes or "(no notes)", colour=product.get("colour") or "the colour in the photos"),
                images, max_tokens=2500).strip()
    open(os.path.join(folder, "product_info.txt"), "w", encoding="utf-8").write(text + "\n")
    return text


def contact_sheet(folder, rel_paths, cell=360, cols=6):
    rel_paths = rel_paths[:cols * 14]
    rows = max(1, -(-len(rel_paths) // cols))
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 30)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    f = font(15, bold=False)
    for i, rel in enumerate(rel_paths):
        im = Image.open(os.path.join(folder, rel)).convert("RGB")
        im.thumbnail((cell - 8, cell - 8))
        x, y = (i % cols) * cell, (i // cols) * (cell + 30)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        d.text((x + 6, y + cell + 6), f"{i + 1}. {os.path.basename(rel)[:40]}", fill=(230, 230, 230), font=f)
    return sheet


# ----------------------------------------------------------------------------- sheet: plan (slots, not free planning)

PLAN = """You fill the slots of a product reference sheet: one square image made of panels, which an image or video
model will use as its only reference of the product. PRODUCT: {name} ({category}).

PRODUCT INFO
{info}

AVAILABLE PHOTOS (the picture shows them numbered in this same order; "generated" = made earlier by an image model):
{listing}

MODE: {mode}
- "real": only photographs. - "mixed": photographs for whole views, controls and labels; a generated view only where no
  clean photograph exists. - "generated": generated views everywhere; photographs are references.
At most {max_gen} NEW generated views (reusing a photo marked generated costs nothing and is preferred).

THE SLOTS, in this order. Fill every slot that applies to this product; do not add other slots.
A. WHOLE (2 or 3 slots, kind "whole", required): the complete product from the front, then from a three-quarter angle,
   then (optional) from the other side. For a set of several pieces: all pieces together, then all pieces laid out apart.
B. CONTROLS (kind "controls", required if the product has controls or printed labels): ONE close-up showing ALL
   controls with every label, cut from the cleanest front photograph (never from a generated image). Add a second
   "controls" slot only for a small control group that is easy to miss (an indicator light, a single button apart from
   the rest). Never more than 2 controls slots, never two cuts of the same controls.
C. STATES (kind "state", required if the product opens or has removable parts): the product OPEN is the most important
   panel after the whole views - one slot for the whole product open (door open, lid up, part pulled out), and one for a
   closer view of the open area. Add a slot for a part removed only if that changes how the product looks.
D. PARTS (kind "part"): {parts_rule}
E. DETAILS (kind "detail", optional, at most 2): only something a model would get wrong and no other panel shows.
   Never a plain surface, a base, feet, or the logo when it is already readable in the controls close-up.
Fewer, larger panels are better: 6 to 9 slots in total.

HOW TO FILL A SLOT - exactly one of:
- "photo": the number of a photo that shows exactly this, whole and uncut. For kinds whole / state / part the photo must
  be clean (no food, props, hands, added text). Never give a crop for these: code frames the product.
- "photo" + "crop": ONLY for kind "controls" (and "detail" cut from a clean whole-product photograph on white):
  fractions [x0, y0, x1, y1] of that photo, a little loose, containing every control named.
- "generate": {{"what": "one sentence, exactly what to show", "refs": [1 to 3 photo numbers that show it best],
  "wide": false}} - not allowed for kind "controls", not allowed in real mode.
- "missing": true - when none of the above is possible. A missing slot is acceptable; a wrong picture is not.
Never use two slots for the same thing. Never crop an infographic or a photo of several parts.
When several photos show the same thing equally well, take the one with the most pixels.
Captions: UPPER CASE, at most 38 characters, the view or the part and its state.

Answer one JSON object and nothing else:
{{"slots": [{{"kind": "whole", "caption": "FRONT", "photo": 3}},
            {{"kind": "controls", "caption": "CONTROL PANEL - 4 DIALS", "photo": 3, "crop": [0.1, 0.2, 0.9, 0.4]}},
            {{"kind": "state", "caption": "DOOR OPEN", "generate": {{"what": "...", "refs": [3, 7], "wide": false}}}},
            {{"kind": "part", "caption": "CRUMB TRAY", "missing": true}}]}}"""

KIND_ROLE = {"whole": "hero", "controls": "view", "state": "view", "part": "detail", "detail": "detail"}
WHOLE_VIEWS = ("front", "3/4 left", "3/4 right", "side", "top", "set together")


def candidates(tags, mode="mixed"):
    order = {"hero": 0, "view": 1, "detail": 2, "reference": 3}
    keys = [k for k, v in tags.items() if v.get("use") in order and v.get("same_product") is not False
            and not (mode == "real" and v.get("generated"))]
    return sorted(keys, key=lambda k: (order[tags[k]["use"]], not tags[k].get("clean"), k))[:36]


def plan_sheet(call, folder, product, tags, info, mode, max_gen, model, include_parts=False):
    """Ask the LLM to fill the slots, then enforce in code what it may and may not do. Returns (slots, notes)."""
    cands = candidates(tags, mode)
    listing = "\n".join(f'{i + 1}) {os.path.basename(k)} - {"GENERATED, " if tags[k].get("generated") else ""}{tags[k].get("view")}, '
                        f'{tags[k].get("state") or "no state"}, background {tags[k].get("background")}, '
                        f'{"clean" if tags[k].get("clean") else "NOT clean"}, {"x".join(str(v) for v in Image.open(os.path.join(folder, k)).size)} px: '
                        f'{tags[k].get("desc", "")}' for i, k in enumerate(cands))
    plan = parse_json(llm(call, model, PLAN.format(name=f'{product.get("brand", "")} {product.get("model", "")}'.strip(),
                                                   category=product.get("category", ""), info=info, listing=listing, mode=mode, max_gen=max_gen,
                                                   parts_rule=("one slot per removable part or per piece of a set, each alone." if include_parts else
                                                               "none, unless the product IS a set of pieces: then at most 3 slots, for the pieces with "
                                                               "the most to get wrong.")),
                          [data_url(contact_sheet(folder, cands), 2400, 85)], max_tokens=5000))
    slots, notes, seen, new_gen, n_controls = [], [], set(), 0, 0
    for sl in plan.get("slots", []):
        kind = sl.get("kind") if sl.get("kind") in KIND_ROLE else "detail"
        cap = re.sub(r"[–—]", "-", str(sl.get("caption", ""))).upper()[:40]
        out = {"kind": kind, "caption": cap, "role": KIND_ROLE[kind]}
        n = sl.get("photo")
        gen = sl.get("generate")
        valid = isinstance(n, (int, float)) and 1 <= int(n) <= len(cands)
        if valid and (mode != "generated" or tags[cands[int(n) - 1]].get("generated")):
            key = cands[int(n) - 1]
            t = tags[key]
            crop = sl.get("crop")
            if crop and len(crop) == 4 and max(crop) > 1.001:
                crop = [c / 10 for c in crop] if max(crop) <= 10 else None
            plain_whole = t.get("clean") and t.get("background") == "white" and t.get("view") in WHOLE_VIEWS and not t.get("generated")
            if crop and not (kind in ("controls", "detail") and plain_whole):
                notes.append(f"{cap}: crop refused (a close-up may only be cut from a clean whole-product photograph)")
                crop = None
                if kind == "controls":
                    key = None
            if key and kind in ("whole", "state", "part") and not t.get("clean"):
                notes.append(f"{cap}: photo refused (not clean)")
                if kind != "whole" and mode != "real" and not gen:
                    gen = {"what": f"{cap.lower()} - the same state as in the reference photo, but with nothing else in the picture",
                           "refs": [int(n)], "wide": False}
                    notes[-1] += " - a clean view will be generated from it"
                key = None
            if key and kind == "controls":
                n_controls += 1
                if n_controls > 2:
                    notes.append(f"{cap}: dropped (at most 2 control close-ups)")
                    continue
            if key and kind == "controls" and not crop and not (t.get("view") == "detail close-up" and t.get("clean")):
                notes.append(f"{cap}: photo refused (not a clean control close-up and no crop given)")
                key = None
            if key and (key, tuple(crop or ())) not in seen:
                seen.add((key, tuple(crop or ())))
                out.update(image=key, crop=crop, generated=bool(t.get("generated")))
        if "image" not in out and gen and mode != "real" and kind != "controls" and new_gen < max_gen:
            refs = [cands[int(r) - 1] for r in gen.get("refs", []) if isinstance(r, (int, float)) and 1 <= int(r) <= len(cands)][:3]
            if refs and gen.get("what"):
                new_gen += 1
                out.update(generate={"what": gen["what"], "refs": refs, "wide": bool(gen.get("wide"))}, generated=True)
        if "image" not in out and "generate" not in out:
            out["missing"] = True
        slots.append(out)
    return slots, notes


# ----------------------------------------------------------------------------- sheet: hard gates

def touches_edge(im, tol=45, run=0.12):   # tol is high on purpose: a soft floor shadow reaching the edge is not a cut product
    """True when the subject of a plain-background picture is cut off by the picture edge.

    A tightly framed product touches an edge at a point; a cut-off one runs along it. So an edge counts only when
    more than  of its length is subject, not background."""
    rgb = im.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, rgb.getpixel((2, 2)))).convert("L")
    mask = diff.point(lambda v: 255 if v > tol else 0)
    w, h = mask.size
    strips = (mask.crop((0, 0, w, 3)), mask.crop((0, h - 3, w, h)), mask.crop((0, 0, 3, h)), mask.crop((w - 3, 0, w, h)))
    for strip in strips:
        hist = strip.histogram()
        if hist[255] / float(strip.width * strip.height) > run:
            return True
    return False


def gate_report(slots, built, rows, size):
    """What the finished sheet covers, as plain facts. `fatal` lists gates that must stop delivery."""
    area = lambda r: r["box"][2] * r["box"][3]                                           # noqa: E731
    hero = sum(area(r) for r in rows if r["role"] == "hero") / float(size * size)
    biggest_hero = max([area(r) for r in rows if r["role"] == "hero"] or [0])
    fatal = []
    if not any(b.get("kind") == "whole" for b in built):
        fatal.append("G1/G3: no whole-product view could be placed")
    if hero < 0.38:
        fatal.append(f"G1: whole-product panels take {hero:.0%} of the sheet (minimum 40 %)")
    tallest_hero = max([r["box"][3] for r in rows if r["role"] == "hero"] or [0])
    if any(r["box"][3] > tallest_hero * 1.02 for r in rows if r["role"] != "hero"):
        fatal.append("G1: a panel is taller than the whole views")       # height, not area: a full-width control strip is fine
    controls = [r for r in rows if r.get("kind") == "controls"]
    if controls and max(r["box"][2] for r in controls) < 0.3 * size and max(r["scale"] for r in controls) < 0.6:
        fatal.append("G7: the control close-up is small: narrower than half the sheet and shown below its source size")
    gone = [s for s in slots if s.get("missing") or s.get("dropped")]
    return {"hero_area": round(hero, 2), "panels": len(rows), "missing": [s["caption"] for s in gone],
            "required_missing": [s["caption"] for s in gone if s["kind"] in ("whole", "controls")], "fatal": fatal}


# ----------------------------------------------------------------------------- sheet: clean, generate

GENERATE = ("Product reference photo for a catalogue. Show: {what}. "
            "The reference images show the real product: reproduce it EXACTLY - same shape, proportions, materials, colours, "
            "every button, dial, display, printed label, logo, handle and marking, in the same places. Do not add, remove, "
            "restyle or relabel anything. No food, no liquid, no hands, no people, no props, no text overlay. Plain pure white "
            "background, soft even studio light, a soft neutral shadow under the product, the whole subject inside the frame "
            "with a small margin.")


def trim_plain(im, tol=14, pad=0.03):
    """Crop away a plain background (white, cream, any flat colour taken from the corner)."""
    rgb = im.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, rgb.getpixel((2, 2)))).convert("L")
    box = diff.point(lambda v: 255 if v > tol else 0).getbbox()
    if not box:
        return im
    px, py = int(im.width * pad), int(im.height * pad)
    return im.crop((max(0, box[0] - px), max(0, box[1] - py), min(im.width, box[2] + px), min(im.height, box[3] + py)))


def banner_crop(im, where, share=0.2):
    w, h = im.size
    return {"bottom": im.crop((0, 0, w, int(h * (1 - share)))), "top": im.crop((0, int(h * share), w, h)),
            "left": im.crop((int(w * share), 0, w, h)), "right": im.crop((0, 0, int(w * (1 - share)), h))}.get(where, im)


def isolate(call, im):
    """Background removal: the product's real pixels on white, trimmed to the product."""
    res = call(BG_REMOVE, {"image_url": data_url(im, quality=95)})
    cut = Image.open(io.BytesIO(fetch(res["image"]["url"], timeout=120))).convert("RGBA")
    white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    white.alpha_composite(cut)
    box = cut.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    return white.convert("RGB").crop(box) if box else im


def generate_view(call, folder, what, refs, wide=False):
    size = {"width": 2048, "height": 1536} if wide else {"width": 2048, "height": 2048}
    urls = [data_url(Image.open(os.path.join(folder, r)), 1536, 90) for r in refs]
    res = call(IMAGE_EDIT, {"prompt": GENERATE.format(what=what), "image_urls": urls, "image_size": size, "quality": "high",
                            "num_images": 1, "output_format": "png"})
    return Image.open(io.BytesIO(fetch(res["images"][0]["url"], timeout=180))).convert("RGB")


CONTROLS_CHECK = """This picture should be a close-up showing ALL the controls of {name} ({caption}): every dial, button,
display and printed label complete, and little else.
Answer one JSON object: {{"cut": ["top" | "bottom" | "left" | "right" for each side where a control or its label is cut off
or still outside the picture], "too_wide": true if the controls fill less than half of the picture height or width}}"""


def refine_controls(call, folder, slot, name, model, rounds=4):
    """Fit the control close-up by looking at the result, not by trusting coordinates.

    The LLM's box is 5-10 % off, which is exactly a cut dial. So: cut, show the cut to a cheap vision model, grow the
    side it says is cut (or tighten when the controls are small in the picture), and look again."""
    im = Image.open(os.path.join(folder, slot["image"])).convert("RGB")
    box = [min(1.0, max(0.0, float(c))) for c in slot["crop"]]
    first = list(box)
    fw, fh = first[2] - first[0], first[3] - first[1]
    limit = [first[0] - 0.15 * fw, first[1] - 0.5 * fh, first[2] + 0.15 * fw, first[3] + 0.5 * fh]   # first run: grew to the whole product
    for _ in range(rounds):
        cut = im.crop((int(box[0] * im.width), int(box[1] * im.height), int(box[2] * im.width), int(box[3] * im.height)))
        try:
            ans = parse_json(llm(call, model, CONTROLS_CHECK.format(name=name, caption=slot["caption"]), [data_url(cut, 1024, 85)], max_tokens=300))
        except Exception:
            break
        sides = [s for s in ans.get("cut", []) if s in ("top", "bottom", "left", "right")]
        if not sides and not ans.get("too_wide"):
            break
        w, h = box[2] - box[0], box[3] - box[1]
        if sides:
            box[1] -= 0.35 * h if "top" in sides else 0
            box[3] += 0.35 * h if "bottom" in sides else 0
            box[0] -= 0.12 * w if "left" in sides else 0
            box[2] += 0.12 * w if "right" in sides else 0
        else:
            box = [box[0] + 0.08 * w, box[1] + 0.12 * h, box[2] - 0.08 * w, box[3] - 0.12 * h]
        box = [max(limit[0], box[0]), max(limit[1], box[1]), min(limit[2], box[2]), min(limit[3], box[3])]
        box = [min(1.0, max(0.0, v)) for v in box]
    return box


def prepare_panel(call, folder, panel, tags, isolate_busy):
    """The image a panel shows: cropped by rule, never by the LLM's eye, except for close-up cuts."""
    im = Image.open(os.path.join(folder, panel["image"])).convert("RGB")
    t = tags.get(panel["image"], {})
    crop = panel.get("crop")
    if crop:
        x0, y0, x1, y1 = [min(1.0, max(0.0, float(c))) for c in crop]
        if x1 - x0 > 0.05 and y1 - y0 > 0.05:
            px, py = (0.0, 0.0) if panel.get("crop_exact") else (0.02, (y1 - y0) * 0.08)   # a small safety margin unless hand-set
            x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(1, x1 + px), min(1, y1 + py)
            return im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
    im = banner_crop(im, t.get("banner", "none"))
    if t.get("background") in ("white", "graphic") and t.get("banner", "none") in ("none", None):
        return trim_plain(im)
    if isolate_busy and panel.get("kind") in ("whole", "state", "part") and t.get("view") not in ("interior", "detail close-up", "infographic"):
        try:
            return isolate(call, im)
        except Exception as e:
            logger.warning("background removal failed for %s: %s", panel["image"], e)
    return im


VERIFY = """These {n} pictures are panels of a product reference sheet for: {name}. Their captions, in order: {caps}
For each panel answer whether the caption is true for what the picture shows, and whether the picture is usable:
not usable = the named part is cut off or missing, the picture shows a different part than the caption says, or it contains
added marketing text, badges or callout dots. Answer one JSON object {{"panels": [{{"n": 1, "ok": true, "why": ""}}]}}."""


def verify_panels(call, name, built, model):
    """Drop panels whose picture does not match its caption. Returns (kept, dropped captions with the reason)."""
    if len(built) < 2:
        return built, []
    caps = "; ".join(f'{i + 1}) {b["caption"].replace(" (GENERATED)", "")}' for i, b in enumerate(built))
    try:
        got = parse_json(llm(call, model, VERIFY.format(n=len(built), name=name, caps=caps),
                             [data_url(b["im"], 768, 80) for b in built], max_tokens=3000))["panels"]
    except Exception as e:
        logger.warning("panel check failed, keeping every panel: %s", e)
        return built, []
    bad = {int(g["n"]) - 1: g.get("why", "") for g in got if g.get("ok") is False}
    kept = [b for i, b in enumerate(built) if i not in bad]
    if len(kept) < 3:                                   # never let the check empty the sheet
        return built, []
    return kept, [f'{built[i]["caption"]}: {why}' for i, why in bad.items()]


# ----------------------------------------------------------------------------- sheet: layout + compose

def _splits(items, max_rows):
    n = len(items)
    if n == 0:
        return [[]]
    out = []

    def rec(start, rows):
        if start == n:
            out.append(rows)
            return
        if len(rows) == max_rows:
            return
        for end in range(start + 1, n + 1):
            rec(end, rows + [items[start:end]])
    rec(0, [])
    return out


AREA_WEIGHT = {"whole": 1.2, "controls": 1.9, "state": 1.6, "part": 0.8, "detail": 0.8}


def layout(panels, W, H, gap, cap_h, max_up, pad=25, hero_weight=None):
    """Pack the panels into rows that fill the sheet: every way of cutting the ordered list into rows is tried and the
    one that puts the most picture on the sheet wins.

    A row is as tall as its width allows (all its pictures at one height, edge to edge). If the rows together are
    taller than the sheet they are all shrunk by the same factor; if shorter, the spare height is shared out. The score
    is the picture area actually drawn, weighted by what the panel is (whole views, controls and open states count
    more than small details), with penalties when a whole view is not in the tallest row or a row is too thin to read.
    The earlier version gave each kind of panel its own rows and a fixed height share, which left a third of a sheet
    white around a single control panel (coffee maker, 2026-11-10).
    """
    order = {"hero": 0, "view": 1, "detail": 2}
    items = sorted(panels, key=lambda q: order.get(q["role"], 2))       # stable: plan order inside each role
    n = len(items)
    asp = [q["im"].width / q["im"].height for q in items]
    cap = [q["im"].height * q.get("max_up", max_up) for q in items]
    wgt = [AREA_WEIGHT.get(q.get("kind"), 1.0) for q in items]
    best = None
    for mask in range(1 << (n - 1)):
        rows, start = [], 0
        for i in range(n - 1):
            if mask >> i & 1:
                rows.append(list(range(start, i + 1)))
                start = i + 1
        rows.append(list(range(start, n)))
        if len(rows) > 6 or max(len(r) for r in rows) > 6:
            continue
        fixed = len(rows) * (cap_h + 2 * pad) + gap * (len(rows) - 1)
        if H - fixed <= 0:
            continue
        nat = [(W - gap * (len(r) - 1) - 2 * pad * len(r)) / sum(asp[i] for i in r) for r in rows]
        nat = [min(h, max(cap[i] for i in r)) for h, r in zip(nat, rows)]          # never taller than its pictures can be drawn
        f = min(1.0, (H - fixed) / sum(nat))
        hts = [h * f for h in nat]
        score = 0.0
        for r, h in zip(rows, hts):
            for i in r:
                d = min(h, cap[i])
                score += asp[i] * d * d * wgt[i]
        hero_rows = [h for r, h in zip(rows, hts) if any(items[i]["role"] == "hero" for i in r)]
        other = [h for r, h in zip(rows, hts) if not any(items[i]["role"] == "hero" for i in r)]
        if hero_rows and other and min(hero_rows) < max(other):
            score *= 0.55                                                           # whole views are the tallest rows
        if any(items[i]["role"] == "hero" for r in rows for i in r) and any(
                {items[i]["role"] for i in r} > {"hero"} for r in rows):
            score *= 0.8                                                            # do not mix whole views with small details
        if min(hts) < 300:
            score *= 0.5
        if best is None or score > best[0]:
            best = (score, rows, hts)
    if best is None:
        raise ValueError("no layout fits: too many panels")
    _, rows, hts = best
    out = []
    for r, h in zip(rows, hts):
        cells = [(items[i], asp[i] * h + 2 * pad) for i in r]
        extra = (W - gap * (len(r) - 1) - sum(c[1] for c in cells)) / len(r)
        out.append(([(q, cw + extra) for q, cw in cells], h + cap_h + 2 * pad))
    slack = max(0, H - sum(rh for _, rh in out) - gap * (len(out) - 1)) / len(out)
    return out, slack


def compose(panels, title, subtitle, size=5000, margin=60, gap=30, cap_h=110, max_up=1.5, hero_weight=None):
    """panels: [{"im": PIL, "role": hero|view|detail, "caption": str}] -> (sheet, info rows)."""
    bg, cell, ink, sub, line = (232, 232, 232), (255, 255, 255), (20, 20, 20), (90, 90, 90), (200, 200, 200)
    sheet = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(sheet)

    def fit(text, max_w, s):
        while s > 24 and d.textlength(text, font=font(s)) > max_w:
            s -= 2
        return font(s)

    y = margin
    if title:
        d.rectangle((margin, y, size - margin, y + 210), fill=cell)
        d.text((margin + 50, y + 28), title, fill=ink, font=fit(title, size - 2 * margin - 100, 84))
        d.text((margin + 50, y + 128), subtitle or "", fill=sub, font=fit(subtitle or "", size - 2 * margin - 100, 58))
        y += 210 + gap
    rows, slack = layout(panels, size - 2 * margin, size - margin - y, gap, cap_h, max_up, hero_weight=hero_weight)
    info = []
    for cells, rh in rows:
        x, rh = margin, rh + slack
        for p, cw in cells:
            d.rectangle((x, y, x + cw, y + rh), fill=cell, outline=line, width=3)
            im = p["im"]
            scale = min((cw - 50) / im.width, (rh - cap_h - 50) / im.height, p.get("max_up", max_up))
            im2 = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
            sheet.paste(im2, (int(x + (cw - im2.width) / 2), int(y + 25 + (rh - cap_h - 50 - im2.height) / 2)))
            f = fit(p["caption"], cw - 50, 60)
            d.text((x + (cw - d.textlength(p["caption"], font=f)) // 2, y + rh - cap_h + 14), p["caption"], fill=ink, font=f)
            info.append({"caption": p["caption"], "role": p["role"], "kind": p.get("kind"), "solo": bool(p.get("solo")),
                         "source": p.get("source"), "generated": bool(p.get("generated")),
                         "scale": round(scale, 2), "box": [int(x), int(y), int(cw), int(rh)]})
            x += cw + gap
        y += rh + gap
    return sheet, info
