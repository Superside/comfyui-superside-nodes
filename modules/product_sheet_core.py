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


def parse_json(text, want=None):
    """The first JSON value in a model's answer. Models add a sentence before or after it, or a second object: the
    value is read from its first bracket and whatever follows is ignored (a greedy match broke on trailing text).
    want=dict reads the first object even when a list comes earlier in the answer."""
    text = text or ""
    starts = [i for i in ((text.find("{"),) if want is dict else (text.find("{"), text.find("["))) if i >= 0]
    if not starts:
        raise ValueError("the model did not answer with JSON")
    for start in sorted(starts):
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except ValueError:
            continue
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
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
If a link is given, the product is the one on that link: when the photo shows something else, describe the link's
product and add "photo_matches": false.
Makers sell near-identical models under different numbers. Say "high" only if the model number is printed in the photo or
given by the user; if you recognise the product line but cannot tell the models apart, say "low" and list them."""

SEARCH = """Find up to {n} web pages that show product photos of this exact product: {query}.
Wanted this time: {focus}. One page per site, each from a different site.
Leave out the big marketplaces and chains whose pages cannot be read by a script (Amazon, Walmart, Target, Wayfair,
Best Buy, Home Depot, Lowe's, Macy's, eBay){exclude}.
Answer with a JSON list of URLs only."""

TAG = """PRODUCT: {name} ({category}), colour / finish: {colour}
You get {n} photos collected for this product, in this order: {names}
For EACH photo answer one object:
- "image": the file name exactly as given
- "view": the category of the photo, exactly one of:
  "front", "3/4 left", "3/4 right", "side", "back", "top" - the whole product alone, from that angle;
  "set together" - the product with its accessories, or all pieces of a set;
  "part alone" - one or more accessories or removable parts without the product;
  "controls close-up" - the buttons, switches, dials or display fill most of the picture;
  "logo close-up" - the brand logo or model name on the product fills most of the picture;
  "detail close-up" - another part of the product seen close;
  "interior" - the product open, showing its inside;
  "in hand" - a hand holds the product and little else is in the picture;
  "in use" - a person is using it;
  "packaging" - the box, closed or open;
  "infographic" - mostly added text, callouts, charts or several pictures in one;
  "not the product"
- "angle": the angle the main product is seen from, whatever else is in the picture: "front", "3/4 left",
  "3/4 right", "side", "back", "top", or "none" if the main product is not in the picture
- "colour": the colour / finish of the main product in two or three words ("brushed steel", "black stainless",
  "white"); "" if the main product is not in the picture
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


def lookup_link(call, url):
    """What a product link sells, asked of a model with web search - for pages a script cannot read."""
    try:
        return llm(call, SEARCH_MODEL, "Which exact product is sold on this page? Give brand, model number and product name in one "
                                       f"line, nothing else. If you cannot tell, answer: unknown.\n{url}", max_tokens=200).strip()[:240]
    except Exception:
        return ""


def lookup_photo(call, image):
    """Brand and model of the product in a photo, asked of a vision model with web search - for photos with no visible name."""
    try:
        return llm(call, SEARCH_MODEL, "Search the web for the exact product shown in this photo. Give brand, model number and "
                                       "product name in one line, nothing else. If you are not sure it is that exact product, "
                                       "answer: unknown.", [data_url(image, 1024)], max_tokens=200).strip()[:240]
    except Exception:
        return ""


def identify(call, image, hint, model):
    hint_txt = f" (the user says: {hint})" if hint else ""
    return parse_json(llm(call, model, IDENTIFY.format(hint=hint_txt), [data_url(image, 1024)], max_tokens=600))


SEARCH_FOCUS = ("the maker's own sites in several countries (.com, .ca, .co.uk, .com.au ...) and specialist or regional retailers",
                "hands-on reviews, unboxings, comparison tests and blog posts that show their own photos of the product",
                "smaller online shops, kitchen or appliance specialists, price-comparison and manual / spare-parts sites")


def search_pages(call, query, n=8, exclude_hosts=(), round_no=1):
    exclude = (" and these sites, already tried: " + ", ".join(sorted(exclude_hosts))) if exclude_hosts else ""
    try:
        urls = parse_json(llm(call, SEARCH_MODEL, SEARCH.format(n=n, query=query, exclude=exclude, focus=SEARCH_FOCUS[(round_no - 1) % 3]), max_tokens=1500))
    except Exception as e:
        logger.warning("search failed: %s", e)
        return []
    out = [u for u in urls if isinstance(u, str) and u.startswith("http")]
    return out[:n]


# sites that state they do not allow automated agents: nothing is fetched from them, pages or image servers
REFUSED_HOSTS = ("amazon.", "media-amazon.", "ssl-images-amazon.")
IMAGE_QUERIES = ("{q}", "{q} in use in hand", "{q} buttons controls close up", "{q} open inside parts")


def search_images(query, per_query=60, min_side=700, patterns=None):
    """Image results for a product from the open-source ddgs metasearch (no key). Returns (results, note); each result
    is {"image", "page", "title"}. Needs the 'ddgs' package: without it the list is empty and the note says so."""
    try:
        from ddgs import DDGS
    except Exception:
        return [], "image search skipped: the 'ddgs' package is not installed (run: pip install ddgs)"
    host = lambda u: urllib.parse.urlparse(u or "").netloc.lower()      # noqa: E731
    out, seen, refused, small, errors = [], set(), 0, 0, 0
    engine = DDGS()
    for pattern in (patterns or IMAGE_QUERIES):
        try:
            found = engine.images(pattern.format(q=query), max_results=per_query, size="Large")
        except Exception as e:
            logger.warning("image search failed: %s", e)
            errors += 1
            continue
        for r in found or []:
            img, page = r.get("image") or "", r.get("url") or ""
            if not img.startswith("http") or img in seen:
                continue
            seen.add(img)
            if any(t in host(img) or t in host(page) for t in REFUSED_HOSTS):
                refused += 1
                continue
            try:
                known = max(int(r.get("width") or 0), int(r.get("height") or 0))
            except Exception:
                known = 0
            if known and known < min_side:
                small += 1
                continue
            out.append({"image": img, "page": page, "title": (r.get("title") or "")[:120]})
    note = f"image search: {len(out)} candidates ({small} too small, {refused} on sites that refuse automated access"
    return out, note + (f", {errors} searches failed)" if errors else ")")


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
    junk = r"logo|icon|sprite|swatch|badge|thumb|avatar|banner|promo|footer|header|flag|payment|rating|star"
    keep, others = [], []
    for u in found:
        low = urllib.parse.unquote(u).lower()
        if re.search(junk, low):
            continue
        (keep if any(t in low for t in tokens) else others).append(u)
    # images the page itself declares as the product's: og:image and schema.org "image" (their file names are often hashes)
    declared = re.findall(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', html, re.I)
    for block in re.findall(r'"image"\s*:\s*(\[[^\]]{0,4000}\]|"[^"]+")', html):
        declared += re.findall(r'https?://[^"\s\\]+', block)
    keep += [u for u in declared if u not in keep]
    if len(keep) < 4:                                   # nothing named after the model: take the page's other photos; the
        keep += [u for u in others if u not in keep][:12]   # tagger decides later which of them show the product
    text = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html, flags=re.S)
    return list(dict.fromkeys(keep)), re.sub(r"\s+", " ", text)[:6000]


def best_variant(url):
    """The largest version of an image URL: drop resize parameters and CDN resize prefixes."""
    u = url.split("?")[0] if re.search(r"[?&](sw|sh|w|h|width|height|wid|hei)=", url) else url
    return re.sub(r"/dw/image/v2/[A-Z_]+/on/", "/on/", u)


def download_images(urls, out_dir, max_images, min_side=700, referer=None, seen=None, timeout=60):
    """seen: a set shared between calls, so the same picture found on two sites is saved once."""
    os.makedirs(out_dir, exist_ok=True)
    seen, saved = (seen if seen is not None else set()), []
    for u in urls:
        if len(saved) >= max_images:
            break
        for cand in dict.fromkeys((best_variant(u), u)):
            if any(t in urllib.parse.urlparse(cand).netloc.lower() for t in REFUSED_HOSTS):
                break                                       # hosted by a site that refuses automated access
            try:
                raw = fetch(cand, timeout=timeout, referer=referer)
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


TAG_VERSION = 4          # 2 = finer categories (2026-11-11); 3 = angle; 4 = colour of the product in each photo

FINISHES = """These are colour / finish descriptions written for photos of ONE product model, which is sold in several
finishes: {colours}
Group the descriptions that mean the same finish. Answer one JSON object: {{"groups": [["...", "..."], ["..."]]}}"""


def other_model_in_name(name, model):
    """True when a file name carries another model number of the same family: 'ab70x' for an AB-60. Makers sell
    near-identical models, shops reuse each other's pictures, and a vision model cannot tell them apart - the file
    name often can."""
    m = re.match(r"\s*([a-z]{2,5})[-_ ]?(\d{1,5})", (model or "").lower())
    if not m:
        return False
    found = re.findall(m.group(1) + r"[-_ ]?(\d{1,5})", name.lower())
    return bool(found) and m.group(2) not in found


def finish_tone(path):
    """How light the product's body is in a photo on a plain background: the 75th percentile of the luminance of the
    product's own pixels (0-255), or None when the photo has no plain background to separate it from. A vision model
    gave a brushed-steel and a black-stainless version of one coffee maker the same colour words; this number does
    not - about 200 against about 110."""
    try:
        return tone_of(Image.open(path))
    except Exception:
        return None


def tone_of(im):
    """finish_tone for an image already in memory (a prepared panel sits on white, so it can always be measured)."""
    im = im.convert("RGB")
    if not plain_background(im):
        return None
    im = im.copy()
    im.thumbnail((256, 256))
    diff = ImageChops.difference(im, Image.new("RGB", im.size, im.getpixel((2, 2)))).convert("L")
    vals = sorted(v for v, d in zip(im.convert("L").getdata(), diff.getdata()) if d > 24)
    return vals[int(len(vals) * 0.75)] if len(vals) > 400 else None


def reference_tone(folder, tags):
    """The tone of the product's finish: the middle value over the whole-product photos research kept."""
    vals = sorted(v for v in (finish_tone(os.path.join(folder, k)) for k, t in tags.items()
                              if t.get("same_product") is not False and t.get("view") in WHOLE_VIEWS and not k.startswith("generated/"))
                  if v is not None)
    return vals[len(vals) // 2] if vals else None


def lock_tone(folder, product, tags, reference=None):
    """One finish, measured. Whole-product photos on a plain background are grouped by finish_tone; the largest group
    (or the group of the input photo, when there is one) is the product's finish and the photos of the other groups
    are kept but not used. Returns a note or ""."""
    tones = {}
    for k, t in tags.items():
        if t.get("same_product") is not False and t.get("view") in WHOLE_VIEWS and not k.startswith("generated/"):
            v = finish_tone(os.path.join(folder, k))
            if v is not None:
                tones[k] = v
    if len(tones) < 3:
        return ""
    # one split, at the widest jump between neighbouring tones, and only if that jump is wide: exposure alone moves a
    # photo by 20-30, a different finish by more (measured: 116 -> 151 between dark and brushed steel)
    ordered = sorted(tones.items(), key=lambda kv: kv[1])
    jump, at = max((ordered[i + 1][1] - ordered[i][1], i) for i in range(len(ordered) - 1))
    if jump < 30:
        return ""
    groups = [ordered[:at + 1], ordered[at + 1:]]
    ref = finish_tone(reference) if reference else None
    centre = lambda g: sum(v for _k, v in g) / len(g)                               # noqa: E731
    best = min(groups, key=lambda g: abs(centre(g) - ref)) if ref is not None else max(groups, key=len)
    off = 0
    for g in groups:
        if g is not best:
            for k, _v in g:
                tags[k]["same_product"], tags[k]["why_not"] = False, "another finish (lighter or darker body)"
                off += 1
    return (f"{len(groups)} finishes measured in the whole-product photos: the {'input photo' if ref is not None else 'most common'} one "
            f"was kept, {off} photos of the others are kept, not used") if off else ""


def lock_variant(call, folder, product, tags, model, had_photo):
    """One model, one finish. (1) A photo whose file name carries another model number of the same family is not this
    product. (2) With no input photo to say which finish is meant, the finish seen in most whole-product photos becomes
    the product's finish and the others are kept but not used. Without this a sheet built from a name alone mixed a
    dark and a steel version of the same coffee maker, and a sheet for one oven model was built from photos of its sibling model (2026-11-11).
    Returns notes for the report."""
    notes, wrong = [], 0
    for k, t in tags.items():
        if t.get("same_product") is not False and other_model_in_name(os.path.basename(k), product.get("model", "")):
            t["same_product"], t["why_not"] = False, "another model number in the file name"
            wrong += 1
    if wrong:
        notes.append(f"{wrong} photos carry another model number in their file name: kept, not used")
    if not had_photo:
        seen = {}
        for k, t in tags.items():
            c = (t.get("colour") or "").strip().lower()
            if c and t.get("same_product") is not False and t.get("view") in WHOLE_VIEWS:
                seen[c] = seen.get(c, 0) + 1
        if len(seen) > 1:
            try:
                groups = parse_json(llm(call, model, FINISHES.format(colours="; ".join(sorted(seen))), max_tokens=600)).get("groups") or []
                groups = [[str(c).strip().lower() for c in g] for g in groups if g]
                best = max(groups, key=lambda g: sum(seen.get(c, 0) for c in g))
                others = {c for g in groups if g is not best for c in g}
                product["colour"] = max(best, key=lambda c: seen.get(c, 0))
                off = 0
                for t in tags.values():
                    if (t.get("colour") or "").strip().lower() in others and t.get("same_product") is not False:
                        t["same_product"], t["why_not"] = False, "another finish of the product"
                        off += 1
                if off:
                    notes.append(f'no input photo, {len(groups)} finishes found: "{product["colour"]}" is the most common and was kept; '
                                 f"{off} photos of other finishes are kept, not used")
            except Exception as e:
                logger.warning("finishes not grouped: %s", e)
    photo = os.path.join(folder, "images", "input_photo.jpg")
    tone = lock_tone(folder, product, tags, reference=photo if (had_photo and os.path.exists(photo)) else None)
    if tone:
        notes.append(tone)
    if notes:
        json.dump(tags, open(os.path.join(folder, "image_tags.json"), "w", encoding="utf-8"), indent=1)
        json.dump(product, open(os.path.join(folder, "product.json"), "w", encoding="utf-8"), indent=1)
    return notes

# what a sheet is built from, in the order of its hierarchy: label, the views that belong to it, the search that finds more
CATEGORIES = (
    ("PRODUCT ALONE", ("front", "3/4 left", "3/4 right", "side", "top", "back"), "{q} product photo white background"),
    ("CONTROLS CLOSE-UP", ("controls close-up",), "{q} buttons controls close up"),
    ("LOGO CLOSE-UP", ("logo close-up",), "{q} logo close up"),
    ("OTHER DETAILS", ("detail close-up", "interior"), ""),
    ("ACCESSORIES", ("part alone",), "{q} attachments accessories parts"),
    ("PRODUCT WITH ACCESSORIES", ("set together",), ""),
    ("IN HAND", ("in hand",), "{q} held in hand"),
    ("IN USE", ("in use",), ""),
    ("PACKAGING", ("packaging",), ""),
    ("INFOGRAPHICS", ("infographic",), ""),
)


def category_needed(label, profile):
    """Whether an empty category is worth a search for this product: no controls photos for cookware, no in-hand
    photos for an oven, no 'product alone' for a set (the set is the product)."""
    if label == "CONTROLS CLOSE-UP":
        return applies(profile, "has_controls")
    if label == "IN HAND":
        return applies(profile, "handheld")
    if label == "ACCESSORIES":
        return applies(profile, "has_accessories") and not (profile and profile.get("is_set"))
    if label == "PRODUCT ALONE":
        return not (profile and profile.get("is_set"))
    return True


def categorize(folder, tags):
    """The photos of the product (its own colour and model) sorted into the categories a sheet is built from; inside a
    category the photos with no added text come first, then the largest."""
    out = {}
    for label, views, _q in CATEGORIES:
        keys = [k for k, t in tags.items() if t.get("view") in views and t.get("same_product") is not False and not k.startswith("generated/")]
        out[label] = sorted(keys, key=lambda k: (tags[k].get("banner", "none") not in ("none", None), not tags[k].get("clean"),
                                                 -min(Image.open(os.path.join(folder, k)).size)))
    return out


def best_of(folder, tags, view, used=()):
    """The best photo of one category for a panel: no marketing text all over it, largest first."""
    keys = [k for k, t in tags.items() if t.get("view") == view and t.get("same_product") is not False and not k.startswith("generated/")
            and t.get("banner", "none") != "all over" and k not in used]
    keys.sort(key=lambda k: (tags[k].get("banner", "none") not in ("none", None), -min(Image.open(os.path.join(folder, k)).size)))
    return keys[0] if keys and min(Image.open(os.path.join(folder, keys[0])).size) >= 350 else None


def category_sheet(folder, cats, cell=300, cols=8):
    """One picture of everything found, grouped under the name of its category - so a person sees at a glance what the
    sheet can be built from and what is missing."""
    f, fh = font(15, bold=False), font(22)
    rows = [(label, keys[:cols * 2]) for label, keys in cats.items()]
    height = sum(46 + max(1, -(-len(keys) // cols)) * (cell + 26) for _l, keys in rows)
    sheet = Image.new("RGB", (cols * cell, max(height, 200)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    y = 0
    for label, keys in rows:
        total = len(cats[label])
        d.text((8, y + 10), f"{label}  -  {total} photo{'s' if total != 1 else ''}" + ("" if total else "   (NONE FOUND)"),
               fill=(255, 214, 90) if total else (255, 110, 90), font=fh)
        y += 46
        for i, rel in enumerate(keys):
            im = Image.open(os.path.join(folder, rel)).convert("RGB")
            im.thumbnail((cell - 8, cell - 8))
            x, yy = (i % cols) * cell, y + (i // cols) * (cell + 26)
            sheet.paste(im, (x + (cell - im.width) // 2, yy + (cell - im.height) // 2))
            d.text((x + 6, yy + cell + 4), os.path.basename(rel)[:34], fill=(210, 210, 210), font=f)
        y += max(1, -(-len(keys) // cols)) * (cell + 26)
    return sheet


def tag_images(call, folder, product, model, subfolders=("images",), batch=10):
    path = os.path.join(folder, "image_tags.json")
    tags = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    tags = {k: v for k, v in tags.items() if os.path.exists(os.path.join(folder, k))}      # a deleted file leaves no tag behind
    todo = [f"{s}/{f}" for s in subfolders for f in list_images(os.path.join(folder, s))
            if f"{s}/{f}" not in tags or tags[f"{s}/{f}"].get("v") != TAG_VERSION]      # older tags lack the newer categories
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
                g["v"] = TAG_VERSION
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


SCENE = """Write the SCENE of one photo of this product in real use, for an image model that already has the product's
reference sheet. PRODUCT: {name} ({category}).

PRODUCT INFO
{info}

Rules:
- Where and how people really use this product at home or at work: the right room, the right surface or hand, the
  product in a normal working state (on, open, in the hand, on the counter ...), doing what it is for.
- It stands or is held the way it is in real use: upright, square to the wall if it sits on a surface, never rotated,
  tilted or floating. Respect anything the product info says about safe placement.
- One or two props that belong with it, without any brand name. If a person appears, only what is needed (a hand, or a
  person seen from the side or behind), described plainly.
- One familiar object or body part that shows its true size.
- Say where the camera is, as someone taking the picture with a phone would stand.
- Do not describe the product's own details (the sheet and the product info carry them) and do not describe lighting
  or style. 60 to 110 words, plain sentences, no list, no title."""


def write_scene(call, folder, product, info, model):
    """One scene for the render step, written for this product, saved as scene_prompt.txt."""
    text = llm(call, model, SCENE.format(name=f'{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")}'.strip(),
                                         category=product.get("category", ""), info=info or "(none)"), max_tokens=500).strip()
    text = "SCENE: " + text.replace("SCENE:", "").strip()
    open(os.path.join(folder, "scene_prompt.txt"), "w", encoding="utf-8").write(text + "\n")
    return text


def read_scene(call, folder, product, info, model):
    path = os.path.join(folder, "scene_prompt.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read().strip()
    return write_scene(call, folder, product, info, model)


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

PROFILE OF THIS PRODUCT: {profile}
The profile decides which slots exist. "no" for controls: no controls slot. "no" for opens: no state slot. "no" for
accessories and for set: no accessories slot. Not held in the hand: the in-use slot is optional, not required. A set:
the whole views are all pieces together, then all pieces laid out apart.

AVAILABLE PHOTOS (the picture shows them numbered in this same order; "generated" = made earlier by an image model):
{listing}

MODE: {mode}
- "real": only photographs. - "mixed": photographs for whole views, controls and labels; a generated view only where no
  clean photograph exists. - "generated": generated views everywhere; photographs are references.
At most {max_gen} NEW generated views (reusing a photo marked generated costs nothing and is preferred).

THE SLOTS, in order of importance. This order is the hierarchy of the sheet: what comes first is drawn largest.
Fill every slot that applies to this product; do not add other slots.
A. WHOLE (2 or 3 slots, kind "whole", required): slot 1 is THE PRODUCT ITSELF, ALONE - the main body without its
   accessories, complete, from the front or a three-quarter angle. Slot 2: the product alone from another angle.
   Each angle once: never two whole views from the same angle. A photo of the product WITH its accessories is not a
   whole view when the product alone is available - the accessories have their own slot. For a set of equal pieces: all
   pieces together, then all pieces laid out apart. Prefer photos on plain white; a photo on a coloured or graphic
   background is a last choice (code cuts its background away).
B. CONTROLS (kind "controls", required if the product has any button, switch, dial, display or printed label): ONE
   close-up showing ALL controls with every label, cut from the cleanest REAL photograph of the whole product (never
   from a photo marked GENERATED; the background does not matter, code removes it). Add a second
   "controls" slot only for a small control group that is easy to miss (an indicator light, a single button apart from
   the rest). Never more than 2 controls slots, never two cuts of the same controls. Only if no photograph can give
   it: a generated close-up (say how many buttons, their shape, order and icons, as the product info describes them).
C. STATES (kind "state", required if the product opens): the product OPEN - one slot for the whole product open (door
   open, lid up, part pulled out), and one for a closer view of the open area.
D. LOGO (kind "detail", caption starting with "LOGO", required if the product carries a brand name, logo or model
   name): ONE close-up of it, cut from a real photograph of the whole product (photo + crop).
   DETAILS (kind "detail", at most 2 more): what makes this product recognisable and a model would get wrong - a
   light ring, a display, an unusual vent or outlet. If the only photo of it has added text or props, ask for a
   generated clean view from that photo. Never a plain surface, a base or feet.
E. ACCESSORIES (kind "part"): {parts_rule}
   Accessories matter less than the product: they are drawn smaller than it.
F. IN USE (kind "use", required for anything held, worn or operated by hand): ONE panel of the product in a hand or
   with a person, used the way it is meant to be, at true scale. This tells a model how big the product is and how
   it is held. A photograph of a hand or person using it is right for this slot (it does not have to be "clean"),
   as long as it has no added text or graphics; otherwise ask for a generated view (say which hand, which grip).
Every button, switch and indicator named in the product info must be visible in a controls or detail slot.
Fewer, larger panels are better: 6 to 9 slots in total.

FILL EACH SLOT FROM ITS OWN CATEGORY FIRST. The listing gives every photo's category. Controls: a "controls close-up"
photo. Logo: a "logo close-up" photo. In use: an "in hand" photo, else an "in use" photo. Accessories: a "part alone"
photo. Whole: "front", "3/4", "side" photos. Only when a category has no usable photo, cut the close-up from a photo
of the whole product, and only when that is impossible too, ask for a generated view.

HOW TO FILL A SLOT - exactly one of:
- "photo": the number of a photo that shows exactly this, whole and uncut. For kinds whole / state / part the photo must
  be clean (no food, props, hands, added text). Never give a crop for these: code frames the product.
- "photo" + "crop": ONLY for kind "controls" (and "detail") cut from a clean photograph of the whole product:
  fractions [x0, y0, x1, y1] of that photo, a little loose, containing every control named.
- "generate": {{"what": "one sentence, exactly what to show", "refs": [1 to 3 photo numbers that show it best],
  "wide": false}} - for kind "controls" only when no photograph shows them; not allowed in real mode.
- "missing": true - when none of the above is possible. A missing slot is acceptable; a wrong picture is not.
Never use two slots for the same thing. Never crop an infographic or a photo of several parts.
When several photos show the same thing equally well, take the one with the most pixels.
Captions: UPPER CASE, at most 38 characters, the view or the part and its state.

Answer one JSON object and nothing else:
{{"slots": [{{"kind": "whole", "caption": "FRONT", "photo": 3}},
            {{"kind": "controls", "caption": "CONTROL PANEL - 4 DIALS", "photo": 3, "crop": [0.1, 0.2, 0.9, 0.4]}},
            {{"kind": "state", "caption": "DOOR OPEN", "generate": {{"what": "...", "refs": [3, 7], "wide": false}}}},
            {{"kind": "part", "caption": "CRUMB TRAY", "missing": true}}]}}"""

KIND_ROLE = {"whole": "hero", "controls": "view", "state": "view", "use": "view", "part": "detail", "detail": "detail"}
WHOLE_VIEWS = ("front", "3/4 left", "3/4 right", "side", "top", "set together")


LOCATE = """This is a photograph of {name} with a numbered grid drawn over it ({cols} columns, {rows} rows; the number
is in the top-left corner of each cell). Find: {what}.
Which cells does it sit in? One thing: one group of cells. Things far apart on the product (a button on the front, a
switch on the side): one group per place, at most {n} groups. Give only cells where it is really visible; a group is
{size}.
Answer one JSON object: {{"groups": [[12, 13], [27]]}}, or {{"groups": []}} if it is not clearly visible in this photograph."""


def _product_box(im, tol=16):
    """Where the product is in a photo on a plain background (the whole frame otherwise), in pixels."""
    rgb = im.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, rgb.getpixel((2, 2)))).convert("L")
    box = diff.point(lambda v: 255 if v > tol else 0).getbbox()
    return box if box and (box[2] - box[0]) > 0.2 * im.width and (box[3] - box[1]) > 0.2 * im.height else (0, 0, im.width, im.height)


def locate_in_photo(call, folder, photos, name, what, model, min_px=130, grow=0.3, max_share=0.2, n=2, cols=6, rows=7, max_cells=6):
    """Find a named thing (the controls, the logo, a detail) in a real photograph of the whole product, so close-ups
    can be cut from real pixels. An image model asked for a close-up from whole-product references returns the whole
    product again, so close-ups are cut, not generated (first seen on a hair dryer, 2026-11-11).

    The model is not asked for coordinates: its boxes are 5-10 % off, which misses a small button altogether. A
    numbered grid is drawn over the product and it names the cells - a choice, not a measurement. Each group of cells
    becomes one tight cut; a cut that covers a large share of the photo is refused (that is the whole view again).
    Returns (photo, [boxes as fractions]) or (None, [])."""
    for rel in photos:
        try:
            im = Image.open(os.path.join(folder, rel)).convert("RGB")
            px0, py0, px1, py1 = _product_box(im)
            cw, ch = (px1 - px0) / cols, (py1 - py0) / rows
            view = im.crop((px0, py0, px1, py1))
            scale = 1500 / max(view.size)
            view = view.resize((max(1, round(view.width * scale)), max(1, round(view.height * scale))), Image.LANCZOS)
            d = ImageDraw.Draw(view)
            f = font(max(16, view.width // 46))
            for c in range(1, cols):
                d.line([(c * view.width / cols, 0), (c * view.width / cols, view.height)], fill=(255, 40, 40), width=2)
            for r in range(1, rows):
                d.line([(0, r * view.height / rows), (view.width, r * view.height / rows)], fill=(255, 40, 40), width=2)
            for r in range(rows):
                for c in range(cols):
                    x, y = c * view.width / cols + 5, r * view.height / rows + 3
                    label = str(r * cols + c + 1)
                    d.rectangle([x - 2, y - 1, x + f.getlength(label) + 3, y + f.size + 3], fill=(255, 255, 255))
                    d.text((x, y), label, fill=(220, 0, 0), font=f)
            got = parse_json(llm(call, model, LOCATE.format(name=name, what=what, n=n, cols=cols, rows=rows,
                                                          size="1 to 4 neighbouring cells" if max_cells <= 6 else "every cell it covers"),
                                 [data_url(view, 1500, 88)], max_tokens=600))
        except Exception as e:
            logger.warning("locate failed on %s: %s", rel, e)
            continue
        boxes = []
        for group in (got.get("groups") or [])[:n]:
            cells = [int(v) - 1 for v in group if isinstance(v, (int, float)) and 1 <= int(v) <= cols * rows][:max_cells]
            if not cells:
                continue
            c0, c1 = min(v % cols for v in cells), max(v % cols for v in cells) + 1
            r0, r1 = min(v // cols for v in cells), max(v // cols for v in cells) + 1
            x0, y0, x1, y1 = px0 + c0 * cw, py0 + r0 * ch, px0 + c1 * cw, py0 + r1 * ch
            gw, gh = grow * (x1 - x0) / max(1, c1 - c0), grow * (y1 - y0) / max(1, r1 - r0)     # a margin of part of one cell
            x0, y0, x1, y1 = max(0, x0 - gw), max(0, y0 - gh), min(im.width, x1 + gw), min(im.height, y1 + gh)
            if (x1 - x0) * (y1 - y0) <= max_share * im.width * im.height and min(x1 - x0, y1 - y0) >= min_px:
                boxes.append([x0 / im.width, y0 / im.height, x1 / im.width, y1 / im.height])
        if boxes:
            return rel, boxes
    return None, []


def real_whole_photos(folder, tags, keys):
    """Clean real photographs that show the whole product: the product alone first (a close-up is cut from the
    PRODUCT ALONE category; in a photo of the product with its accessories it is smaller and other things get into
    the cut), then the largest."""
    ok = [k for k in keys if tags.get(k, {}).get("view") in WHOLE_VIEWS and tags[k].get("clean") and not tags[k].get("generated")
          and not k.startswith("generated/") and tags[k].get("same_product") is not False]
    return sorted(ok, key=lambda k: (tags[k].get("view") == "set together", -min(Image.open(os.path.join(folder, k)).size)))


# One set of rules for every product; a small PROFILE says which conditional parts apply. Rules written per category
# (one for hair dryers, one for ovens ...) would not scale to the next product and would end up contradicting each
# other. The profile is about traits, not names: a coffee maker with a carafe shares more with an oven than with a
# capsule machine. It is decided once, saved in product.json, and read by the planner, the gates and the final QA -
# before it existed the QA asked every sheet for "the product in a hand", which an oven can never satisfy.
PROFILE_KEYS = ("handheld", "opens", "has_controls", "has_accessories", "is_set")
PROFILE = """PRODUCT: {name} ({category}).
{info}

Answer one JSON object, true or false for each, about the product itself:
{{"handheld": it is held in the hand while it works (a hair dryer, a drill, a razor) - not merely carried, opened or touched,
  "opens": it has a door, lid or drawer the user opens, or an inside the user looks into,
  "has_controls": it has buttons, switches, dials, a display or printed control labels on it,
  "has_accessories": it comes with separate accessories or removable parts,
  "is_set": it IS a set of several pieces of the same kind (a cookware set, a knife set) with no single main body}}"""


def product_profile(call, folder, product, info, model):
    """The product's profile: read from product.json, or decided now and saved there. None if it cannot be decided -
    then nothing is switched off and every rule applies as before."""
    saved = product.get("profile")
    if isinstance(saved, dict) and all(k in saved for k in PROFILE_KEYS):
        return saved
    try:
        got = parse_json(llm(call, model, PROFILE.format(name=f'{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")}'.strip(),
                                                         category=product.get("category", ""), info=(info or "")[:3000]), max_tokens=300))
        profile = {k: bool(got.get(k)) for k in PROFILE_KEYS}
    except Exception as e:
        logger.warning("profile not decided: %s", e)
        return None
    product["profile"] = profile
    try:
        json.dump(product, open(os.path.join(folder, "product.json"), "w", encoding="utf-8"), indent=1)
    except Exception as e:
        logger.warning("profile not saved: %s", e)
    return profile


def applies(profile, key):
    """Whether a conditional rule applies. With no profile every rule applies."""
    return True if not profile or key not in profile else bool(profile[key])


def profile_text(profile):
    if not profile:
        return "(not decided: treat every slot as applicable)"
    yes = lambda k: "yes" if profile.get(k) else "no"                           # noqa: E731
    return (f"held in the hand while it works: {yes('handheld')}; opens / has an inside: {yes('opens')}; has controls: {yes('has_controls')}; "
            f"comes with accessories: {yes('has_accessories')}; is itself a set of pieces: {yes('is_set')}")


def candidates(tags, mode="mixed"):
    order = {"hero": 0, "view": 1, "detail": 2, "reference": 3}
    keys = [k for k, v in tags.items() if v.get("use") in order and v.get("same_product") is not False
            and not (mode == "real" and v.get("generated"))]
    first = ("controls close-up", "logo close-up", "in hand")          # scarce categories are never pushed out of the list
    return sorted(keys, key=lambda k: (tags[k].get("view") not in first, order[tags[k]["use"]], not tags[k].get("clean"), k))[:36]


def plan_sheet(call, folder, product, tags, info, mode, max_gen, model, include_parts=False, profile=None):
    """Ask the LLM to fill the slots, then enforce in code what it may and may not do. Returns (slots, notes)."""
    cands = candidates(tags, mode)
    listing = "\n".join(f'{i + 1}) {os.path.basename(k)} - {"GENERATED, " if tags[k].get("generated") else ""}{tags[k].get("view")}, '
                        f'{tags[k].get("state") or "no state"}, background {tags[k].get("background")}, '
                        f'{"clean" if tags[k].get("clean") else "NOT clean"}, {"x".join(str(v) for v in Image.open(os.path.join(folder, k)).size)} px: '
                        f'{tags[k].get("desc", "")}' for i, k in enumerate(cands))
    plan = parse_json(llm(call, model, PLAN.format(name=f'{product.get("brand", "")} {product.get("model", "")}'.strip(),
                                                   category=product.get("category", ""), info=info, listing=listing, mode=mode, max_gen=max_gen,
                                                   profile=profile_text(profile),
                                                   parts_rule=("one slot per accessory, removable part or piece of a set, each alone." if include_parts else
                                                               "ONE slot showing all accessories / removable parts together, WITHOUT the product itself "
                                                               "(a clean photo of them; if the only photo that shows them also shows the product, give "
                                                               "that photo: code cuts the product away). "
                                                               "If the product IS a set of pieces: at most 3 slots, for the pieces with the most to get wrong.")),
                          [data_url(contact_sheet(folder, cands), 2400, 85)], max_tokens=5000))
    if isinstance(plan, list):                      # the slots given bare, without the wrapping object
        plan = {"slots": [x for x in plan if isinstance(x, dict)]}
    slots, notes, seen, new_gen, n_controls = [], [], set(), 0, 0
    whole_refs = real_whole_photos(folder, tags, cands)[:2]
    name = f'{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")}'.strip()
    for sl in plan.get("slots", []):
        kind = sl.get("kind") if sl.get("kind") in KIND_ROLE else "detail"
        cap = re.sub(r"[–—]", "-", str(sl.get("caption", ""))).upper()[:40]
        off = ("it has no controls" if kind == "controls" and not applies(profile, "has_controls") else
               "it does not open" if kind == "state" and not applies(profile, "opens") else
               "it has no accessories" if kind == "part" and not (applies(profile, "has_accessories") or applies(profile, "is_set")) else "")
        if off:
            notes.append(f"{cap}: not planned - the product profile says {off}")
            continue
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
            # any background: a coloured one is cut away before the close-up is cut (prepare_panel)
            plain_whole = t.get("clean") and t.get("view") in WHOLE_VIEWS and not t.get("generated")
            use_photo = kind == "use" and t.get("view") in ("in use", "in hand") and t.get("background") != "graphic" \
                and t.get("banner", "none") in ("none", None)
            closeup_photo = kind in ("controls", "detail") and t.get("view") in ("controls close-up", "logo close-up", "detail close-up") \
                and t.get("banner", "none") != "all over"
            if crop and (crop[2] - crop[0]) * (crop[3] - crop[1]) > (0.45 if kind == "controls" else 0.3):
                notes.append(f"{cap}: crop refused (it covers most of the photo: that is the whole view again, not a close-up)")
                crop, key = None, None
            if crop and not (kind in ("controls", "detail") and plain_whole):
                notes.append(f"{cap}: crop refused (a close-up may only be cut from a clean whole-product photograph)")
                crop = None
                if kind == "controls":
                    key = None
            if key and kind in ("whole", "state", "part", "use", "detail") and not t.get("clean") and not crop and not use_photo \
                    and not closeup_photo:
                notes.append(f"{cap}: photo refused (not clean)")
                if mode != "real" and not gen:
                    gen = {"what": (f"{cap.lower()} - one hand holding the product as in the reference photo" if kind == "use" else
                                    f"{cap.lower()} - the same state as in the reference photo, but with nothing else in the picture"),
                           "refs": [int(n)], "wide": False}
                    notes[-1] += " - a clean view will be generated from it"
                key = None
            if key and kind == "controls":
                n_controls += 1
                if n_controls > 2:
                    notes.append(f"{cap}: dropped (at most 2 control close-ups)")
                    continue
            if key and kind == "controls" and not crop and not closeup_photo:
                notes.append(f"{cap}: photo refused (not a clean control close-up and no crop given)")
                key = None
            if key and (key, tuple(crop or ())) not in seen:
                seen.add((key, tuple(crop or ())))
                out.update(image=key, crop=crop, generated=bool(t.get("generated")))
        if "image" not in out and (kind == "controls" or cap.startswith("LOGO")):
            # the slot's own category first: a real close-up photograph beats any cut
            key = best_of(folder, tags, "controls close-up" if kind == "controls" else "logo close-up", {k for k, _c in seen})
            if key:
                n_controls += kind == "controls"
                gen = None
                seen.add((key, ()))
                out.update(image=key, crop=None, generated=False)
                notes.append(f"{cap}: taken from the close-up photo {os.path.basename(key)}")
        if "image" not in out and kind == "controls" and whole_refs and n_controls < 2:
            # controls are mandatory, and a close-up is cut from a real photograph, never generated: find them in the
            # largest real photo of the whole product
            key, boxes = locate_in_photo(call, folder, whole_refs, name, f"all the controls ({cap.lower()}): every button, switch, dial, "
                                         "display and indicator light, with their labels", model)
            if key:
                n_controls += 1
                gen = None
                out.update(image=key, crop=boxes[0], generated=False, crop_fitted=True, **({"crops": boxes} if len(boxes) > 1 else {}))
                notes.append(f"{cap}: located in {os.path.basename(key)} and cut from it")
        if "image" not in out and gen and mode != "real" and new_gen < max_gen:
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


def gate_report(slots, built, rows, size, profile=None, height=None):
    """What the finished sheet covers, as plain facts. `fatal` lists gates that must stop delivery."""
    area = lambda r: r["box"][2] * r["box"][3]                                           # noqa: E731
    hero = sum(area(r) for r in rows if r["role"] == "hero") / float(size * (height or size))
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
    if any(area(r) > 0.8 * biggest_hero for r in rows if r.get("kind") == "part"):
        fatal.append("G8: an accessory is drawn as large as the product itself")
    gone = [s for s in slots if s.get("missing") or s.get("dropped")]
    needed = [k for k, trait in (("controls", "has_controls"), ("use", "handheld")) if applies(profile, trait)]
    required = [s["caption"] for s in gone if s["kind"] in needed
                and not any(b.get("kind") == s["kind"] for b in built)]
    required += [s["caption"] for s in gone if s["caption"].startswith("LOGO")
                 and not any(re.search(r"LOGO|BADGE|BRAND", b.get("caption", "")) for b in built)]
    if required:
        fatal.append("G9: a required panel is missing: " + ", ".join(required))
    return {"hero_area": round(hero, 2), "panels": len(rows), "missing": [s["caption"] for s in gone],
            "required_missing": required, "fatal": fatal}


# ----------------------------------------------------------------------------- sheet: clean, generate

GENERATE = ("Product reference photo for a catalogue. Show: {what}. "
            "ONE single photograph showing one view of one thing: never a collage, a grid, several views, zoom bubbles, callout circles, lines or labels. The references may show more than is asked for (accessories, other views, a room): show ONLY what is asked. "
            "The reference images show the real product: reproduce it EXACTLY - same shape, proportions, materials, colours, "
            "every button, dial, display, printed label, logo, handle and marking, in the same places. Do not add, remove, "
            "restyle or relabel anything. No food, no liquid, no hands, no people, no props, no text overlay. Plain pure white "
            "background, soft even studio light, a soft neutral shadow under the product, the whole subject inside the frame "
            "with a small margin.")


GENERATE_USE = ("Product reference photo for a catalogue. Show: {what}. "
                "ONE single photograph showing one view of one thing: never a collage, a grid, several views, zoom bubbles, callout circles, lines or labels. The references may show more than is asked for (accessories, other views, a room): show ONLY what is asked. One adult hand and forearm hold the product the "
                "way it is really used, fingers on the grip, at true scale: the product is neither enlarged nor shrunk next to "
                "the hand. The reference images show the real product: reproduce it EXACTLY - same shape, proportions, "
                "materials, colours, every button, display, printed label, logo and marking, in the same places. Do not add, "
                "remove, restyle or relabel anything. No face, no props, no text overlay. Plain pure white background, soft "
                "even studio light, the whole product and the hand inside the frame with a small margin.")


def plain_background(im, light=226, share=0.88):
    """True when the picture's border is plain white or near white - what every panel of a sheet must sit on."""
    px = im.convert("RGB").resize((96, 96)).load()
    border = [px[x, y] for x in range(96) for y in (0, 1, 94, 95)] + [px[x, y] for y in range(96) for x in (0, 1, 94, 95)]
    return sum(1 for p in border if min(p) >= light and max(p) - min(p) <= 20) >= share * len(border)


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


def isolate(call, im, trim=True):
    """Background removal: the product's real pixels on white, trimmed to the product (trim=False keeps the frame)."""
    res = call(BG_REMOVE, {"image_url": data_url(im, quality=95)})
    cut = Image.open(io.BytesIO(fetch(res["image"]["url"], timeout=120))).convert("RGBA")
    if cut.size != im.size:
        cut = cut.resize(im.size, Image.LANCZOS)
    white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    white.alpha_composite(cut)
    box = cut.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    return white.convert("RGB").crop(box) if (box and trim) else white.convert("RGB")


def isolated(call, folder, rel, im):
    """The photo with its background removed, same frame, kept in clean/ so it is paid for once."""
    path = os.path.join(folder, "clean", os.path.splitext(os.path.basename(rel))[0] + "_white.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    out = isolate(call, im, trim=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.save(path)
    return out


def generate_view(call, folder, what, refs, wide=False, kind=None, feedback=""):
    size = {"width": 2048, "height": 1536} if wide else {"width": 2048, "height": 2048}
    urls = [data_url(Image.open(os.path.join(folder, r)), 1536, 90) for r in refs]
    prompt = (GENERATE_USE if kind == "use" else GENERATE).format(what=what)
    if feedback:
        prompt += f" An earlier attempt was rejected for this reason, fix exactly that: {feedback[:300]}"
    res = call(IMAGE_EDIT, {"prompt": prompt, "image_urls": urls, "image_size": size, "quality": "high",
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
    boxes = panel.get("crops") or ([panel["crop"]] if panel.get("crop") else [])
    tiles = []
    for crop in boxes:
        x0, y0, x1, y1 = [min(1.0, max(0.0, float(c))) for c in crop]
        if x1 - x0 > 0.01 and y1 - y0 > 0.01:
            if isolate_busy and not tiles and not plain_background(im):       # a close-up cut from a photo on colour: white first
                try:
                    im = isolated(call, folder, panel["image"], im)
                except Exception as e:
                    logger.warning("background removal failed for %s: %s", panel["image"], e)
            px, py = (0.0, 0.0) if (panel.get("crop_exact") or panel.get("crop_fitted")) else (0.02, (y1 - y0) * 0.08)
            x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(1, x1 + px), min(1, y1 + py)
            tiles.append(im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height))))
    if len(tiles) == 1:
        return tiles[0]
    if tiles:                                           # several groups of one thing, side by side at one height
        h = max(t.height for t in tiles)
        tiles = [t.resize((max(1, round(t.width * h / t.height)), h), Image.LANCZOS) for t in tiles]
        gap = max(12, h // 14)
        out = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), h), (255, 255, 255))
        x = 0
        for t in tiles:
            out.paste(t, (x, 0))
            x += t.width + gap
        return out
    im = banner_crop(im, t.get("banner", "none"))
    if plain_background(im):
        return trim_plain(im)
    # anything else - a flat colour, a gradient, a room - is cut away: the product sits on white in every panel.
    # In the in-use panel the hand or person stays and the room goes. Not for pictures with no outline to cut.
    if isolate_busy and t.get("view") not in ("interior", "detail close-up", "controls close-up", "logo close-up", "infographic"):
        try:
            return trim_plain(isolated(call, folder, panel["image"], im))
        except Exception as e:
            logger.warning("background removal failed for %s: %s", panel["image"], e)
    return im


VERIFY = """These {n} pictures are panels of a product reference sheet for: {name}. Their captions, in order: {caps}
For each panel answer whether the caption is true for what the picture shows, and whether the picture is usable:
not usable = the named part is cut off or missing, the picture shows a different part than the caption says, it shows
more than the caption says (other parts, accessories, several views or a collage in one picture), or it contains added
marketing text, badges, zoom bubbles or callout dots. A caption that names a close-up or one part (controls, buttons,
an outlet, a grille, a badge, a display) is NOT true for a picture of the whole product seen from afar: the named thing
must be the clear subject of the picture and large enough to read. A hand or a person is right in a panel captioned as in use / held. Answer one JSON object {{"panels": [{{"n": 1, "ok": true, "why": ""}}]}}."""


# ----------------------------------------------------------------------------- composition: what the product is made of, counted

SPEC = """Write the COMPOSITION of {name}: what it is made of, counted. It is used to keep every picture of the product
consistent, so count from the photos (they are the truth) and use the text only for names, labels and sizes.

PRODUCT INFO
{info}

NOTES FROM PRODUCT PAGES
{notes}

Answer one JSON object:
{{"summary": "one sentence listing the counted parts, e.g. 4 dials, 1 power light, 1 light button, 1 glass door with a bar handle, 4 feet",
  "components": [{{"name": "", "count": 1, "where": "where it sits on the product", "label": "the text printed on or next to it, exactly, or empty"}}],
  "accessories": [{{"name": "", "count": 1}}],
  "materials": ["material and finish of each main part"],
  "dimensions": {{"width": "", "depth": "", "height": "", "weight": "", "capacity": "", "source": "where the numbers come from, or: not stated"}}}}
At most 12 components, the ones a picture could get wrong first. Write a dimension only if it is stated in the notes or
the product info; never estimate one - leave it empty."""


def product_spec(call, folder, product, info, tags, model, notes=""):
    """The product's composition, counted: read from product_spec.json or written now. None if it cannot be written."""
    path = os.path.join(folder, "product_spec.json")
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    order = {"hero": 0, "detail": 1, "view": 2}
    picks = sorted((k for k, v in tags.items() if v.get("use") in order and v.get("same_product") is not False and not k.startswith("generated/")),
                   key=lambda k: (order[tags[k]["use"]], not tags[k].get("clean")))[:5]
    try:
        spec = parse_json(llm(call, model, SPEC.format(name=f'{product.get("brand", "")} {product.get("model", "")} {product.get("name", "")}'.strip(),
                                                       info=(info or "(none)")[:3500], notes=(notes or "(none)")[:3500]),
                              [data_url(Image.open(os.path.join(folder, k)), 1536, 88) for k in picks], max_tokens=2000), want=dict)
    except Exception as e:
        logger.warning("composition not written: %s", e)
        return None
    json.dump(spec, open(path, "w", encoding="utf-8"), indent=1)
    return spec


def _wrap(d, text, f, max_w):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if d.textlength((cur + " " + w).strip(), font=f) <= max_w:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def spec_panel(spec, width=2800, columns=2):
    """The composition as a panel of text: counted parts with what is printed on them, accessories, materials, sizes.
    Text sits in a panel of its own, never over a photograph, so an image model does not read it as printed on the
    product. Two columns: a tall narrow column of text forced every other panel of the sheet into one thin row."""
    head, body = font(46), font(38, bold=False)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    blocks = []
    if spec.get("summary"):
        blocks.append(("MADE OF", [spec["summary"]]))
    comps = [f'{c.get("count", 1)} x {c.get("name", "")}' + (f' - reads: {c["label"]}' if c.get("label") else "") for c in (spec.get("components") or [])[:12] if c.get("name")]
    if comps:
        blocks.append(("PARTS, COUNTED", comps))
    acc = [f'{a.get("count", 1)} x {a.get("name", "")}' for a in (spec.get("accessories") or [])[:8] if a.get("name")]
    if acc:
        blocks.append(("ACCESSORIES", acc))
    if spec.get("materials"):
        blocks.append(("MATERIALS", [str(m) for m in spec["materials"][:5]]))
    dims = spec.get("dimensions") or {}
    sizes = [f"{k}: {dims[k]}" for k in ("width", "depth", "height", "weight", "capacity") if dims.get(k)]
    if sizes:
        blocks.append(("SIZE (" + str(dims.get("source") or "stated by the maker")[:60] + ")", sizes))
    if not blocks:
        return None
    col_w = (width - 80 * (columns + 1)) // columns
    laid = []                                            # per block: its lines (text, font, indent) and its height
    for title, items in blocks:
        lines = [(title, head, 0)]
        for it in items:
            for i, ln in enumerate(_wrap(probe, it, body, col_w - 90)):
                lines.append((ln, body, 0 if i == 0 else 44))
        laid.append((lines, 58 * len(lines) + 30))
    total = sum(h for _l, h in laid)
    cols, cur, run = [[]], 0, 0
    for lines, h in laid:                                # fill a column to about its share, then start the next
        if cols[cur] and run + h / 2 > total / columns and cur < columns - 1:
            cols.append([])
            cur, run = cur + 1, 0
        cols[cur].append((lines, h))
        run += h
    height = 80 + max(sum(h for _l, h in c) for c in cols)
    im = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for ci, col in enumerate(cols):
        x0, y = 80 + ci * (col_w + 80), 40
        for lines, _h in col:
            for text, f, indent in lines:
                if f is body and not indent:
                    d.ellipse((x0 + 2, y + 16, x0 + 16, y + 30), fill=(20, 20, 20))
                d.text((x0 + (40 + indent if f is body else 0), y), text, fill=(20, 20, 20) if f is head else (40, 40, 40), font=f)
                y += 58
            y += 30
    return im


def dimension_panel(im, dims):
    """A whole view with its stated width and height drawn as dimension lines (never estimated: only what the maker states)."""
    w_txt, h_txt = (dims or {}).get("width"), (dims or {}).get("height")
    if not (w_txt or h_txt):
        return None
    im = im.convert("RGB")
    scale = 1400 / max(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    m, f, ink = 190, font(52), (20, 20, 20)
    out = Image.new("RGB", (im.width + 2 * m, im.height + 2 * m), (255, 255, 255))
    out.paste(im, (m, m))
    d = ImageDraw.Draw(out)
    if w_txt:
        y = m + im.height + 70
        d.line([(m, y), (m + im.width, y)], fill=ink, width=6)
        for x in (m, m + im.width):
            d.line([(x, y - 26), (x, y + 26)], fill=ink, width=6)
        d.text((m + (im.width - d.textlength(str(w_txt), font=f)) / 2, y + 30), str(w_txt), fill=ink, font=f)
    if h_txt:
        x = m + im.width + 70
        d.line([(x, m), (x, m + im.height)], fill=ink, width=6)
        for y in (m, m + im.height):
            d.line([(x - 26, y), (x + 26, y)], fill=ink, width=6)
        label = Image.new("RGB", (int(d.textlength(str(h_txt), font=f)) + 20, 70), (255, 255, 255))
        ImageDraw.Draw(label).text((10, 4), str(h_txt), fill=ink, font=f)
        label = label.rotate(90, expand=True)
        out.paste(label, (x + 28, m + (im.height - label.height) // 2))
    if (dims or {}).get("depth"):
        d.text((m, 60), "depth: " + str(dims["depth"]), fill=ink, font=f)
    return out


def _shape(im):
    """The outline of what a panel shows, as a small mask: two whole views with the same outline are the same angle."""
    im = im.convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).convert("L").point(lambda v: 255 if v > 24 else 0)
    box = diff.getbbox()
    if not box:
        return None
    return [v > 0 for v in diff.crop(box).resize((48, 48)).getdata()]


def same_shape(a, b):
    sa, sb = _shape(a), _shape(b)
    if not sa or not sb:
        return False
    union = sum(1 for x, y in zip(sa, sb) if x or y)
    return union > 0 and sum(1 for x, y in zip(sa, sb) if x and y) / union > 0.9


def _box_of(slot):
    boxes = slot.get("crops") or ([slot["crop"]] if slot.get("crop") else [])
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _look(im):
    import numpy as np
    arr = np.asarray(im.convert("L").resize((48, 48)), dtype=np.float32)
    arr = arr - arr.mean()
    return arr / (float(np.linalg.norm(arr)) or 1.0)


def drop_repeats(built, slots):
    """No view twice (owner's rule, 2026-11-11: the same front view was on the sheet four times - once whole and three
    times as loose "close-ups" cut from it). Panels are taken in the order of the hierarchy; a later panel is dropped
    when it repeats one already kept:
    - it is cut from a photo that is already shown whole, and the cut covers more than a quarter of that photo;
    - it is cut from the same photo as another cut and they share most of their area;
    - it simply looks the same as a kept panel.
    Returns (kept, [(panel, reason)])."""
    order = sorted(range(len(built)), key=lambda i: KIND_RANK.get(built[i].get("kind"), 3))
    kept, gone = [], []
    for i in order:
        b = built[i]
        box = _box_of(slots[b["slot"]]) if 0 <= b.get("slot", -1) < len(slots) else None
        b["_box"], b["_look"] = box, _look(b["im"])
        why = ""
        for k in kept:
            same_photo = k["source"] == b["source"] and not b.get("generated")
            limit = 0.45 if b.get("kind") == "controls" else 0.25
            if same_photo and box and not k["_box"] and (box[2] - box[0]) * (box[3] - box[1]) > limit:
                why = f'shows again most of the photo already used whole in "{k["caption"]}"'
            elif same_photo and box and k["_box"]:
                kb = k["_box"]
                inter = max(0.0, min(box[2], kb[2]) - max(box[0], kb[0])) * max(0.0, min(box[3], kb[3]) - max(box[1], kb[1]))
                small = min((box[2] - box[0]) * (box[3] - box[1]), (kb[2] - kb[0]) * (kb[3] - kb[1]))
                if small and inter / small > 0.5:
                    why = f'shows the same area of the same photo as "{k["caption"]}"'
            if not why and float((b["_look"] * k["_look"]).sum()) > 0.9:
                why = f'looks the same as "{k["caption"]}"'
            if not why and b.get("kind") == "whole" and k.get("kind") == "whole" and same_shape(b["im"], k["im"]):
                why = f'is the same angle as "{k["caption"]}"'
            if why:
                break
        if why:
            gone.append((b, why))
        else:
            kept.append(b)
    kept_ids = {id(b) for b in kept}
    return [b for b in built if id(b) in kept_ids], gone


def verify_panels(call, name, built, model, keep_min=3):
    """Drop panels whose picture does not match its caption. Returns (kept, dropped captions with the reason).
    keep_min=0 checks even a single panel (used when one panel is made again)."""
    if len(built) < 2 and keep_min:
        return built, []
    caps = "; ".join(f'{i + 1}) {b["caption"].replace(" (GENERATED)", "")}' for i, b in enumerate(built))
    try:
        got = parse_json(llm(call, model, VERIFY.format(n=len(built), name=name, caps=caps),
                             [data_url(b["im"], 768, 80) for b in built], max_tokens=3000))["panels"]
    except Exception as e:
        logger.warning("panel check failed, keeping every panel: %s", e)
        return built, []
    bad = {int(g["n"]) - 1: g.get("why", "") for g in got if g.get("ok") is False}
    bad = {i: why for i, why in bad.items() if 0 <= i < len(built)
           and not (built[i].get("kind") == "whole" and not built[i].get("generated"))}
    kept = [b for i, b in enumerate(built) if i not in bad]
    if len(kept) < keep_min:                            # never let the check empty the sheet
        return built, []
    for i, why in bad.items():
        if 0 <= i < len(built):
            built[i]["why"] = why
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


# the hierarchy of a sheet: the product itself, then its controls and details, then accessories, then the hand / person
KIND_RANK = {"whole": 0, "controls": 1, "state": 2, "detail": 3, "spec": 3.5, "part": 4, "use": 5}
AREA_WEIGHT = {"whole": 1.6, "controls": 1.9, "state": 1.6, "detail": 1.2, "spec": 1.1, "part": 0.6, "use": 1.0}

QA = """You are the last check before a product reference sheet is delivered. PRODUCT: {name}. An image model will use
this sheet as its only reference of the product, so judge it strictly. Panel captions, in order: {caps}

What a good sheet is for THIS product, in order of importance:
{expect}
Judge only against this list: do not ask for a panel the list does not ask for.

{composition}The title and the one-line note under it are part of the sheet format, not a problem. So are a panel of
specification text and a panel with dimension lines. A panel marked (GENERATED) is
allowed. The in-use panel shows a hand or a person: that is what it is for.

Report only real problems, each with one of these codes:
coloured_background, product_cut, artefact (smear, leftover shadow, damaged cut-out, distorted shape), accessory_too_large,
no_controls_closeup, no_in_use, duplicate (two panels showing the same thing), wrong_caption, marketing_text,
empty_space (a large blank area), product_unclear (the product alone is not the largest, clearest thing on the sheet).
Answer one JSON object: {{"pass": true or false, "issues": [{{"panel": "the caption, or SHEET", "code": "...", "why": "one sentence"}}]}}"""


def qa_expect(profile):
    """What the final QA may ask of this product: the hierarchy, with the parts its profile switches off left out."""
    whole = ("All pieces of the set together, complete, large and clear, on plain white, then laid out apart."
             if profile and profile.get("is_set") else "The product itself, alone, complete, large and clear, on plain white.")
    lines = [whole + " No coloured or graphic background anywhere."]
    lines.append("Close-ups of its own details: its buttons, switches, displays, labels and logo." if applies(profile, "has_controls")
                 else "Close-ups of its logo and distinctive details. It has no controls: do not ask for a controls close-up.")
    if applies(profile, "opens"):
        lines.append("The product open, showing its inside.")
    if applies(profile, "has_accessories") and not (profile and profile.get("is_set")):
        lines.append("Its accessories - present, complete, and clearly smaller than the product.")
    lines.append("The product in a hand or with a person, showing its true size." if applies(profile, "handheld")
                 else "It is not held in the hand: a panel with a hand or a person is optional, do not ask for one.")
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))


def qa_sheet(call, name, sheet, captions, model, profile=None, composition=""):
    """A vision model judges the finished sheet as a whole. Returns {"pass": bool, "issues": [...]}; on failure of the
    call itself the sheet is not blocked and the answer says the check did not run."""
    try:
        got = parse_json(llm(call, model, QA.format(name=name, caps="; ".join(captions), expect=qa_expect(profile),
                                                    composition=(f"The product is made of: {composition}. A panel that shows a different number "
                                                                 "of these parts is an artefact.\n" if composition else "")),
                             [data_url(sheet, 2000, 85)], max_tokens=1500))
        off = {code for code, trait in (("no_in_use", "handheld"), ("no_controls_closeup", "has_controls")) if not applies(profile, trait)}
        issues = [i for i in got.get("issues", []) if isinstance(i, dict) and i.get("code") and i.get("code") not in off]
        return {"pass": bool(got.get("pass")) and not issues, "issues": issues}
    except Exception as e:
        logger.warning("sheet QA failed to run: %s", e)
        return {"pass": True, "issues": [], "not_run": str(e)[:200]}


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
    order = {"hero": 0, "view": 2, "detail": 3}
    items = sorted(panels, key=lambda q: KIND_RANK.get(q.get("kind"), order.get(q["role"], 3)))   # stable: plan order inside a kind
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
        if len(rows) > 1 or n > 3:                                                  # the product row owns at most half the sheet,
            nat = [min(h, 0.52 * (H - fixed)) if any(items[i]["role"] == "hero" for i in r) else h    # so its details stay readable
                   for h, r in zip(nat, rows)]
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
        if min(hts) < 0.5 * max(hts):
            score *= (min(hts) / (0.5 * max(hts))) ** 2                             # details are smaller than the product, not tiny
        if any(b > a * 1.05 for a, b in zip(hts, hts[1:])):
            score *= 0.3                                                            # rows follow the hierarchy: never taller further down
        if n > 1 and any(len(r) == 1 and asp[r[0]] * min(h, cap[r[0]]) < 0.6 * (W - 2 * pad) for r, h in zip(rows, hts)):
            score *= 0.35                                                           # one narrow picture alone in a row: white on both sides
        if any(len(r) == 1 and items[r[0]].get("kind") not in ("whole", "controls", "state") for r in rows) and n > 1:
            score *= 0.4                                                            # no accessory, detail or in-use panel gets a row to itself
        areas = {i: asp[i] * min(h, cap[i]) ** 2 for r, h in zip(rows, hts) for i in r}
        product = max([areas[i] for i in areas if items[i].get("kind") == "whole"] or [0])
        if product and any(areas[i] > 0.6 * product for i in areas if items[i].get("kind") == "part"):
            score *= 0.4                                                            # an accessory never rivals the product
        if product and any(areas[i] > 0.9 * product for i in areas if items[i].get("kind") in ("use", "detail")):
            score *= 0.7
        drawn = sum(asp[i] * min(h, cap[i]) ** 2 for r, h in zip(rows, hts) for i in r)
        cells = sum((W - gap * (len(r) - 1) - 2 * pad * len(r)) * h for r, h in zip(rows, hts)) + (H - fixed - sum(hts)) * W
        score *= (drawn / max(cells, 1.0)) ** 1.5                                   # white space is paid for
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


def compose(panels, title, subtitle, size=5000, margin=60, gap=30, cap_h=110, max_up=1.5, hero_weight=None, height=None):
    """panels: [{"im": PIL, "role": hero|view|detail, "caption": str}] -> (sheet, info rows). size is the width;
    height defaults to it (square), or is given for a 16:9 sheet."""
    bg, cell, ink, sub, line = (232, 232, 232), (255, 255, 255), (20, 20, 20), (90, 90, 90), (200, 200, 200)
    height = height or size
    sheet = Image.new("RGB", (size, height), bg)
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
    rows, slack = layout(panels, size - 2 * margin, height - margin - y, gap, cap_h, max_up, hero_weight=hero_weight)
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
