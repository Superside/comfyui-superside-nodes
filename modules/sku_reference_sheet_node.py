"""SKU Reference Sheet: every product view on one sheet, at one shared scale.

This replaces a chain of Normalize -> Detail Crop -> ImageStitch -> Resize that
could not produce a consistent sheet however it was tuned. Each Normalize sized
its own canvas from its own product, so the three views came out at three
different heights; ImageStitch then matched edges by resizing a whole panel,
and the frame ended up at a different apparent size in each one. Measured on
one catalogue SKU: the product was 2117 / 2105 / 2114 px wide across the views -
already consistent - but 754 / 808 / 891 px tall, so the canvases came out
897 / 961 / 1060 and the stitch rescaled them against each other.

Holding every view in one node is what fixes it: one pixels-per-unit is chosen
for all of them, so a millimetre of frame is the same number of pixels in every
panel, and each cell is trimmed to its own content instead of being padded to a
shared rectangle.

The detail strip carries the close-ups the downstream editing model cannot
resolve at working resolution. Its first slot can find the brand logo on its
own: Florence-2 proposes one candidate per view, and the vision model picks
which candidate actually carries a mark. That two-step matters because the logo
is in a different place on different brands - printed on the lens and readable
only from the front on one, an emblem on the temple visible only at an angle on
another - and Florence returns no confidence to choose by. Measured on two
brands, the pair picked the right view and read the mark on both.
"""

import json
import logging

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .base_node import API_KEY_INPUT_SPEC

logger = logging.getLogger(__name__)

VIEW_SLOTS = (
    ("front", "FRONT"),
    ("side", "SIDE"),
    ("three_quarter", "3/4"),
    ("three_quarter_additional", "3/4 ADDITIONAL"),
)

# The bridge and the temple-to-front joint sit in the same place on any pair of
# glasses, so they are cut from the product box rather than detected. That is not
# a shortcut: asked to ground "the bridge between the two lenses", Florence
# returned the whole frame on 3 of 5 frames (1806x643 out of a 2400x1200 photo on
# one), because a bridge is part of a continuous structure with no edge of its
# own. The same fractions were right on 5 of 5. The logo is the opposite case -
# its position moves by brand, so that one is detected.
# Fractions of the product box: (x1, y1, x2, y2).
# The joint region is right on 10 of 11 frames. It misses on wrap-around sports
# frames (an Oakley Sutro, a Flak), whose temple sits further back than the outer
# corner, so the crop lands on lens. Sliding the window inward by edge density
# was tried and rejected: a hinge is edge dense, but so is the bridge - more so,
# between nose pads, screws and pad arms - so the window drifted to the bridge
# and made two frames that had been right worse. Bounding the slide to the outer
# third did not stop the drift. The fixed fraction beat both, so a wrap frame is
# handled by overriding that slot through detail_boxes instead.
BRIDGE_REGION = (0.37, 0.00, 0.63, 0.72)
JOINT_REGION = (0.00, 0.00, 0.26, 0.62)

# Fonts are looked up by name so the sheet still renders on a machine that has
# none of them; PIL's default bitmap font is the last resort.
_FONT_CANDIDATES = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")


def _font(size):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


class SupersideSkuReferenceSheetNode:
    """Compose three or four product views plus a detail strip into one sheet."""

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "build_sheet"
    DISPLAY_NAME = "SKU Reference Sheet"
    DESCRIPTION = (
        "Lay three or four product views out on one sheet at a single shared "
        "scale, with a DETAILS strip of close-ups. Connect front (required) and "
        "any of side / 3-4 / 3-4 additional; the layout follows how many are "
        "wired. Every view is placed at the same pixels-per-unit, so the frame "
        "is the same physical size in every panel - which a chain of separate "
        "Normalize and Stitch nodes cannot do, because each one only ever sees "
        "one image. Set auto_logo to have the brand mark found and cropped "
        "without naming where it is."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": ("IMAGE",),
            },
            "optional": {
                "side": ("IMAGE",),
                "three_quarter": ("IMAGE",),
                "three_quarter_additional": ("IMAGE",),
                "max_long_side": ("INT", {
                    "default": 5000, "min": 512, "max": 8192, "step": 64,
                    "tooltip": "Longest side of the finished sheet, in pixels.",
                }),
                "margin_percent": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 25.0, "step": 0.5,
                    "tooltip": "Breathing room around the product inside each panel, "
                               "as a percentage of the panel's short side.",
                }),
                "gap_px": ("INT", {
                    "default": 14, "min": 0, "max": 120, "step": 2,
                    "tooltip": "Width of the dividers between panels. These exist so the "
                               "editing model reads the panels as separate photographs "
                               "rather than one blended image.",
                }),
                "background_threshold": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 80.0, "step": 1.0,
                    "tooltip": "How far a pixel must differ from the backdrop to count as "
                               "product. Raise it when a soft shadow is being picked up, "
                               "lower it when a pale frame is being cut off.",
                }),
                "detail_boxes": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Close-ups for the DETAILS strip, as a JSON list of "
                               "{view, x1, y1, x2, y2, caption}. view is 0-based over the "
                               "connected views; x1..y2 are 0-1 fractions of that view's "
                               "product box. Leave empty for no hand-picked details.",
                }),
                "auto_bridge": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Put a close-up of the bridge in the DETAILS strip, cut from "
                               "the centre of the front view. No API call - the bridge is in "
                               "the same place on every frame.",
                }),
                "auto_joint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Put a close-up of the joint between the lateral support and "
                               "the front in the DETAILS strip, cut from the outer upper "
                               "corner of a three-quarter view (or the side view). No API call.",
                }),
                "auto_logo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Find the brand logo and put it first in the DETAILS strip. "
                               "Florence-2 proposes one candidate per view and the vision "
                               "model picks the one carrying a mark, so the logo is found "
                               "whether it sits on the lens or on the temple. Needs api_key; "
                               "costs one detection per view plus one vision call, once per "
                               "SKU.",
                }),
                "layout": (["standard", "corner"], {
                    "default": "standard",
                    "tooltip": "standard: one DETAILS strip. corner: the strip becomes THE CORNER, "
                               "ENLARGED (taller, larger slots), and the close-ups captioned CORNER: "
                               "SIDE VIEW, TEMPLE..., BRIDGE... move to a full-width row under the sheet "
                               "(THE SIDE AND THE BRIDGE, ENLARGED). Built for an edit model that must "
                               "redraw the hinge corner and the temple from the product. With "
                               "detail_boxes empty, corner uses its own five close-ups (corner from the "
                               "front, the three-quarter view at the joint measured on this product, the "
                               "side; the whole temple; the bridge) and the auto_* crops are skipped.",
                }),
                "canvas": (["fit", "square"], {
                    "default": "fit",
                    "tooltip": "fit: the sheet is as tall as its content. square: a square document "
                               "(max_long_side x max_long_side, e.g. 5000 x 5000); in the corner layout "
                               "the side-and-bridge row grows to fill it, and any remaining space is "
                               "neutral grey. Nothing is stretched or cropped.",
                }),
                "api_key": API_KEY_INPUT_SPEC,
            },
        }

    # ------------------------------------------------------------------ pixels

    @staticmethod
    def _to_pil(image):
        arr = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    @staticmethod
    def _backdrop(arr):
        """Median of the four corner patches - the catalogue's own backdrop."""
        h, w = arr.shape[:2]
        p = max(4, min(h, w) // 40)
        corners = np.concatenate([
            arr[:p, :p].reshape(-1, 3), arr[:p, -p:].reshape(-1, 3),
            arr[-p:, :p].reshape(-1, 3), arr[-p:, -p:].reshape(-1, 3),
        ], axis=0)
        return np.median(corners, axis=0)

    def _product_box(self, img, threshold):
        arr = np.array(img)
        diff = np.abs(arr.astype(np.int16) - self._backdrop(arr).astype(np.int16)).max(axis=2)
        ys, xs = np.nonzero(diff > threshold)
        if len(xs) == 0:
            return (0, 0, img.width, img.height)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    # ------------------------------------------------------------------ logo

    def _find_logo(self, views, api_key, threshold):
        """Florence proposes a candidate per view; the vision model picks one.

        Returns (crop, caption) or None. Any failure here degrades to "no logo
        slot" rather than failing the sheet - a reference sheet without the logo
        close-up is still a usable sheet.
        """
        from .any_llm_vision_node import SupersideAnyLLMVisionNode
        from .florence_2_region_selector_node import SupersideFlorence2RegionSelectorNode

        finder = SupersideFlorence2RegionSelectorNode()
        reader = SupersideAnyLLMVisionNode()
        candidates = []
        for label, img in views:
            try:
                out = finder.select_region(
                    image=self._pil_to_tensor(img), api_key=api_key, region_type="object",
                    custom_text="brand logo", selection_mode="largest", padding_percent=0,
                    return_rect_mask=False, mask_blur_percent=0, detection_mode="auto",
                    upload_max_dimension=2048)
                out = out["result"] if isinstance(out, dict) else out
                cx, cy, cw, ch = int(out[3]), int(out[4]), int(out[5]), int(out[6])
            except Exception as exc:
                logger.info("SKU sheet: no logo candidate on %s (%s)", label, exc)
                continue
            # Clamp to the view: an unclamped pad runs off the edge on a logo
            # near the border and PIL returns the crop silently truncated.
            pad = int(max(cw, ch) * 0.6)
            box = (max(0, cx - cw // 2 - pad), max(0, cy - ch // 2 - pad),
                   min(img.width, cx + cw // 2 + pad), min(img.height, cy + ch // 2 + pad))
            candidates.append((label, img.crop(box)))
        if not candidates:
            return None

        system = (
            "You are shown numbered close-up crops taken from photographs of one pair of "
            "eyewear. Exactly one of them, or none, carries the brand's own logo - a word "
            "or an emblem applied to an outward-facing surface. Printing on an inner temple "
            "surface is not a brand mark, and neither is a size or fit marking such as "
            "'52 18 140' or a colour code - those are specification text, and a crop showing "
            "only those is a 'none'. Reply with one line: the number of the crop carrying the "
            "brand mark, then a colon, then the word you can read, or the shape of the emblem. "
            "Reply '0: none' when no crop carries a brand mark. No other text."
        )
        try:
            imgs = {"image_%d" % (i + 1): self._pil_to_tensor(c)
                    for i, (_l, c) in enumerate(candidates[:6])}
            answer = reader.generate(
                prompt="Which numbered crop carries the brand logo?", api_key=api_key,
                system_prompt=system, model="anthropic/claude-sonnet-4.6", reasoning=False,
                priority="throughput", auto_rescale_images=True, max_image_dimension=1024,
                temperature=0, max_tokens=200, **imgs)
            answer = str((answer["result"] if isinstance(answer, dict) else answer)[0]).strip()
            index = int(answer.split(":")[0].strip())
        except Exception as exc:
            logger.info("SKU sheet: could not pick a logo candidate (%s)", exc)
            return None
        if index < 1 or index > len(candidates):
            return None
        return candidates[index - 1][1], "LOGO"

    @staticmethod
    def _pil_to_tensor(img):
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

    # ------------------------------------------------------------------ build

    def build_sheet(self, front, side=None, three_quarter=None, three_quarter_additional=None,
                    max_long_side=5000, margin_percent=6.0, gap_px=14,
                    background_threshold=12.0, detail_boxes="", auto_bridge=True,
                    auto_joint=True, auto_logo=False, api_key="", layout="standard", canvas="fit"):
        try:
            supplied = (front, side, three_quarter, three_quarter_additional)
            views = [(label, self._to_pil(img))
                     for (_key, label), img in zip(VIEW_SLOTS, supplied) if img is not None]
            if not views:
                raise ValueError("Connect at least the front view.")

            products = [(label, img.crop(self._product_box(img, background_threshold)))
                        for label, img in views]
            if layout == "corner" and not (detail_boxes or "").strip():
                detail_boxes = json.dumps(self._corner_boxes(products, background_threshold))
                auto_bridge = auto_joint = auto_logo = False

            # The sheet is laid out at its natural width and scaled once at the
            # end, so max_long_side never compounds with the per-panel rounding.
            # A square canvas is laid out at its final size instead (u = how much
            # larger than 2600 px), so a 5000 px document carries real detail and
            # is not an enlarged 2600 px sheet.
            u = max(1.0, max_long_side / 2600.0) if canvas == "square" else 1.0
            gap = int(round(gap_px * u))
            label_h = int(round(28 * u))
            sheet_w = int(round(2600 * u))
            inner = sheet_w - 2 * gap
            rows = [products[:2], products[2:]]
            rows = [r for r in rows if r]
            margin = max(0, int(round(min(p.height for _l, p in products) * margin_percent / 100.0)))

            scale = min((inner - 2 * margin * len(r) - gap * (len(r) - 1))
                        / float(sum(p.width for _l, p in r))
                        for r in rows)
            if scale <= 0:
                raise ValueError("margin_percent leaves no room for the views; lower it.")

            def panel(label, product):
                w = max(1, int(product.width * scale))
                h = max(1, int(product.height * scale))
                cell = Image.new("RGB", (w + 2 * margin, h + 2 * margin + label_h), (255, 255, 255))
                cell.paste(product.resize((w, h), Image.LANCZOS), (margin, margin + label_h))
                ImageDraw.Draw(cell).text((margin, int(6 * u)), label, fill=(122, 122, 128), font=_font(int(22 * u)))
                return cell

            details = self._collect_details(products, detail_boxes, auto_bridge, auto_joint,
                                            auto_logo, api_key, background_threshold)
            if layout == "corner":
                # main views first, the frame the same height in each; the close-ups last
                sheet, view_scales = self._corner_sheet(products, details, u, gap, background_threshold, canvas)
                scale = view_scales[0]
            else:
                blocks = [[panel(l, p) for l, p in r] for r in rows]
                if len(products) == 4 or len(blocks) == 1:
                    strip_w = inner if len(products) == 4 else blocks[0][0].width
                else:
                    # the strip shares the second row with the last view: it gets the room that view leaves, so the
                    # row never runs past the sheet (a wide 3/4 view used to push the strip off the left edge)
                    strip_w = max(int(400 * u), inner - sum(c.width for c in blocks[1]) - gap * len(blocks[1]))
                strip = self._detail_strip(details, strip_w, gap, label_h) if details else None

                if strip is not None:
                    if len(products) == 4:
                        blocks.insert(1, [strip])          # a band between the two rows
                    elif len(blocks) > 1:
                        blocks[1].insert(0, strip)         # three views leave a cell free
                    else:
                        blocks.append([strip])

                height = sum(max(c.height for c in r) for r in blocks) + gap * (len(blocks) + 1)
                sheet = Image.new("RGB", (sheet_w, height), (188, 188, 194))
                y = gap
                for row in blocks:
                    row_w = sum(c.width for c in row) + gap * (len(row) - 1)
                    x = (sheet_w - row_w) // 2
                    for cell in row:
                        sheet.paste(cell, (x, y))
                        x += cell.width + gap
                    y += max(c.height for c in row) + gap

            if canvas == "square" and sheet.width != sheet.height:
                side_px = max(sheet.size)
                square = Image.new("RGB", (side_px, side_px), (188, 188, 194))
                square.paste(sheet, ((side_px - sheet.width) // 2, (side_px - sheet.height) // 2))
                sheet = square

            longest = max(sheet.size)
            if longest > max_long_side:
                f = max_long_side / float(longest)
                sheet = sheet.resize((max(1, int(sheet.width * f)),
                                      max(1, int(sheet.height * f))), Image.LANCZOS)

            info = json.dumps({
                "views": [l for l, _p in products],
                "shared_scale": round(scale, 5),
                "view_scales": [round(v, 5) for v in view_scales] if layout == "corner" else None,
                "details": [c for _crop, c in details],
                "sheet_size": list(sheet.size),
                "max_long_side": max_long_side,
            })
            return (self._pil_to_tensor(sheet), info)

        except Exception as exc:
            logger.error("SKU reference sheet failed: %s", exc)
            raise RuntimeError("SKU reference sheet failed: %s" % exc) from exc

    # ------------------------------------------------------- the corner layout's own close-ups
    # Settled on a 422-row eyewear review (2026-11): most failures were at the near corner (hinge, end piece, where
    # the temple leaves the front) and on the temple itself, so the corner is shown from three angles and the
    # temple whole. Boxes are fractions of each view's product box; the three-quarter one is measured per product.
    NEAR_JOINT_DEFAULT = (0.40, 0.22, 0.74, 0.78)
    SHADOW_PEAK = 0.40          # a column run whose darkest pixel stays under 40 % of the frame's contrast is shadow

    def _solid_mask(self, product, threshold):
        """The product without its floor shadow. Catalogue shots with a soft shadow ("_shad_") pass the background
        threshold in a grey band under the frame (contrast ~30-75 against ~200+ for the frame), and that band read
        as the near temple and stretched the joint and the frame height (Clearly Batch 5, 2026-11-09). Per column,
        a run of product pixels is kept only if it reaches SHADOW_PEAK of the frame's contrast somewhere, so clear
        acetate stays (its run holds the dark edge) and pure shadow goes. Frames without a shadow are unchanged."""
        arr = np.asarray(product.convert("RGB")).astype(np.int16)
        d = np.abs(arr - self._backdrop(arr)).max(2)
        m = d > threshold
        if not m.any():
            return m
        strong = max(threshold, self.SHADOW_PEAK * float(np.percentile(d[m], 99)))
        out = np.zeros_like(m)
        for x in np.nonzero(m.any(0))[0]:
            r = np.nonzero(m[:, x])[0]
            for run in np.split(r, np.nonzero(np.diff(r) > 1)[0] + 1):
                if d[run, x].max() >= strong:
                    out[run, x] = True
        return out

    def _near_joint(self, product, threshold):
        """The near front-to-temple joint on a three-quarter view (temple running to the right): the right end of
        the last run of tall columns (the front), ignoring the ear tip. -> (x1, y1, x2, y2) fractions."""
        m = self._solid_mask(product, threshold)
        H, W = m.shape
        ext = np.zeros(W)
        for x in range(W):
            r = np.nonzero(m[:, x])[0]
            if len(r):
                ext[x] = r.max() - r.min()
        if not (ext > 0).any():
            return self.NEAR_JOINT_DEFAULT
        tall = ext >= 0.55 * np.percentile(ext[ext > 0], 95)
        runs, x = [], 0
        while x < W:
            if tall[x]:
                s = x
                while x < W and tall[x]:
                    x += 1
                runs.append((s, x))
            x += 1
        runs = [r for r in runs if r[1] - r[0] >= 0.004 * W]
        if len(runs) > 1 and runs[-1][0] > 0.80 * W and runs[-1][0] - runs[-2][1] > 0.06 * W:
            runs = runs[:-1]                                   # the ear tip
        if not runs:
            return self.NEAR_JOINT_DEFAULT
        jx = runs[-1][1]
        return (round(max(0, jx - 0.08 * W) / W, 3), 0.12, round(min(W, jx + 0.26 * W) / W, 3), 0.78)

    def _temple_box(self, product, threshold):
        """The temple alone on the side view (front on the left): from just before the front ends to the tip,
        cut tight to the temple's own rows, so it is shown larger than in the full side view."""
        m = self._solid_mask(product, threshold)
        H, W = m.shape
        ext = np.array([(np.nonzero(c)[0].max() - np.nonzero(c)[0].min()) if c.any() else 0 for c in m.T])
        tall = ext >= 0.55 * np.percentile(ext[ext > 0], 95)
        idx = np.nonzero(tall[: int(0.45 * W)])[0]           # the front: tall columns at the left
        fx = int(idx.max()) if len(idx) else int(0.25 * W)
        x1 = max(0, fx - int(0.02 * W))                       # keep the hinge in the cut
        # Only the NEAR temple (owner, 2026-11-09): on a side view it is the lower one, the far one shows above it.
        # Per column past the front, the lowest run of product pixels is the near temple; the crop starts just above
        # its straight part, so the far temple stays out (at most its tip, where it drops into the crop).
        tops, bottoms = [], []
        for x in range(min(W - 1, fx + int(0.02 * W)), W):
            r = np.nonzero(m[:, x])[0]
            if not len(r):
                continue
            breaks = np.nonzero(np.diff(r) > 2)[0]
            start = r[breaks[-1] + 1] if len(breaks) else r[0]       # the lowest run
            tops.append((x, start))
            bottoms.append(r[-1])
        if not tops:
            return (round(x1 / W, 3), 0.0, 1.0, 1.0)
        straight = [t for x, t in tops if x < fx + 0.55 * (W - fx)] or [t for _x, t in tops]
        mg = int(0.03 * H)
        y1 = max(0, int(min(straight)) - mg)
        y2 = min(H, int(max(bottoms)) + mg)
        return (round(x1 / W, 3), round(y1 / H, 3), 1.0, round(y2 / H, 3))

    def _corner_boxes(self, products, threshold):
        labels = [label for label, _p in products]
        out = []

        def add(label, box, caption):
            if label in labels:
                out.append({"view": labels.index(label), "x1": box[0], "y1": box[1], "x2": box[2],
                            "y2": box[3], "caption": caption})

        add("FRONT", (0.00, 0.00, 0.15, 0.62), "CORNER: FRONT VIEW")
        qt = "3/4" if "3/4" in labels else ("3/4 ADDITIONAL" if "3/4 ADDITIONAL" in labels else None)
        if qt:
            try:
                joint = self._near_joint(products[labels.index(qt)][1], threshold)
            except Exception:
                joint = self.NEAR_JOINT_DEFAULT
            add(qt, joint, "CORNER: THREE-QUARTER VIEW")
        add("SIDE", (0.00, 0.15, 0.40, 0.65), "CORNER: SIDE VIEW")
        if "SIDE" in labels:
            try:
                temple = self._temple_box(products[labels.index("SIDE")][1], threshold)
            except Exception:
                temple = (0.25, 0.00, 1.00, 1.00)
            add("SIDE", temple, "TEMPLE (SIDE VIEW)")
        add("FRONT", (0.34, 0.00, 0.66, 0.55), "BRIDGE (FRONT VIEW)")
        return out

    def _region_crop(self, product, region, threshold):
        x1, y1, x2, y2 = region
        crop = product.crop((int(x1 * product.width), int(y1 * product.height),
                             int(x2 * product.width), int(y2 * product.height)))
        return crop.crop(self._product_box(crop, threshold))

    def _collect_details(self, products, detail_boxes, auto_bridge, auto_joint,
                         auto_logo, api_key, threshold):
        details = []
        labels = [label for label, _p in products]

        if auto_bridge:
            details.append((self._region_crop(products[0][1], BRIDGE_REGION, threshold),
                            "BRIDGE"))
        if auto_joint:
            # The joint reads on an angled view; head-on the temple is edge-on.
            for wanted in ("3/4", "3/4 ADDITIONAL", "SIDE", "FRONT"):
                if wanted in labels:
                    product = products[labels.index(wanted)][1]
                    details.append((self._region_crop(product, JOINT_REGION, threshold),
                                    "JOINT"))
                    break
        if auto_logo:
            if not api_key:
                logger.warning("SKU sheet: auto_logo is on but api_key is empty; skipping "
                               "the logo slot.")
            else:
                found = self._find_logo(products, api_key, threshold)
                if found:
                    details.append(found)
                else:
                    logger.info("SKU sheet: no brand logo found; leaving that slot out.")

        text = (detail_boxes or "").strip()
        if not text:
            return details
        try:
            entries = json.loads(text)
        except Exception as exc:
            raise ValueError("detail_boxes is not valid JSON: %s" % exc)
        for entry in entries:
            idx = int(entry.get("view", 0))
            if idx < 0 or idx >= len(products):
                continue
            _label, product = products[idx]
            x1 = int(float(entry.get("x1", 0.0)) * product.width)
            y1 = int(float(entry.get("y1", 0.0)) * product.height)
            x2 = int(float(entry.get("x2", 1.0)) * product.width)
            y2 = int(float(entry.get("y2", 1.0)) * product.height)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = product.crop((x1, y1, x2, y2))
            crop = crop.crop(self._product_box(crop, threshold))
            details.append((crop, str(entry.get("caption", "DETAIL"))))
        return details

    # A slot never grows past this, so one detail sits in a strip its own size
    # instead of floating in the middle of a full-width band.
    MAX_SLOT_W = 620

    def _detail_strip(self, details, width, gap, label_h):
        height = 340
        top = label_h + 22
        box_h = height - top - 18
        slot_w = min(self.MAX_SLOT_W,
                     (width - 4 * gap - gap * (len(details) - 1)) // len(details))
        needed = 4 * gap + len(details) * slot_w + gap * (len(details) - 1)
        width = min(width, needed)
        strip = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        draw.text((gap * 2, 6), "DETAILS", fill=(122, 122, 128), font=_font(22))
        for i, (crop, caption) in enumerate(details):
            fit = min(slot_w / crop.width, box_h / crop.height)
            w, h = max(1, int(crop.width * fit)), max(1, int(crop.height * fit))
            x = gap * 2 + i * (slot_w + gap)
            strip.paste(crop.resize((w, h), Image.LANCZOS),
                        (x + (slot_w - w) // 2, top + (box_h - h) // 2))
            draw.text((x, label_h + 2), caption, fill=(158, 158, 164), font=_font(17))
            if i:
                rule = x - gap // 2
                draw.line([(rule, top), (rule, top + box_h)], fill=(188, 188, 194), width=2)
        return strip

    # layout="corner" (owner, 2026-11-09): the three main views in one row with the FRAME at the same height in each
    # - catalogue photos frame every view to fill the width, so the same multiplier for all left the side ~17 % and
    # the 3/4 ~13 % smaller than the front, and an edit model reads a smaller view as a smaller frame. Then the
    # close-ups, last, with large dark captions.
    def _frame_height(self, product, threshold):
        """The frame's height in a view: the tallest columns of the product (the rim; temples are thin)."""
        m = self._solid_mask(product, threshold)
        cols = m.any(0)
        if not cols.any():
            return float(product.height)
        top = np.where(cols, m.argmax(0), 0)
        bottom = np.where(cols, m.shape[0] - 1 - m[::-1].argmax(0), 0)
        ext = (bottom - top)[cols]
        return max(1.0, float(np.percentile(ext, 99)))

    def _corner_sheet(self, products, details, u, gap, threshold, canvas):
        """Owner's pick (2026-11-09, option D): FRONT | SIDE, then 3/4 | TEMPLE, then the corner close-ups and the
        bridge in one row. Every main view at the same frame height (one scale k times the view's frame factor);
        a close-up sharing a row with a view takes the width left; close-up rows share the height left."""
        SW = int(round(2600 * u))
        inner = SW - 2 * gap
        grey, white, dark, mid = (188, 188, 194), (255, 255, 255), (35, 35, 40), (70, 70, 78)
        pad, lab = int(24 * u), int(56 * u)
        view_font, cap_font = _font(int(34 * u)), _font(int(30 * u))

        heights = [self._frame_height(p, threshold) for _l, p in products]
        rel = [heights[0] / h for h in heights]
        vw = [p.width * r for (_l, p), r in zip(products, rel)]
        vh = [p.height * r for (_l, p), r in zip(products, rel)]
        det = {cap: crop for crop, cap in details}
        temple = next((c for c in det if c.startswith("TEMPLE")), None)

        views = list(range(len(products)))
        rows = [views[:2]]
        if len(views) > 2:
            rows.append(views[2:] + ([temple] if temple and len(views) == 3 else []))
        elif temple:
            rows.append([temple])
        placed = {i for r in rows for i in r if isinstance(i, str)}
        rest = [c for c in det if c not in placed]
        vrows = [r for r in rows if r]

        min_close = int(0.27 * inner)
        k = min((inner - gap * (len(r) - 1) - 2 * pad * sum(isinstance(i, int) for i in r)
                 - min_close * sum(isinstance(i, str) for i in r))
                / sum(vw[i] for i in r if isinstance(i, int)) for r in vrows)

        def row_heights(k):
            return [int(max(vh[i] for i in r if isinstance(i, int)) * k) + lab + 2 * pad for r in vrows]

        def shown(per, h):
            """Total close-up area when `per` share a row of height h (to pick the fullest arrangement)."""
            w = (inner - gap * (per - 1)) // per
            return sum(min((w - 2 * pad) / det[c].width, (h - lab - 2 * pad) / det[c].height) ** 2
                       * det[c].width * det[c].height for c in rest)

        per = 4
        if canvas == "square" and rest:
            while True:
                left = SW - gap * (len(vrows) + 1) - sum(row_heights(k))
                if left >= int(0.17 * SW) + gap:
                    break
                k *= 0.97
            best = None
            for cand in (4, 3, 2):
                n_rows = -(-len(rest) // cand)
                h = (left - gap * n_rows) // n_rows
                if h >= int(0.15 * SW):
                    a = shown(cand, h)
                    if best is None or a > best[0]:
                        best = (a, cand, h)
            per, close_h = (best[1], best[2]) if best else (len(rest), left - gap)
        else:
            close_h = int(0.26 * SW)
        crows = [rest[j:j + per] for j in range(0, len(rest), per)]
        rows = vrows + crows
        rh = dict(zip(map(id, vrows), row_heights(k)))
        for r in crows:
            rh[id(r)] = close_h

        height = gap * (len(rows) + 1) + sum(rh.values())
        sheet = Image.new("RGB", (SW, height), grey)
        y = gap
        for r in rows:
            h = rh[id(r)]
            vws = {i: int(vw[i] * k) + 2 * pad for i in r if isinstance(i, int)}
            n_close = sum(isinstance(i, str) for i in r)
            cw = (inner - gap * (len(r) - 1) - sum(vws.values())) // max(1, n_close)
            x = gap
            for i in r:
                if isinstance(i, int):
                    label, p = products[i]
                    w, ih = int(p.width * rel[i] * k), int(p.height * rel[i] * k)
                    c = Image.new("RGB", (vws[i], h), white)
                    c.paste(p.resize((max(1, w), max(1, ih)), Image.LANCZOS), ((vws[i] - w) // 2, h - pad - ih))
                    ImageDraw.Draw(c).text((pad, int(10 * u)), label, fill=dark, font=view_font)
                else:
                    crop = det[i]
                    c = Image.new("RGB", (cw, h), white)
                    f = min((cw - 2 * pad) / crop.width, (h - lab - 2 * pad) / crop.height)
                    w, ih = max(1, int(crop.width * f)), max(1, int(crop.height * f))
                    c.paste(crop.resize((w, ih), Image.LANCZOS), ((cw - w) // 2, lab + pad + (h - lab - 2 * pad - ih) // 2))
                    ImageDraw.Draw(c).text((pad, int(12 * u)), i, fill=mid, font=cap_font)
                sheet.paste(c, (x, y))
                x += c.width + gap
            y += h + gap
        return sheet, [k * r for r in rel]
