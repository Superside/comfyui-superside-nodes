"""Tiled upscale in one node - crop, upscale every tile, stitch.

The tiling technique here is Aaron Amortegui's: the method presets, the
edge-aware feathering, the overlap colour matching and the crop/stitch geometry
all come from his own repository, which is the original and holds the rights to
the approach:

    https://github.com/amortegui84/comfyui-tile-upscale-AM

Everything between the two banner comments below is ported from it unchanged,
so a sheet built here matches one built there with the same settings. What is
new in this file is only the packaging.

What the packaging buys: the published workflows in that repo wire five node
types by hand - Tile Crop, one Tile Extract per tile, one upscaler per tile,
Tile Collect, Tile Stitch - so the graph grows with the grid and a 3x3 needs
nine of each. And the upscaler in those graphs is a local lanczos placeholder,
so the model call was never part of the graph at all; it had to be dropped in
by hand, per tile. Here the fan-out is a loop, the model is a dropdown, and the
fal key has one home.

fal bills per output megapixel per call and every tile is one call, so the
overlap is paid for. `info` reports the finished size, the print size at 300
dpi and the billed megapixels; set preview_only to see all three before
spending anything.
"""

import io
import json
import logging
import os
import struct
import time
import urllib.request
import zlib

import folder_paths
import numpy as np
import torch
from PIL import Image

from .base_node import API_KEY_INPUT_SPEC

logger = logging.getLogger(__name__)


# ===========================================================================
# PORTED from comfyui-tile-upscale-AM (Aaron Amortegui) - do not drift from it
# ===========================================================================


METHOD_PRESETS: dict[str, dict] = {
    "nb2": {
        "category": "regenerative",
        "changes_content": True,
        "recommended_overlap_pct": 20.0,
        "feather_mode": "strong",
        "color_match": True,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Nano Banana 2 — generative reinterpretation, strong blending",
    },
    "image_2": {
        "category": "regenerative",
        "changes_content": True,
        "recommended_overlap_pct": 20.0,
        "feather_mode": "strong",
        "color_match": True,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "GPT Image 2 — generative reinterpretation, strong blending",
    },
    "topaz": {
        "category": "faithful",
        "changes_content": False,
        "recommended_overlap_pct": 8.0,
        "feather_mode": "minimal",
        "color_match": False,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Topaz-style faithful upscale — minimal blending, preserve structure",
    },
    "seedv2": {
        "category": "faithful",
        "changes_content": False,
        "recommended_overlap_pct": 10.0,
        "feather_mode": "strong",
        "color_match": True,
        "recommended_cols": 3,
        "recommended_rows": 2,
        "description": "SeedVR2-style faithful upscale - 3x2 tiles, color match on",
    },
    "passthrough": {
        "category": "passthrough",
        "changes_content": False,
        "recommended_overlap_pct": 4.0,
        "feather_mode": "minimal",
        "color_match": False,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Passthrough / external upscale — exact alignment, near-zero blending",
    },
    "custom": {
        "category": "custom",
        "changes_content": False,
        "recommended_overlap_pct": 12.0,
        "feather_mode": "medium",
        "color_match": False,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Custom — user controls all parameters",
    },
}

GRID_PRESETS: dict[str, tuple[int, int] | None] = {
    "method default": None,
    "fixed 2x2": (2, 2),
    "fixed 2×2": (2, 2),
    "3x2 horizontal": (3, 2),
    "2x3 vertical": (2, 3),
    "3x3": (3, 3),
}

_FEATHER_POWER = {"strong": 1.0, "medium": 2.0, "minimal": 4.0}

def _np_to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray((a * 255).clip(0, 255).astype(np.uint8), "RGB")

def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB")).astype(np.float32) / 255.0

def _t2np(t: torch.Tensor) -> np.ndarray:
    """(H,W,C) float32 tensor → float32 numpy [0,1]."""
    return t.cpu().float().numpy()

def _np2t(a: np.ndarray) -> torch.Tensor:
    """float32 numpy [0,1] → (H,W,C) float32 tensor."""
    return torch.from_numpy(a.astype(np.float32))

def _resize_np(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize (H,W,C) float32 array to (h, w, C) via LANCZOS."""
    if a.shape[1] == w and a.shape[0] == h:
        return a
    return _pil_to_np(_np_to_pil(a).resize((w, h), Image.LANCZOS))

def _normalize_grid_label(label: str) -> str:
    return str(label).strip().lower().replace("×", "x")

def _resolve_grid(method: str, grid_preset: str, grid_cols: int, grid_rows: int) -> tuple[int, int, str]:
    preset = METHOD_PRESETS[method]
    label = _normalize_grid_label(grid_preset)

    if label in ("method default", "auto"):
        cols = int(preset["recommended_cols"])
        rows = int(preset["recommended_rows"])
        policy = "method_default"
    elif label in ("fixed 2x2", "2x2"):
        cols, rows = 2, 2
        policy = "preset"
    elif label in ("3x2 horizontal", "3x2"):
        cols, rows = 3, 2
        policy = "preset"
    elif label in ("2x3 vertical", "2x3"):
        cols, rows = 2, 3
        policy = "preset"
    elif label in ("3x3", "9 tiles", "9_tiles"):
        cols, rows = 3, 3
        policy = "preset"
    else:
        cols, rows = int(grid_cols), int(grid_rows)
        policy = "custom"

    cols = max(1, min(cols, 8))
    rows = max(1, min(rows, 8))
    return cols, rows, policy

def _uniform_tile_size(src_size: int, count: int, overlap_fraction: float) -> int:
    if count <= 1:
        return src_size
    coverage = 1.0 + (count - 1) * (1.0 - overlap_fraction)
    return max(1, min(src_size, int(np.ceil(src_size / coverage))))

def _tile_starts(src_size: int, tile_size: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    max_start = max(0, src_size - tile_size)
    return [round(i * max_start / (count - 1)) for i in range(count)]

def _edge_feather_axis(
    n: int,
    overlap: int,
    power: float,
    fade_start: bool,
    fade_end: bool,
) -> np.ndarray:
    ramp = np.ones(n, dtype=np.float64)
    if overlap <= 0:
        return ramp
    overlap = min(overlap, n // 2)
    t = np.linspace(0.0, 1.0, overlap, endpoint=False)
    fade = np.power(t, 1.0 / power)
    if fade_start:
        ramp[:overlap] *= fade
    if fade_end:
        ramp[-overlap:] *= fade[::-1]
    return ramp

def _tile_feather_mask(
    h: int,
    w: int,
    ov_h: int,
    ov_w: int,
    power: float,
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    row_w = _edge_feather_axis(h, ov_h, power, row > 0, row < rows - 1)
    col_w = _edge_feather_axis(w, ov_w, power, col > 0, col < cols - 1)
    return np.outer(row_w, col_w)

def _color_match_to_canvas(
    tile: np.ndarray,
    canvas: np.ndarray,
    canvas_weights: np.ndarray,
    feather: np.ndarray,
) -> np.ndarray:
    """
    Adjust tile mean/std to match the already-composited canvas in the overlap zone.

    tile, canvas  — (H, W, 3) float32
    canvas_weights — (H, W, 1) accumulated weight (0 where nothing placed yet)
    feather        — (H, W) weight for this tile
    Returns corrected tile (H, W, 3) float32.
    """
    # Overlap = where feather < 1 AND canvas already has content.
    placed = canvas_weights[:, :, 0] > 1e-6
    in_overlap = (feather < 0.999) & placed
    if not in_overlap.any():
        return tile

    # Normalised canvas values in overlap zone.
    ref = (canvas / np.where(canvas_weights > 0, canvas_weights, 1)).astype(np.float64)

    mask = in_overlap.astype(np.float64)
    eps = 1e-6

    ref_mean = (ref * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps)
    tile_mean = (tile * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps)

    ref_std = np.sqrt(
        ((ref - ref_mean) ** 2 * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps) + eps
    )
    tile_std = np.sqrt(
        ((tile - tile_mean) ** 2 * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps) + eps
    )

    corrected = (tile - tile_mean) * (ref_std / tile_std) + ref_mean
    # Blend correction proportional to in_overlap (full correction in overlap, none elsewhere).
    blend = np.clip(mask[:, :, None], 0, 1)
    return (tile * (1 - blend) + corrected * blend).clip(0, 1).astype(np.float32)

# -- crop and stitch, ported from TileCropAM / TileStitchAM -----------------

def crop_tiles(
    image: torch.Tensor,
    method: str,
    grid_preset: str,
    grid_cols: int,
    grid_rows: int,
    overlap_percent: float,
    target_tile_width: int = 0,
    target_tile_height: int = 0,
):
    # image: (B, H, W, C) — take the first frame; drop alpha if present.
    src_np = _t2np(image[0])           # (H, W, C)
    if src_np.shape[2] == 4:
        src_np = src_np[:, :, :3]
    src_h, src_w = src_np.shape[:2]

    preset = METHOD_PRESETS[method]

    requested_grid_cols = int(grid_cols)
    requested_grid_rows = int(grid_rows)
    requested_grid_preset = grid_preset

    grid_cols, grid_rows, tile_count_policy = _resolve_grid(
        method, grid_preset, grid_cols, grid_rows
    )

    eff_overlap = overlap_percent if overlap_percent >= 0.0 else preset["recommended_overlap_pct"]
    overlap_fraction = max(0.0, min(float(eff_overlap) / 100.0, 0.75))

    tile_w = _uniform_tile_size(src_w, grid_cols, overlap_fraction)
    tile_h = _uniform_tile_size(src_h, grid_rows, overlap_fraction)
    xs = _tile_starts(src_w, tile_w, grid_cols)
    ys = _tile_starts(src_h, tile_h, grid_rows)

    # Nominal overlap in pixels for feathering. Actual start positions are
    # stored per tile and used for exact placement during stitch.
    ov_w = max(0, round(tile_w * overlap_fraction))
    ov_h = max(0, round(tile_h * overlap_fraction))

    base_w = tile_w
    base_h = tile_h

    # Build tile crop regions.
    tile_infos: list[dict] = []
    crops: list[np.ndarray] = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            x0 = xs[col]
            y0 = ys[row]
            x1 = min(src_w, x0 + tile_w)
            y1 = min(src_h, y0 + tile_h)

            crop = src_np[y0:y1, x0:x1]

            tile_infos.append(
                {
                    "index": row * grid_cols + col,
                    "row": row,
                    "col": col,
                    "src_x0": x0,
                    "src_y0": y0,
                    "src_x1": x1,
                    "src_y1": y1,
                    "src_w": x1 - x0,
                    "src_h": y1 - y0,
                }
            )
            crops.append(crop)

    # Determine uniform tile output dimensions.
    # If target not specified, use the natural size of the first tile.
    unif_w = target_tile_width if target_tile_width > 0 else crops[0].shape[1]
    unif_h = target_tile_height if target_tile_height > 0 else crops[0].shape[0]

    tensors: list[torch.Tensor] = []
    for crop in crops:
        resized = _resize_np(crop, unif_w, unif_h)
        tensors.append(_np2t(resized))

    batch = torch.stack(tensors, dim=0)   # (N, unif_h, unif_w, C)

    metadata = {
        "method": method,
        "preset": preset,
        "requested_grid_preset": requested_grid_preset,
        "requested_grid_cols": requested_grid_cols,
        "requested_grid_rows": requested_grid_rows,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "tile_count_policy": tile_count_policy,
        "grid_preset": grid_preset,
        "overlap_pct": eff_overlap,
        "overlap_w": ov_w,
        "overlap_h": ov_h,
        "base_tile_w": base_w,
        "base_tile_h": base_h,
        "src_w": src_w,
        "src_h": src_h,
        "uniform_tile_w": unif_w,
        "uniform_tile_h": unif_h,
        "tiles": tile_infos,
    }

    return (batch, json.dumps(metadata, indent=2), len(tensors))

def stitch_tiles(
    tiles: torch.Tensor,
    tile_metadata: str,
    color_match_override: str = "auto",
    feather_mode_override: str = "auto",
    upscale_factor: float = 0.0,
):
    meta = json.loads(tile_metadata)
    preset: dict = meta["preset"]
    tile_infos: list[dict] = meta["tiles"]

    src_w: int = meta["src_w"]
    src_h: int = meta["src_h"]
    ov_w: int = meta["overlap_w"]
    ov_h: int = meta["overlap_h"]
    unif_w: int = meta["uniform_tile_w"]
    unif_h: int = meta["uniform_tile_h"]
    grid_cols: int = meta.get("grid_cols", 1)
    grid_rows: int = meta.get("grid_rows", 1)

    n_tiles = tiles.shape[0]
    if n_tiles != len(tile_infos):
        raise ValueError(
            f"TileStitchAM: received {n_tiles} tiles but metadata describes "
            f"{len(tile_infos)}. Check your tile count."
        )

    # Determine upscale factor from processed tile size vs uniform tile size,
    # unless an explicit workflow factor is connected for deterministic sizing.
    proc_h, proc_w = tiles.shape[1], tiles.shape[2]
    if upscale_factor > 0.0:
        scale_x = scale_y = float(upscale_factor)
    else:
        scale_x = proc_w / unif_w
        scale_y = proc_h / unif_h

    canvas_w = round(src_w * scale_x)
    canvas_h = round(src_h * scale_y)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_acc = np.zeros((canvas_h, canvas_w, 1), dtype=np.float64)

    # Resolve blend settings.
    do_color_match = (
        preset["color_match"]
        if color_match_override == "auto"
        else (color_match_override == "on")
    )
    feather_mode = (
        preset["feather_mode"]
        if feather_mode_override == "auto"
        else feather_mode_override
    )
    power = _FEATHER_POWER[feather_mode]

    # Scaled overlap sizes.
    ov_w_s = round(ov_w * scale_x)
    ov_h_s = round(ov_h * scale_y)

    for idx, tinfo in enumerate(tile_infos):
        tile_np = _t2np(tiles[idx])      # (proc_h, proc_w, C)
        if tile_np.shape[2] == 4:        # drop alpha if upscaler returned RGBA
            tile_np = tile_np[:, :, :3]

        # Destination canvas region (scaled from source coordinates).
        dst_x0 = round(tinfo["src_x0"] * scale_x)
        dst_y0 = round(tinfo["src_y0"] * scale_y)
        dst_w = round(tinfo["src_w"] * scale_x)
        dst_h = round(tinfo["src_h"] * scale_y)
        dst_x1 = min(dst_x0 + dst_w, canvas_w)
        dst_y1 = min(dst_y0 + dst_h, canvas_h)
        dst_w = dst_x1 - dst_x0
        dst_h = dst_y1 - dst_y0

        # Resize processed tile to exactly fit destination region.
        tile_r = _resize_np(tile_np, dst_w, dst_h)

        # Build an edge-aware feather mask. Image-boundary edges stay fully
        # weighted; only edges with neighboring tiles fade into overlaps.
        feather = _tile_feather_mask(
            dst_h,
            dst_w,
            ov_h_s,
            ov_w_s,
            power,
            int(tinfo.get("row", 0)),
            int(tinfo.get("col", 0)),
            grid_rows,
            grid_cols,
        )

        # Optional: colour-match tile to already-placed canvas in overlap zone.
        if do_color_match:
            tile_r = _color_match_to_canvas(
                tile_r,
                canvas[dst_y0:dst_y1, dst_x0:dst_x1],
                weight_acc[dst_y0:dst_y1, dst_x0:dst_x1],
                feather,
            )

        canvas[dst_y0:dst_y1, dst_x0:dst_x1] += tile_r * feather[:, :, None]
        weight_acc[dst_y0:dst_y1, dst_x0:dst_x1] += feather[:, :, None]

    # Normalise — safe even where weight==0 (black where no tile placed).
    safe_w = np.where(weight_acc > 0, weight_acc, 1.0)
    result = (canvas / safe_w).clip(0, 1).astype(np.float32)
    return (_np2t(result).unsqueeze(0),)


# ===========================================================================
# NEW here: the model call, and the node that drives the whole thing.
# ===========================================================================


def _faithful(endpoint, preset, extra=None):
    """A plain upscaler: an image and a factor in, one image back."""
    return {"endpoint": endpoint, "preset": preset, "generative": False,
            "extra": extra or {}}


def _generative(endpoint, preset):
    """A model that redraws each tile, so it needs a prompt."""
    return {"endpoint": endpoint, "preset": preset, "generative": True, "extra": {}}


# The method preset drives overlap, feathering and colour matching: a faithful
# upscaler needs almost no blending, a model that redraws a tile needs a lot.
MODELS = {
    "topaz": _faithful("fal-ai/topaz/upscale/image", "topaz", {"model": "Standard V2"}),
    "seedvr2": _faithful("fal-ai/seedvr/upscale/image/seamless", "seedv2"),
    "crystal": _faithful("fal-ai/crystal-upscaler", "topaz"),
    "nano banana 2": _generative("fal-ai/nano-banana-2/edit", "nb2"),
    "gpt image 2": _generative("openai/gpt-image-2/edit", "image_2"),
    "none (local lanczos)": {"endpoint": None, "preset": "passthrough",
                             "generative": False, "extra": {}},
}

TILE_CHOICES = ["1x1 (no tiling)", "2x2", "3x2", "2x3", "3x3"]
_TILE_GRID = {"1x1 (no tiling)": (1, 1), "2x2": (2, 2), "3x2": (3, 2),
              "2x3": (2, 3), "3x3": (3, 3)}


def _tile_to_pil(tile):
    arr = tile.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return Image.fromarray(
        (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)[..., :3], "RGB")


def _pil_to_tensor(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0)


def _png_bytes(pil):
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return buffer.getvalue()


def _first_url(result):
    """fal's upscalers answer with `image`, the editing models with `images`."""
    if isinstance(result.get("image"), dict) and result["image"].get("url"):
        return result["image"]["url"]
    images = result.get("images")
    if isinstance(images, list) and images and isinstance(images[0], dict):
        url = images[0].get("url")
        if url:
            return url
    raise RuntimeError("the model returned no image: %s" % str(result)[:200])


class SupersideTileUpscaleNode:
    """Crop into tiles, upscale each one through fal, stitch them back."""

    CATEGORY = "Superside"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "run"
    DISPLAY_NAME = "Tile Upscale"
    DESCRIPTION = (
        "Upscale an image in tiles through one node: connect the image, paste "
        "the fal key, pick a model and a grid. Each tile is sent to the model "
        "and the results are stitched back with the overlap, feathering and "
        "colour matching that model calls for. `info` gives the finished size, "
        "the print size at 300 dpi and the billed megapixels - turn on "
        "preview_only to read all three before spending anything.\n\n"
        "The tiling technique is Aaron Amortegui's, from "
        "github.com/amortegui84/comfyui-tile-upscale-AM, which is the original."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(MODELS.keys()), {
                    "default": "topaz",
                    "tooltip": "Which upscaler runs on every tile. topaz, seedvr2 and "
                               "crystal keep the content; nano banana 2 and gpt image 2 "
                               "redraw it and need a prompt. 'none' upscales locally with "
                               "lanczos and costs nothing - useful for checking the "
                               "geometry before paying for it.",
                }),
                "tiles": (TILE_CHOICES, {
                    "default": "2x2",
                    "tooltip": "How the image is split. More tiles means more detail per "
                               "tile - and one API call each, so cost rises with the count.",
                }),
                "scale": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5,
                    "tooltip": "How much bigger the finished image is than the original.",
                }),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Only used by the generative models. Describe the output, "
                               "not the input, and keep it identical for every tile - a "
                               "prompt that drifts between tiles leaves seams that no "
                               "amount of feathering can hide.",
                }),
                "preview_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Work out the finished size, the print size and the billed "
                               "megapixels without calling the model. Returns the input "
                               "image untouched.",
                }),
                "overlap_percent": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Overlap between tiles, as a percentage of tile size. -1 "
                               "uses what the chosen model calls for. Raise it if seams show.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 600, "min": 60, "max": 3600, "step": 30,
                    "tooltip": "How long to wait on a single tile before giving up.",
                }),
            },
        }

    def _upscale_tile(self, client, tile, spec, scale, prompt, timeout):
        pil = _tile_to_pil(tile)
        url = client.upload(_png_bytes(pil), "image/png")

        if spec["generative"]:
            arguments = {"prompt": prompt, "image_urls": [url]}
        else:
            arguments = {"image_url": url, "upscale_factor": float(scale)}
        arguments.update(spec["extra"])

        result = client.subscribe(spec["endpoint"], arguments=arguments,
                                  client_timeout=timeout)
        raw = urllib.request.urlopen(_first_url(result), timeout=120).read()
        out = Image.open(io.BytesIO(raw)).convert("RGB")

        # A generative model answers at whatever size it likes, and the stitcher
        # needs every tile the same size to lay them back down.
        want = (int(pil.width * scale), int(pil.height * scale))
        if out.size != want:
            out = out.resize(want, Image.LANCZOS)
        return _pil_to_tensor(out)

    def run(self, image, model, tiles, scale, api_key, prompt="",
            preview_only=False, overlap_percent=-1.0, timeout_seconds=600):
        spec = MODELS[model]
        cols, rows = _TILE_GRID[tiles]
        count = cols * rows

        # "custom" is not a known grid label, which is how the ported resolver
        # is told to take grid_cols / grid_rows literally.
        cropped, metadata, _n = crop_tiles(
            image=image, method=spec["preset"], grid_preset="custom",
            grid_cols=cols, grid_rows=rows, overlap_percent=overlap_percent)

        meta = json.loads(metadata)
        src_w, src_h = int(meta["src_w"]), int(meta["src_h"])
        tile_h, tile_w = int(cropped.shape[1]), int(cropped.shape[2])
        out_w, out_h = int(round(src_w * scale)), int(round(src_h * scale))
        up_w, up_h = int(tile_w * scale), int(tile_h * scale)
        billed_mp = count * (up_w * up_h) / 1e6
        final_mp = (out_w * out_h) / 1e6

        report = [
            "model        %s" % model,
            "grid         %s  (%d tiles, %d API call%s)"
            % (tiles, count, 0 if spec["endpoint"] is None else count,
               "" if count == 1 else "s"),
            "source       %d x %d  (%.1f MP)" % (src_w, src_h, src_w * src_h / 1e6),
            "each tile    %d x %d  ->  %d x %d" % (tile_w, tile_h, up_w, up_h),
            "FINAL        %d x %d  (%.1f MP)" % (out_w, out_h, final_mp),
            "at 300 dpi   %.1f x %.1f in   /   %.1f x %.1f cm"
            % (out_w / 300.0, out_h / 300.0, out_w / 300.0 * 2.54, out_h / 300.0 * 2.54),
            "billed       %.1f MP across the tiles (%.1f MP more than the final image, "
            "which is the overlap)" % (billed_mp, billed_mp - final_mp),
        ]

        if preview_only:
            report += ["", "preview_only is ON - nothing was sent and nothing was charged."]
            return (image, "\n".join(report))

        if spec["endpoint"] is None:
            upscaled = torch.stack([
                _pil_to_tensor(
                    _tile_to_pil(cropped[i]).resize((up_w, up_h), Image.LANCZOS))
                for i in range(cropped.shape[0])])
        else:
            if spec["generative"] and not prompt.strip():
                raise ValueError(
                    "%s redraws each tile, so it needs a prompt. Describe the output you "
                    "want, and use the same text for the whole run." % model)
            if not api_key.strip():
                raise ValueError("api_key is empty, and %s is called through fal." % model)

            import fal_client

            client = fal_client.SyncClient(key=api_key.strip())
            done = []
            started = time.time()
            for index in range(cropped.shape[0]):
                tile_started = time.time()
                done.append(self._upscale_tile(client, cropped[index], spec, scale,
                                               prompt, timeout_seconds))
                logger.info("Tile upscale: tile %d/%d in %.0fs", index + 1,
                            cropped.shape[0], time.time() - tile_started)
            upscaled = torch.stack(done)
            report.append("took         %.0fs for %d tiles" % (time.time() - started, count))

        stitched = stitch_tiles(tiles=upscaled, tile_metadata=metadata,
                                upscale_factor=float(scale))[0]

        got_h, got_w = int(stitched.shape[1]), int(stitched.shape[2])
        if (got_w, got_h) != (out_w, out_h):
            report.append("note         stitched to %d x %d, %d x %d was expected"
                          % (got_w, got_h, out_w, out_h))
        return (stitched, "\n".join(report))


# -- also ported: saving with DPI, which is what makes a TIFF printable ----

def _write_png_dpi(path: str, img_pil: Image.Image, dpi: int) -> None:
    """
    Write a PNG file with a pHYs chunk encoding the given DPI.

    DPI metadata tells viewers / printers how to scale the image for display
    or print.  It does NOT add image detail — pixel count is unchanged.
    We inject the pHYs chunk manually after IHDR to guarantee it is present
    regardless of PIL version.
    """
    import io

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    raw = buf.getvalue()

    # PNG signature is 8 bytes; IHDR chunk follows immediately.
    # Chunk format: [4-byte length][4-byte type][data][4-byte CRC]
    ihdr_data_len = struct.unpack(">I", raw[8:12])[0]
    ihdr_end = 8 + 4 + 4 + ihdr_data_len + 4   # sig + len + type + data + crc

    # pHYs payload: x_ppm (4), y_ppm (4), unit=1/metre (1) = 9 bytes.
    ppm = round(dpi / 0.0254)
    phys_payload = struct.pack(">IIB", ppm, ppm, 1)
    phys_crc = zlib.crc32(b"pHYs" + phys_payload) & 0xFFFFFFFF
    phys_chunk = (
        struct.pack(">I", len(phys_payload))
        + b"pHYs"
        + phys_payload
        + struct.pack(">I", phys_crc)
    )

    out = raw[:ihdr_end] + phys_chunk + raw[ihdr_end:]
    with open(path, "wb") as f:
        f.write(out)

class SupersideSaveImageWithDPINode:
    """
    Save an image with embedded DPI metadata.

    IMPORTANT: DPI is metadata that tells viewers/printers the intended display
    size.  Changing DPI does NOT create new image detail.  A 1000×1000 image
    saved at 300 DPI will print at ~8.5cm × 8.5cm; saved at 72 DPI it prints
    at ~35cm × 35cm — same pixel count, different physical size interpretation.

    Supported formats and DPI support:
    - PNG:  pHYs chunk (lossless, always embedded).
    - TIFF: ResolutionUnit / XResolution / YResolution tags (lossless with LZW).
    - JPEG: APP0/JFIF DPI fields (lossy, quality-controlled).
    """

    CATEGORY = "Superside"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "print_size_info")
    OUTPUT_NODE = True
    FUNCTION = "save_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "tile_upscale"}),
                "dpi": (
                    ["72", "150", "300", "600"],
                    {
                        "default": "300",
                        "tooltip": (
                            "DPI metadata only — does not resize or add detail. "
                            "72=screen/web, 150=draft print, 300=quality print, 600=high-res print."
                        ),
                    },
                ),
                "format": (["png", "tiff", "jpeg"], {"default": "png"}),
            },
            "optional": {
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 95,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "JPEG only. Higher = better quality, larger file.",
                    },
                ),
                "output_subfolder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Subfolder inside ComfyUI output directory. Empty = root output dir.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save_image(
        self,
        image: torch.Tensor,
        filename_prefix: str,
        dpi: str,
        format: str,
        jpeg_quality: int = 95,
        output_subfolder: str = "",
        prompt=None,
        extra_pnginfo=None,
    ):
        dpi_val = int(dpi)
        fmt = format  # avoid shadowing the built-in

        # Resolve output directory — fall back gracefully if folder_paths unavailable.
        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
        except ImportError:
            base_dir = os.path.join(os.path.dirname(__file__), "output")

        out_dir = os.path.join(base_dir, output_subfolder) if output_subfolder else base_dir
        os.makedirs(out_dir, exist_ok=True)

        ext = {"png": ".png", "tiff": ".tif", "jpeg": ".jpg"}[fmt]

        # Find next available filename (non-overwriting).
        counter = 1
        while True:
            fname = f"{filename_prefix}_{counter:04d}{ext}"
            path = os.path.join(out_dir, fname)
            if not os.path.exists(path):
                break
            counter += 1

        # Convert tensor to PIL — first frame only; strip alpha if present.
        img_t = _t2np(image[0])
        if img_t.shape[2] == 4:
            img_t = img_t[:, :, :3]
        img_np = (img_t * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, "RGB")
        w, h = img_pil.size

        if fmt == "png":
            _write_png_dpi(path, img_pil, dpi_val)
        elif fmt == "tiff":
            img_pil.save(path, format="TIFF", dpi=(dpi_val, dpi_val), compression="lzw")
        elif fmt == "jpeg":
            img_pil.save(path, format="JPEG", quality=jpeg_quality, dpi=(dpi_val, dpi_val))

        # Physical print size (DPI is metadata only — pixel count is unchanged).
        w_in = w / dpi_val
        h_in = h / dpi_val
        w_cm = w_in * 2.54
        h_cm = h_in * 2.54
        size_info = (
            f"{w}×{h} px  |  {dpi_val} DPI\n"
            f"Print size: {w_in:.1f}\" × {h_in:.1f}\"  ({w_cm:.1f} cm × {h_cm:.1f} cm)\n"
            f"Saved: {path}"
        )
        print(f"[SaveImageWithDPI] {size_info}")

        # ComfyUI preview output.
        results = [{"filename": fname, "subfolder": output_subfolder, "type": "output"}]
        return {"ui": {"images": results}, "result": (path, size_info)}
