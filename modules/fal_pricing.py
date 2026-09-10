"""
fal.ai price table for the endpoints this package calls.

Every price here was read from fal's own model catalogue (the
`pricingInfoOverride` field of https://fal.ai/api/models, or the "Cost per X"
table on the model's page) on 2026-09-09. `SOURCE_DATE` below is what the cost
report prints so it is obvious how fresh the numbers are.

Two kinds of entry:

* **Priced** - fal publishes a per-call price, so `estimator` can compute the
  cost of one call from the request arguments and the response.
* **Unpriced** - fal bills the endpoint by GPU-second or by token consumption
  and publishes no per-call figure. `estimator` is None and `reason` explains
  why. These calls are counted and listed separately in the report instead of
  being guessed at, so the total is never silently wrong.

Prices change. `python -m modules.fal_pricing --check` re-reads fal's catalogue
and prints any endpoint whose published price text no longer matches the note
recorded here.
"""

import math

SOURCE_DATE = "2026-09-09"
SOURCE_URL = "https://fal.ai/api/models"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _output_count(arguments, result):
    """How many images the call actually produced (falls back to the request)."""
    images = (result or {}).get("images")
    if isinstance(images, list) and images:
        return len(images)
    return int(arguments.get("num_images") or 1)


def _input_count(arguments):
    urls = arguments.get("image_urls")
    if isinstance(urls, list):
        return len(urls)
    return 1 if arguments.get("image_url") else 0


def _output_megapixels(arguments, result):
    """Output megapixels from the response, rounded up (fal bills per started MP)."""
    images = (result or {}).get("images") or []
    total = 0.0
    for image in images:
        if not isinstance(image, dict):
            continue
        width, height = image.get("width"), image.get("height")
        if width and height:
            total += (width * height) / 1_000_000.0
    return total


def _duration_seconds(arguments):
    raw = arguments.get("duration")
    if raw is None:
        return None
    try:
        return float(str(raw).rstrip("s"))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# per-endpoint estimators -> (usd, detail)
# --------------------------------------------------------------------------

def _grok_v2_edit(arguments, result):
    per_image = {
        ("1k", "low"): 0.04,
        ("1k", "medium"): 0.06,
        ("2k", "low"): 0.06,
        ("2k", "medium"): 0.08,
    }
    resolution = str(arguments.get("resolution") or "1k").lower()
    quality = str(arguments.get("quality") or "medium").lower()
    rate = per_image.get((resolution, quality))
    if rate is None:
        return None, "unrecognised resolution/quality ({}/{})".format(resolution, quality)
    outputs = _output_count(arguments, result)
    inputs = _input_count(arguments)
    usd = rate * outputs + 0.01 * inputs
    return usd, "{} x {} {} image @ ${:.2f} + {} input @ $0.01".format(
        outputs, resolution.upper(), quality, rate, inputs
    )


def _grok_quality_edit(arguments, result):
    per_image = {"1k": 0.05, "2k": 0.07}
    resolution = str(arguments.get("resolution") or "1k").lower()
    rate = per_image.get(resolution)
    if rate is None:
        return None, "unrecognised resolution ({})".format(resolution)
    outputs = _output_count(arguments, result)
    inputs = _input_count(arguments)
    return rate * outputs + 0.01 * inputs, "{} x {} image @ ${:.2f} + {} input @ $0.01".format(
        outputs, resolution.upper(), rate, inputs
    )


def _nano_banana_2_edit(arguments, result):
    multiplier = {"0.5k": 0.75, "1k": 1.0, "2k": 1.5, "4k": 2.0}
    resolution = str(arguments.get("resolution") or "1k").lower()
    factor = multiplier.get(resolution)
    if factor is None:
        return None, "unrecognised resolution ({})".format(resolution)
    outputs = _output_count(arguments, result)
    usd = 0.08 * factor * outputs
    detail = "{} x image @ ${:.3f} ({} = {}x rate)".format(outputs, 0.08 * factor, resolution.upper(), factor)
    if arguments.get("enable_web_search"):
        usd += 0.015
        detail += " + web search $0.015"
    if str(arguments.get("thinking_level") or "").lower() == "high":
        usd += 0.002
        detail += " + high thinking $0.002"
    return usd, detail


def _nano_banana_pro_edit(arguments, result):
    resolution = str(arguments.get("resolution") or "1k").lower()
    factor = 2.0 if resolution == "4k" else 1.0
    outputs = _output_count(arguments, result)
    usd = 0.15 * factor * outputs
    detail = "{} x image @ ${:.3f} ({})".format(outputs, 0.15 * factor, resolution.upper())
    if arguments.get("enable_web_search"):
        usd += 0.015
        detail += " + web search $0.015"
    return usd, detail


def _flat_per_request(rate):
    def estimator(arguments, result):
        return rate, "1 request @ ${:.4f}".format(rate)
    return estimator


def _flat_per_output_image(rate):
    def estimator(arguments, result):
        outputs = _output_count(arguments, result)
        return rate * outputs, "{} x image @ ${:.4f}".format(outputs, rate)
    return estimator


def _per_output_megapixel(rate, round_up=True):
    def estimator(arguments, result):
        megapixels = _output_megapixels(arguments, result)
        if not megapixels:
            return None, "output dimensions not reported by fal"
        billed = math.ceil(megapixels) if round_up else megapixels
        return rate * billed, "{:.2f} MP output -> {} MP billed @ ${:.3f}/MP".format(
            megapixels, billed, rate
        )
    return estimator


def _seedream_v5_pro_edit(arguments, result):
    outputs = _output_count(arguments, result)
    extra_inputs = max(0, _input_count(arguments) - 1)
    usd = 0.0675 * outputs + 0.0045 * extra_inputs
    return usd, "{} x output @ $0.0675 + {} extra input @ $0.0045 (tentative fal pricing)".format(
        outputs, extra_inputs
    )


def _wan_25_image_to_video(arguments, result):
    per_second = {"480p": 0.05, "720p": 0.10, "1080p": 0.15}
    resolution = str(arguments.get("resolution") or "").lower()
    rate = per_second.get(resolution)
    seconds = _duration_seconds(arguments)
    if rate is None or seconds is None:
        return None, "resolution/duration not resolvable ({}/{})".format(resolution, seconds)
    return rate * seconds, "{:g}s of {} @ ${:.2f}/s".format(seconds, resolution, rate)


def _kling_video(base_5s, per_extra_second):
    def estimator(arguments, result):
        seconds = _duration_seconds(arguments) or 5.0
        extra = max(0.0, seconds - 5.0)
        usd = base_5s + per_extra_second * extra
        return usd, "{:g}s video: ${:.2f} base (5s) + {:g}s x ${:.3f}".format(
            seconds, base_5s, extra, per_extra_second
        )
    return estimator


def _seedvr_upscale_video(arguments, result):
    video = (result or {}).get("video") or {}
    width, height = video.get("width"), video.get("height")
    frames = video.get("frames") or video.get("num_frames")
    if not (width and height and frames):
        return None, "fal did not report output width/height/frames"
    megapixels = (width * height * frames) / 1_000_000.0
    return 0.001 * megapixels, "{}x{} x {} frames = {:.1f} MP @ $0.001/MP".format(
        width, height, frames, megapixels
    )


def _z_image_trainer(arguments, result):
    steps = int(arguments.get("steps") or arguments.get("num_steps") or 1000)
    billed = max(500, steps)
    usd = 0.85 * billed / 1000.0
    detail = "{} steps".format(steps)
    if billed != steps:
        detail += " -> billed at the {}-step floor".format(billed)
    return usd, detail + " @ $0.85 / 1000 steps"


def _seedance_tokens(usd_per_million_tokens, reference_note):
    """Seedance bills video tokens: (height x width x fps x duration) / 1024."""
    def estimator(arguments, result):
        video = (result or {}).get("video") or {}
        width = video.get("width")
        height = video.get("height")
        fps = video.get("fps") or 24
        seconds = _duration_seconds(arguments)
        if not (width and height and seconds):
            return None, "fal did not report output size; " + reference_note
        tokens = (height * width * fps * seconds) / 1024.0
        usd = tokens / 1_000_000.0 * usd_per_million_tokens
        return usd, "{}x{} @ {}fps x {:g}s = {:.0f}k tokens @ ${:g}/1M".format(
            width, height, fps, seconds, tokens / 1000.0, usd_per_million_tokens
        )
    return estimator


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

def _priced(note, estimator):
    return {"note": note, "estimator": estimator, "reason": None}


def _unpriced(note, reason):
    return {"note": note, "estimator": None, "reason": reason}


GPU_TIME = "fal bills this endpoint per GPU-second, so the cost depends on how long inference runs - there is no per-call price to quote"
TOKEN_BILLED = "fal bills this endpoint by token consumption, which depends on prompt and image size - no per-call price is published"
NO_PUBLISHED_PRICE = "fal publishes no price for this endpoint id - check the model page or your fal usage dashboard"

PRICES = {
    # ---- image edit / generate, priced per image -------------------------
    "xai/grok-imagine-image/v2.0/edit": _priced(
        "$0.04 (low) / $0.06 (medium) per 1K image, $0.06 / $0.08 per 2K image, plus $0.01 per input image",
        _grok_v2_edit,
    ),
    "xai/grok-imagine-image/quality/edit": _priced(
        "$0.05 per 1K output image, $0.07 per 2K output image, plus $0.01 per input image",
        _grok_quality_edit,
    ),
    "fal-ai/nano-banana-2/edit": _priced(
        "$0.08 per image (0.5K x0.75, 2K x1.5, 4K x2), +$0.015 with web search, +$0.002 with high thinking",
        _nano_banana_2_edit,
    ),
    "fal-ai/nano-banana-pro/edit": _priced(
        "$0.15 per image (4K charged at double), +$0.015 with web search",
        _nano_banana_pro_edit,
    ),
    "fal-ai/wan-25-preview/image-to-image": _priced(
        "$0.05 per image",
        _flat_per_output_image(0.05),
    ),
    "fal-ai/bytedance/seedream/v4.5/edit": _priced(
        "$0.04 per edit",
        _flat_per_output_image(0.04),
    ),
    "bytedance/seedream/v5/pro/edit": _priced(
        "~$0.0675 per output image plus $0.0045 per additional input image (fal calls this tentative pricing)",
        _seedream_v5_pro_edit,
    ),
    "fal-ai/bria/background/remove": _priced(
        "$0.018 per image",
        _flat_per_output_image(0.018),
    ),
    "fal-ai/sam-3/image": _priced(
        "$0.005 per request",
        _flat_per_request(0.005),
    ),
    # ---- priced per megapixel -------------------------------------------
    "fal-ai/flux-pro/v1/fill": _priced(
        "$0.05 per output megapixel, rounded up to the next megapixel",
        _per_output_megapixel(0.05),
    ),
    # ---- video ----------------------------------------------------------
    "fal-ai/wan-25-preview/image-to-video": _priced(
        "$0.05 per second at 480p, $0.10 at 720p, $0.15 at 1080p",
        _wan_25_image_to_video,
    ),
    "fal-ai/kling-video/v2.1/standard/image-to-video": _priced(
        "$0.28 for 5s, then $0.056 per extra second",
        _kling_video(0.28, 0.056),
    ),
    "fal-ai/kling-video/v2.1/pro/image-to-video": _priced(
        "$0.49 for 5s, then $0.098 per extra second",
        _kling_video(0.49, 0.098),
    ),
    "fal-ai/kling-video/v2.1/master/image-to-video": _priced(
        "$1.40 for 5s, then $0.28 per extra second",
        _kling_video(1.40, 0.28),
    ),
    "fal-ai/kling-video/v2.5-turbo/pro/image-to-video": _priced(
        "$0.35 for 5s, then $0.07 per extra second",
        _kling_video(0.35, 0.07),
    ),
    "fal-ai/bytedance/seedance/v1/pro/image-to-video": _priced(
        "video tokens at $2.50 per 1M - a 1080p 5s video is roughly $0.62",
        _seedance_tokens(2.5, "a 1080p 5s video is roughly $0.62"),
    ),
    "fal-ai/bytedance/seedance/v1/lite/reference-to-video": _priced(
        "video tokens at $1.80 per 1M - a 720p 5s video is roughly $0.18",
        _seedance_tokens(1.8, "a 720p 5s video is roughly $0.18"),
    ),
    "fal-ai/seedvr/upscale/video": _priced(
        "$0.001 per megapixel of video (width x height x frames)",
        _seedvr_upscale_video,
    ),
    # ---- training -------------------------------------------------------
    "fal-ai/z-image-turbo-trainer-v2": _priced(
        "$0.85 per 1000 training steps, billed with a 500-step floor",
        _z_image_trainer,
    ),
    # ---- no per-call price published -------------------------------------
    "openai/gpt-image-2/edit": _unpriced(
        "token-billed: text $5.00/1M in and $10.00/1M out, image $8.00/1M in and $30.00/1M out; the quality parameter moves this a lot",
        TOKEN_BILLED,
    ),
    "openai/gpt-image-2.5/sunburst/edit": _unpriced(
        "token-billed: text $5.00/1M in and $10.00/1M out, image $8.00/1M in and $30.00/1M out; the quality parameter moves this a lot",
        TOKEN_BILLED,
    ),
    "google/gemini-omni-flash/edit": _unpriced(
        "token-billed: $1.875 per 1M input tokens, $21.875 per 1M output tokens",
        TOKEN_BILLED,
    ),
    "fal-ai/florence-2-large/caption-to-phrase-grounding": _unpriced(
        "billed per GPU-second on an A100", GPU_TIME,
    ),
    "fal-ai/florence-2-large/more-detailed-caption": _unpriced(
        "billed per GPU-second on an A100", GPU_TIME,
    ),
    "fal-ai/florence-2-large/referring-expression-segmentation": _unpriced(
        "billed per GPU-second on an A100", GPU_TIME,
    ),
    "rundiffusion-fal/juggernaut-flux/pro/image-to-image": _unpriced(
        "billed per GPU-second on an A100", GPU_TIME,
    ),
    "fal-ai/bria/background/replace": _unpriced(
        "billed per GPU-second on an H100", GPU_TIME,
    ),
    "bria/replace-background": _unpriced(
        "no published per-call price", NO_PUBLISHED_PRICE,
    ),
    "fal-ai/flux-pro/kontext/max/multi": _unpriced(
        "no published per-call price", NO_PUBLISHED_PRICE,
    ),
    "fal-ai/ideogram/upscale": _unpriced(
        "no published per-call price", NO_PUBLISHED_PRICE,
    ),
    "fal-ai/image-editing/retouch": _unpriced(
        "no published per-call price", NO_PUBLISHED_PRICE,
    ),
    "fal-ai/pasd": _unpriced(
        "no published per-call price", NO_PUBLISHED_PRICE,
    ),
    "fal-ai/seedvr/upscale/image/seamless": _unpriced(
        "no published per-call price", NO_PUBLISHED_PRICE,
    ),
    "fal-ai/z-image/turbo/inpaint/lora": _priced(
        "$0.02 per output megapixel",
        # No rounding: fal states rounding for flux-pro fill but not for this
        # endpoint, so bill the literal megapixel count rather than overstating.
        _per_output_megapixel(0.02, round_up=False),
    ),
    "fal-ai/crystal-upscaler": _unpriced(
        "not in fal's catalogue under this id; clarityai/crystal-upscaler is $0.016 per output megapixel",
        NO_PUBLISHED_PRICE,
    ),
    # ---- LLM routers: billed by the upstream model's own token rates -------
    "openrouter/router/vision": _unpriced(
        "billed at the routed model's own token rates (the `model` widget picks it)",
        TOKEN_BILLED,
    ),
    "openrouter/router": _unpriced(
        "billed at the routed model's own token rates (the `model` widget picks it)",
        TOKEN_BILLED,
    ),
    "fal-ai/any-llm/vision": _unpriced(
        "billed at the routed model's own token rates (the `model` widget picks it)",
        TOKEN_BILLED,
    ),
    "fal-ai/any-llm": _unpriced(
        "billed at the routed model's own token rates (the `model` widget picks it)",
        TOKEN_BILLED,
    ),
    "fal-ai/topaz/upscale/image": _unpriced(
        "not in fal's catalogue under this id; the topaz/upscale/image/* variants bill $0.08 per started 2-24 MP depending on model",
        NO_PUBLISHED_PRICE,
    ),
}


# Which fal endpoint(s) each node class calls. Used to show the price note on
# the node itself. A node that can pick between endpoints lists all of them.
NODE_ENDPOINTS = {
    "SupersideAnyLLMTextNode": ["openrouter/router", "fal-ai/any-llm"],
    "SupersideAnyLLMVisionNode": ["openrouter/router/vision", "fal-ai/any-llm/vision"],
    "SupersideBriaBackgroundReplaceNode": ["fal-ai/bria/background/replace"],
    "SupersideBriaBackgroundStandardizerNode": ["fal-ai/bria/background/remove"],
    "SupersideBriaReplaceBackgroundNode": ["bria/replace-background"],
    "SupersideCrystalUpscalerNode": ["fal-ai/crystal-upscaler"],
    "SupersideFlorence2CaptionNode": ["fal-ai/florence-2-large/more-detailed-caption"],
    "SupersideFlorence2RegionSelectorNode": [
        "fal-ai/florence-2-large/caption-to-phrase-grounding",
        "fal-ai/florence-2-large/referring-expression-segmentation",
    ],
    "SupersideFluxKontextMaxMultiImageNode": ["fal-ai/flux-pro/kontext/max/multi"],
    "SupersideFluxProFillNode": ["fal-ai/flux-pro/v1/fill"],
    "SupersideGeminiOmniFlashEditNode": ["google/gemini-omni-flash/edit"],
    "SupersideGPTImage2EditNode": ["openai/gpt-image-2/edit"],
    "SupersideGPTImage25SunburstEditNode": ["openai/gpt-image-2.5/sunburst/edit"],
    "SupersideGrokImagineImageQualityEditNode": ["xai/grok-imagine-image/quality/edit"],
    "SupersideGrokImagineImageV2EditNode": ["xai/grok-imagine-image/v2.0/edit"],
    "SupersideIdeogramUpscaleNode": ["fal-ai/ideogram/upscale"],
    "SupersideImageRetouchNode": ["fal-ai/image-editing/retouch"],
    "SupersideJuggernautFluxProImg2ImgNode": ["rundiffusion-fal/juggernaut-flux/pro/image-to-image"],
    "SupersideKling21ImageToVideoNode": [
        "fal-ai/kling-video/v2.1/standard/image-to-video",
        "fal-ai/kling-video/v2.1/pro/image-to-video",
        "fal-ai/kling-video/v2.1/master/image-to-video",
    ],
    "SupersideKling25TurboProImageToVideoNode": ["fal-ai/kling-video/v2.5-turbo/pro/image-to-video"],
    "SupersideNanoBananaProEditNode": ["fal-ai/nano-banana-pro/edit"],
    "SupersideNanoBananaV2EditNode": ["fal-ai/nano-banana-2/edit"],
    "SupersidePASDUpscalerNode": ["fal-ai/pasd"],
    "SupersidePortraitSectionsNode": ["fal-ai/sam-3/image"],
    "SupersideSAM3RegionSelectorNode": ["fal-ai/sam-3/image"],
    "SupersideSceneExclusionMaskNode": ["fal-ai/sam-3/image"],
    "SupersideSeedanceLiteImageToVideoNode": ["fal-ai/bytedance/seedance/v1/lite/reference-to-video"],
    "SupersideSeedanceProImageToVideoNode": ["fal-ai/bytedance/seedance/v1/pro/image-to-video"],
    "SupersideSeedreamV45EditNode": ["fal-ai/bytedance/seedream/v4.5/edit"],
    "SupersideSeedreamV5ProEditNode": ["bytedance/seedream/v5/pro/edit"],
    "SupersideSeedVR2UpscaleImageNode": ["fal-ai/seedvr/upscale/image/seamless"],
    "SupersideSeedVRUpscaleVideoNode": ["fal-ai/seedvr/upscale/video"],
    "SupersideSmartDetailSheetNode": ["fal-ai/florence-2-large/caption-to-phrase-grounding"],
    "SupersideTopazUpscaleImageNode": ["fal-ai/topaz/upscale/image"],
    "SupersideWan25ImageToImageNode": ["fal-ai/wan-25-preview/image-to-image"],
    "SupersideWan25ImageToVideoNode": ["fal-ai/wan-25-preview/image-to-video"],
    "SupersideZImageInpaintLoraNode": ["fal-ai/z-image/turbo/inpaint/lora"],
    "SupersideZImageLoraTrainerNode": ["fal-ai/z-image-turbo-trainer-v2"],
}


# Snapshot of the dollar amounts fal published for each endpoint when the notes
# above were written. `--check` diffs the live catalogue against this, so a
# reworded blurb stays quiet and a real price move always shows up. An empty
# list means fal published no price text for that endpoint id at capture time
# (its note, where there is one, came from the model page's own cost table).
PUBLISHED_AMOUNTS = {
    'bria/replace-background': [],
    'bytedance/seedream/v5/pro/edit': ['$0.0045', '$0.0675', '$0.135'],
    'fal-ai/bria/background/remove': [],
    'fal-ai/bria/background/replace': [],
    'fal-ai/bytedance/seedance/v1/lite/reference-to-video': [],
    'fal-ai/bytedance/seedance/v1/pro/image-to-video': ['$0.62', '$2.5'],
    'fal-ai/bytedance/seedream/v4.5/edit': [],
    'fal-ai/crystal-upscaler': [],
    'fal-ai/florence-2-large/caption-to-phrase-grounding': [],
    'fal-ai/florence-2-large/more-detailed-caption': [],
    'fal-ai/florence-2-large/referring-expression-segmentation': [],
    'fal-ai/flux-pro/kontext/max/multi': [],
    'fal-ai/flux-pro/v1/fill': [],
    'fal-ai/ideogram/upscale': [],
    'fal-ai/image-editing/retouch': [],
    'fal-ai/kling-video/v2.1/master/image-to-video': ['$0.28', '$1.40'],
    'fal-ai/kling-video/v2.1/pro/image-to-video': ['$0.098', '$0.49'],
    'fal-ai/kling-video/v2.1/standard/image-to-video': ['$0.056', '$0.28'],
    'fal-ai/kling-video/v2.5-turbo/pro/image-to-video': ['$0.07', '$0.35'],
    'fal-ai/nano-banana-2/edit': ['$0.002', '$0.015', '$0.08', '$1.00'],
    'fal-ai/nano-banana-pro/edit': ['$0.015', '$0.15', '$1.00'],
    'fal-ai/pasd': [],
    'fal-ai/sam-3/image': ['$0.005'],
    'fal-ai/seedvr/upscale/image/seamless': [],
    'fal-ai/seedvr/upscale/video': ['$0.001', '$0.25'],
    'fal-ai/topaz/upscale/image': [],
    'fal-ai/wan-25-preview/image-to-image': ['$0.05'],
    'fal-ai/wan-25-preview/image-to-video': ['$0.05', '$0.10', '$0.15'],
    'fal-ai/z-image-turbo-trainer-v2': ['$0.85', '$1.70'],
    'fal-ai/z-image/turbo/inpaint/lora': [],
    'google/gemini-omni-flash/edit': ['$0.13', '$1.875', '$21.875'],
    'openai/gpt-image-2/edit': ['$0.0001', '$1.25', '$10.00', '$2.00', '$30.00', '$5.00', '$8.00'],
    'rundiffusion-fal/juggernaut-flux/pro/image-to-image': [],
    'xai/grok-imagine-image/quality/edit': ['$0.01', '$0.05', '$0.07'],
    'xai/grok-imagine-image/v2.0/edit': ['$0.01', '$0.04', '$0.06', '$0.08'],
}

# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

# Per-call prices measured by hand, for the endpoints fal bills by GPU-second
# or by token and publishes no per-call figure for. Read the real number off
# fal's usage dashboard (a single request's cost) and add it here; the cost
# report then folds those calls into the total instead of listing them as
# unpriced, and labels them as measured rather than published.
#
#   MANUAL_PRICES = {
#       "fal-ai/florence-2-large/caption-to-phrase-grounding": 0.0012,
#       "openrouter/router/vision": 0.004,
#   }
#
# Keys are endpoint ids, values are USD per call.
MANUAL_PRICES = {}


def get_note(endpoint):
    """The human-readable fal price for one endpoint, or None if unknown."""
    entry = PRICES.get(endpoint)
    return entry["note"] if entry else None


def is_priced(endpoint):
    entry = PRICES.get(endpoint)
    return bool(entry and entry["estimator"])


def estimate(endpoint, arguments, result):
    """
    Cost of a single call.

    Returns a dict: usd (float or None), detail (str), note (str or None),
    reason (str or None - why usd is None).
    """
    manual = MANUAL_PRICES.get(endpoint)
    entry = PRICES.get(endpoint)

    if entry is None:
        if manual is not None:
            return {
                "usd": float(manual),
                "detail": "${:.4f} per call, measured by hand".format(manual),
                "note": None,
                "reason": None,
            }
        return {
            "usd": None,
            "detail": "endpoint not in the price table",
            "note": None,
            "reason": "this endpoint is not listed in modules/fal_pricing.py yet",
        }

    if entry["estimator"] is None:
        if manual is not None:
            return {
                "usd": float(manual),
                "detail": "${:.4f} per call, measured by hand ({})".format(manual, entry["note"]),
                "note": entry["note"],
                "reason": None,
            }
        return {
            "usd": None,
            "detail": entry["note"],
            "note": entry["note"],
            "reason": entry["reason"],
        }

    try:
        usd, detail = entry["estimator"](arguments or {}, result or {})
    except Exception as exc:  # never let pricing break a generation
        return {
            "usd": None,
            "detail": "cost estimate failed ({})".format(exc.__class__.__name__),
            "note": entry["note"],
            "reason": "the cost estimator raised {}: {}".format(exc.__class__.__name__, exc),
        }

    return {
        "usd": usd,
        "detail": detail,
        "note": entry["note"],
        "reason": None if usd is not None else "not enough information in the response to price this call",
    }


def _short_labels(endpoints):
    """
    Shortest labels that still tell a node's endpoints apart, e.g. the three
    Kling tiers 'standard' / 'pro' / 'master' rather than their full ids.
    """
    parts = [endpoint.split("/") for endpoint in endpoints]
    shortest = min(len(p) for p in parts)
    varying = [
        i for i in range(shortest)
        if len({p[i] for p in parts}) > 1
    ]
    if not varying:
        return [endpoint.split("/")[-1] for endpoint in endpoints]
    return ["/".join(p[i] for i in varying) for p in parts]


def price_note_for_node(class_name):
    """
    One-line price note for a node class, e.g. for its tooltip.

    Returns None for nodes that make no fal call (local/utility nodes).
    """
    endpoints = NODE_ENDPOINTS.get(class_name)
    if not endpoints:
        return None

    notes = [get_note(endpoint) or "price unknown" for endpoint in endpoints]

    if len(set(notes)) == 1:
        # One price covers every endpoint the node can call - no need to label.
        body = notes[0]
    else:
        labels = _short_labels(endpoints)
        body = " | ".join(
            "{}: {}".format(label, note) for label, note in zip(labels, notes)
        )
    return "fal price ({}): {}".format(SOURCE_DATE, body)


def _check_against_fal():  # pragma: no cover - manual maintenance helper
    """
    Re-read fal's catalogue and report drift against PUBLISHED_AMOUNTS.

    Comparing the amount sets rather than the prose means a reworded blurb does
    not raise a false alarm, while a genuine price move always does.
    """
    import re
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    live = {}
    page, pages = 1, 1
    while page <= pages:
        response = session.get(
            "https://fal.ai/api/models", params={"page": page, "size": 40}, timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        pages = payload["pages"]
        for item in payload["items"]:
            live[item["id"]] = item.get("pricingInfoOverride")
        page += 1

    def resolves(endpoint):
        """
        Is the endpoint id still live?

        fal's model listing is not a complete inventory - aliases and unlisted
        endpoints are missing from it while working perfectly. The per-endpoint
        queue schema is the real existence check: 200 means callable, 404 means
        gone.
        """
        try:
            response = session.get(
                "https://fal.ai/api/openapi/queue/openapi.json",
                params={"endpoint_id": endpoint},
                timeout=45,
            )
            return response.status_code == 200
        except Exception:
            return None  # network problem, not a verdict

    amount_re = re.compile(r"\$[0-9]+(?:\.[0-9]+)?")
    findings = 0

    print("checked {} endpoints against {} live fal models\n".format(len(PRICES), len(live)))
    for endpoint in sorted(PRICES):
        snapshot = PUBLISHED_AMOUNTS.get(endpoint, [])
        if endpoint not in live:
            callable_now = resolves(endpoint)
            if callable_now is True:
                print("[UNLISTED] {}\n     callable (its queue schema resolves) but absent from "
                      "fal's listing, so no price is published for it - fine to keep "
                      "calling".format(endpoint))
            elif callable_now is False:
                print("[GONE    ] {}\n     fal no longer serves this endpoint id - the node "
                      "calling it needs a new id".format(endpoint))
                findings += 1
            else:
                print("[UNKNOWN ] {}\n     not in the listing and the schema probe failed - "
                      "check your connection".format(endpoint))
                findings += 1
            continue
        current = sorted(set(amount_re.findall(live[endpoint] or "")))
        if current == snapshot:
            continue
        if not snapshot:
            print("[NEW     ] {}\n     fal now publishes a price: {}".format(endpoint, current))
        else:
            print("[CHANGED ] {}\n     recorded : {}\n     now      : {}".format(
                endpoint, snapshot, current
            ))
        findings += 1

    if findings:
        print("\n{} endpoint(s) need attention - update the note, PUBLISHED_AMOUNTS and "
              "SOURCE_DATE.".format(findings))
    else:
        print("no drift: every recorded price still matches fal's catalogue.")


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--check" in sys.argv:
        _check_against_fal()
    else:
        for name in sorted(NODE_ENDPOINTS):
            print("{}\n    {}".format(name, price_note_for_node(name)))
