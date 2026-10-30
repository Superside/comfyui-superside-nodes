"""Regenerate modules/video_model_specs.json from fal's published schemas.

    python tools/refresh_video_specs.py

The video nodes build their widgets from that file, so this is how the package
picks up a new option, a changed enum or a new tier on any of these models.
Run it, read the diff, and commit the result with a note on what moved.

Endpoints with no public OpenAPI document are reported and skipped; they keep
whatever is already in the table.
"""

import json
import os
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "modules", "video_model_specs.json")
SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id="

_CACHE = {}


def load_input(endpoint):
    """Fetch an endpoint's input schema from fal."""
    if endpoint not in _CACHE:
        url = SCHEMA_URL + urllib.parse.quote(endpoint, safe="")
        with urllib.request.urlopen(url, timeout=60) as response:
            doc = json.load(response)
        schemas = doc.get("components", {}).get("schemas", {})
        name = next((k for k in schemas if k.lower().endswith("input")), None)
        _CACHE[endpoint] = schemas.get(name, {})
    return _CACHE[endpoint]

# One node per family x modality. The tier is a dropdown inside the node:
# tiers of one modality are the same control surface at a different price.
NODES = [
    ("wan_30_image_to_video", "Wan30ImageToVideo", "Wan 3.0 Image-to-Video",
     {"standard": "alibaba/wan-3.0/image-to-video",
      "prime": "alibaba/wan-3.0-prime/image-to-video"}),
    ("wan_30_text_to_video", "Wan30TextToVideo", "Wan 3.0 Text-to-Video",
     {"standard": "alibaba/wan-3.0/text-to-video",
      "prime": "alibaba/wan-3.0-prime/text-to-video"}),
    ("wan_30_reference_to_video", "Wan30ReferenceToVideo", "Wan 3.0 Reference-to-Video",
     {"standard": "alibaba/wan-3.0/reference-to-video",
      "prime": "alibaba/wan-3.0-prime/reference-to-video"}),

    ("kling_v3_image_to_video", "KlingV3ImageToVideo", "Kling 3 Image-to-Video",
     {"pro": "fal-ai/kling-video/v3/pro/image-to-video",
      "standard": "fal-ai/kling-video/v3/standard/image-to-video",
      "4k": "fal-ai/kling-video/v3/4k/image-to-video",
      "turbo pro": "fal-ai/kling-video/v3/turbo/pro/image-to-video",
      "turbo standard": "fal-ai/kling-video/v3/turbo/standard/image-to-video"}),
    ("kling_v3_text_to_video", "KlingV3TextToVideo", "Kling 3 Text-to-Video",
     {"pro": "fal-ai/kling-video/v3/pro/text-to-video",
      "standard": "fal-ai/kling-video/v3/standard/text-to-video",
      "4k": "fal-ai/kling-video/v3/4k/text-to-video",
      "turbo pro": "fal-ai/kling-video/v3/turbo/pro/text-to-video",
      "turbo standard": "fal-ai/kling-video/v3/turbo/standard/text-to-video"}),
    ("kling_v3_motion_control", "KlingV3MotionControl", "Kling 3 Motion Control",
     {"pro": "fal-ai/kling-video/v3/pro/motion-control",
      "standard": "fal-ai/kling-video/v3/standard/motion-control"}),

    ("kling_v4_image_to_video", "KlingV4ImageToVideo", "Kling 4 Image-to-Video",
     {"default": "kling/v4/image-to-video"}),
    ("kling_v4_text_to_video", "KlingV4TextToVideo", "Kling 4 Text-to-Video",
     {"default": "kling/v4/text-to-video"}),
    ("kling_v4_reference_to_video", "KlingV4ReferenceToVideo", "Kling 4 Reference-to-Video",
     {"default": "kling/v4/reference-to-video"}),
    ("kling_v4_edit_video", "KlingV4EditVideo", "Kling 4 Edit Video",
     {"default": "kling/v4/video-to-video/edit"}),

    ("seedance_25_image_to_video", "Seedance25ImageToVideo", "Seedance 2.5 Image-to-Video",
     {"default": "bytedance/seedance-2.5/us/image-to-video"}),
    ("seedance_25_text_to_video", "Seedance25TextToVideo", "Seedance 2.5 Text-to-Video",
     {"default": "bytedance/seedance-2.5/us/text-to-video"}),
    ("seedance_25_reference_to_video", "Seedance25ReferenceToVideo", "Seedance 2.5 Reference-to-Video",
     {"default": "bytedance/seedance-2.5/us/reference-to-video"}),

    ("minimax_h3_image_to_video", "MiniMaxH3ImageToVideo", "MiniMax H3 Image-to-Video",
     {"h3": "minimax/h3/image-to-video",
      "h3 max": "minimax/h3-max/image-to-video",
      "h3 max turbo": "minimax/h3-max-turbo/image-to-video"}),
    ("minimax_h3_text_to_video", "MiniMaxH3TextToVideo", "MiniMax H3 Text-to-Video",
     {"h3": "minimax/h3/text-to-video",
      "h3 max": "minimax/h3-max/text-to-video",
      "h3 max turbo": "minimax/h3-max-turbo/text-to-video"}),
    ("minimax_h3_reference_to_video", "MiniMaxH3ReferenceToVideo", "MiniMax H3 Reference-to-Video",
     {"h3": "minimax/h3/reference-to-video",
      "h3 max": "minimax/h3-max/reference-to-video"}),
    ("minimax_h3_reference_to_video_lora", "MiniMaxH3ReferenceToVideoLora",
     "MiniMax H3 Reference-to-Video LoRA",
     {"default": "minimax/h3/reference-to-video/lora"}),
    ("minimax_h3_lip_sync", "MiniMaxH3LipSync", "MiniMax H3 Max Lip Sync",
     {"default": "minimax/h3-max/lip-sync/image-to-video"}),
    ("minimax_h3_camera_controls", "MiniMaxH3CameraControls", "MiniMax H3 Max Camera Controls",
     {"default": "minimax/h3-max/camera-controls"}),
    ("minimax_h3_extend_video", "MiniMaxH3ExtendVideo", "MiniMax H3 Extend Video",
     {"h3 max": "minimax/h3-max/extend-video",
      "h3 max turbo": "minimax/h3-max-turbo/extend-video"}),
    ("minimax_h3_reframe_video", "MiniMaxH3ReframeVideo", "MiniMax H3 Max Reframe Video",
     {"default": "minimax/h3-max/reframe-video"}),
    ("minimax_h3_recast", "MiniMaxH3Recast", "MiniMax H3 Max Recast",
     {"default": "minimax/h3-max/recast"}),
    ("minimax_h3_3d_to_video", "MiniMaxH33DToVideo", "MiniMax H3 Max 3D-to-Video",
     {"default": "minimax/h3-max/3d-to-video"}),
    ("minimax_video_01_director", "MiniMaxVideo01Director", "MiniMax Video 01 Director",
     {"text-to-video": "fal-ai/minimax/video-01-director",
      "image-to-video": "fal-ai/minimax/video-01-director/image-to-video"}),
]

IMAGE_FIELDS = {"image_url", "start_image_url", "end_image_url", "middle_image_url"}
IMAGE_LIST_FIELDS = {"image_urls", "reference_image_urls", "middle_image_urls"}
VIDEO_FIELDS = {"video_url"}
VIDEO_LIST_FIELDS = {"video_urls", "reference_video_urls"}
URL_LIST_FIELDS = {"reference_audio_urls", "audio_urls"}
STR_LIST_FIELDS = {"voice_ids"}
# Arrays of objects, not of URLs: the caller supplies them as JSON.
JSON_FIELDS = {"elements", "multi_prompt", "loras", "reference_conditions",
               "camera_trajectory"}
# sync_mode returns the video base64-inline instead of a CDN URL, which is not
# what these nodes do with the result, so it is never offered.
SKIP_FIELDS = {"sync_mode"}
TEXT_URL_FIELDS = {"audio_url", "target_audio_url", "file_url", "web_url"}
MULTILINE = {"prompt", "negative_prompt", "multi_prompt", "elements", "loras",
             "reference_conditions", "camera_trajectory"}


def unwrap(prop):
    """Collapse fal's anyOf[T, null] into T, plus a nullable flag."""
    if "anyOf" in prop:
        concrete = [a for a in prop["anyOf"] if a.get("type") != "null"]
        if len(concrete) == 1:
            merged = dict(concrete[0])
            for key in ("default", "description", "title", "examples"):
                if key in prop and key not in merged:
                    merged[key] = prop[key]
            return merged, True
    return prop, False


def field_spec(name, raw, required):
    prop, nullable = unwrap(raw)
    typ = prop.get("type")
    base = {"name": name, "desc": (prop.get("description") or "").strip(),
            "required": required}

    if name in IMAGE_FIELDS:
        return dict(base, kind="image")
    if name in IMAGE_LIST_FIELDS:
        return dict(base, kind="image_list")
    if name in VIDEO_FIELDS:
        return dict(base, kind="video")
    if name in VIDEO_LIST_FIELDS:
        return dict(base, kind="video_list")
    if name in URL_LIST_FIELDS:
        return dict(base, kind="url_list")
    if name in STR_LIST_FIELDS:
        return dict(base, kind="str_list")
    if name in JSON_FIELDS:
        return dict(base, kind="json")
    if name in TEXT_URL_FIELDS:
        return dict(base, kind="url")
    if prop.get("enum"):
        return dict(base, kind="enum", options=list(prop["enum"]),
                    default=prop.get("default", prop["enum"][0]))
    # prompt_expansion_mode ships its choices as examples rather than an enum.
    if prop.get("examples") and typ == "string" and name.endswith("_mode"):
        opts = list(prop["examples"])
        return dict(base, kind="enum", options=opts,
                    default=prop.get("default", opts[0]))
    if typ == "boolean":
        return dict(base, kind="bool", default=bool(prop.get("default", False)))
    if typ == "integer":
        return dict(base, kind="int", default=prop.get("default", 0),
                    min=prop.get("minimum", 0),
                    max=prop.get("maximum", 2147483647), nullable=nullable)
    if typ == "number":
        return dict(base, kind="float", default=prop.get("default", 0.0),
                    min=prop.get("minimum", 0.0),
                    max=prop.get("maximum", 100.0), nullable=nullable)
    if typ == "array":
        return dict(base, kind="url_list")
    return dict(base, kind="text", multiline=name in MULTILINE,
                default=prop.get("default", "") or "")


def main():
    out = []
    for node_id, stem, display, variants in NODES:
        per_tier = {}
        for tier, endpoint in variants.items():
            try:
                schema = load_input(endpoint)
            except Exception as exc:
                print("  no schema for %s (%s) - skipped" % (endpoint, exc))
                continue
            required = set(schema.get("required", []))
            per_tier[tier] = {n: field_spec(n, p, n in required)
                              for n, p in schema.get("properties", {}).items()}

        if not per_tier:
            print("  %s has no reachable endpoint - left out" % stem)
            continue
        variants = {t: e for t, e in variants.items() if t in per_tier}

        ordered = []
        for tier in variants:
            for name in per_tier[tier]:
                if name not in ordered and name not in SKIP_FIELDS:
                    ordered.append(name)

        fields = []
        for name in ordered:
            tiers = [t for t in variants if name in per_tier[t]]
            spec = dict(next(per_tier[t][name] for t in tiers))
            spec["tiers"] = tiers
            # Which endpoints demand it. A field can be required on one tier and
            # optional on another (MiniMax asks h3-max for prompt_expansion_mode
            # but lets h3 omit it), and ComfyUI has no conditional required, so
            # the node offers it and checks against the chosen tier at run time.
            spec["required_in"] = [t for t in tiers if per_tier[t][name]["required"]]
            spec["required"] = (len(tiers) == len(variants)
                                and len(spec["required_in"]) == len(tiers))

            if spec["kind"] == "enum":
                by_tier = {t: list(per_tier[t][name].get("options") or []) for t in tiers}
                if len({tuple(v) for v in by_tier.values()}) > 1:
                    # Same field, different choices per tier - MiniMax h3 offers
                    # 2K and 4K where h3-max offers 1080P. The widget shows the
                    # union and the node rejects a value the chosen tier cannot
                    # take, which beats hiding options the user paid for.
                    union = []
                    for tier in tiers:
                        for option in by_tier[tier]:
                            if option not in union:
                                union.append(option)
                    shared = [o for o in union if all(o in by_tier[t] for t in tiers)]
                    defaults = [per_tier[t][name].get("default") for t in tiers]
                    preferred = next((d for d in defaults if d in shared), None)
                    spec["options"] = union
                    spec["options_by_tier"] = by_tier
                    spec["default"] = preferred or (shared[0] if shared else union[0])
            fields.append(spec)

        out.append({"id": node_id, "class": "Superside%sNode" % stem,
                    "display": "Superside %s" % display,
                    "variants": variants, "fields": fields})

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print("wrote", OUT)
    print("nodes:", len(out))
    for node in out:
        kinds = {}
        for f in node["fields"]:
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        print("  %-46s tiers=%d fields=%2d %s"
              % (node["class"], len(node["variants"]), len(node["fields"]), kinds))


if __name__ == "__main__":
    main()
