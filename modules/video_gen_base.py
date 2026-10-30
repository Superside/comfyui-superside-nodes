"""Shared machinery for the fal video-generation nodes.

Every one of these models takes the same shape of request - a prompt, some
media, a handful of knobs - and answers with a video URL. What differs between
them is only which fields exist, what each field accepts, and what it defaults
to. Writing that out by hand once per node means two dozen near-identical files
that drift away from fal the first time a model gains an option.

So the field tables live in `video_model_specs.json`, generated from fal's own
published OpenAPI schemas (see tools/refresh_video_specs.py), and the widgets
are built from them here. Enums, ranges, defaults and tooltips are whatever fal
says they are. Regenerating the file picks up upstream changes without touching
this code.

A node may cover several endpoints that share one control surface - Kling 3
pro/standard/4k/turbo, MiniMax h3/h3-max/h3-max-turbo, Wan 3.0 standard/prime.
Those become a `model` dropdown rather than separate nodes, since they differ in
price and quality rather than in how you drive them. A field that only some of
those endpoints accept is still offered, and simply left out of the request when
the chosen one does not take it.

Nodes return ComfyUI's native VIDEO alongside the URL, so a SaveVideo or
PreviewVideo node wires straight in.
"""

import json
import logging
import os

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    VideoProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)

SPEC_PATH = os.path.join(os.path.dirname(__file__), "video_model_specs.json")

# How many sockets to offer for a field that takes a list of media. fal accepts
# more in some cases, but a node with twenty optional sockets is unusable; past
# this, pass URLs through the matching *_urls text field instead.
MEDIA_LIST_SLOTS = 4


def load_specs():
    """Read the generated model table that ships beside this module."""
    with open(SPEC_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sentinel(minimum):
    """The widget value that means "leave this out and let the model decide".

    Numeric fields fal declares as nullable need a way to say "unset" through a
    ComfyUI widget, which always holds a number. One below the floor would read
    oddly for a duration (1 second looks like a real request), so anything with
    a positive floor uses 0 and a seed, whose floor is 0, uses -1.
    """
    return -1 if minimum <= 0 else 0


def _tooltip(field):
    """fal's own description, plus whatever the widget adds on top of it."""
    parts = []
    if field.get("desc"):
        parts.append(field["desc"])
    kind = field["kind"]
    if field.get("options_by_tier"):
        parts.append("Not every model takes every value here: "
                     + "; ".join("%s accepts %s" % (tier, ", ".join(options))
                                 for tier, options in field["options_by_tier"].items())
                     + ".")
    if field.get("required_in") and not field["required"]:
        parts.append("Required by: %s." % ", ".join(field["required_in"]))
    if kind in ("int", "float") and field.get("nullable"):
        parts.append("%s = let the model choose." % _sentinel(field.get("min", 0)))
    elif kind == "url_list":
        parts.append("One URL per line.")
    elif kind == "str_list":
        parts.append("Comma-separated.")
    elif kind == "json":
        parts.append("JSON, as fal documents this field.")
    return " ".join(parts)


def _widget(field):
    """Translate one generated field spec into a ComfyUI input declaration."""
    kind = field["kind"]
    tip = _tooltip(field)

    if kind == "image":
        return ("IMAGE", {"tooltip": tip})
    if kind == "video":
        return ("VIDEO", {"tooltip": tip})
    if kind == "enum":
        options = list(field["options"])
        default = field.get("default", options[0])
        if default not in options:
            options.insert(0, default)
        return (options, {"default": default, "tooltip": tip})
    if kind == "bool":
        return ("BOOLEAN", {"default": bool(field.get("default", False)),
                            "tooltip": tip})
    if kind == "int":
        minimum = int(field.get("min", 0))
        default = field.get("default")
        low = _sentinel(minimum) if field.get("nullable") else minimum
        if default is None:
            default = low
        return ("INT", {"default": int(default), "min": low,
                        "max": int(field.get("max", 2147483647)),
                        "tooltip": tip})
    if kind == "float":
        minimum = float(field.get("min", 0.0))
        default = field.get("default")
        low = float(_sentinel(minimum)) if field.get("nullable") else minimum
        if default is None:
            default = low
        return ("FLOAT", {"default": float(default), "min": low,
                          "max": float(field.get("max", 100.0)),
                          "step": 0.01, "tooltip": tip})
    multiline = kind in ("url_list", "json") or field.get("multiline", False)
    return ("STRING", {"default": str(field.get("default", "") or ""),
                       "multiline": multiline, "tooltip": tip})


def _socket_names(field):
    """The ComfyUI input name(s) a field is exposed under."""
    if field["kind"] in ("image_list", "video_list"):
        stem = field["name"]
        for suffix in ("_urls", "_url"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return ["%s_%d" % (stem, i) for i in range(1, MEDIA_LIST_SLOTS + 1)]
    return [field["name"]]


class SupersideVideoGenNode(
    SupersideFalNode,
    ImageProcessingMixin,
    VideoProcessingMixin,
    APIClientMixin,
):
    """Base for the generated fal video nodes. Subclasses only set SPEC."""

    SPEC = None

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_url")
    FUNCTION = "generate"
    CATEGORY = "Superside"

    # ------------------------------------------------------------- interface

    @classmethod
    def INPUT_TYPES(cls):
        spec = cls.SPEC
        required = {}
        optional = {}

        if len(spec["variants"]) > 1:
            labels = list(spec["variants"])
            required["model"] = (labels, {
                "default": labels[0],
                "tooltip": "Which endpoint to call. These share one control "
                           "surface and differ in quality, speed and price:\n"
                           + "\n".join("  %s - %s" % (k, v)
                                       for k, v in spec["variants"].items()),
            })

        # The prompt leads, where there is one, because it is what the node is for.
        fields = sorted(spec["fields"], key=lambda f: f["name"] != "prompt")
        for field in fields:
            for name in _socket_names(field):
                if field["kind"] == "image_list":
                    decl = ("IMAGE", {"tooltip": _tooltip(field)})
                elif field["kind"] == "video_list":
                    decl = ("VIDEO", {"tooltip": _tooltip(field)})
                else:
                    decl = _widget(field)
                # Only a field every endpoint demands is required; a media list
                # stays optional so the caller can fill as many slots as needed.
                if field["required"] and field["kind"] not in ("image_list", "video_list"):
                    required[name] = decl
                else:
                    optional[name] = decl

        required["api_key"] = API_KEY_INPUT_SPEC
        return {"required": required, "optional": optional}

    @classmethod
    def endpoint_for(cls, model=None):
        variants = cls.SPEC["variants"]
        if model and model in variants:
            return variants[model]
        return next(iter(variants.values()))

    # -------------------------------------------------------------- request

    def _field_value(self, client, field, kwargs):
        """Turn one node input into the value fal expects, or None to omit it."""
        kind = field["kind"]
        name = field["name"]

        if kind == "image":
            image = kwargs.get(name)
            return self.upload_image(client, image) if image is not None else None

        if kind == "video":
            video = kwargs.get(name)
            return self.upload_video(client, video) if video is not None else None

        if kind in ("image_list", "video_list"):
            uploads = []
            for socket in _socket_names(field):
                media = kwargs.get(socket)
                if media is None:
                    continue
                uploads.append(self.upload_image(client, media)
                               if kind == "image_list"
                               else self.upload_video(client, media))
            return uploads or None

        if kind == "url_list":
            raw = (kwargs.get(name) or "").strip()
            urls = [line.strip() for line in raw.splitlines() if line.strip()]
            return urls or None

        if kind == "str_list":
            raw = (kwargs.get(name) or "").strip()
            items = [part.strip() for part in raw.split(",") if part.strip()]
            return items or None

        if kind == "json":
            raw = (kwargs.get(name) or "").strip()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "%s expects JSON, as fal documents it, and this did not "
                    "parse: %s" % (name, exc)
                ) from exc

        if kind == "bool":
            value = kwargs.get(name)
            return None if value is None else bool(value)

        if kind in ("int", "float"):
            value = kwargs.get(name)
            if value is None:
                return None
            if field.get("nullable") and value == _sentinel(field.get("min", 0)):
                return None
            return int(value) if kind == "int" else float(value)

        text = kwargs.get(name)
        if text is None:
            return None
        text = str(text)
        # An empty optional string means "not set", but a required one is the
        # caller's actual answer and goes through as it stands.
        if not text.strip() and not field["required"]:
            return None
        return text

    def _tier(self, model):
        """The tier label in play, which is the only one when there is one."""
        if model is not None:
            return model
        return next(iter(self.SPEC["variants"]))

    def prepare_arguments(self, client, model, **kwargs):
        tier = self._tier(model)
        arguments = {}

        for field in self.SPEC["fields"]:
            if tier not in field["tiers"]:
                # This endpoint does not take the field; sending it anyway is
                # how you earn a 422 from fal.
                continue
            value = self._field_value(client, field, kwargs)
            if value is None:
                continue
            # A field can offer more choices than the chosen endpoint accepts:
            # MiniMax h3 renders 2K and 4K, h3-max does not. Say so here rather
            # than letting fal answer with a validation error.
            legal = (field.get("options_by_tier") or {}).get(tier)
            if legal is not None and value not in legal:
                raise ValueError(
                    "%s=%r is not available on %s, which accepts %s."
                    % (field["name"], value, tier, ", ".join(legal))
                )
            arguments[field["name"]] = value

        missing = [f["name"] for f in self.SPEC["fields"]
                   if tier in f.get("required_in", [])
                   and f["name"] not in arguments]
        if missing:
            raise ValueError(
                "%s on %s needs %s - fal rejects the request without %s."
                % (self.SPEC["display"], tier, ", ".join(missing),
                   "them" if len(missing) > 1 else "it")
            )
        return arguments

    # -------------------------------------------------------------- response

    def process_video_result(self, result):
        """Pull the video URL out of the reply.

        The upscalers and most generators answer with `video`, a few with a
        `videos` list, so both are accepted rather than assuming one.
        """
        video = result.get("video")
        if isinstance(video, dict) and video.get("url"):
            return video["url"]
        videos = result.get("videos")
        if isinstance(videos, list) and videos and isinstance(videos[0], dict):
            url = videos[0].get("url")
            if url:
                return url
        raise RuntimeError(
            "No video came back from the model. Response was: %s"
            % str(result)[:300]
        )

    def generate(self, api_key, model=None, **kwargs):
        label = self.SPEC["display"]
        try:
            client = self.get_client(api_key)
            endpoint = self.endpoint_for(model)
            arguments = self.prepare_arguments(client, model, **kwargs)
            logger.info("%s: calling %s", label, endpoint)

            result = self.call_api(client, endpoint, arguments)
            video_url = self.process_video_result(result)
            return (self.download_video(video_url), video_url)
        except Exception as exc:
            logger.error("%s failed: %s", label, exc)
            raise RuntimeError("%s failed: %s" % (label, exc)) from exc


def build_node_class(spec):
    """Create the ComfyUI node class for one entry of the generated table."""
    endpoints = "\n".join("  %s - %s" % (tier, ep)
                          for tier, ep in spec["variants"].items())
    description = (
        "%s on fal. Returns a native VIDEO plus the URL, so SaveVideo or "
        "PreviewVideo connects straight in.\n\nEndpoint%s:\n%s"
        % (spec["display"].replace("Superside ", ""),
           "s" if len(spec["variants"]) > 1 else "", endpoints)
    )
    return type(spec["class"], (SupersideVideoGenNode,), {
        "SPEC": spec,
        "DISPLAY_NAME": spec["display"].replace("Superside ", ""),
        "DESCRIPTION": description,
        "__doc__": description,
    })


def build_all():
    """Every video node, keyed by class name, plus its display name."""
    classes = {}
    display = {}
    for spec in load_specs():
        classes[spec["class"]] = build_node_class(spec)
        display[spec["class"]] = spec["display"]
    return classes, display
