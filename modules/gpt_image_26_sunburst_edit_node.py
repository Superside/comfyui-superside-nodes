import io
import logging

import numpy as np
import torch
from PIL import Image, ImageSequence

from .gpt_image_25_sunburst_edit_node import SupersideGPTImage25SunburstEditNode

logger = logging.getLogger(__name__)

# fal's own bounds for the GIF controls, so the widgets cannot ask for
# something the endpoint will refuse.
FRAME_COUNT_MIN, FRAME_COUNT_MAX, FRAME_COUNT_DEFAULT = 2, 16, 9
FRAME_MS_MIN, FRAME_MS_MAX, FRAME_MS_DEFAULT = 20, 2000, 200


class SupersideGPTImage26SunburstEditNode(SupersideGPTImage25SunburstEditNode):
    """
    GPT Image 2.6 Sunburst Edit: openai/gpt-image-2.6/sunburst/edit.

    Everything 2.5 takes - multi-image references, a real mask_url, the
    image_size / quality / output_format controls, background and
    output_compression - so this subclasses that node and adds only what 2.6
    has on top:

      * reasoning_effort (low / medium / high, default high): how much the
        model plans before drawing, separate from quality. Higher helps with
        dense text and complex layouts, and costs latency.
      * markup_image: a transparent PNG of annotations - circles, arrows,
        notes - that directs the edit of the first image, rather than a mask
        saying where it may happen. fal refuses markup and mask together, so
        this node says so before spending the call.
      * GIF output: output_format gains "gif", with frame_count and
        frame_duration_ms driving the animation. The frames come back as an
        ordinary ComfyUI image batch.

    Tokens are priced exactly as 2.5 (text $5 in / $10 out, image $8 in / $30
    out per 1M), so the ledger records the call without a dollar estimate, like
    the rest of the gpt-image family. A bench on 2026-11-05 put 2.6 and 2.5
    about level on quality at the same cost per call, so the reason to reach
    for this node is the three features above rather than better output.
    """

    ENDPOINT = "openai/gpt-image-2.6/sunburst/edit"

    REASONING_OPTIONS = ["low", "medium", "high"]
    # 2.5 offers png / jpeg / webp; 2.6 adds an animated gif.
    OUTPUT_FORMATS = ["png", "jpeg", "webp", "gif"]

    DESCRIPTION = (
        "Edit images using OpenAI GPT Image 2.6 (Sunburst) on fal. Everything "
        "2.5 does - multi-image references, mask-based editing, preset or custom "
        "sizes, quality up to 'max' - plus the three things 2.6 adds: "
        "reasoning_effort (planning depth, separate from quality), markup_image "
        "(a transparent PNG of arrows and notes that directs the edit instead of "
        "a mask), and animated GIF output, whose frames arrive as an image batch. "
        "Cost is per token and rises steeply with quality."
    )

    # ------------------------------------------------------------- interface

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        opt = spec["optional"]

        # Replace the inherited three-way format with the one 2.6 accepts.
        opt["output_format"] = (cls.OUTPUT_FORMATS, {
            "default": "png",
            "tooltip": "Output format. 'gif' returns an animation, and its "
                       "frames leave this node as an image batch - shape it "
                       "with frame_count and frame_duration_ms.",
        })
        opt["reasoning_effort"] = (cls.REASONING_OPTIONS, {
            "default": "high",
            "tooltip": "How much the model plans before drawing, separate from "
                       "quality. 'high' is the API default; lower is faster.",
        })
        opt["markup_image"] = ("IMAGE", {
            "tooltip": "Annotations - circles, arrows, notes - drawn over the "
                       "first image to direct the edit. Must be the same size "
                       "as image_1, and cannot be combined with a mask (set "
                       "mask_mode to 'off'). Needs markup_transparency too, so "
                       "the model sees the marks and not a repainted frame.",
        })
        opt["markup_transparency"] = ("MASK", {
            "tooltip": "Which parts of markup_image are see-through, in "
                       "LoadImage's convention: WHITE where the PNG is "
                       "transparent, BLACK on the annotations themselves. Wire "
                       "LoadImage's MASK output straight in.",
        })
        opt["frame_count"] = ("INT", {
            "default": FRAME_COUNT_DEFAULT,
            "min": FRAME_COUNT_MIN, "max": FRAME_COUNT_MAX, "step": 1,
            "tooltip": "GIF only: how many frames. Each is rendered at "
                       "image_size, so a large size and a high count together "
                       "can exceed the request's pixel budget. Ignored by the "
                       "still formats.",
        })
        opt["frame_duration_ms"] = ("INT", {
            "default": FRAME_MS_DEFAULT,
            "min": FRAME_MS_MIN, "max": FRAME_MS_MAX, "step": 10,
            "tooltip": "GIF only: milliseconds each frame is held, rounded to "
                       "10 ms - 200 ms is five frames a second. Ignored by the "
                       "still formats.",
        })
        return spec

    # --------------------------------------------------------------- request

    def _markup_url(self, client, **kwargs):
        """Upload the annotation layer as a transparent PNG, or return None."""
        markup = kwargs.get("markup_image")
        if markup is None:
            return None

        if kwargs.get("_send_mask"):
            raise ValueError(
                "markup_image and a mask cannot be sent together - fal rejects "
                "the request. Set mask_mode to 'off' to use the markup, or "
                "disconnect markup_image to use the mask."
            )

        transparency = kwargs.get("markup_transparency")
        if transparency is None:
            raise ValueError(
                "markup_image needs markup_transparency as well. fal expects a "
                "PNG carrying only the annotations, so without it the whole "
                "frame goes up as opaque paint over the image. Wire the MASK "
                "output of the LoadImage node that loaded the markup."
            )

        rgb = self._tensor_to_pil(markup)

        base = kwargs.get("image_1")
        if base is not None:
            base_size = self._tensor_to_pil(base).size
            if rgb.size != base_size:
                raise ValueError(
                    "markup_image is %dx%d but image_1 is %dx%d - fal requires "
                    "them to match, since the annotations are read against the "
                    "first image's pixels."
                    % (rgb.size[0], rgb.size[1], base_size[0], base_size[1])
                )

        # ComfyUI hands transparency over as a MASK that is white where the
        # image is see-through, so the alpha channel is its inverse.
        mask = transparency
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask, dtype=np.float32)
        while mask.ndim > 2:
            mask = mask[0]
        alpha = np.clip(1.0 - mask, 0.0, 1.0)

        alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), "L")
        if alpha_image.size != rgb.size:
            alpha_image = alpha_image.resize(rgb.size, Image.NEAREST)

        rgba = rgb.convert("RGBA")
        rgba.putalpha(alpha_image)
        return self._upload_pil_png(client, rgba)

    def prepare_arguments(self, client, prompt, **kwargs):
        arguments = super().prepare_arguments(client, prompt, **kwargs)

        effort = kwargs.get("reasoning_effort")
        if effort:
            arguments["reasoning_effort"] = effort

        markup_url = self._markup_url(client, **kwargs)
        if markup_url:
            arguments["markup_image_url"] = markup_url

        # The frame controls mean nothing outside an animation, and sending
        # them for a png would only muddy what was actually requested.
        if kwargs.get("output_format") == "gif":
            if kwargs.get("frame_count"):
                arguments["frame_count"] = int(kwargs["frame_count"])
            if kwargs.get("frame_duration_ms"):
                arguments["frame_duration_ms"] = int(kwargs["frame_duration_ms"])

        return arguments

    # -------------------------------------------------------------- response

    def process_images(self, result):
        """As the base version, except an animation arrives as a frame batch.

        The inherited path opens whatever came back with PIL and keeps the
        first frame, which for a GIF would throw the animation away silently.
        """
        entries = result.get("images")
        if not entries:
            raise RuntimeError("No images were generated by the API.")

        frames = []
        for index, info in enumerate(entries):
            if not isinstance(info, dict) or "url" not in info:
                logger.error("Invalid image info for image %d", index)
                continue
            try:
                raw = self._load_image_bytes_from_url(info["url"])
                image = Image.open(io.BytesIO(raw))
                count = getattr(image, "n_frames", 1)
                if count > 1:
                    logger.info("%s returned an animation of %d frames",
                                self.ENDPOINT, count)
                for frame in ImageSequence.Iterator(image):
                    array = np.asarray(frame.convert("RGB")).astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(array).unsqueeze(0))
            except Exception as exc:
                logger.error("Failed to process image %d: %s", index, exc)
                continue

        if not frames:
            raise RuntimeError("Failed to process any generated images.")
        return (torch.cat(frames, dim=0),)

    def generate(self, prompt, api_key, **kwargs):
        """The parent's flow, with failures reported under this node's name.

        The parent wraps everything as "GPT Image 2 edit failed", which sends
        anyone debugging this node to the wrong model.
        """
        try:
            output, info = super().generate(prompt, api_key, **kwargs)
        except Exception as exc:
            message = str(exc).replace("GPT Image 2 edit failed: ", "")
            raise RuntimeError(
                "GPT Image 2.6 Sunburst edit failed: %s" % message) from exc

        if kwargs.get("output_format") == "gif" and int(output.shape[0]) == 1:
            logger.warning(
                "gif was requested but one frame came back - check frame_count, "
                "and whether image_size times frame_count exceeded the pixel "
                "budget."
            )
        return (output, info)
