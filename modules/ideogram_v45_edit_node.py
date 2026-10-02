"""Ideogram V4.5 Edit: ideogram/v4.5/edit on fal.

Shares this package's mask vocabulary with the GPT Image nodes, so it is a
drop-in alternative wherever one of those sits: the same image_1 plus reference
images, the same mask_mode (off / soft / lock outside mask (hard)), the same
WHITE = edit convention, and the same local composite in hard mode.

Three things about this endpoint differ underneath, and are handled here rather
than left to the caller:

  * Ideogram reads BLACK as the area to edit - the opposite of every other node
    in this package. The mask is inverted on its way out, so a graph that feeds
    the same mask to Sunburst and to this node behaves the same in both.
  * references go in their own `reference_image_urls` list beside `image_url`,
    not in one combined list, and at most three are accepted alongside a mask.
  * it is seeded, so a fixed seed reproduces a run.

Measured against Sunburst on the same pipeline inputs (five eyewear cases, one
run each): on detail repair Ideogram at quality high fixed 3/3 where Sunburst
fixed 1/3, and it changed only the defect while Sunburst redrew the whole frame.
On reshaping an outline it was 0/2 in every configuration - it holds the
geometry it is given - where Sunburst was 2/2. So this is the node for correcting
detail, and not the one for changing a shape.
"""

import logging

from PIL import Image

from .gpt_image_2_edit_node import SupersideGPTImage2EditNode

logger = logging.getLogger(__name__)

# fal accepts at most three references when a mask is sent.
MAX_REFERENCES_WITH_MASK = 3


class SupersideIdeogramV45EditNode(SupersideGPTImage2EditNode):
    """Edit an image with Ideogram V4.5, with this package's mask conventions."""

    ENDPOINT = "ideogram/v4.5/edit"

    QUALITY_OPTIONS = ["high", "medium", "low", "turbo"]
    PRECISION_OPTIONS = ["high", "regular"]

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    DISPLAY_NAME = "Ideogram V4.5 Edit"
    DESCRIPTION = (
        "Edit an image with Ideogram V4.5 on fal. Takes the image to edit plus "
        "up to three reference images, and the same mask_mode as the GPT Image "
        "nodes - give it a WHITE = edit mask and it inverts internally, since "
        "Ideogram reads black as the editable area. Measured against Sunburst "
        "on five eyewear cases: better at repairing a detail (3/3 vs 1/3) and it "
        "changes only the defect, but it will not reshape an outline (0/2) - it "
        "keeps the geometry it is given."
    )

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        optional = dict(base.get("optional", {}))

        # Ideogram takes image_1 plus at most three references; drop the slots
        # the GPT nodes offer beyond that rather than accept them and silently
        # discard the extras at call time.
        for key in ("image_5", "image_6"):
            optional.pop(key, None)
        # `size` goes too: this endpoint is always called with image_size "auto",
        # so leaving the widget would be a control that silently does nothing.
        # output_format goes as well: the endpoint rejects it outright with
        # "extra_forbidden", so offering the widget would only produce a 422.
        for key in ("quality", "size", "image_size", "width", "height", "background",
                    "output_compression", "output_format", "sync_mode", "resolution"):
            optional.pop(key, None)

        optional["quality"] = (cls.QUALITY_OPTIONS, {
            "default": "high",
            "tooltip": "Rendering quality. 'high' repaired a detail 3/3 in testing "
                       "and 'medium' 2/3, at roughly a quarter of the price.",
        })
        optional["edit_precision"] = (cls.PRECISION_OPTIONS, {
            "default": "high",
            "tooltip": "How closely the edit is held to the masked region. 'high' "
                       "keeps more of the surrounding image untouched.",
        })
        optional["seed"] = ("INT", {
            "default": -1, "min": -1, "max": 2147483647,
            "tooltip": "Fixed seed for a reproducible run. -1 lets the service pick, "
                       "and the seed it used comes back in `info`.",
        })
        return {"required": base["required"], "optional": optional}

    def generate(self, prompt, api_key, **kwargs):
        """Inherit the parent's flow, but report failures under this node's name.

        The parent wraps every error as "GPT Image 2 edit failed", which sends
        anyone debugging this node to the wrong model.
        """
        try:
            return super().generate(prompt, api_key, **kwargs)
        except Exception as exc:
            message = str(exc).replace("GPT Image 2 edit failed: ", "")
            raise RuntimeError("Ideogram V4.5 edit failed: %s" % message) from exc

    def prepare_arguments(self, client, prompt, **kwargs):
        """Build Ideogram's argument shape.

        It differs from the GPT family in three ways, all of them handled here:
        the image to edit rides in `image_url` on its own, references go in
        `reference_image_urls`, and the mask is inverted because Ideogram treats
        black as the area to change.
        """
        image_1 = kwargs.get("image_1")
        if image_1 is None:
            raise ValueError("image_1 is required - it is the image being edited.")

        edited = self._tensor_to_pil(image_1)
        arguments = {
            "prompt": prompt,
            "image_url": self._upload_pil_png(client, edited),
            "image_size": "auto",
            "num_images": int(kwargs.get("num_images", 1) or 1),
            "quality": kwargs.get("quality", "high"),
            "edit_precision": kwargs.get("edit_precision", "high"),
        }

        seed = kwargs.get("seed", -1)
        if seed is not None and int(seed) >= 0:
            arguments["seed"] = int(seed)

        references = []
        for index in (2, 3, 4):
            reference = kwargs.get("image_%d" % index)
            if reference is not None:
                references.append(self._upload_pil_png(client, self._tensor_to_pil(reference)))

        mask_gray = kwargs.get("_mask_gray")
        sending_mask = bool(kwargs.get("_send_mask")) and mask_gray is not None

        if references:
            if sending_mask and len(references) > MAX_REFERENCES_WITH_MASK:
                logger.warning(
                    "Ideogram V4.5 accepts %d reference images alongside a mask; "
                    "%d were connected, so the extras were left out.",
                    MAX_REFERENCES_WITH_MASK, len(references))
                references = references[:MAX_REFERENCES_WITH_MASK]
            arguments["reference_image_urls"] = references

        if sending_mask:
            # Every other node in this package uses WHITE = edit. Ideogram reads
            # BLACK as edit, so invert on the way out and leave the caller's mask
            # convention alone. Resized to the edited image first: a mask at a
            # different size comes back misaligned.
            if mask_gray.size != edited.size:
                mask_gray = mask_gray.resize(edited.size, Image.LANCZOS)
            arguments["mask_url"] = self._upload_pil_png(
                client, Image.eval(mask_gray, lambda p: 255 - p))

        return arguments
