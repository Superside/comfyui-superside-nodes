import base64
import io
import json
import logging

import numpy as np
import requests
import torch
from PIL import Image

from .base_node import SupersideFalNode, ImageProcessingMixin, APIClientMixin, API_KEY_INPUT_SPEC

logger = logging.getLogger(__name__)

# Toggle name -> SAM 3 text prompt. Consolidates the original local
# face-parsing model's 19-class taxonomy (which had separate l_eye/r_eye,
# l_ear/r_ear, u_lip/l_lip, l_brow/r_brow classes) into one open-vocabulary
# SAM 3 prompt per toggle - SAM 3's "merge_all" selection mode already
# picks up both left+right instances of e.g. "eyes" in a single call, so
# this needs far fewer fal calls than one-per-side would.
SECTION_PROMPTS = {
    "skin": "facial skin",
    "nose": "nose",
    "eyes": "eyes",
    "eyebrows": "eyebrows",
    "ears": "ears",
    "mouth": "mouth",
    "lips": "lips",
    "hair": "hair",
    "hat": "hat",
    # "eyeglasses" (unqualified) reads as the whole glasses silhouette,
    # frame + lens together, to SAM 3 - which pulls the visible under-eye
    # skin behind the lens into the exclusion too, so that skin never gets
    # retouched. Worded to target the rigid frame/temples specifically;
    # overridable per-run via glasses_prompt_override below without editing
    # code, since how well this text-only distinction lands can vary by
    # glasses style/photo.
    "glasses": "eyeglasses frame only - the metal or plastic rim and temples, not the lens or the skin behind it",
    "earrings": "earrings",
    "neck": "neck",
    "necklace": "necklace",
    "clothing": "clothing",
}

# Defaults mirror the production values baked into the original local
# FaceParsingResultsParser "EXCLUSION" node in the re-skin workflow: nose,
# eyes, ears, mouth, lips, hair and hat were excluded from retouching;
# skin, eyebrows, neck and clothing were left in (retouched).
#
# "glasses" is added on top of that original set (the old 19-class local
# face-parsing model had no eyeglasses class at all, so it couldn't have
# been in that baseline). Rigid, thin, high-contrast objects like glasses
# frames are exactly what a diffusion inpaint pass struggles to reproduce
# pixel-for-pixel - if "glasses" isn't excluded, fal regenerates that area
# too and the result is a ghosted/doubled frame in the final composite,
# since nothing pastes the true original glasses pixels back afterward.
DEFAULT_ON = {"nose", "eyes", "ears", "mouth", "lips", "hair", "hat", "glasses"}

# One distinct color per section, purely for the color_preview visualization
# (like the old local FaceParsing node's colorized class overlay) - the
# actual "mask" output stays a plain grayscale merge, unaffected by this.
SECTION_COLORS = {
    "skin": (255, 0, 0),
    "nose": (0, 255, 0),
    "eyes": (0, 0, 255),
    "eyebrows": (255, 255, 0),
    "ears": (255, 0, 255),
    "mouth": (0, 255, 255),
    "lips": (255, 128, 0),
    "hair": (128, 0, 255),
    "hat": (0, 128, 128),
    "glasses": (128, 128, 0),
    "earrings": (255, 192, 203),
    "neck": (0, 160, 0),
    "necklace": (160, 160, 160),
    "clothing": (139, 69, 19),
}


class SupersidePortraitSectionsNode(SupersideFalNode, ImageProcessingMixin, APIClientMixin):
    """
    Superside Portrait Sections: in-house, fal-based replacement for the
    local comfyui_face_parsing package (FaceParsingProcessorLoader +
    FaceParsingModelLoader + FaceParse + FaceParsingResultsParser). Lets
    you toggle which facial/portrait sections to include in one output
    mask - e.g. as an EXCLUSION mask so a retouch pass skips eyes/lips/hair
    - using SAM 3 (fal-ai/sam-3/image) instead of a local segmentation
    model, one call per *active* toggle (not per class), merged with OR.

    Trade-off vs. the original local model: SAM 3 is open-vocabulary and
    promptable rather than a fixed, pixel-labeled taxonomy, so the exact
    boundary of e.g. "lips" may be slightly less crisp than a dedicated
    face-parsing network - and each active toggle is one extra fal.ai
    call (cost/latency), unlike the original's single local pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        toggles = {
            name: ("BOOLEAN", {"default": name in DEFAULT_ON})
            for name in SECTION_PROMPTS
        }
        # Per-section opacity: how strongly each section is written into the
        # exclusion mask. 1.0 (default) = fully white = the downstream composite
        # pastes 100% of the ORIGINAL back there (no retouch) - identical to the
        # old binary behavior. 0.5 = gray = composite blends 50% original / 50%
        # generated, i.e. a partial retouch on that section (e.g. clothing).
        # Only applies when the section's toggle is ON.
        opacities = {
            f"{name}_opacity": (
                "FLOAT",
                {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        f"Opacity of '{name}' in the exclusion mask (only when '{name}' is ON). "
                        "1.0 = fully preserve the original here (no retouch). 0.5 = partial "
                        "(50% original / 50% generated in the final composite). 0.0 = no exclusion."
                    ),
                },
            )
            for name in SECTION_PROMPTS
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                **toggles,
                **opacities,
                "glasses_prompt_override": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": SECTION_PROMPTS["glasses"],
                        "tooltip": (
                            "SAM 3 text query used for the 'glasses' toggle above. Edit "
                            "this to tune how tightly it targets just the frame/temples "
                            "vs. the whole glasses silhouette (frame+lens) - no code "
                            "changes needed. Only used when 'glasses' is ON."
                        ),
                    },
                ),
                "glasses_box_center_x": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 1_000_000,
                        "tooltip": (
                            "Optional grounding box for the 'glasses' section, in the "
                            "old GroundingDINO+SAM style: wire in a Florence-2 Smart "
                            "Region Selector's center_x/center_y/crop_width/crop_height "
                            "(query e.g. 'eyeglasses frame') and SAM 3 will refine that "
                            "box into a precise mask instead of relying on text alone. "
                            "Leave at -1 (unconnected) to use text-only prompting."
                        ),
                    },
                ),
                "glasses_box_center_y": ("INT", {"default": -1, "min": -1, "max": 1_000_000}),
                "glasses_box_width": ("INT", {"default": -1, "min": -1, "max": 1_000_000}),
                "glasses_box_height": ("INT", {"default": -1, "min": -1, "max": 1_000_000}),
                "padding_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "partial_feather_percent": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 25.0, "step": 0.5,
                        "tooltip": (
                            "Feather (soft edge) applied ONLY to sections whose opacity is "
                            "below 1.0 (e.g. clothing at 0.5). Softens the partial region's "
                            "border so the downstream composite blends it gradually instead "
                            "of showing a hard line / doubled edge. Full-opacity (1.0) "
                            "sections are untouched here - they keep their crisp edge for the "
                            "downstream grow/blur to handle. 0 = off."
                        ),
                    },
                ),
                "partial_contract_percent": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 25.0, "step": 0.5,
                        "tooltip": (
                            "Contract (erode inward) the mask of partial-opacity sections "
                            "BEFORE feathering, so the soft edge sits INSIDE the region "
                            "instead of bleeding outward past its true border (which shows as "
                            "a faint ring/edge on the surrounding skin). Set it near "
                            "partial_feather_percent to pull the feathered ramp back to the "
                            "real edge. Only affects partial (<1.0 opacity) sections. 0 = off."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "info", "color_preview")
    FUNCTION = "build_mask"
    DESCRIPTION = (
        "Toggle which portrait sections (eyes, lips, hair, ...) to include "
        "in one merged mask, using SAM 3 on fal.ai - in-house replacement "
        "for the local face-parsing model's per-region checklist."
    )

    @staticmethod
    def _scale_opacity(mask_uint8, opacity, feather_px=0.0, contract_px=0.0):
        """Scale a 0/255 section mask by a 0-1 opacity, optionally contracting
        (eroding inward) then feathering the edge - only meaningful for partial
        sections. opacity>=1 -> unchanged (identical to the old binary behavior)."""
        if opacity >= 1.0:
            return mask_uint8
        work = mask_uint8.astype(np.float32)
        if contract_px and contract_px > 0:
            from scipy.ndimage import grey_erosion
            k = int(round(contract_px)) * 2 + 1
            work = grey_erosion(work, size=(k, k), mode="constant", cval=0.0)
        scaled = work * max(0.0, float(opacity))
        if feather_px and feather_px > 0:
            from scipy.ndimage import gaussian_filter
            scaled = gaussian_filter(scaled, sigma=float(feather_px))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def _download_mask_array(self, image_ref, width, height):
        url = image_ref.get("url") if isinstance(image_ref, dict) else image_ref
        if not url:
            raise RuntimeError("SAM 3 returned a section without a mask URL.")
        if str(url).startswith("data:"):
            header, _, payload = url.partition(",")
            content = base64.b64decode(payload)
        else:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            content = response.content
        pil_image = Image.open(io.BytesIO(content))
        pil_image.load()
        if pil_image.mode == "RGBA":
            arr = np.asarray(pil_image.getchannel("A"), dtype=np.uint8)
        else:
            arr = np.asarray(pil_image.convert("L"), dtype=np.uint8)
        if arr.shape[:2] != (height, width):
            arr = np.asarray(
                Image.fromarray(arr, mode="L").resize((width, height), Image.BILINEAR), dtype=np.uint8
            )
        return arr

    def _sam3_mask_for_prompt(self, client, image_url, prompt, width, height, box_prompts=None):
        result = self.call_api(
            client,
            "fal-ai/sam-3/image",
            {
                "image_url": image_url,
                "prompt": prompt,
                "point_prompts": [],
                # A box here reproduces the old GroundingDINO+SAM two-stage
                # pattern: the box (e.g. from Florence-2 grounding) tells SAM
                # 3 exactly where to look, so it refines that region into a
                # precise mask instead of relying on text alone to both find
                # AND segment the target - text-only tends to grab the whole
                # object (e.g. glasses lens+frame) when a sub-part is meant.
                "box_prompts": box_prompts or [],
                "apply_mask": False,
                "output_format": "png",
                "return_multiple_masks": True,
                "max_masks": 8,
                "include_scores": False,
                "include_boxes": False,
            },
        )
        entries = result.get("masks") if isinstance(result, dict) else None
        if not entries:
            primary = result.get("image") if isinstance(result, dict) else None
            entries = [primary] if primary else []

        merged = np.zeros((height, width), dtype=np.uint8)
        for entry in entries:
            try:
                arr = self._download_mask_array(entry, width, height)
            except Exception as e:
                logger.warning(f"SAM 3 mask download failed for prompt '{prompt}': {e}")
                continue
            merged = np.maximum(merged, arr)
        return merged

    def build_mask(
        self,
        image,
        api_key,
        padding_percent=0.0,
        partial_feather_percent=0.0,
        partial_contract_percent=0.0,
        glasses_prompt_override="",
        glasses_box_center_x=-1,
        glasses_box_center_y=-1,
        glasses_box_width=-1,
        glasses_box_height=-1,
        **toggles,
    ):
        try:
            client = self.get_client(api_key)

            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError("Superside Portrait Sections currently supports batch size 1 only.")
            height, width = image.shape[1], image.shape[2]
            image_url = self.upload_image(client, image, max_dimension=2048)

            active = [name for name in SECTION_PROMPTS if toggles.get(name, False)]

            # Optional GroundingDINO+SAM-style hint for "glasses": a box from
            # an upstream grounding model (e.g. Florence-2 Smart Region
            # Selector with a "eyeglasses frame" query) tells SAM 3 exactly
            # where to look, so it refines that box into a mask rather than
            # having to both find AND segment the frame from text alone.
            glasses_box_prompts = []
            if (
                glasses_box_center_x >= 0
                and glasses_box_center_y >= 0
                and glasses_box_width > 0
                and glasses_box_height > 0
            ):
                x1 = max(0, glasses_box_center_x - glasses_box_width // 2)
                y1 = max(0, glasses_box_center_y - glasses_box_height // 2)
                x2 = min(width, glasses_box_center_x + glasses_box_width // 2)
                y2 = min(height, glasses_box_center_y + glasses_box_height // 2)
                # fal's box_prompts schema is a list of BoxPrompt *objects*
                # (x_min/y_min/x_max/y_max), not plain [x1,y1,x2,y2] arrays -
                # sending arrays 422s ("Input should be a valid dictionary or
                # object"), which was silently swallowed by the try/except
                # below and skipped the whole 'glasses' section (no box, no
                # text fallback), so nothing ever got excluded.
                glasses_box_prompts = [{"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}]

            # Feather / contract (in px) for partial sections, from the smaller edge.
            feather_px = min(width, height) * float(partial_feather_percent) / 100.0
            contract_px = min(width, height) * float(partial_contract_percent) / 100.0

            merged = np.zeros((height, width), dtype=np.uint8)
            used = []
            section_masks = []  # [(name, mask_uint8), ...] - kept for color_preview
            for name in active:
                prompt = SECTION_PROMPTS[name]
                if name == "glasses" and glasses_prompt_override and glasses_prompt_override.strip():
                    prompt = glasses_prompt_override.strip()
                box_prompts = glasses_box_prompts if name == "glasses" else []
                opacity = float(toggles.get(f"{name}_opacity", 1.0))
                try:
                    section_mask = self._sam3_mask_for_prompt(
                        client, image_url, prompt, width, height, box_prompts=box_prompts
                    )
                except Exception as e:
                    if box_prompts:
                        # Don't let a bad/failed box silently drop the whole
                        # section - fall back to text-only prompting so
                        # "glasses" still gets excluded even if the grounding
                        # box itself was rejected or came back empty.
                        logger.warning(
                            f"Section '{name}': box-prompted SAM 3 call failed ({e}); "
                            f"retrying with text-only prompt."
                        )
                        try:
                            section_mask = self._sam3_mask_for_prompt(
                                client, image_url, prompt, width, height, box_prompts=[]
                            )
                            merged = np.maximum(merged, self._scale_opacity(section_mask, opacity, feather_px, contract_px))
                            used.append(name)
                            section_masks.append((name, section_mask))
                            continue
                        except Exception as e2:
                            e = e2
                    logger.warning(f"Skipping section '{name}' (SAM 3 call failed): {e}")
                    continue
                merged = np.maximum(merged, self._scale_opacity(section_mask, opacity, feather_px, contract_px))
                used.append(name)
                section_masks.append((name, section_mask))

            # Colorized overlay preview - one distinct color per section,
            # blended over the original image. Purely visual (like the old
            # local FaceParsing node's colored class map); the real "mask"
            # output above stays a plain grayscale merge.
            base_np = image[0].detach().cpu().numpy()
            if base_np.max() <= 1.0:
                base_np = base_np * 255.0
            preview_np = np.clip(base_np, 0, 255).astype(np.float32)[:, :, :3].copy()
            alpha = 0.5
            for name, section_mask in section_masks:
                color = np.array(SECTION_COLORS.get(name, (255, 255, 255)), dtype=np.float32)
                m = (section_mask.astype(np.float32) / 255.0) * alpha
                m3 = m[:, :, None]
                preview_np = preview_np * (1.0 - m3) + color[None, None, :] * m3
            preview_tensor = torch.from_numpy(np.clip(preview_np, 0, 255).astype(np.float32) / 255.0).unsqueeze(0)

            if padding_percent > 0 and merged.max() > 0:
                # Uniform grow of the mask. Grayscale dilation with the same
                # cross structuring element / iteration count the old
                # binary_dilation used, so the grown SHAPE is unchanged and it's
                # byte-identical on a binary mask - but partial (per-section
                # opacity) gray values are now preserved through the grow.
                from scipy.ndimage import grey_dilation
                pad_px = int(round(min(width, height) * padding_percent / 100.0))
                if pad_px > 0:
                    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
                    for _ in range(pad_px):
                        merged = grey_dilation(merged, footprint=cross, mode="constant", cval=0)

            mask_float = np.clip(merged, 0, 255).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)

            info = json.dumps({"active_sections": used, "padding_percent": padding_percent})
            return (mask_tensor, info, preview_tensor)

        except Exception as e:
            logger.error(f"Portrait sections failed: {str(e)}")
            raise RuntimeError(f"Portrait sections failed: {str(e)}") from e
