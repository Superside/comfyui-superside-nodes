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


class SupersideSceneExclusionMaskNode(SupersideFalNode, ImageProcessingMixin, APIClientMixin):
    """
    Superside Scene Exclusion Mask: build a generic EXCLUSION mask for a scene
    enhancement pass - the parts that must be PROTECTED from the enhancer and
    pasted back from the original.

    This is the generic, non-face counterpart of Superside Portrait Sections.
    Instead of fixed facial toggles, you give it a plain list of things to
    protect (one per line, e.g. "dimmer switch", "wall outlet", "brand logo")
    plus an "exclude people" toggle. Each is segmented with SAM 3 and merged
    (OR) into one mask, using the same re-skin principle: enhance the region you
    want, but composite the ORIGINAL back over everything this mask covers.

    Wire the output into a grow/blur node and then Superside Image Composite
    Masked (source = original, destination = enhanced), exactly like the re-skin
    graph - just for arbitrary scene elements instead of eyes/glasses/skin.

    Opacity < 1.0 protects a target only partially (partial blend in the
    composite); feather/contract soften the protected edge without a hard seam.
    """

    CATEGORY = "Superside"
    SELECTION_MODE_OPTIONS = ["merge_all", "largest", "first"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "exclude_people": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Protect people from the enhancement (adds 'person' to the exclusion set). Turn OFF to let the enhancer touch people too.",
                }),
                "exclude_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "One thing to protect per line, e.g.\ndimmer switch\nwall outlet plate\nbrand logo",
                    "tooltip": "Plain-language targets to EXCLUDE (protect), one per line (commas also work). Each is segmented with SAM 3 and merged into the exclusion mask.",
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly the targets are protected. 1.0 = fully preserve the original there. 0.5 = partial (the composite blends 50% original / 50% enhanced).",
                }),
                "feather_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 25.0, "step": 0.5,
                    "tooltip": "Soften the protected edge so the composite blends gradually instead of a hard line.",
                }),
                "contract_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 25.0, "step": 0.5,
                    "tooltip": "Erode the mask inward before feathering, so the soft edge stays inside the target instead of bleeding onto surroundings. Set near/above feather.",
                }),
                "padding_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Grow the whole merged exclusion mask outward by this much (after per-target opacity/feather).",
                }),
                "selection_mode": (cls.SELECTION_MODE_OPTIONS, {
                    "default": "merge_all",
                    "tooltip": "Per target: merge_all catches every instance (e.g. two switches); largest keeps only the biggest; first keeps the top-scored.",
                }),
                "max_masks": ("INT", {"default": 6, "min": 1, "max": 32, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "info", "color_preview")
    FUNCTION = "build_mask"
    DESCRIPTION = (
        "Build a generic EXCLUSION mask for a scene enhancement pass: protect "
        "people (toggle) and any listed objects (one prompt per line) via SAM 3, "
        "merged into one mask. Pair with grow/blur + Image Composite Masked to "
        "enhance a scene while keeping the protected elements pixel-original."
    )

    # ---- SAM 3 helpers (self-contained; same convention as Portrait Sections) ----
    def _download_mask_array(self, image_ref, width, height):
        url = image_ref.get("url") if isinstance(image_ref, dict) else image_ref
        if not url:
            raise RuntimeError("SAM 3 returned a target without a mask URL.")
        if str(url).startswith("data:"):
            _, _, payload = url.partition(",")
            content = base64.b64decode(payload)
        else:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            content = resp.content
        pil = Image.open(io.BytesIO(content))
        pil.load()
        arr = np.asarray(pil.getchannel("A"), dtype=np.uint8) if pil.mode == "RGBA" else np.asarray(pil.convert("L"), dtype=np.uint8)
        if arr.shape[:2] != (height, width):
            arr = np.asarray(Image.fromarray(arr, "L").resize((width, height), Image.BILINEAR), dtype=np.uint8)
        return arr

    def _sam3_mask(self, client, image_url, prompt, width, height, selection_mode, max_masks):
        result = self.call_api(client, "fal-ai/sam-3/image", {
            "image_url": image_url,
            "prompt": prompt,
            "point_prompts": [],
            "box_prompts": [],
            "apply_mask": False,
            "output_format": "png",
            "return_multiple_masks": True,
            "max_masks": int(max_masks),
            "include_scores": False,
            "include_boxes": False,
        })
        entries = result.get("masks") if isinstance(result, dict) else None
        if not entries:
            primary = result.get("image") if isinstance(result, dict) else None
            entries = [primary] if primary else []
        masks = []
        for e in entries:
            try:
                masks.append(self._download_mask_array(e, width, height))
            except Exception as ex:
                logger.warning(f"SAM 3 mask download failed for '{prompt}': {ex}")
        if not masks:
            return np.zeros((height, width), dtype=np.uint8)
        if selection_mode == "first":
            return masks[0]
        if selection_mode == "largest":
            return max(masks, key=lambda m: int(np.count_nonzero(m > 0)))
        merged = np.zeros((height, width), dtype=np.uint8)
        for m in masks:
            merged = np.maximum(merged, m)
        return merged

    @staticmethod
    def _scale_opacity(mask_uint8, opacity, feather_px=0.0, contract_px=0.0):
        if opacity >= 1.0 and feather_px <= 0 and contract_px <= 0:
            return mask_uint8
        work = mask_uint8.astype(np.float32)
        if contract_px and contract_px > 0:
            from scipy.ndimage import grey_erosion
            k = int(round(contract_px)) * 2 + 1
            work = grey_erosion(work, size=(k, k), mode="constant", cval=0.0)
        work = work * max(0.0, float(opacity))
        if feather_px and feather_px > 0:
            from scipy.ndimage import gaussian_filter
            work = gaussian_filter(work, sigma=float(feather_px))
        return np.clip(work, 0, 255).astype(np.uint8)

    @staticmethod
    def _parse_targets(exclude_people, exclude_prompts):
        targets = []
        if exclude_people:
            targets.append("person")
        for line in (exclude_prompts or "").splitlines():
            for part in line.split(","):
                p = part.strip()
                if p:
                    targets.append(p)
        # dedupe, keep order
        seen, out = set(), []
        for t in targets:
            if t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
        return out

    def build_mask(self, image, api_key, exclude_people=True, exclude_prompts="",
                   opacity=1.0, feather_percent=0.0, contract_percent=0.0,
                   padding_percent=0.0, selection_mode="merge_all", max_masks=6):
        try:
            client = self.get_client(api_key)
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError("Scene Exclusion Mask supports batch size 1 only.")
            height, width = int(image.shape[1]), int(image.shape[2])
            targets = self._parse_targets(exclude_people, exclude_prompts)
            if not targets:
                empty = torch.zeros((1, height, width), dtype=torch.float32)
                black = torch.from_numpy(np.asarray(image[0]).astype(np.float32)) if False else image
                return (empty, json.dumps({"targets": [], "note": "nothing to exclude"}), image)

            image_url = self.upload_image(client, image, max_dimension=2048)
            feather_px = min(width, height) * float(feather_percent) / 100.0
            contract_px = min(width, height) * float(contract_percent) / 100.0

            merged = np.zeros((height, width), dtype=np.uint8)
            used, sections = [], []
            for t in targets:
                try:
                    m = self._sam3_mask(client, image_url, t, width, height, selection_mode, max_masks)
                except Exception as e:
                    logger.warning(f"Skipping target '{t}' (SAM 3 failed): {e}")
                    continue
                if int(np.count_nonzero(m > 0)) == 0:
                    logger.info(f"Target '{t}' produced an empty mask.")
                    continue
                merged = np.maximum(merged, self._scale_opacity(m, opacity, feather_px, contract_px))
                used.append(t); sections.append(m)

            if padding_percent > 0 and merged.max() > 0:
                from scipy.ndimage import grey_dilation
                pad_px = int(round(min(width, height) * padding_percent / 100.0))
                cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
                for _ in range(pad_px):
                    merged = grey_dilation(merged, footprint=cross, mode="constant", cval=0)

            # color preview: red overlay where protected
            base = image[0].detach().cpu().numpy()
            base = (base * 255.0) if base.max() <= 1.0 else base
            prev = np.clip(base, 0, 255).astype(np.float32)[:, :, :3].copy()
            mnorm = (merged.astype(np.float32) / 255.0)[:, :, None] * 0.5
            prev = prev * (1.0 - mnorm) + np.array([220.0, 60.0, 50.0])[None, None, :] * mnorm
            preview = torch.from_numpy(np.clip(prev, 0, 255).astype(np.float32) / 255.0).unsqueeze(0)

            mask_tensor = torch.from_numpy((merged.astype(np.float32) / 255.0)).unsqueeze(0)
            info = json.dumps({"targets_used": used, "opacity": opacity,
                               "feather_percent": feather_percent, "contract_percent": contract_percent,
                               "padding_percent": padding_percent, "selection_mode": selection_mode})
            return (mask_tensor, info, preview)
        except Exception as e:
            logger.error(f"Scene exclusion mask failed: {str(e)}")
            raise RuntimeError(f"Scene exclusion mask failed: {str(e)}") from e
