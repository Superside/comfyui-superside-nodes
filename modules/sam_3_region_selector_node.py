import base64
import io
import json
import logging
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image

from .base_node import (
    APIClientMixin,
    SupersideFalNode,
    ImageProcessingMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideSAM3RegionSelectorNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Select a single edit region from an image using SAM 3 on fal.ai.

    This mirrors the Florence region selector contract so existing crop/stitch
    chains can swap between Florence and SAM.
    """

    REGION_TYPE_OPTIONS = [
        "face",
        "upper_body",
        "lower_body",
        "full_body",
        "hair",
        "glasses",
        "hat",
        "shirt",
        "top",
        "bra",
        "pants",
        "skirt",
        "dress",
        "shoes",
        "bag",
        "car",
        "vehicle",
        "wheel",
        "object",
    ]
    SELECTION_MODE_OPTIONS = ["largest", "first", "merge_all"]
    REGION_QUERY_MAP = {
        "face": "face",
        "upper_body": "upper body",
        "lower_body": "lower body",
        "full_body": "full body person",
        "hair": "hair",
        "glasses": "glasses",
        "hat": "hat",
        "shirt": "shirt",
        "top": "top garment",
        "bra": "bra",
        "pants": "pants",
        "skirt": "skirt",
        "dress": "dress",
        "shoes": "shoes",
        "bag": "bag",
        "car": "car",
        "vehicle": "vehicle",
        "wheel": "wheel",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region_type": (cls.REGION_TYPE_OPTIONS, {"default": "object"}),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "custom_text": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Required when region_type is object",
                    },
                ),
                "selection_mode": (cls.SELECTION_MODE_OPTIONS, {"default": "largest"}),
                "padding_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "return_rect_mask": ("BOOLEAN", {"default": False}),
                "return_multiple_masks": ("BOOLEAN", {"default": True}),
                "max_masks": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                "include_scores": ("BOOLEAN", {"default": True}),
                "include_boxes": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "mask",
        "mask_image",
        "info",
        "center_x",
        "center_y",
        "crop_width",
        "crop_height",
    )
    FUNCTION = "select_region"
    DISPLAY_NAME = "SAM 3 Smart Region Selector"
    DESCRIPTION = (
        "Select one semantic region at a time using SAM 3. Supports body, "
        "garment, vehicle, or custom object prompts."
    )

    def _normalize_image_array(self, image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.asarray(image)
        if image_np.ndim == 4 and image_np.shape[0] == 1:
            image_np = image_np[0]
        elif image_np.ndim == 3 and image_np.shape[0] in (3, 4):
            image_np = np.transpose(image_np, (1, 2, 0))
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        return image_np

    def _build_query(self, region_type: str, custom_text: str) -> str:
        region_type = (region_type or "object").strip()
        custom_text = (custom_text or "").strip()
        if region_type == "object":
            if not custom_text:
                raise ValueError("custom_text is required when region_type is object.")
            return custom_text
        if custom_text:
            raise ValueError("custom_text can only be used when region_type is object.")
        return self.REGION_QUERY_MAP[region_type]

    def _download_mask_image(self, image_ref):
        url = ""
        if isinstance(image_ref, dict):
            url = str(image_ref.get("url") or "").strip()
        elif isinstance(image_ref, str):
            url = image_ref.strip()
        if not url:
            raise RuntimeError("SAM 3 returned a mask without a URL.")
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise RuntimeError("Unsupported SAM 3 data URI mask payload.")
            content = base64.b64decode(payload)
        else:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            content = response.content
        pil_image = Image.open(io.BytesIO(content))
        pil_image.load()
        if pil_image.mode == "RGBA":
            alpha = np.asarray(pil_image.getchannel("A"), dtype=np.uint8)
            rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
            return np.maximum(np.max(rgb, axis=-1), alpha)
        return np.asarray(pil_image.convert("L"), dtype=np.uint8)

    def _mask_bbox(self, mask_uint8: np.ndarray):
        ys, xs = np.nonzero(mask_uint8 > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("SAM 3 did not return a usable region.")
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _apply_padding(self, bbox, width, height, padding_percent):
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(round(box_w * float(padding_percent) / 100.0))
        pad_y = int(round(box_h * float(padding_percent) / 100.0))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def _select_mask(self, masks_uint8, selection_mode):
        if not masks_uint8:
            raise RuntimeError("SAM 3 returned no usable masks.")
        if selection_mode == "merge_all":
            merged = np.zeros_like(masks_uint8[0], dtype=np.uint8)
            for mask in masks_uint8:
                merged = np.maximum(merged, mask)
            return merged, "merge_all", len(masks_uint8)
        if selection_mode == "first":
            return masks_uint8[0], "first", len(masks_uint8)
        return max(masks_uint8, key=lambda m: int(np.count_nonzero(m > 0))), "largest", len(masks_uint8)

    def _rect_mask_from_bbox(self, width, height, bbox):
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
        return mask

    def _mask_to_outputs(self, mask_uint8: np.ndarray):
        mask_float = mask_uint8.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)
        mask_rgb = np.stack([mask_float] * 3, axis=-1)
        mask_image_tensor = torch.from_numpy(mask_rgb).unsqueeze(0)
        return mask_tensor, mask_image_tensor

    def select_region(
        self,
        image,
        region_type,
        api_key,
        custom_text="",
        selection_mode="largest",
        padding_percent=0.0,
        return_rect_mask=False,
        return_multiple_masks=True,
        max_masks=3,
        include_scores=True,
        include_boxes=True,
    ):
        try:
            client = self.get_client(api_key)
            if not isinstance(image, torch.Tensor):
                raise ValueError("image input must be a ComfyUI IMAGE tensor.")
            if image.ndim != 4:
                raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(image.shape)}.")
            if image.shape[0] != 1:
                raise ValueError("SAM 3 Smart Region Selector currently supports batch size 1 only.")

            query = self._build_query(region_type, custom_text)
            image_url = self.upload_image(client, image, max_dimension=2048)
            image_np = self._normalize_image_array(image[0:1])
            height, width = image_np.shape[:2]
            result = self.call_api(
                client,
                "fal-ai/sam-3/image",
                {
                    "image_url": image_url,
                    "prompt": query,
                    "point_prompts": [],
                    "box_prompts": [],
                    "apply_mask": False,
                    "output_format": "png",
                    "return_multiple_masks": bool(return_multiple_masks),
                    "max_masks": int(max_masks),
                    "include_scores": bool(include_scores),
                    "include_boxes": bool(include_boxes),
                },
            )
            mask_entries = result.get("masks") if isinstance(result, dict) else None
            if not mask_entries:
                primary = result.get("image") if isinstance(result, dict) else None
                mask_entries = [primary] if primary else []

            masks_uint8 = []
            for entry in mask_entries:
                mask_uint8 = self._download_mask_image(entry)
                if mask_uint8.shape[:2] != (height, width):
                    pil_mask = Image.fromarray(mask_uint8, mode="L")
                    pil_mask = pil_mask.resize((width, height), Image.BILINEAR)
                    mask_uint8 = np.asarray(pil_mask, dtype=np.uint8)
                masks_uint8.append(mask_uint8)

            selected_mask, selected_source, mask_count = self._select_mask(masks_uint8, selection_mode)
            selected_mask = np.where(selected_mask > 0, 255, 0).astype(np.uint8)
            bbox = self._mask_bbox(selected_mask)
            padded_bbox = self._apply_padding(bbox, width, height, padding_percent)
            output_mask = self._rect_mask_from_bbox(width, height, padded_bbox) if return_rect_mask else selected_mask

            center_x = int(round((padded_bbox[0] + padded_bbox[2]) / 2.0))
            center_y = int(round((padded_bbox[1] + padded_bbox[3]) / 2.0))
            crop_width = int(padded_bbox[2] - padded_bbox[0])
            crop_height = int(padded_bbox[3] - padded_bbox[1])
            mask_tensor, mask_image_tensor = self._mask_to_outputs(output_mask)
            info = {
                "region_type": region_type,
                "query": query,
                "source": "fal-ai/sam-3/image",
                "selection_mode": selection_mode,
                "selected_mask_source": selected_source,
                "returned_mask_count": int(mask_count),
                "padding_percent": float(padding_percent),
                "bbox": {
                    "x1": int(padded_bbox[0]),
                    "y1": int(padded_bbox[1]),
                    "x2": int(padded_bbox[2]),
                    "y2": int(padded_bbox[3]),
                },
                "center_x": center_x,
                "center_y": center_y,
                "crop_width": crop_width,
                "crop_height": crop_height,
            }
            return (
                mask_tensor,
                mask_image_tensor,
                json.dumps(info),
                center_x,
                center_y,
                crop_width,
                crop_height,
            )
        except Exception as e:
            logger.error(f"SAM 3 region selection failed: {str(e)}")
            raise RuntimeError(f"SAM 3 region selection failed: {str(e)}") from e
