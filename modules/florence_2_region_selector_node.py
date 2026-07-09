import json
import logging
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

from .base_node import (
    APIClientMixin,
    SupersideFalNode,
    ImageProcessingMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Florence2RegionSelectorNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Select a single edit region from an image using Florence-2 on fal.ai.

    This node is designed around a strict single-region UX:
    - one region type at a time
    - free text only when region_type == object
    - segmentation first, grounding fallback second
    """

    REGION_TYPE_OPTIONS = [
        "face",
        "upper_body",
        "lower_body",
        "full_body",
        "object",
    ]

    SELECTION_MODE_OPTIONS = ["largest", "merge_all"]

    REGION_QUERY_MAP = {
        "face": "face",
        "upper_body": "upper body",
        "lower_body": "lower body",
        "full_body": "full body person",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region_type": (cls.REGION_TYPE_OPTIONS, {"default": "face"}),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "custom_text": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Only used when region_type is object",
                    },
                ),
                "selection_mode": (
                    cls.SELECTION_MODE_OPTIONS,
                    {"default": "largest"},
                ),
                "padding_percent": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                    },
                ),
                "return_rect_mask": ("BOOLEAN", {"default": False}),
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
    DISPLAY_NAME = "Florence-2 Smart Region Selector"
    DESCRIPTION = (
        "Select one semantic region at a time using Florence-2. "
        "Supports face, upper body, lower body, full body, or a custom object prompt."
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

        return image_np

    def _build_query(self, region_type: str, custom_text: str) -> str:
        custom_text = (custom_text or "").strip()
        if region_type == "object":
            if not custom_text:
                raise ValueError(
                    "custom_text is required when region_type is set to object."
                )
            return custom_text

        if custom_text:
            raise ValueError(
                "custom_text can only be used when region_type is object."
            )

        return self.REGION_QUERY_MAP[region_type]

    def _is_point_pair(self, value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and all(isinstance(v, (int, float)) for v in value[:2])
        )

    def _extract_points_recursive(self, value: Any) -> list[tuple[float, float]] | None:
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                x = value["x"]
                y = value["y"]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    return [(float(x), float(y))]
            for nested in value.values():
                points = self._extract_points_recursive(nested)
                if points and len(points) >= 3:
                    return points
            return None

        if isinstance(value, (list, tuple)):
            if len(value) >= 3 and all(self._is_point_pair(item) for item in value):
                return [(float(item[0]), float(item[1])) for item in value]

            collected: list[tuple[float, float]] = []
            for item in value:
                points = self._extract_points_recursive(item)
                if points and len(points) >= 3:
                    return points
                if points and len(points) == 1:
                    collected.extend(points)
            if len(collected) >= 3:
                return collected

        return None

    def _extract_bbox(self, value: Any) -> tuple[float, float, float, float] | None:
        if isinstance(value, dict):
            if all(key in value for key in ("x1", "y1", "x2", "y2")):
                coords = (value["x1"], value["y1"], value["x2"], value["y2"])
                if all(isinstance(v, (int, float)) for v in coords):
                    return tuple(float(v) for v in coords)
            if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
                coords = (value["xmin"], value["ymin"], value["xmax"], value["ymax"])
                if all(isinstance(v, (int, float)) for v in coords):
                    return tuple(float(v) for v in coords)
            if "bbox" in value:
                bbox = self._extract_bbox(value["bbox"])
                if bbox:
                    return bbox
            for nested in value.values():
                bbox = self._extract_bbox(nested)
                if bbox:
                    return bbox
            return None

        if isinstance(value, (list, tuple)) and len(value) >= 4:
            if all(isinstance(v, (int, float)) for v in value[:4]):
                x1, y1, x2, y2 = [float(v) for v in value[:4]]
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)

        return None

    def _polygon_area(self, points: Iterable[tuple[float, float]]) -> float:
        pts = list(points)
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for index, (x1, y1) in enumerate(pts):
            x2, y2 = pts[(index + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _bbox_area(self, bbox: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _coerce_polygon_entries(self, result: dict[str, Any]) -> list[list[tuple[float, float]]]:
        polygons = result.get("results", {}).get("polygons", [])
        output: list[list[tuple[float, float]]] = []
        for entry in polygons:
            points = self._extract_points_recursive(entry)
            if points and len(points) >= 3:
                output.append(points)
        return output

    def _coerce_bbox_entries(self, result: dict[str, Any]) -> list[tuple[float, float, float, float]]:
        bboxes = result.get("results", {}).get("bboxes", [])
        output: list[tuple[float, float, float, float]] = []
        for entry in bboxes:
            bbox = self._extract_bbox(entry)
            if bbox:
                output.append(bbox)
        return output

    def _render_mask_from_polygons(
        self,
        width: int,
        height: int,
        polygons: list[list[tuple[float, float]]],
        selection_mode: str,
    ) -> np.ndarray:
        if selection_mode == "largest":
            polygons = [max(polygons, key=self._polygon_area)]

        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)
        for polygon in polygons:
            drawer.polygon(polygon, fill=255)
        return np.array(mask_image, dtype=np.uint8)

    def _render_mask_from_bboxes(
        self,
        width: int,
        height: int,
        bboxes: list[tuple[float, float, float, float]],
        selection_mode: str,
    ) -> np.ndarray:
        if selection_mode == "largest":
            bboxes = [max(bboxes, key=self._bbox_area)]

        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)
        for x1, y1, x2, y2 in bboxes:
            drawer.rectangle((x1, y1, x2, y2), fill=255)
        return np.array(mask_image, dtype=np.uint8)

    def _apply_padding(
        self,
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
        padding_percent: float,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(round(box_w * (padding_percent / 100.0)))
        pad_y = int(round(box_h * (padding_percent / 100.0)))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def _mask_bbox(self, mask_uint8: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(mask_uint8 > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("Florence did not return a usable region.")
        return (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )

    def _rect_mask_from_bbox(
        self, width: int, height: int, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
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

    def _call_segmentation(self, client, image_url: str, query: str) -> dict[str, Any]:
        return self.call_api(
            client,
            "fal-ai/florence-2-large/referring-expression-segmentation",
            {"image_url": image_url, "text_input": query},
        )

    def _call_grounding(self, client, image_url: str, query: str) -> dict[str, Any]:
        return self.call_api(
            client,
            "fal-ai/florence-2-large/caption-to-phrase-grounding",
            {"image_url": image_url, "text_input": query},
        )

    def select_region(
        self,
        image,
        region_type,
        api_key,
        custom_text="",
        selection_mode="largest",
        padding_percent=8.0,
        return_rect_mask=False,
    ):
        try:
            client = self.get_client(api_key)

            if not isinstance(image, torch.Tensor):
                raise ValueError("image input must be a ComfyUI IMAGE tensor.")

            if image.ndim != 4:
                raise ValueError(
                    f"Expected IMAGE tensor with shape [B,H,W,C], got {tuple(image.shape)}."
                )

            if image.shape[0] != 1:
                raise ValueError(
                    "Florence-2 Smart Region Selector currently supports batch size 1 only."
                )

            query = self._build_query(region_type, custom_text)
            image_url = self.upload_image(client, image, max_dimension=2048)
            image_np = self._normalize_image_array(image[0:1])
            height, width = image_np.shape[:2]

            logger.info(
                "Running Florence region selector with region_type=%s query=%s",
                region_type,
                query,
            )

            mask_uint8 = None
            source = None

            segmentation_result = self._call_segmentation(client, image_url, query)
            polygons = self._coerce_polygon_entries(segmentation_result)
            if polygons:
                mask_uint8 = self._render_mask_from_polygons(
                    width, height, polygons, selection_mode
                )
                source = "referring-expression-segmentation"

            if mask_uint8 is None:
                grounding_result = self._call_grounding(client, image_url, query)
                bboxes = self._coerce_bbox_entries(grounding_result)
                if not bboxes:
                    raise RuntimeError(
                        "Florence returned no polygons and no bounding boxes for this region."
                    )
                mask_uint8 = self._render_mask_from_bboxes(
                    width, height, bboxes, selection_mode
                )
                source = "caption-to-phrase-grounding"

            bbox = self._mask_bbox(mask_uint8)
            padded_bbox = self._apply_padding(bbox, width, height, padding_percent)

            if return_rect_mask:
                output_mask_uint8 = self._rect_mask_from_bbox(width, height, padded_bbox)
            else:
                output_mask_uint8 = mask_uint8.copy()

            center_x = int(round((padded_bbox[0] + padded_bbox[2]) / 2.0))
            center_y = int(round((padded_bbox[1] + padded_bbox[3]) / 2.0))
            crop_width = int(padded_bbox[2] - padded_bbox[0])
            crop_height = int(padded_bbox[3] - padded_bbox[1])

            mask_tensor, mask_image_tensor = self._mask_to_outputs(output_mask_uint8)
            info = {
                "region_type": region_type,
                "query": query,
                "source": source,
                "selection_mode": selection_mode,
                "padding_percent": padding_percent,
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
            logger.error(f"Florence region selection failed: {str(e)}")
            raise RuntimeError(f"Florence region selection failed: {str(e)}") from e
