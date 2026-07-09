import io
import logging

import numpy as np
import torch
from PIL import Image, ImageFilter

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideBriaBackgroundStandardizerNode(
    SupersideFalNode, ImageProcessingMixin, APIClientMixin
):
    """
    Bria Background Standardizer Node: Cut out the subject with
    fal-ai/bria/background/remove, then composite it locally onto a solid
    hex-color canvas.

    Unlike prompt-driven background replacement, this node never regenerates
    the subject. The cutout is pixel-identical to the source, and the
    background is an exact hex color instead of a diffusion model's
    interpretation of a color description - this makes it suitable for
    batch-homogenizing avatar backgrounds without any quality drift.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hex_color": (
                    "STRING",
                    {
                        "default": "#FFFFFF",
                        "placeholder": "#RRGGBB (e.g. #F5F5F5)",
                    },
                ),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "edge_feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 15,
                        "tooltip": "Softens the cutout edge by this many pixels to avoid a jagged silhouette. 0 = pixel-perfect edge, no softening.",
                    },
                ),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Standardize an avatar's background to an exact hex color. Removes the "
        "background with Bria RMBG 2.0 (fal-ai/bria/background/remove) and "
        "composites the untouched subject onto a solid color locally - the "
        "subject is never regenerated, only the background pixels change."
    )

    @staticmethod
    def _parse_hex_color(hex_color):
        value = hex_color.strip().lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        if len(value) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
            raise ValueError(
                f"Invalid hex_color '{hex_color}'. Use a format like '#FFFFFF' or '#FFF'."
            )
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))

    def generate(self, image, hex_color, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            rgb = self._parse_hex_color(hex_color)

            image_url = self.upload_image(client, image)
            arguments = {
                "image_url": image_url,
                "sync_mode": kwargs.get("sync_mode", False),
            }
            result = self.call_api(client, "fal-ai/bria/background/remove", arguments)

            cutout_data = result.get("image")
            if not isinstance(cutout_data, dict) or "url" not in cutout_data:
                raise RuntimeError("No cutout image was returned by the background removal API.")

            cutout_url = cutout_data["url"]
            cutout_bytes = self._load_image_bytes_from_url(cutout_url)
            cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")

            edge_feather = kwargs.get("edge_feather", 0)
            if edge_feather:
                alpha = cutout.split()[-1].filter(ImageFilter.GaussianBlur(edge_feather))
                cutout.putalpha(alpha)

            background = Image.new("RGBA", cutout.size, rgb + (255,))
            composed = Image.alpha_composite(background, cutout).convert("RGB")

            img_np = np.array(composed).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)

            info = f"background=#{''.join(f'{c:02X}' for c in rgb)}; source_cutout={cutout_url}"

            if unique_id is not None and extra_pnginfo is not None:
                if (
                    isinstance(extra_pnginfo, list)
                    and isinstance(extra_pnginfo[0], dict)
                    and "workflow" in extra_pnginfo[0]
                ):
                    workflow = extra_pnginfo[0]["workflow"]
                    node = next(
                        (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id)),
                        None,
                    )
                    if node:
                        node["widgets_values"] = [info]

            return {"ui": {"text": [info]}, "result": (img_tensor, info)}
        except Exception as e:
            logger.error(f"Bria background standardizer failed: {str(e)}")
            raise RuntimeError(f"Bria background standardizer failed: {str(e)}") from e
