import io
import logging

import numpy as np
import torch
from PIL import Image

from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SupersideGPTImage2EditNode(SupersideFalNode, ImageProcessingMixin, APIClientMixin):
    """
    GPT Image 2 Edit Node: Edit images using openai/gpt-image-2/edit.

    Supports one or more reference images plus an optional mask image for more
    precise inpainting-style edits.
    """

    # One "size" control drives everything. Pick the SHAPE here:
    #   - "match input + resolution": keep the input image's aspect ratio, but
    #     scale it up to the chosen resolution (portrait stays portrait, sized 4K)
    #   - "match input (original)": let the model infer/keep the input's own size
    #   - an aspect ratio (e.g. "16:9"): the resolution control below sets how big
    #   - "custom pixels": use the width/height fields
    SIZE_OPTIONS = [
        "match input + resolution",
        "match input (original)",
        "1:1",
        "4:5",
        "5:4",
        "4:3",
        "3:4",
        "16:9",
        "9:16",
        "3:2",
        "2:3",
        "custom pixels",
    ]

    ASPECT_RATIOS = {"1:1", "4:5", "5:4", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"}

    RESOLUTION_OPTIONS = ["1K", "2K", "4K"]

    # How mask_image is used. Off = ignore it (crop-stitch pipelines); soft =
    # send it to the model as guidance; hard = also composite the result back
    # only inside the mask so the outside stays pixel-identical.
    MASK_OFF = "off - edit whole image"
    MASK_SOFT = "guide model (soft)"
    MASK_HARD = "lock outside mask (hard)"
    MASK_MODE_OPTIONS = [MASK_OFF, MASK_SOFT, MASK_HARD]

    # Long-edge target per resolution. GPT Image 2 caps total output to about
    # 8 megapixels, so at "4K" the long edge lands at ~3840 px for 16:9
    # (true UHD), ~3520 for 3:2, ~2880 for 1:1 - fal clamps to the budget while
    # preserving the aspect ratio.
    LONG_EDGE_MAP = {"1K": 1024, "2K": 2048, "4K": 4096}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Describe the edit you want to apply",
                    },
                ),
                "image_1": ("IMAGE",),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "size": (
                    cls.SIZE_OPTIONS,
                    {
                        "default": "match input + resolution",
                        "tooltip": "Output shape. 'match input + resolution' keeps your image's aspect (portrait stays portrait) and scales it to 'resolution' below - just pick 4K for the biggest. 'match input (original)' keeps the input's own size. Or pick a fixed aspect ratio / 'custom pixels'.",
                    },
                ),
                "resolution": (
                    cls.RESOLUTION_OPTIONS,
                    {
                        "default": "2K",
                        "tooltip": "How large the output is when 'size' is an aspect ratio. GPT Image 2 caps total size to ~8 MP, so 4K gives ~3840 px on the long edge at 16:9 (true UHD), less for squarer ratios (~2880 at 1:1).",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 16,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Only used when size is 'custom pixels'. Must be a multiple of 16.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 16,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Only used when size is 'custom pixels'. Must be a multiple of 16.",
                    },
                ),
                "mask_mode": (
                    cls.MASK_MODE_OPTIONS,
                    {
                        "default": cls.MASK_OFF,
                        "tooltip": (
                            "How to use mask_image:\n"
                            "- 'off - edit whole image': ignore the mask, edit everything. Use this in crop-stitch pipelines where a separate stitch node does the masking (this is how it worked before).\n"
                            "- 'guide model (soft)': send the mask to GPT so it focuses edits on the white area (the model may still re-render the rest).\n"
                            "- 'lock outside mask (hard)': same, plus paste the result back only inside the mask so everything outside stays pixel-identical to the input. Best for standalone inpainting."
                        ),
                    },
                ),
                "invert_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only used when mask_mode is not 'off'. Mask convention is WHITE = edit this area, BLACK = keep. Turn ON if your mask is inverted (the area you want to change is black).",
                    },
                ),
                "quality": (["auto", "low", "medium", "high"], {"default": "high"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Edit images using OpenAI GPT Image 2 on fal. "
        "Supports multi-image references, mask-based editing, preset sizes, "
        "aspect-ratio + resolution sizing, and custom output sizes."
    )

    def _round_to_multiple_of_16(self, value):
        """Round a numeric dimension to the nearest lower multiple of 16."""
        return max(16, int(value) // 16 * 16)

    def _calculate_image_size_from_aspect_ratio(self, aspect_ratio, resolution):
        """Convert aspect ratio + resolution preset to explicit dimensions."""
        long_edge = self.LONG_EDGE_MAP.get(resolution, 2048)

        width_ratio, height_ratio = aspect_ratio.split(":")
        width_ratio = int(width_ratio)
        height_ratio = int(height_ratio)

        if width_ratio >= height_ratio:
            width = long_edge
            height = long_edge * height_ratio / width_ratio
        else:
            height = long_edge
            width = long_edge * width_ratio / height_ratio

        width = self._round_to_multiple_of_16(width)
        height = self._round_to_multiple_of_16(height)

        return {
            "width": width,
            "height": height,
        }

    def _size_from_input_aspect(self, image, resolution):
        """
        Build {width,height} that keeps the input image's aspect ratio, scaled so
        the long edge targets the chosen resolution. GPT Image 2 clamps to its
        ~8 MP budget while preserving the aspect, so a tall portrait stays tall.
        Falls back to "auto" if the input dimensions can't be read.
        """
        try:
            h = int(image.shape[-3])
            w = int(image.shape[-2])
        except Exception:
            return "auto"
        if w <= 0 or h <= 0:
            return "auto"

        long_edge = self.LONG_EDGE_MAP.get(resolution, 2048)
        if w >= h:
            width = long_edge
            height = long_edge * h / w
        else:
            height = long_edge
            width = long_edge * w / h
        return {
            "width": self._round_to_multiple_of_16(width),
            "height": self._round_to_multiple_of_16(height),
        }

    def _resolve_image_size(self, **kwargs):
        """
        Turn the single `size` control into the API's image_size value:
          - "match input + resolution"     -> {width,height} from input aspect + resolution
          - "match input (original)"/None   -> "auto" (model keeps the input's size)
          - an aspect ratio ("16:9", ...)  -> {width,height} from ratio + resolution
          - "custom pixels"                -> {width,height} from the width/height fields
        A legacy bridge keeps old workflows saved with `size_mode` working.
        """
        size = kwargs.get("size")

        # Backwards compatibility with the previous multi-control layout.
        if size is None and kwargs.get("size_mode") is not None:
            legacy_mode = kwargs.get("size_mode")
            if legacy_mode == "aspect_ratio":
                size = kwargs.get("aspect_ratio", "16:9")
            elif legacy_mode == "custom":
                size = "custom pixels"
            else:  # "preset"
                preset = kwargs.get("image_size", "auto")
                return preset if preset else "auto"

        if size == "match input + resolution":
            return self._size_from_input_aspect(
                kwargs.get("image_1"), kwargs.get("resolution", "2K")
            )

        # "match input (original)", the old "match input (auto)", or None -> auto
        if size is None or size in ("match input (original)", "match input (auto)"):
            return "auto"

        if size == "custom pixels":
            width = kwargs.get("width", 1920)
            height = kwargs.get("height", 1080)
            if width % 16 != 0 or height % 16 != 0:
                raise ValueError("Custom width and height must both be multiples of 16")
            return {"width": width, "height": height}

        if size in self.ASPECT_RATIOS:
            return self._calculate_image_size_from_aspect_ratio(
                size, kwargs.get("resolution", "2K")
            )

        # Any other string (e.g. a legacy preset like "square_hd") passes through.
        return size

    @staticmethod
    def _tensor_to_pil(tensor):
        """ComfyUI IMAGE/MASK tensor -> RGB PIL. Accepts [B,H,W,C], [H,W,C],
        [B,H,W] or [H,W] (a single-channel MASK)."""
        arr = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:  # single-channel mask
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr[..., :3], "RGB")

    def _resolve_mask_mode(self, **kwargs):
        """
        Return (send_mask, do_composite) from mask_mode, bridging the older
        send_mask_to_model / keep_unmasked_area booleans if mask_mode is absent.
        """
        mode = kwargs.get("mask_mode")
        if mode is None:  # legacy workflow saved before mask_mode existed
            if kwargs.get("send_mask_to_model", False):
                hard = kwargs.get("keep_unmasked_area", True)
                return True, bool(hard)
            return False, False

        if mode == self.MASK_SOFT:
            return True, False
        if mode == self.MASK_HARD:
            return True, True
        return False, False  # MASK_OFF or anything unexpected

    def _mask_to_grayscale(self, mask_tensor, invert):
        """Build the grayscale mask GPT Image 2 expects: WHITE = edit, BLACK =
        keep. Optionally invert if the user's mask uses the opposite convention."""
        gray = self._tensor_to_pil(mask_tensor).convert("L")
        if invert:
            gray = Image.eval(gray, lambda p: 255 - p)
        return gray

    @staticmethod
    def _upload_pil_png(client, pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return client.upload(buffered.getvalue(), "image/png")

    def _composite_unmasked(self, out_tensor, orig_tensor, mask_gray):
        """Keep everything outside the mask identical to the input: blend the
        model output with the original using the mask (white=use output)."""
        out = out_tensor[0].cpu().numpy().astype(np.float32)  # H,W,C in 0..1
        h, w = out.shape[:2]
        orig_pil = self._tensor_to_pil(orig_tensor).resize((w, h), Image.LANCZOS)
        orig = np.asarray(orig_pil).astype(np.float32) / 255.0
        m = np.asarray(mask_gray.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
        m = m[..., None]
        blended = out[..., :3] * m + orig * (1.0 - m)
        return torch.from_numpy(blended.astype(np.float32)).unsqueeze(0)

    def prepare_image_urls(self, client, **kwargs):
        """Prepare list of image URLs from input images."""
        image_urls = []

        for i in range(1, 7):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                try:
                    url = self.upload_image(client, kwargs[image_key])
                    image_urls.append(url)
                    logger.info(f"Uploaded {image_key}: {url}")
                except Exception as e:
                    logger.warning(f"Failed to upload {image_key}: {str(e)}")
                    continue

        return image_urls

    def prepare_arguments(self, client, prompt, **kwargs):
        """Prepare arguments for the API call."""
        image_urls = self.prepare_image_urls(client, **kwargs)
        if not image_urls:
            raise ValueError("At least one image is required for GPT Image 2 editing")

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
        }

        # Only send a mask when mask_mode is not 'off'. Crop-stitch pipelines
        # feed a mask for their OWN nodes and expect GPT to edit the whole crop,
        # so the node ignores mask_image by default. The fal endpoint field is
        # `mask_url`, and GPT Image 2 reads it as grayscale (WHITE = edit).
        if kwargs.get("_send_mask"):
            mask_gray = kwargs.get("_mask_gray")
            if mask_gray is not None:
                arguments["mask_url"] = self._upload_pil_png(client, mask_gray)
            elif kwargs.get("mask_image") is not None:
                arguments["mask_url"] = self.upload_image(client, kwargs["mask_image"])

        image_size = self._resolve_image_size(**kwargs)
        if image_size is not None:
            arguments["image_size"] = image_size

        if kwargs.get("quality") is not None:
            arguments["quality"] = kwargs["quality"]

        if kwargs.get("num_images") is not None:
            arguments["num_images"] = kwargs["num_images"]

        if kwargs.get("output_format") is not None:
            arguments["output_format"] = kwargs["output_format"]

        if kwargs.get("sync_mode") is not None:
            arguments["sync_mode"] = kwargs["sync_mode"]

        return arguments

    def generate(self, prompt, api_key, **kwargs):
        """Main image editing function."""
        try:
            client = self.get_client(api_key)

            # Resolve how the mask should be used (with a bridge for the older
            # send_mask_to_model / keep_unmasked_area layout).
            send_mask, do_composite = self._resolve_mask_mode(**kwargs)
            kwargs["_send_mask"] = send_mask

            # Build the grayscale mask once (honoring invert) so it can drive
            # both the API call and the local composite - only when in use.
            # 'off' by default so crop-stitch pipelines are unaffected.
            mask_gray = None
            if send_mask and kwargs.get("mask_image") is not None:
                mask_gray = self._mask_to_grayscale(
                    kwargs["mask_image"], kwargs.get("invert_mask", False)
                )
                kwargs["_mask_gray"] = mask_gray

            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, "openai/gpt-image-2/edit", arguments)

            images = self.process_images(result)
            output = images[0]

            # In 'hard' mode, GPT re-renders the whole frame so pixels outside
            # the mask drift too; composite the result back only inside the mask
            # to keep the rest identical to the input.
            if mask_gray is not None and do_composite:
                orig = kwargs.get("image_1")
                if orig is not None:
                    output = self._composite_unmasked(output, orig, mask_gray)

            info = ""
            if result.get("images") and isinstance(result["images"], list):
                first_url = result["images"][0].get("url", "")
                if isinstance(first_url, str) and first_url.startswith("data:"):
                    info = "data-uri image returned"
                else:
                    info = first_url

            return (output, info)
        except Exception as e:
            logger.error(f"GPT Image 2 edit failed: {str(e)}")
            raise RuntimeError(f"GPT Image 2 edit failed: {str(e)}") from e
