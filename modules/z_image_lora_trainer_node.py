import io
import logging
import zipfile

import numpy as np
import torch
from PIL import Image

from .base_node import SupersideFalNode, APIClientMixin, API_KEY_INPUT_SPEC

logger = logging.getLogger(__name__)

# fal recommends at least this many training images; fewer than this usually
# undertrains the LoRA. This only produces a warning, not a hard failure -
# some quick style tests genuinely use fewer.
RECOMMENDED_MIN_IMAGES = 10


class SupersideZImageLoraTrainerNode(SupersideFalNode, APIClientMixin):
    """
    Z-Image LoRA Trainer Node: train a custom LoRA on Z-Image Turbo from a
    batch of images (fal endpoint "fal-ai/z-image-turbo-trainer-v2") - e.g.
    close-up skin/imperfection references for a realistic-skin LoRA.

    Pair the resulting `lora_file_url` with Superside Z-Image Inpaint+LoRA's
    `lora_url` input - both nodes target the same base model (Z-Image
    Turbo), so the trained LoRA is guaranteed to apply correctly there
    (unlike training on one model family and inpainting with another).

    Connect a batch of IMAGE tensors (e.g. several LoadImage nodes through an
    Image Batch node, or a folder loader). Every image gets the same
    `default_caption` - a per-image caption workflow isn't exposed here to
    keep this node simple; for per-image captions, zip the images with
    matching .txt files yourself and pass the zip's URL via `images_zip_url`
    instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Batch of training images (ignored when images_zip_url is set). At least 10 recommended."},
                ),
                "default_caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "e.g. \"sks skin texture, close-up photo, realistic pores and imperfections\"",
                        "tooltip": "Applied to every image (no per-image caption files here). Include your trigger word.",
                    },
                ),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "steps": (
                    "INT",
                    {"default": 2000, "min": 100, "max": 10000, "step": 100, "tooltip": "Number of training steps."},
                ),
                "learning_rate": (
                    "FLOAT",
                    {"default": 0.0005, "min": 0.00001, "max": 0.01, "step": 0.00001, "tooltip": "Training learning rate."},
                ),
                "images_zip_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "URL of a pre-built zip (images + matching .txt captions). Overrides the IMAGE batch input.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_file_url",)
    FUNCTION = "train_lora"
    CATEGORY = "Superside"
    DESCRIPTION = (
        "Train a LoRA on Z-Image Turbo from a batch of images (e.g. skin "
        "close-ups for a realistic-skin LoRA). Pair the output URL with "
        "Superside Z-Image Inpaint+LoRA's lora_url - same base model, "
        "guaranteed compatible."
    )

    def _tensor_to_png_bytes(self, image_tensor):
        image_np = image_tensor.detach().cpu().numpy() if isinstance(image_tensor, torch.Tensor) else np.asarray(image_tensor)
        if image_np.dtype != np.uint8:
            image_np = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()

    def _build_and_upload_zip(self, client, images):
        count = images.shape[0] if hasattr(images, "shape") else len(images)
        if count < RECOMMENDED_MIN_IMAGES:
            logger.warning(
                f"Only {count} training image(s) provided; fal recommends at "
                f"least {RECOMMENDED_MIN_IMAGES} for a reliable LoRA."
            )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(count):
                png_bytes = self._tensor_to_png_bytes(images[i])
                zf.writestr(f"{i + 1:04d}.png", png_bytes)
        zip_bytes = zip_buffer.getvalue()

        url = client.upload(zip_bytes, "application/zip")
        logger.info(f"Uploaded training zip ({count} images). URL: {url}")
        return url

    def train_lora(
        self,
        images,
        default_caption,
        api_key,
        steps=2000,
        learning_rate=0.0005,
        images_zip_url="",
    ):
        try:
            client = self.get_client(api_key)

            zip_url = images_zip_url.strip() if images_zip_url else ""
            if not zip_url:
                if not default_caption or not default_caption.strip():
                    raise ValueError(
                        "default_caption is required when training from an IMAGE "
                        "batch (no per-image caption files are generated here). "
                        "Provide a short caption/trigger word, or pass a "
                        "pre-built images_zip_url with its own .txt captions."
                    )
                zip_url = self._build_and_upload_zip(client, images)

            arguments = {
                "image_data_url": zip_url,
                "steps": int(steps),
                "learning_rate": float(learning_rate),
            }
            if default_caption and default_caption.strip():
                arguments["default_caption"] = default_caption.strip()

            result = self.call_api(client, "fal-ai/z-image-turbo-trainer-v2", arguments)

            lora_url = result["diffusers_lora_file"]["url"]
            return (lora_url,)

        except Exception as e:
            logger.error(f"Z-Image LoRA training failed: {str(e)}")
            raise RuntimeError(f"Z-Image LoRA training failed: {str(e)}") from e
