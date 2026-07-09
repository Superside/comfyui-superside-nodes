import base64
import io
import json
import logging

import fal_client
import numpy as np
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# Shared INPUT_TYPES entry for the required api_key STRING input. Every node
# in this package uses this exact spec so the widget looks and behaves the
# same everywhere.
API_KEY_INPUT_SPEC = (
    "STRING",
    {
        "multiline": False,
        "default": "",
        "placeholder": "fal.ai API key (ask your project lead)",
    },
)

# Endpoints that must be called through fal.ai's queue (submit + poll) instead
# of the synchronous run() helper. Add an endpoint here if it requires queued
# execution.
QUEUED_ENDPOINTS = {
    "openai/gpt-image-2/edit",
}


class SupersideFalNode:
    """
    Base class for Superside's fal.ai ComfyUI nodes.

    Unlike the internal fal-flux-nodes package this is derived from, there is
    no config.ini or FAL_KEY environment variable fallback here. Every node
    requires an explicit `api_key` input, provided per-workflow by whoever set
    it up (the project lead hands out the key - it is never baked into this
    repo).
    """

    CATEGORY = "Superside"

    @staticmethod
    def get_client(api_key):
        """Build a fal.ai client scoped to the given key. No global state."""
        if not api_key or not api_key.strip():
            raise ValueError(
                "api_key is required. Ask your project lead for the fal.ai "
                "API key and paste it into the api_key input of this node."
            )
        return fal_client.SyncClient(key=api_key.strip())


class ImageProcessingMixin:
    """Shared image upload/download helpers for Superside fal.ai nodes."""

    def _load_image_bytes_from_url(self, img_url):
        """Load image bytes from either an HTTP URL or a data URI."""
        if not isinstance(img_url, str) or not img_url:
            raise ValueError("Image URL is missing or invalid.")

        if img_url.startswith("data:"):
            try:
                header, encoded = img_url.split(",", 1)
            except ValueError as e:
                raise ValueError("Invalid data URI image format.") from e

            if ";base64" not in header:
                raise ValueError("Only base64-encoded data URI images are supported.")

            try:
                return base64.b64decode(encoded)
            except Exception as e:
                raise ValueError("Failed to decode base64 image data.") from e

        response = requests.get(img_url)
        response.raise_for_status()
        return response.content

    def upload_image(self, client, image_tensor, max_dimension=None):
        """
        Upload an image tensor to fal.ai and return the URL.

        Args:
            client: fal_client.SyncClient scoped to the node's api_key input.
            image_tensor: PyTorch tensor representing the image.
            max_dimension: Optional maximum width/height. Images larger than
                this will be downscaled before upload while preserving aspect
                ratio.

        Returns:
            str: URL of the uploaded image.
        """
        try:
            if isinstance(image_tensor, torch.Tensor):
                image_np = image_tensor.cpu().numpy()
            else:
                image_np = image_tensor

            if image_np.ndim == 4 and image_np.shape[0] == 1:  # (1, H, W, C)
                image_np = image_np.squeeze(0)
            elif image_np.ndim == 3 and image_np.shape[0] == 3:  # (C, H, W)
                image_np = np.transpose(image_np, (1, 2, 0))

            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

            image = Image.fromarray(image_np)

            if max_dimension is not None:
                width, height = image.size
                longest_edge = max(width, height)
                if longest_edge > max_dimension:
                    scale = max_dimension / float(longest_edge)
                    resized_size = (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    )
                    image = image.resize(resized_size, Image.LANCZOS)
                    logger.info(
                        "Downscaled image before upload from "
                        f"{width}x{height} to {resized_size[0]}x{resized_size[1]}"
                    )

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            url = client.upload(img_bytes, "image/png")
            logger.info(f"Image uploaded successfully. URL: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload image: {str(e)}")
            raise

    def process_images(self, result):
        """Process the API response and convert images to tensors."""
        if "images" not in result or not result["images"]:
            raise RuntimeError("No images were generated by the API.")

        output_images = []
        for index, img_info in enumerate(result["images"]):
            try:
                if not isinstance(img_info, dict) or "url" not in img_info:
                    logger.error(f"Invalid image info for image {index}")
                    continue

                img_url = img_info["url"]
                image_bytes = self._load_image_bytes_from_url(img_url)
                img = Image.open(io.BytesIO(image_bytes))

                if img.mode != "RGB":
                    img = img.convert("RGB")

                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                output_images.append(img_tensor)

            except Exception as e:
                logger.error(f"Failed to process image {index}: {str(e)}")
                continue

        if not output_images:
            raise RuntimeError("Failed to process any generated images.")

        output_tensor = torch.cat(output_images, dim=0)
        return (output_tensor,)


class APIClientMixin:
    """Shared fal.ai API-call helper for Superside nodes."""

    def call_api(self, client, endpoint, arguments):
        """
        Call the fal.ai API with the given endpoint and arguments.

        Args:
            client: fal_client.SyncClient scoped to the node's api_key input.
            endpoint (str): The API endpoint to call.
            arguments (dict): Arguments to pass to the API.

        Returns:
            dict: API response.
        """
        logger.debug(f"API request payload: {json.dumps(arguments, indent=2)}")

        try:
            if endpoint in QUEUED_ENDPOINTS:
                logger.info(f"Using queued API call to {endpoint}")
                handle = client.submit(endpoint, arguments=arguments)
                logger.info("Submitted fal.ai request %s to %s", handle.request_id, endpoint)
                for status in handle.iter_events(with_logs=False, interval=5.0):
                    if status.__class__.__name__ == "Completed":
                        break
                result = handle.get()
            else:
                logger.info(f"Using synchronous API call to {endpoint}")
                result = client.run(endpoint, arguments=arguments)

            logger.debug(f"API response: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Failed to call fal.ai API: {str(e)}") from e
