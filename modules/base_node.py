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
# of the synchronous run() helper. A single long-held HTTP connection (what
# run() does under the hood) is prone to mid-flight disconnects on slow
# generations - the queue's short-lived polling requests are far more
# resilient for endpoints that commonly take several minutes.
QUEUED_ENDPOINTS = {
    "openai/gpt-image-2/edit",
    "bytedance/seedream/v5/pro/edit",
    "bytedance/seedream/v4.5/edit",
    "google/gemini-omni-flash/edit",
}

# Without a client-side timeout, fal_client.subscribe() waits indefinitely -
# a genuinely stuck job would hang the ComfyUI node forever instead of
# failing with a clear error. 20 minutes comfortably covers the slowest
# generations we've observed while still giving up eventually.
QUEUED_CLIENT_TIMEOUT = 1200


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


class VideoProcessingMixin:
    """
    Shared video upload/download helpers for Superside fal.ai nodes.

    Uses ComfyUI's native VIDEO type (comfy_api.latest.Input.Video /
    InputImpl.VideoFromFile) so users can wire a LoadVideo node straight in
    and a SaveVideo/PreviewVideo node straight out - no manual URL copying.
    The import is done lazily (not at module load time) so the rest of this
    package still works on older ComfyUI installs that predate this API;
    only the video nodes themselves would fail, with a clear error.
    """

    @staticmethod
    def _video_input_impl():
        try:
            from comfy_api.latest import InputImpl
            return InputImpl
        except ImportError as e:
            raise RuntimeError(
                "This node requires ComfyUI's native VIDEO type "
                "(comfy_api.latest.InputImpl), which isn't available in this "
                "ComfyUI install. Please update ComfyUI."
            ) from e

    def upload_video(self, client, video_input):
        """
        Upload a ComfyUI VIDEO input to fal.ai and return the URL.

        Args:
            client: fal_client.SyncClient scoped to the node's api_key input.
            video_input: a comfy_api VideoInput (e.g. from a LoadVideo node).

        Returns:
            str: URL of the uploaded video.
        """
        try:
            source = video_input.get_stream_source()
            if isinstance(source, str):
                url = client.upload_file(source)
            else:
                source.seek(0)
                url = client.upload(source.read(), "video/mp4")

            logger.info(f"Video uploaded successfully. URL: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to upload video: {str(e)}")
            raise

    def download_video(self, video_url):
        """
        Download a video from a URL and wrap it as a native ComfyUI VIDEO
        object, ready to connect to SaveVideo/PreviewVideo.
        """
        InputImpl = self._video_input_impl()
        response = requests.get(video_url)
        response.raise_for_status()
        return InputImpl.VideoFromFile(io.BytesIO(response.content))


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
                result = client.subscribe(
                    endpoint,
                    arguments=arguments,
                    on_enqueue=lambda request_id: logger.info(
                        "Submitted fal.ai request %s to %s", request_id, endpoint
                    ),
                    client_timeout=QUEUED_CLIENT_TIMEOUT,
                )
            else:
                logger.info(f"Using synchronous API call to {endpoint}")
                result = client.run(endpoint, arguments=arguments)

            logger.debug(f"API response: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Failed to call fal.ai API: {str(e)}") from e


class DetailSheetCompositionMixin:
    """
    Shared crop/compose logic for "detail sheet" nodes (Smart Detail Sheet's
    AI-driven detection and Manual Detail Sheet's user-drawn boxes both build
    on this): crop a region from the source photo at native resolution,
    upscale it locally (Lanczos, no AI), and composite a set of such crops
    alongside the original into one final image - the original photo stays
    the dominant element, but the detail crops stay legible.
    """

    # A crop whose pixel std-dev is below this is treated as a flat/blank
    # region (the detection/selection missed the actual detail) and is
    # dropped from the final sheet instead of showing an empty patch. Kept
    # low because legitimate details can be subtle (e.g. silver hardware on
    # a white background) - a truly blank/background crop is much closer to 0.
    BLANK_CROP_STD_THRESHOLD = 4.0

    # The detail block's corresponding dimension (column height for the side
    # layout, row width for the below layout) is kept within this fraction
    # range of the original's own dimension: never below the floor (so
    # details stay legible) and never above the ceiling of 1.0 (so the
    # original photo stays the dominant element). Floor is always well below
    # the ceiling, so applying it can never overshoot the cap.
    MIN_DETAIL_BLOCK_RATIO = 0.35
    MAX_DETAIL_BLOCK_RATIO = 1.0

    @staticmethod
    def _tensor_to_pil(image):
        """Convert a ComfyUI IMAGE tensor (or numpy array) to an RGB PIL image."""
        image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        if image_np.ndim == 4:
            image_np = image_np[0]
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        return Image.fromarray(image_np).convert("RGB")

    def _crop_rect_scale_check_blank(self, source_img, x1, y1, x2, y2, crop_scale, label="detail"):
        """
        Crop an exact pixel rect from source_img, discard it if it looks
        blank/flat (a missed detection or an empty area the user boxed by
        mistake), and upscale it uniformly (never distorts aspect ratio).
        """
        width, height = source_img.size
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(width, int(round(x2)))
        y2 = min(height, int(round(y2)))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop region for '{label}' - skipping")
            return None

        cropped = source_img.crop((x1, y1, x2, y2))

        crop_std = float(np.array(cropped.convert("L")).std())
        if crop_std < self.BLANK_CROP_STD_THRESHOLD:
            logger.warning(f"'{label}' crop looks blank (std={crop_std:.1f}) - discarding")
            return None

        new_size = (max(1, int(cropped.width * crop_scale)), max(1, int(cropped.height * crop_scale)))
        return cropped.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _rescale_crops(crops, scale):
        """Uniformly rescale every crop by the same factor (never distorts)."""
        if scale == 1.0:
            return crops
        rescaled = []
        for c in crops:
            new_size = (max(1, int(c.width * scale)), max(1, int(c.height * scale)))
            rescaled.append(c.resize(new_size, Image.LANCZOS))
        return rescaled

    def _fit_block_to_original(self, resized_crops, natural_size, original_size):
        """
        Uniformly rescale a column/row of crops so its overall size sits
        within [MIN_DETAIL_BLOCK_RATIO, MAX_DETAIL_BLOCK_RATIO] of the
        original's corresponding dimension. Scales up if the details would
        otherwise be too small to read, scales down if they'd overwhelm the
        original - same factor applied to every crop, so proportions never
        distort.
        """
        floor = original_size * self.MIN_DETAIL_BLOCK_RATIO
        ceiling = original_size * self.MAX_DETAIL_BLOCK_RATIO

        if natural_size < floor:
            fit_scale = floor / natural_size
        elif natural_size > ceiling:
            fit_scale = ceiling / natural_size
        else:
            return resized_crops

        return self._rescale_crops(resized_crops, fit_scale)

    def _compose_detail_sheet(self, original_img, detail_crops):
        """
        Portrait/tall originals (width < height, e.g. 4:5) get their detail
        crops arranged in a column beside the image, so the canvas doesn't
        keep growing taller. Landscape/square originals (width >= height,
        e.g. 16:9) get their crops in a row underneath instead.

        The original image must stay the dominant element, but the details
        must also stay legible: the detail column/row as a whole is kept
        within MIN_DETAIL_BLOCK_RATIO-MAX_DETAIL_BLOCK_RATIO of the original's
        corresponding dimension (height for the side layout, width for the
        below layout). The whole block of crops is scaled uniformly (same
        factor for every crop, so their individual proportions stay
        untouched) to land inside that range.
        """
        is_portrait = original_img.width < original_img.height
        gap = 24

        if is_portrait:
            # Column of details to the right, each normalized to the same width.
            col_width = max(c.width for c in detail_crops)
            resized_crops = []
            for c in detail_crops:
                if c.width != col_width:
                    scale = col_width / c.width
                    c = c.resize((col_width, max(1, int(c.height * scale))), Image.LANCZOS)
                resized_crops.append(c)

            col_height = sum(c.height for c in resized_crops) + gap * (len(resized_crops) - 1)

            resized_crops = self._fit_block_to_original(resized_crops, col_height, original_img.height)
            col_width = max(c.width for c in resized_crops)
            col_height = sum(c.height for c in resized_crops) + gap * (len(resized_crops) - 1)

            canvas_width = original_img.width + gap + col_width
            canvas_height = max(original_img.height, col_height)

            canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
            canvas.paste(original_img, (0, (canvas_height - original_img.height) // 2))

            x = original_img.width + gap
            y = (canvas_height - col_height) // 2
            for c in resized_crops:
                canvas.paste(c, (x, y))
                y += c.height + gap

            return canvas

        # Landscape/square: row of details below, each normalized to the same height.
        row_height = max(c.height for c in detail_crops)
        resized_crops = []
        for c in detail_crops:
            if c.height != row_height:
                scale = row_height / c.height
                c = c.resize((max(1, int(c.width * scale)), row_height), Image.LANCZOS)
            resized_crops.append(c)

        row_width = sum(c.width for c in resized_crops) + gap * (len(resized_crops) - 1)

        resized_crops = self._fit_block_to_original(resized_crops, row_width, original_img.width)
        row_height = max(c.height for c in resized_crops)
        row_width = sum(c.width for c in resized_crops) + gap * (len(resized_crops) - 1)

        canvas_width = max(original_img.width, row_width)
        canvas_height = original_img.height + gap + row_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

        orig_x = (canvas_width - original_img.width) // 2
        canvas.paste(original_img, (orig_x, 0))

        row_x = (canvas_width - row_width) // 2
        y = original_img.height + gap
        for c in resized_crops:
            canvas.paste(c, (row_x, y))
            row_x += c.width + gap

        return canvas
