import logging
from .base_node import (
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
    API_KEY_INPUT_SPEC,
)

logger = logging.getLogger(__name__)


class SupersideAnyLLMVisionNode(
    SupersideFalNode,
    ImageProcessingMixin,
    APIClientMixin,
):
    """
    Any LLM Vision Node: Use any vision language model from the fal.ai catalogue
    (powered by OpenRouter).

    This node provides access to multiple vision-capable LLMs including Claude, Gemini,
    GPT-4o, and Llama models for image analysis and understanding tasks.
    """

    MODEL_OPTIONS = [
        # Google
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "google/gemini-2.0-flash-001",
        "google/gemini-flash-1.5-8b",
        "google/gemini-flash-1.5",
        "google/gemini-pro-1.5",
        # Anthropic
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        # OpenAI
        "openai/gpt-4o",
        "openai/gpt-4.1",
        "openai/gpt-5-chat",
        "openai/gpt-oss-120b",
        # Meta
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
        "meta-llama/llama-3.2-90b-vision-instruct",
        # Other
        "moonshotai/kimi-k2.5",
    ]

    VISION_UPLOAD_MAX_DIMENSION = 1536

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Enter your prompt",
                    },
                ),
                "api_key": API_KEY_INPUT_SPEC,
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "System prompt to provide instructions",
                    },
                ),
                "model": (
                    cls.MODEL_OPTIONS,
                    {
                        "default": "google/gemini-2.5-flash-lite",
                        "tooltip": "Premium models (3x rate): Claude 4.6, Gemini 2.5 Pro, GPT-4o/4.1/5, Llama 90b/4",
                    },
                ),
                "reasoning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include reasoning in the response",
                    },
                ),
                "priority": (
                    ["latency", "throughput"],
                    {
                        "default": "latency",
                        "tooltip": "Latency: faster response | Throughput: better for batch processing",
                    },
                ),
                "auto_rescale_images": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically downscale large input images before upload.",
                    },
                ),
                "max_image_dimension": (
                    "INT",
                    {
                        "default": cls.VISION_UPLOAD_MAX_DIMENSION,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Maximum width or height for uploaded vision images when auto-rescale is enabled.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Lower values = more predictable, Higher values = more creative",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 32768,
                        "tooltip": "Maximum tokens in response",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "reasoning")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Use any vision language model for image analysis via OpenRouter on fal.ai. "
        "Supports Claude 4.6, Gemini 2.5, GPT-4o, Llama, Kimi and more."
    )

    # OpenRouter vision is primary; fal-ai/any-llm/vision as fallback
    ENDPOINT_CANDIDATES = [
        "openrouter/router/vision",
        "fal-ai/any-llm/vision",
    ]

    def prepare_image_urls(self, client, **kwargs):
        image_urls = []
        auto_rescale = kwargs.get("auto_rescale_images", True)
        max_dimension = kwargs.get("max_image_dimension", self.VISION_UPLOAD_MAX_DIMENSION)
        upload_max_dimension = max_dimension if auto_rescale else None

        for i in range(1, 7):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                try:
                    url = self.upload_image(client, kwargs[image_key], max_dimension=upload_max_dimension)
                    image_urls.append(url)
                    logger.info(f"Uploaded {image_key}: {url}")
                except Exception as e:
                    logger.warning(f"Failed to upload {image_key}: {str(e)}")
        return image_urls

    def prepare_arguments(self, client, prompt, **kwargs):
        arguments = {"prompt": prompt}

        image_urls = self.prepare_image_urls(client, **kwargs)
        if image_urls:
            arguments["image_urls"] = image_urls

        if kwargs.get("system_prompt"):
            arguments["system_prompt"] = kwargs["system_prompt"]

        if kwargs.get("model") is not None:
            arguments["model"] = kwargs["model"]

        if kwargs.get("reasoning") is not None:
            arguments["reasoning"] = kwargs["reasoning"]

        if kwargs.get("priority") is not None:
            arguments["priority"] = kwargs["priority"]

        if kwargs.get("temperature") is not None:
            arguments["temperature"] = kwargs["temperature"]

        if kwargs.get("max_tokens") is not None:
            arguments["max_tokens"] = kwargs["max_tokens"]

        return arguments

    def generate(self, prompt, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            model = kwargs.get("model", "google/gemini-2.5-flash-lite")
            logger.info(f"Any LLM Vision: model={model}, prompt='{prompt[:50]}...'")

            result = None
            last_error = None

            for endpoint in self.ENDPOINT_CANDIDATES:
                try:
                    logger.info(f"Trying endpoint: {endpoint}")
                    result = self.call_api(client, endpoint, arguments)
                    break
                except Exception as endpoint_error:
                    last_error = endpoint_error
                    logger.warning(f"Endpoint failed: {endpoint} - {str(endpoint_error)}")

            if result is None:
                error_text = str(last_error)
                if (
                    "worker_exceeded_resources" in error_text
                    or "Error 1102" in error_text
                    or "Worker exceeded resource limits" in error_text
                ):
                    raise RuntimeError(
                        "OpenRouter vision backend exceeded resource limits. "
                        "Try fewer images, smaller images, a shorter prompt, or a lighter model. "
                        f"Last error: {error_text}"
                    )
                raise RuntimeError(f"All endpoints failed. Last error: {error_text}")

            if "error" in result and result["error"]:
                raise RuntimeError(f"API returned error: {result['error']}")

            output = result.get("output", "")
            reasoning = result.get("reasoning", "")

            if not output:
                raise RuntimeError("No output was generated by the API.")

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
                        node["widgets_values"] = [output]

            return {"ui": {"text": [output]}, "result": (output, reasoning)}

        except Exception as e:
            logger.error(f"Any LLM Vision generation failed: {str(e)}")
            raise RuntimeError(f"Vision LLM generation failed: {str(e)}") from e
