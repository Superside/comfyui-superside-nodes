import logging

from .base_node import SupersideFalNode, APIClientMixin, API_KEY_INPUT_SPEC

logger = logging.getLogger(__name__)


class AnyLLMTextNode(SupersideFalNode, APIClientMixin):
    """
    Any LLM Text Node: Run text-only chat/completion models via fal.ai OpenRouter.

    This node is intended for pure text generation and reasoning tasks without image
    inputs. It uses the OpenRouter-backed fal endpoint and exposes a curated list of
    current text model options.
    """

    MODEL_OPTIONS = [
        # Google
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-pro",
        # Anthropic
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.5-sonnet",
        # OpenAI
        "openai/gpt-4o",
        "openai/gpt-4.1",
        "openai/gpt-5-chat",
        "openai/gpt-oss-120b",
        # Meta
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
        # Other
        "x-ai/grok-4-fast",
        "moonshotai/kimi-k2.5",
    ]

    ENDPOINT_CANDIDATES = [
        "openrouter/router",
        "fal-ai/any-llm",
    ]

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
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional system prompt",
                        "tooltip": "System prompt to provide context or instructions to the model",
                    },
                ),
                "model": (
                    cls.MODEL_OPTIONS,
                    {
                        "default": "google/gemini-2.5-flash",
                        "tooltip": "Text model to use via fal OpenRouter.",
                    },
                ),
                "reasoning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include reasoning when supported by the selected model.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Lower values = more predictable, higher values = more creative.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 32768,
                        "tooltip": "Maximum number of output tokens.",
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
        "Generate text with OpenRouter-backed LLMs on fal.ai. "
        "Supports Gemini, GPT, Claude 4.6, Llama, Grok, and Kimi models."
    )

    def prepare_arguments(self, prompt, **kwargs):
        arguments = {"prompt": prompt}

        if kwargs.get("system_prompt"):
            arguments["system_prompt"] = kwargs["system_prompt"]

        if kwargs.get("model") is not None:
            arguments["model"] = kwargs["model"]

        if kwargs.get("reasoning") is not None:
            arguments["reasoning"] = kwargs["reasoning"]

        if kwargs.get("temperature") is not None:
            arguments["temperature"] = kwargs["temperature"]

        if kwargs.get("max_tokens") is not None:
            arguments["max_tokens"] = kwargs["max_tokens"]

        return arguments

    def generate(self, prompt, api_key, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(prompt, **kwargs)
            model = kwargs.get("model", "google/gemini-2.5-flash")
            logger.info(f"Any LLM Text: model={model}, prompt='{prompt[:50]}...'")

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
                raise RuntimeError(
                    f"All Any LLM Text endpoints failed. Last error: {str(last_error)}"
                )

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
            logger.error(f"Any LLM Text generation failed: {str(e)}")
            raise RuntimeError(f"Any LLM Text generation failed: {str(e)}") from e
