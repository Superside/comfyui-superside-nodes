import logging

logger = logging.getLogger(__name__)

MAX_SPLITS = 10


class SupersidePromptSplitterNode:
    """
    Splits a prompt string by a configurable separator symbol into up to 10 individual
    STRING outputs. Unused slots return an empty string. Displays the split result in the UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Text 1 * Text 2 * Text 3 ...",
                    },
                ),
                "separator": (
                    "STRING",
                    {
                        "default": "*",
                        "multiline": False,
                        "tooltip": "Symbol used to split the prompt. Default is *",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",) * MAX_SPLITS
    RETURN_NAMES = tuple(f"text_{i+1}" for i in range(MAX_SPLITS))
    FUNCTION = "split"
    CATEGORY = "Superside"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Split a prompt into up to 10 separate text outputs using a separator symbol. "
        "Useful for feeding individual prompts into multi-image nodes."
    )

    def split(self, prompt, separator, unique_id=None, extra_pnginfo=None):
        if not separator:
            separator = "*"

        parts = [p.strip() for p in prompt.split(separator)]
        parts = parts[:MAX_SPLITS]
        while len(parts) < MAX_SPLITS:
            parts.append("")

        display_lines = [f"[{i+1}] {p}" for i, p in enumerate(parts) if p]
        display = "\n".join(display_lines) if display_lines else "(empty)"

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
                    node["widgets_values"] = [display]

        logger.debug(f"PromptSplitter: {len([p for p in parts if p])} active parts")
        return {"ui": {"text": [display]}, "result": tuple(parts)}
