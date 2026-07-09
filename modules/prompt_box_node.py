class PromptBoxNode:
    """
    A simple text box node. Type any prompt and connect the output STRING
    to any node that accepts text input. Displays the text in the node UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "placeholder": "Write your prompt here...",
                        "dynamicPrompts": True,
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "passthrough"
    CATEGORY = "Superside"
    OUTPUT_NODE = True
    DESCRIPTION = "A simple prompt text box. Write your text and connect the output to any node."

    def passthrough(self, prompt, unique_id=None, extra_pnginfo=None):
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
                    node["widgets_values"] = [prompt]

        return {"ui": {"text": [prompt]}, "result": (prompt,)}
