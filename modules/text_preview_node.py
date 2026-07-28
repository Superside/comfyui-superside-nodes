class SupersideTextPreviewNode:
    """
    Superside Text Preview: in-house replacement for the sibling
    'superside-utility-nodes' package's Text Preview Node, so this repo has
    no dependency on that other package either. Displays an incoming
    STRING (e.g. a trained LoRA's file URL) in the node itself.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    DESCRIPTION = "Display an incoming STRING in the node graph (e.g. a trained LoRA URL)."

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if isinstance(extra_pnginfo, list) and extra_pnginfo and isinstance(extra_pnginfo[0], dict) and "workflow" in extra_pnginfo[0]:
                workflow = extra_pnginfo[0]["workflow"]
                node = next((x for x in workflow["nodes"] if str(x["id"]) == str(unique_id)), None)
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": [text]}, "result": (text,)}
