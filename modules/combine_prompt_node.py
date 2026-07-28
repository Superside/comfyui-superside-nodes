class SupersideCombinePromptNode:
    """
    Superside Combine Prompt: in-house replacement for Comfyroll's
    "CR Combine Prompt". Same 4-parts + separator contract: literal
    concatenation part1 + sep + part2 + sep + part3 + sep + part4 - empty
    parts still produce the separator (no "skip empty" logic), matching the
    original exactly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "part1": ("STRING", {"default": "", "multiline": True}),
                "part2": ("STRING", {"default": "", "multiline": True}),
                "part3": ("STRING", {"default": "", "multiline": True}),
                "part4": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": ",", "multiline": False}),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_value"
    DESCRIPTION = "Concatenate up to 4 text parts with a separator - in-house equivalent of Comfyroll's CR Combine Prompt."

    def get_value(self, part1="", part2="", part3="", part4="", separator=""):
        prompt = part1 + separator + part2 + separator + part3 + separator + part4
        return (prompt,)
