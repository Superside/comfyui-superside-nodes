class SupersideSkinIntensityPromptNode:
    """
    Superside Skin Intensity: a single dial to raise or lower how strongly
    the skin-texture LoRA reads in the final image.

    Prompt wording alone is a soft, unreliable lever for a trained LoRA - the
    two numeric params that actually move the needle are lora_scale and
    strength on Superside Z-Image Turbo Inpaint+LoRA. This node ties a single
    "level" choice to a matched set of: a prompt fragment (to plug into
    Superside Combine Prompt's part2), a lora_scale, and a strength - so
    picking one level moves all three together instead of hand-tuning three
    separate fields every time.
    """

    LEVELS = [
        "1 - very subtle",
        "2 - subtle",
        "3 - medium (default)",
        "4 - strong",
        "5 - extreme",
    ]

    # level -> (prompt_fragment, lora_scale, strength)
    _TABLE = {
        "1 - very subtle": (
            "soft natural skin, smooth even skin tone, minimal visible pores, gentle realistic texture",
            0.5,
            0.22,
        ),
        "2 - subtle": (
            "natural skin texture, soft visible pores, light realistic imperfections",
            0.75,
            0.30,
        ),
        "3 - medium (default)": (
            "realistic skin texture, natural pores and fine imperfections, visible skin grain, photorealistic skin",
            1.0,
            0.40,
        ),
        "4 - strong": (
            "highly detailed skin texture, pronounced visible pores, fine lines and imperfections, hyper-realistic macro skin detail",
            1.3,
            0.50,
        ),
        "5 - extreme": (
            "extreme macro skin detail, heavily pronounced pores, deep skin texture, raw unretouched skin, "
            "visible micro-wrinkles and imperfections, hyper-detailed macro photography skin",
            1.6,
            0.60,
        ),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "level": (cls.LEVELS, {"default": "3 - medium (default)"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("prompt_fragment", "lora_scale", "strength")
    FUNCTION = "get_settings"
    CATEGORY = "Superside"
    DESCRIPTION = (
        "One dial for skin-texture intensity: outputs a matched prompt "
        "fragment + lora_scale + strength for the chosen level. Wire "
        "prompt_fragment into Superside Combine Prompt's part2, and "
        "lora_scale/strength into Superside Z-Image Turbo Inpaint+LoRA."
    )

    def get_settings(self, level="3 - medium (default)"):
        fragment, lora_scale, strength = self._TABLE.get(level, self._TABLE["3 - medium (default)"])
        return (fragment, float(lora_scale), float(strength))
