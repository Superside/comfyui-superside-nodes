class SupersideSceneRealismPromptNode:
    """
    Superside Scene Realism Dial: one dial for how strongly to push
    photorealism when enhancing an arbitrary scene region (not skin-specific).

    Generic counterpart of Superside Skin Intensity Dial. Wire it into the same
    places: `prompt_fragment` -> Superside Combine Prompt's part2, and
    `strength` -> Superside Z-Image Turbo Inpaint+LoRA. `lora_scale` is only used
    if you also wire a generic detail LoRA into the inpaint node; leave it
    ignored for the plain (no-LoRA) realism pass.

    Use it to add realistic material texture and surface detail to a masked
    region of a scene - a sofa, a wall, a product, a dimmer plate - the same
    mask -> refine -> composite-back principle as the re-skin pipeline, just not
    tied to a skin LoRA.
    """

    CATEGORY = "Superside"

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
            "subtle photographic realism, clean natural materials, gentle realistic surface detail, true colors",
            0.5,
            0.20,
        ),
        "2 - subtle": (
            "natural photographic realism, realistic material texture, soft surface detail, physically plausible lighting",
            0.75,
            0.30,
        ),
        "3 - medium (default)": (
            "photorealistic, realistic material textures and surface detail, physically accurate lighting and shadows, "
            "sharp focus, high detail, natural true colors",
            1.0,
            0.40,
        ),
        "4 - strong": (
            "highly detailed photorealism, rich material texture, fine surface detail and micro-contrast, crisp "
            "physically-accurate lighting and reflections, sharp focus",
            1.3,
            0.50,
        ),
        "5 - extreme": (
            "extreme photorealistic detail, macro surface texture, richly detailed realistic materials, hyper-real "
            "lighting, reflections and micro-detail, ultra sharp",
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
    DESCRIPTION = (
        "One dial for scene realism intensity (generic, not skin): outputs a "
        "matched realism prompt fragment + lora_scale + strength for the chosen "
        "level. Wire prompt_fragment into Superside Combine Prompt's part2, and "
        "strength into Superside Z-Image Turbo Inpaint+LoRA. lora_scale only "
        "matters if a generic detail LoRA is wired in."
    )

    def get_settings(self, level="3 - medium (default)"):
        fragment, lora_scale, strength = self._TABLE.get(level, self._TABLE["3 - medium (default)"])
        return (fragment, float(lora_scale), float(strength))
