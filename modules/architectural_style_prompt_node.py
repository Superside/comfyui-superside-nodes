"""
Superside Architectural Style Dial.

Builds a prompt fragment for interior / real-estate image generation from three
axes:

    style  x  room  x  realism level

Design principle (important): the LoRA already learns the *look* of each style -
its woods, fabrics, metals, lighting and color grading. So the prompt should
describe the style through GENERAL material / texture / finish / palette / light
categories, NOT an exhaustive furniture inventory. Dumping a long list of
specific objects ("tufted Chesterfield + wingbacks + carved table + persian rug
+ brass lamps + candelabra ...") into one inference prompt makes the model try
to cram them all into the frame, which causes clutter, duplicated / melted
objects and malformations. Categories steer the render toward the style and let
the model compose the scene naturally.

So:
  * style  -> a general material / fabric / surface / metal / palette / light
              descriptor (the character of the style).
  * room   -> a MINIMAL scene anchor (only the pieces that define the room type,
              e.g. a bed = a bedroom), not a styled furniture list.
  * level  -> photographic-detail intensity, also driving strength / lora_scale.

The exhaustive per-room object catalogs live in
`architectural_styles_glossary.txt` as REFERENCE (useful for training captions
or human lookup) - deliberately kept out of the runtime prompt.

Wire it like the other dials: `prompt_fragment` -> Superside Combine Prompt's
part2, `strength` -> an inpaint node, `lora_scale` -> the LoRA scale input
(only used if a style / detail LoRA is actually wired in).
"""


class SupersideArchitecturalStylePromptNode:
    CATEGORY = "Superside"

    STYLES = ["transitional", "traditional", "modern"]

    ROOMS = [
        "any",
        "living_room",
        "bedroom",
        "kitchen",
        "dining_room",
        "bathroom",
        "hallway",
    ]

    LEVELS = [
        "1 - very subtle",
        "2 - subtle",
        "3 - medium (default)",
        "4 - strong",
        "5 - extreme",
    ]

    # ------------------------------------------------------------------ #
    # Style character - GENERAL material / fabric / surface / metal /
    # palette / light categories, not an object inventory. Master copy of
    # the full vocabulary is architectural_styles_glossary.txt.
    # ------------------------------------------------------------------ #
    _STYLE_CORE = {
        "transitional": (
            "transitional interior, a warm balanced blend of classic and "
            "contemporary, oak and soft painted wood, linen and wool textiles, "
            "quartz and marble surfaces, subway tile, brushed brass and bronze "
            "accents, warm evening lighting, greige cream and taupe palette with "
            "muted navy and sage accents, coffered ceilings and light millwork"
        ),
        "traditional": (
            "traditional interior, a warm formal heritage feel, rich stained "
            "cherry walnut and mahogany wood, brick and stone, tufted leather "
            "velvet damask and toile textiles, persian rugs, marble and "
            "butcher-block surfaces, aged brass and bronze, warm incandescent "
            "lighting, warm brown cream deep-green navy and burgundy palette, "
            "wainscoting and beadboard"
        ),
        "modern": (
            "modern interior, clean minimal and contemporary, light oak walnut "
            "and charcoal wood, quartz honed stone and concrete, matte black "
            "metal, flat-weave wool boucle and jute textiles, large black-framed "
            "glass, balanced natural lighting, white warm-oak and charcoal "
            "palette with sage accents, uncluttered surfaces"
        ),
    }

    # Minimal, style-agnostic scene anchor: only the pieces that define the room
    # type. The style core + LoRA supply the styling; this just sets the scene.
    _ROOM_ANCHOR = {
        "living_room": "a living room with a sofa, coffee table and fireplace",
        "bedroom": "a bedroom with a bed, nightstands and a dresser",
        "kitchen": "a kitchen with an island, cabinetry and pendant lighting",
        "dining_room": "a dining room with a dining table and chairs",
        "bathroom": "a bathroom with a vanity, mirror and shower",
        "hallway": "a hallway with a console table and a runner rug",
    }

    # Shared photographic base - the LoRA's common real-estate look.
    _BASE = (
        "warm evening interior real-estate photography, interior lights on, "
        "inviting magazine-quality staging"
    )

    # level -> (detail phrase, lora_scale, strength)
    _LEVEL = {
        "1 - very subtle": (
            "subtle photographic realism, clean natural materials, gentle "
            "surface detail, true colors",
            0.5,
            0.20,
        ),
        "2 - subtle": (
            "natural photographic realism, realistic material texture, soft "
            "surface detail, physically plausible lighting",
            0.75,
            0.30,
        ),
        "3 - medium (default)": (
            "photorealistic, realistic material textures and surface detail, "
            "physically accurate lighting and shadows, sharp focus, natural true "
            "colors",
            1.0,
            0.40,
        ),
        "4 - strong": (
            "highly detailed photorealism, rich material texture and "
            "micro-contrast, crisp physically-accurate lighting and reflections, "
            "sharp focus, professional interior photography",
            1.3,
            0.50,
        ),
        "5 - extreme": (
            "extreme photorealistic detail, richly detailed realistic materials, "
            "hyper-real lighting reflections and micro-detail, ultra sharp, "
            "magazine-quality interior photography",
            1.6,
            0.60,
        ),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (cls.STYLES, {"default": "transitional"}),
                "room": (cls.ROOMS, {"default": "any"}),
                "level": (cls.LEVELS, {"default": "3 - medium (default)"}),
            },
            "optional": {
                "trigger_word": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "optional LoRA trigger token, prepended verbatim",
                        "tooltip": "If your style LoRA was trained with a trigger "
                        "word/token, put it here and it is prepended to the "
                        "prompt fragment. Leave empty if the LoRA has none.",
                    },
                ),
                "include_base": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Append the shared 'warm evening real-estate "
                        "photography' base look that all three styles share. "
                        "Disable if your own base prompt already sets the scene.",
                    },
                ),
                "include_room_anchor": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prepend a minimal scene anchor for the chosen "
                        "room (e.g. 'a bedroom with a bed and nightstands'). "
                        "Kept intentionally general to avoid over-constraining "
                        "the composition. Ignored when room = any.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("prompt_fragment", "lora_scale", "strength")
    FUNCTION = "get_settings"
    DESCRIPTION = (
        "Interior architectural-style prompt driver: pick a style "
        "(transitional / traditional / modern), a room, and a realism level. "
        "The style is described through GENERAL material / fabric / surface / "
        "palette / light categories (not a furniture inventory) so the LoRA can "
        "compose the scene without malformations; room adds only a minimal scene "
        "anchor. Outputs a prompt fragment + lora_scale + strength. Vocabulary "
        "master copy: architectural_styles_glossary.txt. Wire prompt_fragment "
        "into Superside Combine Prompt's part2, strength into an inpaint node, "
        "lora_scale into a style/detail LoRA scale."
    )

    def get_settings(
        self,
        style="transitional",
        room="any",
        level="3 - medium (default)",
        trigger_word="",
        include_base=True,
        include_room_anchor=True,
    ):
        core = self._STYLE_CORE.get(style, self._STYLE_CORE["transitional"])
        level_detail, lora_scale, strength = self._LEVEL.get(
            level, self._LEVEL["3 - medium (default)"]
        )

        parts = []
        if trigger_word and trigger_word.strip():
            parts.append(trigger_word.strip())

        # Scene anchor first (sets the subject), then the style character.
        if include_room_anchor:
            anchor = self._ROOM_ANCHOR.get(room, "")
            if anchor:
                parts.append(anchor)

        parts.append(core)

        if include_base:
            parts.append(self._BASE)

        parts.append(level_detail)

        fragment = ", ".join(p for p in parts if p)
        return (fragment, float(lora_scale), float(strength))
