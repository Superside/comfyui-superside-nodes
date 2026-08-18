"""
Superside Architectural Style Dial.

Builds a prompt fragment for interior / real-estate image generation from three
axes:

    style  x  room  x  realism level

The style + room vocabulary DEFINES the look (materials, cabinetry, furniture,
lighting, ceilings, floors, fixtures, palette); the realism level controls how
hard to push photographic detail and also drives `strength` / `lora_scale` for a
downstream inpaint / LoRA pass.

Designed as the prompt driver for a LoRA trained on three interior styles -
TRANSITIONAL, TRADITIONAL, MODERN. The full human-readable vocabulary lives in
`architectural_styles_glossary.txt` next to this file; the condensed,
prompt-ready phrases below are mirrored from it. Expand the .txt first, then
mirror new entries here so the two stay in sync.

Wire it like the other dials: `prompt_fragment` -> Superside Combine Prompt's
part2, `strength` -> an inpaint node, `lora_scale` -> the LoRA scale input
(only used if a detail / style LoRA is actually wired in).
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
    # Style vocabulary - condensed, prompt-ready. Master copy of the full
    # descriptions is architectural_styles_glossary.txt (keep in sync).
    # ------------------------------------------------------------------ #
    _STYLE_CORE = {
        "transitional": (
            "transitional interior style, a balanced blend of classic and "
            "contemporary, warm greige and soft white walls, crown molding and "
            "coffered ceilings, wainscoting, medium oak hardwood floors, neutral "
            "linen upholstery, layered muted-pattern rugs, mixed painted and "
            "dark-wood furniture, warm evening lighting with brass and bronze "
            "accents, calm neutral palette with muted navy and sage accents"
        ),
        "traditional": (
            "traditional interior style, classic and formal with heritage "
            "character, brick and masonry fireplaces, rich stained cherry and "
            "walnut wood, carved furniture, tufted leather and skirted "
            "upholstery, wingback chairs, herringbone and wide-plank floors, "
            "oriental persian rugs, framed oil landscape paintings in gilt "
            "frames, wainscoting and beadboard, warm incandescent lighting, "
            "palette of warm brown, cream, deep green, navy and burgundy"
        ),
        "modern": (
            "modern interior style, clean minimalist contemporary design, "
            "flat-panel slab cabinetry in light oak walnut or charcoal, "
            "waterfall quartz surfaces, matte black hardware and fixtures, large "
            "black-framed floor-to-ceiling windows, linear gas fireplace, "
            "mid-century walnut furniture, wide-plank light oak floors, low-pile "
            "jute rugs, sculptural minimal lighting, uncluttered palette of "
            "white warm oak and charcoal with sage accents"
        ),
    }

    # _STYLE_ROOM[style][room] -> room-specific element phrase.
    _STYLE_ROOM = {
        "transitional": {
            "living_room": (
                "coffered ceiling living room, built-in shelving flanking a "
                "fireplace, neutral sectional and accent chairs, wood coffee "
                "table, layered area rug, glass globe or drum-shade chandelier, "
                "white-shade table lamps, framed landscape art"
            ),
            "bedroom": (
                "upholstered tufted headboard, layered neutral bedding, matching "
                "nightstands and table lamps, drum or lantern pendant, bench at "
                "the foot of the bed, patterned area rug, soft lamplight"
            ),
            "kitchen": (
                "shaker cabinets in white grey or navy, quartz or marble "
                "waterfall island, subway tile backsplash, glass and metal "
                "pendant lights over the island, stainless or paneled "
                "appliances, upholstered counter stools"
            ),
            "dining_room": (
                "rectangular wood dining table, upholstered parsons or spindle "
                "chairs, linear or candelabra chandelier, sideboard buffet, "
                "wainscoting, framed art"
            ),
            "bathroom": (
                "quartz-top vanity, framed mirror, porcelain tile, "
                "glass-enclosed shower, brushed nickel or matte black fixtures, "
                "sconce lighting"
            ),
            "hallway": (
                "console table with a lamp, patterned runner rug, framed gallery "
                "wall, lantern or flush-mount pendant, wainscoting"
            ),
        },
        "traditional": {
            "living_room": (
                "brick fireplace with a wood mantel, built-in bookshelves, "
                "tufted leather Chesterfield or skirted sofa, wingback club "
                "chairs, carved wood coffee table, persian rug, brass table "
                "lamps, candelabra chandelier, oil landscape paintings"
            ),
            "bedroom": (
                "carved four-poster or spindle wood bed, damask and toile "
                "textiles, oriental rug, brass and ceramic table lamps, "
                "upholstered wingback chair, wood dresser, warm lamplight"
            ),
            "kitchen": (
                "raised-panel stained cherry or painted wood cabinets, "
                "farmhouse sink, natural stone or butcher-block counters, "
                "handmade subway tile, bronze or brass pendants, dark vent hood, "
                "warm wood floors"
            ),
            "dining_room": (
                "formal wood dining table, upholstered or wood dining chairs, "
                "shaded candelabra chandelier, wall paneling and wainscoting, "
                "sideboard with lamps, wall sconces, framed landscape art, "
                "persian rug"
            ),
            "bathroom": (
                "furniture-style wood vanity, marble counter, classic subway or "
                "mosaic tile, polished nickel or brass fixtures, framed mirror, "
                "sconce lighting"
            ),
            "hallway": (
                "wood staircase with turned balusters, console table, framed oil "
                "paintings, wall sconces, schoolhouse or lantern pendant, runner "
                "rug, warm lamplight"
            ),
        },
        "modern": {
            "living_room": (
                "low-profile upholstered sofa in grey blue or olive, walnut "
                "coffee table, minimal styling, large black-framed windows, "
                "linear gas fireplace with a tile or stone surround, recessed "
                "and sculptural lighting, abstract art, floating media console"
            ),
            "bedroom": (
                "low platform bed, light oak or walnut nightstands, minimal "
                "neutral bedding with a sage or charcoal accent, sculptural "
                "paper-lantern pendant or flush mount, large windows, low-pile "
                "rug, a single framed artwork, clean walls"
            ),
            "kitchen": (
                "flat-panel light oak charcoal or white slab cabinets, waterfall "
                "quartz island, integrated or stainless appliances, matte black "
                "faucet and hardware, glass globe or black pendants, minimal "
                "open shelving, wide-plank oak floor"
            ),
            "dining_room": (
                "round or rectangular oak or dark-wood table, sculptural curved "
                "or cane dining chairs, cluster or linear pendant, abstract art, "
                "minimal sideboard, large windows"
            ),
            "bathroom": (
                "floating wood vanity, quartz top, backlit LED mirror, "
                "large-format porcelain tile, matte black fixtures, frameless "
                "glass shower"
            ),
            "hallway": (
                "minimal console table, black-framed gallery wall, wall sconces "
                "or recessed lighting, clean walls, oak floor, simple-railing "
                "staircase"
            ),
        },
    }

    # Shared photographic base - applied to every style so the LoRA's common
    # real-estate look is always present.
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
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("prompt_fragment", "lora_scale", "strength")
    FUNCTION = "get_settings"
    DESCRIPTION = (
        "Interior architectural-style prompt driver: pick a style "
        "(transitional / traditional / modern), a room, and a realism level, "
        "and it outputs a matched prompt fragment + lora_scale + strength. "
        "Style + room define the materials/furniture/lighting; level controls "
        "photographic detail. Vocabulary master copy: "
        "architectural_styles_glossary.txt. Wire prompt_fragment into Superside "
        "Combine Prompt's part2, strength into an inpaint node, lora_scale into "
        "a style/detail LoRA scale."
    )

    def get_settings(
        self,
        style="transitional",
        room="any",
        level="3 - medium (default)",
        trigger_word="",
        include_base=True,
    ):
        core = self._STYLE_CORE.get(style, self._STYLE_CORE["transitional"])
        level_detail, lora_scale, strength = self._LEVEL.get(
            level, self._LEVEL["3 - medium (default)"]
        )

        parts = []
        if trigger_word and trigger_word.strip():
            parts.append(trigger_word.strip())
        parts.append(core)

        room_vocab = self._STYLE_ROOM.get(style, {}).get(room, "")
        if room_vocab:
            parts.append(room_vocab)

        if include_base:
            parts.append(self._BASE)

        parts.append(level_detail)

        fragment = ", ".join(p for p in parts if p)
        return (fragment, float(lora_scale), float(strength))
