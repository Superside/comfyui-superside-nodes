"""A pool of interchangeable prompt fragments, and a template to drop one into.

Written for try-on generation, where one long instruction prompt stays fixed and
only a short fragment - the pose - changes per run. Keeping the pool in one node
means the ten poses live in one place instead of ten copies of the long prompt,
and the pick is driven by the seed, so which pose produced which image is
recorded in the workflow rather than lost to a hidden shuffle.
"""

import random
import re

BLOCK_SEPARATOR = "---"


def split_variants(raw):
    """Split the pool on lines that are exactly '---'.

    A block whose first line starts with '#' takes that line as its label; the
    label is not part of the emitted text. Blank blocks are dropped, so a
    trailing separator or a stray blank line is harmless.
    """
    blocks = re.split(r"(?m)^\s*" + re.escape(BLOCK_SEPARATOR) + r"\s*$", raw or "")
    out = []
    for i, block in enumerate(blocks):
        lines = block.strip("\n").split("\n")
        label = ""
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and lines[0].lstrip().startswith("#"):
            label = lines.pop(0).lstrip().lstrip("#").strip()
        text = "\n".join(lines).strip()
        if not text:
            continue
        out.append((label or "variant {}".format(len(out) + 1), text))
    return out


class SupersidePromptVariantsNode:
    """
    Superside Prompt Variants: hold a numbered pool of prompt fragments and
    emit one of them.

    Blocks are separated by a line containing only `---`, and a block may open
    with a `# label` line naming it. Three ways to pick:

      * `random` - seeded, so the same seed always gives the same block
      * `cycle`  - block (seed mod count), which walks the pool in order when
                   the seed widget is set to increment after each run
      * `fixed`  - the block at `index`, for testing one on its own

    Every mode is a pure function of the inputs: there is no counter hidden in
    the node, so re-running a workflow reproduces the prompt it produced before,
    and the `index` and `label` outputs record which block was used.
    """

    MODES = ["random", "cycle", "fixed"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variants": ("STRING", {
                    "default": (
                        "# 01 frontal, loose smile\n"
                        "Front-facing studio headshot, weight shifted subtly onto one side so the\n"
                        "shoulders sit at a natural loose diagonal, head straight to camera, soft\n"
                        "subtle smile, arms down at the sides out of frame. No hands visible.\n"
                        "---\n"
                        "# 02 three-quarter, temple detail\n"
                        "Body and shoulders rotated to a 3/4 angle away from the head, head\n"
                        "counter-rotated back toward the camera to reveal the temple detail of the\n"
                        "glasses, soft subtle smile, arms down at the sides out of frame. No hands\n"
                        "visible.\n"
                    ),
                    "multiline": True,
                    "tooltip": "Blocks separated by a line containing only ---. "
                               "A leading '# label' line names the block and is not emitted.",
                }),
                "mode": (cls.MODES, {
                    "default": "random",
                    "tooltip": "random and cycle both read the seed; fixed reads index.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Drives random and cycle. Set the widget to 'increment' "
                               "to walk the pool one block per run.",
                }),
                "index": ("INT", {
                    "default": 1, "min": 1, "max": 999,
                    "tooltip": "1-based block to emit in fixed mode. Out of range is clamped.",
                }),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("text", "label", "index", "count")
    FUNCTION = "pick"
    DESCRIPTION = (
        "Hold a pool of prompt fragments (blocks split by ---) and emit one, "
        "picked at random, cycled by seed, or fixed by index. Pairs with "
        "Superside Prompt Slots."
    )

    def pick(self, variants, mode, seed, index):
        blocks = split_variants(variants)
        if not blocks:
            raise ValueError(
                "Prompt Variants: no blocks found. Separate them with a line "
                "containing only '{}'.".format(BLOCK_SEPARATOR)
            )

        count = len(blocks)
        if mode == "fixed":
            chosen = min(max(int(index), 1), count) - 1
        elif mode == "cycle":
            chosen = int(seed) % count
        else:
            chosen = random.Random(int(seed)).randrange(count)

        label, text = blocks[chosen]
        return (text, label, chosen + 1, count)


class SupersidePromptSlotsNode:
    """
    Superside Prompt Slots: one long template with named holes in it.

    The template carries `{pose}`, `{spec}` and so on; each slot input fills the
    placeholder named by its key. Built so a 4,000-character instruction prompt
    stays in one editable place while the parts that change per run - a pose
    fragment, a live SKU description - arrive on wires.

    An unfilled slot collapses to nothing rather than leaving `{pose}` in the
    text for the model to read. The `info` output names every mismatch it found
    - a placeholder nothing fills, a slot whose key is not in the template -
    because a wire that silently does nothing is the failure worth catching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "default": "Your instruction prompt.\n\nPOSE: {pose}\n\nFRAME: {spec}\n",
                    "multiline": True,
                    "tooltip": "Placeholders are {key}, matching the key_N widgets below.",
                }),
                "key_1": ("STRING", {"default": "pose", "multiline": False}),
                "key_2": ("STRING", {"default": "spec", "multiline": False}),
                "key_3": ("STRING", {"default": "slot3", "multiline": False}),
                "key_4": ("STRING", {"default": "slot4", "multiline": False}),
            },
            "optional": {
                "slot_1": ("STRING", {"forceInput": True}),
                "slot_2": ("STRING", {"forceInput": True}),
                "slot_3": ("STRING", {"forceInput": True}),
                "slot_4": ("STRING", {"forceInput": True}),
            },
        }

    CATEGORY = "Superside"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "info")
    FUNCTION = "compose"
    DESCRIPTION = (
        "Fill {key} placeholders in a long template from connected string "
        "inputs. Reports placeholders nothing fills and slots the template "
        "never uses."
    )

    def compose(self, template, key_1, key_2, key_3, key_4,
                slot_1=None, slot_2=None, slot_3=None, slot_4=None):
        pairs = [(key_1, slot_1), (key_2, slot_2), (key_3, slot_3), (key_4, slot_4)]
        text = template or ""
        notes = []
        filled = set()

        for key, value in pairs:
            key = (key or "").strip()
            if not key:
                continue
            token = "{" + key + "}"
            present = token in text
            if value is None:
                if present:
                    # Drop it rather than hand the model a literal "{pose}" to
                    # read; the note is what tells you the wire is missing.
                    text = text.replace(token, "")
                    notes.append("{}: placeholder in the template, nothing connected - removed".format(token))
                continue
            if not present:
                notes.append("{}: connected, but the template has no such placeholder".format(token))
                continue
            text = text.replace(token, value)
            filled.add(key)

        for token in sorted(set(re.findall(r"\{([A-Za-z0-9_]+)\}", text))):
            notes.append("{{{}}}: left in the text - no slot uses that key".format(token))

        info = "filled: {}".format(", ".join(sorted(filled)) or "none")
        if notes:
            info += "\n" + "\n".join(notes)
        return (text, info)
