import logging

from . import fal_cost_ledger

logger = logging.getLogger(__name__)


class SupersideFalCostReportNode:
    """
    Fal Cost Report: what this workflow called on fal.ai and what it cost.

    Every Superside node routes its fal call through the same helper, so the
    report covers all of them with nothing to wire up per node - drop this node
    at the end of the graph and run it.

    ComfyUI does not guarantee that a node with no inputs runs last, so pass
    the final image (or any string) of your pipeline through `after_image` /
    `after_text` to force this node to run after the work is done.
    """

    SCOPES = ["this run", "session (since ComfyUI started)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scope": (cls.SCOPES, {"default": "this run"}),
            },
            "optional": {
                "after_image": ("IMAGE",),
                "after_text": ("STRING", {"forceInput": True}),
                "clear_after_report": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("report", "total_usd", "calls")
    FUNCTION = "report"
    OUTPUT_NODE = True
    CATEGORY = "Superside"
    DESCRIPTION = (
        "Adds up every fal.ai call made by Superside nodes and reports the cost. "
        "'this run' covers the calls made since this node last reported; "
        "'session' covers everything since ComfyUI started. Wire the last image "
        "of your pipeline into after_image so this node runs last."
    )

    # Per-node cursor into the ledger, so "this run" means "since I last
    # reported" even with several report nodes in one graph.
    _cursors = {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Inputs rarely change, but the ledger does - always re-run.
        return float("NaN")

    def report(
        self,
        scope,
        after_image=None,
        after_text=None,
        clear_after_report=False,
        unique_id=None,
    ):
        key = str(unique_id)
        if scope.startswith("session"):
            rows = fal_cost_ledger.entries()
            title = "Superside fal.ai cost - session total"
        else:
            cursor = self._cursors.get(key, 0)
            rows = fal_cost_ledger.entries(after_seq=cursor)
            title = "Superside fal.ai cost - this run"

        self._cursors[key] = fal_cost_ledger.last_seq()

        text = fal_cost_ledger.format_report(rows, title)
        summary = fal_cost_ledger.totals(rows)

        if clear_after_report:
            dropped = fal_cost_ledger.clear()
            self._cursors[key] = fal_cost_ledger.last_seq()
            text += "\n\nLedger cleared ({} entries dropped).".format(dropped)

        logger.info(
            "Fal cost report (%s): $%.4f over %d call(s)",
            scope, summary["usd"], summary["calls"],
        )

        return {
            "ui": {"text": [text]},
            "result": (text, float(summary["usd"]), int(summary["calls"])),
        }
