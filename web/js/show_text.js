import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Adds a read-only text display widget to Superside nodes that return text output.
// Mirrors the pattern used by ComfyUI-Custom-Scripts ShowText node.

const TEXT_DISPLAY_NODES = [
    "SupersidePromptBoxNode",
    "SupersideTextPreviewNode",
    "SupersidePromptSplitterNode",
    "SupersideAnyLLMVisionNode",
    "SupersideAnyLLMTextNode",
    "SupersideGrokImagineImageQualityEditNode",
    "SupersideGrokImagineImageV2EditNode",
    "SupersideSeedreamV45EditNode",
    "SupersideSeedreamV5ProEditNode",
    "SupersideBriaBackgroundStandardizerNode",
    "SupersideBriaReplaceBackgroundNode",
];

app.registerExtension({
    name: "comfyui-superside-nodes.ShowText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TEXT_DISPLAY_NODES.includes(nodeData.name)) return;

        function populate(text) {
            // Remove previously added display widgets only
            if (this.widgets) {
                for (let i = this.widgets.length - 1; i >= 0; i--) {
                    if (this.widgets[i]._supersideDisplay) {
                        this.widgets[i].onRemove?.();
                        this.widgets.splice(i, 1);
                    }
                }
            }

            const values = Array.isArray(text) ? text : [text];
            for (const entry of values) {
                const items = Array.isArray(entry) ? entry : [entry];
                for (const item of items) {
                    const w = ComfyWidgets["STRING"](
                        this,
                        "output_" + (this.widgets?.length ?? 0),
                        ["STRING", { multiline: true }],
                        app
                    ).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = item;
                    w._supersideDisplay = true;
                    // Never serialise the display. ComfyUI maps widgets_values
                    // positionally, so a saved display value sits in the slot
                    // that the next widget added on the Python side will claim -
                    // and then a STRING lands where an INT is expected and the
                    // node refuses to run. (Verified in the frontend: `serialize`
                    // on the widget is honoured, `options.serialize` is not.)
                    w.serialize = false;
                }
            }

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) sz[0] = this.size[0];
                if (sz[1] < this.size[1]) sz[1] = this.size[1];
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // Show text when node finishes executing
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (message?.text) {
                populate.call(this, message.text);
            }
        };

        // The display is a view of the last run, not part of the document, so
        // there is deliberately nothing to restore on load. The previous
        // version re-displayed the LAST entry of widgets_values, which only
        // held the display text while the display happened to be the last
        // widget - once a real widget is appended on the Python side that
        // restore shows the new widget's value as if it were model output.
    },
});
