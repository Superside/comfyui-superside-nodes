import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Adds a read-only text display widget to Superside nodes that return text output.
// Mirrors the pattern used by ComfyUI-Custom-Scripts ShowText node.

const TEXT_DISPLAY_NODES = [
    "SupersidePromptBoxNode",
    "SupersidePromptSplitterNode",
    "SupersideAnyLLMVisionNode",
    "SupersideAnyLLMTextNode",
    "SupersideGrokImagineImageQualityEditNode",
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

        // Preserve widget values so text survives workflow save/load
        const VALUES = Symbol();
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            this[VALUES] = arguments[0]?.widgets_values;
            return configure?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            if (this[VALUES]?.length) {
                requestAnimationFrame(() => {
                    populate.call(this, this[VALUES].slice(-1));
                });
            }
        };
    },
});
