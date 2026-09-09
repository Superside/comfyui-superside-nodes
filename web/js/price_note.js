import { app } from "../../../scripts/app.js";

// Draws each Superside node's fal.ai price on the node itself.
//
// The price already reaches the browser inside the node's `description` (the
// Python side appends it at registration - see modules/fal_pricing.py), but a
// description is only a hover tooltip. This paints it just under the node so
// the cost of a graph is readable at a glance.
//
// Deliberately drawn on the canvas rather than added as a widget: ComfyUI maps
// `widgets_values` positionally, so adding a widget would shift every saved
// workflow's values.

const PRICE_PREFIX = "fal price";
const FONT = "10px system-ui, -apple-system, Segoe UI, sans-serif";
const COLOR = "#6faf6f";
const MIN_WIDTH = 260;
const MIN_ZOOM = 0.5;

function extractPrice(description) {
    if (!description) return null;
    const line = description
        .split("\n")
        .find((entry) => entry.trim().startsWith(PRICE_PREFIX));
    if (!line) return null;
    // "fal price (2026-09-09): $0.04 per image" -> "$0.04 per image"
    const marker = line.indexOf("): ");
    return marker === -1 ? line.trim() : line.slice(marker + 3).trim();
}

function fitToWidth(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let cut = text;
    while (cut.length > 4 && ctx.measureText(cut + "...").width > maxWidth) {
        cut = cut.slice(0, -1);
    }
    return cut.trimEnd() + "...";
}

app.registerExtension({
    name: "comfyui-superside-nodes.PriceNote",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const price = extractPrice(nodeData.description);
        if (!price) return;

        nodeType.prototype._supersidePrice = price;

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed || !this._supersidePrice) return;

            // 10px text is an unreadable smear once the canvas is zoomed out -
            // skip it rather than add noise to a wide graph view.
            const scale = app.canvas?.ds?.scale ?? 1;
            if (scale < MIN_ZOOM) return;

            ctx.save();
            ctx.font = FONT;
            ctx.fillStyle = COLOR;
            ctx.textAlign = "left";
            const maxWidth = Math.max(this.size[0], MIN_WIDTH);
            const label = fitToWidth(ctx, "fal " + this._supersidePrice, maxWidth);
            // Below the node body, so it never covers a widget.
            ctx.fillText(label, 2, this.size[1] + 13);
            ctx.restore();
        };
    },
});
