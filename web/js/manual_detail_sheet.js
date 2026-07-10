import { app } from "../../../scripts/app.js";

// Interactive box-selector widget for Superside Manual Detail Sheet.
// A row of numbered on/off buttons above the image turns each of the
// MAX_BOXES boxes on or off; active boxes are drawn on the image, where you
// drag one to move it and scroll over it to resize it. State is serialized
// as JSON into the hidden "boxes" STRING widget so the Python side can read
// it at run time.

const NODE_NAME = "SupersideManualDetailSheetNode";
const MAX_BOXES = 6;
const DEFAULT_ACTIVE_COUNT = 4;
const COLORS = ["#ff5555", "#55aaff", "#55ff88", "#ffaa33", "#cc88ff", "#ffee55"];
const DEFAULT_SIDE_FRAC = 0.2; // default square side, as a fraction of image height

// Force a box to be a true square in IMAGE PIXELS, keeping its center. The
// box is stored as x/y fractions of width/height independently, so a square
// in pixels needs a smaller width-fraction than height-fraction on a wide
// photo. `aspect` is imageWidth/imageHeight. `sideH` is the desired square
// side as a fraction of image height. The result is clamped to stay inside
// the image while remaining square (the side shrinks if it can't fit).
function squareBox(cx, cy, sideH, aspect, active) {
    let h = Math.min(Math.max(sideH, 0.02), 1);
    let w = h / aspect;
    if (w > 1) {
        w = 1;
        h = w * aspect;
    }
    const halfW = w / 2;
    const halfH = h / 2;
    const ccx = Math.min(Math.max(cx, halfW), 1 - halfW);
    const ccy = Math.min(Math.max(cy, halfH), 1 - halfH);
    return {
        x1: ccx - halfW,
        y1: ccy - halfH,
        x2: ccx + halfW,
        y2: ccy + halfH,
        active,
    };
}

// A box's square "side" expressed as a fraction of image height (its height
// fraction is the source of truth; width is always derived from it).
function boxSideH(b) {
    return b.y2 - b.y1;
}

function defaultBoxes(aspect) {
    const a = aspect || 1.5;
    const boxes = [];
    for (let i = 0; i < MAX_BOXES; i++) {
        const cx = 0.2 + (i % 3) * 0.3;
        const cy = 0.28 + Math.floor(i / 3) * 0.44;
        boxes.push(squareBox(cx, cy, DEFAULT_SIDE_FRAC, a, i < DEFAULT_ACTIVE_COUNT));
    }
    return boxes;
}

app.registerExtension({
    name: "Superside.ManualDetailSheet",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            const boxesWidget = node.widgets?.find((w) => w.name === "boxes");
            if (boxesWidget) {
                boxesWidget.hidden = true;
                boxesWidget.computeSize = () => [0, -4];
            }

            let boxes = defaultBoxes();
            if (boxesWidget?.value) {
                try {
                    const parsed = JSON.parse(boxesWidget.value);
                    if (Array.isArray(parsed?.boxes) && parsed.boxes.length === MAX_BOXES) {
                        boxes = parsed.boxes;
                    }
                } catch (e) {
                    // keep defaults
                }
            }

            function syncWidget() {
                if (boxesWidget) {
                    boxesWidget.value = JSON.stringify({ boxes });
                }
            }
            syncWidget();

            const wrapper = document.createElement("div");
            wrapper.style.cssText = "width:100%;display:flex;flex-direction:column;gap:4px;";

            // Row of external on/off buttons - one per box. Turning a button
            // on makes that box appear on the image below; turning it off
            // hides it. This keeps activation off the image itself so it never
            // competes with dragging/resizing.
            const toolbar = document.createElement("div");
            toolbar.style.cssText =
                "display:flex;flex-wrap:wrap;gap:4px;padding:2px;";
            const toggleButtons = [];
            for (let i = 0; i < MAX_BOXES; i++) {
                const btn = document.createElement("button");
                btn.textContent = String(i + 1);
                btn.style.cssText =
                    "flex:1;min-width:28px;height:24px;border-radius:4px;border:2px solid " +
                    COLORS[i] + ";background:transparent;color:#ccc;font-weight:bold;" +
                    "cursor:pointer;font-family:sans-serif;font-size:12px;";
                btn.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    boxes[i].active = !boxes[i].active;
                    updateButtons();
                    syncWidget();
                    render();
                });
                toolbar.appendChild(btn);
                toggleButtons.push(btn);
            }
            wrapper.appendChild(toolbar);

            function updateButtons() {
                toggleButtons.forEach((btn, i) => {
                    const on = boxes[i]?.active !== false;
                    btn.style.background = on ? COLORS[i] : "transparent";
                    btn.style.color = on ? "#000" : "#888";
                    btn.title = on ? `Detail ${i + 1}: ON (click to hide)` : `Detail ${i + 1}: OFF (click to show)`;
                });
            }

            const canvasBox = document.createElement("div");
            canvasBox.style.cssText = "width:100%;position:relative;background:#111;border-radius:6px;overflow:hidden;";

            const canvas = document.createElement("canvas");
            canvas.style.cssText = "width:100%;display:block;cursor:default;";
            canvasBox.appendChild(canvas);

            const hint = document.createElement("div");
            hint.textContent = "buttons = show/hide  ·  drag = move  ·  scroll = resize";
            hint.style.cssText =
                "position:absolute;bottom:2px;left:6px;font-size:9px;color:#ccc;" +
                "pointer-events:none;text-shadow:0 1px 2px #000;font-family:sans-serif;";
            canvasBox.appendChild(hint);
            wrapper.appendChild(canvasBox);

            const domWidget = node.addDOMWidget("box_preview", "BoxPreviewWidget", wrapper, {
                serialize: false,
                getMinHeight: () => 250,
            });

            const img = new Image();
            let imgLoaded = false;
            let lastSrc = null;

            // Image width/height ratio - drives square boxes. Falls back to
            // 1.5 until the real image loads.
            function aspect() {
                return imgLoaded && img.naturalHeight ? img.naturalWidth / img.naturalHeight : 1.5;
            }

            // Re-derive every box as a true pixel-square for the current
            // aspect ratio, preserving each box's center and side. Called
            // whenever the aspect changes (image load) or boxes are restored.
            function resquareAll() {
                boxes = boxes.map((b) =>
                    squareBox((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, boxSideH(b), aspect(), b.active !== false)
                );
            }

            function getUpstreamImageSrc() {
                const inputIndex = node.inputs?.findIndex((i) => i.name === "image");
                if (inputIndex == null || inputIndex === -1) return null;
                const input = node.inputs[inputIndex];
                if (!input || input.link == null) return null;
                const link = node.graph?.links?.[input.link];
                if (!link) return null;
                const srcNode = node.graph.getNodeById(link.origin_id);
                if (!srcNode) return null;
                if (srcNode.imgs && srcNode.imgs.length > 0) {
                    return srcNode.imgs[0].src;
                }
                return null;
            }

            function render() {
                // Base the drawing size on the canvas's ACTUAL on-screen width
                // (its CSS width is 100% of the widget), not on node.size - so
                // the coordinates we draw boxes at exactly match the
                // coordinates getBoundingClientRect() reports when hit-testing
                // clicks. Using two different widths here was why boxes
                // couldn't be grabbed/moved. Fall back to a node.size estimate
                // only before the canvas has been laid out (clientWidth 0).
                const measured = canvas.clientWidth || canvas.getBoundingClientRect().width;
                const displayWidth = Math.max(measured || (node.size?.[0] || 300) - 20, 100);
                const displayHeight = displayWidth / aspect();

                const dpr = window.devicePixelRatio || 1;
                canvas.width = displayWidth * dpr;
                canvas.height = displayHeight * dpr;
                canvas.style.height = displayHeight + "px";

                const ctx = canvas.getContext("2d");
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.clearRect(0, 0, displayWidth, displayHeight);

                if (imgLoaded) {
                    ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
                } else {
                    ctx.fillStyle = "#222";
                    ctx.fillRect(0, 0, displayWidth, displayHeight);
                    ctx.fillStyle = "#888";
                    ctx.font = "12px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Connect an image to preview", displayWidth / 2, displayHeight / 2);
                    ctx.textAlign = "left";
                }

                // Only active boxes are drawn on the image; inactive ones are
                // simply hidden (toggle them back on via the buttons above).
                boxes.forEach((b, i) => {
                    if (b.active === false) return;
                    const x = b.x1 * displayWidth;
                    const y = b.y1 * displayHeight;
                    const w = (b.x2 - b.x1) * displayWidth;
                    const h = (b.y2 - b.y1) * displayHeight;
                    const color = COLORS[i % COLORS.length];

                    ctx.save();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, w, h);
                    ctx.fillStyle = color;
                    ctx.globalAlpha = 0.12;
                    ctx.fillRect(x, y, w, h);
                    ctx.restore();

                    ctx.save();
                    ctx.fillStyle = "#000";
                    ctx.font = "bold 13px sans-serif";
                    ctx.fillText(String(i + 1), x + 6, y + 17);
                    ctx.fillStyle = color;
                    ctx.fillText(String(i + 1), x + 5, y + 16);
                    ctx.restore();
                });

                updateButtons();
            }
            node._sdsRender = render;
            node._sdsSetBoxes = (newBoxes) => {
                boxes = newBoxes;
                resquareAll();
                syncWidget();
                render();
            };

            // All hit-testing is done directly against the canvas's LIVE
            // on-screen rect and the boxes' 0-1 fractional coords, so it never
            // depends on whatever width render() happened to draw at. Returns
            // mouse position as fractions of the displayed canvas.
            function eventFracs(e) {
                const rect = canvas.getBoundingClientRect();
                return {
                    fx: (e.clientX - rect.left) / rect.width,
                    fy: (e.clientY - rect.top) / rect.height,
                    rect,
                };
            }

            function hitTest(fx, fy) {
                // Only active (drawn) boxes are grabbable. Topmost/last wins.
                for (let i = boxes.length - 1; i >= 0; i--) {
                    const b = boxes[i];
                    if (b.active === false) continue;
                    if (fx >= b.x1 && fx <= b.x2 && fy >= b.y1 && fy <= b.y2) {
                        return { index: i };
                    }
                }
                return null;
            }

            let drag = null;

            function onMouseDown(e) {
                const { fx, fy } = eventFracs(e);
                const hit = hitTest(fx, fy);
                if (!hit) return;
                e.preventDefault();
                e.stopPropagation();

                drag = {
                    index: hit.index,
                    startFx: fx,
                    startFy: fy,
                    origBox: { ...boxes[hit.index] },
                };
            }

            function onMouseMove(e) {
                if (!drag) return;
                const { fx, fy } = eventFracs(e);
                const dxFrac = fx - drag.startFx;
                const dyFrac = fy - drag.startFy;

                const w = drag.origBox.x2 - drag.origBox.x1;
                const h = drag.origBox.y2 - drag.origBox.y1;
                const nx1 = Math.min(Math.max(drag.origBox.x1 + dxFrac, 0), 1 - w);
                const ny1 = Math.min(Math.max(drag.origBox.y1 + dyFrac, 0), 1 - h);

                const b = boxes[drag.index];
                b.x1 = nx1;
                b.y1 = ny1;
                b.x2 = nx1 + w;
                b.y2 = ny1 + h;
                render();
            }

            function onMouseUp() {
                if (drag) {
                    syncWidget();
                    drag = null;
                }
            }

            function onWheel(e) {
                const { fx, fy } = eventFracs(e);
                const hit = hitTest(fx, fy);
                if (!hit) return;
                e.preventDefault();
                e.stopPropagation();

                const b = boxes[hit.index];
                const cx = (b.x1 + b.x2) / 2;
                const cy = (b.y1 + b.y2) / 2;
                const factor = e.deltaY > 0 ? 1 / 1.06 : 1.06;
                // Scale the square's side (tracked as a height fraction) and
                // rebuild it as a true pixel-square - so resizing never turns
                // a square into a rectangle.
                const newSide = boxSideH(b) * factor;
                boxes[hit.index] = squareBox(cx, cy, newSide, aspect(), b.active);
                syncWidget();
                render();
            }

            canvas.addEventListener("mousedown", onMouseDown);
            window.addEventListener("mousemove", onMouseMove);
            window.addEventListener("mouseup", onMouseUp);
            canvas.addEventListener("wheel", onWheel, { passive: false });

            // LoadImage-style upstream nodes set `.imgs` once the user picks
            // a file, but there's no single reliable event for that across
            // every possible upstream node type - a light poll is the
            // simplest robust way to notice a new/changed source image.
            const pollInterval = setInterval(() => {
                const src = getUpstreamImageSrc();
                if (src && src !== lastSrc) {
                    lastSrc = src;
                    imgLoaded = false;
                    img.onload = () => {
                        imgLoaded = true;
                        // Now that the true image aspect ratio is known,
                        // rebuild the boxes as real pixel-squares.
                        resquareAll();
                        syncWidget();
                        render();
                    };
                    img.src = src;
                } else if (!src && lastSrc) {
                    lastSrc = null;
                    imgLoaded = false;
                    render();
                }
            }, 500);

            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                clearInterval(pollInterval);
                window.removeEventListener("mousemove", onMouseMove);
                window.removeEventListener("mouseup", onMouseUp);
                return onRemoved?.apply(this, arguments);
            };

            render();

            requestAnimationFrame(() => {
                const sz = node.computeSize();
                if (sz[1] > node.size[1]) {
                    node.setSize([node.size[0], sz[1]]);
                }
            });

            return result;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            const result = onResize?.apply(this, arguments);
            this._sdsRender?.();
            return result;
        };

        // ComfyUI applies a saved workflow's widgets_values (via configure())
        // AFTER onNodeCreated has already run and populated the "boxes"
        // widget with defaults - so on every real workflow load, that
        // default value gets overwritten by whatever was actually saved.
        // Re-parse the widget's value here, once configure() has settled it,
        // so the on-canvas boxes reflect what was actually persisted instead
        // of staying stuck on the defaults computed at node-creation time.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            const boxesWidget = this.widgets?.find((w) => w.name === "boxes");
            let restored = null;
            if (boxesWidget?.value) {
                try {
                    const parsed = JSON.parse(boxesWidget.value);
                    if (Array.isArray(parsed?.boxes) && parsed.boxes.length === MAX_BOXES) {
                        restored = parsed.boxes;
                    }
                } catch (e) {
                    // fall through to defaults
                }
            }
            this._sdsSetBoxes?.(restored || defaultBoxes());
            return result;
        };
    },
});
