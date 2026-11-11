import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Adds a browse button under the folder text input of the product nodes.
// The dialog opens on the machine that runs ComfyUI (see modules/folder_picker.py).

// mode "folder": the folder dialog (choose where a new product is saved; it lists folders only).
// mode "file": the ordinary Open dialog, with files and thumbnails; the folder of the picked file is used.
const FOLDER_WIDGETS = {
    SupersideProductResearchNode: { widget: "output_dir", mode: "folder", label: "Choose save folder..." },
    SupersideProductSheetAutoNode: { widget: "output_dir", mode: "folder", label: "Choose save folder..." },
    SupersideProductSheetNode: { widget: "product_dir", mode: "file", label: "Browse product folder..." },
    SupersideProductVideoReferenceNode: { widget: "product_dir", mode: "file", label: "Browse product folder..." },
};

app.registerExtension({
    name: "comfyui-superside-nodes.FolderPicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = FOLDER_WIDGETS[nodeData.name];
        if (!config) return;
        const widgetName = config.widget;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const button = node.addWidget("button", config.label, null, async () => {
                const target = node.widgets?.find((w) => w.name === widgetName);
                if (!target) return;
                button.name = "Choosing...";
                try {
                    const response = await api.fetchApi("/superside/pick_folder", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ mode: config.mode }),
                    });
                    const data = await response.json();
                    if (data.path) {
                        target.value = data.path;
                        target.callback?.(data.path);
                    } else if (data.error) {
                        alert("Could not open the folder dialog: " + data.error);
                    }
                } catch (e) {
                    alert("Could not open the folder dialog: " + e);
                } finally {
                    button.name = config.label;
                    node.setDirtyCanvas(true, true);
                }
            });
            button.serialize = false;
            return result;
        };
    },
});
