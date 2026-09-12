import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "sensenova-interleave-preview-styles";
const STYLE_CSS = `
.sn-interleave {
    min-height: 180px;
    padding: 10px;
    box-sizing: border-box;
    overflow: auto;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #1e1e1e);
    border: 1px solid var(--border-color, #333);
    border-radius: 6px;
    word-break: break-word;
}
.sn-interleave > * { margin: 0 0 10px 0; }
.sn-interleave-text { white-space: pre-wrap; }
.sn-interleave-think {
    padding: 6px 8px;
    border-left: 3px solid var(--node-selected-color, #6c757d);
    background: var(--comfy-menu-bg, #2a2a2a);
    color: var(--descrip-text, #aaa);
    font-style: italic;
    white-space: pre-wrap;
}
.sn-interleave-think summary {
    cursor: pointer;
    font-style: normal;
    font-weight: 600;
}
.sn-interleave-think > div { margin-top: 4px; }
.sn-interleave-image { text-align: center; }
.sn-interleave-image img {
    max-width: 100%;
    max-height: 480px;
    border-radius: 4px;
    border: 1px solid var(--border-color, #333);
}
.sn-interleave-placeholder {
    color: var(--descrip-text, #888);
    font-style: italic;
}
.sn-thinking-preview {
    min-height: 96px;
    padding: 10px;
    box-sizing: border-box;
    overflow: auto;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #1e1e1e);
    border: 1px solid var(--border-color, #333);
    border-radius: 6px;
    white-space: pre-wrap;
    word-break: break-word;
}
`;

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE_CSS;
    document.head.appendChild(style);
}

function buildImageUrl(part) {
    const params = new URLSearchParams({
        filename: part.filename || "",
        type: part.image_type || "temp",
        subfolder: part.subfolder || "",
        rand: Math.random().toString(36).slice(2),
    });
    return api?.apiURL ? api.apiURL(`/view?${params}`) : `/view?${params}`;
}

function renderPart(part) {
    if (part.type === "text") {
        const element = document.createElement("div");
        element.className = "sn-interleave-text";
        element.textContent = part.text || "";
        return element;
    }
    if (part.type === "think") {
        const element = document.createElement("details");
        element.className = "sn-interleave-think";
        const summary = document.createElement("summary");
        summary.textContent = "think";
        element.appendChild(summary);
        const body = document.createElement("div");
        body.textContent = part.text || "";
        element.appendChild(body);
        return element;
    }
    if (part.type === "image") {
        const element = document.createElement("div");
        element.className = "sn-interleave-image";
        if (part.missing || !part.filename) {
            const placeholder = document.createElement("span");
            placeholder.className = "sn-interleave-placeholder";
            placeholder.textContent = `[image:${part.index} missing]`;
            element.appendChild(placeholder);
        } else {
            const image = document.createElement("img");
            image.alt = `image ${part.index}`;
            image.src = buildImageUrl(part);
            element.appendChild(image);
        }
        return element;
    }
    return null;
}

function renderParts(container, parts) {
    container.replaceChildren();
    if (!parts?.length) {
        const empty = document.createElement("div");
        empty.className = "sn-interleave-placeholder";
        empty.textContent = "(no interleaved output)";
        container.appendChild(empty);
        return;
    }
    for (const part of parts) {
        const element = renderPart(part);
        if (element) container.appendChild(element);
    }
}

function previewText(message) {
    if (Array.isArray(message?.text)) return message.text.join("\n");
    if (typeof message?.text === "string") return message.text;
    return "";
}

function registerThinkingPreview(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = onNodeCreated?.apply(this, arguments);
        const container = document.createElement("div");
        container.className = "sn-thinking-preview sn-interleave-placeholder";
        container.textContent = "Thinking preview will appear here after execution.";
        this.addDOMWidget?.("preview", "thinking_preview", container, {
            serialize: false,
            hideOnZoom: false,
        });
        this._snThinkingContainer = container;
        return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (!this._snThinkingContainer) return;
        const text = previewText(message);
        this._snThinkingContainer.textContent = text || "(no thinking output)";
        this._snThinkingContainer.classList.toggle(
            "sn-interleave-placeholder",
            !text,
        );
        this.setDirtyCanvas?.(true, true);
    };
}

app.registerExtension({
    name: "sensenova.interleave_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "SenseNovaThinkingPreview") {
            ensureStyles();
            registerThinkingPreview(nodeType);
            return;
        }
        if (nodeData?.name !== "SenseNovaInterleavePreview") return;
        ensureStyles();

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const container = document.createElement("div");
            container.className = "sn-interleave";
            const hint = document.createElement("div");
            hint.className = "sn-interleave-placeholder";
            hint.textContent = "Interleave preview will appear here after execution.";
            container.appendChild(hint);
            this.addDOMWidget?.("preview", "interleave_preview", container, {
                serialize: false,
                hideOnZoom: false,
            });
            this._snInterleaveContainer = container;
            this.imgs = [];
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (!this._snInterleaveContainer) return;
            renderParts(
                this._snInterleaveContainer,
                Array.isArray(message?.parts) ? message.parts : [],
            );
            this.imgs = [];
            this.setDirtyCanvas?.(true, true);
        };
    },
});
