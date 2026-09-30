import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DOWNLOADING = ["active", "waiting"];

function statusText(model) {
    switch (model.status) {
        case "installed": return "Installed";
        case "missing": return "";
        case "waiting": return "Queued";
        case "active": return `${model.progress.toFixed(0)}% · ${model.speed} · ${model.eta}`;
        case "error": return `Error: ${model.error}`;
        default: return model.status;
    }
}

function render(el) {
    const selected = new Set();
    let models = [];
    let wasDownloading = false;

    el.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:8px;padding:8px;height:100%;box-sizing:border-box">
            <input class="filter" placeholder="Filter models" style="padding:6px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px">
            <button class="install" style="padding:6px;cursor:pointer">Install selected</button>
            <div class="error" style="color:var(--error-text)"></div>
            <div class="models" style="overflow-y:auto;flex:1"></div>
        </div>`;
    const filter = el.querySelector(".filter");
    const install = el.querySelector(".install");
    const error = el.querySelector(".error");
    const list = el.querySelector(".models");

    function row(model) {
        const label = document.createElement("label");
        label.style = "display:flex;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid var(--border-color)";

        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = selected.has(model.name);
        box.disabled = model.status !== "missing" && model.status !== "error";
        box.onchange = () => box.checked ? selected.add(model.name) : selected.delete(model.name);

        const info = document.createElement("div");
        info.style = "flex:1;min-width:0";
        const name = document.createElement("div");
        name.textContent = model.name;
        const detail = document.createElement("div");
        detail.style = "font-size:0.85em;opacity:0.7;overflow-wrap:anywhere";
        detail.textContent = `${model.folder}/${model.filename}`;
        const status = document.createElement("div");
        status.style = "font-size:0.85em";
        status.textContent = statusText(model);
        info.append(name, detail, status);

        if (model.status === "active") {
            const bar = document.createElement("progress");
            bar.max = 100;
            bar.value = model.progress;
            bar.style = "width:100%";
            info.append(bar);
        }
        label.append(box, info);
        return label;
    }

    function draw() {
        const query = filter.value.toLowerCase();
        list.replaceChildren(...models.filter((m) => `${m.name} ${m.folder}/${m.filename}`.toLowerCase().includes(query)).map(row));
    }

    async function refresh() {
        const resp = await api.fetchApi("/colab_models");
        if (!resp.ok) {
            error.textContent = `Failed to load models: ${resp.status} ${resp.statusText}`;
            return;
        }
        models = await resp.json();
        const downloading = models.some((m) => DOWNLOADING.includes(m.status));
        if (wasDownloading && !downloading) app.refreshComboInNodes();
        wasDownloading = downloading;
        draw();
    }

    filter.oninput = draw;
    install.onclick = async () => {
        if (!selected.size) return;
        install.disabled = true;
        const resp = await api.fetchApi("/colab_models/install", { method: "POST", body: JSON.stringify([...selected]) });
        error.textContent = resp.ok ? "" : `Install failed: ${resp.status} ${resp.statusText}`;
        selected.clear();
        install.disabled = false;
        refresh();
    };

    refresh();
    const timer = setInterval(() => el.isConnected ? refresh() : clearInterval(timer), 1000);
}

app.registerExtension({
    name: "colab.ModelManager",
    setup() {
        app.extensionManager.registerSidebarTab({
            id: "colab-models",
            icon: "pi pi-download",
            title: "Models",
            tooltip: "Install models",
            type: "custom",
            render,
        });
    },
});
