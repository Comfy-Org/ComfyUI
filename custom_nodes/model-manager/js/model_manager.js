import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DOWNLOADING = ["active", "waiting", "paused", "copying"];

function formatSize(bytes) {
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

function statusText(model) {
    switch (model.status) {
        case "installed": return "Installed";
        case "missing": return "";
        case "waiting": return "Queued";
        case "paused": return `Paused ${model.progress.toFixed(0)}%`;
        case "active": return `${model.progress.toFixed(0)}% · ${model.speed} · ${model.eta}`;
        case "copying": return `Copying ${model.progress.toFixed(0)}%`;
        case "error": return `Error: ${model.error}`;
        default: return model.status;
    }
}

function addModel(url, folder, filename) {
    return api.fetchApi("/model-manager/add", { method: "POST", body: JSON.stringify({ url, folder, filename }) });
}

// The Download button in the workflow's missing models list calls the desktop app's bridge when there is one, otherwise it
// downloads to the browser. Providing the bridge makes it download on the server instead, which also works remotely.
async function downloadMissingModel(url, name, directory) {
    app.extensionManager.sidebarTab.activeSidebarTabId = "model-manager";
    const resp = await addModel(url, directory, name);
    if (!resp.ok) app.extensionManager.toast.add({ severity: "error", summary: `Download of ${name} failed`, detail: await resp.text(), life: 10000 });
}

function render(el) {
    const selected = new Set();
    const collapsed = new Set();  // folder names, kept across the redraw every second
    let models = [];
    let wasDownloading = false;

    el.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:8px;padding:8px;height:100%;box-sizing:border-box">
            <input class="filter" placeholder="Filter models" style="padding:6px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px">
            <button class="install" style="padding:6px;cursor:pointer">Install selected</button>
            <div style="display:flex;gap:8px">
                <button class="upload" style="flex:1;padding:6px;cursor:pointer">Upload models.yaml</button>
                <button class="reset" style="padding:6px;cursor:pointer" title="Go back to the default model list">Reset</button>
            </div>
            <input class="file" type="file" accept=".yaml,.yml" hidden>
            <details>
                <summary style="cursor:pointer">Add model by URL or path</summary>
                <form class="add" style="display:flex;flex-direction:column;gap:8px;padding-top:8px">
                    <input name="url" required placeholder="Hugging Face or Civitai URL, or file path" style="padding:6px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px">
                    <select name="folder" required style="padding:6px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px"></select>
                    <input name="filename" placeholder="Filename (default: from URL or path)" style="padding:6px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px">
                    <button style="padding:6px;cursor:pointer">Add and install</button>
                </form>
            </details>
            <div class="error" style="color:var(--error-text)"></div>
            <div class="models" style="overflow-y:auto;flex:1"></div>
        </div>`;
    const filter = el.querySelector(".filter");
    const install = el.querySelector(".install");
    const upload = el.querySelector(".upload");
    const reset = el.querySelector(".reset");
    const file = el.querySelector(".file");
    const add = el.querySelector(".add");
    const error = el.querySelector(".error");
    const list = el.querySelector(".models");

    function row(model) {
        const label = document.createElement("label");
        label.style = "display:flex;gap:8px;align-items:center;padding:6px 0 6px 12px;border-bottom:1px solid var(--border-color)";

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
        detail.textContent = model.size ? `${model.filename} · ${formatSize(model.size)}` : model.filename;
        const status = document.createElement("div");
        status.style = "font-size:0.85em";
        status.textContent = statusText(model);
        info.append(name, detail, status);

        if (model.status === "active" || model.status === "paused" || model.status === "copying") {
            const bar = document.createElement("progress");
            bar.max = 100;
            bar.value = model.progress;
            bar.style = "width:100%";
            info.append(bar);
        }
        label.append(box, info);

        if (model.status === "active" || model.status === "waiting" || model.status === "paused") {
            const action = model.status === "paused" ? "resume" : "pause";
            const button = document.createElement("button");
            button.textContent = model.status === "paused" ? "Resume" : "Pause";
            button.style = "padding:4px 8px;cursor:pointer";
            button.onclick = async (event) => {
                event.preventDefault();
                const resp = await api.fetchApi(`/model-manager/${action}`, { method: "POST", body: JSON.stringify(model.name) });
                error.textContent = resp.ok ? "" : await resp.text();
                refresh();
            };
            label.append(button);
        }
        return label;
    }

    function group(folder, folderModels) {
        const details = document.createElement("details");
        details.open = !collapsed.has(folder);
        details.ontoggle = () => details.open ? collapsed.delete(folder) : collapsed.add(folder);
        const summary = document.createElement("summary");
        summary.style = "cursor:pointer;padding:6px 0;font-weight:bold;border-bottom:1px solid var(--border-color)";
        summary.textContent = `${folder} (${folderModels.length})`;
        details.append(summary, ...folderModels.map(row));
        return details;
    }

    function draw() {
        const query = filter.value.toLowerCase();
        const folders = Map.groupBy(models.filter((m) => `${m.name} ${m.folder}/${m.filename}`.toLowerCase().includes(query)), (m) => m.folder);
        list.replaceChildren(...[...folders].map(([folder, folderModels]) => group(folder, folderModels)));
    }

    async function refresh() {
        const resp = await api.fetchApi("/model-manager/models");
        if (!resp.ok) {
            error.textContent = `Failed to load models: ${resp.status} ${resp.statusText}`;
            return;
        }
        models = await resp.json();
        const downloading = models.some((m) => DOWNLOADING.includes(m.status));
        const finished = wasDownloading && !downloading;
        wasDownloading = downloading;
        draw();
        if (finished) {
            await app.refreshComboInNodes();
            app.refreshMissingModels({ reloadDefs: false });
        }
    }

    filter.oninput = draw;
    install.onclick = async () => {
        if (!selected.size) return;
        install.disabled = true;
        const resp = await api.fetchApi("/model-manager/install", { method: "POST", body: JSON.stringify([...selected]) });
        error.textContent = resp.ok ? "" : `Install failed: ${resp.status} ${resp.statusText}`;
        selected.clear();
        install.disabled = false;
        refresh();
    };

    async function setCatalog(options) {
        const resp = await api.fetchApi("/model-manager/catalog", options);
        error.textContent = resp.ok ? "" : await resp.text();
        selected.clear();
        refresh();
    }

    upload.onclick = () => file.click();
    file.onchange = async () => {
        const body = await file.files[0].text();
        file.value = "";
        setCatalog({ method: "POST", body });
    };
    reset.onclick = () => setCatalog({ method: "DELETE" });

    api.fetchApi("/experiment/models").then((resp) => resp.json()).then((folders) => {
        add.folder.replaceChildren(...folders.map((f) => new Option(f.name, f.name)));
    });
    add.onsubmit = async (event) => {
        event.preventDefault();
        const resp = await addModel(add.url.value, add.folder.value, add.filename.value.trim());
        error.textContent = resp.ok ? "" : await resp.text();
        if (resp.ok) {
            add.url.value = "";
            add.filename.value = "";
        }
        refresh();
    };

    refresh();
    const timer = setInterval(() => el.isConnected ? refresh() : clearInterval(timer), 1000);
}

app.registerExtension({
    name: "model-manager",
    setup() {
        window.__comfyDesktop2 ??= { downloadModel: downloadMissingModel };
        app.extensionManager.registerSidebarTab({
            id: "model-manager",
            icon: "pi pi-download",
            title: "Download Models",
            tooltip: "Download models or copy them from a path",
            type: "custom",
            render,
        });
    },
});
