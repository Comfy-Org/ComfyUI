// "Model Compatibility" sidebar tab: will the installed models run on this machine?
// Each model is judged from its size on disk against the detected GPU and RAM
// (/model_compat/local). Everything stays local; nothing goes over the network.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE = `
.mc { --bg:#0f1115; --panel:#161a21; --raised:#1d222b; --line:#2a303b; --text:#e6e9ef; --muted:#8d96a5;
  --faint:#5c6573; --accent:#d97757; --ok:#3fb950; --warn:#e3b341; --err:#f85149;
  --mono:"JetBrains Mono","Fira Code",ui-monospace,Menlo,Consolas,monospace;
  display:flex; flex-direction:column; height:100%; background:var(--bg); color:var(--text);
  font:13px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
.mc button { font:inherit; cursor:pointer; border:none; }
.mc button:disabled { opacity:.5; cursor:default; }
.mc button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.mc-head { padding:12px 12px 10px; border-bottom:1px solid var(--line); }
.mc-top { display:flex; align-items:center; margin-bottom:10px; }
.mc-title { font-weight:600; font-size:14px; }
.mc-refresh { margin-left:auto; display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:8px;
  background:var(--raised); color:var(--text); font-weight:500; }
.mc-refresh:hover:not(:disabled) { background:var(--line); }
.mc-hw { display:flex; gap:10px; align-items:center; padding:10px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
.mc-hw > .pi { font-size:20px; color:var(--accent); }
.mc-hw-name { font-weight:600; font-size:12.5px; }
.mc-chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:4px; }
.mc-chip { font-size:11px; padding:1px 7px; border-radius:999px; background:var(--raised); color:var(--muted); white-space:nowrap; }
.mc-chip.yes { color:#7ee787; } .mc-chip.no { color:var(--faint); text-decoration:line-through; }
.mc-body { flex:1; overflow-y:auto; padding:12px; }
.mc-summary { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }
.mc-group { font-size:11px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--faint); margin:14px 0 6px; }
.mc-group:first-of-type { margin-top:0; }
.mc-card { padding:9px 10px; margin-bottom:6px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
.mc-row { display:flex; align-items:center; gap:8px; }
.mc-name { flex:1; min-width:0; font:12px var(--mono); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mc-badge { display:inline-flex; align-items:center; gap:4px; font-size:11px; font-weight:600; padding:2px 8px; border-radius:999px; white-space:nowrap; }
.mc-badge.fits { color:#7ee787; background:rgba(63,185,80,.14); }
.mc-badge.offload { color:#f2cc60; background:rgba(227,179,65,.14); }
.mc-badge.no { color:#ffa198; background:rgba(248,81,73,.14); }
.mc-meta { display:flex; align-items:center; gap:6px; margin-top:6px; font-size:11.5px; color:var(--muted); }
.mc-tag { font:10.5px var(--mono); padding:0 5px; border-radius:4px; background:var(--raised); color:var(--text); }
.mc-bar { position:relative; height:6px; margin-top:7px; border-radius:3px; background:var(--raised); overflow:hidden; }
.mc-bar i { position:absolute; inset:0 auto 0 0; border-radius:3px; }
.mc-bar i.fits { background:var(--ok); } .mc-bar i.offload { background:var(--warn); } .mc-bar i.no { background:var(--err); }
.mc-notes { margin:6px 0 0; padding-left:16px; font-size:11.5px; color:var(--muted); }
.mc-empty { display:flex; flex-direction:column; align-items:center; gap:8px; text-align:center; color:var(--muted); padding:30px 12px; }
.mc-empty .pi { font-size:26px; color:var(--faint); }
.mc-error { color:#ffa198; background:rgba(248,81,73,.08); border:1px solid rgba(248,81,73,.3); border-radius:10px; padding:10px; }
.mc-spin { width:16px; height:16px; border:2px solid var(--line); border-top-color:var(--accent); border-radius:50%; animation:mc-spin .8s linear infinite; }
.mc-legend { padding:10px 12px; border-top:1px solid var(--line); font-size:11px; color:var(--faint); display:flex; flex-wrap:wrap; gap:4px 12px; }
.mc-legend b { font-weight:600; }
@keyframes mc-spin { to { transform:rotate(360deg); } }
`;

const LEVELS = {
  fits: { icon: "pi-check-circle", text: "Fits on GPU" },
  offload: { icon: "pi-exclamation-triangle", text: "Needs offload" },
  no: { icon: "pi-times-circle", text: "Too large" },
};

function esc(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function gb(bytes) {
  const value = bytes / 1024 ** 3;
  return value >= 0.1 ? `${value.toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

function badge(level) {
  return `<span class="mc-badge ${level}"><i class="pi ${LEVELS[level].icon}"></i>${LEVELS[level].text}</span>`;
}

function renderHardware(hw) {
  const chip = (ok, label) => `<span class="mc-chip ${ok ? "yes" : "no"}">${label}</span>`;
  const gpu = hw.type === "cuda";
  return `<div class="mc-hw"><i class="pi pi-microchip"></i><div>
    <div class="mc-hw-name">${esc(hw.name)}</div>
    <div class="mc-chips">
      ${gpu ? `<span class="mc-chip">${gb(hw.vram_total)} VRAM</span>` : `<span class="mc-chip">No GPU</span>`}
      <span class="mc-chip">${gb(hw.ram_total)} RAM</span>
      ${gpu ? chip(hw.supports.bf16, "bf16") + chip(hw.supports.fp8, "fp8") + chip(hw.supports.fp4, "fp4") : ""}
    </div></div></div>`;
}

function renderCard(file, hw) {
  const isGpu = hw.type === "cuda";
  const share = isGpu ? Math.min(100, (file.size / hw.vram_total) * 100) : 0;
  const notes = file.notes.length ? `<ul class="mc-notes">${file.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : "";
  return `<div class="mc-card">
    <div class="mc-row"><span class="mc-name" title="${esc(file.path)}">${esc(file.name)}</span>${badge(file.level)}</div>
    <div class="mc-meta">
      <span>${gb(file.size)}</span>
      ${file.precision ? `<span class="mc-tag">${esc(file.precision)}</span>` : ""}
      <span style="margin-left:auto">${isGpu ? `${Math.round((file.size / hw.vram_total) * 100)}% of VRAM` : ""}</span>
    </div>
    ${isGpu ? `<div class="mc-bar"><i class="${file.level}" style="width:${share}%"></i></div>` : ""}
    ${notes}
  </div>`;
}

function renderFiles(files, hw) {
  if (!files.length) return `<div class="mc-empty"><i class="pi pi-inbox"></i>No models found in the models folders.</div>`;
  const count = (level) => files.filter((f) => f.level === level).length;
  let html = `<div class="mc-summary">${["fits", "offload", "no"]
    .filter(count)
    .map((level) => `<span class="mc-badge ${level}">${count(level)} · ${LEVELS[level].text}</span>`)
    .join("")}</div>`;
  let folder = null;
  for (const file of files) {
    if (file.folder !== folder) {
      folder = file.folder;
      html += `<div class="mc-group">${esc(folder.replace(/_/g, " "))}</div>`;
    }
    html += renderCard(file, hw);
  }
  return html;
}

function createPanel(el) {
  if (!document.getElementById("mc-style")) {
    const style = document.createElement("style");
    style.id = "mc-style";
    document.head.append(style);
  }
  document.getElementById("mc-style").textContent = STYLE;

  el.innerHTML = `
    <div class="mc">
      <div class="mc-head">
        <div class="mc-top"><span class="mc-title">Model Compatibility</span>
          <button class="mc-refresh" title="Scan the models folders again"><i class="pi pi-refresh"></i>Refresh</button></div>
        <div class="mc-hwbox"><div class="mc-hw"><div class="mc-spin"></div><span>Reading hardware…</span></div></div>
      </div>
      <div class="mc-body"></div>
      <div class="mc-legend">
        <span><b style="color:#7ee787">Fits</b>: whole on the GPU, full speed</span>
        <span><b style="color:#f2cc60">Offload</b>: partly in RAM, slower</span>
        <span><b style="color:#ffa198">Too large</b>: will not run</span>
        <span>Assumes ~1.5 GB headroom for image generation; video models need more.</span>
      </div>
    </div>`;
  const hwBox = el.querySelector(".mc-hwbox");
  const body = el.querySelector(".mc-body");
  const refresh = el.querySelector(".mc-refresh");

  async function scan() {
    refresh.disabled = true;
    body.innerHTML = `<div class="mc-empty"><div class="mc-spin"></div>Scanning installed models…</div>`;
    try {
      const response = await api.fetchApi("/model_compat/local");
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error ?? `Server error (${response.status})`);
      hwBox.innerHTML = renderHardware(data.hardware);
      body.innerHTML = renderFiles(data.files, data.hardware);
    } catch (error) {
      body.innerHTML = `<div class="mc-error">${esc(error.message)}</div>`;
    } finally {
      refresh.disabled = false;
    }
  }

  refresh.addEventListener("click", scan);
  scan();
}

app.registerExtension({
  name: "Comfy.ModelCompat",
  setup() {
    app.extensionManager.registerSidebarTab({
      id: "model-compat",
      icon: "pi pi-gauge",
      title: "Model Compatibility",
      tooltip: "Which installed models run on this machine?",
      type: "custom",
      render: createPanel,
    });
  },
});
