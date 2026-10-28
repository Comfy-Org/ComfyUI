import { app } from "../../scripts/app.js"
import { api } from "../../scripts/api.js"

// Routes served by the remote in remote mode: execution, node definitions, models, files, system
// information and logs. Workflows, settings, extensions and the rest stay local.
const REMOTE_ROUTES = ["/prompt", "/queue", "/history", "/interrupt", "/jobs", "/free", "/view", "/view_metadata", "/upload", "/object_info", "/models", "/experiment/models", "/embeddings", "/system_stats"]
// ComfyUI-Manager on the remote. Its HTTP client fixes "/api/v2/" as base URL when created, so these
// requests are redirected when sent instead of through api.apiURL.
const MANAGER_URL = /^\/api(\/v2\/(?:manager|customnode|externalmodel|snapshot)(?:[/?].*)?)$/
const REMOTE_INTERNAL_ROUTES = ["/logs", "/folder_paths", "/files"]
const REMEMBER_SETTING = "RemoteExecution.RememberMode"

const state = {
  gateway: null,
  gatewayError: null,
  // Remote mode never falls back to local execution: while the remote is unavailable its requests fail.
  mode: "local",
  url: storageGet("RemoteExecution.url", ""),
  currentTargetId: null,
  proxyTargetId: null,
  // Which server the loaded node definitions came from.
  defsFrom: "local",
  targets: new Map(),
  // Model downloads running on the remote, and its model files; both from remote_companion.py.
  downloads: new Map(),
  // Finished downloads that started before this time (seconds) are hidden by "Clear finished".
  clearedBefore: Number(storageGet("RemoteExecution.clearedBefore", "0")),
  files: null,
  filesError: null,
  // File key ("directory/name") whose Delete is awaiting confirmation.
  confirming: null,
  // Download links from the bundled templates, by file name, and URLs typed for missing models.
  links: null,
  urlDrafts: {},
  watchingMissing: false,
  missingKey: null,
  // Whether Hugging Face / Civitai keys are saved in the gateway; the keys themselves stay there.
  keyStatus: null,
  tab: "connection",
  busy: null,
  error: null,
  localVersion: null,
}
let mounted = null

function storageGet(key, fallback) {
  try {
    return localStorage.getItem(key) ?? fallback
  } catch {
    return fallback
  }
}

function storageSet(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch {}
}

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag)
  for (const [k, v] of Object.entries(props)) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v)
    else if (v !== undefined && v !== null && v !== false) el.setAttribute(k, v === true ? "" : v)
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : String(c))
  return el
}

function toast(severity, summary, detail) {
  app.extensionManager?.toast?.add({ severity, summary, detail, life: severity === "error" ? 8000 : 3000 })
}

async function loadGateway() {
  const resp = await api.fetchApi("/remote_execution/session")
  const data = await resp.json()
  if (!resp.ok) throw new Error(data.error)
  // Same hostname as the page keeps the gateway same-site, so its session cookie reaches the native socket.
  const hostname = ["127.0.0.1", "localhost"].includes(location.hostname) ? location.hostname : "127.0.0.1"
  state.gateway = { base: `http://${hostname}:${data.port}`, host: `${hostname}:${data.port}`, token: data.token }
}

async function gw(path, body, retried = false) {
  if (!state.gateway) await loadGateway()
  let resp
  try {
    resp = await fetch(state.gateway.base + path, {
      method: body === undefined ? "GET" : "POST",
      credentials: "include",
      headers: { "X-Gateway-Token": state.gateway.token, ...(body !== undefined && { "Content-Type": "application/json" }) },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new Error(`The gateway at ${state.gateway.base} is not reachable.`)
  }
  if (resp.status === 401 && !retried) {
    // The gateway restarted with a new session token.
    state.gateway = null
    return gw(path, body, true)
  }
  const data = await resp.json()
  if (!resp.ok) throw new Error(data.error ?? `Gateway returned HTTP ${resp.status}`)
  return data
}

function isRemote(routes, route) {
  const path = route.replace(/^\/api(?=\/)/, "").split("?")[0]
  return state.mode === "remote" && routes.some((r) => path === r || path.startsWith(r + "/"))
}

function remoteURL(path) {
  if (!state.gateway) throw new Error("Remote mode is on but the gateway is not running.")
  return `${state.gateway.base}/remote${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(state.gateway.token)}`
}

function currentTarget() {
  return state.targets.get(state.currentTargetId)
}

// A frontend Pinia store by id, when it has been created.
function store(id) {
  return document.getElementById("vue-app")?.__vue_app__?.config.globalProperties.$pinia?._s.get(id)
}

async function refreshDefs() {
  // Reloads node definitions and re-runs the missing model/media checks against the active server.
  await app.extensionManager.command.execute("Comfy.RefreshNodeDefinitions")
  // The Manager UI is enabled from the server's launch arguments in system stats, which the
  // frontend otherwise loads once at startup.
  await store("systemStats")?.refetchSystemStats()
}

// The Issues panel's missing models: { nodeId, nodeType, widgetName, name, directory, isMissing }.
function missingModels() {
  return (store("missingModel")?.missingModelCandidates ?? []).filter((c) => c.isMissing === true)
}

// Execution ids are "outer:inner" paths for nodes inside subgraphs.
function nodeByExecutionId(id) {
  let graph = app.rootGraph
  let node = null
  for (const part of String(id).split(":")) {
    node = graph?.getNodeById(part)
    graph = node?.subgraph
  }
  return node
}

async function loadLinks() {
  state.links ??= new Map((await gw("/model_links")).map((l) => [l.name, l]))
}

// Writes known download links into the nodes so the Issues panel shows its own Download button,
// which saves to the remote in remote mode. The links stay in the workflow for next time.
async function addKnownLinks() {
  if (state.mode !== "remote") return
  await loadLinks()
  let added = false
  for (const c of missingModels()) {
    const link = state.links.get(c.name)
    const node = nodeByExecutionId(c.nodeId)
    if (!link || !node || node.properties.models?.some((m) => m.name === c.name)) continue
    node.properties.models = [...(node.properties.models ?? []), { name: link.name, url: link.url, directory: link.directory }]
    added = true
  }
  if (added) await app.refreshMissingModels({ reloadDefs: false, silent: true })
}

function watchMissingModels() {
  const missing = store("missingModel")
  if (!missing || state.watchingMissing) return
  state.watchingMissing = true
  missing.$subscribe(() => {
    addKnownLinks().catch((e) => (state.error = `Model links: ${e.message}`))
    update()
  })
}

// Precision and packaging tags that distinguish variants of the same model file.
const VARIANT_TAGS = /^(fp8|fp16|fp32|bf16|int8|int4|nvfp4|mxfp8|fp4|scaled|e4m3fn|e5m2|convrot|fast|q[2-8](_k(_[sml])?)?|q[2-8]_[01])$/

function baseName(name) {
  return name
    .toLowerCase()
    .replace(/\.[a-z0-9]+$/, "")
    .split(/[_\-.\s]+/)
    .filter((t) => t && !VARIANT_TAGS.test(t))
    .join("_")
}

// Another variant (precision, scaling) of the same model.
function similar(a, b) {
  return baseName(a) === baseName(b)
}

function useModel(candidate, value) {
  const widget = nodeByExecutionId(candidate.nodeId)?.widgets?.find((w) => w.name === candidate.widgetName)
  if (!widget) return toast("warn", "Node not found", `#${candidate.nodeId}`)
  widget.value = value
  widget.callback?.(value)
  app.canvas.setDirty(true, true)
  app.refreshMissingModels({ reloadDefs: false, silent: true })
}

// Points remote mode at the connected server once it is ready: after switching, reloading or reconnecting.
async function syncRemote() {
  const target = currentTarget()
  if (state.mode !== "remote" || target?.status !== "ready") return
  try {
    if (state.proxyTargetId !== target.id) {
      state.proxyTargetId = target.id
      await gw("/proxy", { target_id: target.id })
    }
    if (state.defsFrom !== target.id) {
      state.defsFrom = target.id
      await refreshDefs()
    }
    watchMissingModels()
    await addKnownLinks()
  } catch (e) {
    state.error = `Remote mode: ${e.message}`
    update()
  }
}

// The missing-model Download button hands (url, name, directory) to a desktop host when one is
// present; in remote mode this extension is that host and the remote saves the file.
let desktopHost
function useRemoteDownloads() {
  if (state.mode === "remote") {
    window.__comfyDesktop2 = { downloadModel: downloadToRemote }
    pollDownloads()
  } else if (desktopHost === undefined) delete window.__comfyDesktop2
  else window.__comfyDesktop2 = desktopHost
}

// Calls a remote_companion.py route on the remote.
async function companion(path, body) {
  const resp = await fetch(remoteURL(path), body === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  const data = await resp.json().catch(() => ({}))
  if (resp.status === 404 && !data.error) throw new Error("The remote has no remote_companion.py: install it in its custom_nodes (see the README).")
  if (!resp.ok) throw new Error(data.error ?? `HTTP ${resp.status}`)
  return data
}

function formatSize(bytes) {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

async function loadFiles() {
  if (state.mode !== "remote" || currentTarget()?.status !== "ready") return
  try {
    state.files = await companion("/remote_execution/models/files")
    state.filesError = null
  } catch (e) {
    state.filesError = e.message
  }
  update()
}

function deleteFile(file) {
  return run("Delete", async () => {
    const result = await companion("/remote_execution/models/files/delete", { directory: file.directory, name: file.name })
    state.confirming = null
    // A loaded model keeps its deleted file's disk space until the remote unloads it.
    await fetch(remoteURL("/api/free"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ unload_models: true, free_memory: true }) })
    toast("success", "Deleted on the remote", `${result.deleted}: ${formatSize(result.freed)} freed once loaded models are unloaded`)
    await loadFiles()
    // The deleted file leaves the dropdowns.
    await refreshDefs()
  })
}

function showTab(tab) {
  state.tab = tab
  state.confirming = null
  update()
  if (tab === "files") {
    loadFiles()
    gw("/keys").then((status) => ((state.keyStatus = status), update()))
  }
}

function saveKey(name, input) {
  return run("Save key", async () => {
    state.keyStatus = await gw("/keys", { [name]: input.value })
    input.value = ""
  })
}

async function downloadToRemote(url, name, directory) {
  let data
  try {
    data = await companion("/remote_execution/models/download", { url, name, directory })
  } catch (e) {
    return toast("error", `Download ${name}`, e.message)
  }
  toast("info", "Downloading on the remote", `${directory}/${name}`)
  state.downloads.set(data.id, data)
  update()
  pollDownloads()
}

let polling = false
async function pollDownloads() {
  if (polling) return
  polling = true
  try {
    while (state.mode === "remote") {
      const resp = await fetch(remoteURL("/remote_execution/models/downloads"))
      if (!resp.ok) break
      const finished = []
      for (const d of await resp.json()) {
        if (state.downloads.get(d.id)?.status === "downloading" && d.status !== "downloading") finished.push(d)
        state.downloads.set(d.id, d)
      }
      update()
      for (const d of finished) toast(d.status === "completed" ? "success" : "error", `Download ${d.status}`, d.status === "completed" ? `${d.directory}/${d.name}` : `${d.name}: ${d.error ?? d.status}`)
      // The new file appears in dropdowns and clears the missing-model check.
      if (finished.some((d) => d.status === "completed")) {
        await refreshDefs()
        await loadFiles()
      }
      if (![...state.downloads.values()].some((d) => d.status === "downloading")) break
      await new Promise((resolve) => setTimeout(resolve, 2000))
    }
  } catch (e) {
    state.error = `Downloads: ${e.message}`
  } finally {
    polling = false
  }
}

function cancelDownload(id) {
  return run("Cancel download", () => fetch(remoteURL(`/remote_execution/models/downloads/${id}/cancel`), { method: "POST" }))
}

function useGatewaySocket() {
  // The frontend reconnects its socket to the new api_host; queue, history and logs resync on reconnect.
  const host = state.mode === "remote" ? state.gateway.host : location.host
  if (api.api_host !== host) {
    api.api_host = host
    api.socket?.close()
  }
}

async function setMode(mode) {
  if (mode === "remote" && currentTarget()?.status !== "ready") throw new Error("Connect to a remote server first.")
  state.mode = mode
  storageSet("RemoteExecution.mode", mode)
  useGatewaySocket()
  useRemoteDownloads()
  if (mode === "remote") return syncRemote()
  state.proxyTargetId = null
  state.defsFrom = "local"
  await refreshDefs()
  await gw("/proxy", { target_id: null })
}

async function startGateway() {
  try {
    await loadGateway()
    await gw("/session", {})
    state.gatewayError = null
    if (state.mode === "remote") {
      useGatewaySocket()
      useRemoteDownloads()
    }
    const ws = new WebSocket(`${state.gateway.base.replace(/^http/, "ws")}/events?token=${encodeURIComponent(state.gateway.token)}`)
    ws.onmessage = (e) => onMessage(JSON.parse(e.data))
    ws.onclose = () => {
      state.gatewayError = "Lost the gateway connection; retrying."
      update()
      setTimeout(startGateway, 3000)
    }
  } catch (e) {
    state.gatewayError = e.message
    setTimeout(startGateway, 5000)
  }
  update()
}

function onMessage(m) {
  if (m.type === "state") {
    state.targets = new Map(m.targets.map((t) => [t.id, t]))
    state.proxyTargetId = m.proxy_target_id
    if (!currentTarget()) state.currentTargetId = m.proxy_target_id ?? m.targets.find((t) => t.status === "ready")?.id ?? m.targets[0]?.id ?? null
    state.gatewayError = null
  } else if (m.type === "target") {
    state.targets.set(m.target.id, m.target)
    if (!currentTarget()) state.currentTargetId = m.target.id
  }
  update()
  syncRemote()
}

async function run(label, fn) {
  state.busy = label
  state.error = null
  update()
  try {
    await fn()
  } catch (e) {
    state.error = `${label}: ${e.message}`
    toast("error", label, e.message)
  } finally {
    state.busy = null
    update()
  }
}

function connect() {
  return run("Connect", async () => {
    const url = state.url.trim()
    if (!["http:", "https:"].includes(new URL(url).protocol)) throw new Error("Use an http:// or https:// address.")
    storageSet("RemoteExecution.url", url)
    const target = await gw("/targets/connect", { url })
    // Status comes from the ordered event stream; this reply can be older than an event already received.
    if (!state.targets.has(target.id)) state.targets.set(target.id, target)
    state.currentTargetId = target.id
  })
}

function disconnect() {
  return run("Disconnect", () => gw(`/targets/${state.currentTargetId}/disconnect`, {}))
}

function statusLine(target) {
  if (state.gatewayError) return h("div", { class: "re-warn" }, state.gatewayError)
  if (!target) return h("div", { class: "re-muted" }, "Not connected.")
  const parts = [{ ready: "Connected", connecting: "Connecting…", disconnected: "Disconnected", incompatible: "Incompatible" }[target.status] ?? target.status]
  for (const d of target.devices ?? []) parts.push(`${d.name.split(" : ")[0]} · ${(d.vram_total / 1024 ** 3).toFixed(1)} GB`)
  if (target.comfyui_version) parts.push(`ComfyUI ${target.comfyui_version}`)
  if (target.queue_remaining !== null && target.queue_remaining !== undefined) parts.push(`queue ${target.queue_remaining}`)
  return h(
    "div",
    {},
    h("div", { class: target.status === "ready" ? "re-ok" : "re-warn" }, parts.join(" · ")),
    target.detail && h("div", { class: "re-muted" }, target.detail),
    state.localVersion && target.comfyui_version && state.localVersion !== target.comfyui_version && h("div", { class: "re-warn" }, `Local ComfyUI is ${state.localVersion}; node definitions may differ.`)
  )
}

function update() {
  if (!mounted) return
  const target = currentTarget()
  const ready = target?.status === "ready"
  mounted.status.replaceChildren(statusLine(target))
  mounted.connect.disabled = !!state.busy
  mounted.disconnect.disabled = !!state.busy || !target || target.status === "disconnected"
  mounted.mode.value = state.mode
  mounted.mode.disabled = !!state.busy
  mounted.modeNote.replaceChildren(
    state.error ? h("div", { class: "re-err" }, state.error) : "",
    state.mode === "remote"
      ? h("div", { class: ready ? "re-ok" : "re-warn" }, ready ? `Running on ${new URL(target.url).host}: its nodes, models, files, queue, system info and logs. Workflows and settings stay local.` : "Remote mode is on but the server is not connected: Run fails instead of running locally.")
      : h("div", { class: "re-muted" }, ready ? "Switch to Remote to run on the connected server." : "Connect to enable remote mode.")
  )
  mounted.downloads.replaceChildren(...downloadsView())
  for (const [tab, button] of Object.entries(mounted.tabButtons)) button.classList.toggle("re-tab-active", tab === state.tab)
  mounted.connectionTab.hidden = state.tab !== "connection"
  mounted.filesTab.hidden = state.tab !== "files"
  mounted.files.replaceChildren(...filesView(ready).filter(Boolean))
  for (const [name, row] of Object.entries(mounted.keys)) {
    const saved = state.keyStatus?.[name]
    row.status.textContent = saved === undefined ? "" : saved ? "saved" : "not set"
    row.status.className = saved ? "re-ok" : "re-muted"
    row.clear.hidden = !saved
  }
  // Rebuilt only when the list changes, so a URL being typed is not replaced by status updates.
  const missing = state.mode === "remote" && ready ? missingModels() : []
  const key = JSON.stringify([missing.map((c) => [c.nodeId, c.name]), state.links?.size, state.busy])
  if (key !== state.missingKey) {
    state.missingKey = key
    mounted.missing.replaceChildren(...missingView(missing))
  }
}

function missingView(missing) {
  if (!missing.length) return []
  return [
    h("div", { class: "re-sub" }, "Missing in this workflow"),
    ...missing.map((c) => {
      const link = state.links?.get(c.name)
      const options = nodeByExecutionId(c.nodeId)?.widgets?.find((w) => w.name === c.widgetName)?.options?.values ?? []
      // Variants the remote already has, then variants with a known link that it does not have yet.
      const onRemote = options.filter((o) => typeof o === "string" && similar(o, c.name)).slice(0, 3)
      const stem = c.name.replace(/\.[a-z0-9]+$/i, "")
      const closest = (l) => (l.name.startsWith(stem) ? 0 : 1)
      const linked = [...(state.links?.values() ?? [])]
        .filter((l) => l.directory === c.directory && l.name !== c.name && !options.includes(l.name) && similar(l.name, c.name))
        .sort((a, b) => closest(a) - closest(b) || a.name.length - b.name.length)
        .slice(0, 2)
      const url = h("input", { class: "re-input", placeholder: "https://huggingface.co/…/resolve/main/…", value: state.urlDrafts[c.name] ?? "", oninput: (e) => (state.urlDrafts[c.name] = e.target.value) })
      return h(
        "div",
        { class: "re-card" },
        h("div", {}, h("b", {}, c.name), h("span", { class: "re-muted" }, ` · ${c.directory ?? "unknown folder"} · node #${c.nodeId}`)),
        link && h("div", { class: "re-row" }, h("button", { class: "re-btn", onclick: () => downloadToRemote(link.url, c.name, link.directory) }, "Download"), h("span", { class: "re-muted" }, new URL(link.url).pathname.split("/").slice(1, 3).join("/"))),
        onRemote.map((o) => h("div", { class: "re-row" }, h("button", { class: "re-btn", onclick: () => useModel(c, o) }, "Use instead"), h("span", {}, o))),
        linked.map((l) => h("div", { class: "re-row" }, h("button", { class: "re-btn", onclick: () => downloadToRemote(l.url, l.name, l.directory) }, "Download"), h("span", {}, l.name), h("span", { class: "re-muted" }, "(then Use instead)"))),
        !link && c.directory && h("div", { class: "re-row" }, url, h("button", { class: "re-btn", onclick: () => (state.urlDrafts[c.name] ? downloadToRemote(state.urlDrafts[c.name].trim(), c.name, c.directory) : toast("warn", "Download", "Paste a Hugging Face or Civitai file URL first.")) }, "Download"))
      )
    }),
  ]
}

function downloadStatus(d) {
  if (d.status === "downloading") return d.total ? `${Math.floor((100 * d.received) / d.total)}% of ${formatSize(d.total)}` : formatSize(d.received)
  if (d.status === "completed") return "done"
  if (d.status === "failed") return d.error?.startsWith("Access denied") ? "access denied" : "failed"
  return d.status
}

function downloadsView() {
  // The latest attempt per file; it replaces earlier failed attempts of the same file.
  const latest = new Map()
  for (const d of [...state.downloads.values()].sort((a, b) => a.started - b.started)) latest.set(`${d.directory}/${d.name}`, d)
  const shown = [...latest.values()].filter((d) => d.status === "downloading" || d.started > state.clearedBefore).sort((a, b) => b.started - a.started)
  if (!shown.length) return []
  const finished = shown.some((d) => d.status !== "downloading")
  const denied = shown.some((d) => d.status === "failed" && d.error?.startsWith("Access denied"))
  return [
    h("div", { class: "re-row" }, h("span", { class: "re-sub" }, "Remote downloads"), finished && h("button", { class: "re-btn", onclick: clearFinished }, "Clear finished")),
    ...shown.map((d) =>
      h(
        "div",
        { class: "re-file", title: d.error ?? d.url },
        h("span", { class: `re-name ${d.status === "failed" ? "re-err" : d.status === "completed" ? "re-ok" : ""}` }, d.name, h("span", { class: "re-muted" }, ` · ${d.directory}`)),
        h("span", { class: "re-muted" }, downloadStatus(d)),
        d.status === "downloading" && h("button", { class: "re-btn", onclick: () => cancelDownload(d.id) }, "Cancel")
      )
    ),
    denied && h("div", { class: "re-muted" }, "Access denied: save a key under Access keys and accept the model's terms on its page, then download again."),
  ].filter(Boolean)
}

function clearFinished() {
  state.clearedBefore = Date.now() / 1000
  storageSet("RemoteExecution.clearedBefore", String(state.clearedBefore))
  update()
}

function filesView(ready) {
  if (state.mode !== "remote" || !ready) return [h("div", { class: "re-muted" }, "Switch to remote mode with a connected server to manage its model files.")]
  if (state.filesError) return [h("div", { class: "re-err" }, state.filesError)]
  if (!state.files) return [h("div", { class: "re-muted" }, "Loading…")]
  const { disk, files } = state.files
  const groups = new Map()
  for (const file of files) groups.set(file.directory, [...(groups.get(file.directory) ?? []), file])
  const total = (list) => list.reduce((sum, f) => sum + f.size, 0)
  return [
    h("div", { class: "re-row" }, h("span", {}, `${formatSize(disk.free)} free of ${formatSize(disk.total)}`), h("button", { class: "re-btn", onclick: loadFiles }, "Refresh")),
    h("div", { class: "re-progress" }, h("div", { style: `width: ${(100 * (disk.total - disk.free)) / disk.total}%` })),
    files.length === 0 && h("div", { class: "re-muted" }, "No model files on the remote."),
    ...[...groups.entries()]
      .sort((a, b) => total(b[1]) - total(a[1]))
      .flatMap(([directory, list]) => [
        h("div", { class: "re-sub" }, `${directory} · ${formatSize(total(list))}`),
        ...list
          .sort((a, b) => b.size - a.size)
          .map((file) => {
            const key = `${file.directory}/${file.name}`
            return h(
              "div",
              { class: "re-file" },
              h("span", { class: "re-name" }, file.name),
              h("span", { class: "re-muted" }, formatSize(file.size)),
              state.confirming === key
                ? [h("button", { class: "re-btn re-danger", onclick: () => deleteFile(file), disabled: !!state.busy }, "Delete"), h("button", { class: "re-btn", onclick: () => ((state.confirming = null), update()) }, "Keep")]
                : h("button", { class: "re-btn", onclick: () => ((state.confirming = key), update()) }, "Delete…")
            )
          }),
      ]),
  ]
}

function mount(el) {
  const url = h("input", { class: "re-input", value: state.url, placeholder: "https://…trycloudflare.com", oninput: (e) => (state.url = e.target.value) })
  mounted = {
    // Read the choice before run() re-renders, which resets the select to the current mode.
    mode: h("select", { class: "re-input", onchange: (e) => ((mode) => run("Switch mode", () => setMode(mode)))(e.target.value) }, h("option", { value: "local" }, "Local"), h("option", { value: "remote" }, "Remote")),
    modeNote: h("div"),
    status: h("div"),
    connect: h("button", { class: "re-btn", onclick: connect }, "Connect"),
    disconnect: h("button", { class: "re-btn", onclick: disconnect }, "Disconnect"),
    downloads: h("div"),
    missing: h("div", { class: "re-section" }),
    files: h("div"),
    tabButtons: {
      connection: h("button", { class: "re-tab", onclick: () => showTab("connection") }, "Connection"),
      files: h("button", { class: "re-tab", onclick: () => showTab("files") }, "Model files"),
    },
  }
  mounted.connectionTab = h("div", { class: "re-section" }, h("div", { class: "re-sub" }, "Remote server"), url, h("div", { class: "re-row" }, mounted.connect, mounted.disconnect), mounted.status, h("label", { class: "re-row" }, "Execution", mounted.mode), mounted.modeNote)
  mounted.keys = {}
  const keyRows = [
    ["huggingface", "Hugging Face token", "hf_…"],
    ["civitai", "Civitai API key", "Civitai API key"],
  ].map(([name, label, placeholder]) => {
    const input = h("input", { class: "re-input", type: "password", autocomplete: "off", placeholder })
    const row = { status: h("span"), clear: h("button", { class: "re-btn", onclick: () => ((input.value = ""), saveKey(name, input)) }, "Clear") }
    mounted.keys[name] = row
    return h("div", { class: "re-section" }, h("div", { class: "re-row" }, h("span", {}, label), row.status), h("div", { class: "re-row" }, input, h("button", { class: "re-btn", onclick: () => saveKey(name, input) }, "Save"), row.clear))
  })
  const keys = h(
    "div",
    { class: "re-section" },
    h("div", { class: "re-sub" }, "Access keys"),
    h("div", { class: "re-muted" }, "For gated or login-only models. Kept on this PC by the gateway and sent only with downloads from that site."),
    ...keyRows
  )
  mounted.filesTab = h("div", { class: "re-section" }, mounted.missing, mounted.downloads, mounted.files, keys)
  state.missingKey = null
  el.replaceChildren(h("div", { class: "re-panel" }, h("div", { class: "re-tabs" }, Object.values(mounted.tabButtons)), mounted.connectionTab, mounted.filesTab))
  update()
  if (state.tab === "files") showTab("files")
}

const STYLE = `
.re-panel { display: flex; flex-direction: column; gap: 8px; padding: 12px; font-size: 13px; color: var(--fg-color); overflow-y: auto; height: 100%; box-sizing: border-box; }
.re-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.re-input { background: var(--comfy-input-bg); color: var(--input-text, var(--fg-color)); border: 1px solid var(--border-color); border-radius: 6px; padding: 5px 8px; font: inherit; min-width: 0; flex: 1; }
.re-btn { background: var(--comfy-input-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 6px; padding: 5px 10px; cursor: pointer; font: inherit; }
.re-btn:disabled { opacity: 0.5; cursor: default; }
.re-sub { font-weight: 600; margin-top: 6px; }
.re-muted { color: var(--descrip-text, #999); }
.re-ok { color: #22c55e; } .re-warn { color: #f59e0b; } .re-err { color: #ef4444; }
.re-section { display: flex; flex-direction: column; gap: 8px; }
.re-section[hidden] { display: none; }
.re-tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border-color); }
.re-tab { background: none; border: none; border-bottom: 2px solid transparent; color: var(--descrip-text, #999); padding: 6px 10px; cursor: pointer; font: inherit; }
.re-tab-active { color: var(--fg-color); border-bottom-color: var(--fg-color); }
.re-danger { background: #dc2626; border-color: #dc2626; color: #fff; }
.re-file { display: flex; gap: 8px; align-items: center; }
.re-name { flex: 1; min-width: 0; overflow-wrap: anywhere; }
.re-progress { height: 4px; background: var(--border-color); border-radius: 2px; overflow: hidden; }
.re-progress div { height: 100%; background: var(--descrip-text, #999); }
.re-card { border: 1px solid var(--border-color); border-radius: 8px; padding: 8px; display: flex; flex-direction: column; gap: 6px; overflow-wrap: anywhere; }
`

app.registerExtension({
  name: "Comfy.RemoteExecution",
  settings: [
    {
      id: REMEMBER_SETTING,
      category: ["Remote execution", "Mode", "Remember"],
      name: "Stay in remote mode after reloading the page",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async setup() {
    document.head.append(h("style", {}, STYLE))
    desktopHost = window.__comfyDesktop2
    const apiURL = api.apiURL.bind(api)
    const internalURL = api.internalURL.bind(api)
    fetch(apiURL("/system_stats")).then(async (r) => ((state.localVersion = (await r.json()).system?.comfyui_version), update()))
    // In remote mode these routes resolve to the gateway's remote proxy; the native frontend
    // (Run, queue, progress, previews, outputs, dropdowns, Issues, logs) then works unchanged.
    api.apiURL = (route) => (isRemote(REMOTE_ROUTES, route) ? remoteURL(route.startsWith("/api/") ? route : `/api${route}`) : apiURL(route))
    api.internalURL = (route) => (isRemote(REMOTE_INTERNAL_ROUTES, route) ? remoteURL(`/internal${route}`) : internalURL(route))
    const open = XMLHttpRequest.prototype.open
    XMLHttpRequest.prototype.open = function (method, url, ...rest) {
      const manager = state.mode === "remote" && typeof url === "string" && url.match(MANAGER_URL)
      return open.call(this, method, manager ? remoteURL(`/api${manager[1]}`) : url, ...rest)
    }

    if (app.extensionManager.setting.get(REMEMBER_SETTING) && storageGet("RemoteExecution.mode", "local") === "remote") state.mode = "remote"

    app.extensionManager.registerSidebarTab({ id: "remote-execution", icon: "pi pi-cloud", title: "Remote", tooltip: "Remote execution", type: "custom", render: mount, destroy: () => (mounted = null) })
    startGateway()
  },
})
