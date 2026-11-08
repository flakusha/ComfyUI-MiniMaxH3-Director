// Shared widget reads and edits, also used by presets so edits notify the plan panel.
export function newContextId(source = typeof crypto === "undefined" ? null : crypto) {
  // Identity only, never a credential. getRandomValues also works on ordinary LAN HTTP pages.
  const bytes = new Uint8Array(16);
  if (source?.getRandomValues) source.getRandomValues(bytes);
  else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return "ctx_" + Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

// A copied card, or a preset loaded onto a second card, carried the first card's ID and name: the Connector then
// stopped the run with "id used twice (Mage and Mage)". The card that came later gets its own ID and name.
export function ownIdentity(own, others, makeId = newContextId) {
  const changes = {};
  if (own.context_id && others.some((other) => other.context_id === own.context_id)) changes.context_id = makeId();
  const taken = new Set(others.map((other) => String(other.name || "").trim().toLowerCase()).filter(Boolean));
  const name = String(own.name || "").trim();
  if (name && taken.has(name.toLowerCase())) {
    let n = 2;
    while (taken.has(`${name} ${n}`.toLowerCase())) n++;
    changes.name = `${name} ${n}`;
  }
  return changes;
}

// ComfyUI draws a DOM widget (widget.width ?? node.width) wide. Showing the widget in a side panel (Workflow Overview,
// node properties) writes that panel's width (about 100 px) into widget.width, and it stuck: the Studio and the Context
// cards then lay out 80 px wide, in one column, thousands of pixels tall (user report, 2026-10-11). Our widgets are always
// as wide as their node; the panel passes its width to draw() itself.
export function followNodeWidth(widget) {
  Object.defineProperty(widget, "width", { configurable: true, enumerable: false, get: () => undefined, set: () => {} });
  return widget;
}

// The id ng/core/context_build.py (card_id) gives a card whose Context ID field is empty: media and a slug of its label.
export function contextId(widgets, media) {
  if (widgets.context_id) return String(widgets.context_id);
  const name = String(widgets.name || "").trim(), source = String(widgets[media] || "");
  const file = source.replace(" [input]", "").trim().split(/[\\/]/).pop().replace(/\.[^.]*$/, "");
  const label = name || (media === "text" ? source.slice(0, 24).trim() || "note" : file || media);
  return `${media}_${label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "item"}`;
}

export function values(node) {
  return Object.fromEntries((node.widgets || []).filter((w) => w.name).map((w) => [w.name, w.value]));
}

export function analysisFingerprint(media,widgets) {
  return JSON.stringify([widgets[media],widgets.description,widgets.content_roles||widgets.content]);
}

export function nativeAIFingerprint(node) {
  const port=node.inputs?.find((input)=>input.name==="clip"),link=node.graph?.links?.[port?.link];
  const encoder=link&&node.graph.getNodeById(link.origin_id);
  return JSON.stringify([values(node),encoder?.id,encoder?values(encoder):null]);
}

// Edits made from a panel do not pass ComfyUI's own input events, so the workflow would not notice them:
// no unsaved mark, no undo step and the browser restore keeps the old graph. Ask the change tracker once they settle.
let tracking = null;
export function noteChange() {
  clearTimeout(tracking);
  tracking = setTimeout(() => globalThis.window?.comfyAPI?.app?.app?.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(), 300);
}

export function applyValues(node, changes) {
  for (const widget of node.widgets || []) {
    if (!(widget.name in changes)) continue;
    const old = widget.value;
    widget.value = changes[widget.name];
    widget.callback?.(widget.value);
    // the hook ComfyUI's own widgets call: it drops a "missing model" mark once another file is chosen
    node.onWidgetChanged?.(widget.name, widget.value, old, widget);
  }
  node.setDirtyCanvas?.(true, true);
  document.dispatchEvent(new CustomEvent("ng-context-changed"));
  noteChange();
}

// An AI Loader address that is no HTTP(S) URL (left over or shifted in an older workflow) is cleared, so the provider's own address is used.
export const clearedAddress = (address) => (address && !/^https?:\/\/[^\s/?#]+/i.test(String(address)) ? { base_url: "" } : {});

// A workflow saved before Trim start/end existed restores positionally: the Context ID lands in Trim start,
// the colour in Trim end, and the JSON fields shift along after them (12 values; 9 when it also predates the
// role fields). Re-seat by name, so the ID never reaches the node's float conversion. A save the frontend
// already restored by name maps to the same values, so this is a no-op there (h3s_node.js guards the same way).
export function reseatPreTrimAudio(node, info) {
  const saved = info?.widgets_values;
  if (!Array.isArray(saved) || (saved.length !== 12 && saved.length !== 9)) return;
  const named = info?.widgets_values_named;
  const names = ["audio", "name", "description", "content", "usage", "preserve", "speaker",
    "trim_start", "trim_end", "context_id", "color", "content_roles", "usage_roles", "context_options"];
  if (named && typeof named === "object" && !Array.isArray(named) && names.every((name) => Object.hasOwn(named, name))) return;
  const at = Object.fromEntries((node.widgets || []).filter((w) => w?.name).map((w) => [w.name, w]));
  ["audio", "name", "description", "content", "usage", "preserve", "speaker"].forEach((name, i) => { if (at[name]) at[name].value = saved[i]; });
  if (at.trim_start) at.trim_start.value = 0;
  if (at.trim_end) at.trim_end.value = 0;
  ["context_id", "color", "content_roles", "usage_roles", "context_options"].forEach((name, i) => { if (at[name] && saved[7 + i] !== undefined) at[name].value = saved[7 + i]; });
  if (saved.length === 9) for (const name of ["content_roles", "usage_roles", "context_options"]) if (at[name]) at[name].value = "";
}

export function element(tag, className = "", text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export async function request(api, path, payload, signal) {
  // The frontend's fetchApi aborts a request after its own built-in 60 s unless timeoutMs says otherwise.
  // An AI request is ended by the provider, at the AI Loader's Timeout, and the route answers with the
  // transport's real error ("did not answer within Ns") — so the browser must not cut the connection first.
  // TODO(env): timeoutMs is the ComfyUI frontend 1.58.5 fetchApi option (requirements.txt pin; its
  // watchdog default is 6e4 ms). Older frontends drop the unknown key harmlessly; when the pinned
  // frontend package moves and its fetchApi contract changes, re-check this option name here.
  const options = payload ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), timeoutMs: null, signal }
                          : { timeoutMs: null, signal };
  const response = await api.fetchApi(path, options);
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || data.reason || "The request failed.");
  return data;
}
