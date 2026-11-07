// Real Context card extension hooks, offline host DOM/API: a linked audio_in mirrors the origin's file into the
// card's own audio widget, locks the file controls, re-syncs on origin changes and restores the own file on unlink.
import assert from "node:assert/strict";
import test from "node:test";
import { installDom } from "./fake_dom.mjs";

const settle = (ms = 90) => new Promise((resolve) => setTimeout(resolve, ms));

installDom();
const proto = Object.getPrototypeOf(document.body);
proto.querySelectorAll = function (selector) { return this.find((node) => selector.split(",").map((v) => v.trim().toUpperCase()).includes(node.tagName)); };
proto.querySelector = function (selector) { return this.querySelectorAll(selector)[0] || null; };
proto.pause = function () {};
globalThis.requestAnimationFrame = () => {};
const state = { presetNames: ["P"], preset: {} };
const payloads = (url) => url.includes("/ng/presets") ? { ok: true, names: state.presetNames }
  : url.includes("/ng/preset?") ? { ok: true, widgets: state.preset }
  : url.includes("/ng/waveform") ? { ok: true, peaks: [0.5], duration_s: 3, sample_rate: 44100, channels: 2 }
  : { ok: true };
let fetches = 0;
const apiCalls = () => fetches;
const listeners = new Map();
const listen = (name, fn) => listeners.set(name, [...(listeners.get(name) || []), fn]);
const unlisten = (name, fn) => listeners.set(name, (listeners.get(name) || []).filter((entry) => entry !== fn));
const emit = (name, detail) => { for (const fn of listeners.get(name) || []) fn({ detail }); };
const api = { apiURL: (path) => path, fetchApi: async (url) => { fetches++; return { ok: true, json: async () => payloads(String(url)) }; },
  addEventListener: listen, removeEventListener: unlisten };
const extensions = [];
globalThis.window = { comfyAPI: { app: { app: { registerExtension: (extension) => extensions.push(extension) } }, api: { api } } };
await import("../../../js/ng/h3c_card.js");
const extension = extensions.find((entry) => entry.name === "mmxd.ng.context_cards");
assert.ok(extension, "execute the actual product extension");

const { values } = await import("../../../js/ng/ng_widgets.js");

function makeCard(widgetValues, comfyClass = "NGContextAudio") {
  const card = { id: 1, comfyClass, properties: {}, size: [800, 800], inputs: [], outputs: [{ links: [1] }],
    widgets: Object.entries(widgetValues).map(([name, value]) => ({ name, value, options: { values: [value] } })),
    setSize(size) { this.size = size; }, setDirtyCanvas() {},
    addDOMWidget(name, type, root) { this.root = root; const widget = { name, type, serialize: false }; this.widgets.push(widget); return widget; } };
  const connector = { id: 2, comfyClass: "NGContextConnector", inputs: [{ name: "contexts.0", link: 1 }], widgets: [], outputs: [] };
  const director = { id: 3, comfyClass: "NGH3Director", inputs: [{ name: "context", link: 2 }], widgets: [] };
  const all = [card, connector, director];
  const graph = { _nodes: all, links: { 1: { origin_id: 1, target_id: 2 }, 2: { origin_id: 2, target_id: 3 } },
    getNodeById: (id) => all.find((node) => node.id === id) };
  for (const node of all) node.graph = graph;
  return { card, graph };
}

function makeLoader(graph, comfyClass, widgets, linkId, title = comfyClass, id = 10 + linkId) {
  const loader = { id, comfyClass, title, widgets, inputs: [], outputs: [{ links: [] }] };
  graph._nodes.push(loader);
  graph.links[linkId] = { origin_id: id, target_id: 1 };
  return loader;
}

function link(card, linkId, port = "audio_in") {
  card.inputs.push({ name: port, link: linkId });
  card.onConnectionsChange?.();
}

function unlink(card, linkId, port = "audio_in") {
  card.inputs.find((entry) => entry.name === port && entry.link === linkId).link = null;
  card.onConnectionsChange?.();
}

const fileSelect = (card) => card.root.find((node) => node.attributes?.["aria-label"] === "File")[0];
const uploadButton = (card) => card.root.find((node) => node.tagName === "BUTTON" && node.innerText === "Upload")[0];
const chip = (card) => card.root.find((node) => node.classList?.contains("h3c-link"))[0];

const AUDIO_WIDGETS = { audio: "voice.wav", name: "Voice", description: "", content: "voice", usage: "voice", preserve: "high",
  speaker: "", trim_start: 0.0, trim_end: 0.0, context_id: "ctx_voice", color: "", content_roles: "", usage_roles: "", context_options: "", subject_ids: "" };
const TEXT_WIDGETS = { text: "card note", name: "Rules", description: "", content: "rules", usage: "rule", preserve: "high",
  context_id: "ctx_text", color: "", content_roles: "", usage_roles: "", context_options: "", subject_ids: "" };

test("A linked external input owns the card's file: mirror, lock, re-sync, preset filter and restore", async (t) => {
  const { card, graph } = makeCard({ ...AUDIO_WIDGETS });
  extension.nodeCreated(card);
  assert.equal(values(card).audio, "voice.wav");
  assert.equal(fileSelect(card).disabled, false, "unlinked, the file picker stays editable");
  assert.equal(chip(card), undefined);

  await t.test("linking mirrors the origin's file, locks the controls and names the origin", async () => {
    const loader = makeLoader(graph, "LoadAudio", [{ name: "audio", value: "song.wav" }], 7, "Load Audio");
    link(card, 7);
    await settle();
    assert.equal(values(card).audio, "song.wav", "the origin's file is mirrored into the card's own widget");
    assert.equal(card.properties.ng_external_saved.audio, "voice.wav", "the card's own file is kept for unlink");
    assert.equal(fileSelect(card).disabled, true, "the file picker locks");
    assert.equal(uploadButton(card).disabled, true, "Upload locks");
    assert.match(chip(card).innerText, /Linked · Load Audio · song\.wav/);
    assert.ok(loader.widgets[0].ngExternalHooked, "the origin widget's callback is wrapped for this card");
    assert.equal(card.root.find((node) => node.classList?.contains("h3c-warn"))[0], undefined, "no warning while the origin carries a file");
  });

  await t.test("a saved workflow that loads while linked keeps its stored own file", async () => {
    const linked = makeCard({ ...AUDIO_WIDGETS, audio: "song.wav" });
    linked.card.properties.ng_external_saved = { audio: "own.wav" };
    linked.card.inputs.push({ name: "audio_in", link: 7 });
    makeLoader(linked.graph, "LoadAudio", [{ name: "audio", value: "song.wav" }], 7, "Load Audio");
    extension.nodeCreated(linked.card);
    await settle();
    assert.equal(values(linked.card).audio, "song.wav");
    assert.equal(linked.card.properties.ng_external_saved.audio, "own.wav", "the mirror must not overwrite the saved own value");
    unlink(linked.card, 7);
    await settle();
    assert.equal(values(linked.card).audio, "own.wav");
  });

  await t.test("changing the origin's file re-syncs without any graph event", async () => {
    const loader = graph._nodes.find((node) => node.comfyClass === "LoadAudio");
    loader.widgets[0].value = "other.wav";
    loader.widgets[0].callback("other.wav");
    await settle();
    assert.equal(values(card).audio, "other.wav", "the mirror follows the origin");
    assert.match(chip(card).innerText, /other\.wav/);
    const before = apiCalls();   // baseline only once the mirror has fully settled
    await settle(250);
    assert.equal(apiCalls(), before, "the mirror settles: no render/request loop keeps running");
  });

  await t.test("loading a preset on a linked card skips the file key", async () => {
    state.preset = { audio: "preset.wav", name: "From preset", description: "A preset." };
    const load = card.root.find((node) => node.tagName === "BUTTON" && node.innerText === "Load")[0];
    assert.ok(load && !load.disabled, "the preset list arrived, Load is enabled");
    load.onclick();
    await settle();
    assert.equal(values(card).name, "From preset", "the preset's name lands");
    assert.equal(values(card).audio, "other.wav", "the preset's file never overwrites the linked one");
  });

  await t.test("unlinking restores the card's previous own file", async () => {
    const loader = graph._nodes.find((node) => node.comfyClass === "LoadAudio");
    unlink(card, 7);
    await settle();
    assert.equal(values(card).audio, "voice.wav", "the value captured at connect time comes back");
    assert.equal(card.properties.ng_external_saved?.audio, undefined);
    assert.equal(fileSelect(card).disabled, false, "the picker unlocks");
    assert.equal(uploadButton(card).disabled, false);
    assert.equal(chip(card), undefined, "the link chip is gone");
    assert.equal(loader.widgets[0].ngExternalWatch.size, 0, "this card unsubscribed from the origin");
  });

  await t.test("a linked origin without a file warns and leaves the card's value alone", async () => {
    makeLoader(graph, "EmptyAudio", [{ name: "sample_rate", value: 44100 }, { name: "duration", value: 3 }], 8);
    link(card, 8);
    await settle();
    assert.equal(values(card).audio, "voice.wav", "nothing to mirror; the card's own file stays");
    assert.ok(card.root.find((node) => node.classList?.contains("h3c-warn")).length >= 1, "chip and status warn");
    assert.match(card.root.innerText, /carries no file/);
    assert.match(chip(card).innerText, /EmptyAudio/);
    assert.equal(graph._nodes.find((node) => node.comfyClass === "LoadAudio").widgets[0].ngExternalWatch.size, 0,
      "the old origin stays unsubscribed; the new one has no file widget to watch");
  });
});

test("A linked text_in is the note: shown read-only, the card's own text untouched", async (t) => {
  const { card, graph } = makeCard({ ...TEXT_WIDGETS }, "NGContextText");
  extension.nodeCreated(card);
  const editor = () => card.root.find((node) => node.attributes?.["aria-label"] === "Note")[0];
  assert.equal(editor().disabled, false);

  await t.test("linking shows the origin's text read-only", async () => {
    makeLoader(graph, "PrimitiveString", [{ name: "string", value: " STYLE: noir. " }], 7, "Primitive String");
    link(card, 7, "text_in");
    await settle();
    assert.equal(editor().disabled, true, "the note editor locks");
    assert.equal(editor().value, " STYLE: noir. ", "the linked text is shown");
    assert.equal(values(card).text, "card note", "the card's own text is not mirrored over");
    const insert = card.root.find((node) => node.tagName === "BUTTON" && node.innerText.startsWith("+ "))[0];
    insert.onclick();
    assert.equal(values(card).text, "card note", "quick insert does not edit a linked note");
    assert.match(card.root.innerText, /note is linked/);
  });

  await t.test("unlinking brings the own note back", async () => {
    unlink(card, 7, "text_in");
    await settle();
    assert.equal(editor().disabled, false);
    assert.equal(editor().value, "card note");
  });
});

test("A linked video locks the mask box: picking a mask would silently unmirror it", async (t) => {
  const { maskBox } = await import("../../../js/ng/h3c_mask.js");
  const calls = [];
  const actions = new Proxy({}, { get: (_t, name) => (...args) => { calls.push([name, ...args]); } });
  const base = { opts: { control_mask: "", control_result: "", mask_shape: "box", mask_prompt: "person", mask_grow: 16 },
    values: { video: "clip.mp4" }, files: ["mask.mp4"], busy: "", els: { mask: document.createElement("video") }, api };
  await t.test("unlinked, the mask controls stay editable", async () => {
    const view = maskBox({ ...base, external: { linked: false } }, actions);
    assert.equal(view.find((node) => node.attributes?.["aria-label"] === "Mask video")[0].disabled, false);
  });
  await t.test("linked, choose/shape/prompt/grow/upload/generate all lock", async () => {
    const view = maskBox({ ...base, external: { linked: true, label: "Load Video" } }, actions);
    assert.equal(view.find((node) => node.attributes?.["aria-label"] === "Mask video")[0].disabled, true);
    assert.equal(view.find((node) => node.attributes?.["aria-label"] === "Mask shape")[0].disabled, true);
    assert.equal(view.find((node) => node.attributes?.["aria-label"] === "Mask object")[0].disabled, true);
    assert.equal(view.find((node) => node.attributes?.["aria-label"] === "Mask grow")[0].disabled, true);
    assert.equal(view.find((node) => node.attributes?.["aria-label"] === "Upload mask video")[0].disabled, true);
    const generate = view.find((node) => node.tagName === "BUTTON" && node.innerText.startsWith("Generate mask"))[0];
    assert.equal(generate.disabled, true);
  });
});

test("A trim-era save repairs non-JSON role/options fields on configure", async (t) => {
  await t.test("mid-length values holding filenames reset by name, valid JSON survives", async () => {
    const card = makeCard({ ...AUDIO_WIDGETS, content_roles: "voice.wav", usage_roles: "[\"voice\"]",
      context_options: "song.mp3", subject_ids: "" });
    extension.nodeCreated(card.card);
    card.card.onConfigure({ widgets_values: Array(12).fill("junk") });
    const byName = Object.fromEntries(card.card.widgets.map((widget) => [widget.name, widget.value]));
    assert.equal(byName.content_roles, "", "a filename in a JSON field resets");
    assert.equal(byName.context_options, "", "a filename in the options field resets");
    assert.equal(byName.usage_roles, "[\"voice\"]", "valid JSON is left alone");
  });
});

test("A run that saves a linked sound pushes its name into the card's empty widget", async (t) => {
  const bare = makeCard({ ...AUDIO_WIDGETS, audio: "" });
  bare.card.id = 4;
  extension.nodeCreated(bare.card);
  makeLoader(bare.graph, "EmptyAudio", [{ name: "sample_rate", value: 44100 }], 8);
  link(bare.card, 8);
  await settle();
  assert.equal(values(bare.card).audio, "", "nothing to mirror; the widget stays empty");

  await t.test("another card's push is ignored", async () => {
    emit("minimax_h3_ng_audio", { node_id: 99, audio: "h3-external/other.flac" });
    await settle();
    assert.equal(values(bare.card).audio, "");
  });

  await t.test("the card's own push lands, previews and clears the warning", async () => {
    const before = apiCalls();
    emit("minimax_h3_ng_audio", { node_id: 4, audio: "h3-external/abc123.flac" });
    await settle();
    assert.equal(values(bare.card).audio, "h3-external/abc123.flac", "the pushed name becomes the card's file");
    assert.ok(apiCalls() > before, "the new file previews (waveform request)");
    assert.match(bare.card.root.innerText, /h3-external\/abc123\.flac/);
    assert.doesNotMatch(bare.card.root.innerText, /carries no file/, "a saved sound needs no run");
  });

  await t.test("an occupied widget is never overwritten", async () => {
    emit("minimax_h3_ng_audio", { node_id: 4, audio: "h3-external/def456.flac" });
    await settle();
    assert.equal(values(bare.card).audio, "h3-external/abc123.flac");
  });
});

test("The materialized push fills only an empty audio widget of its own card", async () => {
  const { watchMaterialized } = await import("../../../js/ng/ng_external.js");
  const heard = [];
  const events = new Map();
  const stub = { addEventListener: (name, fn) => events.set(name, fn), removeEventListener: (name) => events.delete(name) };
  const node = { id: 5, properties: {}, widgets: [{ name: "audio", value: "" }] };
  const stop = watchMaterialized(node, "audio", stub, (file) => heard.push(file));
  const fire = events.get("minimax_h3_ng_audio");
  fire({ detail: { node_id: 5, audio: "h3-external/a.flac" } });
  assert.deepEqual(heard, ["h3-external/a.flac"]);
  fire({ detail: { node_id: 6, audio: "h3-external/b.flac" } });
  fire({ detail: { node_id: 5, audio: 42 } });
  fire({});
  node.widgets[0].value = "mirror.wav";
  fire({ detail: { node_id: 5, audio: "h3-external/c.flac" } });
  assert.deepEqual(heard, ["h3-external/a.flac"], "wrong card, non-string, empty detail and occupied widgets are ignored");
  stop();
  assert.equal(events.has("minimax_h3_ng_audio"), false, "unsubscribe removes the listener");
  watchMaterialized({ id: 5, properties: {}, widgets: [{ name: "audio", value: "" }] }, "image", stub, (file) => heard.push(file));
  events.get("minimax_h3_ng_audio")({ detail: { node_id: 5, audio: "h3-external/d.flac" } });
  assert.deepEqual(heard, ["h3-external/a.flac"], "image cards never take an audio push");
});
