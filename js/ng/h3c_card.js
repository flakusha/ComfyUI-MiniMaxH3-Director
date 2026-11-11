// The four H3 Context nodes as full cards in the graph (design: Context-Nodes v1). The node's own widgets stay
// the saved values; the card draws them, redraws on change and grows the node to its content so cards never overlap.
import { h, keepKeys } from "./h3s_dom.js";
import { ensureCardStyle } from "./h3c_style.js";
import { TEMPLATES, usableFields, offerFor } from "./h3c_model.js";
import { CARD_MODELS, modelState, da3Available } from "./ng_models.js";
import { mediaURL, wirePlayback, moveHeads } from "./h3c_media.js";
import { stampLine, deviceNote, storeEngine, wireEngines } from "./h3c_lyrics.js";
import { values, applyValues, newContextId, ownIdentity, analysisFingerprint, request, followNodeWidth } from "./ng_widgets.js";
import { contextOptions, setOptions, selectedRoles } from "./ng_context_controls.js";
import { hideNative } from "./ng_editor_fields.js";
import { externalSource, watchOrigin, watchMaterialized } from "./ng_external.js";
import { contextProvider, relatedContexts } from "./ng_read.js";
import { downstreamNodes } from "./ng_graph.js";
import { aiAction } from "./ng_ai_request.js";
import { offeredFields, applyChanges, reasonLine } from "./ng_assist.js";
import { autoColor } from "./ng_identity.js";
import { FAMILY } from "./ng_context.js";
import { assetChoices, assetSubjects, restoreAssetWidget } from "./ng_asset_relations.js";
import { subjectChoices } from "./ng_subjects.js";
import { followNoteRenames, knownNames } from "./ng_references.js";
import { noteGrowth, followDraggedHeight, pushOnGrowth } from "./ng_layout.js";
import { dropTarget } from "./ng_drop.js";
import { controlGraph, controlSetup, reseatPreTrim } from "./h3c_control_graph.js";
import { layout, WIDTH } from "./h3c_layout.js";
import { runJob } from "./h3b_queue.js";
import { uploadMask } from "./h3c_mask.js";
import { uploadVideo, videoSource } from "./h3c_upload.js";
import { openPictureEditor } from "./h3c_editor.js";
import { cardContentHeight } from "./h3c_measure.js";

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;
const MEDIA = { NGContextImage: "image", NGContextVideo: "video", NGContextAudio: "audio", NGContextText: "text" };
// One preset list request per medium for all cards that load together (six image cards asked six times); saving refreshes it.
const presetLists = new Map();
function presetNames(media, fresh = false) {
  if (fresh || !presetLists.has(media)) {
    const pending = request(api, "/ng/presets?media=" + media).then((data) => data.names);
    pending.catch(() => presetLists.delete(media));
    presetLists.set(media, pending);
  }
  return presetLists.get(media);
}

function repairNullableText(node) {
  for (const widget of node.widgets || [])
    if (["name", "description", "speaker", "color"].includes(widget.name) && widget.value == null) widget.value = "";
}

function attach(node, media) {
  ensureCardStyle();
  repairNullableText(node);
  node.properties ||= {};
  for (const key of ["ng_context_expanded"]) delete node.properties[key];
  const identity = () => node.widgets?.find((widget) => widget.name === "context_id");
  const colorWidget = () => node.widgets?.find((widget) => widget.name === "color");
  function assignColor(force = false) {
    if (!colorWidget() || (colorWidget().value && !force)) return;
    const used = (node.graph?._nodes || []).filter((other) => other !== node).map((other) => values(other).color);
    colorWidget().value = autoColor(media, identity()?.value, used);
  }
  if (identity() && !identity().value) identity().value = newContextId();
  assignColor();
  node.properties.ng_preview_settings ||= { volume: 1, fade_in: 0, fade_out: 0 };

  const root = h("div", { class: "h3c", dataset: { media } });
  keepKeys(root);
  const S = { media, node, api, els: {}, files: [], presets: [], presetName: node.properties.ng_preset_name || "", suggestion: null, busy: false,
    envelope: null, waveError: "", dragging: false, pending: false, status: "", statusWarn: false, listen: node.properties.ng_preview_settings };
  if (media === "image") { S.els.image = new Image(); S.els.image.onload = () => schedule(); }
  if (media === "video") {
    S.els.video = h("video", { controls: true, preload: "metadata", muted: false }); S.els.video.onloadedmetadata = () => schedule();
    S.els.original = h("video", { controls: true, preload: "metadata", muted: true });
    S.els.mask = h("video", { controls: true, preload: "metadata", muted: true });
    da3Available(api).then((available) => { S.da3 = available; schedule(); });
  }
  if (media === "audio") { S.els.audio = h("audio", { preload: "metadata" }); wirePlayback(S, () => schedule(), S.els.audio, () => moveHeads(root, S.els.audio)); wireEngines(api, S, schedule); }
  if (media === "video") wirePlayback(S, () => {}, S.els.video, () => moveHeads(root, S.els.video));
  let source = null, controller = null, removed = false, timer = null;

  const say = (text, warn = false) => { S.status = text; S.statusWarn = warn; render(); };
  const directors = () => downstreamNodes(node).filter((entry) => entry.comfyClass === "NGH3Director").map((entry) => values(entry));
  // Cards earlier in the graph keep their ID and name (a loaded workflow); a pasted card, or a preset loaded here, gives way.
  function keepOwnIdentity(all = false) {
    const nodes = node.graph?._nodes || [], at = nodes.indexOf(node);
    const others = nodes.filter((other, index) => other !== node && (all || at < 0 || index < at)
      && other.widgets?.some((widget) => widget.name === "context_id")).map(values);
    const before = values(node), changes = ownIdentity(before, others);
    if (!Object.keys(changes).length) return;
    applyValues(node, changes);
    if (changes.context_id) assignColor(true);
    say(`Another card already had ${changes.name ? `the name “${before.name}”` : "this ID"}; this card is now ${changes.name ? `“${changes.name}”` : "its own"}.`, true);
  }

  // The linked external input (audio_in and friends) owns the card's file: mirror the origin's file into the
  // media widget, lock the card's file controls, and keep the card's own value for the day the link is removed.
  // Synced on every render; the origin widget's callback re-syncs this card when it changes.
  let unwatch = null;
  const externalLinked = () => Boolean(S.external?.linked);
  function syncExternal() {
    const before = S.external;
    S.external = externalSource(node, media);
    if (!S.external.linked) {
      if (before?.linked) {
        // the link is gone: the card's previous own file comes back and the card is self-contained again
        const saved = node.properties.ng_external_saved?.[media];
        const widget = node.widgets?.find((entry) => entry.name === media);
        if (saved != null && widget && widget.value !== saved) applyValues(node, { [media]: saved });
        if (node.properties.ng_external_saved) delete node.properties.ng_external_saved[media];
        S.status = ""; S.statusWarn = false;
      }
      unwatch?.(); unwatch = null;
      return;
    }
    if (!before?.linked) {
      // freshly linked: the card's own file steps aside before the mirror writes over it. A saved workflow that
      // loads while linked already carries the saved own value in its properties - keep it, don't store the mirror.
      node.properties.ng_external_saved ||= {};
      if (!(media in node.properties.ng_external_saved)) node.properties.ng_external_saved[media] = S.values[media] ?? "";
    }
    unwatch?.(); unwatch = null;
    if (S.external.origin) unwatch = watchOrigin(S.external.origin, media === "text" ? null : media, schedule);
    if (media === "text") {
      // no mirror: the backend receives the linked string itself; an empty one falls back to the card's own note
      if (!S.external.file && !S.busy) { S.status = "The linked text is empty; the card's own note is used until the origin says something."; S.statusWarn = false; }
      return;
    }
    if (S.external.file && S.external.file !== S.values[media]) {
      applyValues(node, { [media]: S.external.file });
      S.values = values(node);
    }
    if (!S.external.file && !String(S.values[media] || "").startsWith("h3-external/") && !S.busy) { S.status = `Linked ${S.external.label || "node"} carries no file - ` + (media === "audio" ? "run the card once so Whisper can hear it." : "pick one on it; a run would stop here."); S.statusWarn = true; }
  }

  function load() {
    S.values = values(node);
    S.opts = contextOptions(node);
    S.content = selectedRoles(node, "content");
    S.usage = selectedRoles(node, "usage")[0] || S.values.usage;
    S.ai = Boolean(contextProvider(node));
    S.analyzed = node.properties.ng_analysis?.fingerprint === analysisFingerprint(media, S.values);
    if (media === "text") node.properties.ng_context_names = knownNames(node.properties.ng_context_names, relatedContexts(node));
    S.mode = directors().find((entry) => entry.mode && entry.mode !== "auto")?.mode || "auto";
    const entries = relatedContexts(node), current = assetSubjects({ widgets: S.values });
    S.owners = { current, choices: assetChoices(entries, current, S.values.context_id) };
    S.speakers = subjectChoices(entries, S.values.speaker || "");
    if (media === "audio") {
      const linked = (name) => (node.inputs || []).some((entry) => entry.widget?.name === name && entry.link != null);
      S.trimLinked = { start: linked("trim_start"), end: linked("trim_end") };
      // DECISION (2026-11-07): the native trim widgets stay visible when linked (inspectable values);
      // the card's lock hint carries the range display, so both show and neither edits past the link.
      // the range the card plays: a linked Trim value wins over the stored one (0 = unset), matching the node's execute
      S.trim = { start: Number(S.values.trim_start) > 0 ? Number(S.values.trim_start) : (S.opts.start_s || 0),
        end: Number(S.values.trim_end) > 0 ? Number(S.values.trim_end) : (S.opts.end_s || 0) };
      // a LINKED 0 means "the file end" to the link's author, but the node's fold reads the stored options first;
      // keep them equal to the linked values so card and run can never disagree
      if ((S.trimLinked.start || S.trimLinked.end) && (S.opts.start_s !== S.trim.start || S.opts.end_s !== S.trim.end))
        setOptions(node, { start_s: S.trim.start, end_s: S.trim.end });
    }
    syncExternal();
    S.files = (node.widgets?.find((widget) => widget.name === media)?.options?.values || []).slice();
    if (S.values[media] && !S.files.includes(S.values[media])) S.files.unshift(S.values[media]);
    S.accent = /^#[0-9a-f]{6}$/i.test(S.values.color || "") ? S.values.color : FAMILY[node.comfyClass].color;
    if (media !== "text") refreshSource();
  }

  // A new file: new preview, and for sound a new waveform.
  function refreshSource() {
    const now = S.values[media] || "";
    if (now === source) return;
    source = now; S.envelope = null; S.waveError = "";
    if (!now) return;
    const url = mediaURL(now, api);
    if (media === "image") S.els.image.src = url;
    if (media === "video") S.els.video.src = url;
    if (media === "audio") S.els.audio.src = url;
    // the waveform: the song on an Audio card, the video's own sound under the range on a Video card
    if (media === "audio" || media === "video") {
      request(api, "/ng/waveform?" + new URLSearchParams({ source: now }))
        .then((data) => { if (source === now) { S.envelope = data; schedule(); } })
        .catch((error) => { if (source === now) { S.waveError = error.message; schedule(); } });
    }
  }

  async function refreshPresets(fresh = false) {
    try { S.presets = await presetNames(media, fresh); } catch (error) { S.presets = []; S.status = error.message; }
    schedule();
  }

  const writeRoles = (type, list) => applyValues(node, { [type + "_roles"]: JSON.stringify(list), [type]: list[0] || values(node)[type] });
  const actions = {
    resize() { fit(); },
    set(name, value, redraw = true) {
      if (media === "video" && name === "video" && value !== values(node).video) setOptions(node, { control_original: "", control_result: "", control_mask: "" });
      applyValues(node, { [name]: value }); node.graph?.change?.(); if (!redraw) S.values = values(node);
    },
    upload() {
      if (externalLinked()) return say("The file is linked; the origin card owns it.", true);
      const widget = node.widgets?.find((entry) => entry.name === "upload" || entry.name === "choose file to upload");
      if (widget?.callback) widget.callback(); else say("Use ComfyUI's file upload for this node type.", true);
    },
    async uploadVideo(file) {
      if (S.busy) return;
      if (externalLinked()) return say("The video is linked; the origin card owns it.", true);
      S.busy = "upload-video"; const before = JSON.stringify(values(node)); say("Uploading video …");
      try {
        const uploaded = await uploadVideo(api, file);
        if (!removed && JSON.stringify(values(node)) === before) { actions.set("video", uploaded); say("Video uploaded."); }
        else say("The card changed during upload. Choose the saved video from input/h3-video.", true);
      } catch (error) { say(error.message, true); }
      finally { S.busy = false; if (!removed) render(); }
    },
    toggleContent(key) { const list = S.content.includes(key) ? S.content.filter((value) => value !== key) : [...S.content, key]; writeRoles("content", list); },
    pickUsage(key) { writeRoles("usage", [key]); },
    setKeep(key, level) { setOptions(node, { preserve: { ...contextOptions(node).preserve, [key]: level } }); },
    setOptions(changes) { setOptions(node, changes); },
    controlKind(kind) {
      if (externalLinked()) return say("The video is linked; control flows need the card's own file.", true);
      const opts = contextOptions(node);
      if (opts.control_original && opts.control_result === values(node).video) applyValues(node, { video: opts.control_original });
      setOptions(node, { control_kind: kind, control_original: "", control_result: "", control_mask: "" });
    },
    controlEngine(engine) { S.depthEngine = engine; render(); },
    openEditor() {
      if (externalLinked()) return say("The picture is linked; the origin card owns it.", true);
      openPictureEditor(node, api, (text, warn) => { if (text) say(text, warn); else render(); });
    },
    setMask(file) {
      if (externalLinked()) return say("The video is linked; the origin card owns it.", true);
      setOptions(node, { control_mask: file, control_result: values(node).video });
    },
    async uploadMask(file) {
      if (S.busy) return;
      if (externalLinked()) return say("The video is linked; the origin card owns it.", true);
      S.busy = "upload-mask"; const before = JSON.stringify(values(node)); say("Uploading mask …");
      try {
        const uploaded = await uploadMask(api, file);
        if (!removed && JSON.stringify(values(node)) === before) { actions.setMask(uploaded); say("Mask uploaded. White marks the region to regenerate."); }
        else say("The card changed during upload. Choose the saved mask from input/h3-control.", true);
      } catch (error) { say(error.message, true); }
      finally { S.busy = false; if (!removed) render(); }
    },
    async generateControl() {
      if (S.busy || !values(node).video) return;
      if (externalLinked()) return say("The video is linked; the origin card owns it.", true);
      S.busy = "control"; say("Generating control video through ComfyUI's queue …");
      const before = JSON.stringify(values(node)), opts = contextOptions(node);
      const source = (opts.control_result === values(node).video && opts.control_original) || values(node).video, kind = opts.control_kind || "canny";
      try {
        const setup = await controlSetup(api, kind, S.depthEngine || "moge2", node.properties.ng_models || {});
        const prefix = globalThis.crypto.randomUUID();
        const saved = await runJob(api, controlGraph(kind, source, { ...setup, prompt: opts.mask_prompt ?? "person" }, prefix), "save");
        if (removed || JSON.stringify(values(node)) !== before) { say("The card changed during generation. Its saved result was left in output/h3-control; the card was not replaced.", true); return; }
        const imported = await request(api, "/ng/control/import", saved);
        const importedSource = kind === "mask" ? imported.source : await videoSource(api, imported.source);
        if (removed || JSON.stringify(values(node)) !== before) return;
        const result = kind === "mask" ? { control_mask: imported.source, control_result: source, control_original: "" }
          : { control_original: source, control_result: importedSource, control_mask: "" };
        applyValues(node, { video: kind === "mask" ? source : importedSource, content: "control", content_roles: JSON.stringify(["control"]),
          context_options: JSON.stringify({ ...opts, ...result, control_kind: kind, video_audio: "ignore" }) });
        node.graph?.change?.(); say("Control video saved. Drag this card onto the Director's Control lane.");
      } catch (error) {
        say(kind === "depth" && error.message.includes("DepthAnything3Net")
          ? "The installed Depth Anything 3 package cannot import DepthAnything3Net. Choose MoGe 2 or repair that package; this card was kept." : error.message, true);
      } finally { S.busy = false; if (!removed) render(); }
    },
    setRange(start, end) {
      // the audio card keeps the range in its Trim widgets (the node folds them into context_options at run time);
      // the stored copy follows so an unlinked drag and a linked value never disagree on the fallback
      if (media === "audio") { applyValues(node, { trim_start: start, trim_end: end }); setOptions(node, { start_s: start, end_s: end }); }
      else setOptions(node, { start_s: start, end_s: end });
    },
    setOwner(id) { applyValues(node, { subject_ids: JSON.stringify(id ? [id] : []) }); node.graph?.change?.(); },
    setListen(key, value) { node.properties.ng_preview_settings[key] = value; S.listen = node.properties.ng_preview_settings; render(); },
    beginPointer() { S.dragging = true; },
    endPointer() { S.dragging = false; if (S.pending) render(); },
    autoColor() { assignColor(true); applyValues(node, { color: colorWidget()?.value || "" }); },
    presetName(value) { S.presetName = value; },
    async savePreset() {
      try {
        const data = await request(api, "/ng/presets", { media, name: S.presetName, widgets: values(node) });
        node.properties.ng_preset_name = S.presetName = data.name;
        say(`Saved preset “${data.name}”. Saving the same name again replaces it.`); refreshPresets(true);
      } catch (error) { say(error.message, true); }
    },
    async loadPreset(name) {
      if (!name) return;
      try {
        const data = await request(api, "/ng/preset?" + new URLSearchParams({ media, name }));
        const widgets = { ...data.widgets };
        if (externalLinked()) delete widgets[media];   // the link owns the file; a preset's file must not overwrite it
        applyValues(node, widgets); node.properties.ng_preset_name = S.presetName = name; say(`Loaded preset “${name}”.`);
        keepOwnIdentity(true);
      } catch (error) { say(error.message, true); }
    },
    async analyze() {
      const provider = contextProvider(node);
      if (!provider || S.busy) return;
      S.busy = "analyze"; S.suggestion = null; controller = new AbortController(); say("");
      const before = JSON.stringify(values(node));
      try {
        const data = await aiAction(app, api, "analyze", { media, widgets: values(node), provider, offer: offerFor(media) }, controller.signal);
        if (JSON.stringify(values(node)) !== before) { say("The card changed while the AI was looking; its answer was discarded. Analyze again.", true); return; }
        const fields = usableFields(media, offeredFields(data.suggestion));
        S.suggestion = Object.keys(fields).length ? { raw: data.suggestion, fields, why: reasonLine(data.suggestion), basis: before, model: provider.model } : null;
        if (!S.suggestion) say("The AI answered, but suggested nothing new.", true);
      } catch (error) { say(error.message === "AI action cancelled." ? "AI cancelled, nothing changed." : `AI failed: ${error.message}`, error.message !== "AI action cancelled."); }
      finally { S.busy = ""; controller = null; render(); }
    },
    accept() {
      const suggestion = S.suggestion; S.suggestion = null;
      if (!suggestion || JSON.stringify(values(node)) !== suggestion.basis) { say("The card changed after the AI answered; analyze again.", true); return; }
      applyValues(node, applyChanges(suggestion.fields, Object.keys(suggestion.fields)));
      node.properties.ng_analysis = { fingerprint: analysisFingerprint(media, values(node)), model: suggestion.model };
      node.graph?.change?.(); say("Accepted the AI suggestion.");
    },
    discard() { S.suggestion = null; render(); },
    async toggleModels() {
      S.modelsOpen = !S.modelsOpen; render();
      if (S.modelsOpen) { S.models = await modelState(api, CARD_MODELS[media]).catch(() => ({})); render(); }
    },
    setModel(purpose, file) { node.properties.ng_models = { ...node.properties.ng_models, [purpose]: file || undefined }; node.graph?.change?.(); render(); },
    // Lyrics: Whisper times the lines that are there (or writes down what it hears); the words stay the user's.
    // Linked with no file behind it, the card sends `linked`: one run saves the sound, then Whisper hears it.
    setEngine(engine) { storeEngine(node, engine); render(); },
    async timeLyrics() {
      if (S.busy) return;
      S.busy = "lyrics"; say("");
      try {
        const data = await request(api, "/ng/lyrics", { source: values(node).audio, lyrics: S.els.lyrics?.value ?? contextOptions(node).lyrics ?? "",
          model: node.properties.ng_models?.whisper || "", engine: node.properties.ng_engine || "", linked: externalLinked() });
        setOptions(node, { lyrics: data.lrc });
        say(`Timed ${data.lines} line(s) (${data.language}, ${deviceNote(data.device)}). Check them while the song plays.`);
      } catch (error) { say(`Lyrics: ${error.message}`, true); }
      finally { S.busy = ""; render(); }
    },
    stampLyric() {
      const editor = S.els.lyrics, text = editor?.value ?? "";
      if (!editor) return;
      const result = stampLine(text, editor.selectionStart ?? 0, S.els.audio.currentTime || 0);
      setOptions(node, { lyrics: result.text }); render();
      queueMicrotask(() => { S.els.lyrics?.focus(); S.els.lyrics?.setSelectionRange(result.caret, result.caret); });
    },
    // Decision 3: the provider's audio model is a second hearing beside Whisper, which stays the one
    // clock. The card's range (or the whole file) goes out; the plain words come back and are shown for
    // review - only acceptTranscript writes the Lyrics, so nothing is ever replaced silently.
    async transcribe() {
      const provider = contextProvider(node);
      if (!provider || S.busy || !values(node).audio) return;
      if (provider.source === "clip") { say("Transcription needs a provider that accepts audio (input_audio); the connected CLIP encoder has no audio input. ⟳ Time with Whisper writes down the words instead.", true); return; }
      S.busy = "transcribe"; S.transcript = null; controller = new AbortController(); say("");
      const before = JSON.stringify(values(node));
      try {
        const data = await request(api, "/ng/transcribe", { media, widgets: values(node), provider,
          start_s: S.trim?.start || 0, end_s: S.trim?.end || 0 }, controller.signal);
        if (JSON.stringify(values(node)) !== before) { say("The card changed while the AI was listening; its answer was discarded. Transcribe again.", true); return; }
        if (!String(data.text || "").trim()) say("The AI heard nothing it could write down.", true);
        else S.transcript = data.text;
      } catch (error) { say(error.message === "AI action cancelled." ? "AI cancelled, nothing changed." : `Transcribe failed: ${error.message}`, error.message !== "AI action cancelled."); }
      finally { S.busy = ""; controller = null; render(); }
    },
    acceptTranscript(keep = false) {
      const text = String(S.els.transcript?.value ?? S.transcript ?? "").replace(/\r\n?/g, "\n").trim();
      S.transcript = null;
      if (!text) { say("The transcript is empty; the lyrics were left alone.", true); return; }
      // "Keep both" appends the hearing below the user's own lines (times kept); the plain
      // accept still replaces, exactly as before. Either way Whisper adds the times afterwards.
      const base = keep ? String(S.els.lyrics?.value ?? contextOptions(node).lyrics ?? "").replace(/\r\n?/g, "\n").trim() : "";
      setOptions(node, { lyrics: base ? `${base}\n${text}` : text });
      say(keep ? "Transcript appended below your lyrics, without times. ⟳ Time with Whisper adds when each line is sung."
                : "Lyrics set from the transcript, without times. ⟳ Time with Whisper adds when each line is sung.");
      render();
    },
    discardTranscript() { S.transcript = null; render(); },
    cancel() { controller?.abort(); },
    setNote(text) { applyValues(node, { text }); S.values = values(node); },
    insert(key) {
      if (externalLinked()) return say("The note is linked; the origin node owns it.", true);
      const editor = S.els.editor, text = values(node).text || "", at = editor?.selectionStart ?? text.length, end = editor?.selectionEnd ?? at, token = `[${key}]`;
      actions.setNote(text.slice(0, at) + token + text.slice(end)); render();
      queueMicrotask(() => { S.els.editor?.focus(); S.els.editor?.setSelectionRange(at, at + token.length); });
    },
    useTemplate(name) {
      if (externalLinked()) return say("The note is linked; the origin node owns it.", true);
      const text = values(node).text || "", body = TEMPLATES[name] || ""; actions.setNote(text.trim() ? `${text}\n${body}` : body); render();
    },
  };

  // Redraw everything; the field being typed in gets its focus and caret back by its place among the fields.
  function render() {
    if (removed) return;
    if (S.dragging) { S.pending = true; return; }
    S.pending = false;
    load();
    const fields = () => [...root.querySelectorAll("input,textarea,select")];
    const active = document.activeElement, before = fields(), index = before.indexOf(active);
    const caret = index >= 0 && typeof active.selectionStart === "number" ? [active.selectionStart, active.selectionEnd] : null;
    root.style.setProperty("--h3c-accent", S.accent);
    // One dark frame for every card; the medium shows in the card itself. ng_context.js colours titles by medium after us.
    node.color = "#192842"; node.bgcolor = "#101a2c";
    root.replaceChildren(...layout(S, actions));
    const after = fields(), next = index >= 0 && after.length === before.length ? after[index] : null;
    if (next && next.tagName === active.tagName) { next.focus({ preventScroll: true }); if (caret && next.setSelectionRange) try { next.setSelectionRange(...caret); } catch {} }
    fit();
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(render, 40); }

  // The node is as tall as its card, so the DOM never spills over the nodes below it. ComfyUI hides DOM widgets of
  // nodes outside the view; a hidden card measures 0, so only a visible card is measured, again whenever it resizes.
  let measured = { image: 800, video: 820, audio: 820, text: 740 }[media];
  const grown = () => node.properties.ng_grow || 0;   // the height a drag gave the growing part (ng_layout.followDraggedHeight)
  function fit() {
    requestAnimationFrame(() => {
      const height = cardContentHeight(root);
      if (removed || !root.isConnected || height < 120) return;
      measured = height;
      const target = (node.computeSize?.()[1] ?? measured) + grown();
      const width = Math.max(node.size?.[0] || 0, WIDTH[media]);
      if (!node.size || Math.abs(node.size[1] - target) > 2 || node.size[0] < WIDTH[media]) node.setSize([width, target]);
      node.setDirtyCanvas?.(true, true);
    });
  }
  const watcher = typeof ResizeObserver === "function" ? new ResizeObserver(() => fit()) : null;
  watcher?.observe(root);
  // dragged taller, the preview (Image, Video), the waveform (Audio) or the note (Text) takes the extra height
  followDraggedHeight(node, (px) => root.style.setProperty("--h3c-grow", `${px}px`), fit);
  pushOnGrowth(node);

  const names = () => (node.widgets || []).filter((widget) => widget !== dom).map((widget) => widget.name);
  const dom = followNodeWidth(node.addDOMWidget("h3c_card", "h3c_card", root, { serialize: false, hideOnZoom: false }));
  // the node's minimum is its content without the grown part, so it can be dragged smaller again
  dom.computeSize = () => [WIDTH[media] - 20, measured - grown()];
  dom.computeLayoutSize = () => ({ minHeight: measured - grown(), maxHeight: measured, minWidth: WIDTH[media] - 20 });
  hideNative(node, names());
  // Upload previews and players arrive after nodeCreated; they are hidden as they come.
  for (const method of ["addWidget", "addCustomWidget"]) {
    const original = node[method];
    if (!original) continue;
    node[method] = function (...args) { const result = original.apply(this, args); hideNative(node, names()); return result; };
  }
  for (const widget of node.widgets || []) {
    if (widget === dom) continue;
    const previous = widget.callback;
    widget.callback = function (...args) { const result = previous?.apply(this, args); schedule(); return result; };
  }
  const configured = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = configured?.apply(this, args);
    repairNullableText(node);
    restoreAssetWidget(node);
    // Old workflows end after ID/colour; browser upload widgets must not fill newly added JSON fields.
    // A pre-trim save restores positionally (Context ID in Trim start); re-seat by name.
    const saved = args[0]?.widgets_values;
    if (Array.isArray(saved) && saved.length <= (media === "audio" ? 9 : 8))
      for (const widget of node.widgets || []) if (["content_roles", "usage_roles", "context_options"].includes(widget.name)) widget.value = "";
    if (media === "audio") reseatPreTrim(node, args[0]);
    for (const widget of node.widgets || []) {
      if (!["content_roles", "usage_roles", "context_options", "subject_ids"].includes(widget.name)) continue;
      if (!widget.value) continue;
      try { JSON.parse(widget.value); } catch { widget.value = ""; }
    }
    if (identity() && !identity().value) identity().value = newContextId();
    assignColor(); delete node.properties.ng_context_expanded;
    queueMicrotask(() => { if (!removed) keepOwnIdentity(); });
    // Saved by the former, narrower Context node: the card needs the room once, and the nodes around it move along.
    if (node.size?.[0] < WIDTH[media] - 20) { noteGrowth(node, [...node.size]); node.setSize([WIDTH[media], Math.max(node.size[1], measured)]); }
    S.presetName = node.properties.ng_preset_name || ""; S.listen = node.properties.ng_preview_settings ||= { volume: 1, fade_in: 0, fade_out: 0 };
    schedule(); return result;
  };
  const connected = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) { const result = connected?.apply(this, args); schedule(); return result; };
  // A renamed Context keeps the @Name mentions in this note.
  const followRenames = () => {
    if (media !== "text") return;
    const now = values(node).text || "", renamed = followNoteRenames(node.properties, now, relatedContexts(node));
    if (renamed !== now) applyValues(node, { text: renamed });
  };
  const outside = () => { followRenames(); schedule(); };
  document.addEventListener("ng-context-changed", outside);
  const stopSound = watchMaterialized(node, media, api, (file) => { applyValues(node, { [media]: file }); say(""); });
  const gone = node.onRemoved;
  node.onRemoved = function (...args) {
    removed = true; unwatch?.(); stopSound?.(); clearTimeout(timer); controller?.abort(); watcher?.disconnect(); S.els.audio?.pause(); S.els.video?.pause?.(); S.els.original?.pause?.(); S.els.mask?.pause?.();
    document.removeEventListener("ng-context-changed", outside); return gone?.apply(this, args);
  };
  // A file dragged onto the card goes through ComfyUI's own upload of this node (the same as dropping it on a Load Image).
  if (media !== "text") dropTarget(root, async (event) => {
    if (externalLinked()) return say("The file is linked; drop it on the origin card instead.", true);
    if (!(node.onDragOver?.(event) && await node.onDragDrop?.(event))) say(`Drop a ${media} file here.`, true);
  });
  // The card shows its own picture or video. ComfyUI's preview of the upload widget would come below it, and in Vue mode
  // (measured 2026-10-25: 228 px) outside the node's size, over the node below.
  node.hideOutputImages = true;
  node.setSize([WIDTH[media], node.size?.[1] || 600]);
  refreshPresets(); render();
}


app.registerExtension({ name: "mmxd.ng.context_cards", nodeCreated(node) { if (MEDIA[node.comfyClass]) attach(node, MEDIA[node.comfyClass]); } });
