// External inputs: the four Context cards accept a typed socket (audio_in, video_in, image_in, text_in) whose
// connection is the authority over the card's own file. The render pipeline needs a file NAME and the payload
// (AUDIO/IMAGE tensors) carries none - so the card mirrors the origin's file widget into its own media widget,
// which every consumer (previews, waveform, lyrics, fingerprints, the Studio panel) already reads, and locks its
// file controls while the link holds. Text needs no mirror: the backend receives the string itself.
import { upstreamNode } from "./ng_read.js";
import { values } from "./ng_widgets.js";

// Schema ids must be unique next to the widgets they drive (a second input named like the widget is rejected),
// so the ports carry the `_in` suffix and only the display label shows the plain medium.
export const EXTERNAL_PORT = { image: "image_in", video: "video_in", audio: "audio_in", text: "text_in" };

export function externalInput(node, media) {
  return (node.inputs || []).find((entry) => entry.name === EXTERNAL_PORT[media] && entry.link != null) || null;
}

// The origin's file lives in the widget named like the medium (LoadAudio's "audio", LoadImage's "image",
// LoadVideo's "video" - and a Context card chained in). A text origin is any node whose widgets hold a plain
// string (Primitive String and friends); combos are skipped, so a file list never reads as text.
export function originWidget(origin, media) {
  const widgets = origin?.widgets || [];
  if (media) return widgets.find((widget) => widget.name === media) || null;
  return widgets.find((widget) => (widget.name === "string" || widget.name === "text") && typeof widget.value === "string")
    || widgets.find((widget) => typeof widget.value === "string" && !Array.isArray(widget.options?.values)) || null;
}

// What the connection points at: `{ linked: false }` unlinked, `{ linked: true, origin, label, file }` linked
// (file "" when the origin carries nothing to mirror, e.g. decoded audio - the card warns and a run stops).
export function externalSource(node, media) {
  const input = externalInput(node, media);
  if (!input) return { linked: false, origin: null, label: "", file: "" };
  const origin = upstreamNode(node.graph, input);
  if (!origin) return { linked: true, origin: null, label: "", file: "" };
  const widget = originWidget(origin, media === "text" ? null : media);
  const label = origin.title || origin.comfyClass || origin.type || "node";
  return { linked: true, origin, label, file: widget && typeof widget.value === "string" ? String(widget.value) : "" };
}

// Changing the origin's file must re-sync this card even when no graph event reaches it: the origin widget's
// callback is wrapped once and every watching card subscribes. Returns the unsubscribe call.
export function watchOrigin(origin, media, notify) {
  const widget = originWidget(origin, media);
  if (!widget) return null;
  widget.ngExternalWatch ||= new Set();
  if (!widget.ngExternalHooked) {
    const previous = widget.callback;
    widget.ngExternalHooked = true;
    widget.callback = function (...args) {
      const result = previous?.apply(this, args);
      for (const hook of widget.ngExternalWatch) hook();
      return result;
    };
  }
  widget.ngExternalWatch.add(notify);
  return () => widget.ngExternalWatch.delete(notify);
}

// A run saved the card's linked sound to input/h3-external/ and pushed its name: the name is content-hashed,
// so the card cannot learn it any other way. Keep it as the card's file while the widget is still empty, so
// Whisper (and the next run) reads it; an occupied widget (mirror, user file) is never overwritten.
// Audio-only: other media always mirror a file already. Returns the unsubscribe call.
export function watchMaterialized(node, media, api, apply) {
  const savedSound = ({ detail = {} }) => {
    const file = typeof detail.audio === "string" && detail.audio;
    if (media === "audio" && file && String(detail.node_id ?? "") === String(node.id) && !values(node)[media]) apply(file);
  };
  api.addEventListener?.("minimax_h3_ng_audio", savedSound);
  return () => api.removeEventListener?.("minimax_h3_ng_audio", savedSound);
}
