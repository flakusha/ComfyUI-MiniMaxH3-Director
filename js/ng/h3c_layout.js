// How a Context card is arranged: which boxes each medium shows and in what order. The card
// (h3c_card.js) owns the state and redraws on change; this module only places the pieces.
import { h } from "./h3s_dom.js";
import { box, topRow, modelsBox, contentBox, usageBox, keepBox, ownerBox, descriptionBox, suggestionBox, presetRow, statusLine } from "./h3c_parts.js";
import { imageBlock, videoBlock, videoSoundBox, audioBlock, listenBox } from "./h3c_media.js";
import { controlBox } from "./h3c_control.js";
import { textBody } from "./h3c_text.js";
import { lyricsBox } from "./h3c_lyrics.js";

// The node is as wide as its card; narrower and the two-column rows would fold.
export const WIDTH = { image: 660, video: 660, audio: 700, text: 780 };

function speakerBox(S, actions) {
  const choose = h("select", { class: "h3c-in", "aria-label": "Whose voice", onchange: (event) => actions.set("speaker", event.target.value) },
    h("option", { value: "" }, "— nobody (narrator)"), S.speakers.map((entry) => h("option", { value: entry.id }, entry.name)));
  choose.value = S.values.speaker || "";
  return box("Whose voice", choose, h("span", { class: "h3c-hint" }, "For “Voice”, and for who sings this card's Lyrics. Who speaks a line in a shot is set in the Studio."));
}

export function layout(S, actions) {
  const two = (...children) => h("div", { class: "h3c-grid", style: { gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" } }, ...children);
  const parts = [topRow(S, actions), modelsBox(S, actions)];
  if (S.media === "image") parts.push(h("div", { class: "h3c-grid", style: { gridTemplateColumns: "minmax(0,1.15fr) minmax(0,1fr)" } }, imageBlock(S), contentBox(S, actions)),
    suggestionBox(S, actions), h("div", { class: "h3c-grid", style: { gridTemplateColumns: "minmax(0,1fr) minmax(0,1.25fr)" } }, usageBox(S, actions), keepBox(S, actions)),
    two(ownerBox(S, actions, `Independent: this picture is a figure of its own; write @${S.values.name || "Name"} in the Studio.`), descriptionBox(S, actions)));
  if (S.media === "video") parts.push(videoBlock(S, actions), two(contentBox(S, actions), usageBox(S, actions)), controlBox(S, actions), suggestionBox(S, actions),
    S.usage === "control" ? null : two(videoSoundBox(S, actions), ownerBox(S, actions, "E.g. a walk cycle for a figure.")), descriptionBox(S, actions));
  if (S.media === "audio") parts.push(audioBlock(S, actions), two(contentBox(S, actions), usageBox(S, actions)), suggestionBox(S, actions),
    two(speakerBox(S, actions), listenBox(S, actions)), lyricsBox(S, actions), descriptionBox(S, actions));
  if (S.media === "text") parts.push(...textBody(S, actions));
  parts.push(presetRow(S, actions), statusLine(S));
  return parts.filter(Boolean);
}
