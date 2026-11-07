// "Transcribe with AI" on an Audio card (decision 3, 2026-11-07): the AI Provider's audio model writes
// down what it hears as plain lines - the trimmed range the card plays, or the whole file. The words are
// shown here for review; only Accept touches the card's Lyrics, and Whisper ("⟳ Time with Whisper")
// stays the one clock that adds times. A separate module on purpose: the card file is near its line
// budget, and this box is audio-only.
import { h } from "./h3s_dom.js";

export function transcribeBox(S, actions) {
  const run = [h("button", { class: "h3c-btn ai", disabled: !S.values.audio || Boolean(S.busy),
      title: "Let the connected AI provider write down the words of this sound (plain text, no times). You accept before anything changes.",
      onclick: () => actions.transcribe() }, S.busy === "transcribe" ? "Transcribing …" : "✦ Transcribe with AI"),
    S.busy === "transcribe" ? h("button", { class: "h3c-btn", onclick: () => actions.cancel() }, "Cancel") : null];
  if (!S.transcript) {
    return h("div", { class: "h3c-box" }, h("div", { class: "h3c-cap" }, "Transcribe", h("span", { class: "h3c-grow" }), ...run),
      h("span", { class: "h3c-hint" }, "Optional. The AI's hearing of the selected range (or the whole file), as plain lines. Accept puts them into Lyrics without times; ⟳ Time with Whisper then adds when each line is sung."));
  }
  const existing = String(S.opts.lyrics || "").split("\n").filter((line) => line.trim()).length;
  const editor = h("textarea", { class: "h3c-in", rows: Math.min(12, Math.max(4, String(S.transcript).split("\n").length + 1)),
    value: S.transcript, spellcheck: false, "aria-label": "AI transcript, one line per sung or spoken line" });
  S.els.transcript = editor;
  return h("div", { class: "h3c-box h3c-suggest" },
    h("span", { class: "h3c-lbl", style: { color: "#C9BBFF" } },
      `AI transcript — nothing changes until you accept${existing ? `; it would replace the ${existing} lyric line(s) on the card` : ""}`),
    editor,
    h("div", { class: "h3c-row" },
      h("button", { class: "h3c-btn ai", onclick: () => actions.acceptTranscript() }, "Use as lyrics"),
      h("button", { class: "h3c-btn", onclick: () => actions.discardTranscript() }, "Discard")));
}
