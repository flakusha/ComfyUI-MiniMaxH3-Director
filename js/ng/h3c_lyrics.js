// Lyrics of a song card: LRC lines "[mm:ss.xx] words" in seconds of the file. Where the card plays exactly (lock or
// copy), the Director puts each line into the render part it is sung in. Whisper fills in the times; Stamp sets one by ear.
import { h } from "./h3s_dom.js";

const STAMP = /^\s*\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]\s*/;

export function stamp(seconds) {
  const m = Math.floor(Math.max(0, seconds) / 60), s = Math.max(0, seconds) - m * 60;
  return `[${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}]`;
}

// The line at the caret gets the player's time, replacing an earlier stamp; the next line is selected for the next press.
export function stampLine(text, caret, seconds) {
  const start = text.lastIndexOf("\n", Math.max(0, caret - 1)) + 1, cut = text.indexOf("\n", caret), end = cut < 0 ? text.length : cut;
  const line = `${stamp(seconds)} ${text.slice(start, end).replace(STAMP, "")}`.trimEnd();
  const next = text.slice(0, start) + line + text.slice(end);
  return { text: next, caret: Math.min(next.length, start + line.length + 1) };
}

export function timedCount(text) {
  return String(text || "").split("\n").filter((line) => STAMP.test(line) && line.replace(STAMP, "").trim()).length;
}

// Where Whisper ran: an older backend (or a non-200 path) answers without a device, which reads as CPU.
export function deviceNote(device) {
  const name = String(device || "");
  return name.startsWith("cpu") || !name ? "CPU while a render runs" : "GPU";
}

// One Lyrics subtab: the card's own words (LRC/Whisper/Stamp) plus the AI transcribe offer.
// Transcription belongs here, not in a second box: it starts from the existing lyrics and time
// them, so operating on the same text keeps "what Whisper timed" and "what the AI heard" together.
export function lyricsBox(S, actions) {
  const text = S.opts.lyrics || "", timed = timedCount(text);
  const lines = String(text).split("\n").filter((line) => line.replace(STAMP, "").trim() && !/^\s*\[la:/i.test(line)).length;
  const editor = h("textarea", { class: "h3c-in", rows: Math.min(12, Math.max(4, lines + 1)), value: text, spellcheck: false, "aria-label": "Lyrics",
    placeholder: "Paste the lyrics, one line per sung line. ⟳ Time with Whisper adds when each line is sung.",
    onchange: (event) => actions.setOptions({ lyrics: event.target.value }) });
  S.els.lyrics = editor;
  const sung = ["lock", "copy"].includes(S.usage);
  const note = !text.trim() ? "Optional. Only for a song or speech whose words should be sung or spoken in the video."
    : !sung ? "Sung where this song plays exactly: Lock or Copy here, or a Lock or Copy clip on a Studio audio lane."
    : `${timed} of ${lines} line(s) timed; each render part gets the lines sung in it.`;
  // The file widget gate covers both sources: unlinked it holds the chosen file, linked the mirror of the
  // origin's file (ng_external.js) - so the buttons stay lit exactly when Whisper has something to hear.
  const heard = S.values.audio || "";
  const transcript = transcriptRow(S, actions);
  return h("div", { class: "h3c-box" },
    h("div", { class: "h3c-cap" }, "Lyrics", h("span", { class: "h3c-grow" }),
      h("button", { class: "h3c-btn ai", disabled: !heard || Boolean(S.busy),
        title: "Let the connected AI provider write down the words of this sound (plain text, no times). Your lyrics stay until you accept.",
        onclick: () => actions.transcribe() }, S.busy === "transcribe" ? "Transcribing …" : "✦ Transcribe with AI"),
      h("button", { class: "h3c-btn", disabled: !heard || Boolean(S.busy), title: "Time each line with the local Whisper model; without lines it writes down what it hears",
        onclick: () => actions.timeLyrics() }, S.busy === "lyrics" ? "Listening …" : "⟳ Time with Whisper"),
      h("button", { class: "h3c-btn", disabled: !heard || Boolean(S.busy), title: "Play the song and press at the start of the selected line",
        onclick: () => actions.stampLyric() }, "Stamp line")),
    editor, transcript, h("span", { class: "h3c-hint" }, note));
}

// The transcribe offer inside the same Lyrics box: idle hint, running cancel, review editor with
// keep/accept/discard. Nothing writes Lyrics except accept/keep; the review editor drafts on a copy.
function transcriptRow(S, actions) {
  if (S.busy === "transcribe")
    return h("div", { class: "h3c-row" },
      h("button", { class: "h3c-btn", onclick: () => actions.cancel() }, "Cancel"));
  if (!S.transcript) {
    const existing = String(S.opts.lyrics || "").split("\n").filter((line) => line.trim()).length;
    // one idle hint: the keeping-your-lines clause lives here, not in a second stacked note.
    return h("span", { class: "h3c-hint" },
      existing
        ? `“Transcribe with AI” drafts fresh lines from the selected range (or the whole file), keeping your ${existing} lyric line(s) until you accept.`
        : "“Transcribe with AI” drafts the selected range (or the whole file) as plain lines; accept puts them into Lyrics without times, then ⟳ Time with Whisper adds when each line is sung.");
  }
  const existing = String(S.opts.lyrics || "").split("\n").filter((line) => line.trim()).length;
  const editor = h("textarea", { class: "h3c-in", rows: Math.min(12, Math.max(4, String(S.transcript).split("\n").length + 1)),
    value: S.transcript, spellcheck: false, "aria-label": "AI transcript, one line per sung or spoken line" });
  S.els.transcript = editor;
  return h("div", { class: "h3c-suggest" },
    h("span", { class: "h3c-lbl", style: { color: "#C9BBFF" } },
      `AI transcript — nothing changes until you accept${existing ? `; it would replace the ${existing} lyric line(s) on the card` : ""}`),
    editor,
    h("div", { class: "h3c-row" },
      existing
        ? h("button", { class: "h3c-btn", title: "Append the transcript below the existing lyrics instead of replacing them",
          onclick: () => actions.acceptTranscript(true) }, "Keep both") : null,
      h("button", { class: "h3c-btn ai", onclick: () => actions.acceptTranscript() }, "Use as lyrics"),
      h("button", { class: "h3c-btn", onclick: () => actions.discardTranscript() }, "Discard")));
}
