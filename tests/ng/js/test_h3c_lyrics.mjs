// The Audio card's lyrics: stamping a line by ear, and what the box says and does for a song card.
import assert from "node:assert/strict";
import { installDom } from "./fake_dom.mjs";

installDom();
const { stamp, stampLine, timedCount, lyricsBox, deviceNote } = await import("../../../js/ng/h3c_lyrics.js");

// Stamp replaces an earlier time on the line at the caret and moves on to the next line.
{
  assert.equal(stamp(75.5), "[01:15.50]");
  const first = stampLine("My master\nI still wait", 3, 1.234);
  assert.equal(first.text, "[00:01.23] My master\nI still wait");
  assert.equal(first.caret, "[00:01.23] My master\n".length);
  assert.equal(stampLine(first.text, 2, 2).text, "[00:02.00] My master\nI still wait", "a second press replaces the time");
  assert.equal(timedCount("[la:English]\n[00:01.00] a\n[00:03.00]\nb"), 1, "a bare stamp is a pause, an unstamped line is untimed");
}

function recorder() {
  const calls = [];
  return { calls, actions: new Proxy({}, { get: (_t, name) => (...args) => { calls.push([name, ...args]); } }) };
}
const state = (extra = {}) => ({ media: "audio", values: { audio: "song.mp3" }, opts: { lyrics: "" }, usage: "lock", busy: "", els: {}, ...extra });

// Whisper, Stamp and Transcribe are buttons on the one Lyrics subtab; the hint says when lines reach the prompt.
{
  const { actions, calls } = recorder();
  const view = lyricsBox(state({ opts: { lyrics: "[00:01.00] My master\n[00:04.00] I still wait\nThe halls" } }), actions);
  view.button("⟳ Time with Whisper").onclick();
  assert.deepEqual(calls.at(-1), ["timeLyrics"]);
  view.button("Stamp line").onclick();
  assert.deepEqual(calls.at(-1), ["stampLyric"]);
  view.button("✦ Transcribe with AI").onclick();
  assert.deepEqual(calls.at(-1), ["transcribe"]);
  assert.ok(view.innerText.includes("2 of 3 line(s) timed"), view.innerText);
  assert.ok(lyricsBox(state({ opts: { lyrics: "[00:01.00] a\n[00:03.00]\n[00:05.00] b\n[00:07.00]" } }), actions).innerText.includes("2 of 2 line(s)"),
    "a pause stamp is not a line");
  assert.ok(lyricsBox(state({ opts: { lyrics: "My master\nI still wait" } }), actions).innerText.includes(
    "0 of 2 line(s) timed; each render part gets the lines sung in it. Untimed lines are skipped; press \u27F3 Time with Whisper."), "untimed lock lines point at Whisper");
  assert.ok(!lyricsBox(state({ usage: "music_style", opts: { lyrics: "My master" } }), actions).innerText.includes("untimed lines are skipped"),
    "no timing nudge off lock/copy");
  assert.ok(lyricsBox(state({ usage: "music_style", opts: { lyrics: "[00:01.00] x" } }), actions).innerText.includes("Lock or Copy clip on a Studio audio lane"));
  assert.equal(lyricsBox(state({ busy: "lyrics" }), actions).button("Listening …").disabled, true);
  assert.equal(lyricsBox(state({ values: { audio: "" } }), actions).button("Stamp line").disabled, true);
}

// The button gate reads the file widget for Transcribe and Stamp; Whisper alone stays lit for a linked
// sound with no file behind it (a decoded sound): the mirror holds the origin's file when there is one
// (ng_external.js), and one run materializes the rest (input/h3-external/) - the route names that run.
{
  const { actions } = recorder();
  assert.equal(lyricsBox(state({ values: { audio: "song.wav" } }), actions).button("⟳ Time with Whisper").disabled, false,
    "the mirrored file keeps Whisper lit");
  const linked = { linked: true, origin: null, label: "node", file: "" };
  assert.equal(lyricsBox(state({ values: { audio: "" }, external: linked }), actions).button("⟳ Time with Whisper").disabled, false,
    "a linked sound keeps Whisper lit until one run saves it");
  assert.equal(lyricsBox(state({ values: { audio: "" }, external: linked }), actions).button("Stamp line").disabled, true,
    "Stamp needs the file itself to play");
  assert.equal(lyricsBox(state({ values: { audio: "" }, external: linked }), actions).button("✦ Transcribe with AI").disabled, true,
    "Transcribe sends the widget, so it needs the file itself");
  assert.equal(lyricsBox(state({ values: { audio: "" } }), actions).button("⟳ Time with Whisper").disabled, true,
    "unlinked with no file leaves Whisper dark");
}

// One action at a time: a busy box disables Transcribe, Whisper and Stamp together (a second
// request would orphan the first and Cancel would no longer cancel it).
{
  const { actions } = recorder();
  for (const busy of ["transcribe", "lyrics", "analyze"]) {
    const view = lyricsBox(state({ busy }), actions);
    assert.equal(view.button(busy === "transcribe" ? "Transcribing …" : "✦ Transcribe with AI").disabled, true, `transcribe during ${busy}`);
    assert.equal(view.button(busy === "lyrics" ? "Listening …" : "⟳ Time with Whisper").disabled, true, `whisper during ${busy}`);
    assert.equal(view.button("Stamp line").disabled, true, `stamp during ${busy}`);
  }
}

// A device-less answer still reads: older backends and non-200 paths carry no device.
assert.equal(deviceNote(undefined), "CPU while a render runs");
assert.equal(deviceNote("cpu"), "CPU while a render runs");
assert.equal(deviceNote("cuda:0"), "GPU");

// One Lyrics subtab: the transcript review lives on the same text, with Keep both / Use as lyrics / Discard.
{
  const { actions, calls } = recorder();
  const idle = lyricsBox(state({ opts: { lyrics: "My master" } }), actions);
  assert.ok(idle.innerText.includes("keeping your 1 lyric line(s)"), idle.innerText);
  const review = lyricsBox(state({ opts: { lyrics: "My master" }, transcript: "I still wait" }), actions);
  assert.ok(review.innerText.includes("replace the 1 lyric line(s)"), review.innerText);
  review.button("Keep both").onclick();
  assert.deepEqual(calls.at(-1), ["acceptTranscript", true]);
  review.button("Use as lyrics").onclick();
  assert.deepEqual(calls.at(-1), ["acceptTranscript"]);
  review.button("Discard").onclick();
  assert.deepEqual(calls.at(-1), ["discardTranscript"]);
  const busy = lyricsBox(state({ busy: "transcribe" }), actions);
  busy.button("Cancel").onclick();
  assert.deepEqual(calls.at(-1), ["cancel"]);
}

console.log("h3c lyrics ok");
