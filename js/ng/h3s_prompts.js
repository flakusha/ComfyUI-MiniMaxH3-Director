// Prompts per render part: exactly what goes to the sampler, and each part's own prompt. The AI writes a part out from the
// compiled one (checked on the server: cuts, sung lines and labels unchanged), or the user edits it; either is kept
// with the key of the compiled prompt it came from, and an older one is not used after the plan changes. A shot's own
// text is kept per share instead, so a re-time that keeps the shot keeps its prose.
import { h, icon } from "./h3s_dom.js";
import { fmt, partColor, chunkBudget, partPromptAt, partPromptLegacyAt } from "./h3s_model.js";
import { issueLine } from "./h3s_issues.js";

const SOURCE = { compiled: ["compiled", "h3s-muted"], ai: ["AI-written", ""], user: ["edited", ""], manual: ["manual prompt", "h3s-warn"] };

// A part has its own prompt, or one of its shot shares does, but the plan compiled it again: the plan changed since,
// and the compiled text is used. A legacy index-keyed part prompt (no story_start_f) is always outdated: the plan it
// was written from cannot be recognised across re-times, so it shows the outdated badge and is never silently applied.
export function outdated(S, chunk) {
  if (partPromptLegacyAt(S.partPrompts || [], chunk)) return true;
  if (partPromptAt(S.partPrompts || [], chunk) && chunk.prompt_source === "compiled") return true;
  return (chunk.shares || []).some((share) => (S.shotPrompts || []).some((entry) => entry.shot_id === share.shot_id && entry.basis !== share.basis));
}

// One shot's compiled share of a part: the "Made from" text of that shot, what the sampler uses without an own share.
function shareExcerpt(chunk, shotId) {
  const text = ((chunk.sources || []).find((entry) => entry.kind === "shot" && entry.id === shotId)?.text || "").trim();
  return text && text !== "(no text)" ? text : "";
}

function shareEntry(S, shotId) {
  return (S.shotPrompts || []).find((entry) => entry.shot_id === shotId);
}

// This share's state: no own text (compiled), a stale own text (outdated), or whose it is. Without a basis from the
// preview the staleness cannot be judged, so an entry simply counts as its author's.
function shareState(S, share) {
  const entry = shareEntry(S, share.shot_id);
  if (!entry || typeof share.basis !== "string") return entry ? entry.by : "compiled";
  return entry.basis !== share.basis ? "outdated" : entry.by;
}

const SHARE_LABEL = { compiled: ["compiled", "h3s-muted"], ai: ["AI-written", ""], user: ["edited", ""], outdated: ["outdated", "h3s-warn"] };

// The part badge aggregates its shares: every share's own text, only some (mixed), or how many went outdated.
function badgeFor(S, chunk) {
  if (chunk.prompt_source === "manual") return SOURCE.manual;
  const shares = chunk.shares || [], legacy = partPromptAt(S.partPrompts || [], chunk) || partPromptLegacyAt(S.partPrompts || [], chunk);
  const covered = shares.some((share) => shareEntry(S, share.shot_id));
  if (!shares.length || (!covered && legacy)) {
    const [label, tone] = SOURCE[chunk.prompt_source] || SOURCE.compiled;
    return outdated(S, chunk) ? ["own prompt outdated", "h3s-warn"] : [label, tone];
  }
  const states = shares.map((share) => shareState(S, share));
  const bad = states.filter((state) => state === "outdated").length;
  if (bad) return [`${bad} outdated`, "h3s-warn"];
  const own = states.filter((state) => state !== "compiled");
  if (!own.length) return SOURCE.compiled;
  if (own.length < states.length) return ["mixed", ""];
  return own.every((state) => state === "ai") ? SOURCE.ai : SOURCE.user;
}

// What this part's prompt is made from (ng/core/prompt_sources.py): one chip per shot, card, Look, rule and setting; hover shows
// its text, a shot chip selects the shot, the manual prompt opens Settings.
const KINDS = { manual: ["#F2A65A", "Manual"], shot: ["#6CB6FF", "Shot"], look: ["#C9A2FF", "Look"], rule: ["#F7D774", "Rule"],
  picture: ["#5BD08A", "Picture"], video: ["#FF8A8A", "Video"], audio: ["#B38CF5", "Audio"], setting: ["#9AA6B8", "Setting"] };

export function sourcesRow(S, actions, chunk) {
  const sources = chunk.sources || [];
  if (!sources.length) return null;
  return h("div", { class: "h3s-row", style: { padding: "6px 12px 0", gap: "6px" } }, h("span", { class: "h3s-lbl" }, "Made from"),
    sources.map((entry) => {
      const empty = entry.kind === "shot" && entry.text === "(no text)";
      const [color, kind] = empty ? ["#F2A65A", "Shot without text"] : KINDS[entry.kind] || KINDS.setting;
      const label = entry.kind === "shot" && entry.id && actions.shotLabel ? actions.shotLabel(entry.id) : entry.label;
      const click = entry.kind === "shot" && entry.id ? () => actions.select("shot", entry.id) : entry.kind === "manual" ? () => actions.toggle("settingsOpen") : null;
      return h(click ? "button" : "span", { class: "h3s-chip", title: `${kind}: ${entry.text}`, onclick: click,
        style: { border: `1px solid ${color}`, color, background: "transparent", cursor: click ? "pointer" : "help" } }, empty ? `${label}: no text` : label);
    }));
}

// One share of a part: the shot's chip, what the plan compiled for it, and its own text with Edit/Save/Reset.
// A share without a basis yet (the backend predates shares, or the shot is new) edits against the excerpt.
function shareRow(S, actions, chunk, share, compiled) {
  const label = actions.shotLabel ? actions.shotLabel(share.shot_id) : "Shot";
  const own = shareEntry(S, share.shot_id);
  const state = shareState(S, share);
  const [stateLabel, stateTone] = SHARE_LABEL[state] || SHARE_LABEL.compiled;
  const editing = S.shareEdit?.shotId === share.shot_id && S.shareEdit?.index === chunk.index;
  const entry = (chunk.sources || []).find((item) => item.kind === "shot" && item.id === share.shot_id);
  const chip = h("button", { class: "h3s-chip", title: `${label}: ${entry?.text || "(no text)"}`, onclick: () => actions.select("shot", share.shot_id),
    style: { border: "1px solid #6CB6FF", color: "#6CB6FF", background: "transparent", cursor: "pointer" } }, label);
  const head = h("div", { class: "h3s-row", style: { padding: "6px 12px 0" } }, chip,
    h("span", { class: `h3s-chip ${stateTone}`, title: state === "outdated" ? "The plan changed after this share's own text was written; the compiled share is used." : "" }, stateLabel),
    h("span", { class: "h3s-grow" }),
    own ? h("button", { class: "h3s-btn ghost small", onclick: () => actions.resetShare(chunk.index, share.shot_id), title: "Use the compiled share again" }, "Reset") : null);
  const body = editing
    ? h("div", { style: { padding: "8px 12px", display: "grid", gap: "6px" } },
      h("textarea", { class: "h3s-in", rows: 6, value: S.shareEdit.text, "aria-label": `Share of ${label} in part ${chunk.index + 1}`,
        oninput: (event) => { S.shareEdit.text = event.target.value; } }),
      h("div", { class: "h3s-row" }, h("span", { class: "h3s-lbl" }, "Your text replaces this shot's share as written, until the plan of this share changes."),
        h("span", { class: "h3s-grow" }), h("button", { class: "h3s-btn ai", onclick: () => actions.saveShare(chunk.index, share.shot_id) }, "Save"),
        h("button", { class: "h3s-btn ghost", onclick: () => actions.editShare(null) }, "Cancel")))
    : own
      ? [h("span", { class: "h3s-lbl", style: { padding: "6px 12px 0" } }, state === "outdated" ? "Compiled share (own text outdated)" : "Own text"),
        h("pre", { style: { margin: "0 12px" } }, state === "outdated" ? compiled : own.text),
        own.text.trim()
          ? h("div", { class: "h3s-row", style: { padding: "0 12px" } }, h("span", { class: "h3s-grow" }),
            h("button", { class: "h3s-btn small", disabled: editing, onclick: () => actions.editShare(chunk.index, share.shot_id, own.text) }, "Edit"))
          : null]
      : [h("span", { class: "h3s-lbl", style: { padding: "6px 12px 0" } }, "Compiled share"),
        h("pre", { style: { margin: "0 12px" } }, compiled || "(no text)"),
        h("div", { class: "h3s-row", style: { padding: "0 12px" } }, h("span", { class: "h3s-grow" }),
          h("button", { class: "h3s-btn small", disabled: editing, onclick: () => actions.editShare(chunk.index, share.shot_id, compiled) }, "Edit"))];
  return h("div", { style: { display: "flex", flexDirection: "column", gap: "2px" } }, head, ...[body].flat().filter(Boolean));
}

function partView(S, actions, chunk) {
  const [color, bg] = partColor(chunk.index), use = chunkBudget(chunk), i = chunk.index;
  const [label, tone] = badgeFor(S, chunk);
  const draft = S.partDrafts?.[i], editing = S.partEdit?.index === i, busy = S.busy === `write${i}`;
  const members = chunk.shares || chunk.shots || [];
  const own = partPromptAt(S.partPrompts || [], chunk) || partPromptLegacyAt(S.partPrompts || [], chunk)
    || (S.shotPrompts || []).some((entry) => members.some((share) => share.shot_id === entry.shot_id));
  const tools = chunk.prompt_source === "manual" ? [] : [
    h("button", { class: "h3s-btn ai small", disabled: !S.ai || Boolean(S.busy), onclick: () => actions.writePart(i),
      title: S.ai ? "The AI writes this part's prompt out in full; cuts, sung lines and labels stay as planned" : "Connect an H3 AI Loader to the Connector for this." },
      icon("spark"), busy ? "Writing …" : "Write with AI"),
    h("button", { class: "h3s-btn small", disabled: editing, onclick: () => actions.editPart(i, chunk.prompt) }, "Edit"),
    own ? h("button", { class: "h3s-btn ghost small", onclick: () => actions.resetPart(i), title: "Use the compiled prompt again" }, "Reset") : null];
  const shares = chunk.shares?.length ? chunk.shares : (chunk.shots || []).map((entry) => ({ shot_id: entry.shot_id }));
  return h("div", { class: "h3s-prompt", style: { borderColor: color } },
    h("div", { class: "h3s-prompt-head", style: { background: bg } }, h("span", { class: "h3s-swatch", style: { background: color, width: "10px", height: "10px" } }),
      h("strong", {}, `${chunk.picture ? "Picture" : "Part"} ${i + 1}`), h("span", { class: "h3s-muted", style: { fontSize: "12px" } }, `${fmt(chunk.start_s)} – ${fmt(chunk.end_s)}`),
      h("span", { class: `h3s-chip ${tone}`, title: "What the shares of this part's shots carry: own text, or the compiled text." }, label),
      h("span", { class: "h3s-grow" }), ...tools, h("span", { class: "h3s-lbl" }, chunk.mode.toUpperCase())),
    sourcesRow(S, actions, chunk),
    h("span", { class: "h3s-lbl", style: { padding: "6px 12px 0" }, title: "Reference slots H3 gets in this part (the model's limits)" },
      `Images ${use.image}/9 · Video ${use.video}/3 · Audio ${use.audio}/3 · total ${use.total}/12`),
    shares.map((share) => shareRow(S, actions, chunk, share, shareExcerpt(chunk, share.shot_id))),
    draft ? h("div", { class: "h3s-banner", style: { margin: "8px 12px", flexDirection: "column", alignItems: "stretch" } },
      draft.problems.length
        ? h("span", {}, `The AI's version was not used: ${draft.problems.join("; ")}. Nothing changed; write it again or edit by hand.`)
        : h("div", {}, draft.unchanged ? h("span", { class: "h3s-lbl" }, "The AI kept the compiled prompt as it is; write again or edit by hand for more detail.") : null,
          h("pre", { style: { margin: 0 } }, draft.text)),
      h("div", { class: "h3s-row" }, h("span", { class: "h3s-grow" }),
        draft.problems.length ? null : h("button", { class: "h3s-btn ai", onclick: () => actions.acceptPart(i) }, "Accept"),
        h("button", { class: "h3s-btn ghost", onclick: () => actions.dropPart(i) }, draft.problems.length ? "Dismiss" : "Discard"))) : null,
    editing ? h("div", { style: { padding: "8px 12px", display: "grid", gap: "6px" } },
      h("textarea", { class: "h3s-in", rows: 14, value: S.partEdit.text, "aria-label": `Prompt of part ${i + 1}`,
        oninput: (event) => { S.partEdit.text = event.target.value; } }),
      h("div", { class: "h3s-row" }, h("span", { class: "h3s-lbl" }, "Your text goes to the sampler as written, until the plan of this part changes."),
        h("span", { class: "h3s-grow" }), h("button", { class: "h3s-btn ai", onclick: () => actions.savePart(i) }, "Save"),
        h("button", { class: "h3s-btn ghost", onclick: () => actions.editPart(null) }, "Cancel"))) : h("pre", {}, chunk.prompt));
}

export function drawerView(S, actions) {
  const chunks = S.plan?.chunks || [], issues = S.plan?.issues || [];
  const head = h("button", { class: "h3s-drawer-head", "aria-expanded": String(S.promptOpen), onclick: () => actions.toggle("promptOpen") },
    h("span", { style: { display: "inline-flex", color: "#9AA4B5", transform: `rotate(${S.promptOpen ? 90 : 0}deg)` } }, icon("chevron")),
    h("span", { class: "h3s-strong" }, "Prompts per render part"),
    chunks.map((chunk) => { const [color] = partColor(chunk.index);
      return h("span", { class: "h3s-chip", style: { border: `1px solid ${color}`, display: "inline-flex", alignItems: "center", gap: "6px" } },
        h("span", { class: "h3s-swatch", style: { background: color } }), `Part ${chunk.index + 1}`); }),
    issues.length ? h("span", { class: "h3s-chip h3s-warn", style: { border: "1px solid #6B4A2E" } }, `${issues.length} note${issues.length === 1 ? "" : "s"}`) : null,
    h("span", { class: "h3s-lbl" }, "Exactly the prompt that goes to the sampler."));
  if (!S.promptOpen) return h("section", { class: "h3s-drawer" }, head);
  return h("section", { class: "h3s-drawer" }, head,
    chunks.length > 1 && chunks.some((chunk) => chunk.prompt_source !== "manual") ? h("div", { class: "h3s-row", style: { padding: "0 20px 10px" } },
      h("button", { class: "h3s-btn ai", disabled: !S.ai || Boolean(S.busy), onclick: actions.writeAllParts }, icon("spark"), "Write all parts with AI"),
      h("span", { class: "h3s-lbl" }, "One request per part; each answer is checked and waits for your Accept.")) : null,
    issues.length ? h("div", { class: "h3s-issues", style: { padding: "0 20px 12px" } },
      issues.map((issue) => h("span", {}, `⚠ ${issueLine(issue)}`))) : null,
    h("div", { class: "h3s-prompts" }, chunks.map((chunk) => partView(S, actions, chunk))));
}
