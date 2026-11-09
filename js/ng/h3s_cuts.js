// The studio's plan-edit actions: a render part's own prompt (AI draft or hand edit) and the document's render cuts.
// Both write through save() like every other edit — one write, then a replan — so they live here instead of growing
// the openStudio action table.
import { setPartPrompt, setShotPrompt, upsertCut } from "./h3s_model.js";

// A part's own prompt replaces any earlier one of that part: an accepted AI draft (its key must still be the
// compiled one), a hand edit, or a reset. Entries are keyed by the part's starting story frame so a re-time that
// shifts part indices keeps the prose; a shot's share (setShotPrompt) overrides per shot.
export function promptActions(S, { save, render, stale }) {
  const actions = {
    acceptPart(index) {
      const draft = S.partDrafts[index], chunk = S.plan?.chunks?.[index];
      delete S.partDrafts[index];
      if (!draft || !chunk || draft.key !== chunk.prompt_key) { stale(); return; }
      S.partPrompts = setPartPrompt(S.partPrompts, index, draft.key, draft.text, "ai", chunk.story_start_f); save(); render();
    },
    dropPart(index) { delete S.partDrafts[index]; render(); },
    editPart(index, text = "") { S.partEdit = index === null ? null : { index, text }; S.shareEdit = null; render(); },
    savePart(index, shotId) {
      const chunk = S.plan?.chunks?.[index], edit = S.partEdit;
      S.partEdit = null;
      if (!chunk || !edit) { render(); return; }
      if (shotId) {
        const basis = (chunk.shares || []).find((share) => share.shot_id === shotId)?.basis;
        S.shotPrompts = setShotPrompt(S.shotPrompts, shotId, typeof basis === "string" ? basis : chunk.prompt_key, edit.text, "user", index);
      } else S.partPrompts = setPartPrompt(S.partPrompts, index, chunk.prompt_key, edit.text, "user", chunk.story_start_f);
      save(); render();
    },
    editShare(index, shotId, text = "") { S.shareEdit = index === null ? null : { index, shotId, text }; S.partEdit = null; render(); },
    saveShare(index, shotId) {
      const chunk = S.plan?.chunks?.[index], edit = S.shareEdit;
      S.shareEdit = null;
      if (!chunk || !edit || edit.shotId !== shotId) { render(); return; }
      const basis = (chunk.shares || []).find((share) => share.shot_id === shotId)?.basis;
      S.shotPrompts = setShotPrompt(S.shotPrompts, shotId, typeof basis === "string" ? basis : chunk.prompt_key, edit.text, "user", index);
      save(); render();
    },
    resetShare(index, shotId) { S.shotPrompts = setShotPrompt(S.shotPrompts, shotId, "", "", "user", index); save(); render(); },
    resetPart(index, shotId) {
      if (shotId) return actions.resetShare(index, shotId);
      const chunk = S.plan?.chunks?.[index], inPart = chunk?.shares || chunk?.shots || [];
      for (const entry of inPart) S.shotPrompts = setShotPrompt(S.shotPrompts, entry.shot_id, "", "", "user", index);
      S.partPrompts = setPartPrompt(S.partPrompts, index, "", "", "user", chunk?.story_start_f); save(); render();
    },
  };
  return actions;
}

// Render cuts: forced part boundaries, kept in the document's render_cuts. An empty list plans exactly as before.
// NOTE: warn-only — a bad cut renders cut_moved/cut_dropped/cut_beyond_end plan issues; the list itself is never rewritten.
export function cutActions(S, { save, render }) {
  return {
    setRenderCut(sec) { S.renderCuts = upsertCut(S.renderCuts, sec); save(); render(); },
    removeRenderCut(sec) { S.renderCuts = (S.renderCuts || []).filter((at) => Math.abs(at - sec) > 0.05); save(); render(); },
    resetCuts() { S.renderCuts = []; save(); render(); },
  };
}