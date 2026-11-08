// Panel sizing for the AI Loader card (split out of ng_ai_panel.js to keep both modules under
// the size limit; behavior is unchanged). The Loader is as tall as its panel, like the Context
// cards: nothing in it can use more height, and a taller node only showed an empty area under
// the panel (user 2026-10-19: "wrong sizes, height and width").
import { draggedByUser } from "./ng_layout.js";

// Below this width the three source tiles and the provider / model pair no longer fit: at 210 the Model field was cut
// off with a scrollbar under it (GUI test 2026-10-19). The Context cards and the Director snap back to theirs the same way.
export const MIN_HEIGHT = 240, MIN_WIDTH = 440;

export function panelHeight(body, header) {
  const content = Array.from(body.children).filter((child) => !child.hidden).reduce((height, child) => {
    const style = getComputedStyle(child);
    return height + child.offsetHeight + (parseFloat(style.marginTop) || 0) + (parseFloat(style.marginBottom) || 0);
  }, 0);
  return Math.max(MIN_HEIGHT, Math.ceil(content + header.offsetHeight) + 32);
}

// Grow when the content needs room, but only shrink when the panel itself changed shape. Shrinking
// unconditionally would fight the user's drag and snap the node back while they resize it.
export function fitPanel(node, body, header, allowShrink = false) {
  const width = Math.max(node.size[0], MIN_WIDTH), wanted = panelHeight(body, header);
  if (width !== node.size[0] || wanted > node.size[1] || (allowShrink && wanted < node.size[1] - 2)) node.setSize([width, Math.max(wanted, allowShrink ? 0 : node.size[1])]);
  node.setDirtyCanvas?.(true, true);
}

export function watchPanel(node, body, header, widget, advanced, sync) {
  widget.computeSize = undefined;
  widget.computeLayoutSize = () => { const height = panelHeight(body, header); return { minHeight: height, maxHeight: height, minWidth: MIN_WIDTH - 20 }; };
  const fit = (allowShrink = false) => fitPanel(node, body, header, allowShrink);
  advanced.addEventListener("toggle", () => fit(true));
  // After a drag the node goes back to the panel's height; during it, it follows the pointer.
  let settling = 0;
  const resized = node.onResize; node.onResize = function (...args) { const result = resized?.apply(this, args);
    clearTimeout(settling); settling = setTimeout(() => fit(true), draggedByUser(node) ? 200 : 0); return result; };
  const observer = new ResizeObserver(() => fit()); observer.observe(body);
  const changed = node.onConnectionsChange; node.onConnectionsChange = function (...args) { const result = changed?.apply(this, args); queueMicrotask(sync); return result; };
  const tested = () => sync(); document.addEventListener("ng-ai-tested", tested);
  return { fit, dispose() { observer.disconnect(); document.removeEventListener("ng-ai-tested", tested); } };
}
