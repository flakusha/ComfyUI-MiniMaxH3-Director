// Per-purpose model overrides for the H3 AI Loader panel. The loader carries one Model for every
// action; these three optional fields let one provider serve several models - a small audio model
// beside a big planner - without a second loader node. Each field is empty until the user fills it
// in, and empty means "use the Model above", so nothing changes for an untouched node.
import { element } from "./ng_widgets.js";
import { control } from "./ng_editor_fields.js";

// The purposes the backend resolves (ng/adapters/analysis.py endpoint(config, purpose)). Image and
// video analysis share one field because both ride the same vision transport.
const OVERRIDES = [
  ["model_text", "Text model", "file", "Scene split, prompts and part writing. Empty = use the Model above."],
  ["model_caption", "Vision model", "eye", "Image and video analysis. Empty = use the Model above."],
  ["model_audio", "Audio model", "audio", "Audio analysis on the same server. Empty = use the Model above."],
];

// `listId` is the Model tile's datalist: one "Load models" answer suggests names for every override.
export function modelOverrides(node, listId) {
  const row = element("div", "ng-row");
  row.setAttribute("aria-label", "Model overrides (optional)");
  const fields = OVERRIDES.map(([name, labelText, iconName, help]) => {
    const field = control(node, name, labelText, { tile: true, icon: iconName });
    field.field.setAttribute("list", listId);
    if (field.field.tagName === "INPUT") field.field.placeholder = "Empty = use the Model above";
    field.setHelp(help);
    row.append(field.label);
    return field;
  });
  return { row, fields };
}
