// The AI provider registry (/ng/providers): fallback shape, loader, lookups and source cards.
// Split out of ng_ai_panel.js to keep both modules under the size limit; behavior is unchanged.
import { element, values } from "./ng_widgets.js";
import { icon } from "./ng_icons.js";

const { api: fetch } = window.comfyAPI.api;

// The provider list, the source cards and every address come from the server's registry
// (/ng/providers). They used to be written out here as well, which is how the two drifted apart.
// A field typed in twice is a field that will be wrong in one of the two places.
//
// Until the registry answers, an empty panel is worse than one with a clear default, so the first
// paint uses this minimal shape and refreshes once the route returns.
const FALLBACK = {
  sources: [
    { id: "local", title: "Local", hint: "Load LLM / AI Model", icon: "monitor",
      providers: ["ollama"], key: false, address: "", note: "Runs on your own machine." },
    { id: "api", title: "API", hint: "Online provider", icon: "globe",
      providers: [], key: true, address: "", note: "Needs an API key." },
    { id: "clip", title: "CLIP / Encoder", hint: "Text Encoder only", icon: "paperclip",
      providers: [], key: false, address: "", note: "Connect Load CLIP." },
  ],
  providers: [],
};

let REGISTRY = FALLBACK;
const BY_ID = {};
let loaded = false;

async function loadRegistry() {
  try {
    const response = await fetch.fetchApi("/ng/providers");
    const data = await response.json();
    if (!Array.isArray(data.sources) || !data.sources.length || !Array.isArray(data.providers)) {
      console.warn("[NG] /ng/providers answered an unexpected shape; keeping the fallback", data);
      return;
    }
    REGISTRY = data;
    for (const entry of data.providers) BY_ID[entry.id] = entry;
    loaded = true;
  } catch (error) {
    // The panel still works with the fallback, but the reason is written to the console. A silent
    // catch here hid a real failure for a whole session: the panel quietly showed one provider
    // and there was no way to tell that it was a fallback rather than the truth.
    console.warn("[NG] the provider registry could not be loaded; using the built-in fallback", error);
  }
}

export function SOURCES() { return REGISTRY.sources; }
export function providerEntry(id) { return BY_ID[id] || null; }
export function registryLoaded() { return loaded; }

export function sourceOf(node) {
  const config = values(node);
  const cards = SOURCES();
  const found = cards.find((entry) => entry.id === config.source);
  return found || cards.find((entry) => entry.providers.includes(config.provider)) || cards[0];
}

export function sourceCard(entry, onPick) {
  const button = element("button", "ng-source-card");
  button.type = "button";
  button.dataset.source = entry.id;
  button.title = `${entry.title} — ${entry.hint}`;
  const badge = icon(entry.icon, "ng-source-icon");
  const text = element("span", "ng-source-text");
  text.append(element("strong", "", entry.title), element("small", "", entry.hint));
  button.append(badge, text);
  button.onclick = () => onPick(entry);
  return button;
}

export { loadRegistry };
