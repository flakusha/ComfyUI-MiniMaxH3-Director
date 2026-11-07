// Studio lab node parity with the product Director (review F3): the panel's AI and control indicators
// resolve through the Studio's own ai_provider/model_patch sockets, and its cards read through its own
// contexts autogrow inputs. Run: node tests/ng/js/test_studio_parity.mjs
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

globalThis.document = { createElement: () => ({}), dispatchEvent: () => {} };
globalThis.CustomEvent = class { constructor(type) { this.type = type; } };

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")),
  "..", "..", "..", "js", "ng");
const load = (name) => import(pathToFileURL(path.join(root, name)).href);

const { connectedProvider, controlPatchConnected, contextNodes, readContexts } = await load("ng_read.js");

function node(comfyClass, widgets = {}, inputs = [], graph = null) {
  return { comfyClass, widgets: Object.entries(widgets).map(([name, value]) => ({ name, value })), inputs, graph };
}

// --- AI indicator: the Studio carries its own ai_provider socket, not a Connector -------------------
{
  const provider = node("NGAIProvider", { provider: "custom", base_url: "http://localhost:1234", model: "local",
    vision: true, audio: false, api_key_env: "", timeout_seconds: 45, max_tokens: 1024, keep_alive: 60 });
  const studio = node("NGDirectorStudio", { idea: "Alice walks." }, [{ name: "ai_provider", link: 4 }]);
  const graph = { links: { 4: { origin_id: 3 } }, getNodeById: (id) => ({ 3: provider }[id]) };
  studio.graph = graph; provider.graph = graph;
  assert.equal(connectedProvider(studio).model, "local", "a linked AI Loader lights the Studio AI indicator");
  studio.inputs[0].link = null;
  assert.equal(connectedProvider(studio), null, "unplugged AI reads as off, not as a stale provider");
}

// --- card AI: a card wired straight into the Studio shares the Studio's own provider ---------------
{
  const { contextProvider } = await load("ng_read.js");
  const provider = node("NGAIProvider", { provider: "custom", base_url: "http://localhost:1234", model: "local",
    vision: true, audio: false, api_key_env: "", timeout_seconds: 45, max_tokens: 1024, keep_alive: 60 });
  const studio = node("NGDirectorStudio", { idea: "Alice walks." }, [{ name: "ai_provider", link: 4 }]);
  const card = node("NGContextImage", { name: "Alice" });
  card.outputs = [{ links: [6] }];
  const graph = { links: { 4: { origin_id: 3 }, 6: { origin_id: 7, target_id: 8 } },
    getNodeById: (id) => ({ 3: provider, 8: studio }[id]) };
  studio.graph = graph; provider.graph = graph; card.graph = graph;
  card.id = 7;
  assert.equal(contextProvider(card).model, "local", "a Studio-wired card shares the Studio provider");
  studio.inputs[0].link = null;
  assert.equal(contextProvider(card), null, "unplugging the Studio provider darkens its cards too");
}

// --- control indicator: own model_patch socket on the Studio, Connector socket on the Director --------
{
  const patch = node("ModelPatchLoader", {});
  const studio = node("NGDirectorStudio", {}, [{ name: "model_patch", link: 7 }]);
  studio.graph = { links: { 7: { origin_id: 9 } }, getNodeById: () => patch };
  assert.equal(controlPatchConnected(studio), true, "a linked patch lights the Studio control indicator");
  studio.inputs[0].link = null;
  assert.equal(controlPatchConnected(studio), false, "an unwired patch socket reads as unconnected");

  const connector = node("NGContextConnector", {}, [{ name: "model_patch", link: 8 }]);
  const director = node("NGH3Director", {}, [{ name: "context", link: 5 }]);
  const graph = { links: { 5: { origin_id: 6 }, 8: { origin_id: 9 } },
    getNodeById: (id) => ({ 6: connector, 9: patch }[id]) };
  connector.graph = graph; director.graph = graph;
  assert.equal(controlPatchConnected(director), true, "the Director still reads the Connector patch");
  connector.inputs[0].link = null;
  assert.equal(controlPatchConnected(director), false, "disconnecting the Connector patch reads as off");
}

// --- contexts: the Studio reads its own autogrow inputs, the Director reads through its Connector ------
{
  const alice = node("NGContextImage", { name: "Alice", content: "character" });
  const studio = node("NGDirectorStudio", { idea: "x" }, [{ name: "contexts.context_0", link: 10 }]);
  studio.graph = { links: { 10: { origin_id: 1 } }, getNodeById: (id) => ({ 1: alice }[id]) };
  assert.deepEqual(contextNodes(studio).map((c) => c.media), ["image"], "Studio cards resolve without a Connector");
  assert.equal(readContexts(studio)[0].widgets.name, "Alice");
}
