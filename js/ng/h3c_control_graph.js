// Small queued preprocessing graphs using the installed ComfyUI nodes. Keep their original frame rate.
import { missingText, modelState, resolveModel } from "./ng_models.js";

// The files of each kind: the card's Models menu choice, else the official name (ng_models.js / js/ng/models.json).
export async function controlSetup(api, kind, engine = "moge2", chosen = {}) {
  if (kind === "canny") return {};
  const purposes = kind === "mask" ? ["sam3"] : kind === "pose" ? ["pose_detector", "pose"] : kind === "depth" ? [engine === "da3" ? "da3" : "moge"] : null;
  if (!purposes) throw new Error("Choose Canny, Pose or Depth.");
  const state = await modelState(api, purposes);
  const files = purposes.map((purpose) => resolveModel(state, purpose, chosen[purpose]));
  const lacking = purposes.filter((_, n) => !files[n]).map((purpose) => missingText(state[purpose]));
  if (lacking.length) throw new Error(`${lacking.join(" ")}${kind === "depth" && engine !== "da3" ? " Or choose installed Depth Anything 3." : ""}`);
  if (kind === "mask") return { checkpoint: files[0] };
  if (kind === "pose") return { detector: files[0], pose: files[1] };
  return { depth: files[0], engine: engine === "da3" ? "da3" : "moge2" };
}

export function controlGraph(kind, source, setup, prefix) {
  if (kind === "mask") return maskGraph(source, setup, prefix);
  const node = (class_type, inputs) => ({ class_type, inputs });
  const graph = { load: node("LoadVideo", { file: source }), components: node("GetVideoComponents", { video: ["load", 0] }),
    batch: node("RebatchImages", { images: ["components", 0], batch_size: kind === "canny" ? 16 : 4 }) };
  let images;
  if (kind === "canny") {
    graph.edges = node("Canny", { image: ["batch", 0], low_threshold: 0.4, high_threshold: 0.8 });
    images = ["edges", 0];
  } else if (kind === "pose") {
    graph.detector = node("UNETLoader", { unet_name: setup.detector, weight_dtype: "default" });
    graph.boxes = node("RTDETR_detect", { model: ["detector", 0], image: ["batch", 0], threshold: 0.5, class_name: "person", max_detections: 2 });
    graph.pose = node("CheckpointLoaderSimple", { ckpt_name: setup.pose });
    graph.points = node("SDPoseKeypointExtractor", { model: ["pose", 0], vae: ["pose", 2], image: ["batch", 0], batch_size: 4, bboxes: ["boxes", 0] });
    graph.draw = node("SDPoseDrawKeypoints", { keypoints: ["points", 0], draw_body: true, draw_hands: true, draw_face: true, draw_feet: true,
      stick_width: 4, face_point_size: 2, score_threshold: 0.5, draw_head: true });
    images = ["draw", 0];
  } else if (kind === "depth" && setup.engine === "da3") {
    graph.depth = node("DownloadAndLoadDepthAnythingV3Model", { model: setup.depth, precision: "fp16", attention: "auto" });
    graph.infer = node("DepthAnything_V3", { da3_model: ["depth", 0], images: ["batch", 0], normalization_mode: "V2-Style" });
    images = ["infer", 0];
  } else if (kind === "depth") {
    graph.depth = node("LoadMoGeModel", { model_name: setup.depth });
    graph.infer = node("MoGeInference", { moge_model: ["depth", 0], image: ["batch", 0], resolution_level: 9, fov_x_degrees: 0,
      batch_size: 4, force_projection: true, apply_mask: true, refine_steps: 3 });
    graph.draw = node("MoGeRender", { moge_geometry: ["infer", 0], output: "depth" });
    images = ["draw", 0];
  } else throw new Error("Choose Canny, Pose or Depth.");
  graph.join = node("RebatchImages", { images, batch_size: 4096 });
  graph.video = node("CreateVideo", { images: ["join", 0], fps: ["components", 2] });
  graph.save = node("SaveVideo", { video: ["video", 0], filename_prefix: `h3-control/${prefix}_${kind}`, format: "mp4", "format.codec": "h264" });
  return graph;
}

// Pre-trim audio save (Trim start/end added later): the Context ID lands in Trim start, the colour
// in Trim end, the JSON fields shift along. Re-seat by name, so the ID never reaches float conversion.
// A save the frontend restored by name maps to the same values, so this is a no-op there.
// 12 values; 9 when the save also predates the role fields.
export function reseatPreTrim(node, info) {
  const saved = info?.widgets_values;
  if (!Array.isArray(saved) || (saved.length !== 12 && saved.length !== 9)) return;
  const named = info?.widgets_values_named;
  const names = ["audio", "name", "description", "content", "usage", "preserve", "speaker",
    "trim_start", "trim_end", "context_id", "color", "content_roles", "usage_roles", "context_options"];
  if (named && typeof named === "object" && !Array.isArray(named) && names.every((name) => Object.hasOwn(named, name))) return;
  const at = Object.fromEntries((node.widgets || []).filter((w) => w?.name).map((w) => [w.name, w]));
  ["audio", "name", "description", "content", "usage", "preserve", "speaker"].forEach((name, i) => { if (at[name]) at[name].value = saved[i]; });
  if (at.trim_start) at.trim_start.value = 0;
  if (at.trim_end) at.trim_end.value = 0;
  ["context_id", "color", "content_roles", "usage_roles", "context_options"].forEach((name, i) => { if (at[name] && saved[7 + i] !== undefined) at[name].value = saved[7 + i]; });
  if (saved.length === 9) for (const name of ["content_roles", "usage_roles", "context_options"]) if (at[name]) at[name].value = "";
}

function maskGraph(source, setup, prefix) {
  const node = (class_type, inputs) => ({ class_type, inputs });
  if (!setup.prompt?.trim()) throw new Error("Describe what to mask, e.g. person or red ball.");
  return {
    load: node("LoadVideo", { file: source }), components: node("GetVideoComponents", { video: ["load", 0] }),
    batch: node("RebatchImages", { images: ["components", 0], batch_size: 1 }),
    sam: node("CheckpointLoaderSimple", { ckpt_name: setup.checkpoint }),
    text: node("CLIPTextEncode", { clip: ["sam", 1], text: setup.prompt }),
    detect: node("SAM3_Detect", { model: ["sam", 0], image: ["batch", 0], conditioning: ["text", 0], threshold: 0.5,
      refine_iterations: 2, individual_masks: false }),
    image: node("MaskToImage", { mask: ["detect", 0] }),
    join: node("RebatchImages", { images: ["image", 0], batch_size: 4096 }),
    video: node("CreateVideo", { images: ["join", 0], fps: ["components", 2] }),
    save: node("SaveVideo", { video: ["video", 0], filename_prefix: `h3-control/${prefix}_mask`, format: "mp4", "format.codec": "h264" }),
  };
}
