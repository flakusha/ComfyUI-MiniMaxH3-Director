"""The connectable Director: model inputs and contexts directly to a finished video."""
from comfy_api.latest import io

from ..adapters.render_graph import director_sequence
from ..core.limits import RESIZE_METHODS
from .context_nodes import CATEGORY, NGContext
from .director_node import NGDirector
from .product_director_node import turbo_inputs
from .provider_node import NGAI
from .render_node import model_inputs, sampling_inputs, render_outputs, split_render_inputs

# The planner inputs the Studio shares with the product Director: everything but the context wiring (raw cards
# here, the Connector there) and Output (each owner declares its own switch).
PLANNING = [item for item in NGDirector.define_schema().inputs if item.id not in ("contexts", "output")]


class NGDirectorStudio(io.ComfyNode):
    # NOTE (review F3, IMPROVEMENT path): this lab node mirrors NGH3Director's extras below (same widgets, same
    # defaults, appended last so older saves keep their values). The shared Studio panel reads and writes the node's
    # own widgets; AI and control indicators resolve through its own ai_provider/model_patch sockets. Style and prompt
    # override shape the live plan preview; execution plans through the same NGDirector node as before.
    @classmethod
    def define_schema(cls):
        contexts = io.Autogrow.TemplatePrefix(input=NGContext.Input("context", raw_link=True), prefix="context_", min=0, max=15)
        return io.Schema(node_id="NGDirectorStudio", display_name="NG Studio Laboratory", category=CATEGORY + "/Custom loop",
                         description="Write your idea, connect contexts and H3 models, then Run. Plans and renders the entire project with native progress and interrupt.",
                         inputs=[*model_inputs(), *PLANNING, io.Autogrow.Input("contexts", template=contexts, optional=True),
                                 *sampling_inputs(),
                                 # Product extras, same widgets and defaults as NGH3Director, appended last.
                                 io.Float.Input("duration_seconds", default=0.0, min=0.0, max=3600.0, step=0.01, optional=True, force_input=True,
                                                tooltip="Automation (connection-only). Desired length in SECONDS, overriding the Length "
                                                        "widget; 0 or unconnected uses the widget. The plan snaps the video to H3's "
                                                        "17k+5 frame grid, so the rendered length can differ by a fraction of a "
                                                        "second — the exact value is on the length_seconds output."),
                                 io.Combo.Input("style", options=["None", "Cinematic", "Anime", "Documentary", "Photorealistic"], default="None"),
                                 io.String.Input("prompt_override", multiline=True, default="", optional=True, advanced=True),
                                 *turbo_inputs(),
                                 io.Combo.Input("resize_method", options=list(RESIZE_METHODS), default="crop", advanced=True,
                                                tooltip="How a first or last frame picture is fitted to the video size."),
                                 io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True,
                                                tooltip="Reference pictures at the video's pixel area (fast) or a 2048 px short edge (more identity, slower)."),
                                 io.Boolean.Input("resume", default=True, advanced=True, optional=True,
                                                  tooltip="Keep every finished render part in output/h3-director-resume; after a crash the same "
                                                          "plan renders only the parts that are missing."),
                                 io.Boolean.Input("resume_delete_done", default=True, advanced=True, optional=True,
                                                  tooltip="Delete the kept render parts as soon as the whole video is put together, so they "
                                                          "do not fill the disk unseen. Off: they stay for 14 days."),
                                 io.Float.Input("control_strength", default=1.0, min=0.0, max=10.0, step=0.01, optional=True, advanced=True),
                                 io.Float.Input("control_start", default=0.0, min=0.0, max=1.0, step=0.001, optional=True, advanced=True,
                                                tooltip="Start within denoising, not seconds in the video."),
                                 io.Float.Input("control_end", default=1.0, min=0.0, max=1.0, step=0.001, optional=True, advanced=True,
                                                tooltip="End within denoising, not seconds in the video."),
                                 io.String.Input("models", default="", optional=True, advanced=True,
                                                 tooltip="Files the Studio's Models menu chose for place pictures (JSON); empty: the official names. A swap re-renders from the first part; a checkpoint or card-picture swap reuses stale parts, so delete output/h3-director-resume first."),
                                 io.Combo.Input("output", options=["video", "images"], default="video", optional=True,
                                                tooltip="Video, or one picture per shot (text to image, or a First frame picture changed as the shot says)."),
                                 io.Boolean.Input("spectrum", default=False, optional=True, advanced=True,
                                                  tooltip="With Turbo off: the Spectrum pack forecasts some sampling steps (about a third "
                                                          "faster at 20 steps). Needs ComfyUI-Spectrum-MiniMax-H3 installed."),
                                 # The Connector's optional extras as the Studio's own sockets: no Connector here.
                                 NGAI.Input("ai_provider", display_name="AI provider", optional=True),
                                 io.Vae.Input("preview_vae", display_name="preview VAE", raw_link=True, lazy=True, optional=True,
                                              tooltip="Optional H3 TAEHV VAE (taeh3): the render shows a live preview of each part."),
                                 io.ModelPatch.Input("model_patch", raw_link=True, lazy=True, optional=True,
                                                     tooltip="ModelPatchLoader with minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot. Used by control clips on the Studio timeline.")],
                         outputs=[*render_outputs(), io.String.Output(display_name="plan"), io.String.Output(display_name="plan_warnings")],
                         enable_expand=True)

    @classmethod
    def validate_inputs(cls, turbo_lora_fl2va="none", turbo_lora_ref2va="none"):
        # As on NGH3Director: ComfyUI does not check these two against the loras folder; the render checks the loaded one.
        return True

    @classmethod
    def execute(cls, **values):
        inputs, settings = split_render_inputs(values)
        planning = {name: values[name] for name in (item.id for item in PLANNING) if name in values}
        # An automation socket has no minimum, and 0 is what an upstream node hands over when its own value was never
        # set; a negative length is refused by name, 0 falls back to the Length widget (as on NGH3Director).
        project_seconds = float(values["project_seconds"])
        if values.get("duration_seconds") is not None:
            try:
                external_seconds = float(values["duration_seconds"])
            except ValueError:
                raise ValueError("H3 Director: the connected 'duration_seconds' is %r. It is a length in "
                                 "seconds, so it must be a number — a value that was never set arrives "
                                 "here as 0, which means use the Length widget."
                                 % values["duration_seconds"])
            if external_seconds < 0:
                raise ValueError("H3 Director: the connected 'duration_seconds' is %.3gs. It is a length in "
                                 "seconds and cannot be negative. Check the node feeding it — a value that "
                                 "was never set arrives here as 0, which means use the Length widget."
                                 % external_seconds)
            if external_seconds > 0:
                project_seconds = external_seconds
        planning["project_seconds"] = project_seconds
        planning["output"] = values.get("output", "video")
        planning.update({"contexts." + name: link for name, link in (values.get("contexts") or {}).items()})     # None when unwired (BUG-063)
        expanded, outputs = director_sequence(planning, inputs, settings)
        return io.NodeOutput(*outputs, expand=expanded)


NODES = [NGDirectorStudio]
