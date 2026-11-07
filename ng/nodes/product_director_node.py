"""The product Director owns creative planning, with render execution downstream."""
import folder_paths
from comfy_api.latest import io

from ..adapters.director_packets import DirectorResult, RenderSettings
from ..core import frames
from ..core.chunks import ContinuityPlan
from ..core.context import DirectorContext
from ..core.limits import PIN_AUDIO_DEFAULT
from ..adapters.probe import image_size
from ..core.limits import RESIZE_METHODS
from ..core.project import jobs, plan_project
from ..core.stills import plan_stills
from .connector_node import NGInputs
from .context_nodes import CATEGORY
from .director_node import NGChunk, NGDirector
from .render_node import sampling_inputs

NGResult = io.Custom("NG_DIRECTOR_RESULT")


def turbo_inputs():
    """Turbo: a few-step LoRA per H3 model plus fewer steps. The user chooses the LoRA; a known one brings its measured steps,
    sampler and scheduler (core/turbo_profiles.py)."""
    loras = folder_paths.get_filename_list("loras")
    return [io.Boolean.Input("turbo", default=False, advanced=True,
                             tooltip="Apply the turbo LoRA below to the H3 model. The official templates use 4 steps and CFG 1."),
            io.Combo.Input("turbo_lora_fl2va", options=["none", *loras], default="none", advanced=True,
                           tooltip="Turbo LoRA for the FL2VA model (start/end frame and text)."),
            io.Combo.Input("turbo_lora_ref2va", options=["none", *loras], default="none", advanced=True,
                           tooltip="Turbo LoRA for the Ref2VA model (references).")]


class NGH3Director(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGH3Director", display_name="H3 Director", category=CATEGORY,
                         description="Plan your project: idea, reusable contexts, editable scenes and timing. Connect Result to H3 Director Output; sampling remains in the downstream graph.",
                         inputs=[NGInputs.Input("context", display_name="context"),
                                 # The planner's context wiring (raw cards on the lab node, the Connector here) and its Output
                                 # switch (each owner declares its own) stay out: this list is the shared planning widgets.
                                 *[item for item in NGDirector.define_schema().inputs if item.id not in ("contexts", "output")],
                                 # force_input, so it is a socket, not a widget: workflows saved before it keep their
                                 # widget values on the same inputs (test_node's positional-widget trap does not apply).
                                 io.Float.Input("duration_seconds", default=0.0, min=0.0, max=3600.0, step=0.01, optional=True, force_input=True,
                                                tooltip="Automation (connection-only). Desired length in SECONDS, overriding the Length "
                                                        "widget; 0 or unconnected uses the widget. The plan snaps the video to H3's "
                                                        "17k+5 frame grid, so the rendered length can differ by a fraction of a "
                                                        "second — the exact value is on the length_seconds output."),
                                 io.Combo.Input("style", options=["None", "Cinematic", "Anime", "Documentary", "Photorealistic"], default="None"),
                                 io.String.Input("prompt_override", multiline=True, default="", optional=True, advanced=True), *sampling_inputs(),
                                 *turbo_inputs(),
                                 io.Combo.Input("resize_method", options=list(RESIZE_METHODS), default="crop", advanced=True,
                                                tooltip="How a first or last frame picture is fitted to the video size."),
                                 io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True,
                                                tooltip="Reference pictures at the video's pixel area (fast) or a 2048 px short edge (more identity, slower)."),
                                 # last and optional: API graphs saved before it stay valid (it defaults to on), and the Studio
                                 # repairs the widget order of older UI workflows (js/ng/h3s_node.js, repairInsertedWidgets)
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
                                                          "faster at 20 steps). Needs ComfyUI-Spectrum-MiniMax-H3 installed.")],
                         outputs=[NGResult.Output(display_name="result"), NGChunk.Output(display_name="chunks", is_output_list=True),
                                  # appended last: existing workflows keep their output slots
                                  io.Float.Output("length_seconds", display_name="length_seconds",
                                                  tooltip="The video's real length in seconds after the plan snapped it to H3's "
                                                          "17k+5 frame grid (24 fps). Wire it where the downstream timing has to "
                                                          "match what is actually rendered.")])

    @classmethod
    def validate_inputs(cls, turbo_lora_fl2va="none", turbo_lora_ref2va="none"):
        # A project uses one H3 model, so one turbo LoRA. Listed here, ComfyUI does not check these two against the loras
        # folder: a References project was refused because the First / last frame LoRA was not installed (GUI test
        # 2026-10-25). The LoRA a render part loads is checked when it loads it (render_node.NGTurboLora).
        return True

    @classmethod
    def execute(cls, context, **values):
        text_inputs = {name: "" if values.get(name) is None else values[name]
                       for name in ("prepend_text", "prompt_override")}
        # An automation socket has no minimum, and 0 is what an upstream node hands over when
        # its own value was never set; a negative length is refused by name, 0 falls back to
        # the Length widget. Same hazard and same handling as the legacy Director's
        # start/end/duration sockets.
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
        director = DirectorContext(idea=values["idea"], project_seconds=project_seconds,
                                   width=values["width"], height=values["height"], mode=values["mode"],
                                   shots_text=values["shots"], soundscape=values["soundscape"], music=values["music"],
                                   items=context.contexts, style=values.get("style", ""), **text_inputs)
        continuity = ContinuityPlan(mode=values["continuity"], video_frames=int(values["pin_frames"]), audio_frames=PIN_AUDIO_DEFAULT)
        if values.get("output") == "images":
            project = plan_stills(director, image_size)
            if not project.stills:
                raise ValueError("Pictures: write an idea or a shot; each shot becomes one picture.")
        else:
            project = plan_project(director, values["chunk_seconds"], continuity, image_size)
        settings = RenderSettings(**{name: values[name] for name in RenderSettings.__dataclass_fields__ if name in values})
        # the video's real length: the grid the chunks landed on, not the widget's request
        length_seconds = frames.frames_to_seconds(project.plan.total_frames)
        return io.NodeOutput(DirectorResult(context, project, settings), jobs(project), length_seconds)


NODES = [NGH3Director]
