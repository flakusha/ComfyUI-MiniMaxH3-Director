"""Execution packets retain graph links so only the chosen H3 model is evaluated."""
from dataclasses import dataclass

from comfy_execution.graph_utils import GraphBuilder

from ..core.analysis import ProviderConfig
from .ai_provider import NativeAIProvider
from ..core.project import ChunkJob, jobs
from ..core.turbo_profiles import part_settings
from ..core import control_timing
from .render_graph import SPECTRUM, render_sequence, still_sequence, turbo_models


@dataclass(frozen=True)
class DirectorInputs:
    clip: object
    vae: object
    audio_vae: object
    model_fl2va: object = None
    model_ref2va: object = None
    contexts: tuple = ()
    ai_provider: ProviderConfig | NativeAIProvider | None = None
    preview_vae: object = None
    model_patch: object = None

    def links(self):
        return {name: getattr(self, name) for name in ("clip", "vae", "audio_vae", "model_fl2va", "model_ref2va")}


@dataclass(frozen=True)
class RenderSettings:
    seed: int = 0
    steps: int = 20
    sampler_name: str = "res_multistep"
    scheduler: str = "simple"
    shift_video: float = 12.0
    shift_audio: float = 3.0
    turbo: bool = False
    turbo_lora_fl2va: str = "none"
    turbo_lora_ref2va: str = "none"
    resize_method: str = "crop"
    ref_image_size: str = "match"
    resume: bool = True
    resume_delete_done: bool = True
    control_strength: float = 1.0
    control_start: float = 0.0
    control_end: float = 1.0
    models: str = ""            # {purpose: file} from the Studio's Models menu (adapters/model_files.py)
    output: str = "video"       # video | images: one picture per shot (core/stills.py)
    spectrum: bool = False      # Spectrum's step forecast on the sampled model, Turbo off only (adapters/render_graph.py)

    def values(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class DirectorResult:
    inputs: DirectorInputs
    project: object
    settings: RenderSettings


def require_spectrum_pack(settings):
    """Spectrum forecasting needs the user's own Spectrum pack; without it the run would fail at the unknown node."""
    if settings.get("spectrum") and not settings.get("turbo"):
        try:
            from nodes import NODE_CLASS_MAPPINGS
        except ImportError:
            return
        if "SpectrumApplyMiniMaxH3" not in NODE_CLASS_MAPPINGS:
            raise ValueError("Spectrum is on, but ComfyUI-Spectrum-MiniMax-H3 is not installed: install "
                             "xmarre/ComfyUI-Spectrum-MiniMax-H3, or turn Spectrum off (Director, Settings).")


def chunk_job(result, chunk_index, loop_chunk=None):
    if loop_chunk is None:
        chunks = jobs(result.project)
        if not 0 <= chunk_index < len(chunks):
            raise ValueError("Choose a chunk from 0 to %d." % (len(chunks) - 1))
        return chunks[chunk_index]
    job = loop_chunk[0] if isinstance(loop_chunk, list) else loop_chunk
    if isinstance(loop_chunk, list) and len(loop_chunk) != 1:
        raise ValueError("Connect Start Loop's list_item to loop_chunk, not the whole chunks list: it would render chunk 0 every iteration.")
    if not isinstance(job, ChunkJob) or job.project is not result.project:
        raise ValueError("Connect the current chunk to Start Loop's list_item from the same Director's chunks.")
    return job


def prepare_output(result, chunk_index, previous=None, loop_chunk=None, apply_sigma_shift=True, seed=None):
    job = chunk_job(result, chunk_index, loop_chunk)
    require_spectrum_pack(result.settings.values())
    graph = GraphBuilder()
    inputs = result.inputs.links()
    inputs = turbo_models(graph, {name: value for name, value in inputs.items() if value is not None}, result.settings.values())
    conditioning = graph.node("NGChunkConditioning", "conditioning", chunk=job, previous=previous,
                              **inputs, shift_video=result.settings.shift_video, shift_audio=result.settings.shift_audio,
                              apply_sigma_shift=apply_sigma_shift, ref_image_size=result.settings.ref_image_size,
                              resize_method=result.settings.resize_method)
    negative = graph.node("ConditioningZeroOut", "negative", conditioning=conditioning.out(1))
    model = conditioning.out(0)
    if result.settings.spectrum and not result.settings.turbo:
        # the same forecast wrap H3 Render Sequence applies inside its loop (adapters/render_graph.py): a custom-loop
        # user enabling Spectrum gets it on the Output model too, not a silent no-op
        model = graph.node("SpectrumApplyMiniMaxH3", "spectrum", model=model, **SPECTRUM).out(0)
    context = result.project.context
    chunk_seed = (int(result.settings.seed if seed is None else seed) + job.index) % (2**64)
    # a known turbo LoRA on this part's model brings its own steps, sampler and scheduler (core/turbo_profiles.py)
    settings = part_settings(job.compiled.mode, result.settings.values())
    sampler = graph.node("KSamplerSelect", "sampler", sampler_name=settings["sampler_name"])
    sigmas = graph.node("BasicScheduler", "sigmas", model=model, scheduler=settings["scheduler"], steps=settings["steps"], denoise=1.0)
    outputs = (model, conditioning.out(1), negative.out(0), conditioning.out(2),
               24.0, context.width, context.height, conditioning.out(3), result, chunk_seed,
               settings["steps"], 1.0, settings["sampler_name"], settings["scheduler"], sampler.out(0), sigmas.out(0),
               result.inputs.preview_vae)
    return graph.finalize(), outputs


def render_project(result, upscale_2k=False, upscale_model=None):
    require_spectrum_pack(result.settings.values())
    if result.settings.control_strength and control_timing.clips(result.project.context) and result.inputs.model_patch is None:
        raise ValueError("Connect ModelPatchLoader to the Context Connector's model_patch for control clips.")
    upscale = None
    if upscale_2k:
        if upscale_model is None:
            raise ValueError("2K finish is on, but no upscaler model is connected: install a MiniMax H3 latent upscaler in models/latent_upscale_models, load it with LatentUpscaleModelLoader and connect it to upscale_model.")
        if result.project.output == "images":
            raise ValueError("2K finish upscales the video output; the Pictures output has no latent upscale.")
        context = result.project.context
        upscale = {"model": upscale_model, "width": context.width * 2, "height": context.height * 2}
    graph = GraphBuilder()
    chunks = graph.node("NGProjectChunks", "chunks", sequence=result)
    inputs = {**result.inputs.links(), "chunks": chunks.out(0), "preview": True}
    if result.inputs.preview_vae is not None:
        inputs["preview_vae"] = result.inputs.preview_vae
    if result.inputs.model_patch is not None:
        inputs["model_patch"] = result.inputs.model_patch
    sequence = still_sequence if result.project.output == "images" else render_sequence
    settings = result.settings.values()
    if upscale is not None:
        settings = {**settings, "upscale": upscale}
    expanded, outputs = sequence(inputs, settings)
    return {**graph.finalize(), **expanded}, outputs
