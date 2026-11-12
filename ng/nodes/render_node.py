"""User-facing model ports and one render connection; the native graph does the work."""
from comfy_api.latest import io
from comfy.samplers import KSampler
from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, RandomNoise

import json
import logging

from ..adapters import model_files, places, resume
from ..adapters.render_graph import render_sequence
from .chunk_node import _decode_video
from .context_nodes import CATEGORY
from .director_node import NGChunk


def model_inputs():
    return [io.Clip.Input("clip", raw_link=True, lazy=True), io.Vae.Input("vae", raw_link=True, lazy=True, display_name="video_vae"),
            io.Vae.Input("audio_vae", raw_link=True, lazy=True),
            io.Model.Input("model_fl2va", raw_link=True, lazy=True, optional=True),
            io.Model.Input("model_ref2va", raw_link=True, lazy=True, optional=True)]


def sampling_inputs():
    return [io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            io.Int.Input("steps", default=20, min=1, max=100, advanced=True),
            io.Combo.Input("sampler_name", options=KSampler.SAMPLERS, default="res_multistep", advanced=True),
            io.Combo.Input("scheduler", options=KSampler.SCHEDULERS, default="simple", advanced=True),
            io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, advanced=True),
            io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, advanced=True)]


def render_outputs():
    return [io.Video.Output(display_name="video"), io.Image.Output(display_name="images"),
            io.Audio.Output(display_name="audio"), io.String.Output(display_name="render_warnings")]


def split_render_inputs(values):
    names = ("chunks", "clip", "vae", "audio_vae", "model_fl2va", "model_ref2va", "preview_vae", "model_patch")
    inputs = {name: values[name] for name in names if name in values}
    if not any(inputs.get(name) for name in ("model_fl2va", "model_ref2va")):
        raise ValueError("Connect an H3 model to model_fl2va (text/frames) or model_ref2va (references).")
    # The Studio lab node carries the product Director's extras; Render Sequence has no such widgets, so its
    # values simply lack these keys and the graph keeps its defaults (review F3).
    settings = {name: values[name] for name in ("seed", "steps", "sampler_name", "scheduler", "shift_video", "shift_audio",
                                                "resize_method", "ref_image_size",
                                                "resume", "resume_delete_done", "control_strength", "control_start", "control_end",
                                                "models", "output", "spectrum") if name in values}
    return inputs, settings


class NGRenderSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGRenderSequence", display_name="NG Chunk Sequence", category=CATEGORY + "/Custom loop",
                         description="Connect the Director's chunks and your models. Renders and joins the entire sequence using native sampling, progress and interrupt.",
                         inputs=[NGChunk.Input("chunks", raw_link=True), *model_inputs(), *sampling_inputs()],
                         outputs=render_outputs(), enable_expand=True)

    @classmethod
    def execute(cls, **values):
        inputs, settings = split_render_inputs(values)
        expanded, outputs = render_sequence(inputs, settings)
        return io.NodeOutput(*outputs, expand=expanded)


class NGChunkNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGChunkNoise", display_name="NG Chunk Noise", category=CATEGORY + "/Technical", is_dev_only=True,
                         inputs=[NGChunk.Input("chunk"), io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff)],
                         outputs=[io.Noise.Output()])

    @classmethod
    def execute(cls, chunk, seed):
        job = chunk[0] if isinstance(chunk, list) else chunk
        return RandomNoise.execute(noise_seed=(int(seed) + job.index) % (1 << 64))


class NGChunkSampling(io.ComfyNode):
    """Sampler and sigmas of one render part, from the Director's Steps, Sampler and Scheduler."""
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGChunkSampling", display_name="NG Chunk Sampling", category=CATEGORY + "/Technical", is_dev_only=True,
                         inputs=[NGChunk.Input("chunk"), io.Model.Input("model"), io.String.Input("settings")],
                         outputs=[io.Sampler.Output(), io.Sigmas.Output()])

    @classmethod
    def execute(cls, chunk, model, settings):
        values = json.loads(settings)
        sampler = KSamplerSelect.execute(sampler_name=values["sampler_name"]).args[0]
        sigmas = BasicScheduler.execute(model=model, scheduler=values["scheduler"], steps=values["steps"], denoise=1.0).args[0]
        return io.NodeOutput(sampler, sigmas)


class NGResumeChunk(io.ComfyNode):
    """A finished render part comes from disk when the same plan is queued again (adapters/resume.py); its sampler then
    does not run at all."""
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGResumeChunk", display_name="NG Resume Chunk", category=CATEGORY + "/Technical", is_dev_only=True,
                         inputs=[NGChunk.Input("chunk"), io.String.Input("settings"),
                                 io.Latent.Input("sampled", lazy=True, optional=True)],
                         outputs=[io.Latent.Output()])

    @classmethod
    def check_lazy_status(cls, chunk, settings, sampled=None):
        job = chunk[0] if isinstance(chunk, list) else chunk
        return [] if resume.has(resume.part_key(job, json.loads(settings))) else ["sampled"]

    @classmethod
    def execute(cls, chunk, settings, sampled=None):
        job = chunk[0] if isinstance(chunk, list) else chunk
        key = resume.part_key(job, json.loads(settings))
        if resume.has(key):
            logging.info("[H3 Director] render part %d comes from %s (resume)", job.index + 1, resume.folder())
            return io.NodeOutput(resume.load(key))
        resume.save(key, sampled)
        return io.NodeOutput(sampled)


class NGPlaceStill(io.ComfyNode):
    """After a part is sampled: the pictures later shots take their place from ("Place like Shot 1", core/places.py)."""
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGPlaceStill", display_name="NG Place Still", category=CATEGORY + "/Technical", is_dev_only=True,
                         inputs=[NGChunk.Input("chunk"), io.Latent.Input("sampled"), io.Vae.Input("vae"),
                                 io.String.Input("models", default="", optional=True)],
                         outputs=[io.Latent.Output()])

    @classmethod
    def execute(cls, chunk, sampled, vae, models=""):
        job = chunk[0] if isinstance(chunk, list) else chunk
        for path, how in places.write(job, sampled, lambda video: _decode_video(vae, video), model_files.chosen(models)):
            logging.info("[H3 Director] render part %d: place picture %s (%s)", job.index + 1, path, how)
        return io.NodeOutput(sampled)


NODES = [NGRenderSequence, NGChunkNoise, NGChunkSampling, NGResumeChunk, NGPlaceStill]

