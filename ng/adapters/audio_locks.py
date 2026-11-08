"""Restore declared source audio at native sampler stages and inside output Lock spans."""
from dataclasses import dataclass

import torch
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy.utils import pack_latents, unpack_latents

from ...minimax_core import core
from ...minimax_director import lock_audio_latent
from ..core.limits import FPS
from . import media
from .latents import validate_chunk


@dataclass(frozen=True)
class AudioLockSource:
    """Restores one part's locked audio from the encode its conditioning build already made (chunk_builder lock).
    Deriving it here again decoded the whole source file and ran the audio VAE at every sampler call, while the
    diffusion model was streaming its own weights back in - on a memory-starved GPU that interleaving is exactly the
    load the multi-part renders froze under (2026-11-06), and the result was the same encode every time."""

    encoded: object
    spans: tuple
    sampled_frames: int

    def __call__(self, executor, *args, **kwargs):
        # OUTER_SAMPLE sees native packed latents before latent-format normalization.
        values = list(args)
        shapes = kwargs["latent_shapes"]
        if len(shapes) != 2:
            raise ValueError("Locked H3 audio needs the full AV latent at every sampler stage.")
        video, audio = unpack_latents(values[1], shapes)
        latent = {"samples": NestedTensor((video, audio))}
        validate_chunk(latent, self.sampled_frames, "Audio lock")
        held = lock_audio_latent(latent, self.encoded, self.spans, core().AUDIO_LATENT_FPS)
        source = held["samples"].unbind()[1].to(audio)
        video_mask, audio_mask = held["noise_mask"].unbind()
        restored = torch.where(audio_mask.to(audio.device) == 0, source, audio)
        values[1], _ = pack_latents((video, restored))
        lock_mask, _ = pack_latents((video_mask, audio_mask))
        lock_mask = lock_mask.to(values[1])
        values[4] = lock_mask if values[4] is None else values[4] * lock_mask.to(values[4])
        return executor(*values, **kwargs)


def bind_source(model, job, lock):
    if not job.compiled.of_kind("lock"):
        return model
    bound = model.clone()
    bound.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, "ng_audio_lock")
    encoded, spans = lock
    bound.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "ng_audio_lock",
                               AudioLockSource(encoded, tuple(spans), job.chunk.sampled_frames))
    return bound


def _mux_fallback(wave, fallback):
    """Fill silent chunks from the fallback AUDIO dict; generated or locked sound is untouched.

    Only chunks with no audio at all take the fallback (peak below -60 dBFS): a chunk that generated
    silence on purpose keeps its silence only when nothing is wired — wiring *is* the opt-in."""
    samples = fallback.get("waveform") if isinstance(fallback, dict) else None
    if samples is None or not hasattr(samples, "shape") or samples.shape[-1] <= 0:
        return wave
    if float(wave.abs().max() or 0.0) > 0.001:
        return wave
    take = samples[0] if samples.ndim == 3 else samples
    take = take.to(device=wave.device, dtype=wave.dtype)
    if take.shape[0] != wave.shape[1]:
        take = take.mean(dim=0, keepdim=True).expand(wave.shape[1], -1) if wave.shape[1] > 1 else take[:1]
        if take.shape[0] != wave.shape[1]:
            return wave
    count = min(wave.shape[-1], take.shape[-1])
    if count <= 0:
        return wave
    out = wave.clone()
    out[..., :count] = take[..., :count]
    return out


def restore_waveform(job, wave, sample_rate, fallback=None):
    """Use source samples inside Lock spans; keep decoded model sound everywhere else.

    `fallback` (P3 opt-in mux) is an AUDIO dict the caller mixed in only where the wave is
    still silent: sampled chunks that carry no audio at all (silent windows, muted VAE) keep the
    fallback's samples there, never over generated or locked sound. None (default) is byte-identical."""
    refs = job.compiled.of_kind("lock")
    if fallback is not None:
        wave = _mux_fallback(wave, fallback)
    if not refs:
        return wave
    chunk = job.chunk
    origin = round(chunk.story_start_f / float(FPS) * sample_rate)
    window_start = chunk.story_start_f - chunk.head_frames
    mix = torch.zeros_like(wave)
    covered = torch.zeros_like(wave[:, :1], dtype=torch.bool)
    for ref in refs:
        start_f = window_start + ref.start_f
        first = max(origin, round(start_f / float(FPS) * sample_rate))
        last = min(origin + wave.shape[-1], round((start_f + ref.length_f) / float(FPS) * sample_rate))
        if last <= first:
            continue
        source = media.load_audio_source(ref, sample_rate).to(wave)
        if wave.shape[1] == 1:
            source = source.mean(dim=1, keepdim=True)
        offset = round((ref.trim_start_f - start_f) / float(FPS) * sample_rate)
        source_start = first + offset
        destination = first - origin
        covered[..., destination:destination + last - first] = True
        count = min(last - first, source.shape[-1] - source_start)
        if count > 0:
            mix[..., destination:destination + count] += source[..., source_start:source_start + count]
    return torch.where(covered, mix.clamp(-1, 1), wave)
