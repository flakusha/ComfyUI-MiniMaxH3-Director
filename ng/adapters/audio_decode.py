"""Decode a slice of an audio file without ffmpeg on PATH. The Audio card's trimmed output and the Whisper timing
share this one decode; neither re-implements PyAV seek, resample or the empty-window check."""
import av
import numpy as np
import torch

from .paths import input_path


def decode_audio(source, start_s=0.0, end_s=0.0, rate=None, layout=None):
    """(waveform, sample_rate) for [start_s, end_s) seconds of `source` (end 0 = the file end), as (channels, frames)
    float, resampled to `rate`/`layout` when given. Raises when the file has no audio or the window holds no sound."""
    path = input_path(source)
    if not path:
        raise ValueError("Choose a source containing audio.")
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("Choose a source containing audio.")
        stream = container.streams.audio[0]
        rate = int(rate or stream.rate)
        resampler = av.AudioResampler(format="fltp", layout=layout or stream.layout, rate=rate)
        chunks, origin = [], None
        sought = start_s > 0
        if sought:                          # a keyframe near the start; the frames before it are cut below
            try:
                container.seek(int(start_s / stream.time_base), stream=stream, backward=True)
            except av.error.FFmpegError:
                container.seek(0)
        try:
            for frame in container.decode(stream):
                if origin is None and frame.pts is not None:
                    origin = float(frame.pts * stream.time_base)
                chunks.extend(sound.to_ndarray() for sound in resampler.resample(frame))
            chunks.extend(sound.to_ndarray() for sound in resampler.resample(None))
        except av.error.FFmpegError as error:
            raise ValueError("The audio could not be decoded: %s" % error) from error
    if origin is None:                      # streams without stamps: the seek point is the best origin we have
        origin = float(start_s) if sought else 0.0
    if end_s > 0 and end_s <= start_s:
        raise ValueError("The selected range holds no audio; set Trim end past Trim start and inside the file.")
    total = np.concatenate(chunks, axis=1) if chunks else np.zeros((1, 0), dtype=np.float32)
    first = round(max(0.0, float(start_s) - (origin or 0.0)) * rate)
    first = min(first, total.shape[1])
    last = min(max(first, round((float(end_s) - origin) * rate) if end_s > 0 else total.shape[1]), total.shape[1])
    if not (total.shape[1] - first):
        raise ValueError("The selected range holds no audio; set Trim end past Trim start and inside the file.")
    return torch.from_numpy(np.ascontiguousarray(total[:, first:last])), rate


def trimmed_audio(source, start_s=0.0, end_s=0.0):
    """AUDIO for the selected range, in the file's own rate and channels, in LoadAudio's shape (1, channels, frames)."""
    wave, rate = decode_audio(source, start_s, end_s)
    return {"waveform": wave.unsqueeze(0), "sample_rate": rate}
