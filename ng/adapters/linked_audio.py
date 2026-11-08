"""A connected AUDIO payload read without any file: the batch dim is ignored, nothing is decoded.

Non-file-backed sounds (decoded or recorded audio, composites) leave nothing for the frontend to
mirror into the card's media widget, so the linked execute path materializes them to
input/h3-external/ where the whole file-path pipeline (probe, decode, Whisper, collect) reads them."""
import hashlib
import os

import av
import folder_paths

FOLDER = "h3-external"


def materialize(audio_in):
    """Write a connected AUDIO payload to input/h3-external/<sha>.flac; returns the card-relative name.
    The name derives from the bytes, the rate and the channel count, so the same sound reuses its file and a
d    different one never collides. FLAC is lossless and the native loader reads it back."""
    wave, rate = read_waveform(audio_in)
    data = wave.float().contiguous().cpu().numpy()
    digest = hashlib.sha1(data.tobytes() + b"%d%d" % (int(rate), data.shape[-2])).hexdigest()[:16]
    name = "%s/%s.flac" % (FOLDER, digest)
    folder = os.path.join(folder_paths.get_input_directory(), FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, digest + ".flac")
    if not os.path.isfile(path):
        if data.shape[-2] not in (1, 2):
            raise ValueError("The connected audio has %d channels; only mono and stereo can be materialized - "
                             "reconnect its source, or pick a file on the card." % data.shape[-2])
        layout = "mono" if data.shape[-2] == 1 else "stereo"
        container = av.open(path, mode="w", format="flac")
        try:
            stream = container.add_stream("flac", rate=int(rate), layout=layout)
            frame = av.AudioFrame.from_ndarray(data, format="fltp", layout=layout)
            frame.sample_rate, frame.pts = int(rate), 0
            container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        finally:
            container.close()
    return name


def materialize_image(image_in):
    """Write a connected IMAGE payload to input/h3-external/<sha>.png; returns the card-relative name.
    The name derives from the bytes and the shape, so the same picture reuses its file and a
    different one never collides. PNG is lossless and the pipeline's PIL probe reads it back.
    Only the first batch frame is kept: a Context card stands for one picture, never a batch."""
    frame, shape = read_image(image_in)
    data = frame.contiguous().cpu().numpy()
    digest = hashlib.sha1(data.tobytes() + b"%d%d%d" % shape).hexdigest()[:16]
    name = "%s/%s.png" % (FOLDER, digest)
    folder = os.path.join(folder_paths.get_input_directory(), FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, digest + ".png")
    if not os.path.isfile(path):
        from PIL import Image
        Image.fromarray((data * 255.0).clip(0, 255).astype("uint8"), mode="RGB").save(path)
    return name


def read_image(image_in):
    """(first [H, W, 3] float frame, (h, w, c)) of a connected IMAGE payload."""
    try:
        frame = image_in[0] if getattr(image_in, "dim", lambda: 0)() == 4 else image_in
        height, width, channels = frame.shape[-3], frame.shape[-2], frame.shape[-1]
    except (TypeError, AttributeError, IndexError):
        raise ValueError("The connected image has no readable picture - reconnect its source, or pick a file on the card.") from None
    if channels not in (3, 4) or height <= 0 or width <= 0:
        raise ValueError("The connected image has no readable picture - reconnect its source, or pick a file on the card.")
    return frame[..., :3].float(), (int(height), int(width), int(channels))


def read_waveform(audio_in):
    """(channels, frames) float samples and the rate of a connected AUDIO payload."""
    try:
        rate = float(audio_in["sample_rate"])
        wave = audio_in["waveform"]
        while getattr(wave, "dim", lambda: 0)() > 2:
            wave = wave[0]
        frames = wave.shape[-1]
        channels = wave.shape[-2]
    except (KeyError, TypeError, AttributeError, IndexError, ZeroDivisionError):
        raise ValueError("The connected audio has no readable sound - reconnect its source, or pick a file on the card.") from None
    if rate <= 0 or frames <= 0 or channels <= 0:
        raise ValueError("The connected audio has no readable sound - reconnect its source, or pick a file on the card.")
    return wave, rate


def materialize_video(video_in):
    """Write a connected VIDEO payload to input/h3-external/<sha>.mp4; returns the card-relative name.
    A file-backed payload (LoadVideo) keeps its own path: its stream source is already a file the
    pipeline reads, so nothing is copied. Anything else (composited clips) is re-encoded once with the
    payload's own save_to; the name derives from its bytes, so the same clip reuses its file."""
    source = _video_file_source(video_in)
    if source is not None:
        return _input_relative(source)
    return _encode_video(video_in)


def _video_file_source(video_in):
    """The on-disk path when the payload is already file-backed, else None."""
    get = getattr(video_in, "get_stream_source", None)
    if get is None:
        return None
    try:
        source = get()
    except (OSError, ValueError, AttributeError):
        return None
    import io as _io
    if isinstance(source, _io.BytesIO) or not isinstance(source, str):
        return None
    return source


def _input_relative(path):
    """A real path as a card-relative input-folder reference; empty when it lives outside input."""
    import os
    root = os.path.realpath(folder_paths.get_input_directory())
    real = os.path.realpath(path)
    if os.path.commonpath((root, real)) != root or not os.path.isfile(real):
        return None
    return os.path.relpath(real, root).replace(os.path.sep, "/")


def _encode_video(video_in):
    import os
    save = getattr(video_in, "save_to", None)
    if save is None:
        raise ValueError("The connected video has no readable picture - reconnect its source, or pick a file on the card.")
    try:
        images = video_in.get_components().images.float().contiguous().cpu().numpy()
    except (AttributeError, TypeError):
        raise ValueError("The connected video has no readable picture - reconnect its source, or pick a file on the card.") from None
    if images.ndim != 4 or images.shape[0] <= 0:
        raise ValueError("The connected video has no readable picture - reconnect its source, or pick a file on the card.")
    digest = hashlib.sha1(images.tobytes() + b"%d%d%d%d" % images.shape).hexdigest()[:16]
    name = "%s/%s.mp4" % (FOLDER, digest)
    folder = os.path.join(folder_paths.get_input_directory(), FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, digest + ".mp4")
    if not os.path.isfile(path):
        try:
            save(path)
        except (OSError, ValueError, AttributeError) as error:
            raise ValueError("The connected video could not be written (%s) - reconnect its source, or pick a file on the card." % error) from None
    return name


def notify_materialized(node_cls, name):
    """Tell the owning card its linked sound now has a file: the name is content-hashed, so the card cannot
    learn it any other way, and without it Time with Whisper keeps asking for a run. A failed push stays
    silent - the run itself already succeeded, and the next run pushes again."""
    try:
        from server import PromptServer
        node_id = str(getattr(getattr(node_cls, "hidden", None), "unique_id", None) or "")
        if not node_id:
            return
        client_id = getattr(PromptServer.instance, "client_id", None)
        PromptServer.instance.send_sync("minimax_h3_ng_audio", {"node_id": node_id, "audio": name}, client_id)
    except (AttributeError, OSError, RuntimeError):
        pass
