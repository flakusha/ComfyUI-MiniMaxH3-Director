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
