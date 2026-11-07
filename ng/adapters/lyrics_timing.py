"""Time a song's lyrics with a local Whisper model (openai-whisper, model files in models/whisper or wherever
ComfyUI registers the folder). Official .pt files load through whisper; safetensors checkpoints load here, whatever
their name is called and under HuggingFace transformers or openai-whisper key names."""
import json
import os
import re
import struct

import folder_paths
import numpy as np

from ..core import lyrics
from . import model_files
from .audio_decode import decode_audio

try:
    import whisper
except ImportError:
    whisper = None

# HuggingFace transformers checkpoints (openai/whisper-*) keep Whisper's weights under other names; the shapes are
# the same, so renaming is enough. q/k/v become query/key/value, projections become the attention's pieces and the
# language-model head is dropped - whisper reads its logits off the token embedding itself.
_ATTN = {"q_proj": "query", "k_proj": "key", "v_proj": "value", "out_proj": "out"}
_HF_KEYS = (
    (r"model\.encoder\.conv([12])\.(weight|bias)$", "encoder.conv%s.%s"),
    (r"model\.encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$", "encoder.blocks.%s.attn.%s.%s"),
    (r"model\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$", "encoder.blocks.%s.attn_ln.%s"),
    (r"model\.encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$", "encoder.blocks.%s.mlp_ln.%s"),
    (r"model\.encoder\.layers\.(\d+)\.fc1\.(weight|bias)$", "encoder.blocks.%s.mlp.0.%s"),
    (r"model\.encoder\.layers\.(\d+)\.fc2\.(weight|bias)$", "encoder.blocks.%s.mlp.2.%s"),
    (r"model\.encoder\.layer_norm\.(weight|bias)$", "encoder.ln_post.%s"),
    (r"model\.decoder\.embed_tokens\.weight$", "decoder.token_embedding.weight"),
    (r"model\.decoder\.embed_positions\.weight$", "decoder.positional_embedding"),
    (r"model\.decoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$", "decoder.blocks.%s.attn.%s.%s"),
    (r"model\.decoder\.layers\.(\d+)\.encoder_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$", "decoder.blocks.%s.cross_attn.%s.%s"),
    (r"model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$", "decoder.blocks.%s.attn_ln.%s"),
    (r"model\.decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.(weight|bias)$", "decoder.blocks.%s.cross_attn_ln.%s"),
    (r"model\.decoder\.layers\.(\d+)\.fc1\.(weight|bias)$", "decoder.blocks.%s.mlp.0.%s"),
    (r"model\.decoder\.layers\.(\d+)\.fc2\.(weight|bias)$", "decoder.blocks.%s.mlp.2.%s"),
    (r"model\.decoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$", "decoder.blocks.%s.mlp_ln.%s"),
    (r"model\.decoder\.layer_norm\.(weight|bias)$", "decoder.ln.%s"),
)

def model_file(chosen=""):
    """The Audio card's Models menu choice, else the largest model present: larger ones hear sung words better. The
    name is resolved wherever ComfyUI has the whisper folder registered (extra_model_paths.yaml), models/ only when
    nothing is registered."""
    name = model_files.find("whisper", chosen)
    if name is None:
        raise ValueError("No Whisper model: " + model_files.missing("whisper"))
    path = folder_paths.get_full_path("whisper", name) if "whisper" in folder_paths.folder_names_and_paths else None
    return path or os.path.join(folder_paths.models_dir, "whisper", name)


def _is_safetensors(path):
    """True when the file is safetensors, whatever its name says - a .pt symlink may point at one. Only the header is
    read: a length and the JSON of tensor names behind it."""
    try:
        with open(path, "rb") as file:
            (size,) = struct.unpack("<Q", file.read(8))
            if size > max(os.path.getsize(path) - 8, 0):
                return False
            header = json.loads(file.read(size))
        return isinstance(header, dict) and all(
            isinstance(entry, dict) and "dtype" in entry and "shape" in entry
            for key, entry in header.items() if key != "__metadata__")
    except (OSError, ValueError, struct.error):
        return False


def _openai_key(key):
    """A HuggingFace transformers key renamed to its openai-whisper name, or None (head and helpers are dropped)."""
    for pattern, template in _HF_KEYS:
        match = re.fullmatch(pattern, key)
        if match:
            return template % tuple(_ATTN.get(part, part) for part in match.groups())
    return None


def _dimensions(state, name):
    """The model's dimensions read off the tensor shapes. Every released Whisper keeps head = state // 64, a
    1500-frame audio window and a 448-token text window; the audio sinusoids are rebuilt, not carried
    (HuggingFace files leave them out, so their length comes from whisper itself there)."""
    for key in ("encoder.conv1.weight", "decoder.token_embedding.weight", "decoder.positional_embedding"):
        if key not in state:
            raise ValueError("%s is not a Whisper checkpoint: no %s." % (name, key))

    def side(prefix):
        layers = [int(m.group(1)) for key in state for m in [re.match(prefix + r"\.blocks\.(\d+)\.", key)] if m]
        return 1 + max(layers + [-1])

    conv, pos = state["encoder.conv1.weight"].shape, state["decoder.positional_embedding"].shape
    sinusoids = state.get("encoder.positional_embedding")
    return whisper.model.ModelDimensions(
        n_mels=conv[1], n_audio_ctx=sinusoids.shape[0] if sinusoids is not None else 1500,
        n_audio_state=conv[0], n_audio_head=conv[0] // 64, n_audio_layer=side("encoder"),
        n_vocab=state["decoder.token_embedding.weight"].shape[0],
        n_text_ctx=pos[0], n_text_state=pos[1], n_text_head=pos[1] // 64, n_text_layer=side("decoder"))


def _load_safetensors(path, device):
    """Whisper built from a safetensors file: torch.load reads only its own pickle, so openai-whisper's loader refuses
    the format. What is built is what load_model returns for the same weights (fp16 weights stay fp16, the alignment
    heads default as for a loaded .pt file). Anything that does not fit is an error, not a silently wrong model."""
    from safetensors.torch import load_file
    name = os.path.basename(path)
    state = load_file(path)
    huggingface = any(key.startswith("model.") for key in state)
    converted = {}
    for key, tensor in state.items():
        key = _openai_key(key) if huggingface else key
        if key is not None:
            converted[key] = tensor
    model = whisper.model.Whisper(_dimensions(converted, name))
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if unexpected or set(missing) - {"encoder.positional_embedding"}:
        raise ValueError("%s does not fit Whisper: missing %s, unexpected %s." % (name, sorted(missing), sorted(unexpected)))
    return model.to(device)


def _load_model(path, device):
    """The chosen model file: safetensors content is built here, the official torch pickles go through whisper."""
    return _load_safetensors(path, device) if _is_safetensors(path) else whisper.load_model(path, device=device)


def _samples(source):
    """The song as Whisper reads it, 16 kHz mono float, decoded through the shared decoder (adapters/audio_decode).
    The card-relative name goes in unresolved: the decoder resolves it itself, and a resolved path fed back in
    fails its input-folder check, so every file timed out as missing."""
    wave, _ = decode_audio(source, rate=16000, layout="mono")
    return wave.reshape(-1).numpy().astype(np.float32)


def time_lyrics(source, text, device, chosen=""):
    """LRC text for the song: the user's own lines with recognised times, or what Whisper heard when no lines are given.

    `device` is the caller's choice: the GPU only while nothing renders. `chosen`: the card's Whisper file, if any.
    `source` is the card's file - the chosen file, the mirrored origin file, or a materialized linked sound
    (input/h3-external/): the card always sends a name, because a linked AUDIO payload never crosses
    request.json(). Whisper is the clock either way."""
    if whisper is None:
        raise ValueError("Timing lyrics needs the openai-whisper package in ComfyUI's Python.")
    samples = _samples(source)
    known = lyrics.plain_lines(text)
    language, _ = lyrics.parse(text)
    model = _load_model(model_file(chosen), device=device)
    try:
        result = model.transcribe(samples, word_timestamps=True, fp16=str(device) != "cpu",
                                  initial_prompt=" ".join(known) or None, condition_on_previous_text=False)
    finally:
        del model
    language = language or whisper.tokenizer.LANGUAGES.get(result.get("language", ""), "english").title()
    words = [(w["word"], float(w["start"]), float(w["end"])) for segment in result["segments"] for w in segment.get("words", [])]
    if known:
        lines = lyrics.align(known, words, len(samples) / float(whisper.audio.SAMPLE_RATE))
    else:
        lines = tuple(lyrics.LyricLine(round(float(s["start"]), 2), round(float(s["end"]), 2), s["text"].strip())
                      for s in result["segments"] if s["text"].strip())
    heard = len(words)
    return {"lrc": lyrics.to_lrc(language, lines), "language": language, "lines": len(lines), "heard_words": heard}
