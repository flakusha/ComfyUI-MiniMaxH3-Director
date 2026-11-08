"""ASR backend selection for lyric timing.

Backends are picked by name; the default is always Whisper. An optional backend
(Qwen3-ASR + ForcedAligner) is only selected when its import and weights are
present; otherwise selection falls back to Whisper with a clear log line.
No downloads, no new deps.

Contract (asr-modernization.md §1.3): every backend returns the same
`(language, [(word, start_s, end_s), ...])` triples that `lyrics.align`
consumes; whisper stays the only clock and the stored LRC stays file-time.

Core stays free of ComfyUI: availability probes take injected callables
(version string, model-name lister) so no `folder_paths` import lives here.
"""
import logging
import os

SAMPLE_RATE = 16000
ENV_VARS = ("H3_ASR_BACKEND", "H3_ASR_ENGINE")

log = logging.getLogger(__name__)

_REGISTRY = {}


def register(name, available, factory=None):
    """Register a backend: `available()` -> bool, `factory` -> backend object
    with `transcribe(samples, known, language, chosen, device)`."""
    _REGISTRY[name.lower()] = (available, factory)


def _whisper_available():
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


register("whisper", _whisper_available)


def transformers_version_ok(version=""):
    """True when the transformers version string is >= 5.13 (the first with `qwen3_asr`)."""
    try:
        return tuple(int(part) for part in str(version).split(".")[:2]) >= (5, 13)
    except (ValueError, AttributeError):
        return False


def qwen3_dirs(list_names=None):
    """(asr model dir, aligner model dir) from an injected model-name lister.

    `list_names(folder)` returns the registered names of one model folder; None keeps
    the pair unavailable. The caller (lyrics_timing, which already lives in adapters)
    passes `folder_paths.get_filename_list`. Dirs live under models/asr (see js/ng/models.json)."""
    if list_names is None:
        return None, None
    asr_names = list(list_names("asr") or [])
    asr = next((name for name in asr_names if "qwen3" in str(name).lower() and "align" not in str(name).lower()), None)
    aligner = next((name for name in asr_names if "align" in str(name).lower()), None)
    return asr, aligner


def _qwen3_available(list_names=None):
    """True when the Qwen3-ASR + ForcedAligner pair can run here.

    Needs transformers >= 5.13 and both HF folders registered; the folder check
    runs through the injected `list_names` so core never imports ComfyUI."""
    try:
        import transformers
    except ImportError:
        return False
    if not transformers_version_ok(getattr(transformers, "__version__", "")):
        return False
    asr, aligner = qwen3_dirs(list_names)
    return asr is not None and aligner is not None


def _qwen3_factory(source, text, device, chosen=""):
    """Qwen3-ASR transcript + ForcedAligner word times, in Whisper's shape.

    Registered by ng.adapters.lyrics_timing (adapter layer, folder_paths OK); this stub stays so
    `select` resolves the name before that import. Any failure inside the optional backend raises
    ValueError naming it."""
    raise ValueError("Qwen3-ASR backend is registered but its weights are not on disk yet; "
                     "unset H3_ASR_BACKEND (or place the Qwen3-ASR + ForcedAligner folders) to use Whisper.")


register("qwen3", _qwen3_available, _qwen3_factory)
register("qwen3-asr", _qwen3_available, _qwen3_factory)


def get(name):
    return _REGISTRY.get((name or "").lower())


def requested_engine(engine=None):
    """Explicit engine argument wins, then env, then the Whisper default."""
    if engine:
        return str(engine).strip().lower()
    for var in ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value.lower()
    return "whisper"


def select(engine=None, list_names=None):
    """(name, factory) for the requested backend, Whisper on any absence.

    `list_names(folder)` lists one model folder's registered names; the Qwen3 probe
    uses it to check its two HF folders. Factory may be None (caller resolves by name).
    Falls back to Whisper with a warning when the requested backend is unknown or its
    weights/imports are unavailable, so a missing optional backend never changes behavior.
    """
    name = requested_engine(engine)
    entry = _REGISTRY.get(name)
    if entry is not None:
        available, factory = entry
        try:
            if name in ("qwen3", "qwen3-asr"):
                ok = bool(available(list_names)) if list_names is not None else False
            else:
                ok = bool(available())
        except Exception:
            ok = False
        if ok:
            return name, factory
        log.warning("[H3 Director] ASR backend '%s' requested but unavailable (missing weights or imports); falling back to Whisper.",
                    name)
    elif name != "whisper":
        log.warning("[H3 Director] ASR backend '%s' is not registered; falling back to Whisper.", name)
    return "whisper", _REGISTRY.get("whisper", (None, None))[1]
