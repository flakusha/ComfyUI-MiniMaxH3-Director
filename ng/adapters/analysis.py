"""Reuse legacy text/vision transport. Audio uses the compatible input_audio message only when declared."""
import asyncio
import base64
import io
import wave

import aiohttp
import numpy as np
from PIL import Image

from ... import minimax_media as legacy
from ..core import providers
from ..core.analysis import ProviderConfig, analysis_prompt, parse_suggestion
from ..core.fields import need_dict
from ..core.context_build import widget_text
from ..core.system_prompts import framed, for_node
from . import preview_media
from .audio_decode import decode_audio


def endpoint(config, purpose=""):
    """Where this provider actually lives, and which model to ask for.

    The registry owns the address. `_PROVIDER_DEFAULTS` only ever listed Ollama, LM Studio and
    "custom", so indexing it with a provider id from the registry raised KeyError for OpenAI,
    Claude, Gemini, OpenRouter and llama.cpp - every provider the panel now offers.

    Both are checked here rather than left to aiohttp. A remote provider with no address and no
    model would otherwise become a connection error against an empty string, which names neither the
    missing setting nor the card that needed it.

    `purpose` picks an optional per-purpose model ("text", "caption", "audio"); an empty field,
    an unknown purpose, or no purpose at all all fall through to the Model above, so callers that
    do not care keep the exact behaviour they had.
    """
    entry = providers.get(config.provider)
    defaults = legacy._PROVIDER_DEFAULTS.get(config.provider) or {}
    url = legacy.normalize_base_url(config.base_url, entry.base_url or defaults.get("url", ""))
    override = getattr(config, "model_" + purpose, "") if purpose else ""
    model = override or config.model or entry.sample_model or defaults.get("model", "")
    # `custom` is the only provider with neither, so it is the only one where this can fire. Checked
    # on the provider rather than on an empty url, because a local provider always has an address.
    if entry.id == "custom" and not url:
        raise ValueError("Custom URL is not a usable address. Enter the provider's address.")
    return entry, url, model


def audio_error(status, model, body_text=""):
    """What a failed audio request tells the card, naming the likely cause, not the status.

    llama.cpp answers 500 "audio input is not supported" when the addressed model has no
    audio projector (mmproj) attached - either the wrong model got the sound (empty Audio
    model override, so the text/vision model was asked) or the audio model itself runs
    without its mmproj. The server's own body is quoted (cut short) so a different fault
    stays recognisable instead of being relabelled."""
    body = str(body_text or "").strip()
    if status == 500 and "audio" in body.lower():
        return ("The server refused the audio (HTTP 500, model '%s'): it reports no audio input. "
                "Check the H3 AI Loader's Audio model override - empty means the Model above got the sound "
                "and a text/vision model cannot hear it - and that the audio model loads with its mmproj "
                "(llama-swap entry needs the projector file / --mmproj flag); server said: %s"
                % (model, body[:200]))
    return ("Audio analysis failed (HTTP %d, model '%s'); check whether the model supports input_audio.%s"
            % (status, model, (" server said: " + body[:200]) if body else ""))


async def audio_generate(config, prompt, source, key, wav=None, system=None):
    if config.provider == "ollama":
        raise ValueError("This Ollama transport has no audio input; choose an input_audio-compatible provider or describe the sound manually.")
    # NOTE (intentional, 2026-11-07): Analyze sends the 15 s preview even when the card has a Trim window;
    # Transcribe sends the windowed range. A trimmed card's Analyze hears audio the video does not contain,
    # while Transcribe hears the right range - kept because Analyze answers "what is this sound" from the card's
    # file, while Transcribe writes down the exact range the video plays. Plumbing trim into analyze would
    # change the pinned 15 s cap (test_analysis pins it); revisit only with that decision.
    wav = wav if wav is not None else await asyncio.to_thread(preview_media.audio_wav, source)
    entry, url, model = endpoint(config, "audio")
    if not url or not model:
        raise ValueError("Set the provider's address and audio-capable model name.")
    # The same role text and the same schema framing the other media send. Audio had neither until
    # now, which made this the one card whose answer the parser could not rely on - and an audio
    # card is exactly where a "who is speaking" guess matters most.
    messages = [{"role": "system", "content": system if system is not None else framed(for_node("context", "audio"))},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}}]}]
    payload = {"model": model, "stream": False, "max_tokens": config.max_tokens, "messages": messages}
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)

    async def send():
        async with aiohttp.ClientSession(headers=legacy.registry_auth_headers(entry.auth, key)) as session:
            async with session.post(legacy.chat_completions_url(url), json=payload, timeout=timeout) as response:
                if response.status in (401, 403):
                    raise ValueError("The provider refused authentication; set api_key_env to a variable available to ComfyUI.")
                if response.status != 200:
                    body_text = await response.text()
                    raise ValueError(audio_error(response.status, model, body_text))
                return await response.json()

    body = await legacy.interruptible(send())
    try:
        return legacy.strip_thinking(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        raise ValueError("The audio provider returned an unexpected response.")


# NOTE: a provider could in principle return transcription timestamps (an OpenAI-style verbose
# transcribe format). Deliberately not built: this action sends plain text and accepts plain text only,
# and Whisper stays the one clock (decision 3, 2026-11-07). A verbose-timestamp variant is future work
# behind a provider that really offers such a format - and even then its times would be checked against
# the file-time LRC contract, never pasted in over Whisper's stamps.
def transcript_lines(text):
    """The words, one line per sung or spoken line, the way the Lyrics field holds them.

    A model asked for plain lines still sometimes wraps the answer in a code fence or pads it with
    blank lines; both are dropped. The words themselves are kept verbatim, because Whisper aligns
    exactly what is written here - tidying them would break the match between the words and the sound.
    """
    lines = []
    for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        clean = line.strip()
        if clean and not clean.startswith("```"):
            lines.append(clean)
    return "\n".join(lines)


def transcribe_prompt(content):
    """What the audio model is asked on a Transcribe action, with the card's description for context.

    The role text comes from the shared prompt table (`for_node("context", "audio_transcribe")`) like
    every other purpose framing; `framed`'s schema rule is deliberately left out, because it demands
    JSON and this answer is plain words for the Lyrics field, which no parser reads.
    """
    ask = ("Write down exactly what is said or sung, in the original words and language, one line per "
           "spoken or sung line, in the order heard. Do not add times, headings, speaker names or commentary.")
    if content:
        ask += "\nThe card describes the sound as: %s" % content
    return ask


def window_wav(source, start_s, end_s):
    """The sound to transcribe as a base64 wav (16 kHz mono), [start_s, end_s) seconds; end 0 = the file
    end. One decode (`decode_audio`) also serves the card's trimmed output and the Whisper timing, so
    the range sent here is exactly the range the card plays - and the whole file when no range is set,
    not the 15 s preview the analysis answer is allowed.
    """
    samples, _rate = decode_audio(source, start_s, end_s, rate=preview_media.AUDIO_RATE, layout="mono")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(preview_media.AUDIO_RATE)
        pcm = np.clip(samples[0].numpy(), -1.0, 1.0)
        stream.writeframes(np.round(pcm * 32767.0).astype("<i2").tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def transcribe(payload, *, api_key=None):
    """The AI Provider's audio model writes down what it hears - a second hearing beside Whisper, which
    stays the only clock (decision 3). The card's selected range is sent when it has one, the whole file
    when it has not; the answer comes back as plain lyric lines for the card to show for review, and
    "Time with Whisper" then stamps them. Nothing here writes the card's lyrics: the text returns to
    the route, and accepting it is the card's own action.
    """
    need_dict(payload, "request")
    config = ProviderConfig.from_dict(payload["provider"])
    widgets = need_dict(payload.get("widgets"), "request.widgets")
    source = str(widgets.get("audio") or "")
    if not source:
        raise ValueError("Choose an audio file first; there is nothing to transcribe.")
    refused = _capability_error(config, "audio")
    if refused:
        raise ValueError(refused)
    start, end = max(0.0, float(payload.get("start_s") or 0.0)), max(0.0, float(payload.get("end_s") or 0.0))
    if end and end <= start:
        raise ValueError("Source end must be later than source start; 0 means the file end.")
    key = legacy.resolve_api_key({"api_key_env": config.api_key_env}) if api_key is None else api_key
    prompt = transcribe_prompt(str(widgets.get("content") or ""))
    role = for_node("context", "audio_transcribe")
    try:
        wav = await asyncio.to_thread(window_wav, source, start, end)
        answer = await audio_generate(config, prompt, source, key, wav=wav, system=role)
    except (ValueError, legacy.VLMError) as error:
        raise ValueError(str(error).replace(key, "[redacted]") if key else str(error)) from None
    return {"text": transcript_lines(answer)}


def images(media, source):
    if media == "video":
        return preview_media.video_images(source)
    with Image.open(preview_media.require_path(source)) as image:
        return [preview_media.jpeg(image)]


def _capability_error(config, media):
    """Why this provider cannot read this medium, in the user's words, or None when it can.

    The registry knows what a provider accepts; the old two checkboxes only recorded what the user
    believed, and the request then punished them for being wrong. One registry check replaces both:
    the question has one answer per provider, and that answer is a measured fact.

    `custom` is the honest exception. It is "I type the address myself", so nothing is known about
    what sits behind it - refusing a custom provider outright would break the only way to reach a
    server we do not ship. It therefore keeps the stored flags, which for that provider are the
    only evidence available.

    Both messages name the way out, not just the fault. "This provider cannot read audio" leaves the
    user with nothing to do; naming the Text card that needs no provider at all does not.
    """
    entry = providers.get(config.provider)
    if entry.id != "custom":
        if not entry.supports_media(media):
            return providers.UNSUPPORTED_HINT.get(media, "This provider cannot read %s." % media)
        return None
    if media in ("image", "video") and not config.vision:
        return ("Vision is off on the H3 AI Loader, so the AI is not shown this picture. Switch Vision on if the model "
                "reads images, or write the description yourself.")
    if media == "audio" and not config.audio:
        return ("Audio is off on the H3 AI Loader, so the AI is not given this sound. Switch Audio on only for a model "
                "that accepts audio (input_audio), or write the description yourself.")
    return None


async def analyze(payload, *, api_key=None):
    need_dict(payload, "request")
    config = ProviderConfig.from_dict(payload["provider"])
    media, widgets = payload["media"], payload["widgets"]
    need_dict(widgets, "context.widgets")
    if media not in ("image", "video", "audio", "text"):
        raise ValueError("Choose an image, video, audio or text context.")
    refused = _capability_error(config, media)
    if refused:
        raise ValueError(refused)
    prompt = analysis_prompt(media, widgets.get("content", ""), payload.get("offer"))
    key = legacy.resolve_api_key({"api_key_env": config.api_key_env}) if api_key is None else api_key
    try:
        if media == "audio":
            answer = await audio_generate(config, prompt, widgets.get("audio", ""), key)
        else:
            image_list = await asyncio.to_thread(images, media, widgets.get(media, "")) if media in ("image", "video") else []
            if media == "text":
                prompt += "\n\nContext text:\n" + widget_text(widgets, "text")
            entry, url, model = endpoint(config, {"image": "caption", "video": "caption",
                                                  "audio": "audio"}.get(media, "text"))
            answer = await legacy.vlm_generate(
                image_list, prompt, config.provider, url, model, api_key=key, auth=entry.auth,
                # The role and the schema framing come from one place, and the framing is appended
                # after the role. Sending the role but not the framing is how an answer arrives as
                # prose that parse_suggestion cannot read.
                system_prompt=framed(for_node("context", media)),
                **wire(config))
        return parse_suggestion(answer, payload.get("media", "text")).to_dict()
    except (ValueError, legacy.VLMError) as error:
        raise ValueError(str(error).replace(key, "[redacted]") if key else str(error)) from None


def wire(config):
    """The settings that actually go out, named the way `vlm_generate` names them.

    Built from the config rather than spelled out here, because the sampling matrix in the registry
    is what decides which of these a provider accepts at all. A field Ollama would reject is
    therefore never handed to a transport that would send it.
    """
    values = providers.provider_payload(config.provider, {
        "max_tokens": config.max_tokens, "timeout_seconds": config.timeout_seconds,
        "keep_alive": config.keep_alive, "seed": config.seed, "fixed_seed": config.fixed_seed,
        "temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k,
        "min_p": config.min_p, "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty})
    # provider_payload() nests the sampling knobs under "options" for Ollama, because that is the wire format; it
    # travels to vlm_generate whole as `sampling`. Only max_tokens, keep_alive and timeout used to be handed on, so
    # Temperature, Top P/K, Min P, the penalties and a fixed Seed never left ComfyUI. timeout_seconds belongs to the
    # HTTP client and is passed on its own, or the request would fall back to the library default.
    flat = {name: values[name] for name in ("max_tokens", "keep_alive") if name in values}
    return {"max_tokens": flat.get("max_tokens", config.max_tokens),
            "keep_alive": flat.get("keep_alive", 0),
            "timeout": config.timeout_seconds,
            "sampling": values}
