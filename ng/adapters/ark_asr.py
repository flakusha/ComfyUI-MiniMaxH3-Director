"""Time a song's lyrics with Doubao Seed-ASR 2.0 (Volcengine cloud, opt-in via H3_ASR_BACKEND=ark).

The recorded-file API (openspeech.bytedance.com, resource volc.seedasr.auc) runs submit + query and only accepts an
audio URL, so the song must be reachable online under the name the Audio card shows it: upload it once to any web
address (for example a Volcengine TOS bucket) and set H3_ARK_AUDIO_URL_BASE to that place; the card's file name is
appended. Keys live in the environment, never in the workflow: H3_ARK_API_KEY (new console), or H3_ARK_APP_ID +
H3_ARK_ACCESS_KEY (legacy console). No model files, no GPU: `device` is ignored, the recognition runs in ByteDance's
cloud. Whisper stays the default and the fallback whenever the variables are missing.
"""
import asyncio
import json
import os
import time
import uuid

import aiohttp

from ..core import lyrics

SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
RESOURCE_ID = "volc.seedasr.auc"          # 2.0 (volc.bigasr.auc would be model 1.0)
FORMATS = ("wav", "mp3", "ogg", "spx", "amr", "aac", "m4a")
POLL_TIMEOUT_S = 300                      # a song answers in far less; longer means something is wrong
POLL_STEP_S = 2.0
DONE, WORKING = "20000000", ("20000001", "20000002")


def ark_available():
    """True when an API key, or the legacy APP ID + access token pair, is in the environment."""
    return bool(os.environ.get("H3_ARK_API_KEY") or
                (os.environ.get("H3_ARK_APP_ID") and os.environ.get("H3_ARK_ACCESS_KEY")))


def _headers(task_id):
    """The new console authenticates with one API key; the legacy one with APP ID + access token."""
    headers = {"X-Api-Resource-Id": RESOURCE_ID, "X-Api-Request-Id": task_id}
    if os.environ.get("H3_ARK_API_KEY"):
        headers["X-Api-Key"] = os.environ["H3_ARK_API_KEY"]
    else:
        headers["X-Api-App-Key"] = os.environ.get("H3_ARK_APP_ID", "")
        headers["X-Api-Access-Key"] = os.environ.get("H3_ARK_ACCESS_KEY", "")
    return headers


def _audio_url(source):
    """The song's online address: the base the user uploaded it under, plus the card's file name."""
    base = os.environ.get("H3_ARK_AUDIO_URL_BASE", "").strip()
    if not base:
        raise ValueError("Ark Seed-ASR needs H3_ARK_AUDIO_URL_BASE: the address the song is served from, for example "
                         "https://my-bucket.tos-cn-beijing.volces.com/songs/ (the card's file name is appended).")
    return "%s/%s" % (base.rstrip("/"), os.path.basename(source))


def _submit_body(source, known):
    """The task: word times on, the user's own lines as hotwords, the heard words kept close to the singing."""
    fmt = os.path.splitext(source)[1][1:].lower()
    if fmt not in FORMATS:
        raise ValueError("Ark Seed-ASR reads %s files; upload the song in one of those formats (the card's file is .%s)."
                         % (", ".join(FORMATS), fmt or "unknown"))
    body = {"audio": {"url": _audio_url(source), "format": fmt},
            "request": {"model_name": "bigmodel", "show_utterances": True,
                        # ITN and punctuation rewrite the words into written form; align() matches the user's typed
                        # lyrics against the heard words, so the plain sung form matches better.
                        "enable_itn": False, "enable_punc": False}}
    if known:
        body["request"]["corpus"] = {"context": json.dumps({"hotwords": [{"word": line} for line in known]})}
    return body


async def _submit(session, source, known):
    """Hand the task over; the answer body is empty, success lives in the X-Api-Status-Code header."""
    task_id = str(uuid.uuid4())           # the same id rides every query for this task
    headers = _headers(task_id)
    headers["X-Api-Sequence"] = "-1"
    async with session.post(SUBMIT_URL, json=_submit_body(source, known), headers=headers) as response:
        status = response.headers.get("X-Api-Status-Code", "")
        message = response.headers.get("X-Api-Message", "")
        if response.status != 200 or status != DONE:
            raise ValueError("Ark Seed-ASR refused the task (HTTP %s, status %s, %s); check the key and the audio address."
                             % (response.status, status or "?", message or "no message"))
    return task_id


async def _query(session, task_id):
    """The finished answer's JSON, or None while the task is queued or still running."""
    async with session.post(QUERY_URL, json={}, headers=_headers(task_id)) as response:
        status = response.headers.get("X-Api-Status-Code", "")
        message = response.headers.get("X-Api-Message", "")
        if response.status != 200:
            raise ValueError("Ark Seed-ASR's query failed (HTTP %s, %s)." % (response.status, message or "no message"))
        if status in WORKING:
            return None
        try:
            payload = await response.json(content_type=None)
        except ValueError as error:
            raise ValueError("Ark Seed-ASR's answer was not readable JSON (%s); a proxy or gateway answered instead of the API." % error) from error
        if status != DONE:
            # 20000003 = silent audio, 450000xx = bad request (format, size), 550xxxxx = the service itself; the code
            # table lives in Volcengine's recorded-file recognition docs.
            raise ValueError("Ark Seed-ASR failed (status %s, %s)." % (status or "?", message or "no message"))
        return payload


async def _recognize(source, known):
    """Submit and poll until the answer lands, at most POLL_TIMEOUT_S (the task keeps running past that)."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        task_id = await _submit(session, source, known)
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while True:
            payload = await _query(session, task_id)
            if payload is not None:
                return payload
            if time.monotonic() > deadline:
                raise ValueError("Ark Seed-ASR is still busy after %d s; the task keeps running, try Timing again in a moment."
                                 % POLL_TIMEOUT_S)
            await asyncio.sleep(POLL_STEP_S)


def _words(payload):
    """(language, [(word, start_s, end_s)], utterances) out of the finished answer; the API counts milliseconds."""
    result = payload.get("result") if isinstance(payload, dict) else None
    utterances = (result or {}).get("utterances") or []
    words = [(word["text"], word["start_time"] / 1000.0, word["end_time"] / 1000.0)
             for utterance in utterances for word in (utterance.get("words") or [])]
    if not words:
        raise ValueError("Ark Seed-ASR's answer carries no word times (a wrong audio address, or no audible singing).")
    return (result or {}).get("language", ""), words, utterances


def ark_transcribe(source, text, device, chosen=""):
    """Doubao Seed-ASR's word times, in Whisper's shape (the /ng/lyrics route returns this dict directly).

    The user's own lines keep their words and take only the times (lyrics.align); without lines the API's utterances
    become the lines. `device` and `chosen` are the registry's Whisper-shaped arguments; neither applies here."""
    if not ark_available():
        raise ValueError("Ark Seed-ASR needs H3_ARK_API_KEY (new console) or H3_ARK_APP_ID + H3_ARK_ACCESS_KEY (legacy "
                         "console) in ComfyUI's environment.")
    known = lyrics.plain_lines(text)
    try:
        payload = asyncio.run(_recognize(source, known))
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
        raise ValueError("Ark Seed-ASR could not be reached (%s); check the network and H3_ARK_AUDIO_URL_BASE." % error) from error
    heard, words, utterances = _words(payload)
    language, _ = lyrics.parse(text)
    if known:
        lines = lyrics.align(known, words, payload.get("audio_info", {}).get("duration", 0) / 1000.0)
    else:
        lines = tuple(lyrics.LyricLine(round(u["start_time"] / 1000.0, 2), round(u["end_time"] / 1000.0, 2), u["text"].strip())
                      for u in utterances if u["text"].strip())
    language = language or heard
    return {"lrc": lyrics.to_lrc(language, lines), "language": language, "lines": len(lines), "heard_words": len(words)}


from ..core import asr as _asr_registry
_asr_registry.register("ark", ark_available, ark_transcribe)
_asr_registry.register("seed-asr", ark_available, ark_transcribe)
