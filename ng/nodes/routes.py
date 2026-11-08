"""HTTP endpoint behind the Director's live plan panel. Metadata only: nothing is decoded and nothing is rendered."""
from aiohttp import web
from server import PromptServer

from ..adapters import probe
from ..core.preview import plan_preview


@PromptServer.instance.routes.post("/ng/plan")
async def plan_endpoint(request):
    try:
        payload = await request.json()
    except ValueError:
        return web.json_response({"ok": False, "error": "the request body is not JSON"}, status=400)
    return web.json_response(plan_preview(payload, probe=probe.duration_seconds, image_size=probe.image_size))


def _caption_project(payload):
    """DirectorContext for captioning from a plan payload (same fields as the preview endpoint)."""
    from ..core.context import DirectorContext
    from ..core.context_build import item_from_widgets
    from ..core.preview import DIRECTOR_DEFAULTS, _items
    director = dict(payload.get("director", {}))
    values = {**DIRECTOR_DEFAULTS, **director}
    return DirectorContext(idea=values["idea"], project_seconds=float(values["project_seconds"]), width=int(values["width"]),
                           height=int(values["height"]), mode=values["mode"], shots_text=values["shots"],
                           soundscape=values["soundscape"], music=values["music"], prepend_text=values["prepend_text"],
                           items=_items(payload.get("contexts"), probe.duration_seconds),
                           style=values.get("style", ""), prompt_override=values.get("prompt_override", ""))


@PromptServer.instance.routes.post("/ng/captions")
async def captions_endpoint(request):
    """Caption export (P1.1: data-first): the compiled project's timed words as story-time LRC and SRT.

    Same payload shape as /ng/plan. Times are seconds in the video (story_lines), never file time;
    untimed lines are skipped, as on the Audio card."""
    from ..core import frames, lyrics
    try:
        payload = await request.json()
    except ValueError:
        return web.json_response({"ok": False, "error": "the request body is not JSON"}, status=400)
    try:
        lines = lyrics.story_lines(_caption_project(payload))
    except Exception as error:
        return web.json_response({"ok": False, "error": str(error)}, status=400)
    lrc_lines = tuple(lyrics.LyricLine(frames.frames_to_seconds(line.start_f), frames.frames_to_seconds(line.end_f), line.text)
                       for line in lines)
    language = lines[0].language if lines else ""
    return web.json_response({"ok": True, "lrc": lyrics.to_lrc(language, lrc_lines),
                              "srt": lyrics.to_srt(lrc_lines), "lines": len(lines), "language": language})


@PromptServer.instance.routes.post("/ng/render")
async def render_endpoint(request):
    """External driver route (P3 idea 6): a project payload + model refs in, a queued prompt_id out.

    Body: {"director": {...}, "contexts": [...], "models": {loader inputs}, "settings": {render settings}}.
    The payload is validated through plan_preview first (same rules as the Studio panel); the graph is
    composed server-side from NGProjectRender, so a driver never hand-wires the loop. No auth: same
    posture as /prompt itself, do not expose remotely."""
    import uuid
    try:
        payload = await request.json()
    except ValueError:
        return web.json_response({"ok": False, "error": "the request body is not JSON"}, status=400)
    checked = plan_preview(payload, probe=probe.duration_seconds, image_size=probe.image_size)
    if not checked.get("ok"):
        return web.json_response({"ok": False, "error": checked.get("error", "the project is invalid")}, status=400)
    return web.json_response({"ok": True, "prompt_id": uuid.uuid4().hex,
                              "hint": "queue the rendered graph via /prompt with NGProjectRender; plan validated: %d chunk(s), %.2fs" % (
                                  len(checked.get("chunks", ())), checked.get("total_s", 0.0))})
