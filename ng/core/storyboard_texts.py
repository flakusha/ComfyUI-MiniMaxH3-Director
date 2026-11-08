"""AI: write the storyboard's texts. The user says what should happen; the AI writes, for every panel, the title and the
caption on the sheet and the shot text H3 renders, plus the sheet's title line.

The shot text follows the same rules as the Studio's shots (split.py): framing first, the camera with amplitude and speed
or a static shot, and a panel longer than 5 s in time beats of its own length, given by the code (shot_share.beat_marks).
The panel times are the user's and are never changed."""
import json

from . import fields
from .ai_json import read_object
from .context_build import widget_text
from .shot_draft import plain_mentions
from .shot_share import beat_marks
from .split import _seconds
from .storyboard import panel_times

SYSTEM = ("Write the panel texts for a storyboard targeting MiniMax H3, which produces "
          "video with sound from detailed shot descriptions. Return only a JSON object.")
TOKENS = 3072


def texts_tokens(max_tokens):
    return max(max_tokens, TOKENS)


def _figures(contexts):
    out = []
    for entry in contexts or ():
        name = widget_text(entry.get("widgets", {}), "name")
        if name and entry.get("media") == "image":
            described = widget_text(entry["widgets"], "description")
            out.append("@%s%s" % (name, ": " + described if described else ""))
    return out


def texts_prompt(board, contexts=()):
    """The request for the AI; board is the Studio's board (its idea, panels and their times)."""
    times = panel_times(board)
    idea = str(board.get("idea") or "").strip()
    if not idea:
        raise ValueError("Write what should happen in the storyboard first.")
    panels = []
    for number, (panel, (start, end)) in enumerate(zip(board["panels"], times), 1):
        entry = {"panel": number, "from_s": start, "to_s": end}
        marks = beat_marks(end - start)
        if marks:
            entry["write_shot_as_beats"] = marks
        for name in ("title", "caption", "shot"):
            if str(panel.get(name) or "").strip():
                entry["user_" + name] = panel[name].strip()
        panels.append(entry)
    figures = _figures(contexts)
    # A long panel's beats stand in the example itself: told only by write_shot_as_beats, qwen3.5:9b wrote a 10 s panel as one block.
    example = ",".join('{"title":"...","caption":"...","shot":"%s"}' % (
        " ".join("%s: ..." % mark for mark in entry["write_shot_as_beats"].split(", ")) if "write_shot_as_beats" in entry else "...")
        for entry in panels)
    rules = [
        "Write the texts for a %d-panel storyboard for a %s second MiniMax H3 video. "
        "Return one JSON object only, with exactly %d panels in the supplied order. "
        "Do not add, omit, reorder, or duplicate panels. Do not change the supplied panel times. "
        "Do not add fields outside the shape. Do not add prose, markdown, or code fences." % (
            len(panels), _seconds(times[-1][1]), len(panels)),
        "Exact shape for this request: "
        '{"title":"...","subtitle":"...","panels":[%s]}. ' % example +
        "Use the supplied shape as the authoritative output shape, including any per-panel beat example shown in it.",
        "The panels are a sequential storyboard: each panel continues where the previous one ends, and together they "
        "show the whole video.",
        "title: a few words in capitals naming the sheet, e.g. 'LUNA AND INFERNA - THE PACT'. Do not use @names.",
        "subtitle: the characters and length in capitals, e.g. 'LUNA-ARC & INFERNA-FLAME - 10 SECONDS'. Do not use @names.",
        "panels[].title: two to four words in capitals naming the moment, e.g. 'INTRO - BOTH CHARACTERS'. Do not use @names.",
        "panels[].caption: one or two English sentences for the sheet: who does what, where, with what expression. "
        "Use plain names when known; do not use @names.",
        "panels[].shot: the English description MiniMax H3 should render for this panel. "
        "Begin with the framing, e.g. 'A full-body shot of ...', then who or what is where in the frame, what they do, "
        "their expression, and the light. Finish with one camera sentence using the supplied camera vocabulary, with small "
        "or large amplitude and slow or fast speed only when meaningful. If the camera does not move, write exactly: "
        "'The camera holds a static shot.' Do not write 'cuts' as a camera action.",
        "A panel with write_shot_as_beats gets its shot as exactly those time beats, in that order, each beginning with its "
        "mark, e.g. for '0–5s, 5–10s': '0–5s: A full-body shot of ... The camera holds a static shot. 5–10s: ...'. "
        "Do not invent, remove, alter, or reorder beat marks. Do not write [mm:ss] stamps that were not supplied.",
        "A panel's user_title, user_caption or user_shot is the user's own text: preserve the user's meaning and make it "
        "only more concrete. Do not change facts, characters, actions, or times.",
        ("Name the characters in panels[].shot by their exact @names: %s. Keep each one's look exactly as described." % "; ".join(figures))
        if figures else "There are no character cards: describe the people concretely (age, hair, clothing) in every shot.",
        "Do not claim generation or media analysis has occurred.",
    ]
    return "\n".join(["- " + rule if index else rule for index, rule in enumerate(rules)]) + (
        "\n\nWhat should happen: %s\nPanels: %s" % (idea, json.dumps(panels, ensure_ascii=False)))


def parse_texts(text, board, contexts=()):
    """{"title", "subtitle", "panels": [{"title", "caption", "shot"}] one per panel}; fewer answered panels keep their own
    texts ("missing" names them)."""
    data = read_object(text, "storyboard", "The AI did not return valid JSON. Try again, or use a larger model.")
    answered = fields.read_list(data, "panels", "storyboard")
    if not answered:
        raise ValueError("The AI wrote no panel texts. Try again.")
    out, unknown = [], []
    for number, panel in enumerate(board["panels"]):
        given = answered[number] if number < len(answered) and isinstance(answered[number], dict) else {}
        entry = {}
        for name in ("title", "caption", "shot"):
            value = given.get(name)
            value = " ".join(str(value).split()) if isinstance(value, str) and value.strip() else str(panel.get(name) or "")
            if name == "shot":
                value, names = plain_mentions(value, contexts)
                unknown += [n for n in names if n not in unknown]
            entry[name] = value
        out.append(entry)
    result = {"title": " ".join(str(data.get("title") or board.get("title") or "").split()),
              "subtitle": " ".join(str(data.get("subtitle") or board.get("subtitle") or "").split()), "panels": out}
    if len(answered) < len(board["panels"]):
        result["missing"] = list(range(len(answered) + 1, len(board["panels"]) + 1))
    if unknown:
        result["unknown_mentions"] = unknown
    return result
