"""Make H3 prompt: plan a whole video from a direction, the Content cards and an optional request, in one AI call."""
import json
import re

from . import fields
from .ai_json import read_object
from .auto_direct import story
from .context_build import card_id, card_words, widget_text
from .directions import DIRECTIONS, LANGUAGES, SPEECH, check_allowed, direction, shot_count
from .split import BEAT_RULE, CAMERA_RULE, _seconds, parse_split, shots_example

SYSTEM = ("You are the director of a short video for MiniMax H3, which produces video with sound from a single detailed "
          "prompt. Plan the video concretely, visually, and aurally from the user's reference pictures, direction, and request. Respond with a single JSON object following the provided schema.")
PLAN_TOKENS = 3072
# A place picture binds to the shots showing it; a figure stays in every part (a shot saying only "she" still needs it).
PLACES = ("environment", "location", "prop", "object", "style", "lighting", "composition")
DEFAULT_QUICK = {"direction": "auto", "speech": "none", "language": "English"}


def plan_tokens(max_tokens):
    return max(max_tokens, PLAN_TOKENS)


def read_quick(payload):
    """The Make H3 prompt settings of a request: direction, speech, language; unknown values fall back to the defaults."""
    raw = payload.get("quick") if isinstance(payload, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    quick = dict(DEFAULT_QUICK)
    if isinstance(raw.get("direction"), str) and raw["direction"] in DIRECTIONS:
        quick["direction"] = raw["direction"]
    if isinstance(raw.get("speech"), str) and raw["speech"] in SPEECH:
        quick["speech"] = raw["speech"]
    if raw.get("language") in LANGUAGES:
        quick["language"] = raw["language"]
    return quick


def _roles(widgets, key):
    try:
        values = json.loads(widgets.get(key) or "[]")
    except ValueError:
        values = []
    values = [value for value in values if isinstance(value, str) and value] if isinstance(values, list) else []
    return values or ([widgets[key[:-6]]] if widgets.get(key[:-6]) else [])


def pictures(contexts):
    """The image cards the model is shown, in order: (card, usage roles, content roles)."""
    out = []
    for entry in contexts:
        if entry.get("media") != "image" or not entry["widgets"].get("image"):
            continue
        out.append((entry, _roles(entry["widgets"], "usage_roles"), _roles(entry["widgets"], "content_roles")))
    return out


FRAMES = ("first_frame", "last_frame", "keyframe", "storyboard")


def kinds(contexts):
    """{card name: (kind, owner name)} of the pictures: "frame" (a first/last frame or keyframe), "view" (another picture of
    a subject, "Belongs to"), "place" (a place or thing) or "figure". Only figures and places are named with @ in a shot;
    a view is named by its subject and bound to the shots that use it (uses)."""
    names = {card_id(entry): entry["widgets"].get("name") for entry in contexts}
    out = {}
    for entry, usage, content in pictures(contexts):
        owners = [names.get(i) for i in re.split(r"[\s,\[\]\"]+", entry["widgets"].get("subject_ids") or "") if names.get(i)]
        kind = ("frame" if set(usage) & set(FRAMES) else "view" if owners else
                "place" if content and set(content) <= set(PLACES) else "figure")
        out[entry["widgets"].get("name") or ""] = (kind, owners[0] if owners else "")
    return out


KIND_WORDS = {"frame": "", "view": "another view of @%s", "place": "a place or thing to use", "figure": "a character"}


def _picture_line(number, entry, usage, content, shown, kind):
    widgets = entry["widgets"]
    name = widgets.get("name") or "Picture %d" % number
    sort, owner = kind
    words = {"first_frame": "the FIRST FRAME of the video", "last_frame": "the LAST FRAME of the video"}
    what = (" and ".join(words.get(u, u) for u in usage if u in FRAMES) if sort == "frame"
            else KIND_WORDS[sort] % owner if sort == "view" else KIND_WORDS[sort])
    described = (widgets.get("description") or "").strip()
    label = "@" + name if sort in ("figure", "place") else '"%s"' % name
    return "%s. %s - %s%s%s" % (number, label, what, ": " + described if described else "", "" if shown else " (not shown to you)")


def _audio_line(entry):
    widgets, usage = entry["widgets"], _roles(entry["widgets"], "usage_roles")
    how = ("played as the video's soundtrack, exactly as it is" if {"lock", "copy"} & set(usage) else
           "voice of %s" % widgets.get("speaker") if "voice" in usage and widgets.get("speaker") else
           "a sound reference (%s)" % ", ".join(usage) if usage else "a sound reference")
    described = (widgets.get("description") or "").strip()
    line = "- @%s - audio, %s; contains %s%s" % (widgets.get("name") or "Audio", how,
                                                  ", ".join(_roles(widgets, "content_roles")) or "sound",
                                                  ": " + described if described else "")
    words = card_words([entry])   # the card's own words (BUG-094): the plan paces its shots and writes its lines by them
    return line + ("; its words, to be used as they are ([mm:ss] = when each line is heard in the video; write the words without it): %s" % " / ".join('"%s"' % word for word in words) if words else "")


def _rule_line(entry):
    """A Text card that is not the story: written into every part as it is, so the plan must fit it (2026-10-18)."""
    usage = [u for u in _roles(entry["widgets"], "usage_roles") if u != "scene"]
    text = widget_text(entry["widgets"], "text").strip()
    return '- "%s" (%s): %s' % (entry["widgets"].get("name") or "Note", ", ".join(usage) or "rule", text) if usage and text else None


def _video_line(entry):
    """A Video card: what H3 takes from it. A control video only steers movement and framing; its source's look is never
    described, or the source's figure turns up in the video (T6, 2026-10-16)."""
    usage, name = _roles(entry["widgets"], "usage_roles"), entry["widgets"].get("name") or "Video"
    if "control" in usage:
        return "- %s - a control video: it steers movement and framing of the shots it lies under; do not describe what it shows" % name
    described = (entry["widgets"].get("description") or "").strip()
    return "- @%s - video, %s%s" % (name, ", ".join(usage) or "reference", ": " + described if described else "")


def soundtrack(contexts):
    """True when an audio card plays as the soundtrack (Lock or Copy): its music then is the music."""
    return any(entry.get("media") == "audio" and ({"lock", "copy"} & set(_roles(entry["widgets"], "usage_roles"))) for entry in contexts)


_PACE = "a sung line of 4 to 7 words takes about 3 seconds, so give each shot one line per 3 seconds of its length, and none to a shot shorter than 3 seconds"


def _speech_rule(quick, sung, played, has_request, words=(), played_words=()):
    speech, language = quick["speech"], quick["language"]
    given = " If the request gives words to say (in quotes or after 'says:'), use exactly those words in their own language." if has_request else ""
    if speech == "none":
        return "Speech: nobody speaks or sings. lines is [] in every shot."
    if speech == "dialogue":
        return ("Speech: the characters talk to each other or to the camera.%s Otherwise write short, natural lines in %s that "
                "fit the scene: at most one line per shot, about two words per second of the shot. speaker is the exact name of "
                "who speaks (without @); delivery is says, whispers, shouts or murmurs." % (given, language))
    if speech == "voiceover":
        return ("Speech: a narrator speaks off-screen.%s Otherwise write short narration in %s: at most one line per shot, about "
                "two words per second. speaker is empty; delivery is says." % (given, language))
    if sung:
        return "Speech: the sung lines are already placed at their times (below); lines is []. Show the singer singing them."
    singer = "speaker is the exact name of the singer (without @); delivery is sings."
    if played and played_words:
        # the soundtrack's own words carry no Whisper times: the shots write the lines and pace them (BUG-094)
        return "Speech: the singer sings the soundtrack's own words (see Sound) as they are.%s %s. %s" % (given, _PACE, singer)
    if played:
        return ("Speech: the singer sings along to the soundtrack, whose words are not known: write no lines ([]) and show the "
                "singing performance (mouth, breath, expression) in the action.")
    # A rendered 10 s song (2026-10-06) sang every word, but the 6-word line of a 2 s last shot was cut off at the end.
    if words:
        return "Speech: the character sings the words carried on an audio card (see Sound) as they are.%s %s. %s" % (given, _PACE, singer)
    return "Speech: the character sings.%s Otherwise write short original lyrics in %s: %s. %s" % (given, language, _PACE, singer)


def _frame_rule(found):
    first = [entry["widgets"].get("name") for entry, usage, _ in found if "first_frame" in usage]
    last = [entry["widgets"].get("name") for entry, usage, _ in found if "last_frame" in usage]
    who = " Call the people in it by what they are (for example 'the white-haired elf'), not by the picture's name."
    if first and last:
        return ("The video moves continuously from \"%s\" (its first frame) to \"%s\" (its last frame): use one shot "
                "unless the request asks for more, and describe the path between them: how pose, framing and light change, "
                "ending on the final pose of the last picture named concretely.%s" % (first[0], last[0], who))
    if first:
        return ("The video starts exactly on \"%s\": shot 1 begins with what that picture shows (same people, pose, place, "
                "framing, light) and develops forward from it.%s" % (first[0], who))
    if last:
        return ("The video ends exactly on \"%s\": the last shot arrives at what that picture shows; name its final pose, "
                "framing and light concretely.%s" % (last[0], who))
    return ""


def _action_rule(found, names):
    if not found or not names:
        return ("action: two or three English sentences of what is visible: begin with the framing (for example 'A medium shot "
                "of a woman ...'), then who is where in the frame, what they do and their expression. Describe each character's "
                "look once, concretely (age, hair, clothing).")
    return ("action: two or three English sentences of what is visible: begin with the framing (for example 'A medium shot of "
            "@Name ...', 'A close-up of ...'), then who is where in the frame, what they do and their expression. Name the "
            "characters and places by their exact @name (%s) whenever they are visible; never write the name of a picture in "
            "quotes. Keep every character's face, hair, clothing and colours exactly as the pictures show them; invent no other "
            "main characters." % ", ".join("@" + n for n in names))


def _rules(quick, seconds, found, names, played, parts, sung, kinds_of):
    plan, (fewest, most) = direction(quick["direction"]), shot_count(seconds, quick["direction"])
    music = ("the soundtrack plays as it is: describe its instruments and tempo in one sentence, or N/A if it has no music."
             if played else "one English sentence of %s; instruments and tempo, no mood words." % plan.music)
    look = ("look: one or two English sentences that every shot shares. Start with the style the pictures are made in, judged "
            "from the pictures themselves: a photograph is 'Live-action, photorealistic'; a drawing, painting or anime illustration "
            "is '2D anime' or 'illustrated'; only a rendered 3D image is '3D CG'. Then the place where the video happens, its light "
            "and colour palette. No action. The pictures only say how the characters and places look: when a picture is a "
            "character sheet, a collage, has text or a plain studio background, the video happens in a real place that fits the "
            "request and the direction, never on a sheet or in front of a plain backdrop." if found else
            "look: one or two English sentences that every shot shares: the visual style (for example 'Live-action, cinematic', "
            "'2D anime', '3D CG'), the place, the light and the colour palette. No action.")
    rules = [
        look,
        "shots: %s, ordered and non-overlapping, end to end from 0 to %s s. Start a new shot only when the view changes "
        "(another framing, angle or place); a slow change of distance stays inside one shot as camera movement. Do not "
        "write timestamps, [mm:ss], cut labels, or absolute times in prose." % (
            "exactly 1 shot" if most == 1 else "%d to %d shots" % (fewest, most), _seconds(seconds)),
        _action_rule(found, names),
        CAMERA_RULE,
        BEAT_RULE,
        "lines: each entry is {\"speaker\":\"\",\"delivery\":\"says\",\"language\":\"English\",\"text\":\"...\"}; one sentence "
        "or sung phrase per entry, word for word, in the shot where it is heard. speaker is the exact supplied name, or \"\" "
        "only for a narrator/voiceover; a shot with no line uses []. Never write \"N/A\" for speaker or lines.",
        "soundscape: one or two English sentences of ambience and physical sounds (footsteps, cloth, wind, impacts, breathing); "
        "no speech, no singing, no music.",
        "music: " + music + ' Write "N/A" only in this field, never in another.',
    ]
    usable = [name for name, (sort, _) in kinds_of.items() if sort in ("view", "place")]
    frame = _frame_rule(found)
    if frame:
        rules.insert(2, frame)
    if usable:
        rules.insert(3, "uses: the names of the pictures among %s whose view or place this shot shows, for example [\"%s\"]; "
                     "[] when none. Use each of them in at least one shot." % (", ".join('"%s"' % n for n in usable), usable[0]))
    if quick["direction"] == "spicy":
        rules.append("Everyone in the video is an adult, 18 or older: call them women and men, never girls or boys. If the "
                     "pictures or the request show or suggest anyone under 18, answer only {\"refused\":\"minor\"}.")
    if len(parts) > 1:
        rules.append("The video is rendered in separate parts with hard cuts between them: %s. A shot may cross a part boundary; "
                     "prefer a cut at a part boundary when the sequence allows it."
                     % ", ".join("%s-%s s" % (_seconds(a), _seconds(b)) for a, b in parts))
    if sung:
        rules.append("These sung lines are already placed at their times, so do NOT put them into lines; let the action follow "
                     "them: %s." % "; ".join('%s-%s s "%s"' % (_seconds(a), _seconds(b), words) for a, b, _, words in sung))
    return rules


def _speaks(quick, sung, played, played_words=()):
    """True when the answer's example shot carries a line: speech is wanted and no line is already placed."""
    return quick["speech"] != "none" and not sung and not (quick["speech"] == "singing" and played and not played_words)


def _shape(quick, sung, played, kinds_of, undescribed=(), played_words=()):
    """The answer's JSON shape. With speech, the example shot carries a line: an empty list in the example made qwen3.5:9b
    leave a song without lyrics."""
    speaker = "" if quick["speech"] == "voiceover" else next((n for n, (k, _) in kinds_of.items() if k == "figure" and n), "Name")
    line = '{"speaker":"%s","delivery":"%s","language":"%s","text":"..."}' % (speaker, "sings" if quick["speech"] == "singing" else "says",
                                                                                 quick["language"])
    uses = ',"uses":[]' if any(sort in ("view", "place") for sort, _ in kinds_of.values()) else ""
    describe = ',"describe":{"%s":"..."}' % undescribed[0] if undescribed else ""
    return '{"look":"...","shots":%s,"soundscape":"...","music":"..."%s}' % (
        shots_example(uses, line if _speaks(quick, sung, played, played_words) else ""), describe)


def _opening_lines(quick, seconds, kinds_of, undescribed, played, played_words, request, words, plan, sung=()):
    lines = ["Plan a %s second video. Return only one JSON object that matches the shape below exactly. "
             "Do not add prose, markdown, extra fields, or invented facts." % _seconds(seconds),
             "Shape (authoritative for this request):", _shape(quick, sung, played, kinds_of, undescribed, played_words)]
    if _speaks(quick, sung, played, played_words):
        # An empty example list made qwen3.5:9b leave a song without lyrics: with speech wanted and no line
        # placed, the shape's example line is the structure to use.
        lines.append("The shape example shows the spoken/sung line structure: use it whenever the Request supplies speech or "
                     "song and no line is already placed.")
    return lines + ["", "Direction: %s. %s" % (plan.label, plan.plan),
                    _speech_rule(quick, sung, played, bool(request.strip()), words, played_words),
                    "Request: %s" % (request.strip() or "(none - build the video from the pictures and the direction)")]


def _media_lines(lines, found, shown, kinds_of, contexts):
    if found:
        lines += ["", "Pictures%s:" % (" (attached in this order)" if shown >= len(found) else
                                         " (the first %d attached in this order)" % shown if shown else "")]
        lines += [_picture_line(number, entry, usage, content, number <= shown, kinds_of[entry["widgets"].get("name") or ""])
                  for number, (entry, usage, content) in enumerate(found, 1)]
    sounds = [_audio_line(entry) for entry in contexts if entry.get("media") == "audio"]
    if sounds:
        lines += ["", "Sound:"] + sounds
    videos = [_video_line(entry) for entry in contexts if entry.get("media") == "video"]
    if videos:
        lines += ["", "Videos:"] + videos
    notes = [line for line in (_rule_line(entry) for entry in contexts if entry.get("media") == "text") if line]
    if notes:
        lines += ["", "User rules: the rules below are appended to every rendered part exactly as written. Plan shots "
                  "that obey them. Do not restate, paraphrase, or summarize them inside the JSON answer:"] + notes
    return lines


def plan_prompt(quick, idea, seconds, contexts, shown=0, parts=(), sung=()):
    """The user turn for planning; the first `shown` pictures are attached in the order listed."""
    found = pictures(contexts)
    kinds_of = kinds(contexts)
    undescribed = [entry["widgets"].get("name") for number, (entry, _, _) in enumerate(found, 1) if number <= shown
                   and entry["widgets"].get("name") and not (entry["widgets"].get("description") or "").strip()]
    request = story(idea, contexts)
    check_allowed(quick["direction"], request, *[entry["widgets"].get("description", "") for entry in contexts])
    if not request.strip() and not found:
        raise ValueError("Make H3 prompt needs a picture on a Context card or a few words in the Prompt.")
    played = soundtrack(contexts)
    words, played_words = card_words(contexts, False), card_words(contexts, True)   # the cards' own words (BUG-094)
    lines = _opening_lines(quick, seconds, kinds_of, undescribed, played, played_words, request, words, direction(quick["direction"]), sung)
    lines = _media_lines(lines, found, shown, kinds_of, contexts)
    names = [name for name, (sort, _) in kinds_of.items() if name and sort in ("figure", "place")]
    rules = _rules(quick, seconds, found, names, played, parts, sung, kinds_of)
    if undescribed:
        # A card without a description gave the compiler nothing but "the person shown in <Picture 1>"; the plan's answer
        # fills it (the Studio writes it into the empty card).
        rules.insert(0, "describe: for %s, one English sentence each of what the picture shows that must stay the same: who or "
                     "what it is, face, hair, body, clothing and colours (or the place and its light). Not the pose, not the "
                     "background of a figure." % ", ".join('"%s"' % n for n in undescribed))
    return "\n".join(lines + ["", "Rules:"] + ["- " + rule for rule in rules] +
                        ["", "Do not claim that any generation, rendering, or media analysis has occurred."])


def _uses(data, contexts):
    """Per answer shot, the view and place pictures it uses: named in "uses", or (qwen3.5:9b, 25 s test) written as if they
    were people ("A medium shot of @Cathedral standing ..."). Such a mention becomes the subject the view belongs to."""
    sorts = kinds(contexts)
    usable = {name.casefold(): name for name, (sort, _) in sorts.items() if name and sort in ("view", "place")}
    out = []
    for shot in data.get("shots") or ():
        if not isinstance(shot, dict):
            out.append(set())
            continue
        named = {usable[str(n).strip().lstrip("@").casefold()] for n in (shot.get("uses") if isinstance(shot.get("uses"), list) else ())
                 if str(n).strip().lstrip("@").casefold() in usable}
        for name in usable.values():
            pattern = r"@%s(?![\w])" % re.escape(name)
            if isinstance(shot.get("action"), str) and re.search(pattern, shot["action"], re.I):
                named.add(name)
                if sorts[name][0] == "view":
                    shot["action"] = re.sub(pattern, "@" + sorts[name][1], shot["action"], flags=re.I)
        out.append(named)
    return out


def _bind(result, contexts, uses):
    """Bind each used view or place picture to its shots, so a part without them does not get them as references."""
    data = json.loads(result["shots"])
    ids = {entry["widgets"].get("name"): card_id(entry) for entry in contexts}
    # A figure seen only as a first/last frame is no reference Subject; its lines are spoken by a voice with its name.
    frames = {ids.get(name): name for name, (sort, _) in kinds(contexts).items() if sort == "frame" and ids.get(name)}
    for shot, names in zip(data["shots"], uses):
        shot["media"] += [{"schema_version": 1, "context_id": ids[name], "role": "context"} for name in sorted(names) if ids.get(name)]
        for line in shot["spoken"]:
            if line["speaker_id"] in frames:
                line["voice"] = frames[line["speaker_id"]]
                line["speaker_id"] = ""
    result["shots"] = json.dumps(data, ensure_ascii=False)
    return result


def _speakers(data, contexts, quick):
    """A line whose speaker is no figure card goes to the only figure, when there is exactly one (qwen3.5:9b wrote the
    German name "Magierin" for @Mage); with no figure card at all, to the person of the first frame picture (else every
    line of a first-frame video was "a narrator")."""
    sorts = kinds(contexts)
    figures = [name for name, (sort, _) in sorts.items() if sort == "figure" and name]
    if not figures:
        figures = [entry["widgets"].get("name") for entry, usage, _ in pictures(contexts)
                   if "first_frame" in usage and entry["widgets"].get("name")][:1]
    for shot in data.get("shots") or ():
        for line in (shot.get("lines") if isinstance(shot, dict) and isinstance(shot.get("lines"), list) else ()):
            if not isinstance(line, dict) or quick["speech"] == "voiceover":
                continue
            speaker = str(line.get("speaker") or "").strip().lstrip("@")
            if speaker in figures or sorts.get(speaker, ("",))[0] == "frame":
                continue
            if len(figures) == 1 and speaker.casefold() not in ("narrator", "a narrator"):
                line["speaker"] = figures[0]


def _descriptions(data, contexts):
    """{context id: sentence} for the image cards without a description that the answer described."""
    given = data.get("describe") if isinstance(data.get("describe"), dict) else {}
    given = {str(k).strip().lstrip("@").casefold(): v for k, v in given.items()}
    out = {}
    for entry, _, _ in pictures(contexts):
        text = given.get((entry["widgets"].get("name") or "").casefold())
        if not (entry["widgets"].get("description") or "").strip() and isinstance(text, str) and text.strip():
            out[card_id(entry)] = " ".join(text.split())[:500]
    return out


def read_answer(text):
    """The JSON object of an answer, with the same tolerance as AI: split into shots (fences, a cut-off answer)."""
    if not isinstance(text, str):
        raise ValueError("Make H3 prompt needs a JSON text response.")
    return read_object(text, "plan", "The AI did not return valid JSON. Try again, or use a larger model.")


def parse_plan(text, seconds, contexts, quick, sung=()):
    data = read_answer(text)
    if data.get("refused"):
        raise ValueError("The AI refused: %s. Nothing was changed." % (
            "it saw someone who may be under 18, and Spicy 18+ is for adults only" if data["refused"] == "minor"
            else str(data["refused"])[:200]))
    if quick["speech"] == "none":
        for shot in data.get("shots") or ():
            if isinstance(shot, dict):
                shot["lines"] = []
    uses, descriptions = _uses(data, contexts), _descriptions(data, contexts)
    _speakers(data, contexts, quick)
    result = parse_split(data, seconds, contexts, "", sung)
    if descriptions:
        result["descriptions"] = descriptions
    check_allowed(quick["direction"], result["shots"], result.get("look", ""))
    return _bind(result, contexts, uses)
