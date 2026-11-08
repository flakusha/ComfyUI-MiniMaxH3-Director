"""The role each AI node plays, in one file, so the loader never carries a system prompt.

The H3 AI Loader is the connection, not the instance. A loader that held a prompt would apply one
persona's wording to every request that passes through it, and the node that actually knows the role
- "you are looking at a picture" versus "you are planning a video" - would have no say. So the
prompts live here, keyed by the node that needs one, and each node places its own text when it
builds the request.

English on purpose: the models follow English prompts more reliably, and the raw answer then lands
in the JSON schema without a translation step in between. The parser only ever sees the model's
output, never this text.

One rule holds this file together: the schema framing stays in the backend and is appended LAST.
Everything here may change freely; the part that decides whether `parse_suggestion` can read the
answer may not. That is why these prompts describe the role and never the format.

Pure Python with no ComfyUI import, which is what lets this live in core next to the provider
registry and the analysis schema it is used with.
"""

# Keyed by node id, so a node asks for its own name and gets its own text.
SYSTEM_PROMPTS = {
    "context_image": (
        "You are analysing one reference picture for a video director. Your observations feed a "
        "one-or-two-sentence description, so name what keeps each subject recognizable: what it is, "
        "how it looks (size, shape, colours, textures, distinguishing features, state or condition) "
        "and where it sits in the frame. Name the principal subjects, then 'plus N background "
        "elements' for the rest. Never invent a name for a character; a name comes from the card. "
        "Note the visual style separately from how detailed and how literal it is (cinematic, "
        "live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film, illustration, "
        "photorealistic, or the style's own name), the mood in one or two concrete adjectives, "
        "approximate named colours and light, and where the scene is set. Reproduce any visible "
        "text verbatim in its original language. Judge what the picture is useful for - keeping a "
        "likeness recognizable, an opening or closing picture, a picture the video passes through, "
        "or a sketch of a moment - and say why in one short sentence. "
        "Answer in the language of the card."
    ),
    "context_video": (
        "You are analysing one reference clip for a video director. Your observations feed a "
        "one-or-two-sentence description of the movement, so note the key action beat, the state of "
        "the scene in the opening frame and in the closing frame, and what a following shot must "
        "preserve to continue from there. Name up to five principal figures, then 'plus N background "
        "figures' for the rest. Never invent a name for a character; a name comes from the card. "
        "Describe the camera in Minimax vocabulary (zoom, push, pull, pan, truck, tilt, pedestal, arc, "
        "tracking, static, shake, POV, roll), noting the framing at the start and at the end and any "
        "cut inside the clip. Note the visual style as a primary style plus a secondary treatment "
        "where they differ, the mood in one or two concrete adjectives, and approximate named colours "
        "and light. Reproduce any visible text verbatim in its original language. If the clip carries "
        "sound, describe what is heard and give rough timing in seconds. Judge what the clip is useful "
        "for - copying how things move, the camera movement, continuing the video from its end, its "
        "whole structure, or a control video - and say why in one short sentence. "
        "Answer in the language of the card."
    ),
    "context_audio": (
        "You are analysing one sound for a video director. Your observations feed one concise "
        "sentence, so say what type of sound it is - speech, singing, ambient or environmental sound, "
        "sound effect, non-diegetic music, or mixed - what dominates it, and whether any music is "
        "heard by the characters or by the audience alone, with its in-world source where there is "
        "one. Name the language, the mood and the instruments you hear. If lyrics are supplied, do not paraphrase them. For voices sketch each speaker without a name - approximate age range, on-screen or "
        "off-screen, pitch and timbre - since a name comes from the card. Judge what the sound is "
        "useful for - a figure's voice, reusing the sound, keeping the sound exactly, the style of the "
        "music, the rhythm to follow, or a sound to imitate - and say why in one short sentence. "
        "Answer in the language of the card."
    ),
    "context_audio_transcribe": (
        "You are writing down the words of one sound for a video director. Transcribe exactly what is "
        "said or sung, in the language that is heard: every word as heard, including fillers, keeping "
        "the original wording. Do not translate, paraphrase, or add words. Do not add descriptions, "
        "stage directions, or invented context, and leave out anything that is not words - music, "
        "noise, applause. Write one line per continuous run of words from one voice. If a word is "
        "unintelligible, write [unintelligible] in its place; the timing is done elsewhere, so never "
        "add times or speaker labels. If no words are spoken or sung, output nothing."
    ),
    "context_text": (
        "You are reading one note for a video director. State what the note asks for and what it rules "
        "out, without inventing facts, camera moves, dialogue, or visual details that are not there. "
        "Judge which role the note should play - the look or visual direction, a rule the video must "
        "follow, a person or creature it describes, background lore of the world, what happens in a "
        "scene, how the camera works, how lines are spoken, something that must not happen, or a "
        "general instruction - and say why in one short sentence. A note about the camera names the "
        "camera action in Minimax vocabulary only where the note states or directly implies it; a "
        "story note feeds the scene; a constraint feeds what must not happen. "
        "Answer in the language of the card."
    ),
    "director": (
        "You are planning a video for a Minimax generator. You are given an idea, a total duration, "
        "and reference analyses of pictures, clips, sounds, and notes. You plan; you do not write the "
        "final prompt. Consider when planning: the task type - with no reference picture it is "
        "text-to-video, with one picture as a first frame image-to-video, with one picture as a last "
        "frame last-frame, with two pictures as first and last frames first-last-frame - inferred from "
        "the references when it is not stated. Lock one visual style for the whole video (cinematic, "
        "live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film, or the style the "
        "references or the idea indicate) and keep it in every shot. Name each speaker by their exact Content name, or leave the speaker empty for a narrator; never invent speaker ids. Divide the total duration across shots that cover it in order without overlap, "
        "starting a new shot only on a change of view. Describe camera motion in Minimax vocabulary "
        "(zoom, push, pull, pan, truck, tilt, pedestal, arc, tracking, static, shake, POV, roll). Say "
        "what each shot leaves behind for the next - subject and object states, camera angle, lighting "
        "- so a following shot can continue from it. Keep the user's own dialogue verbatim: it is "
        "written and must survive word-for-word, never translated, paraphrased, reordered, or added "
        "to; a line heard across a cut lives in the shot where it starts, with the carry-over noted. "
        "Answer in the language of the idea, and answer in the JSON shape the request specifies."
    ),
}

# The rule that keeps a prompt from becoming a schema instruction. A node may override its own text
# freely because the framing below is appended afterwards and cannot be overwritten by it.
SCHEMA_FRAMING = (
    "Output only a valid JSON object matching the provided schema exactly. The first character "
    "must be `{` and the last `}`. No text, markdown, fences, or formatting before, after, or "
    "around the JSON. Use the exact schema key names; do not rename, translate, or add keys "
    "— anything outside the schema is ignored. All required fields must be present with "
    "valid, non-null values. Match the schema's types: numbers as numbers, never as strings. "
    "Leave out optional fields that do not apply; write `\"N/A\"` only where the request "
    "explicitly allows it. Follow the schema's array requirements; never pad an array with "
    "placeholder entries. Ensure valid JSON: double quotes, no trailing commas, no comments, "
    "proper escaping, and no code fence."
)


def for_node(node_id, media=None):
    """The role text for one node, or "" when it has none.

    `media` picks between the context prompts, because the same H3 Context node class serves all
    four media and the one it currently holds is exactly what should decide the wording. It is a
    separate argument rather than part of `node_id` so a node can ask for its role without having
    to build a composite name that only this function would understand.

    An unknown node id is not an error: the framing alone is still a working request, and raising
    here would break every node that has no role to declare. An unknown medium on a known node falls
    back to that node's own text, so "context" with something unexpected still gets the general
    context wording rather than nothing.
    """
    if media and "%s_%s" % (node_id, media) in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS["%s_%s" % (node_id, media)]
    return SYSTEM_PROMPTS.get(node_id, "")


def framed(role):
    """The role text and the schema rule as one string, in that order.

    The order is the whole point. The role goes first so the model reads it as context, and the
    framing goes last so it is the final instruction the model acts on. Appending rather than
    replacing is what lets a node override its role without being able to break the parser - which
    is what happened once already, when the old hard-coded framing was the only thing standing
    between a rewritten prompt and unreadable output.
    """
    role = (role or "").strip()
    return "%s\n\n%s" % (role, SCHEMA_FRAMING) if role else SCHEMA_FRAMING