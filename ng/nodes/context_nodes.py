"""The four context nodes: a picture, a clip, a sound or a note, each saying what it MEANS for the video.

They only describe; the Director decides what is sent. Every node returns an NG_CONTEXT, a ContextItem the Director gathers.
"""
import json
import os

import folder_paths
from comfy_api.latest import io

from ..adapters import probe
from ..adapters.audio_decode import trimmed_audio
from ..core import trim_window
from ..core.context import CONTENT_BY_MEDIA, PRESERVE, USAGE_BY_MEDIA
from ..core.context_build import item_from_widgets
from ..core.context_options import ContextOptions
from ..core.analysis import ProviderConfig
from ..adapters.ai_provider import NativeAIProvider

NGContext = io.Custom("NG_CONTEXT")
CATEGORY = "MiniMax H3/NG"
NGAI = io.Custom("NG_AI_PROVIDER")
# CONTENT_BY_MEDIA is imported from core rather than repeated: the analysis parse has to check an
# answer against the same list, and two copies of it is how they start disagreeing.


def _files(kinds):
    folder = folder_paths.get_input_directory()
    os.makedirs(folder, exist_ok=True)
    files, _ = folder_paths.recursive_search(folder, excluded_dir_names=[".git"])
    return sorted(folder_paths.filter_files_content_types(files, kinds))


def _common_inputs(media):
    return [
        io.String.Input("name", default="", tooltip="What the Director and your shot text call this. Write @name in a shot to point at it."),
        io.String.Input("description", multiline=True, default="", tooltip="One sentence about what this is (identity, look). It is written into the prompt."),
        io.Combo.Input("content", options=list(CONTENT_BY_MEDIA[media]), tooltip="What is in it."),
        io.Combo.Input("usage", options=list(USAGE_BY_MEDIA[media]), tooltip="What the model should do with it."),
        io.Combo.Input("preserve", options=list(PRESERVE), default="high", tooltip="How closely the result should stay true to it."),
    ]


def _identity_inputs():
    return [io.String.Input("context_id", default="", optional=True, advanced=True,
                            tooltip="Persistent identity, assigned when a new card is created."),
            io.String.Input("color", default="", optional=True, advanced=True, tooltip="Optional context colour, #rrggbb."),
            io.String.Input("content_roles", default="", optional=True, advanced=True),
            io.String.Input("usage_roles", default="", optional=True, advanced=True),
            io.String.Input("context_options", default="", optional=True, advanced=True),
            NGAI.Input("ai_provider", display_name="AI provider", optional=True)]


def _widget_extras(extras):
    config = extras.pop("ai_provider", None)
    if config is not None and not isinstance(config, (ProviderConfig, NativeAIProvider)):
        raise ValueError("Connect H3 AI Loader to AI provider.")
    return extras


def _external_input(media):
    """The card's optional external socket (`audio_in` and friends). The id needs its suffix because schema ids must
    be unique next to the widget it drives; the label shows the plain medium. A socket adds no widgets_values entry,
    so saved workflows keep restoring by position unchanged."""
    if media == "text":
        return io.String.Input("text_in", display_name="text", optional=True, force_input=True, multiline=True,
                               tooltip="Optional: text from another node (e.g. Primitive String). While linked the "
                                       "delivered text is this card's note; the note editor locks and the card's own "
                                       "text is kept unused.")
    hint = {"image": "Upload, the file picker and the picture editor lock while linked; the linked file fills the card.",
            "video": "Upload and the file picker lock while linked; the linked file fills the card.",
            "audio": "Upload and the file picker lock while linked; the linked file fills the card and its exact "
                     "length comes from the audio itself."}[media]
    kind = {"image": io.Image, "video": io.Video, "audio": io.Audio}[media]
    return kind.Input(media + "_in", display_name=media, optional=True,
                      tooltip="Optional: a %s from another node (Load%s and the like). %s" % (media, media.capitalize(), hint))


def _no_file_error(media):
    return ("The connected %s has no file behind it - connect Load%s, or pick a file on the card."
            % (media, media.capitalize()))


def _linked_file(widget_value, media):
    """A connected external input reaches the render only through the card's media widget: the frontend mirrors the
    origin's file into it, so headless runs and non-file producers (decoded or recorded audio, composite pictures)
    are named here instead of silently keeping a stale file."""
    if not str(widget_value or "").strip():
        raise ValueError(_no_file_error(media))


def _validate_linked_file(linked, widget_value, media):
    """Queue pre-flight: the same rule as execute, returned as the validation message before a run starts."""
    return True if linked is None or str(widget_value or "").strip() else _no_file_error(media)


def _linked_length(audio_in):
    """Exact seconds of a connected AUDIO payload; the batch dim is ignored and no file is decoded."""
    from ..adapters.linked_audio import read_waveform
    wave, rate = read_waveform(audio_in)
    return float(wave.shape[-1]) / rate


def _linked_float(value):
    """A linked trim arrives as None when the upstream is not a constant; that means unset, not 0."""
    return float(value or 0.0)


def _subject_inputs():
    return [io.String.Input("subject_ids", default="", optional=True, advanced=True,
                            tooltip="JSON list of owning Subject context IDs; edited by the card's Subject assignment list.")]


class NGContextImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGContextImage", display_name="H3 Context Image", category=CATEGORY,
                         description="A picture and what it means: a character, an outfit, a place, a first frame.",
                         inputs=[_external_input("image"), io.Combo.Input("image", options=_files(["image"]), upload=io.UploadType.image)] + _common_inputs("image") + _identity_inputs() + _subject_inputs(),
                         outputs=[NGContext.Output(display_name="context")])

    @classmethod
    def fingerprint_inputs(cls, image, **_):
        return probe.file_stamp(image)

    @classmethod
    def validate_inputs(cls, image, image_in=None, **_):
        return _validate_linked_file(image_in, image, "image")

    @classmethod
    def execute(cls, image, name, description, content, usage, preserve, context_id="", color="", image_in=None, **extras):
        if image_in is not None:
            _linked_file(image, "image")
        return io.NodeOutput(item_from_widgets("image", dict(image=image, name=name, description=description, content=content,
                                                            usage=usage, preserve=preserve, context_id=context_id, color=color, **_widget_extras(extras))))


class NGContextVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGContextVideo", display_name="H3 Context Video", category=CATEGORY,
                         description="A video reference, a Canny/Pose/Depth control video, or an Inpaint source with a mask on the Control lane.",
                         inputs=[_external_input("video"), io.Combo.Input("video", options=_files(["video"]), upload=io.UploadType.video)] + _common_inputs("video") + _identity_inputs() + _subject_inputs(),
                         outputs=[NGContext.Output(display_name="context")])

    @classmethod
    def fingerprint_inputs(cls, video, context_options="", **_):
        options = ContextOptions.from_json(context_options)
        mask = probe.file_stamp(options.control_mask) if options.control_kind == "mask" and options.control_mask else ""
        result = probe.file_stamp(options.control_result) if options.control_result else ""
        original = probe.file_stamp(options.control_original) if options.control_original else ""
        return "%s:%s:%s:%s:%s:%s" % (probe.file_stamp(video), mask, result, original,
                                        options.mask_shape, options.mask_grow)

    @classmethod
    def validate_inputs(cls, video, video_in=None, **_):
        return _validate_linked_file(video_in, video, "video")

    @classmethod
    def execute(cls, video, name, description, content, usage, preserve, context_id="", color="", video_in=None, **extras):
        if video_in is not None:
            _linked_file(video, "video")
        widgets = dict(video=video, name=name, description=description, content=content, usage=usage, preserve=preserve,
                       context_id=context_id, color=color, **_widget_extras(extras))
        return io.NodeOutput(item_from_widgets("video", widgets, probe.duration_seconds(video.replace(" [input]", ""))))


class NGContextAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGContextAudio", display_name="H3 Context Audio", category=CATEGORY,
                         description="A sound: a character's voice, a song to keep exactly (lock) or copy, a music style.",
                         inputs=[_external_input("audio"), io.Combo.Input("audio", options=_files(["audio", "video"]), upload=io.UploadType.audio)] + _common_inputs("audio") + [
                             io.String.Input("speaker", default="", tooltip="Voice only: the name of the character who speaks with this voice."),
                             io.Float.Input("trim_start", default=0.0, min=0.0, step=0.01, optional=True,
                                            tooltip="First second of the file this card stands for. A link here drives the card's range; "
                                                    "without one, drag the range on the card."),
                             io.Float.Input("trim_end", default=0.0, min=0.0, step=0.01, optional=True,
                                            tooltip="Last second of the file this card stands for; 0 means the file end. A link here drives "
                                                    "the card's range; without one, drag the range on the card.")] + _identity_inputs(),
                         outputs=[NGContext.Output(display_name="context"),
                                  io.Audio.Output(display_name="trimmed audio",
                                                  tooltip="The selected range as sound, in the file's own rate and channels; wire it into "
                                                          "CreateVideo's audio when no generated soundtrack is used."),
                                  io.Float.Output(display_name="max duration",
                                                  tooltip="Seconds the selected range can play: the number to keep project_seconds under.")])

    @classmethod
    def fingerprint_inputs(cls, audio, trim_start=0.0, trim_end=0.0, context_options="", **_):
        # a linked audio_in arrives as None here (non-constant links resolve only for constants), so the cache key
        # follows the mirrored widget file - which the card rewrites whenever the origin changes its file.
        # A linked trim value is authoritative and the stored options range is the fallback, exactly as in
        # execute: a preset-loaded range alone must bust the key, or a stale cached item is reused.
        options = ContextOptions.from_json(context_options)
        start_s = _linked_float(trim_start) if (trim_start or 0.0) > 0 else options.start_s
        end_s = _linked_float(trim_end) if (trim_end or 0.0) > 0 else options.end_s
        return trim_window.fingerprint(probe.file_stamp(audio), start_s, end_s)

    @classmethod
    def validate_inputs(cls, audio, trim_start=0.0, trim_end=0.0, audio_in=None, **_):
        start_s, end_s = _linked_float(trim_start), _linked_float(trim_end)
        if start_s > 0 and end_s > 0 and end_s <= start_s:
            return "Trim end (%s) must be past Trim start (%s)." % (end_s, start_s)
        return _validate_linked_file(audio_in, audio, "audio")

    @classmethod
    def execute(cls, audio, name, description, content, usage, preserve, speaker, trim_start=0.0, trim_end=0.0,
                context_id="", color="", audio_in=None, **extras):
        widgets = dict(audio=audio, name=name, description=description, content=content, usage=usage, preserve=preserve, speaker=speaker,
                       context_id=context_id, color=color, **_widget_extras(extras))
        options = ContextOptions.from_json(widgets.get("context_options", "")).to_dict()
        # a connected trim value is authoritative; the card's own widgets stay the fallback (0 = the stored range)
        start_s = _linked_float(trim_start) if (trim_start or 0.0) > 0 else options["start_s"]
        end_s = _linked_float(trim_end) if (trim_end or 0.0) > 0 else options["end_s"]
        # a linked audio gives the exact length from its payload; the file header probe stays the unlinked path.
        # The trims stay independent of the link: they cut the mirrored file whatever card it came from.
        if audio_in is not None:
            _linked_file(audio, "audio")
            length_s = _linked_length(audio_in)
        else:
            length_s = probe.duration_seconds(audio.replace(" [input]", ""))
        cut_start, cut_end = trim_window.check_window(start_s, end_s, length_s)
        options["start_s"], options["end_s"] = start_s, end_s      # 0 stays 0: the file end, not today's length
        widgets["context_options"] = json.dumps(options)
        return io.NodeOutput(item_from_widgets("audio", widgets, length_s),
                             trimmed_audio(audio, cut_start, cut_end), trim_window.max_duration(start_s, end_s, length_s))


class NGContextText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="NGContextText", display_name="H3 Context Text (Style & Rules)", category=CATEGORY,
                         description="A note that applies to the whole video: style, camera rules, world.",
                         inputs=[_external_input("text"), io.String.Input("text", multiline=True, default="")] + _common_inputs("text") + _identity_inputs() + _subject_inputs(),
                         outputs=[NGContext.Output(display_name="context")])

    @classmethod
    def execute(cls, text, name, description, content, usage, preserve, context_id="", color="", text_in=None, **extras):
        # the delivered string is the note; the card's own text stays the fallback (an empty link says nothing)
        if text_in is not None and str(text_in).strip():
            text = str(text_in)
        return io.NodeOutput(item_from_widgets("text", dict(text=text, name=name, description=description, content=content,
                                                           usage=usage, preserve=preserve, context_id=context_id, color=color, **_widget_extras(extras))))


NODES = [NGContextImage, NGContextVideo, NGContextAudio, NGContextText]
