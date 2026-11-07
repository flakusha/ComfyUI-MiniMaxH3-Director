"""External typed inputs (audio_in/video_in/image_in/text_in): the connection owns the card's file.

The frontend mirrors the origin's file into the card's media widget; these tests pin the backend half: the
ports exist as plain sockets (no widgets_values shift), a linked input with no file behind it is refused by
validate_inputs and execute alike, the audio length comes from the payload, and a linked text is the note."""
import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "tests" / "ng"))
from _pack import comfy_available, load_full_pack


PORTS = (("NGContextImage", "image_in", "image"), ("NGContextVideo", "video_in", "video"),
         ("NGContextAudio", "audio_in", "audio"), ("NGContextText", "text_in", "text"))
ROLES = {"image": ("character", "reference"), "video": ("motion", "motion"), "audio": ("voice", "voice"),
         "text": ("style", "general")}


class ExternalContextInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        saved_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0], "--cpu"]
            if not comfy_available():
                raise unittest.SkipTest("ComfyUI is not importable")
            cls.pack = load_full_pack()
            cls.io = importlib.import_module("comfy_api.latest._io")
            cls.trim_window = importlib.import_module(cls.pack.__name__ + ".ng.core.trim_window")
            cls.module = importlib.import_module(cls.pack.__name__ + ".ng.nodes.context_nodes")
        finally:
            sys.argv = saved_argv

    def node(self, kind):
        return self.pack.NODE_CLASS_MAPPINGS[kind]

    def input(self, kind, input_id):
        return next(i for i in self.node(kind).GET_SCHEMA().inputs if i.id == input_id)

    def widget_ids(self, kind):
        """The ids the frontend turns into widgets (and widgets_values positions), in object_info order."""
        groups = self.io.create_input_dict_v1(self.node(kind).GET_SCHEMA().inputs)
        widget_types = {"COMBO", "STRING", "INT", "FLOAT", "BOOLEAN"}
        return [name for group in ("required", "optional") for name, (io_type, config) in groups.get(group, {}).items()
                if io_type in widget_types and not config.get("forceInput")]

    def input_groups(self, kind):
        return self.io.create_input_dict_v1(self.node(kind).GET_SCHEMA().inputs)

    def execute(self, kind, **overrides):
        media = next(row[2] for row in PORTS if row[0] == kind)
        content, usage = ROLES[media]
        raw = {media: "hero.png" if media == "image" else "voice.wav" if media == "audio" else "clip.mp4",
               "name": "", "description": "", "content": content, "usage": usage, "preserve": "high",
               "context_id": "ctx_" + media, "color": ""}
        if media == "audio":
            raw["speaker"] = ""
            raw["trim_start"] = raw["trim_end"] = 0.0
        if media == "text":
            raw["text"] = "card note"
        raw.update(overrides)
        return self.node(kind).execute(**raw)

    def test_ports_are_plain_optional_sockets_first_in_the_list(self):
        io_types = {"image": "IMAGE", "video": "VIDEO", "audio": "AUDIO"}
        for kind, port, media in PORTS:
            found = self.input(kind, port)
            self.assertTrue(found.optional, port)
            self.assertEqual(found.display_name, media, port)
            self.assertEqual(self.node(kind).GET_SCHEMA().inputs[0].id, port, "the socket sits on top of the node")
            groups = self.input_groups(kind)
            self.assertNotIn(port, groups.get("required", {}), port + " must not become a widget")
            spec = groups["optional"][port]
            if media == "text":
                self.assertEqual(spec[0], "STRING")
                self.assertTrue(spec[1].get("forceInput"), "String would otherwise shift every saved workflow")
            else:
                self.assertEqual(spec[0], io_types[media], port + " is a plain typed socket")
                self.assertNotIn("forceInput", spec[1])

    def test_the_widgets_keep_their_saved_positions(self):
        common = ["name", "description", "content", "usage", "preserve"]
        advanced = ["context_id", "color", "content_roles", "usage_roles", "context_options"]
        expected = {"NGContextImage": ["image"] + common + advanced + ["subject_ids"],
                    "NGContextVideo": ["video"] + common + advanced + ["subject_ids"],
                    "NGContextAudio": ["audio"] + common + ["speaker", "trim_start", "trim_end"] + advanced,
                    "NGContextText": ["text"] + common + advanced + ["subject_ids"]}
        for kind, ids in expected.items():
            self.assertEqual(self.widget_ids(kind), ids, "saved workflows restore widget values by position")

    def test_linked_audio_takes_its_exact_length_from_the_payload_and_keeps_the_trims(self):
        import torch
        audio_in = {"waveform": torch.zeros(1, 1, 44100 * 3), "sample_rate": 44100}
        captured = {}

        def fake_trimmed(source, start, end):
            captured["source"], captured["window"] = source, (start, end)
            return {"waveform": None, "sample_rate": 0}

        with patch.object(self.module.probe, "duration_seconds", return_value=99.0) as probe_seconds, \
                patch.object(self.module, "trimmed_audio", side_effect=fake_trimmed):
            item, trimmed, max_duration = self.execute("NGContextAudio", audio="voice.wav [input]",
                                                       trim_start=1.0, trim_end=2.5, audio_in=audio_in).args
            self.assertAlmostEqual(item.length_s, 3.0, "the payload's own length, not the file header's")
            probe_seconds.assert_not_called()
            self.assertEqual(captured["source"], "voice.wav [input]", "the cut runs on the mirrored file")
            self.assertEqual(captured["window"], (1.0, 2.5), "the trims stay independent of the link")
            self.assertAlmostEqual(max_duration, 1.5)
            self.assertAlmostEqual(item.options.start_s, 1.0)
            self.assertAlmostEqual(item.options.end_s, 2.5)

    def test_unlinked_audio_still_probes_the_file(self):
        with patch.object(self.module.probe, "duration_seconds", return_value=5.0), \
                patch.object(self.module, "trimmed_audio", return_value={"waveform": None, "sample_rate": 0}):
            item, trimmed, max_duration = self.execute("NGContextAudio", audio_in=None).args
            self.assertAlmostEqual(item.length_s, 5.0)
            self.assertAlmostEqual(max_duration, 5.0)

    def test_a_link_without_a_file_is_refused_before_and_during_the_run(self):
        import torch
        payload = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}
        node = self.node("NGContextAudio")
        with self.assertRaisesRegex(ValueError, "no readable sound"):
            node.execute(audio="", name="", description="", content="voice", usage="voice", preserve="high",
                         speaker="", audio_in={})
        self.assertIn("no file behind it", node.validate_inputs(audio="", audio_in=payload))
        self.assertTrue(node.validate_inputs(audio="voice.wav", audio_in=payload), "a mirrored file passes")
        self.assertTrue(node.validate_inputs(audio="voice.wav", trim_start=1.0, trim_end=2.0, audio_in=payload))
        self.assertTrue(node.validate_inputs(audio="", audio_in=None), "unlinked stays the card's own business")
        for kind, port, media in (("NGContextImage", "image_in", "image"), ("NGContextVideo", "video_in", "video")):
            other = self.node(kind)
            self.assertIn("no file behind it", other.validate_inputs(**{media: "", port: object()}))
            self.assertTrue(other.validate_inputs(**{media: "a.png", port: object()}))
            with self.assertRaises(ValueError):
                other.execute(**{media: "", port: object()}, **dict(name="", description="", preserve="high",
                              content=ROLES[media][0], usage=ROLES[media][1]))

    def test_a_non_file_backed_payload_is_materialized_to_h3_external(self):
        # Decoded/recorded audio has no file to mirror: execute writes it to input/h3-external/ so the
        # linked card runs headlessly, and the item, the trim cut and the duration all read the materialized file.
        import torch
        node = self.node("NGContextAudio")
        payload = {"waveform": torch.zeros(1, 2, 44100 * 2), "sample_rate": 44100}
        captured = {}

        def fake_trimmed(source, start, end):
            captured["source"], captured["window"] = source, (start, end)
            return {"waveform": None, "sample_rate": 0}

        with patch.object(self.module, "trimmed_audio", side_effect=fake_trimmed):
            item, _, max_duration = node.execute(audio="", name="", description="", content="voice", usage="voice",
                                                 preserve="high", speaker="", audio_in=payload).args
        source = captured["source"]
        self.assertTrue(source.startswith("h3-external/") and source.endswith(".flac"), source)
        self.assertEqual(item.source, source.replace(" [input]", ""))
        self.assertAlmostEqual(item.length_s, 2.0)
        self.assertAlmostEqual(max_duration, 2.0)
        self.assertEqual(captured["window"], (0.0, 2.0))
        probe = importlib.import_module(self.pack.__name__ + ".ng.adapters.probe")
        self.assertAlmostEqual(probe.duration_seconds(item.source), 2.0, places=1)
        again = node.execute(audio="", name="", description="", content="voice", usage="voice",
                             preserve="high", speaker="", audio_in=payload).args[0]
        self.assertEqual(again.source, item.source, "the same sound reuses its file")
        bad_channels = {"waveform": torch.zeros(1, 6, 100), "sample_rate": 44100}
        with self.assertRaisesRegex(ValueError, "only mono and stereo"):
            node.execute(audio="", name="", description="", content="voice", usage="voice",
                         preserve="high", speaker="", audio_in=bad_channels)

    def test_a_malformed_payload_names_the_card(self):
        # A linked AUDIO payload the mirror check already passed still has to read: {} and a
        # payload without a waveform name the card instead of raising a raw KeyError.
        import torch
        node = self.node("NGContextAudio")
        for bad in ({}, {"sample_rate": 44100}, {"waveform": None, "sample_rate": 44100}):
            with self.subTest(payload=bad):
                with self.assertRaisesRegex(ValueError, "no readable sound"):
                    node.execute(audio="voice.wav", name="", description="", content="voice", usage="voice",
                                 preserve="high", speaker="", audio_in=bad)

    def test_a_linked_text_is_the_note_and_an_empty_link_falls_back(self):
        item = self.execute("NGContextText", text_in=" STYLE: film noir. ").args[0]
        self.assertEqual(item.source, " STYLE: film noir. ", "the delivered string verbatim")
        self.assertEqual(item.name, "STYLE: film noir.", "the note names the card while Name is empty")
        self.assertEqual(self.execute("NGContextText", text_in=None).args[0].source, "card note")
        self.assertEqual(self.execute("NGContextText", text_in="   ").args[0].source, "card note",
                         "an empty link says nothing")

    def test_fingerprint_still_follows_the_mirrored_widget_file(self):
        node = self.node("NGContextAudio")
        with patch.object(self.module.probe, "file_stamp", return_value="1:2") as stamp:
            self.assertEqual(node.fingerprint_inputs(audio="song.wav", trim_start=0.0, trim_end=2.0, audio_in=None),
                             self.trim_window.fingerprint("1:2", 0.0, 2.0))
            stamp.assert_called_once_with("song.wav")

    def test_a_preset_loaded_range_busts_the_audio_key_without_a_trim_change(self):
        # The preset writes context_options directly; the trim widgets stay 0, so the key must
        # read the stored range or a stale cached item is reused.
        import json
        node = self.node("NGContextAudio")
        options = json.dumps({"schema_version": 1, "start_s": 1.0, "end_s": 2.5})
        with patch.object(self.module.probe, "file_stamp", return_value="1:2"):
            keyed = node.fingerprint_inputs(audio="song.wav", context_options=options)
            plain = node.fingerprint_inputs(audio="song.wav", context_options="")
            self.assertNotEqual(keyed, plain)
            self.assertEqual(keyed, self.trim_window.fingerprint("1:2", 1.0, 2.5))

    def test_a_control_option_change_busts_the_video_key(self):
        # Mask-shape-only or control-file-only edits must re-run the card, not reuse the cached item.
        import json
        node = self.node("NGContextVideo")
        base = json.dumps({"schema_version": 1, "control_kind": "mask", "control_mask": "m.png"})
        shaped = json.dumps({"schema_version": 1, "control_kind": "mask", "control_mask": "m.png",
                             "mask_shape": "outline"})
        resulted = json.dumps({"schema_version": 1, "control_kind": "mask", "control_mask": "m.png",
                               "control_result": "r.mp4"})
        with patch.object(self.module.probe, "file_stamp", side_effect=lambda name: "stamp:" + name):
            self.assertNotEqual(node.fingerprint_inputs(video="clip.mp4", context_options=shaped),
                                node.fingerprint_inputs(video="clip.mp4", context_options=base))
            self.assertNotEqual(node.fingerprint_inputs(video="clip.mp4", context_options=resulted),
                                node.fingerprint_inputs(video="clip.mp4", context_options=base))

    def test_real_validation_resolves_the_link_for_validate_inputs(self):
        """The full V3 path proves OUR override actually runs there: a resolved payload with no widget file fails,
        the same machinery passes once the widget names a file, and an unconnected payload stays valid."""
        from types import SimpleNamespace
        execution = importlib.import_module("execution")
        node = self.node("NGContextAudio")
        base = {"audio": "voice.wav", "name": "", "description": "", "content": "voice", "usage": "voice",
                "preserve": "high", "speaker": "", "trim_start": 0.0, "trim_end": 0.0, "context_id": "ctx",
                "color": "", "audio_in": ["source", 0]}
        import torch
        payload = {"waveform": torch.zeros(1, 1, 44100), "sample_rate": 44100}
        for audio_value, resolved, expect in (("voice.wav", None, True), ("", payload, "no file behind it"), ("", None, True)):
            cache = SimpleNamespace(get_cache=lambda source, target: SimpleNamespace(outputs=[[resolved]]))
            values, missing, v3 = execution.get_input_data({**base, "audio": audio_value}, node, "card", execution_list=cache)
            self.assertFalse(missing)
            nested = self.io.build_nested_inputs({key: entries[0] for key, entries in values.items()}, v3)
            self.assertIn("audio_in", nested, "the port reaches the validator like any input")
            results = asyncio.run(execution._async_map_node_over_list("audit", "card", node, values,
                                                                     "validate_inputs", v3_data=v3))
            if expect is True:
                self.assertEqual(results, [True])
            else:
                self.assertIn(expect, str(results[0]), "the override really runs inside the queue's validation")


if __name__ == "__main__":
    unittest.main()
