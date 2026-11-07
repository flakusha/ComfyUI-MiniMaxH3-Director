"""The audio card's trim window: the seconds math, the errors it names, the decoded slice and the cache key.

window_seconds/check_window/max_duration (ng/core/trim_window) own the effective window; decode_audio/
trimmed_audio (ng/adapters/audio_decode) own the PyAV slice the card's new outputs (and the Whisper timing)
share; fingerprint keeps a linked trim change from being served from cache. NGContextAudio.execute folds a
connected trim value over the card's stored one, so a link drives the range."""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pack import comfy_available, module, stub_prompt_server

comfy_available()          # the adapters import folder_paths/av: put ComfyUI's root on sys.path first

trim_window = module("ng.core.trim_window")
audio_decode = module("ng.adapters.audio_decode")

RATE = 8000


def make_wav(directory, name="trim_slice.wav", seconds=2.0):
    import av
    path = os.path.join(directory, name)
    samples = (np.arange(int(RATE * seconds)) % 1000).astype(np.int16)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("pcm_s16le", rate=RATE)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_mp3(directory, name="trim_slice.mp3", seconds=6.0):
    """A lossy, pts-carrying file: its seek lands on a frame boundary before the start, the path the wav
    tests cannot reach."""
    import av
    path = os.path.join(directory, name)
    samples = np.zeros(int(44100 * seconds), dtype=np.int16)
    samples[44100:44100 * 2] = 5000                     # audible only during the second second
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libmp3lame", rate=44100)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 44100
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


class WindowMath(unittest.TestCase):
    def test_end_zero_means_the_file_end(self):
        self.assertEqual(trim_window.window_seconds(2.0, 0.0, 30.0), (2.0, 30.0))
        self.assertEqual(trim_window.window_seconds(2.0, 30.0, 30.0), (2.0, 30.0))

    def test_the_card_end_cuts_before_the_file(self):
        self.assertEqual(trim_window.window_seconds(2.0, 12.0, 30.0), (2.0, 12.0))

    def test_an_unknown_length_and_end_have_no_window(self):
        self.assertEqual(trim_window.window_seconds(0.0, 0.0, 0.0), (0.0, None))

    def test_max_duration_counts_the_window(self):
        self.assertEqual(trim_window.max_duration(2.0, 12.0, 30.0), 10.0)
        self.assertEqual(trim_window.max_duration(0.0, 0.0, 0.0), 0.0)

    def test_an_inverted_window_is_named(self):
        with self.assertRaisesRegex(ValueError, "later than source start"):
            trim_window.check_window(12.0, 4.0, 30.0)

    def test_a_start_at_or_past_the_file_end_is_named(self):
        with self.assertRaisesRegex(ValueError, "past the file's end"):
            trim_window.check_window(30.0, 0.0, 30.0)

    def test_an_end_past_the_file_is_clamped_to_it(self):
        self.assertEqual(trim_window.check_window(2.0, 99.0, 30.0), (2.0, 30.0))


class Fingerprint(unittest.TestCase):
    def test_a_different_trim_is_a_different_cache_key(self):
        base = trim_window.fingerprint("10:200", 0.0, 0.0)
        self.assertNotEqual(base, trim_window.fingerprint("10:200", 2.0, 0.0))
        self.assertNotEqual(base, trim_window.fingerprint("10:200", 2.0, 5.0))
        self.assertEqual(base, trim_window.fingerprint("10:200", 0.0, 0.0))

    def test_a_replaced_file_is_a_different_cache_key_even_with_the_same_trim(self):
        self.assertNotEqual(trim_window.fingerprint("10:200", 2.0, 5.0), trim_window.fingerprint("12:999", 2.0, 5.0))


class DecodedSlice(unittest.TestCase):
    def decode(self, path, start_s=0.0, end_s=0.0, **kw):
        with patch.object(audio_decode, "input_path", return_value=path):
            return audio_decode.decode_audio(path, start_s, end_s, **kw)

    def test_the_whole_file_at_its_own_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            wave, rate = self.decode(make_wav(directory))
            self.assertEqual((tuple(wave.shape), rate), ((1, 2 * RATE), RATE))

    def test_a_slice_holds_the_seconds_asked_for(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_wav(directory)
            wave, _ = self.decode(path, 0.5, 1.5)
            full, _ = self.decode(path)
            self.assertEqual(tuple(wave.shape), (1, RATE))
            self.assertEqual(wave[0, 0], full[0, RATE // 2])       # the ramp continues where 0.5 s sits
            self.assertEqual(wave[0, -1], full[0, RATE // 2 + RATE - 1])

    def test_an_end_past_the_file_ends_at_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            wave, _ = self.decode(make_wav(directory), 1.0, 99.0)
            self.assertEqual(tuple(wave.shape), (1, RATE))

    def test_a_start_past_the_file_holds_no_audio_and_is_named(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "holds no audio"):
                self.decode(make_wav(directory), 5.0, 0.0)

    def test_a_slice_ending_before_it_starts_holds_no_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "holds no audio"):
                self.decode(make_wav(directory), 1.5, 1.0)

    def test_a_sub_frame_window_inside_the_file_holds_no_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "holds no audio"):
                self.decode(make_wav(directory), 1.0, 1.00001)

    def test_a_file_without_audio_is_named(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "silent.png")
            with open(path, "wb") as f:
                f.write(b"not sound")
            with self.assertRaises(ValueError):
                self.decode(path)

    def test_trimmed_audio_wears_loadaudios_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(audio_decode, "input_path", return_value=make_wav(directory)):
                audio = audio_decode.trimmed_audio(make_wav(directory), 0.0, 1.0)
            self.assertEqual(tuple(audio["waveform"].shape), (1, 1, RATE))
            self.assertEqual(audio["sample_rate"], RATE)

    def test_a_keyframe_seked_slice_starts_where_asked(self):
        """On a lossy file the seek lands before the start on a frame boundary: the frames before the
        asked second are cut, and a slice inside the silent head holds (almost) no energy."""
        with tempfile.TemporaryDirectory() as directory:
            path = make_mp3(directory)
            with patch.object(audio_decode, "input_path", return_value=path):
                head, rate = audio_decode.decode_audio(path, 0.0, 0.9)
                loud, _ = audio_decode.decode_audio(path, 1.2, 1.9)
            self.assertEqual(rate, 44100)
            self.assertLess(float(np.abs(head.numpy()).max()), 0.01)      # the silent first second
            self.assertGreater(float(np.abs(loud.numpy()).max()), 0.1)    # the tone of the second second

    def test_the_whisper_path_resamples_to_16_khz_mono(self):
        timing = module("ng.adapters.lyrics_timing")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(audio_decode, "input_path", return_value=make_wav(directory)):
                samples = timing._samples(make_wav(directory))
            self.assertEqual((samples.dtype, len(samples)), (np.dtype("float32"), 16000 * 2))


class TheAudioCardFoldsTrims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import folder_paths
        comfy_available()
        stub_prompt_server()          # ng.nodes registers /ng/plan routes on import
        cls.nodes = module("ng.nodes.context_nodes")
        cls.directory = tempfile.mkdtemp()
        cls.source = make_wav(cls.directory, "fold_card.wav", seconds=12.0)
        cls.name = "fold_card.wav"
        cls.input_folder = patch.object(folder_paths, "get_input_directory", return_value=cls.directory)
        cls.input_folder.start()

    @classmethod
    def tearDownClass(cls):
        cls.input_folder.stop()

    def run_card(self, trim_start=0.0, trim_end=0.0, options=""):
        output = self.nodes.NGContextAudio.execute(
            audio=self.name, name="Song", description="", content="voice", usage="reference", preserve="high",
            speaker="", trim_start=trim_start, trim_end=trim_end, context_id="audio_song",
            context_options=options)
        return output.args

    def test_an_untrimmed_card_decodes_the_whole_file_and_reports_its_length(self):
        item, audio, longest = self.run_card()
        self.assertEqual((item.options.start_s, item.options.end_s), (0.0, 0.0))
        self.assertEqual(audio["waveform"].shape[2], 12 * RATE)
        self.assertAlmostEqual(longest, 12.0, places=2)

    def test_widget_trims_cut_the_output_and_fold_into_the_options(self):
        item, audio, longest = self.run_card(trim_start=2.0, trim_end=5.0)
        self.assertEqual((item.options.start_s, item.options.end_s), (2.0, 5.0))
        self.assertAlmostEqual(audio["waveform"].shape[2] / audio["sample_rate"], 3.0, places=2)
        self.assertAlmostEqual(longest, 3.0, places=2)

    def test_a_stored_range_still_applies_without_trim_widgets(self):
        stored = json.dumps(dict(schema_version=1, start_s=2.0, end_s=6.0))
        item, audio, longest = self.run_card(options=stored)
        self.assertEqual((item.options.start_s, item.options.end_s), (2.0, 6.0))
        self.assertAlmostEqual(longest, 4.0, places=2)

    def test_an_inverted_window_is_named(self):
        with self.assertRaisesRegex(ValueError, "later than source start"):
            self.run_card(trim_start=5.0, trim_end=2.0)

    def test_a_trim_start_at_the_file_end_is_named(self):
        with self.assertRaisesRegex(ValueError, "past the file's end"):
            self.run_card(trim_start=12.0)

    def test_fingerprint_changes_with_the_trim_floats(self):
        plain = self.nodes.NGContextAudio.fingerprint_inputs(audio=self.name)
        self.assertNotEqual(plain, self.nodes.NGContextAudio.fingerprint_inputs(audio=self.name, trim_start=2.0))
        self.assertNotEqual(plain, self.nodes.NGContextAudio.fingerprint_inputs(audio=self.name, trim_end=4.0))

    def test_a_linked_trim_arrives_as_none_and_means_unset(self):
        plain = self.nodes.NGContextAudio.fingerprint_inputs(audio=self.name)
        self.assertEqual(plain, self.nodes.NGContextAudio.fingerprint_inputs(audio=self.name, trim_start=None, trim_end=None))
        item, _, _ = self.run_card(trim_start=None, trim_end=None)
        self.assertEqual((item.options.start_s, item.options.end_s), (0.0, 0.0))

    def test_a_linked_audio_without_a_sample_rate_names_the_card(self):
        import torch
        with self.assertRaisesRegex(ValueError, "no file behind it"):
            self.nodes.NGContextAudio.execute(
                audio=self.name, name="Song", description="", content="voice", usage="reference", preserve="high",
                speaker="", trim_start=0.0, trim_end=0.0, context_id="audio_song",
                audio_in={"waveform": torch.zeros(1, 1, 8)})

    def test_a_linked_audio_with_a_zero_rate_names_the_card(self):
        import torch
        audio_in = {"waveform": torch.zeros(1, 1, 8), "sample_rate": 0}
        with self.assertRaisesRegex(ValueError, "no file behind it"):
            self.nodes.NGContextAudio.execute(
                audio=self.name, name="Song", description="", content="voice", usage="reference", preserve="high",
                speaker="", trim_start=0.0, trim_end=0.0, context_id="audio_song", audio_in=audio_in)

    def test_pre_flight_rejects_an_inverted_linked_trim(self):
        message = self.nodes.NGContextAudio.validate_inputs(audio=self.name, trim_start=5.0, trim_end=2.0)
        self.assertIn("past Trim start", message)


if __name__ == "__main__":
    unittest.main()
