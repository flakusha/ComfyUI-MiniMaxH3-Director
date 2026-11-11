"""A song's timed lyrics reach only the render part they are sung in; the Look is what every part shares.

User's 27 s elf song (2026-10-05): all eight lines were in each of the three 9 s parts, twice (Prompt and Shot), and only
the first was a sung line at all.
"""
import json
import os
import sys
import unittest

from _pack import COMFY_ROOT, module, stub_prompt_server

if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)
stub_prompt_server()   # the chapter-mark and audio-mux tests import nodes; they need a PromptServer to register against

ctx, opts = module("ng.core.context"), module("ng.core.context_options")
chunks, project, lyrics = module("ng.core.chunks"), module("ng.core.project"), module("ng.core.lyrics")

SONG = """[la:English]
[00:01.00] My master, where did your footsteps fade?
[00:05.00] I still wait where your shadow stayed.
[00:09.50]
[00:12.00] The halls are cold, the candles burn,
[00:16.00] but every flame still waits for your return.
[00:21.00] You taught my hands to shape the light,
"""


def doc(shots=(), audio=(), look=None):
    data = {"format": "h3-director-shots", "schema_version": 1, "shots": list(shots), "audio": list(audio)}
    if look is not None:
        data["look"] = look
    return json.dumps(data)


def shot(identity, start, end, text="She sings in the hall."):
    return dict(id=identity, start=start, end=end, text=text, media=[], spoken=[], locked_fields=[], draft=None)


def lane(start=0.0, end=27.0, offset=0.0, mode="lock"):
    return {"schema_version": 1, "id": "c1", "context_id": "song", "start": start, "end": end, "offset": offset, "mode": mode, "lane": "l1"}


def song(text=SONG, usage=("voice",), speaker="", length=27.0):
    return ctx.ContextItem("song", "Song", "audio", "song.mp3", usage=usage, length_s=length, speaker=speaker,
                           options=opts.ContextOptions(lyrics=text))


def mage():
    return ctx.ContextItem("mage", "Mage", "image", "mage.png", usage=("reference",), content=("character",))


def plan(shots_text, items, mode="fl2va", seconds=27.0, idea="An elf mage sings a sad song."):
    director = ctx.DirectorContext(idea=idea, shots_text=shots_text, items=items, project_seconds=seconds, mode=mode)
    return project.plan_project(director, 0, chunks.ContinuityPlan("video_audio"))


class Parse(unittest.TestCase):
    def test_lines_end_at_the_next_stamp_and_a_bare_stamp_is_a_pause(self):
        language, lines = lyrics.parse(SONG)
        self.assertEqual(language, "English")
        self.assertEqual([(l.start, l.end) for l in lines][:3], [(1.0, 5.0), (5.0, 9.5), (12.0, 16.0)])
        self.assertEqual(lines[-1].end, 21.0 + lyrics.LAST_LINE_SECONDS)

    def test_lrc_roundtrip_keeps_times_and_pauses(self):
        language, lines = lyrics.parse(SONG)
        self.assertEqual(lyrics.parse(lyrics.to_lrc(language, lines)), (language, lines))

    def test_multi_stamp_lines_agree_between_parse_and_story_stamps(self):
        # Backend F2: parse expands one multi-stamp line into one LyricLine per stamp (for the
        # compiler's story_lines), while story_stamps keeps the single text line with every stamp
        # shifted (for the planners' text). Both must carry the same words at the same shifted times.
        multi = "[00:01.00][00:05.00] Echo echo"
        _, parsed = lyrics.parse(multi)
        self.assertEqual([l.text for l in parsed], ["Echo echo", "Echo echo"])
        shifted = lyrics.story_stamps(lyrics.stamped_lines(multi), 1.0)
        self.assertEqual(shifted, ["[00:00.00][00:04.00] Echo echo"])
        self.assertEqual([round(l.start - 1.0, 2) for l in parsed], [0.0, 4.0])

    def test_the_option_survives_the_card_json_and_old_cards_have_none(self):
        options = opts.ContextOptions(lyrics=SONG)
        self.assertEqual(opts.ContextOptions.from_dict(options.to_dict()), options)
        self.assertNotIn("lyrics", opts.ContextOptions().to_dict())


class Align(unittest.TestCase):
    # Whisper small on the user's song (2026-10-05) heard all eight lines; a misheard word must not move a line.
    WORDS = [(" My", 0.0, 0.4), (" master,", 0.4, 1.2), (" where", 1.3, 1.6), (" did", 1.6, 1.8), (" your", 1.8, 2.0),
             (" footsteps", 2.0, 2.6), (" fade?", 2.6, 3.2), (" I", 3.48, 3.6), (" still", 3.6, 4.0), (" wade", 4.0, 4.3),
             (" where", 4.3, 4.6), (" your", 4.6, 4.8), (" shadow", 4.8, 5.4), (" stayed", 5.4, 6.4)]

    def test_the_users_words_keep_the_recognised_times(self):
        lines = lyrics.align(["My master, where did your footsteps fade?", "I still wait where your shadow stayed."], self.WORDS)
        self.assertEqual([l.text for l in lines], ["My master, where did your footsteps fade?", "I still wait where your shadow stayed."])
        self.assertEqual([(l.start, l.end) for l in lines], [(0.0, 3.2), (3.48, 6.4)])

    def test_an_unheard_line_sits_between_its_neighbours(self):
        lines = lyrics.align(["My master, where did your footsteps fade?", "La la la la", "I still wait where your shadow stayed."],
                             self.WORDS)
        self.assertEqual(lines[1].start, 3.2)
        self.assertEqual(lines[1].end, 3.48)
        self.assertEqual(lines[2].start, 3.48)

    def test_lines_after_the_last_heard_one_share_the_rest_of_the_file(self):
        lines = lyrics.align(["My master, where did your footsteps fade?", "Come back to me", "through fire and rain"], self.WORDS[:7], 60.0)
        self.assertEqual([(l.start, l.end) for l in lines[1:]], [(3.2, 31.6), (31.6, 60.0)])

    def test_an_unintelligible_line_is_placed_between_its_neighbours(self):
        # The transcribe role emits [unintelligible] for words it cannot hear; the brackets strip to no
        # token, so the line takes interpolated timing without moving its neighbours (S5).
        lines = lyrics.align(["My master, where did your footsteps fade?", "[unintelligible]", "I still wait where your shadow stayed."],
                             self.WORDS)
        self.assertEqual([(l.start, l.end) for l in lines[::2]], [(0.0, 3.2), (3.48, 6.4)])
        self.assertEqual((lines[1].start, lines[1].end), (3.2, 3.48))
        self.assertEqual(lines[1].text, "[unintelligible]")

    def test_plain_lines_drop_stamps_language_and_blank_lines(self):
        self.assertEqual(lyrics.plain_lines(SONG)[:2], ["My master, where did your footsteps fade?", "I still wait where your shadow stayed."])
        self.assertEqual(len(lyrics.plain_lines(SONG)), 5)


class Placement(unittest.TestCase):
    def prompts(self, result):
        return [c.prompt for c in result.compiled]

    def test_each_part_sings_only_its_own_lines(self):
        result = plan(doc([shot("s1", 0, 27)], [lane()]), (song(),))
        self.assertEqual(len(result.compiled), 3)
        parts = self.prompts(result)
        first, second, third = parts
        self.assertIn("<d>[English] My master, where did your footsteps fade?</d>", first)
        self.assertIn("<d>[English] I still wait where your shadow stayed.</d>", first)
        self.assertNotIn("halls are cold", first)
        self.assertIn("<d>[English] The halls are cold, the candles burn,</d>", second)
        self.assertIn("You taught my hands", third)
        self.assertTrue(all(p.count("My master") <= 1 for p in parts))
        self.assertTrue(all("sings" in p for p in parts))

    def test_a_lane_offset_shifts_the_song_against_the_story(self):
        # The clip starts at story 8 s and plays the file from 0: "My master" (file 1-5 s) is sung at story 9-13 s.
        result = plan(doc([shot("s1", 0, 27)], [lane(start=8.0, end=27.0)]), (song(),))
        self.assertNotIn("My master", result.compiled[0].prompt)
        self.assertIn("My master", result.compiled[1].prompt)

    def test_the_singer_is_the_figure_the_voice_belongs_to(self):
        result = plan(doc([shot("s1", 0, 27, "@Mage sings in the hall.")], [lane()]), (mage(), song(speaker="mage")), mode="ref2va")
        self.assertRegex(result.compiled[0].prompt, r"<Subject 1> \(S1\) sings, <d>\[English\] My master")

    def test_in_first_last_frame_mode_the_figure_is_shown_in_the_picture_and_sings_by_name(self):
        # Elf song in FL2VA (2026-10-05): "<Picture 1> stands in a three-quarter portrait ... the singer (S1) sings".
        first = ctx.ContextItem("mage", "Mage", "image", "mage.png", usage=("first_frame",), content=("character",))
        compiled = plan(doc([shot("s1", 0, 27, "@Mage stands in the hall. @Mage looks up.")], [lane()]), (first, song(speaker="mage"))).compiled
        prompt = compiled[0].prompt
        # the 27 s shot spans several parts: each part gets its own share of the action (ng.core.shot_share, 2026-10-13)
        self.assertIn("[Shot 1] Mage, shown in <Picture 1>, stands in the hall.", prompt)
        self.assertNotIn("looks up", prompt)
        self.assertIn("Mage looks up.", compiled[-1].prompt)
        self.assertIn("Mage (S1) sings, <d>[English] My master", prompt)

    def test_a_reference_song_is_not_sung(self):
        result = plan(doc([shot("s1", 0, 27)]), (song(usage=("music_style",)),))
        self.assertNotIn("<d>", "".join(self.prompts(result)))

    def test_a_song_locked_over_the_whole_video_is_no_dulling_chain(self):
        # A full-video lock requires a file long enough to cover that video (BUG-083).
        long = plan(doc([shot("s1", 0, 177)], [lane(end=177.0)]), (song(length=177.0),), seconds=177.0)
        self.assertNotIn("audio_dulling", {i.code for i in long.issues})
        unlocked = plan(doc([shot("s1", 0, 177)], [lane(end=60.0)]), (song(length=177.0),), seconds=177.0)
        self.assertIn("audio_dulling", {i.code for i in unlocked.issues}, "generated sound after 60 s is still carried")

    def test_a_short_song_cannot_lock_sound_past_its_file_end(self):
        result = plan(doc([shot("s1", 0, 177)], [lane(end=177.0)]), (song(),), seconds=177.0)
        self.assertIn("audio_dulling", {i.code for i in result.issues})
        timing = module("ng.core.audio_timing")
        self.assertEqual(timing.locked_spans(result.context), [(0, 648)])

    def test_a_carried_shot_keeps_its_timed_lines(self):
        result = plan(doc([shot("s1", 0, 27)], [lane()]), (song(),))
        self.assertTrue(result.plan.chunks[1].shots[0].carried)
        self.assertIn("candles burn", result.compiled[1].prompt)


class Captions(unittest.TestCase):
    def test_story_time_lrc_srt_and_chapters_come_from_the_same_window(self):
        lrc = lyrics.card_lrc(SONG)
        self.assertIn("[00:01.00] My master", lrc)
        language, lines = lyrics.parse(SONG)
        srt = lyrics.to_srt(lines)
        self.assertTrue(srt.startswith("1\n00:00:01,000 --> 00:00:05,000\nMy master"))
        # the card trim shifts the caption, like the planner's story_stamps (P1.1)
        trimmed = lyrics.card_lrc(SONG, 4.0, 0.0)
        self.assertIn("[00:01.00] I still wait", trimmed)
        self.assertNotIn("My master", trimmed)
        # the Director warns when the video outlasts the song (P1.2), naming the card
        result = plan(doc([shot("s1", 0, 27)], [lane()]), (song(),))
        codes = {i.code: i for i in result.issues}
        self.assertNotIn("audio_shorter_than_project", codes, "a 27 s song covers a 27 s video")
        long = plan(doc([shot("s1", 0, 60)], [lane(end=60.0)]), (song(),), seconds=60.0)
        self.assertIn("audio_shorter_than_project", {i.code for i in long.issues})

    def test_asr_selection_stays_whisper_without_backends(self):
        asr = module("ng.core.asr")
        self.assertEqual(asr.select("no-such-backend")[0], "whisper")
        self.assertEqual(asr.select()[0], "whisper")

    def test_qwen3_dirs_find_the_pair_in_one_asr_folder(self):
        asr = module("ng.core.asr")
        listing = lambda folder: ["Qwen3-ASR-0.6B-hf", "Qwen3-ForcedAligner-0.6B-hf"] if folder == "asr" else []
        self.assertEqual(asr.qwen3_dirs(listing), ("Qwen3-ASR-0.6B-hf", "Qwen3-ForcedAligner-0.6B-hf"))
        self.assertEqual(asr.qwen3_dirs(lambda folder: []), (None, None))
        self.assertEqual(asr.qwen3_dirs(None), (None, None))

    def test_qwen3_discovery_lists_directories_not_files(self):
        # get_filename_list returns files only (even .cache/huggingface/.gitignore), so dir purposes scan the roots.
        import os
        import tempfile
        model_files = module("ng.adapters.model_files")
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "Qwen3-ASR-1.7B-hf"))
            os.makedirs(os.path.join(root, ".cache", "huggingface"))
            open(os.path.join(root, ".cache", "huggingface", ".gitignore"), "w").close()
            with unittest.mock.patch.object(model_files.folder_paths, "folder_names_and_paths", {"asr": []}), \
                    unittest.mock.patch.object(model_files.folder_paths, "get_folder_paths", return_value=[root]):
                self.assertEqual(model_files.dir_names("asr"), ["Qwen3-ASR-1.7B-hf"])
                self.assertEqual(model_files.names("qwen3_asr"), ["Qwen3-ASR-1.7B-hf"])
                self.assertEqual(model_files.find("qwen3_asr"), "Qwen3-ASR-1.7B-hf")
                self.assertEqual(model_files.find("qwen3_aligner"), None)

    def test_qwen3_empty_transcript_skips_the_aligner(self):
        # Noise can transcribe to nothing: no aligner load, no words, no crash.
        import sys
        import types
        timing = module("ng.adapters.lyrics_timing")
        stub = types.ModuleType("transformers")
        stub.__version__ = "5.19.0"
        aligner_loads = []
        class FakeModel:
            device = "cpu"
            dtype = None
            def generate(self, **kwargs):
                return [[0]]
        class FakeProcessor:
            def apply_transcription_request(self, samples, language=None, prompt=None):
                return {"input_features": samples}
            def batch_decode(self, tokens, skip_special_tokens=True):
                return ["assistant\nlanguage None<asr_text>"]
        stub.AutoModelForMultimodalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel())
        stub.AutoModelForTokenClassification = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: aligner_loads.append(1))
        stub.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor())
        with unittest.mock.patch.dict(sys.modules, {"transformers": stub}), \
                unittest.mock.patch.object(timing, "_qwen3_dirs", return_value=("asr-dir", "aligner-dir")):
            words, kept = timing._qwen3_words([0.0] * 1600, [], "en", "cpu")
        self.assertEqual(words, [])
        self.assertEqual(len(kept), 1)
        self.assertEqual(aligner_loads, [])

    def test_qwen3_factory_returns_whisper_shaped_timings_with_stubbed_transformers(self):
        # No weights on this box: stub transformers + folders, assert the factory returns the same
        # shape time_lyrics returns (lrc/language/lines/heard_words) from the aligner words.
        import sys
        import types
        timing = module("ng.adapters.lyrics_timing")
        asr = module("ng.core.asr")
        heard = [("hello", 0.0, 0.5), ("world", 0.6, 1.0)]
        stub = types.ModuleType("transformers")
        stub.__version__ = "5.19.0"
        class FakeBatchFeature(dict):
            def to(self, device, dtype=None):
                return self
        class FakeModel:
            device = "cpu"
            dtype = None
            config = types.SimpleNamespace(timestamp_token_id=7)
            def generate(self, **kwargs):
                return [[0]]
            def __call__(self, **kwargs):
                return types.SimpleNamespace(logits=[[0.0]])
        class FakeProcessor:
            def apply_transcription_request(self, samples, language=None, prompt=None):
                return {"input_features": samples}
            def batch_decode(self, tokens, skip_special_tokens=True):
                return ["assistant\nlanguage None<asr_text>hello world"]
            def prepare_forced_aligner_inputs(self, part, text, lang):
                return FakeBatchFeature(input_ids=[[1]]), [["hello", "world"]]
            def decode_forced_alignment(self, logits, input_ids, word_lists, timestamp_token_id):
                assert timestamp_token_id == 7
                return [[{"text": word, "start_time": start, "end_time": end} for word, start, end in heard]]
        stub.AutoModelForMultimodalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel())
        stub.AutoModelForTokenClassification = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel())
        stub.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeProcessor())
        FakeModel.generate = lambda self, **kwargs: [[0]]
        with unittest.mock.patch.dict(sys.modules, {"transformers": stub}), \
                unittest.mock.patch.object(timing, "_qwen3_dirs", return_value=("asr-dir", "aligner-dir")), \
                unittest.mock.patch.object(timing, "decode_audio") as decode:
            import torch
            decode.return_value = (torch.zeros(1, 16000), 16000)
            result = timing._qwen3_transcribe("song.mp3", "[00:00.00] hello world", "cpu")
        self.assertEqual(result["heard_words"], 2)
        self.assertEqual(result["lines"], 1)
        self.assertIn("hello world", result["lrc"])
        # the stub factory is registered under both names once lyrics_timing is imported
        self.assertIs(asr.get("qwen3")[1], timing._qwen3_transcribe)
        self.assertIs(asr.get("qwen3-asr")[1], timing._qwen3_transcribe)

    def test_fallback_mux_only_fills_silence(self):
        import torch
        locks = module("ng.adapters.audio_locks")
        silent = torch.zeros(1, 1, 100)
        heard = torch.ones(1, 1, 100)
        fallback = {"waveform": torch.ones(1, 1, 100) * 0.5, "sample_rate": 32000}
        self.assertTrue((locks._mux_fallback(silent, fallback) == 0.5).all())
        self.assertIs(locks._mux_fallback(heard, fallback), heard, "generated sound is untouched")
        self.assertIs(locks._mux_fallback(silent, None), silent, "unconnected is byte-identical")

    def test_lyrics_continuation_narrows_story_lines_per_chunk(self):
        result = plan(doc([shot("s1", 0, 27)], [lane()]), (song(),))
        first, second = result.plan.chunks[0], result.plan.chunks[1]
        # sung_lines is the per-chunk narrowing the compiler already uses: each chunk sings the lines
        # whose middle falls inside it, so a chained video keeps singing where the song is (P3 idea 7).
        first_sung = module("ng.core.lyrics").sung_lines(result.context, first)
        second_sung = module("ng.core.lyrics").sung_lines(result.context, second)
        self.assertTrue(first_sung and second_sung, "both chunks sing lines")
        whole = lyrics.story_lines(result.context)
        self.assertEqual(len(first_sung) + len(second_sung) + len(module("ng.core.lyrics").sung_lines(result.context, result.plan.chunks[2])), len(whole))

    def test_chapter_marks_cover_the_video_end_to_end(self):
        from _pack import module as pack_module
        chunk_node = pack_module("ng.nodes.chunk_node")
        result = plan(doc([shot("s1", 0, 27)], [lane()]), (song(),))
        text = chunk_node.chapter_text(result)
        self.assertIn("Render part 1", text)
        self.assertIn("Render part 3", text)

    def test_caption_polish_prompt_never_writes_times(self):
        native = module("ng.adapters.native_ai")
        prompt, media = native.request_prompt("caption", {"words": "hello world", "language": "English"}, None)
        self.assertEqual(media, "text")
        self.assertIn("hello world", prompt)
        self.assertNotIn("timestamps", prompt.lower())


class Ark(unittest.TestCase):
    ANSWER = {"audio_info": {"duration": 2000},
              "result": {"utterances": [{"text": "hello world", "start_time": 100, "end_time": 1900,
                                         "words": [{"text": "hello", "start_time": 100, "end_time": 900},
                                                   {"text": "world", "start_time": 1000, "end_time": 1900}]}]}}

    def test_ark_factory_returns_whisper_shaped_timings_from_a_canned_answer(self):
        # No keys on this box: stub the submit+poll network call, assert the factory returns the same
        # shape time_lyrics returns (lrc/language/lines/heard_words) from the answer's word times.
        from unittest import mock
        timing = module("ng.adapters.lyrics_timing")
        ark = module("ng.adapters.ark_asr")
        asr = module("ng.core.asr")
        env = {"H3_ARK_API_KEY": "test-key", "H3_ARK_AUDIO_URL_BASE": "http://bucket.example/songs/"}
        with mock.patch.dict(os.environ, env), mock.patch.object(ark, "_recognize", return_value=self.ANSWER):
            result = timing.time_lyrics("song.mp3", "[la:English]\nhello world\n", "cpu", engine="ark")
        self.assertEqual(result["heard_words"], 2)
        self.assertEqual(result["lines"], 1)
        self.assertEqual(result["language"], "English")
        self.assertIn("[00:00.10] hello world", result["lrc"])
        self.assertIs(asr.get("ark")[1], ark.ark_transcribe)
        self.assertIs(asr.get("seed-asr")[1], ark.ark_transcribe)

    def test_ark_names_the_missing_variables(self):
        from unittest import mock
        ark = module("ng.adapters.ark_asr")
        empty = {"H3_ARK_API_KEY": "", "H3_ARK_APP_ID": "", "H3_ARK_ACCESS_KEY": ""}
        with mock.patch.dict(os.environ, empty):
            with self.assertRaisesRegex(ValueError, "H3_ARK_API_KEY"):
                ark.ark_transcribe("song.mp3", "", "cpu")
        with mock.patch.dict(os.environ, {"H3_ARK_API_KEY": "k", "H3_ARK_AUDIO_URL_BASE": ""}):
            with self.assertRaisesRegex(ValueError, "H3_ARK_AUDIO_URL_BASE"):
                ark.ark_transcribe("song.mp3", "", "cpu")

class Look(unittest.TestCase):
    def test_the_prompt_is_not_pasted_into_every_part_once_shots_exist(self):
        result = plan(doc([shot("s1", 0, 27)], look="Dark gothic fantasy, cinematic anime style."), ())
        for n, prompt in enumerate(c.prompt for c in result.compiled):
            self.assertNotIn("An elf mage sings a sad song.", prompt)
            go_on = "Continue the same scene and action from the preceding video, preserving the subject and setting. " if n else ""
            self.assertIn("integrated_multimodal_description: [Shot 1] Dark gothic fantasy, cinematic anime style. %sShe sings" % go_on, prompt)

    def test_a_scene_note_is_story_and_other_notes_go_into_every_part(self):
        # User (2026-10-05): the prompt in H3 Context Text was "not understood" - it was pasted, whole, into every part.
        scene = ctx.ContextItem("t1", "Song scene", "text", "She sings the whole verse: My master ...", usage=("scene",))
        rule = ctx.ContextItem("t2", "Rule", "text", "No camera cuts.", usage=("rule",))
        with_shots = plan(doc([shot("s1", 0, 27)]), (scene, rule))
        self.assertTrue(all("whole verse" not in c.prompt and "No camera cuts." in c.prompt for c in with_shots.compiled))
        without = plan("", (scene,), seconds=8.0)
        self.assertIn("whole verse", without.compiled[0].prompt, "a text-only video without shots still gets its scene")
        direct = module("ng.core.auto_direct")
        card = {"media": "text", "widgets": {"name": "Song scene", "text": "She sings the whole verse.", "usage": "scene"}}
        self.assertIn("She sings the whole verse.", direct.direction_prompt("", 27, [card]), "the split reads it with an empty Prompt")
        with self.assertRaisesRegex(ValueError, "Scene note"):
            direct.direction_prompt("", 27, [{**card, "widgets": {**card["widgets"], "usage": "rule"}}])

    def test_without_shots_the_prompt_still_carries_the_video(self):
        result = plan("", (), seconds=8.0)
        self.assertIn("[Shot 1] An elf mage sings a sad song.", result.compiled[0].prompt)

    def test_reference_mode_keeps_the_look_before_shot_one(self):
        result = plan(doc([shot("s1", 0, 8, "@Mage sings.")], look="Cinematic anime style."), (mage(),), mode="ref2va", seconds=8.0)
        self.assertIn("detailed_description: Cinematic anime style. [Shot 1]", result.compiled[0].prompt)


if __name__ == "__main__":
    unittest.main()
