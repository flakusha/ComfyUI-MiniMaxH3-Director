"""Prompt-level timestamp shift: a card's Trim start moves the [mm:ss] stamps the planning prompts see.

BUG-057 taught the compiler path that a line must never be placed before the story starts (start_s or a lane
offset preceding the first stamp clamps at 0, never negative). The prompt path learned nothing: widget_lyrics
served the stored file-time stamps verbatim, so a card trimmed to begin at file second 10 told the planner
"[00:12.00] ..." for a video that is a few seconds long, and every beat the AI paced off it ran 10 s late.

Fix: lyrics.story_stamps subtracts the card's trim start from every stamp (clamping at 0, dropping lines the
trim removed, shifting pause stamps, keeping multi-stamp lines), and widget_lyrics applies it - the one
chokepoint all three prompt paths (split into shots, draft/redraft meanings, Make H3 prompt) flow through.
The stored LRC stays file-time: the waveform, the player, "Stamp line" and Whisper alignment all speak file
time, and re-trimming must never rewrite the words' true times. story_lines keeps the compiler path.

Example: "[00:12.00] late bells" on a card with Trim start 10 reaches the prompt as "[00:02.00] late bells".

Run:
    python3 tests/ng/regressions/prompt_stamps_shift_with_the_trim_window.py
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _pack import module

lyrics = module("ng.core.lyrics")
context_build = module("ng.core.context_build")
auto_direct = module("ng.core.auto_direct")
context_options = module("ng.core.context_options")

SONG = "[la:English]\n[00:04.00] gone with the morning\n[00:08.00][00:12.00] late bells\n[00:10.00]\n[00:16.00] I wait"
IDEA, SECONDS = "The singer calls her master.", 10.0


def card(lyrics_text, start_s=0.0, end_s=0.0):
    options = dict(schema_version=1)
    if lyrics_text is not None:
        options["lyrics"] = lyrics_text
    if start_s:
        options["start_s"] = start_s
    if end_s:
        options["end_s"] = end_s
    return {"media": "audio", "widgets": {"audio": "song.mp3", "name": "Song", "usage": "voice",
                                          "context_id": "audio_song", "context_options": json.dumps(options)}}


class StoryStamps(unittest.TestCase):
    def test_a_line_at_file_second_12_with_a_trim_start_of_10(self):
        self.assertEqual(lyrics.story_stamps(["[00:12.00] late bells"], 10.0), ["[00:02.00] late bells"])

    def test_multi_stamp_and_pause_stamps_shift_too(self):
        self.assertEqual(lyrics.story_stamps(["[00:08.00][00:12.00] late bells", "[00:10.00]"], 10.0),
                         ["[00:00.00][00:02.00] late bells", "[00:00.00]"])

    def test_lines_sung_inside_the_trim_are_dropped(self):
        self.assertEqual(lyrics.story_stamps(lyrics.stamped_lines(SONG), 10.0),
                         ["[00:00.00][00:02.00] late bells", "[00:00.00]", "[00:06.00] I wait"])

    def test_an_end_drops_lines_that_start_at_or_after_it(self):
        self.assertEqual(lyrics.story_stamps(lyrics.stamped_lines(SONG), 10.0, 14.0),
                         ["[00:00.00][00:02.00] late bells", "[00:00.00]"])

    def test_untimed_lines_pass_through(self):
        self.assertEqual(lyrics.story_stamps(["Words that no Whisper ever timed"], 10.0),
                         ["Words that no Whisper ever timed"])


class StorageStaysFileTime(unittest.TestCase):
    def test_multi_stamp_parse_and_prompt_agree_on_the_repeated_line(self):
        """parse expands each stamp into its own line, so story_stamps sees the same repeated line the prompt serves."""
        language, lines = lyrics.parse(SONG)
        bells = sorted(line.start for line in lines if line.text == "late bells")
        self.assertEqual(bells, [8.0, 12.0])
        self.assertEqual(lyrics.story_stamps(["[00:08.00][00:12.00] late bells"], 10.0),
                         ["".join(lyrics.stamp(max(0.0, round(time - 10.0, 2))) for time in bells) + " late bells"])

    def test_parse_still_reads_the_stored_lrc_in_file_seconds(self):
        language, lines = lyrics.parse(SONG)
        self.assertEqual(language, "English")
        self.assertEqual([(line.start, line.text) for line in lines],
                         [(4.0, "gone with the morning"), (8.0, "late bells"), (12.0, "late bells"), (16.0, "I wait")])

    def test_the_context_options_never_grow_a_shifted_copy(self):
        item_options = context_options.ContextOptions.from_dict(
            json.loads(card(SONG, start_s=10.0)["widgets"]["context_options"]))
        self.assertEqual((item_options.start_s, item_options.end_s, item_options.lyrics), (10.0, 0.0, SONG))


class ThePromptsSeeTheShift(unittest.TestCase):
    def test_widget_lyrics_shift_by_the_card_window(self):
        self.assertEqual(context_build.widget_lyrics(card(SONG, start_s=10.0)["widgets"]),
                         ["[00:00.00][00:02.00] late bells", "[00:00.00]", "[00:06.00] I wait"])

    def test_the_split_prompt_carries_the_shifted_line(self):
        prompt = auto_direct.direction_prompt(IDEA, SECONDS, [card(SONG, start_s=10.0)])
        self.assertIn("[00:02.00] late bells", prompt)
        self.assertNotIn("[00:12.00] late bells", prompt)

    def test_without_a_trim_the_stamps_are_the_stored_ones(self):
        self.assertEqual(context_build.widget_lyrics(card(SONG)["widgets"]),
                         ["[00:04.00] gone with the morning", "[00:08.00][00:12.00] late bells", "[00:10.00]",
                          "[00:16.00] I wait"])


if __name__ == "__main__":
    unittest.main()
