"""BUG-095 Regression Test: a song card's [mm:ss] stamps reach the planning requests verbatim.

BUG-094 carried an audio card's words into the planning, but widget_lyrics read them through
lyrics.plain_lines, which strips every leading [mm:ss] stamp - the timing the planner needs to pace
beats and lines. The stamp was dropped at that one call, so all three request paths (split into shots,
the draft/redraft prompt, Make H3 prompt's Sound line) carried the words without their times.

Fix: lyrics.stamped_lines keeps each line as the card stores it (stamps and all; language tag and blank
lines dropped) and widget_lyrics serves it. plain_lines is untouched for the Whisper alignment, which
must see the words only. Workflows without lyrics stay byte-identical: an empty field still adds no key,
no Sound words and no rule.

Run:
    python3 tests/ng/regressions/bug_095_lyric_stamps_reach_planning_prompts.py
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
builder = module("ng.core.builder")

TIMED = "[la:English]\n[00:01.00] My master, where did you go\n[00:05.00] I wait in the hall"
STAMPED = ["[00:01.00] My master, where did you go", "[00:05.00] I wait in the hall"]
IDEA, SECONDS = "The singer calls her master.", 10.0
QUICK = {"direction": "auto", "speech": "singing", "language": "English"}
DRAFT_SHOTS = json.dumps({"schema_version": 1, "format": "h3-director-shots", "shots": [
    {"id": "shot1", "start": 0, "end": 10, "text": "A singer.", "media": [], "spoken": [], "locked_fields": []}]})


def audio_card(usage, lyrics_text):
    options = {"schema_version": 1}
    if lyrics_text is not None:
        options["lyrics"] = lyrics_text
    return {"media": "audio", "widgets": {"audio": "song.mp3", "name": "Song", "usage": usage,
                                          "context_id": "audio_song",
                                          "context_options": json.dumps(options)}}


class StampsVerbatim(unittest.TestCase):
    def test_bug_095_the_stamps_reach_the_split_prompt(self):
        """AI: split into shots: the Contexts JSON must carry each stamped line verbatim."""
        prompt = auto_direct.direction_prompt(IDEA, SECONDS, [audio_card("voice", TIMED)])
        for line in STAMPED:
            self.assertIn(line, prompt)
        self.assertIn("write the words without the stamp", prompt)

    def test_bug_095_the_stamps_reach_the_draft_prompt(self):
        """An existing shot document (draft/redraft) gets the same meanings: stamps verbatim there too."""
        prompt = auto_direct.direction_prompt(IDEA, SECONDS, [audio_card("voice", TIMED)],
                                              current_shots=DRAFT_SHOTS)
        for line in STAMPED:
            self.assertIn(line, prompt)

    def test_bug_095_the_stamps_reach_the_builder_sound_line(self):
        """Make H3 prompt: the Sound section carries the stamped words verbatim, with the no-copy guidance."""
        prompt = builder.plan_prompt(QUICK, "A singer.", SECONDS, [audio_card("voice", TIMED)])
        for line in STAMPED:
            self.assertIn('"%s"' % line, prompt)
        self.assertIn("[mm:ss] = when each line is heard", prompt)

    def test_bug_095_meanings_carry_the_stamped_lines(self):
        self.assertEqual(auto_direct._meanings([audio_card("voice", TIMED)])[0]["lyrics"], STAMPED)


class Untouched(unittest.TestCase):
    def test_plain_lines_still_strips_for_the_whisper_alignment(self):
        """The aligner diffs words against recognised speech: it must keep reading the stamp-free lines."""
        self.assertEqual(lyrics.plain_lines(TIMED), ["My master, where did you go", "I wait in the hall"])

    def test_bug_095_a_timed_lock_song_keeps_its_compiler_placement(self):
        """Lock/Copy lines stay compiler-placed from the parse, not the meanings: sung hints keep their
        story-time seconds and the plan is told the lines are already placed."""
        parts, sung = auto_direct.timing_hints({"shots": "", "continuity": "none"}, [audio_card("lock", TIMED)])
        self.assertEqual(sung, ((1.0, 5.0, "Song", "My master, where did you go"),
                                (5.0, 9.0, "Song", "I wait in the hall")))
        prompt = auto_direct.direction_prompt(IDEA, SECONDS, [audio_card("lock", TIMED)], timing=(parts, sung))
        self.assertIn("already placed at their times", prompt)

    def test_bug_095_empty_lyrics_give_byte_identical_prompts(self):
        """An empty Lyrics field must not change a single byte of either request."""
        for made in (auto_direct.direction_prompt,  # split prompt
                     lambda idea, s, c: auto_direct.direction_prompt(idea, s, c, current_shots=DRAFT_SHOTS),
                     lambda idea, s, c: builder.plan_prompt(QUICK, idea, s, c)):
            self.assertEqual(made(IDEA, SECONDS, [audio_card("voice", None)]),
                             made(IDEA, SECONDS, [audio_card("voice", "")]))

    def test_bug_095_stamped_lines_edges(self):
        """Language tag and blank lines go; multi-stamp and pause stamps are timing information and stay."""
        self.assertEqual(lyrics.stamped_lines(""), [])
        self.assertEqual(lyrics.stamped_lines(None), [])
        self.assertEqual(lyrics.stamped_lines("A\n\nB"), ["A", "B"])
        self.assertEqual(lyrics.stamped_lines("[la:de]\nX"), ["X"])
        self.assertEqual(lyrics.stamped_lines("[00:01.00][00:04.00] duo"), ["[00:01.00][00:04.00] duo"])
        self.assertEqual(lyrics.stamped_lines("[00:09.00]"), ["[00:09.00]"])
        self.assertEqual(lyrics.stamped_lines("Words that no Whisper ever timed\nA second plain line"),
                         ["Words that no Whisper ever timed", "A second plain line"])


if __name__ == "__main__":
    unittest.main()
