"""BUG-094 Regression Test: an audio card's Lyrics reach the shot and part planning on every card.

The Lyrics field of the H3 Context Audio node (context_options.lyrics) had exactly one consumer:
lyrics.story_lines, which only sees cards that play exactly (Use as Lock/Copy, or a Lock/Copy clip on a
Studio audio lane; lyrics.py::_spans). The default card - Use as Voice - and a Music style card park their
sound as a reference far away (legacy_bridge._audio_segments FAR_AWAY), so for them the words reached
NOTHING: not the AI's scene/shot planning (auto_direct._meanings carried only name, description, usage,
speaker; builder._audio_line described the sound but not its words), and no render part's prompt. Filling
the field on such a card changed nothing - the user's report.

Fix (same channels every comparable audio field already uses): context_build.widget_lyrics reads the
card's words; auto_direct._meanings carries them as "lyrics" (a Text card's "text" did this all along);
split's lines rule names them as the exact words; builder's Sound line carries them and the speech rule
uses them - untimed soundtrack words are written into lines by the plan (timed ones stay compiler-placed,
as before). Workflows without lyrics see no change: empty words add no key and no rule.

Run:
    python3 tests/ng/regressions/bug_094_audio_card_lyrics_reach_shot_planning.py
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _pack import module

context_build = module("ng.core.context_build")
auto_direct = module("ng.core.auto_direct")
builder = module("ng.core.builder")
split = module("ng.core.split")

TIMED = "[la:English]\n[00:01.00] My master, where did you go\n[00:05.00] I wait in the hall"
UNTIMED = "Words that no Whisper ever timed\nA second plain line"


def audio_card(usage, lyrics_text, context_id="audio_song"):
    options = {"schema_version": 1}
    if lyrics_text is not None:
        options["lyrics"] = lyrics_text
    return {"media": "audio", "widgets": {"audio": "song.mp3", "name": "Song", "usage": usage,
                                          "context_id": context_id,
                                          "context_options": json.dumps(options)}}


class WidgetLyrics(unittest.TestCase):
    def test_the_words_survive_their_stamps_and_an_untimed_line_is_kept(self):
        """plain_lines strips stamps and the language tag; words without times stay usable words."""
        self.assertEqual(context_build.widget_lyrics(audio_card("voice", TIMED)["widgets"]),
                         ["My master, where did you go", "I wait in the hall"])
        self.assertEqual(context_build.widget_lyrics(audio_card("voice", UNTIMED)["widgets"]),
                         ["Words that no Whisper ever timed", "A second plain line"])

    def test_broken_or_empty_options_are_no_words(self):
        """An older card with malformed context_options (BUG-030) and a card without lyrics give []."""
        broken = {"context_options": "{ broken json"}
        self.assertEqual(context_build.widget_lyrics(broken), [])
        self.assertEqual(context_build.widget_lyrics({}), [])
        self.assertEqual(context_build.widget_lyrics(audio_card("voice", None)["widgets"]), [])


class Meanings(unittest.TestCase):
    def test_bug_094_an_audio_card_meaning_carries_its_words(self):
        """The card's meaning is what the AI plans the shots from; without the words a Voice or Music style
        card reached it nameless (the report: filling Lyrics changed nothing)."""
        meaning = auto_direct._meanings([audio_card("voice", TIMED)])[0]
        self.assertEqual(meaning["lyrics"], ["My master, where did you go", "I wait in the hall"])
        self.assertEqual(meaning["usage"], "voice")

    def test_green_control_cards_without_lyrics_gain_no_key(self):
        """Workflows that leave Lyrics empty must see no change: no key, no words anywhere."""
        self.assertNotIn("lyrics", auto_direct._meanings([audio_card("voice", None)])[0])
        self.assertNotIn("lyrics", auto_direct._meanings([audio_card("music_style", "")])[0])
        text = {"media": "text", "widgets": {"name": "Note", "text": "a rule"}}
        self.assertNotIn("lyrics", auto_direct._meanings([text])[0])


class SplitPrompt(unittest.TestCase):
    def test_bug_094_the_words_reach_the_shot_planning(self):
        """AI: split into shots plans from the Prompt and the card meanings: the words must be in the request."""
        prompt = auto_direct.direction_prompt("The singer calls her master.", 10.0, [audio_card("voice", TIMED)])
        self.assertIn("My master, where did you go", prompt)
        self.assertIn("exact words", prompt)

    def test_green_control_a_card_without_lyrics_names_no_words(self):
        prompt = auto_direct.direction_prompt("The singer calls her master.", 10.0, [audio_card("voice", None)])
        self.assertNotIn("My master", prompt)
        self.assertNotIn('"lyrics"', prompt)


class PlanPrompt(unittest.TestCase):
    def test_bug_094_untimed_soundtrack_words_are_written_into_lines(self):
        """A Lock song whose words carry no Whisper times had none of its words placed anywhere: the plan was
        told the words were 'not known' and to write no lines. Now it sings the card's own words."""
        parts, sung = auto_direct.timing_hints({"shots": "", "continuity": "none"}, [audio_card("lock", UNTIMED)])
        self.assertEqual(sung, (), "untimed words must not claim placed times")
        prompt = builder.plan_prompt({"direction": "auto", "speech": "singing", "language": "English"},
                                     "A singer.", 10.0, [audio_card("lock", UNTIMED)], len(parts), parts, sung)
        self.assertIn("soundtrack's own words", prompt)
        self.assertNotIn("whose words are not known", prompt)
        self.assertIn("Words that no Whisper ever timed", prompt)

    def test_bug_094_a_reference_cards_words_reach_the_plan(self):
        """The default Voice card parks its sound as a reference: its words must still reach the planning."""
        prompt = builder.plan_prompt({"direction": "auto", "speech": "singing", "language": "English"},
                                     "A singer.", 10.0, [audio_card("voice", TIMED)])
        self.assertIn("words carried on an audio card", prompt)
        self.assertIn("My master, where did you go", prompt)

    def test_green_control_a_soundtrack_without_words_stays_not_known(self):
        """No lyrics: the old behaviour is kept word for word - write no lines, show the performance."""
        prompt = builder.plan_prompt({"direction": "auto", "speech": "singing", "language": "English"},
                                     "A singer.", 10.0, [audio_card("lock", None)])
        self.assertIn("whose words are not known", prompt)
        self.assertNotIn("exact words", prompt)

    def test_green_control_timed_lines_stay_compiler_placed(self):
        """With Whisper times, the plan must keep hands off the lines (the compiler places them per part):
        the sung rule wins, and a duplicated AI line is dropped from the answer."""
        parts, sung = auto_direct.timing_hints({"shots": "", "continuity": "none"}, [audio_card("lock", TIMED)])
        self.assertTrue(sung, "timed lines must be placed by the compiler as before")
        prompt = builder.plan_prompt({"direction": "auto", "speech": "singing", "language": "English"},
                                     "A singer.", 10.0, [audio_card("lock", TIMED)], len(parts), parts, sung)
        self.assertIn("already placed at their times", prompt)
        answer = {"look": "a hall", "shots": [
            {"start_s": 0, "end_s": 5, "action": "A full-body shot of the singer.", "camera": "The camera pushes in.",
             "beats": [], "lines": [{"speaker": "", "delivery": "sings", "language": "English",
                                     "text": "My master, where did you go"},
                                    {"speaker": "", "delivery": "sings", "language": "English",
                                     "text": "a line the AI invented"}]},
            {"start_s": 5, "end_s": 10, "action": "A close-up at the door.", "camera": "The camera holds a static shot.",
             "beats": [], "lines": []}]}
        result = split.parse_split(answer, 10.0, [audio_card("lock", TIMED)], "", sung)
        placed = [line["text"] for line in json.loads(result["shots"])["shots"][0]["spoken"]]
        self.assertIn("a line the AI invented", placed)
        self.assertFalse(any("My master" in text for text in placed), "the compiler places the timed line")


if __name__ == "__main__":
    unittest.main()
