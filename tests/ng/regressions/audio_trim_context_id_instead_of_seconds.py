"""A context ID in a Trim widget must fail validation with a message, never with an exception.

A workflow saved before Trim start/end existed restores positionally: the Context ID lands in
Trim start and the colour in Trim end. NGContextAudio.validate_inputs used to float() the raw value,
so the ValueError escaped into the downstream node's inner-validation handler and was misreported
there as "Exception when validating inner node", hiding this node's own error (user report:
NGContextAudio 25, "could not convert string to float: 'ctx_...'").

Run:
    python3 tests/ng/regressions/audio_trim_context_id_instead_of_seconds.py
"""
import asyncio
import importlib
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _pack import comfy_available, load_full_pack

CTX = "ctx_1b5fa4deee838dc21dc09da934196cfc"


def audio_inputs(**overrides):
    raw = {"audio": "voice.wav", "name": "", "description": "", "content": "voice", "usage": "voice",
           "preserve": "high", "speaker": "", "trim_start": 0.0, "trim_end": 0.0, "context_id": "ctx_song",
           "color": "", "content_roles": "", "usage_roles": "", "context_options": ""}
    raw.update(overrides)
    return raw


@unittest.skipUnless(comfy_available(), "ComfyUI is not importable")
class TrimContextIdInsteadOfSeconds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        saved_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0], "--cpu"]
            cls.pack = load_full_pack()
            cls.node = cls.pack.NODE_CLASS_MAPPINGS["NGContextAudio"]
            cls.execution = importlib.import_module("execution")
            cls.comfy_nodes = importlib.import_module("nodes")
        finally:
            sys.argv = saved_argv

    def test_pre_flight_names_a_context_id_in_trim_start(self):
        message = self.node.validate_inputs(audio="voice.wav", trim_start=CTX, trim_end=0.0)
        self.assertIn("must be seconds", message)
        self.assertIn(CTX, message)

    def test_pre_flight_names_a_context_id_in_trim_end(self):
        message = self.node.validate_inputs(audio="voice.wav", trim_start=0.0, trim_end=CTX)
        self.assertIn("must be seconds", message)

    def test_pre_flight_still_rejects_an_inverted_window(self):
        self.assertIn("past Trim start",
                      self.node.validate_inputs(audio="voice.wav", trim_start=5.0, trim_end=2.0))

    def test_fingerprint_falls_back_to_the_stored_range_on_a_context_id(self):
        """fingerprint_inputs runs during validation: the shifted ID is truthy, so a truthiness
        guard would compare `str > int` and raise TypeError before _linked_float. A non-number
        must fall back to the stored options range instead."""
        from unittest.mock import patch as mock_patch
        stored = '{"schema_version": 1, "start_s": 2.0, "end_s": 6.0}'
        probe = importlib.import_module(self.pack.__name__ + ".ng.adapters.probe")
        with mock_patch.object(probe, "file_stamp", return_value="stamp"):
            shifted = self.node.fingerprint_inputs(audio="voice.wav", trim_start=CTX, trim_end=CTX,
                                                   context_options=stored)
            ranged = self.node.fingerprint_inputs(audio="voice.wav", trim_start=2.0, trim_end=6.0)
            plain = self.node.fingerprint_inputs(audio="voice.wav")
        self.assertEqual(shifted, ranged)
        self.assertNotEqual(shifted, plain)

    def test_real_validation_reports_the_node_itself_not_an_inner_exception(self):
        """Through the actual queue validation path: node 25 holds the shifted ID and is linked
        from a downstream consumer, exactly as in the report (validate prompt for output 24)."""
        prompt = {"25": {"class_type": "NGContextAudio", "inputs": audio_inputs(trim_start=CTX)},
                  "24": {"class_type": "NGContextAudio", "inputs": audio_inputs(audio_in=["25", 1])}}
        with patch.dict(self.comfy_nodes.NODE_CLASS_MAPPINGS, {"NGContextAudio": self.node}):
            validated = {}
            valid, _, _ = asyncio.run(self.execution.validate_inputs("audit", prompt, "24", validated))
        self.assertFalse(valid)
        self.assertIn("25", validated)
        kinds = [reason["type"] for reason in validated["25"][1]]
        self.assertNotIn("exception_during_inner_validation", kinds,
                         "a validator must return its message, never raise into the inner handler")
        self.assertIn("invalid_input_type", kinds)
        self.assertIn("custom_validation_failed", kinds)
        custom = [reason["details"] for reason in validated["25"][1]
                  if reason["type"] == "custom_validation_failed"]
        self.assertTrue(any("must be seconds" in details and CTX in details for details in custom),
                        custom)

    def test_real_validation_of_a_shifted_card_alone_raises_nothing(self):
        prompt = {"25": {"class_type": "NGContextAudio", "inputs": audio_inputs(trim_start=CTX)}}
        with patch.dict(self.comfy_nodes.NODE_CLASS_MAPPINGS, {"NGContextAudio": self.node}):
            validated = {}
            valid, _, _ = asyncio.run(self.execution.validate_inputs("audit", prompt, "25", validated))
        self.assertFalse(valid)
        self.assertEqual(validated["25"][0], False)


if __name__ == "__main__":
    unittest.main()
