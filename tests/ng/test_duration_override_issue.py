"""The Director says when a connected duration_seconds overrides the Length widget.

What happens (user report, 2026-11-12):
A project of 10.1 s on the Length widget rendered 4 parts: a PrimitiveFloat of 30 s was connected
to duration_seconds (it trimmed the song), and the socket silently overrode the widget. The plan
ran 30 s, the shots stopped at 10.1 s, so the later render parts had no text at all — and the
Studio's plan preview still showed 10.1 s, because the payload carries widget values only.

Expected:
1. A connected duration_seconds that differs from the widget by more than 0.05 s adds the
   `length_override` issue ("Rendering 30s: the connected duration_seconds overrides the Length
   widget (10.1s).") to the plan's issues — the same list the grid notes live in.
2. Equal values, values within the epsilon and a disconnected socket add nothing.
3. The Studio's plan preview (ng.core.preview.plan_preview) applies the same override and issue
   when the payload carries duration_seconds, as ng_read.readPayload does for a connected socket.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _pack import comfy_available, load_full_pack, module

preview = module("ng.core.preview")
frames = module("ng.core.frames")

CODE = "length_override"


def preview_issues(director):
    result = preview.plan_preview({"director": {"idea": "A short cinematic loop", "project_seconds": 10.1, **director},
                                   "contexts": []})
    assert result.get("ok"), result.get("error")
    return result, [i for i in result.get("issues", []) if i.get("code") == CODE]


class TestDurationOverridePreview(unittest.TestCase):

    def test_mismatch_is_reported_and_overrides(self):
        result, found = preview_issues({"duration_seconds": 30.0})
        self.assertEqual(len(found), 1)
        self.assertIn("overrides the Length widget", found[0]["text"])
        self.assertEqual(result["requested_frames"], int(round(30.0 * frames.FPS)))
        self.assertLessEqual(abs(result["total_frames"] - result["requested_frames"]), frames.GRID_STEP)

    def test_equal_values_add_no_issue(self):
        _, found = preview_issues({"duration_seconds": 10.1})
        self.assertEqual(found, [])

    def test_within_epsilon_adds_no_issue(self):
        _, found = preview_issues({"duration_seconds": 10.12})
        self.assertEqual(found, [])

    def test_disconnected_adds_no_issue(self):
        _, found = preview_issues({})
        self.assertEqual(found, [])


def run_director(kwargs):
    pack = load_full_pack()
    nodes = pack.NODE_CLASS_MAPPINGS
    provider = nodes["NGAIProvider"].execute(
        source="local", provider="custom", base_url="http://127.0.0.1:18319",
        model="mock", api_key_env="", vision=True, audio=False,
        timeout_seconds=90, max_tokens=256, keep_alive=30, clip=["clip", 0]
    ).args[0]
    text = nodes["NGContextText"].execute(
        text="Slow camera.", name="Camera", description="",
        content="rules", usage="rule", preserve="high"
    ).args[0]
    context = nodes["NGContextConnector"].execute(
        clip=["clip", 0], vae=["vae", 0], audio_vae=["audio", 0],
        model_fl2va=["fl", 0], model_ref2va=["ref", 0], ai_provider=provider,
        context_text={"text_0": text}
    ).args[0]
    base = dict(context=context, idea="A baker opens a shop.", shots="",
                project_seconds=8.0, chunk_seconds=5.0, width=384, height=576,
                mode="auto", continuity="video_audio", pin_frames="22",
                soundscape="Wind", music="N/A")
    base.update(kwargs)
    return nodes["NGH3Director"].execute(**base)


def override_issues(output):
    return [i for i in output.args[0].project.issues if i.code == CODE]


@unittest.skipUnless(comfy_available(), "ComfyUI is not importable")
class TestDurationOverrideNode(unittest.TestCase):

    def test_mismatch_is_reported(self):
        found = override_issues(run_director({"duration_seconds": 12.0}))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].text, "Rendering 12.0s: the connected duration_seconds overrides the "
                                        "Length widget (8.0s).")

    def test_equal_values_add_no_issue(self):
        self.assertEqual(override_issues(run_director({"duration_seconds": 8.0})), [])

    def test_within_epsilon_adds_no_issue(self):
        self.assertEqual(override_issues(run_director({"duration_seconds": 8.04})), [])

    def test_beyond_epsilon_is_reported(self):
        self.assertEqual(len(override_issues(run_director({"duration_seconds": 8.06}))), 1)

    def test_disconnected_adds_no_issue(self):
        self.assertEqual(override_issues(run_director({})), [])

    def test_zero_falls_back_and_adds_no_issue(self):
        self.assertEqual(override_issues(run_director({"duration_seconds": 0.0})), [])


if __name__ == "__main__":
    unittest.main()
