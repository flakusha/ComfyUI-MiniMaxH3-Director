"""Regression test: the H3 Director's external duration control and adapted-length output.

What happens (Ist):
NGH3Director had no way to feed a desired video length in from another node and no way to read
the real length back — the length the planner actually delivers after snapping to H3's 17k+5
frame grid (GRID_STEP 17, GRID_OFFSET 5, 24 fps) can differ from the Length widget by a
fraction of a second.

Expected:
1. `duration_seconds` is an optional connection-only input; when connected and > 0 it overrides
   the `project_seconds` widget. 0 or an omitted socket keeps the widget's value (a value that
   was never set arrives as 0). A negative length is refused with a ValueError naming the input.
2. `length_seconds` is the last output and equals the planned total frames divided by 24 —
   the same number the plan panel reports — for both the widget path and the socket path.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
NG_TESTS = os.path.dirname(HERE)
if NG_TESTS not in sys.path:
    sys.path.insert(0, NG_TESTS)

from _pack import comfy_available, load_full_pack, module

frames = module("ng.core.frames")


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


@unittest.skipUnless(comfy_available(), "ComfyUI is not importable")
class TestBug091DurationSeconds(unittest.TestCase):

    def test_schema_has_input_and_last_output(self):
        schema = load_full_pack().NODE_CLASS_MAPPINGS["NGH3Director"].define_schema()
        self.assertIn("duration_seconds", [i.id for i in schema.inputs])
        self.assertEqual([o.id for o in schema.outputs][-1], "length_seconds")
        duration = next(i for i in schema.inputs if i.id == "duration_seconds")
        self.assertTrue(duration.optional)

    def test_connected_duration_overrides_widget(self):
        output = run_director({"duration_seconds": 12.0})
        length_seconds = output.args[2]
        total_frames = round(length_seconds * frames.FPS)
        # the plan delivers the grid length nearest 12s: within one grid step, on the grid
        self.assertLessEqual(abs(total_frames - int(round(12.0 * frames.FPS))), frames.GRID_STEP)
        self.assertTrue(frames.on_grid(total_frames))

    def test_zero_falls_back_to_widget(self):
        with_socket = run_director({"duration_seconds": 0.0})
        without = run_director({})
        self.assertEqual(with_socket.args[2], without.args[2])

    def test_negative_is_refused(self):
        with self.assertRaises(ValueError):
            run_director({"duration_seconds": -3.0})

    def test_output_matches_planned_frames(self):
        output = run_director({})
        plan = module("ng.core.project")
        ctx_mod = module("ng.core.context")
        chunk_mod = module("ng.core.chunks")
        ctx = ctx_mod.DirectorContext(idea="A baker opens a shop.", project_seconds=8.0,
                                      width=384, height=576, mode="auto", shots_text="",
                                      soundscape="Wind", music="N/A")
        continuity = chunk_mod.ContinuityPlan(mode="video_audio", video_frames=22, audio_frames=176)
        project = plan.plan_project(ctx, 5.0, continuity, None)
        self.assertEqual(output.args[2], frames.frames_to_seconds(project.plan.total_frames))


if __name__ == "__main__":
    unittest.main()
