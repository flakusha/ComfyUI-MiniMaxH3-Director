"""Studio lab node parity with the product Director (review F3, IMPROVEMENT path).

NGDirectorStudio carries NGH3Director's extras as its own trailing widgets, so the shared Studio panel's
writes land and its AI/control indicators resolve through its own sockets.
"""
import unittest

from _pack import comfy_available, load_full_pack

EXTRAS = ("duration_seconds", "style", "prompt_override",
          "resize_method", "ref_image_size", "resume", "resume_delete_done", "control_strength", "control_start",
          "control_end", "models", "output", "spectrum")
SOCKETS = ("ai_provider", "preview_vae", "model_patch")


@unittest.skipUnless(comfy_available(), "ComfyUI is not importable")
class StudioParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_full_pack().NODE_CLASS_MAPPINGS
        cls.studio = cls.nodes["NGDirectorStudio"]
        cls.product = cls.nodes["NGH3Director"]

    def inputs(self, node):
        return {item.id: item for item in node.GET_SCHEMA().inputs}

    def test_studio_carries_every_product_extra_with_matching_defaults(self):
        studio, product = self.inputs(self.studio), self.inputs(self.product)
        for name in EXTRAS:
            with self.subTest(extra=name):
                self.assertIn(name, studio, "Studio lacks the product extra %r" % name)
                self.assertEqual(studio[name].default, product[name].default, name)

    def test_studio_ai_control_and_preview_arrive_on_their_own_sockets(self):
        studio = self.inputs(self.studio)
        for name in SOCKETS:
            with self.subTest(socket=name):
                self.assertIn(name, studio, "Studio lacks the %r socket" % name)
                self.assertTrue(studio[name].optional, name)

    def test_product_schema_is_unchanged(self):
        names = [item.id for item in self.product.GET_SCHEMA().inputs]
        self.assertNotIn("contexts", names)
        self.assertEqual(names[-10:], ["resize_method", "ref_image_size", "resume", "resume_delete_done", "control_strength",
                                       "control_start", "control_end", "models", "output", "spectrum"])

    def test_old_saves_keep_loading_new_widgets_are_appended(self):
        names = [item.id for item in self.studio.GET_SCHEMA().inputs]
        self.assertEqual(names.index("duration_seconds"), names.index("shift_audio") + 1)
        self.assertEqual(names[-3:], list(SOCKETS))

    def values(self, **extra):
        values = dict(clip=["clip", 0], vae=["video_vae", 0], audio_vae=["audio_vae", 0],
                      model_fl2va=["fl2va", 0], model_ref2va=["ref2va", 0], seed=123, steps=8,
                      sampler_name="res_multistep", scheduler="simple", shift_video=12.0, shift_audio=3.0,
                      idea="@Alice walks.", project_seconds=20.0, shots="", chunk_seconds=10.0,
                      width=384, height=576, mode="auto", continuity="video_audio", pin_frames="22",
                      soundscape="wind", music="N/A")
        values.update(extra)
        return values

    def by_type(self, output, kind):
        return [(key, value) for key, value in output.expand.items() if value["class_type"] == kind]

    def test_duration_socket_overrides_the_length_widget_like_the_product(self):
        out = self.studio.execute(**self.values(duration_seconds=30.0))
        plan = self.by_type(out, "NGDirector")[0][1]
        self.assertEqual(plan["inputs"]["project_seconds"], 30.0)
        out = self.studio.execute(**self.values(duration_seconds=0.0))
        plan = self.by_type(out, "NGDirector")[0][1]
        self.assertEqual(plan["inputs"]["project_seconds"], 20.0)
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            self.studio.execute(**self.values(duration_seconds=-5.0))

    def test_pictures_output_expands_the_stills_loop(self):
        out = self.studio.execute(**self.values(output="images"))
        kinds = {value["class_type"] for value in out.expand.values()}
        self.assertIn("NGStillCollect", kinds)
        self.assertNotIn("NGCollect", kinds)
        plan = self.by_type(out, "NGDirector")[0][1]
        self.assertEqual(plan["inputs"]["output"], "images")
        self.assertEqual(len(out.args), 6)

    def test_video_output_stays_the_default(self):
        out = self.studio.execute(**self.values())
        kinds = {value["class_type"] for value in out.expand.values()}
        self.assertIn("NGCollect", kinds)
        self.assertNotIn("NGStillCollect", kinds)

    def test_model_patch_and_preview_vae_reach_the_render_graph(self):
        out = self.studio.execute(**self.values(model_patch=["patch", 0], preview_vae=["preview", 0]))
        control = self.by_type(out, "NGControlChunk")[0][1]
        self.assertEqual(control["inputs"]["model_patch"], ["patch", 0])
        preview = self.by_type(out, "NGH3Preview")[0][1]
        self.assertEqual(preview["inputs"]["vae"], ["preview", 0])


if __name__ == "__main__":
    unittest.main()
