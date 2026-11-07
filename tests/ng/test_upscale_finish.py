"""The opt-in 2K finish: one decode-only learned latent upscale per part, off by default and loud when the model is missing."""
import json
import unittest

from _pack import comfy_available, load_full_pack, module
import test_product_nodes

render_sequence = module("ng.adapters.render_graph").render_sequence


@unittest.skipUnless(comfy_available(), "ComfyUI is not importable")
class UpscaleFinish(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = load_full_pack().NODE_CLASS_MAPPINGS
        cls.helper = test_product_nodes.ProductNodes()
        cls.helper.nodes = cls.nodes

    def result(self):
        director = self.nodes["NGH3Director"]
        values = dict(context=self.helper.packet(), idea="A baker opens a shop.", shots="", project_seconds=8.0,
                      chunk_seconds=5.0, width=384, height=576, mode="auto", continuity="video_audio", pin_frames="22",
                      soundscape="Wind", music="N/A", seed=123, steps=8, sampler_name="res_multistep",
                      scheduler="simple", shift_video=12.0, shift_audio=3.0)
        return director.execute(**values).args[0]

    def by_type(self, expand, kind):
        return {key: value for key, value in expand.items() if value["class_type"] == kind}

    def one(self, expand, kind):
        matches = self.by_type(expand, kind)
        self.assertEqual(len(matches), 1, kind)
        return next(iter(matches.items()))

    def canonical(self, graph):
        """Two builds number their GraphBuilder nodes differently; compare with ids renamed to build order."""
        rename = {key: str(index) for index, key in enumerate(graph)}
        out = {}

        def remap(value):
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            return rename.get(value, value) if isinstance(value, str) else value

        for key, node in graph.items():
            out[rename[key]] = {name: remap(value) for name, value in node.items()}
        return out

    def test_off_is_the_previous_graph_byte_for_byte(self):
        """Upscale off (widget default, or a None carried through the settings): not one node, link or input may change."""
        inputs = dict(chunks=["plan", 0], clip=["clip", 0], vae=["v", 0], audio_vae=["a", 0], model_fl2va=["fl2va", 0])
        settings = dict(seed=1, steps=8, sampler_name="res_multistep", scheduler="simple", shift_video=12.0, shift_audio=3.0)
        plain = render_sequence(inputs, settings)[0]
        self.assertNotIn("MiniMaxH3LatentUpscaleWithModel", {node["class_type"] for node in plain.values()})
        self.assertEqual(json.dumps(self.canonical(plain), sort_keys=True),
                         json.dumps(self.canonical(render_sequence(inputs, {**settings, "upscale": None})[0]), sort_keys=True))
        end = self.one(plain, "EndLoop")[1]
        place = self.one(plain, "NGPlaceStill")[0]
        self.assertEqual(end["inputs"]["output_value"], [place, 0])
        self.assertEqual(end["inputs"]["next_iteration_value"], [place, 0])

    def test_on_upscales_each_part_between_place_still_and_collect(self):
        render = self.nodes["NGProjectRender"].execute(sequence=self.result(), upscale_2k=True, upscale_model=["upscaler", 0])
        expand = render.expand
        upscale_id, upscale = self.one(expand, "MiniMaxH3LatentUpscaleWithModel")
        self.assertEqual(upscale["inputs"]["model"], ["upscaler", 0])
        self.assertEqual((upscale["inputs"]["width"], upscale["inputs"]["height"]), (768, 1152), "2x the Director's canvas")
        start_id, start = self.one(expand, "StartLoop")
        place_id, _ = self.one(expand, "NGPlaceStill")
        self.assertEqual(upscale["inputs"]["samples"], [place_id, 0])
        self.assertIn(upscale_id, start["_loop_body"])
        # the loop still pins the base-resolution latent; only the collected decode runs at 2x
        end_id, end = self.one(expand, "EndLoop")
        self.assertEqual(end["inputs"]["next_iteration_value"], [place_id, 0])
        self.assertEqual(end["inputs"]["output_value"], [upscale_id, 0])
        self.assertEqual(self.one(expand, "NGCollect")[1]["inputs"]["latents"], [end_id, 0])
        # resume cache and still placement stay on the raw sampled latent (adapters/resume.py, core/places.py)
        sample_id, _ = self.one(expand, "SamplerCustomAdvanced")
        resume_id, resume = self.one(expand, "NGResumeChunk")
        self.assertEqual(resume["inputs"]["sampled"], [sample_id, 0])
        self.assertEqual(upscale["inputs"]["samples"], [place_id, 0])

    def test_missing_model_is_a_loud_error_not_a_silent_skip(self):
        with self.assertRaisesRegex(ValueError, "latent_upscale_models"):
            self.nodes["NGProjectRender"].execute(sequence=self.result(), upscale_2k=True)
        message = self.nodes["NGProjectRender"].validate_inputs(upscale_2k=True)
        self.assertIn("latent_upscale_models", message)
        self.assertIs(self.nodes["NGProjectRender"].validate_inputs(upscale_2k=True, upscale_model=["upscaler", 0]), True)

    def test_images_output_refuses_the_finish_instead_of_ignoring_it(self):
        values = dict(context=self.helper.packet(), idea="A baker opens a shop.", shots="", project_seconds=8.0,
                      chunk_seconds=5.0, width=384, height=576, mode="auto", continuity="video_audio", pin_frames="22",
                      soundscape="Wind", music="N/A", seed=123, steps=8, sampler_name="res_multistep",
                      scheduler="simple", shift_video=12.0, shift_audio=3.0, output="images")
        result = self.nodes["NGH3Director"].execute(**values).args[0]
        with self.assertRaisesRegex(ValueError, "Pictures"):
            self.nodes["NGProjectRender"].execute(sequence=result, upscale_2k=True, upscale_model=["upscaler", 0])
        message = self.nodes["NGProjectRender"].validate_inputs(sequence=result, upscale_2k=True,
                                                                  upscale_model=["upscaler", 0])
        self.assertIn("Pictures", message)
        self.assertIs(self.nodes["NGProjectRender"].validate_inputs(sequence=self.result(), upscale_2k=True,
                                                                      upscale_model=["upscaler", 0]), True)

    def test_control_clips_without_a_model_patch_are_flagged_before_the_run(self):
        import json
        from dataclasses import replace
        ctx, options, project, chunks = (module("ng.core." + name) for name in ("context", "context_options", "project", "chunks"))
        card = ctx.ContextItem("pose", "pose", "video", "pose.mp4", usage=("control",), length_s=20,
                               options=options.ContextOptions(control_kind="pose"))
        doc = json.dumps(dict(format="h3-director-shots", schema_version=1, shots=[],
                              control=[dict(schema_version=1, id="a", context_id="pose", start=0, end=4, offset=0, lane="control")]))
        planned = project.plan_project(ctx.DirectorContext(idea="A walk.", project_seconds=12, width=64, height=64,
                                                           mode="fl2va", shots_text=doc, items=(card,)), 4,
                                       chunks.ContinuityPlan("video_audio"))
        packets = module("ng.adapters.director_packets")
        inputs = packets.DirectorInputs(["clip", 0], ["vae", 0], ["audio", 0], ["model", 0])
        result = packets.DirectorResult(inputs, planned, packets.RenderSettings())
        with self.assertRaisesRegex(ValueError, "model_patch"):
            self.nodes["NGProjectRender"].execute(sequence=result)
        self.assertIn("model_patch", self.nodes["NGProjectRender"].validate_inputs(sequence=result))
        patched = replace(result, inputs=replace(inputs, model_patch=["patch", 0]))
        self.assertIs(self.nodes["NGProjectRender"].validate_inputs(sequence=patched), True)

    def test_widget_and_model_port_are_appended_and_the_port_is_optional(self):
        inputs = list(self.nodes["NGProjectRender"].GET_SCHEMA().inputs)
        self.assertEqual([item.id for item in inputs[-2:]], ["upscale_model", "upscale_2k"])
        self.assertEqual(inputs[-2].io_type, "LATENT_UPSCALE_MODEL")
        self.assertTrue(inputs[-2].optional)
        self.assertFalse(inputs[-1].default)
        self.assertTrue(inputs[-1].optional)

    def test_toggling_the_finish_reuses_the_resume_cache(self):
        """sampled_with is identical with/without upscale_2k: toggling 2K reuses the resume cache."""
        inputs = dict(chunks=["plan", 0], clip=["clip", 0], vae=["v", 0], audio_vae=["a", 0], model_fl2va=["fl2va", 0])
        settings = dict(seed=1, steps=8, sampler_name="res_multistep", scheduler="simple", shift_video=12.0, shift_audio=3.0)
        resumes = lambda graph: sorted(node["inputs"]["settings"] for node in graph.values()
                                        if node["class_type"] == "NGResumeChunk")
        upscale = dict(model=["upscaler", 0], width=768, height=1152)
        self.assertEqual(resumes(render_sequence(inputs, settings)[0]),
                         resumes(render_sequence(inputs, {**settings, "upscale": upscale})[0]))


if __name__ == "__main__":
    unittest.main()
