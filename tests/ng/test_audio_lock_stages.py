"""Keep source audio across partial native sampler schedules without replacing free AV values."""
import unittest
from unittest.mock import patch

from _pack import comfy_available, module, load_full_pack

HAVE_COMFY = comfy_available()
if HAVE_COMFY:
    import torch
    from comfy.model_patcher import ModelPatcher
    from comfy.nested_tensor import NestedTensor
    from comfy.patcher_extension import WrapperExecutor, WrappersMP
    from comfy.utils import pack_latents, unpack_latents


@unittest.skipUnless(HAVE_COMFY, "ComfyUI is not importable")
class AudioLockStages(unittest.TestCase):
    def setUp(self):
        self.adapter = module("ng.adapters.audio_locks")
        self.builder = module("ng.adapters.chunk_builder")
        ctx, opts, project, chunks = (module("ng.core." + name) for name in ("context", "context_options", "project", "chunks"))
        lock = ctx.ContextItem("lock", "Backing", "audio", "backing.wav", usage=("lock",), length_s=12,
                               options=opts.ContextOptions(start_s=2, end_s=7))
        result = project.plan_project(ctx.DirectorContext(idea="A walk.", project_seconds=12, items=(lock,)),
                                      5, chunks.ContinuityPlan("video_audio", 22, 24))
        self.jobs = project.jobs(result)
        self.model = ModelPatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
        self.vae = object()
        # what the conditioning build hands to the sampler: the lock's encode and its spans
        self.encoded = torch.arange(178, dtype=torch.float32)[None, None, None, :].expand(1, 32, 2, 178)
        self.spans = [(2.0, 7.0)]
        self.lock = (self.encoded, self.spans)

    def held(self, latent, encoded, spans, fps):
        self.assertIs(encoded, self.encoded)
        self.assertEqual(spans, tuple(self.spans))
        self.assertEqual(fps, 40)
        video, audio = latent["samples"].unbind()
        value = torch.arange(audio.shape[-1], dtype=audio.dtype)[None, None, None, :].expand_as(audio)
        mask = torch.ones_like(audio)
        mask[..., :59] = 0
        return {"samples": NestedTensor((video, value)), "noise_mask": NestedTensor((torch.ones_like(video), mask))}

    def test_each_stage_restores_only_the_locked_audio_and_preserves_native_arguments(self):
        job = self.jobs[1]
        video = torch.arange(32 * 24 * 4 * 4, dtype=torch.float32).reshape(1, 24, 32, 4, 4)
        audio = torch.full((1, 32, 2, 178), 777.0)
        packed, shapes = pack_latents((video, audio))
        noise = torch.zeros_like(packed)
        mask = torch.full_like(packed, 0.5)
        callback, sampler = object(), object()
        calls = []
        def native(*args, **kw):
            calls.append((args, kw))
            return args[1] + 10  # partial schedule output retains noise in held positions
        with patch.object(self.adapter, "lock_audio_latent", side_effect=self.held):
            bound = self.adapter.bind_source(self.model, job, self.lock)
            wrappers = bound.get_all_wrappers(WrappersMP.OUTER_SAMPLE)
            executor = WrapperExecutor.new_executor(native, wrappers)
            for sigmas in (torch.tensor([1.0, 0.5]), torch.tensor([0.5, 0.0])):
                packed = executor.execute(noise, packed, sampler, sigmas, mask, callback, True, 123, latent_shapes=shapes)
        for index, (args, kw) in enumerate(calls):
            actual_video, actual_audio = unpack_latents(args[1], shapes)
            video_mask, audio_mask = unpack_latents(args[4], shapes)
            self.assertTrue(torch.equal(actual_video, video + index * 10))
            self.assertTrue(torch.equal(actual_audio[..., :59], torch.arange(59).expand_as(actual_audio[..., :59])))
            self.assertTrue(torch.all(actual_audio[..., 59:] == 777 + index * 10))
            self.assertTrue(torch.all(video_mask == 0.5))
            self.assertTrue(torch.all(audio_mask[..., :59] == 0))
            self.assertTrue(torch.all(audio_mask[..., 59:] == 0.5))
            self.assertIs(args[0], noise)
            self.assertIs(args[2], sampler)
            self.assertIs(args[5], callback)
            self.assertEqual(args[6:], (True, 123))
            self.assertEqual(kw, {"latent_shapes": shapes})
        self.assertEqual(self.model.get_all_wrappers(WrappersMP.OUTER_SAMPLE), [])

    def test_no_lock_keeps_the_original_model_and_an_existing_wrapper_survives(self):
        marker = lambda *args: None
        self.model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "another_extension", marker)
        self.assertIs(self.adapter.bind_source(self.model, self.jobs[2], self.lock), self.model)
        bound = self.adapter.bind_source(self.model, self.jobs[0], self.lock)
        self.assertIsNot(bound, self.model)
        self.assertIn(marker, bound.get_all_wrappers(WrappersMP.OUTER_SAMPLE))
        again = self.adapter.bind_source(bound, self.jobs[1], self.lock)
        self.assertEqual(len(again.get_wrappers(WrappersMP.OUTER_SAMPLE, "ng_audio_lock")), 1)

    def test_source_is_applied_when_a_sampler_omits_the_noise_mask(self):
        video = torch.zeros(1, 24, 32, 4, 4)
        audio = torch.full((1, 32, 2, 178), 99.0)
        packed, shapes = pack_latents((video, audio))
        with patch.object(self.adapter, "lock_audio_latent", side_effect=self.held):
            bound = self.adapter.bind_source(self.model, self.jobs[1], self.lock)
            wrapper = bound.get_wrappers(WrappersMP.OUTER_SAMPLE, "ng_audio_lock")[0]
            args = wrapper(lambda *args, **kw: args, packed, packed, None, torch.tensor([1.0, 0.0]), None, None, False, 0,
                           latent_shapes=shapes)
        vm, am = unpack_latents(args[4], shapes)
        self.assertTrue(torch.all(vm == 1))
        self.assertTrue(torch.all(am[..., :59] == 0))
        self.assertTrue(torch.all(am[..., 59:] == 1))

    def test_the_restore_reuses_the_parts_encode_instead_of_encoding_at_sampling_time(self):
        # The 2026-11-06 multi-part freeze: the restore derived its own held audio at every sampler call,
        # decoding the source file and running the audio VAE while the diffusion model streamed its weights back in.
        # The wrapper must restore from the encode the conditioning build carried in, and nothing else.
        video = torch.zeros(1, 24, 32, 4, 4)
        audio = torch.full((1, 32, 2, 178), 99.0)
        packed, shapes = pack_latents((video, audio))
        self.assertNotIn("audio_vae", self.adapter.AudioLockSource.__dataclass_fields__)
        with patch.object(self.builder, "_encode_locked_audio",
                          side_effect=AssertionError("the lock must be encoded once, where the part is built")):
            bound = self.adapter.bind_source(self.model, self.jobs[1], self.lock)
            wrapper = bound.get_wrappers(WrappersMP.OUTER_SAMPLE, "ng_audio_lock")[0]
            args = wrapper(lambda *a, **k: a, packed, packed, None, torch.tensor([1.0, 0.0]), None, None, False, 0,
                           latent_shapes=shapes)
        _, restored = unpack_latents(args[1], shapes)
        self.assertTrue(torch.equal(restored[..., 80:], self.encoded[..., 80:]))
        self.assertTrue(torch.all(restored[..., :80] == 99.0))

    def test_conditioning_node_binds_the_current_chunk_to_its_returned_model(self):
        pack = load_full_pack()
        node = pack.NODE_CLASS_MAPPINGS["NGChunkConditioning"]
        import sys
        owner = sys.modules["mmxd_full_pack.ng.nodes.chunk_node"]
        built = self.builder.BuiltChunk([], {}, 0, (), self.lock)
        with patch.object(owner.chunk_builder, "build_chunk", return_value=built), patch.object(owner, "core"), \
             patch.object(owner.audio_locks, "bind_source", return_value="bound") as bind:
            out = node.execute(self.jobs[0], object(), object(), audio_vae=self.vae, model_fl2va=self.model, apply_sigma_shift=False)
        self.assertEqual(out.args[0], "bound")
        bind.assert_called_once_with(self.model, self.jobs[0], self.lock)

    def test_dropped_pin_anchors_are_warned_with_chunk_and_frames(self):
        import logging
        pack = load_full_pack()
        node = pack.NODE_CLASS_MAPPINGS["NGChunkConditioning"]
        import sys
        owner = sys.modules["mmxd_full_pack.ng.nodes.chunk_node"]
        dropped = self.builder.BuiltChunk([], {}, 22, (0, 5), None)
        with patch.object(owner.chunk_builder, "build_chunk", return_value=dropped), patch.object(owner, "core"), \
             patch.object(owner.audio_locks, "bind_source", return_value="bound"), \
             self.assertLogs(owner.log, level=logging.WARNING) as logged:
            node.execute(self.jobs[0], object(), object(), audio_vae=self.vae, model_fl2va=self.model, apply_sigma_shift=False)
        message = " ".join(logged.output)
        self.assertIn("chunk 1", message)
        self.assertIn("0", message)
        self.assertIn("5", message)


if __name__ == "__main__":
    unittest.main()
