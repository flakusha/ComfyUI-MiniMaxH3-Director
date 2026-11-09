"""Resume after a crash: a finished part comes from disk when the same plan is queued again (adapters/resume.py)."""
import json
import os
import tempfile
import unittest
from unittest import mock

from _pack import comfy_available, module

HAVE_COMFY = comfy_available()
if HAVE_COMFY:
    import torch
    from comfy.nested_tensor import NestedTensor
    resume = module("ng.adapters.resume")
    ctx, chunks, project = module("ng.core.context"), module("ng.core.chunks"), module("ng.core.project")

SETTINGS = {"seed": 7, "steps": 4}


def plan(texts):
    records = [{"id": "s%d" % n, "start": 10 * n, "end": 10 * n + 10, "text": text, "media": []} for n, text in enumerate(texts)]
    document = json.dumps({"format": "h3-director-shots", "schema_version": 1, "shots": records})
    result = project.plan_project(ctx.DirectorContext(shots_text=document, project_seconds=10 * len(texts)), 10,
                                  chunks.ContinuityPlan("video_audio"))
    return project.jobs(result)


@unittest.skipUnless(HAVE_COMFY, "ComfyUI is not importable")
class Keys(unittest.TestCase):
    def test_short_ref_source_does_not_reuse_the_old_guide_only_audio_render(self):
        options = module("ng.core.context_options")
        source = ctx.ContextItem("v","Source","video","source.mp4",usage=("continue",),length_s=1.625,
                                 options=options.ContextOptions(video_audio="reference"))
        result = project.plan_project(ctx.DirectorContext(items=(source,),mode="ref2va",project_seconds=4.25),
                                      6,chunks.ContinuityPlan())
        job = project.jobs(result)[0]
        self.assertEqual(job.chunk.head_frames,39)
        with mock.patch.object(resume, "_file", return_value=[123,456]):
            fresh = resume.part_key(job, SETTINGS)
            with mock.patch.object(resume, "uses_sound_prefix", return_value=False):
                previous = resume.part_key(job, SETTINGS)
        self.assertNotEqual(fresh,previous)
        normal = plan(["One."])[0]
        with mock.patch.object(resume, "uses_sound_prefix", return_value=False):
            previous = resume.part_key(normal,SETTINGS)
        self.assertEqual(resume.part_key(normal,SETTINGS),previous)

    def test_the_models_menu_choice_invalidates_every_part(self):
        # the menu picks the files place pictures are made with: a swap re-renders from the first part, like a seed change
        keys = lambda js, models: [resume.part_key(j, {**SETTINGS, "models": models}) for j in js]
        rounds = plan(["One.", "Two."])
        self.assertEqual(keys(rounds, ""), keys(plan(["One.", "Two."]), ""))
        self.assertTrue(all(x != y for x, y in zip(keys(rounds, ""), keys(rounds, '{"sam3": "other.safetensors"}'))))

    def test_a_changed_part_changes_its_key_and_every_later_one(self):
        keys = lambda js: [resume.part_key(j, SETTINGS) for j in js]
        ka = keys(plan(["One.", "Two.", "Three."]))
        kb = keys(plan(["One.", "Two.", "Three changed."]))
        self.assertEqual(ka[0], kb[0], "a part before the change comes from disk (each part also holds a few frames of the next shot)")
        self.assertNotEqual(ka[2], kb[2])
        kc = keys(plan(["One changed.", "Two.", "Three."]))
        self.assertTrue(all(x != y for x, y in zip(ka, kc)), "every later part is pinned to the changed one")
        self.assertEqual(ka, keys(plan(["One.", "Two.", "Three."])), "the same plan gives the same keys")
        self.assertNotEqual(ka[0], resume.part_key(plan(["One.", "Two.", "Three."])[0], {**SETTINGS, "seed": 8}))

    def test_moving_a_cut_changes_keys_only_at_or_after_the_boundary(self):
        base = {"format": "h3-director-shots", "schema_version": 1,
                "shots": [{"id": "s1", "start": 0, "end": 30, "text": "One.", "media": []}]}

        def keys(cuts):
            base["render_cuts"] = cuts
            result = project.plan_project(ctx.DirectorContext(shots_text=json.dumps(base), project_seconds=30), 10,
                                          chunks.ContinuityPlan("video_audio"))
            return [resume.part_key(j, SETTINGS) for j in project.jobs(result)]

        before = keys([6.0, 11.0])
        after = keys([6.0, 12.0])               # the second cut snaps to a different grid frame
        self.assertEqual(len(before), len(after))
        self.assertEqual(before[0], after[0])   # every part up to the first boundary keeps its key and resumes
        self.assertTrue(all(a != b for a, b in zip(before[1:], after[1:])), (before, after))


@unittest.skipUnless(HAVE_COMFY, "ComfyUI is not importable")
class Disk(unittest.TestCase):
    def test_save_load_and_prune(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(resume.folder_paths, "get_output_directory", return_value=root):
            latent = {"samples": NestedTensor((torch.rand(1, 24, 7, 4, 4), torch.rand(1, 32, 2, 50)))}
            self.assertFalse(resume.has("k1"))
            resume.save("k1", latent)
            self.assertTrue(resume.has("k1"))
            video, audio = resume.load("k1")["samples"].unbind()
            self.assertTrue(torch.equal(video, latent["samples"].unbind()[0]))
            self.assertTrue(torch.equal(audio, latent["samples"].unbind()[1]))
            old = os.path.join(resume.folder(), "old.pt")
            open(old, "wb").close()
            os.utime(old, (0, 0))
            resume.save("k2", latent)
            self.assertFalse(os.path.exists(old), "parts older than %d days are removed" % resume.KEEP_DAYS)
            self.assertFalse([n for n in os.listdir(resume.folder()) if n.endswith(".part")])

    def test_a_corrupt_part_names_its_file_and_the_fix(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(resume.folder_paths, "get_output_directory", return_value=root):
            os.makedirs(resume.folder(), exist_ok=True)
            with open(os.path.join(resume.folder(), "k9.pt"), "wb") as file:
                file.write(b"not a tensor")
            with self.assertRaises(ValueError) as raised:
                resume.load("k9")
            self.assertIn("k9.pt is corrupt", str(raised.exception))
            self.assertIn("delete output/h3-director-resume", str(raised.exception))

    def test_forget_deletes_only_the_parts_of_the_finished_video(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(resume.folder_paths, "get_output_directory", return_value=root):
            latent = {"samples": NestedTensor((torch.rand(1, 24, 2, 4, 4), torch.rand(1, 32, 2, 10)))}
            done, other = plan(["One.", "Two."]), plan(["Another."])
            for job in (*done[:1], *other):  # the video's second part was never kept, which forget accepts
                resume.save(resume.part_key(job, SETTINGS), latent)
            resume.forget(done, SETTINGS)
            self.assertFalse(resume.has(resume.part_key(done[0], SETTINGS)))
            self.assertTrue(resume.has(resume.part_key(other[0], SETTINGS)), "another project's part stays")


if __name__ == "__main__":
    unittest.main()
