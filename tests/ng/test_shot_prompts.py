"""Shot-anchored prompt shares: one shot's own text follows the shot through re-times,
resolve hits only on a matching share basis, and the legacy whole-part fallback stays exact."""
import json
import unittest

from _pack import module

doc = module("ng.core.shot_document")
share = module("ng.core.shot_share")
bridge = module("ng.core.legacy_bridge")
ctx = module("ng.core.context")
project = module("ng.core.project")
chunks = module("ng.core.chunks")
preview = module("ng.core.preview")
NONE = chunks.ContinuityPlan("none")


def record(identity, start=0, end=10, text="Walk."):
    return {"id": identity, "start": start, "end": end, "text": text, "media": [],
            "spoken": [], "locked_fields": [], "draft": None}


def document(shots, prompts=None):
    data = {"format": "h3-director-shots", "schema_version": 1, "shots": shots}
    if prompts is not None:
        data["shot_prompts"] = prompts
    return json.dumps(data)


def plan(text, seconds=20):
    return project.plan_project(ctx.DirectorContext(idea="A walk.", shots_text=text, project_seconds=seconds, mode="fl2va"),
                                10, NONE)


def basis_of(result, chunk_index, shot_id):
    chunk = result.plan.chunks[chunk_index]
    inputs = share.share_inputs(result.context, chunk, result.compiled[chunk_index].mode)
    position = next(n for n, shot in enumerate(chunk.shots) if shot.shot_id == shot_id)
    return share.share_key(inputs[position])


SHOTS = [record("s1", 0, 10, "She walks through the hall, then climbs the stairs."),
         record("s2", 10, 20, "She opens the door and looks out.")]


class ShotPrompts(unittest.TestCase):
    def test_share_key_is_the_part_prompt_key(self):
        self.assertEqual(share.share_key("abc"), module("ng.core.part_prompt").prompt_key("abc"))

    def test_a_matching_share_is_used_and_named_mixed(self):
        base = plan(document(SHOTS))
        basis = basis_of(base, 0, "s1")
        owned = plan(document(SHOTS, [{"shot_id": "s1", "text": "SHE WALKS, OWN.", "by": "user",
                                                 "basis": basis, "chunk": 0}]))
        self.assertEqual(owned.compiled[0].source, "mixed")
        self.assertIn("SHE WALKS, OWN.", owned.compiled[0].prompt)
        self.assertNotIn("SHE WALKS, OWN.", owned.compiled[1].prompt)
        self.assertEqual(owned.compiled[1].source, "compiled")
        self.assertEqual(owned.compiled[0].key, base.compiled[0].key)

    def test_a_stale_basis_is_not_used_and_names_the_shot(self):
        owned = plan(document(SHOTS, [{"shot_id": "s1", "text": "OLD.", "by": "user",
                                                 "basis": "deadbeefcafe", "chunk": 0}]))
        self.assertEqual(owned.compiled[0].source, "compiled")
        self.assertNotIn("OLD.", owned.compiled[0].prompt)
        outdated = {(i.code, i.ref) for i in owned.issues if i.code == "part_prompt_outdated"}
        self.assertIn(("part_prompt_outdated", "Shot 1"), outdated)

    def test_legacy_fallback_needs_this_index_and_this_whole_key(self):
        base = plan(document(SHOTS))
        key = base.compiled[0].key
        legacy = json.loads(document(SHOTS))
        legacy["part_prompts"] = [{"index": 0, "key": key, "text": "LEGACY OWN.", "by": "user"}]
        owned = plan(json.dumps(legacy))
        self.assertEqual(owned.compiled[0].source, "user")
        self.assertEqual(owned.compiled[0].prompt, "LEGACY OWN.")
        self.assertEqual(owned.compiled[0].key, key)
        stale = dict(legacy)
        stale["part_prompts"] = [{"index": 0, "key": "000000000000", "text": "OLD.", "by": "ai"}]
        refused = plan(json.dumps(stale))
        self.assertEqual(refused.compiled[0].source, "compiled")
        self.assertIn("part_prompt_outdated", {i.code for i in refused.issues})

    def test_legacy_fallback_only_covers_its_own_index(self):
        base = plan(document(SHOTS))
        legacy = json.loads(document(SHOTS))
        legacy["part_prompts"] = [{"index": 1, "key": base.compiled[1].key, "text": "OTHER PART.", "by": "user"}]
        owned = plan(json.dumps(legacy))
        self.assertEqual(owned.compiled[0].source, "compiled")
        self.assertNotIn("OTHER PART.", owned.compiled[0].prompt)

    def test_orphan_shot_ids_prune_on_the_write_path_only(self):
        text = document([record("s1")], [{"shot_id": "gone", "text": "X.", "by": "user", "basis": "k", "chunk": 0}])
        pruned = doc.prune_shot_prompts(text)
        self.assertEqual([e["shot_id"] for e in doc.read_shot_prompts(pruned)], [])
        # stale bases are kept: the user may Rewrite them.
        kept = document([record("s1")], [{"shot_id": "s1", "text": "X.", "by": "user", "basis": "stale", "chunk": 0}])
        self.assertEqual(len(doc.read_shot_prompts(doc.prune_shot_prompts(kept))), 1)

    def test_re_time_keeps_the_share_without_misattribution(self):
        moved = [record("s1", 0, 12, "She walks through the hall, then climbs the stairs."),
                 record("s2", 12, 20, "She opens the door and looks out.")]
        probe = plan(document(moved))
        basis = None
        for chunk in probe.plan.chunks:
            for shot in chunk.shots:
                if shot.shot_id == "s2":
                    inputs = share.share_inputs(probe.context, chunk, probe.compiled[chunk.index].mode)
                    position = [s.shot_id for s in chunk.shots].index("s2")
                    basis = share.share_key(inputs[position])
        self.assertTrue(basis)
        owned = plan(document(moved, [{"shot_id": "s2", "text": "S2 SURVIVES RETIME.", "by": "user",
                                                  "basis": basis, "chunk": 0}]))
        users = [c for c in owned.compiled if "S2 SURVIVES RETIME." in c.prompt]
        self.assertTrue(users, "no part shows the surviving share")
        for compiled in users:
            chunk = owned.plan.chunks[compiled.index]
            self.assertIn("s2", [s.shot_id for s in chunk.shots], "a part without shot s2 shows its text")

    def test_preview_shares_recompute_the_same_basis(self):
        base = plan(document(SHOTS))
        basis = basis_of(base, 0, "s1")
        view = preview.plan_preview({"director": {"idea": "A walk.", "shots": document(
            SHOTS, [{"shot_id": "s1", "text": "X.", "by": "user", "basis": basis, "chunk": 0}]),
            "project_seconds": 20, "mode": "fl2va"}})
        self.assertTrue(view["ok"], view)
        first = view["chunks"][0]["shares"]
        self.assertEqual(first, [{"shot_id": "s1", "basis": basis, "source": "user"}])


if __name__ == "__main__":
    unittest.main()

