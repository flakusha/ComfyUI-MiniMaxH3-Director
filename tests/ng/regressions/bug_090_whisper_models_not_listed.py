"""BUG-090 (user report, 2026-11-05): the Audio card's Whisper menu stays empty although models/whisper
holds a model, and a Whisper file under a different name cannot load.

Two defects behind it:

1. model_files.names() read models/whisper under ComfyUI's own models dir only. The user's whisper
   folder is registered through extra_model_paths.yaml (`whisper: whisper/` under a central
   base_path; models/whisper does not even exist locally), so every file on the extra path stayed
   invisible. The `list: null` marker in js/ng/models.json means "no ComfyUI loader node lists this
   folder", which the code read as "the folder is not registered" - wrong, the yaml registers whisper
   like every other folder. names() now asks folder_paths whenever the folder is registered and keeps
   the models/ read for folders ComfyUI does not know (GUI test 2026-10-19).

2. lyrics_timing resolved the chosen file under models/whisper and loaded it with whisper.load_model,
   which torch.loads a pickle. A safetensors checkpoint - the format a user with a HuggingFace
   openai/whisper-* download has - failed even under the official .pt name (the report's
   small.pt -> whisper_large_v3_fp16.safetensors symlink). The loader now sniffs the file (a .pt name
   may hide safetensors), builds the Whisper from its state dict with dimensions read off the shapes,
   and renames HuggingFace transformers keys to openai-whisper names; anything that does not fit is a
   named error, not a silently wrong model.

Expected (Soll): files from a registered extra path are listed and resolved, a safetensors Whisper
loads under either key family whatever the file is called, and a non-Whisper safetensors file is
refused with a sentence naming the problem. No server, no GPU, no real model file needed.
"""
import importlib.util
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _pack import comfy_available, module

HAVE_COMFY = comfy_available()
HAVE_WHISPER = importlib.util.find_spec("whisper") is not None


@unittest.skipUnless(HAVE_COMFY, "ComfyUI is not importable")
class WhisperRegisteredFolderAndSafetensors(unittest.TestCase):
    def setUp(self):
        self.files = module("ng.adapters.model_files")
        self.lyrics = module("ng.adapters.lyrics_timing")

    def test_a_registered_folder_is_listed_and_resolved(self):
        """extra_model_paths.yaml registers whisper: the menu follows ComfyUI's list and the file
        resolves on the extra path, not under models/ alone."""
        registered = {"whisper": (["/data/Models/whisper"], set())}
        with mock.patch.dict("folder_paths.folder_names_and_paths", registered), \
             mock.patch("folder_paths.get_filename_list", return_value=["small.pt", "whisper_large_v3_fp16.safetensors"]), \
             mock.patch("folder_paths.get_full_path", return_value="/data/Models/whisper/small.pt") as resolve:
            self.assertEqual(self.files.names("whisper"), ["small.pt", "whisper_large_v3_fp16.safetensors"])
            self.assertEqual(self.files.find("whisper"), "small.pt", "the official name still wins the automatic choice")
            self.assertEqual(self.files.find("whisper", "whisper_large_v3_fp16.safetensors"), "whisper_large_v3_fp16.safetensors")
            self.assertEqual(self.lyrics.model_file("small.pt"), "/data/Models/whisper/small.pt")
            resolve.assert_called_once_with("whisper", "small.pt")

    def test_huggingface_keys_are_renamed_exactly(self):
        cases = {
            "model.encoder.conv1.weight": "encoder.conv1.weight",
            "model.encoder.layers.0.self_attn.v_proj.bias": "encoder.blocks.0.attn.value.bias",
            "model.encoder.layers.2.self_attn_layer_norm.weight": "encoder.blocks.2.attn_ln.weight",
            "model.encoder.layers.1.final_layer_norm.bias": "encoder.blocks.1.mlp_ln.bias",
            "model.encoder.layers.1.fc1.bias": "encoder.blocks.1.mlp.0.bias",
            "model.encoder.layers.1.fc2.bias": "encoder.blocks.1.mlp.2.bias",
            "model.encoder.layer_norm.weight": "encoder.ln_post.weight",
            "model.decoder.embed_tokens.weight": "decoder.token_embedding.weight",
            "model.decoder.embed_positions.weight": "decoder.positional_embedding",
            "model.decoder.layers.4.self_attn.q_proj.weight": "decoder.blocks.4.attn.query.weight",
            "model.decoder.layers.4.encoder_attn.k_proj.weight": "decoder.blocks.4.cross_attn.key.weight",
            "model.decoder.layers.0.self_attn_layer_norm.bias": "decoder.blocks.0.attn_ln.bias",
            "model.decoder.layers.0.encoder_attn_layer_norm.bias": "decoder.blocks.0.cross_attn_ln.bias",
            "model.decoder.layers.3.fc1.weight": "decoder.blocks.3.mlp.0.weight",
            "model.decoder.layers.3.fc2.weight": "decoder.blocks.3.mlp.2.weight",
            "model.decoder.layers.3.final_layer_norm.bias": "decoder.blocks.3.mlp_ln.bias",
            "model.decoder.layer_norm.weight": "decoder.ln.weight",
            "proj_out.weight": None,
        }
        for given, want in cases.items():
            self.assertEqual(self.lyrics._openai_key(given), want, given)

    @unittest.skipUnless(HAVE_WHISPER, "openai-whisper is not installed")
    def test_a_safetensors_whisper_loads_under_both_key_families(self):
        import torch
        from safetensors.torch import save_file
        import whisper as whisper_pkg

        dims = whisper_pkg.model.ModelDimensions(80, 1500, 64, 1, 1, 96, 8, 64, 1, 1)  # 1500: HuggingFace files leave the audio sinusoids out, so the built model gets whisper's own window
        state = dict(whisper_pkg.model.Whisper(dims).state_dict())
        attn = {"query": "q_proj", "key": "k_proj", "value": "v_proj", "out": "out_proj"}
        mlp = {"0": "fc1", "2": "fc2"}

        def hf(key):
            key = re.sub(r"^(encoder|decoder)\.blocks\.(\d+)\.attn\.(query|key|value|out)\.",
                         lambda m: "model.%s.layers.%s.self_attn.%s." % (m[1], m[2], attn[m[3]]), key)
            key = re.sub(r"^decoder\.blocks\.(\d+)\.cross_attn\.(query|key|value|out)\.",
                         lambda m: "model.decoder.layers.%s.encoder_attn.%s." % (m[1], attn[m[2]]), key)
            key = re.sub(r"^(encoder|decoder)\.blocks\.(\d+)\.attn_ln\.", r"model.\1.layers.\2.self_attn_layer_norm.", key)
            key = re.sub(r"^decoder\.blocks\.(\d+)\.cross_attn_ln\.", r"model.decoder.layers.\1.encoder_attn_layer_norm.", key)
            key = re.sub(r"^(encoder|decoder)\.blocks\.(\d+)\.mlp\.([02])\.",
                         lambda m: "model.%s.layers.%s.%s." % (m[1], m[2], mlp[m[3]]), key)
            key = re.sub(r"^(encoder|decoder)\.blocks\.(\d+)\.mlp_ln\.", r"model.\1.layers.\2.final_layer_norm.", key)
            key = re.sub(r"^encoder\.conv([12])\.", r"model.encoder.conv\1.", key)
            key = re.sub(r"^encoder\.ln_post\.", "model.encoder.layer_norm.", key)
            key = key.replace("decoder.token_embedding.", "model.decoder.embed_tokens.")
            key = key.replace("decoder.positional_embedding", "model.decoder.embed_positions.weight")
            key = key.replace("decoder.ln.", "model.decoder.layer_norm.")
            return key

        with tempfile.TemporaryDirectory() as root:
            plain = os.path.join(root, "whisper_openai_keys.safetensors")
            save_file(state, plain)
            self.assertEqual(self.lyrics._load_safetensors(plain, "cpu").dims, dims)

            renamed = os.path.join(root, "whisper_huggingface_keys.safetensors")
            save_file({hf(key): tensor for key, tensor in state.items() if key != "encoder.positional_embedding"}, renamed)
            self.assertEqual(self.lyrics._load_safetensors(renamed, "cpu").dims, dims)

            twin = os.path.join(root, "small.pt")
            os.symlink(renamed, twin)
            self.assertEqual(self.lyrics._load_model(twin, "cpu").dims, dims, "the format is sniffed, not read off the name")

            junk = os.path.join(root, "junk.safetensors")
            save_file({"not_a_whisper": torch.zeros(2)}, junk)
            with self.assertRaisesRegex(ValueError, "is not a Whisper checkpoint: no encoder\\.conv1\\.weight"):
                self.lyrics._load_safetensors(junk, "cpu")


if __name__ == "__main__":
    unittest.main()
