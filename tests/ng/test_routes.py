import json
import os
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch
from _pack import module, comfy_available, stub_prompt_server


@unittest.skipUnless(comfy_available(), "ComfyUI is not importable")
class Routes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        stub_prompt_server()
        self.context = module("ng.nodes.context_routes")
        self.analysis = module("ng.nodes.analysis_routes")
        self.directory = tempfile.TemporaryDirectory()
        self.root = patch.object(self.context, "preset_root", return_value=self.directory.name)
        self.root.start()

    async def asyncTearDown(self):
        self.root.stop()
        self.directory.cleanup()

    async def test_malformed_requests_return_actionable_json(self):
        for payload in (None, [], {"media": "unknown", "widgets": {}, "name": "x"}):
            request = types.SimpleNamespace(json=AsyncMock(return_value=payload))
            for route in (self.context.write_preset, self.analysis.analyze_context):
                with self.subTest(route=route.__name__, payload=payload):
                    response = await route(request)
                    self.assertEqual(response.status, 400)
                    self.assertFalse(json.loads(response.text)["ok"])

    async def test_a_non_dict_body_is_refused_as_json_on_key_and_model_routes(self):
        # A JSON array body has no .get: without the guard the AttributeError escapes as a 500.
        for route in (self.context.list_models, self.context.write_keys):
            request = types.SimpleNamespace(json=AsyncMock(return_value=[]))
            with self.subTest(route=route.__name__):
                response = await route(request)
                self.assertEqual(response.status, 400)
                self.assertFalse(json.loads(response.text)["ok"])

    async def test_analyze_all_route_reports_a_bad_request_as_json(self):
        # The batch route has to answer with the same ok/error shape, otherwise the panel shows a raw exception.
        for payload in (None, [], {"contexts": [], "provider": {"provider": "magic"}}):
            request = types.SimpleNamespace(json=AsyncMock(return_value=payload))
            response = await self.analysis.analyze_every_context(request)
            self.assertEqual(response.status, 400)
            self.assertFalse(json.loads(response.text)["ok"])

    async def test_analyze_all_route_returns_results_and_skipped(self):
        payload = {"provider": {"schema_version": 1, "provider": "custom", "base_url": "", "model": ""},
                   "mode": "manual", "contexts": [{"media": "image", "widgets": {"image": "a.png"}}]}
        response = await self.analysis.analyze_every_context(types.SimpleNamespace(json=AsyncMock(return_value=payload)))
        body = json.loads(response.text)
        self.assertTrue(body["ok"])
        self.assertEqual(body["results"], [])
        self.assertIn("Manual", body["skipped"][0])

    async def test_preset_roundtrip_and_media_mismatch(self):
        payload = {"name": "Camera", "media": "text", "widgets": {"name": "Camera", "text": "Slow camera.", "context_id": "ctx-roundtrip"}}
        response = await self.context.write_preset(types.SimpleNamespace(json=AsyncMock(return_value=payload)))
        self.assertEqual(response.status, 200)
        request = types.SimpleNamespace(query={"name": "Camera", "media": "text"})
        response = await self.context.read_preset(request)
        self.assertEqual(json.loads(response.text)["widgets"]["context_id"], "ctx-roundtrip")
        request.query["media"] = "audio"
        self.assertEqual((await self.context.read_preset(request)).status, 400)

    async def test_media_path_failure_returns_json(self):
        for source in ("../outside.png", 42):
            response = await self.context.media_preview(types.SimpleNamespace(query={"media": "image", "source": source}))
            self.assertEqual(response.status, 400)
            self.assertFalse(json.loads(response.text)["ok"])

    async def test_decoder_and_timing_faults_are_answered_as_json(self):
        # A corrupt-model RuntimeError (whisper.load_model unpickling) or a raw av fault in the
        # waveform decode loop must not escape as a 500 page the card cannot parse.
        with patch.object(self.context.waveform, "envelope", side_effect=RuntimeError("broken stream")):
            response = await self.context.media_waveform(types.SimpleNamespace(query={"source": "song.wav"}))
        self.assertEqual(response.status, 400)
        self.assertFalse(json.loads(response.text)["ok"])
        with patch.object(self.context.lyrics_timing, "time_lyrics", side_effect=RuntimeError("corrupt weights")):
            request = types.SimpleNamespace(json=AsyncMock(return_value={"source": "song.wav", "lyrics": ""}))
            response = await self.context.time_lyrics(request)
        self.assertEqual(response.status, 400)
        self.assertFalse(json.loads(response.text)["ok"])

    async def test_lyrics_route_times_the_mirrored_file_a_linked_card_sends(self):
        # The browser sends JSON only (ng_widgets.js request(): JSON.stringify), so a linked card sends its
        # mirrored file widget as `source` - never the AUDIO payload (a tensor is not JSON-serializable,
        # and the stateless route has no node id to resolve a link with). The route forwards the name untouched.
        seen = {}

        def fake_time(source, text, device, chosen="", engine=None):
            seen["source"] = source
            return {"lrc": "[00:00.00] la", "language": "English", "lines": 1, "heard_words": 0}

        payload = json.loads(json.dumps({"source": "song.wav", "lyrics": "la", "model": "", "linked": True}))
        request = types.SimpleNamespace(json=AsyncMock(return_value=payload))
        queue = patch.object(self.context.PromptServer.instance, "prompt_queue",
                             types.SimpleNamespace(get_tasks_remaining=lambda: 1), create=True)
        with queue, patch.object(self.context.lyrics_timing, "time_lyrics", side_effect=fake_time), \
                patch.object(self.context.comfy.model_management, "soft_empty_cache"):
            response = await self.context.time_lyrics(request)
        body = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(seen["source"], "song.wav")
        self.assertEqual(body["lines"], 1)

    async def test_lyrics_route_names_the_run_a_file_less_linked_sound_needs(self):
        # A linked sound with nothing to mirror (decoded audio) sends an empty source with `linked`: until one
        # run materializes it to input/h3-external/ the route answers with the run to do, not the decode error.
        # An unlinked card with no file keeps the decode error itself.
        def fake_time(source, text, device, chosen="", engine=None):
            if not str(source or "").strip():
                raise ValueError("Choose a source containing audio.")
            return {"lrc": "[00:00.00] la", "language": "English", "lines": 1, "heard_words": 0}

        queue = patch.object(self.context.PromptServer.instance, "prompt_queue",
                             types.SimpleNamespace(get_tasks_remaining=lambda: 1), create=True)
        with queue, patch.object(self.context.lyrics_timing, "time_lyrics", side_effect=fake_time):
            linked = json.loads(json.dumps({"source": "", "lyrics": "la", "model": "", "linked": True}))
            response = await self.context.time_lyrics(types.SimpleNamespace(json=AsyncMock(return_value=linked)))
            self.assertEqual(response.status, 400)
            self.assertIn("run the card once", json.loads(response.text)["error"])
            plain = json.loads(json.dumps({"source": "", "lyrics": "la", "model": ""}))
            response = await self.context.time_lyrics(types.SimpleNamespace(json=AsyncMock(return_value=plain)))
            self.assertEqual(response.status, 400)
            self.assertIn("containing audio", json.loads(response.text)["error"])

    async def test_lyrics_route_keeps_the_file_path_byte_identical(self):
        # Unlinked cards send a file name and no payload: the route passes the name through untouched.
        seen = {}

        def fake_time(source, text, device, chosen="", engine=None):
            seen["source"] = source
            return {"lrc": "[00:00.00] la", "language": "English", "lines": 1, "heard_words": 0}

        payload = {"source": "song.wav", "lyrics": "la", "model": ""}
        request = types.SimpleNamespace(json=AsyncMock(return_value=payload))
        queue = patch.object(self.context.PromptServer.instance, "prompt_queue",
                             types.SimpleNamespace(get_tasks_remaining=lambda: 1), create=True)
        with queue, patch.object(self.context.lyrics_timing, "time_lyrics", side_effect=fake_time), \
                patch.object(self.context.comfy.model_management, "soft_empty_cache"):
            response = await self.context.time_lyrics(request)
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(response.text)["ok"])
        self.assertEqual(seen["source"], "song.wav")

    async def test_lyrics_engines_route_answers_the_menu_the_dropdown_renders(self):
        # The GET on the timing path answers only what the card's engine dropdown needs: the same
        # registry the POST dispatches through decides availability, so menu and dispatch cannot drift.
        # whisper is always offered, qwen3/ark are greyed out unless their weights or keys exist.
        with patch.dict(os.environ, {"H3_ARK_API_KEY": "", "H3_ARK_APP_ID": "", "H3_ARK_ACCESS_KEY": ""}):
            response = await self.context.lyrics_engines(types.SimpleNamespace())
        body = json.loads(response.text)
        offered = {entry["value"]: entry for entry in body["engines"]}
        self.assertEqual(response.status, 200)
        self.assertTrue(offered["whisper"]["available"])
        self.assertFalse(offered["ark"]["available"])
        self.assertIn("available", offered["qwen3"])
        self.assertEqual(body["default"], "whisper")
