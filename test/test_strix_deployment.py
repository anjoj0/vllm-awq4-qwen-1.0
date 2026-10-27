"""Offline checks for checkpoint integrity and incomplete API streams."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_strix", ROOT / "scripts/prepare_strix_halo.py")
smoke = load("smoke_strix", ROOT / "test/smoke_strix.py")


def tensor_file(path, name="weight", payload=b"\0" * 4, dtype="F16", shape=None):
    header = {name: {"dtype": dtype, "shape": [2] if shape is None else shape,
                     "data_offsets": [0, len(payload)]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return header, 8 + len(raw)


class CheckpointTests(unittest.TestCase):
    def test_truncated_payload_and_wrong_shape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            tensor_file(path)
            tensors, _ = prepare.read_header(path)
            self.assertEqual(tensors["weight"]["shape"], [2])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated/trailing payload"):
                prepare.read_header(path)
            tensor_file(path, shape=[3])
            with self.assertRaisesRegex(ValueError, "size/shape mismatch"):
                prepare.read_header(path)

    def test_index_must_point_to_the_actual_tensor_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            repo = cache / "hub/models--example--model"
            revision = "a" * 40
            (repo / "refs").mkdir(parents=True)
            (repo / "refs/main").write_text(revision)
            snap = repo / "snapshots" / revision
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            tensor_file(snap / "model.safetensors")
            _, _, _, report = prepare.inspect_checkpoint(cache, "example/model")
            self.assertEqual(report["revision"], revision)
            self.assertEqual(report["container_path"], f"/models/hub/models--example--model/snapshots/{revision}")
            (snap / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"weight": "missing-shard.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "wrong shard"):
                prepare.inspect_checkpoint(cache, "example/model")

    def test_reduced_dspark_uses_offsets_and_rejects_duplicate_or_invalid_ids(self):
        size, vocab, hidden, rank = 49167, 50000, 4, 2
        target = {"text_config": {"hidden_size": hidden, "vocab_size": vocab, "num_hidden_layers": 64}}
        config = {"hidden_size": hidden, "vocab_size": vocab, "num_target_layers": 64,
                  "target_layer_ids": [1, 16, 31, 46, 61], "architectures": ["Qwen3DSparkModel"],
                  "block_size": 8, "layer_types": ["full_attention"] * 5,
                  "rope_parameters": {"partial_rotary_factor": 1.0},
                  "enable_confidence_head": True, "markov_rank": rank, "draft_vocab_size": size}
        tensors = {name: {"shape": shape} for name, shape in {
            "embed_tokens.weight": [vocab, hidden], "lm_head.weight": [size, hidden],
            "markov_head.markov_w1.weight": [vocab, rank], "markov_head.markov_w2.weight": [size, rank],
            "confidence_head.proj.weight": [1, hidden + rank],
        }.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            for last_offset in (0, -(size - 1), vocab):
                with self.subTest(last_offset=last_offset):
                    # A zero offset for every position is an identity mapping.
                    data = b"\0" * (8 * (size - 1)) + struct.pack("<q", last_offset)
                    header, offset = tensor_file(path, "d2t", data, "I64", [size])
                    tensors.update(header)
                    if last_offset == 0:
                        prepare.validate_draft("dspark-49k", config, tensors, {"d2t": (path, offset)}, target)
                    else:
                        with self.assertRaisesRegex(ValueError, "duplicate/out-of-range"):
                            prepare.validate_draft("dspark-49k", config, tensors, {"d2t": (path, offset)}, target)


class Stream(io.BytesIO):
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}


class FakeClient:
    model = "Qwen3.6-27B-AWQ4"
    chat = smoke.Client.chat

    def __init__(self, frames):
        self.frames = frames

    def request(self, path, payload):
        raw = "".join("data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"
                      for data in self.frames)
        return Stream(raw.encode())


class SmokeStreamTests(unittest.TestCase):
    def test_responses_requires_completed_event_and_usage(self):
        delta = {"type": "response.output_text.delta", "delta": "你好"}
        completed = {"type": "response.completed", "response": {"usage": {"output_tokens": 2}}}
        result = smoke.check_stream(FakeClient([delta, completed]), responses=True)
        self.assertEqual(result["text"], "你好")
        for ending in (["[DONE]"], [], [{"type": "response.incomplete"}],
                       [{"type": "response.completed", "response": {}}]):
            with self.subTest(ending=ending), self.assertRaises(ValueError):
                smoke.check_stream(FakeClient([delta, *ending]), responses=True)

    def test_chat_requires_finish_then_done_and_accepts_usage_after_finish(self):
        frames = [{"choices": [{"delta": {"content": "你好"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"completion_tokens": 2}}, "[DONE]"]
        self.assertEqual(smoke.check_stream(FakeClient(frames))["usage"]["completion_tokens"], 2)
        with self.assertRaisesRegex(ValueError, "Incomplete/empty"):
            smoke.check_stream(FakeClient(frames[:-1]))


if __name__ == "__main__":
    unittest.main()
