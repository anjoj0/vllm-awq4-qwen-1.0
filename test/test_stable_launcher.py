"""Configuration checks run without PyTorch, Docker, model downloads, or GPUs."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stable_launcher", ROOT / "scripts/start_vllm_stable.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def speculative(command):
    return json.loads(command[command.index("--speculative-config") + 1])


class StableLauncherTests(unittest.TestCase):
    def test_bounded_decode_graphs_validate_single_request_and_eager(self):
        self.assertNotIn("--compilation-config", LAUNCHER.build_command({}))
        command = LAUNCHER.build_command({"VLLM_DECODE_GRAPHS": "1",
            "VLLM_ENFORCE_EAGER": "0", "VLLM_SPEC_TOKENS": "4"})
        config = json.loads(command[command.index("--compilation-config") + 1])
        self.assertEqual(config["mode"], 0)
        self.assertEqual(config["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(config["cudagraph_capture_sizes"], [1, 2, 3, 4, 5])
        for settings in ({"VLLM_DECODE_GRAPHS": "1"},
                         {"VLLM_DECODE_GRAPHS": "1", "VLLM_ENFORCE_EAGER": "0",
                          "VLLM_MAX_NUM_SEQS": "2"}):
            with self.assertRaises(ValueError):
                LAUNCHER.build_command(settings)

    def test_research_precision_profiler_and_dspark_topk_are_opt_in(self):
        default = LAUNCHER.build_command({})
        self.assertNotIn("--dtype", default)
        self.assertNotIn("--profiler-config", default)
        command = LAUNCHER.build_command({"VLLM_SPEC_METHOD": "dspark",
            "VLLM_DTYPE": "bfloat16", "VLLM_DSPARK_DRAFT_TOPK": "128",
            "VLLM_PROFILER_DIR": "/tmp/traces"})
        self.assertEqual(command[command.index("--dtype")+1], "bfloat16")
        self.assertEqual(speculative(command)["dspark_draft_topk"], 128)
        config = json.loads(command[command.index("--profiler-config")+1])
        self.assertTrue(config["ignore_frontend"])
        self.assertEqual(config["torch_profiler_dir"], "/tmp/traces")
        for settings in ({"VLLM_DTYPE": "int8"}, {"VLLM_PROFILER_DIR": "relative"},
                         {"VLLM_DSPARK_DRAFT_TOPK": "128"}):
            with self.assertRaises(ValueError):
                LAUNCHER.build_command(settings)

    def test_optional_cache_block_alignment_is_validated(self):
        self.assertNotIn("--block-size", LAUNCHER.build_command({}))
        command = LAUNCHER.build_command({"VLLM_BLOCK_SIZE": "64"})
        self.assertEqual(command[command.index("--block-size") + 1], "64")
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LAUNCHER.build_command({"VLLM_BLOCK_SIZE": value})

    def test_default_pairs_existing_target_with_dflash(self):
        command = LAUNCHER.build_command({})
        self.assertEqual(command[:3], [
            "vllm", "serve", "cyankiwi/Qwen3.6-27B-AWQ-INT4"])
        self.assertEqual(speculative(command), {
            "method": "dflash", "model": "z-lab/Qwen3.6-27B-DFlash",
            "num_speculative_tokens": 8, "draft_tensor_parallel_size": 1})
        self.assertIn("--enforce-eager", command)

    def test_dspark_selects_its_own_checkpoint(self):
        config = speculative(LAUNCHER.build_command({"VLLM_SPEC_METHOD": "dspark"}))
        self.assertEqual(config["method"], "dspark")
        self.assertEqual(config["model"], "abstract-extraordinary/Qwen3.6-27B-DSpark")

    def test_reduced_vocabulary_checkpoint_and_width_override(self):
        command = LAUNCHER.build_command({
            "VLLM_SPEC_METHOD": "dspark",
            "VLLM_DRAFT_MODEL_ID": "abstract-extraordinary/Qwen3.6-27B-DSpark-49k",
            "VLLM_SPEC_TOKENS": "8", "VLLM_DFLASH_N": "4"})
        self.assertTrue(speculative(command)["model"].endswith("-49k"))
        self.assertEqual(speculative(command)["num_speculative_tokens"], 8)

    def test_target_only_omits_speculative_options(self):
        for settings in ({"VLLM_SPEC_METHOD": "none"}, {"VLLM_DISABLE_DFLASH": "1"}):
            self.assertNotIn("--speculative-config", LAUNCHER.build_command(settings))

    def test_prefix_caching_is_explicit_and_can_be_disabled_for_comparison(self):
        for method in ("none", "dflash", "dspark"):
            with self.subTest(method=method):
                command = LAUNCHER.build_command({"VLLM_SPEC_METHOD": method})
                self.assertIn("--enable-prefix-caching", command)
                self.assertIn("--enable-chunked-prefill", command)
                self.assertIn("--enable-prompt-tokens-details", command)
                self.assertIn("--enable-per-request-metrics", command)
                disabled = LAUNCHER.build_command({
                    "VLLM_SPEC_METHOD": method, "VLLM_ENABLE_PREFIX_CACHING": "0"})
                self.assertIn("--no-enable-prefix-caching", disabled)
                self.assertNotIn("--enable-prefix-caching", disabled)

    def test_legacy_width_and_explicit_graph_switch(self):
        command = LAUNCHER.build_command({
            "VLLM_DFLASH_N": "4", "VLLM_SPEC_TOKENS": "",
            "VLLM_ENFORCE_EAGER": "0"})
        self.assertEqual(speculative(command)["num_speculative_tokens"], 4)
        self.assertNotIn("--enforce-eager", command)

    def test_invalid_method_or_width_fails_before_gpu_start(self):
        for settings in ({"VLLM_SPEC_METHOD": "dsparkk"},
                         {"VLLM_SPEC_TOKENS": "0"},
                         {"VLLM_SPEC_TOKENS": "eight"}):
            with self.assertRaises(ValueError):
                LAUNCHER.build_command(settings)

    def test_wrong_image_version_rejected_before_gpu_import(self):
        with patch.object(LAUNCHER.importlib.metadata, "version", return_value="0.20.0"):
            with self.assertRaisesRegex(RuntimeError, "Expected vLLM 0.32.0"):
                LAUNCHER.check_environment({})


if __name__ == "__main__":
    unittest.main()
