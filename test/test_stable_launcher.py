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
