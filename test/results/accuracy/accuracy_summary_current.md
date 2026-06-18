# Accuracy Evaluation Summary

- Run root: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy`
- Model endpoint: `Qwen3.6-27B-AWQ4` on `http://127.0.0.1:8001`
- Runtime profile: AWQ4 + DFlash N=8 + fp8 KV cache, greedy decoding where applicable.
- Caveat: MT-Bench entries are generated answers only until scored by a judge model.
- Caveat: HumanEval executes generated Python tests because `--confirm_run_unsafe_code` is required by lm-eval.

| Benchmark | Capability | Setting | Samples | Metrics | Status | Output |
| --- | --- | --- | ---: | --- | --- | --- |
| GSM8K | math word-problem reasoning | 5-shot, greedy generation | 1/1319 | exact_match,strict-match: 1.0000<br>exact_match,flexible-extract: 1.0000 | scored | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/debug_gsm8k_limit1/Qwen3.6-27B-AWQ4/results_2026-06-15T15-23-03.647332.json` |
| GSM8K | math word-problem reasoning | 5-shot, greedy generation | 5/1319 | exact_match,strict-match: 0.4000<br>exact_match,flexible-extract: 0.4000 | scored | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/gsm8k_dflash_n8_limit5/Qwen3.6-27B-AWQ4/results_2026-06-15T14-54-04.563771.json` |
| HellaSwag | commonsense ending selection | 0-shot, multiple-choice loglikelihood | 5/10042 | acc,none: 0.6000<br>acc_norm,none: 0.8000 | scored | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/hellaswag_dflash_n8_limit5/Qwen3.6-27B-AWQ4/results_2026-06-15T15-04-42.370061.json` |
| MT-Bench | multi-turn chat instruction following | greedy chat answer generation; judge score not run | 3 | answers_generated: yes<br>judge_score: pending | answers only | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/chat_eval/20260615-153105_mt_bench_Qwen3.6-27B-AWQ4_answers.jsonl` |

## Interpretation

These public benchmarks cover mathematical reasoning, commonsense reasoning, science QA, code generation, and multi-turn dialogue. They should be reported as accuracy/quality evidence for the optimized deployment, not as proof of strict mathematical equivalence, because this runtime uses AWQ4 weights and fp8 KV cache in addition to DFlash speculative decoding.
