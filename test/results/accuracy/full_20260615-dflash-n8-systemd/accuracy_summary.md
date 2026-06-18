# Accuracy Evaluation Summary

- Run root: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260615-dflash-n8-systemd`
- Model endpoint: `Qwen3.6-27B-AWQ4` on `http://127.0.0.1:8001`
- Runtime profile: AWQ4 + DFlash N=8 + fp8 KV cache, greedy decoding where applicable.
- Caveat: MT-Bench entries are generated answers only until scored by a judge model.
- Caveat: HumanEval executes generated Python tests because `--confirm_run_unsafe_code` is required by lm-eval.

| Benchmark | Capability | Setting | Samples | Metrics | Status | Output |
| --- | --- | --- | ---: | --- | --- | --- |
| GSM8K | math word-problem reasoning | 5-shot, greedy generation | 1319/1319 | exact_match,strict-match: 0.6710<br>exact_match,flexible-extract: 0.6725 | scored | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260615-dflash-n8-systemd/gsm8k_dflash_n8_full/Qwen3.6-27B-AWQ4/results_2026-06-15T20-07-25.616784.json` |
| HellaSwag | commonsense ending selection | 0-shot, multiple-choice loglikelihood | 10042/10042 | acc,none: 0.6342<br>acc_norm,none: 0.8302 | scored | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260615-dflash-n8-systemd/hellaswag_dflash_n8_full/Qwen3.6-27B-AWQ4/results_2026-06-16T04-18-12.855441.json` |
| ARC Challenge | grade-school science QA | 25-shot, multiple-choice loglikelihood | 1172/1172 | acc,none: 0.7210<br>acc_norm,none: 0.7628 | scored | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260615-dflash-n8-systemd/arc_challenge_dflash_n8_full/Qwen3.6-27B-AWQ4/results_2026-06-16T12-56-44.693993.json` |
| HumanEval | Python code generation | 0-shot, greedy generation, pass@1 | 164/164 | pass@1,create_test: 0.8415 | scored | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260615-dflash-n8-systemd/humaneval_dflash_n8_full/Qwen3.6-27B-AWQ4/results_2026-06-16T15-42-41.632057.json` |
| MT-Bench | multi-turn chat instruction following | greedy chat answer generation; judge score not run | 80 | answers_generated: yes<br>judge_score: pending | answers only | `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260615-dflash-n8-systemd/mt_bench/20260616-125652_mt_bench_Qwen3.6-27B-AWQ4_answers.jsonl` |

## Interpretation

These public benchmarks cover mathematical reasoning, commonsense reasoning, science QA, code generation, and multi-turn dialogue. They should be reported as accuracy/quality evidence for the optimized deployment, not as proof of strict mathematical equivalence, because this runtime uses AWQ4 weights and fp8 KV cache in addition to DFlash speculative decoding.
