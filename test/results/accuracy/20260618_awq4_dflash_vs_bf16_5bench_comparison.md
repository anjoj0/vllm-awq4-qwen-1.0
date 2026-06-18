# AWQ4 + DFlash vs BF16 Accuracy Comparison

- BF16 baseline: `Qwen3.6-27B`, no speculative decoding.
- Optimized path: `Qwen3.6-27B-AWQ4 + DFlash N=8 + fp8 KV cache`.
- MT-Bench score is judged by `deepseek-v4-flash`; it is **not** the official GPT-4 MT-Bench score.

## Primary Metrics

| Benchmark | Capability / setting | Metric | BF16 baseline | AWQ4 + DFlash | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| GSM8K | math reasoning; 5-shot greedy exact_match | `exact_match,flexible-extract` | 0.6505 | 0.6725 | +2.20 pp |
| HellaSwag | commonsense; 0-shot loglikelihood acc_norm | `acc_norm,none` | 0.8401 | 0.8302 | -0.99 pp |
| ARC Challenge | science QA; 25-shot loglikelihood acc_norm | `acc_norm,none` | 0.7637 | 0.7628 | -0.09 pp |
| HumanEval | Python code generation; 0-shot greedy pass@1 | `pass@1,create_test` | 0.8537 | 0.8415 | -1.22 pp |
| MT-Bench | multi-turn dialogue; DeepSeek V4 Flash judge, not official GPT-4 judge | `deepseek_judge_mean` | 4.4000 | 4.0312 | -0.37 / 10 |

## Secondary Metrics

| Metric | BF16 baseline | AWQ4 + DFlash | Delta |
| --- | ---: | ---: | ---: |
| GSM8K strict (`exact_match,strict-match`) | 0.6588 | 0.6710 | +1.21 pp |
| HellaSwag raw acc (`acc,none`) | 0.6409 | 0.6342 | -0.67 pp |
| ARC Challenge raw acc (`acc,none`) | 0.7312 | 0.7210 | -1.02 pp |

## MT-Bench DeepSeek Judge By Category

| Category | BF16 mean | AWQ4 + DFlash mean | Delta |
| --- | ---: | ---: | ---: |
| coding | 4.10 | 3.70 | -0.40 |
| extraction | 3.70 | 3.30 | -0.40 |
| humanities | 3.80 | 3.70 | -0.10 |
| math | 5.70 | 5.40 | -0.30 |
| reasoning | 6.55 | 4.80 | -1.75 |
| roleplay | 3.90 | 4.00 | +0.10 |
| stem | 3.80 | 3.80 | +0.00 |
| writing | 3.65 | 3.55 | -0.10 |

## Interpretation

- On the four objective public benchmarks, AWQ4 + DFlash stays close to BF16: GSM8K improves, ARC Challenge is essentially tied on `acc_norm`, while HellaSwag and HumanEval show small drops around one percentage point.
- MT-Bench with DeepSeek judge is lower for AWQ4 + DFlash. This should be interpreted carefully: the existing AWQ MT-Bench answers were generated with visible reasoning leakage and many `finish_reason=length` cases, which DeepSeek explicitly penalizes. This is an answer-format/generation-policy issue, not direct evidence that the core math/code/multiple-choice capability collapsed.
- For the competition report, the strongest defensible claim is that the optimized inference path greatly improves throughput while keeping standard benchmark quality close to BF16, with MT-Bench requiring a cleaner no-reasoning answer-generation pass for fair dialogue-quality comparison.

## Source Files

- BF16 root: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260616-bf16-systemd`
- AWQ root: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/full_20260615-dflash-n8-systemd`
- JSON summary: `/home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0/test/results/accuracy/20260618_awq4_dflash_vs_bf16_5bench_comparison.json`
