# DFlash Correctness Test Protocol

This protocol is separate from public benchmark scoring. GSM8K, HellaSwag,
ARC Challenge, HumanEval, and MT-Bench measure model quality against public
tasks. The checks below validate whether enabling DFlash and the patched
attention path changes greedy target-model behavior.

## What To Prove

| Question | Evidence |
| --- | --- |
| Does DFlash preserve greedy visible outputs for representative prompts? | `verify_dflash_equivalence.py collect/compare` exact text match |
| Does DFlash actually run and accept draft tokens? | `bench_competition.py --logs-file` parsed `SpecDecoding metrics` |
| Does N=8 beat safer N=1/N=4 on useful workloads? | Stage-B `VLLM_DFLASH_N` sweep with wall, TTFT, decode t/s, acceptance |
| Do patched non-causal and unified-attention paths survive long-context verify? | long prompt runs plus logs showing TRITON_ATTN and no request failure |
| Is failure recovery understood? | client-disconnect/stuck-worker note in README; not part of normal demo |

## A. Greedy Equivalence A/B

Collect no-spec target baseline:

```bash
cd /home/xqhpc/data/AI_project/vllm-awq4-qwen-1.0

VLLM_DISABLE_DFLASH=1 sudo -E docker compose up -d --force-recreate
python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=10).status)'

python3 test/verify_dflash_equivalence.py collect \
  --label awq4_nospec \
  --host http://127.0.0.1:8001 \
  --out-dir test/results/dflash_correctness
```

Collect DFlash candidate:

```bash
VLLM_DISABLE_DFLASH=0 VLLM_DFLASH_N=8 sudo -E docker compose up -d --force-recreate
python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=10).status)'

python3 test/verify_dflash_equivalence.py collect \
  --label awq4_dflash_n8 \
  --host http://127.0.0.1:8001 \
  --out-dir test/results/dflash_correctness
```

Compare outputs:

```bash
python3 test/verify_dflash_equivalence.py compare \
  --a test/results/dflash_correctness/<timestamp>_awq4_nospec_outputs.jsonl \
  --b test/results/dflash_correctness/<timestamp>_awq4_dflash_n8_outputs.jsonl \
  --out test/results/dflash_correctness/<timestamp>_nospec_vs_dflash_n8_compare.json
```

Report the exact-match table from the generated Markdown. If a prompt differs,
record the first divergence and rerun with `VLLM_DFLASH_N=1` to distinguish
multi-token speculative behavior from the basic DFlash path.

## B. Acceptance And Speed Evidence

After starting the DFlash container, preserve logs and run the competition
benchmark:

```bash
sudo docker logs vllm-awq4-qwen > /tmp/vllm-awq4-dflash-n8.log

python3 test/bench_competition.py \
  --label dflash_n8_correctness_speed \
  --host http://127.0.0.1:8001 \
  --cases short_decode_128 mid_prefill_2k_decode_128 paper_8kchars_decode_128 paper_32kchars_decode_128 \
  --runs 1 \
  --mode stream \
  --logs-file /tmp/vllm-awq4-dflash-n8.log \
  --out-dir test/results/dflash_correctness
```

The generated Markdown should include:

- `spec_decoding_last.mean_acceptance_length`
- `spec_decoding_last.avg_draft_acceptance_rate_pct`
- `accepted_tps`
- `drafted_tps`
- API wall time / TTFT / stream decode t/s

## C. Recommended Report Wording

Use conservative wording:

> We validated DFlash separately from public benchmark accuracy. Public
> benchmarks measure model quality, while DFlash A/B checks compare greedy
> visible outputs between the same AWQ4 target with and without speculative
> decoding. Additional runtime logs confirm that the drafter accepts tokens
> on real workloads and that the optimized path improves decode throughput.

Avoid claiming full mathematical equivalence for the whole deployment, because
the deployment also uses AWQ4 weights and fp8 KV cache.
