# W7900 DFlash vLLM patch

This directory contains the source patch used by the W7900 five-route DFlash
study. It is intentionally kept separate from the project wrapper so the
changes can be reviewed and applied to a clean vLLM checkout.

## Baseline

- vLLM main commit: `63e78ce3652f4f94e9f484f40db71ca4cf019f21`
- Tested image: ROCm 7.14, PyTorch 2.11, vLLM `0.23.1.dev1`
- Tested GPU: 8 x Radeon PRO W7900 (`gfx1100`)
- Patch SHA-256: `5ba09e402cf9198cff6fec542b63453eab464d477a82d7488fa8732af8575935`

The patch incorporates or adapts work from upstream PRs `#47914`, `#48113`,
`#50169`, and `#47131`. It also adds the W7900 small-query attention tuning and
V2-runner D-Cut integration used in this repository.

## Apply

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 63e78ce3652f4f94e9f484f40db71ca4cf019f21
git apply --check ../vllm-main-63e78ce-w7900-dflash-five-routes.patch
git apply ../vllm-main-63e78ce-w7900-dflash-five-routes.patch
```

The patch changes 13 existing files and adds
`tests/v1/spec_decode/test_dcut.py`. Run the hardware-independent regression
tests with:

```bash
pytest -q tests/v1/spec_decode/test_dcut.py
```

## Runtime controls

| Control | Purpose | Production value |
|---|---|---|
| `AMD_GFX1100_SMALL_QUERY_ATTN_TILE` | KV tile for the scoped gfx1100 small-query path | `32` |
| `AMD_GFX1100_SMALL_QUERY_ATTN_WARPS` | Triton warps for that path | `4` |
| `VLLM_DFLASH_FORCE_FULL_ATTN` | Force all five draft layers to full attention for A/B | unset / `0` |
| `VLLM_DFLASH_FULL_LAYER_WINDOW` | Restrict the one full draft layer to a recent window | unset |
| `dflash_dcut` in `--speculative-config` | D-Cut: `0`, ratio in `(0,1]`, or `"auto"` | `0` |

The default patched behavior restores checkpoint-aligned DFlash attention:
four sliding-window layers with a 2048-token window and one full-attention
layer. No environment variable is needed for this behavior.

Example production speculative configuration:

```bash
export AMD_GFX1100_SMALL_QUERY_ATTN_TILE=32
export AMD_GFX1100_SMALL_QUERY_ATTN_WARPS=4

vllm serve /models/Qwen3.6-27B-AWQ \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "method": "dflash",
    "model": "/models/Qwen3.6-27B-DFlash",
    "num_speculative_tokens": 4,
    "draft_tensor_parallel_size": 1,
    "dflash_dcut": 0
  }'
```

Use DFlash only for the measured short/mid-context region. The accompanying
router sends single requests up to 14K prompt tokens to DFlash N=4 and sends
longer or batched requests to a target-only TP=4 service.

## Experimental features kept off

- D-Cut is functionally correct in the V2 runner but was 1% to 2% slower in
  the measured single-request and concurrency-4 workloads.
- Restricting the sole full-attention draft layer to an 8K or 16K recent window
  reduced acceptance enough to make 16K and 32K requests slower.
- Draft TP=4 did not beat draft TP=1.

These paths remain in the patch as reproducible research controls, not as
recommended defaults.
