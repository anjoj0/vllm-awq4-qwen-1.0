"""Validate the opt-in Qwen27B attention against FP32 softmax and upstream.

Run inside the pinned ROCm image, with VLLM_QWEN27B_PREFILL_ATTN=1.
Includes chunk boundaries, a 32K context, and fallback binary comparisons.
"""
import json
import statistics

import torch
import triton
from vllm.v1.attention.ops import triton_unified_attention as attention


class Capture:
    def __init__(self, kernel):
        self.kernel = kernel
        self.last = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def call(*args, **kwargs):
            self.last = kwargs
            self.compiled = launch(*args, **kwargs)
            return self.compiled

        return call


def elapsed(fn):
    times = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def run_case(m, context, *, dtype=torch.float16, block=832, heads=24,
             kv_heads=4, dim=256, window=(-1, -1), strided=False,
             verify=False, scratch_rows=None, segments=16):
    length = m + context
    blocks = triton.cdiv(length, block)
    q = torch.randn(m, heads * (2 if strided else 1), dim, device="cuda", dtype=dtype)[:, :heads]
    k = torch.randn(blocks, kv_heads, block, dim, device="cuda", dtype=dtype).permute(0, 2, 1, 3)
    v = torch.randn_like(k)
    table = torch.randperm(blocks, device="cuda").to(torch.int32)[None, :]
    out, baseline = torch.empty_like(q), torch.empty_like(q)
    args = dict(q=q, k=k, v=v, cu_seqlens_q=torch.tensor([0, m], device="cuda", dtype=torch.int32),
                max_seqlen_q=m, seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
                max_seqlen_k=length, softmax_scale=dim ** -0.5, causal=True, window_size=window,
                block_table=table, softcap=0.0, q_descale=None, k_descale=None, v_descale=None)
    if verify:
        scratch_rows = m if scratch_rows is None else scratch_rows
        args.update(seq_threshold_3D=32, num_par_softmax_segments=segments,
                    softmax_segm_output=torch.empty((scratch_rows, heads, segments, dim), device="cuda"),
                    softmax_segm_max=torch.empty((scratch_rows, heads, segments), device="cuda"),
                    softmax_segm_expsum=torch.empty((scratch_rows, heads, segments), device="cuda"))

    def native():
        attention._AMD_QWEN27B_PREFILL_ATTN = False
        attention._AMD_QWEN27B_VERIFY_ATTN = False
        attention.unified_attention(out=baseline, **args)

    def tuned():
        attention._AMD_QWEN27B_PREFILL_ATTN = True
        attention._AMD_QWEN27B_VERIFY_ATTN = True
        attention.unified_attention(out=out, **args)

    native()
    native_binary = attention.kernel_unified_attention.compiled.hash
    tuned()
    tuned_binary = attention.kernel_unified_attention.compiled.hash
    selected = attention.kernel_unified_attention.last["AMD_MASK_PADDED_QUERY"]
    used_3d = attention.kernel_unified_attention.last["IS_3D"]
    scope = dtype == torch.float16 and block == 832 and heads == 24 and dim == 256 and window == (-1, -1)
    verify_expected = verify and 2 <= m <= 9 and length >= 256 and scratch_rows >= m and segments == 16
    expected = scope and (m >= 256 or verify_expected)
    assert selected == expected, (selected, expected)
    if verify_expected and scope:
        assert used_3d
    assert torch.isfinite(out).all()
    fallback_repeat_delta = None
    if not selected:
        assert native_binary == tuned_binary, "Fallback changed the compiled kernel"
        previous = baseline.clone()
        native()
        fallback_repeat_delta = float((previous.float() - baseline.float()).norm() / baseline.float().norm())
    sample = torch.unique(torch.cat((torch.linspace(0, m - 1, 20, device="cuda").long(),
        torch.tensor([x for x in (1, 2, 20, 21, 22, 41, 42, 43, 831, 832) if x < m], device="cuda", dtype=torch.int64))))
    # Gather each query head's actual KV head, independently of Triton's
    # tile layout and cache traversal. Only sampled query rows need FP32.
    flat_k = k[table[0].long()].reshape(-1, kv_heads, dim)[:length].float()
    flat_v = v[table[0].long()].reshape(-1, kv_heads, dim)[:length].float()
    group = torch.arange(heads, device="cuda") // (heads // kv_heads)
    scores = torch.einsum("qhd,khd->hqk", q[sample].float(), flat_k[:, group]) * (dim ** -0.5)
    positions = torch.arange(length, device="cuda")[None, None, :]
    upper = context + sample[None, :, None]
    valid = positions <= upper
    if window[0] >= 0:
        valid &= positions >= upper - window[0]
    scores.masked_fill_(~valid, float("-inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), flat_v[:, group])
    relative = float((out[sample].float() - reference).norm() / reference.norm())
    tolerance = .003 if dtype == torch.float16 else .015
    assert relative < tolerance, relative
    delta = float((out.float() - baseline.float()).norm() / baseline.float().norm())
    assert delta < tolerance, delta
    before, times, after = elapsed(native), elapsed(tuned), elapsed(native)
    return dict(query_tokens=m, context_tokens=context, dtype=str(dtype), block_size=block,
                heads=heads, kv_heads=kv_heads, head_dim=dim, window=window, strided_q=strided,
                verify=verify, scratch_rows=scratch_rows, segments=segments, used_3d=used_3d,
                selected=selected, reference_relative_l2=relative, native_relative_l2=delta,
                fallback_identical_binary=(native_binary == tuned_binary),
                fallback_native_repeat_relative_l2=fallback_repeat_delta,
                before_native_ms=before, tuned_ms=times, after_native_ms=after,
                speedup=statistics.median(before + after) / statistics.median(times), passed=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    assert attention._AMD_QWEN27B_PREFILL_ATTN, "Set VLLM_QWEN27B_PREFILL_ATTN=1"
    torch.set_num_threads(1)
    torch.manual_seed(20261027)
    attention.kernel_unified_attention = Capture(attention.kernel_unified_attention)
    cases = [(256, 0, {}), (257, 831, {"strided": True}), (2496, 0, {}),
             (1600, 2496, {}), (6656, 0, {}), (1536, 6656, {}), (257, 32511, {}),
             (255, 832, {}), (5, 8192, {}), (1, 16384, {}),
             (257, 832, {"dtype": torch.bfloat16}), (257, 832, {"block": 816}),
             (257, 832, {"window": (2047, 0)}), (257, 832, {"heads": 16, "dim": 128})]
    if args.verify_only:
        cases = [(m, length - m, {"verify": True}) for m in (2, 5, 9)
                 for length in (256, 832, 4096, 8192, 32768)]
        cases += [(5, 507, {"verify": True, "scratch_rows": 4}),
                  (5, 507, {"verify": True, "segments": 8}),
                  (5, 122, {"verify": True}), (10, 4086, {"verify": True}),
                  (5, 8187, {})]
    rows = []
    with torch.inference_mode():
        for m, context, kwargs in cases:
            row = run_case(m, context, **kwargs)
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("RESULT_JSON=" + json.dumps({"scope": "Per-layer random-shape attention validation; not serving throughput", "rows": rows}))
