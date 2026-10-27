"""Opt-in gfx1151 tiles for the pinned Qwen27B prefill attention kernel.

The scope is FP16 causal GQA 24:4, head dimension 256, one sequence and
the qualified 832-token cache layout. Other attention modes retain upstream
launch parameters and row masks. This does not change attention semantics.
"""
import ast
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

SOURCE_SHA256 = "a1d13e015969ce0b80a1024ba9e9703bfa50481cd9aa7d8f893f6cc955e657a1"
MARKER = "# AMD_PROJECT_QWEN27B_PREFILL_ATTN_V2"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one attention patch anchor: {old[:80]!r}")
    return source.replace(old, new, 1)


def patch_source(path):
    source = path.read_text()
    if MARKER in source:
        return False
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Attention source differs from the qualified v0.32.0 ROCm file")
    source = replace_once(source, "logger = init_logger(__name__)\n", '''# AMD_PROJECT_QWEN27B_PREFILL_ATTN_V2
import os as _amd_attn_os

_AMD_QWEN27B_PREFILL_ATTN = (
    _amd_attn_os.environ.get("VLLM_QWEN27B_PREFILL_ATTN", "0") == "1"
    and current_platform.is_rocm()
)
_AMD_QWEN27B_VERIFY_ATTN = (
    _amd_attn_os.environ.get("VLLM_QWEN27B_VERIFY_ATTN", "0") == "1"
    and current_platform.is_rocm()
)
if _AMD_QWEN27B_PREFILL_ATTN or _AMD_QWEN27B_VERIFY_ATTN:
    from vllm.platforms.rocm import on_gfx1151 as _amd_on_gfx1151

logger = init_logger(__name__)
''')
    source = replace_once(source, "    MM_PREFIX_CLAMP_SW: tl.constexpr = False,\n", "    MM_PREFIX_CLAMP_SW: tl.constexpr = False,\n    AMD_MASK_PADDED_QUERY: tl.constexpr = False,\n")
    anchor = "    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)\n"
    source = replace_once(source, anchor, anchor + '''    if AMD_MASK_PADDED_QUERY:
        # GQA=6 does not divide a power-of-two tile. Give each output one
        # owning CTA instead of letting the padded rows overlap the next.
        query_mask_0 = query_mask_0 & (offs_m < BLOCK_Q * num_queries_per_kv)
''')
    anchor = "    # Ideally we would launch with kernel with:\n"
    source = replace_once(source, anchor, '''    amd_qwen_attention = (
        (_AMD_QWEN27B_PREFILL_ATTN or _AMD_QWEN27B_VERIFY_ATTN)
        and _amd_on_gfx1151()
        and num_seqs == 1 and num_query_heads == 24 and num_kv_heads == 4
        and head_size == 256 and block_size == 832
        and max_seqlen_k <= 32768
        and q.dtype == k.dtype == v.dtype == out.dtype == torch.float16
        and kv_quant_mode == KVQuantMode.NONE and output_scale is None
        and use_causal and not use_per_seq_causal
        and window_size == (-1, -1) and chunk_lookback == -1
        and not use_mm_prefix and not use_rswa and not use_td
        and not use_alibi_slopes and not use_qq_bias
        and sinks is None and softcap == 0.0
    )
    amd_qwen_prefill = (
        amd_qwen_attention and _AMD_QWEN27B_PREFILL_ATTN
        and 256 <= max_seqlen_q <= 8192
    )
    amd_qwen_verify = (
        amd_qwen_attention and _AMD_QWEN27B_VERIFY_ATTN
        and 2 <= max_seqlen_q <= 9 and max_seqlen_k >= 256
        and num_par_softmax_segments == 16
        and softmax_segm_output is not None
        and softmax_segm_max is not None and softmax_segm_expsum is not None
        and q.shape[0] <= softmax_segm_output.shape[0]
        and q.shape[0] <= softmax_segm_max.shape[0]
        and q.shape[0] <= softmax_segm_expsum.shape[0]
    )
    if amd_qwen_prefill:
        BLOCK_M = 128
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 1
    elif amd_qwen_verify:
        # Reuse the existing 16-way split-K algorithm. The explicit scratch
        # capacity guards above matter: upstream sized it for decode tokens.
        launch_num_warps = 4
        launch_num_stages = 1

''' + anchor)
    source = replace_once(source, "        or max_seqlen_q > 1\n",
                          "        or (max_seqlen_q > 1 and not amd_qwen_verify)\n")
    source = replace_once(source, "        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,\n", "        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,\n        AMD_MASK_PADDED_QUERY=(amd_qwen_prefill or amd_qwen_verify),\n")
    ast.parse(source)
    path.write_text(source)
    return True


if __name__ == "__main__":
    installed = version("vllm")
    if installed.split("+", 1)[0] != "0.32.0":
        raise RuntimeError(f"Expected vLLM 0.32.0, found {installed}")
    root = Path(next(iter(find_spec("vllm").submodule_search_locations)))
    print("Qwen27B prefill attention installed:",
          patch_source(root / "v1/attention/ops/triton_unified_attention.py"))
