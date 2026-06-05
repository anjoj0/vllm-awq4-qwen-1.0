"""
Strix Halo (gfx1151) patch bundle for vLLM source builds.

This is the same patch script the sibling `vllm-qwen` repo ships, kept
in sync verbatim. Two reasons it lives here unchanged:

  1. Every patch is gfx1151-driven, not quant-driven. The AWQ-INT4
     model exercises the same RDNA 3.5 code paths as the BF16 model
     (same custom_ops registration, same AITER overrides, same APU
     VRAM clamp), so the patch surface is identical.

  2. AWQ INT4 dispatches through vLLM's compressed-tensors kernel,
     which is itself unaffected by these patches. The DeltaNet linear-
     attention layers are mixed-precision (FP16/BF16 weights kept
     unquantized in the AWQ checkpoint, see README "DeltaNet under
     AWQ" note), so they take the standard linear-attn path the
     compressed-tensors loader already handles.

If a future tool-call / reasoning-parser PR is needed before merging
upstream, add it here as Patch 13/14/15 (cherry-pick of vllm#40783,
#40785, #40787) and rebuild  -  the existing patch numbers stay stable
so cross-repo references don't drift.
"""
import sys
import re
import site
from pathlib import Path

def patch_vllm():
    print("Applying Strix Halo patches to vLLM (ai-notes modernization)...")

    # Patch 1: vllm/platforms/__init__.py (amdsmi monkey patch  -  PROVEN working for 5 months)
    # Comment out real amdsmi imports and replace with pass stubs.
    # The actual amdsmi library doesn't work on Strix Halo APUs in containers.
    p_init = Path('vllm/platforms/__init__.py')
    if p_init.exists():
        txt = p_init.read_text()
        txt = txt.replace('import amdsmi', '# import amdsmi')
        txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)
        txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)
        txt = txt.replace('amdsmi.amdsmi_init()', 'pass')
        txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')
        p_init.write_text(txt)
        print(" -> Patched vllm/platforms/__init__.py (amdsmi disabled, is_rocm forced True)")

    # Patch 1.5: vllm/platforms/rocm.py (MagicMock amdsmi + force gfx1151)
    # Prepend MagicMock so any remaining amdsmi references in rocm.py silently succeed.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        # Add MagicMock header if not already present
        if 'sys.modules["amdsmi"] = MagicMock()' not in txt:
            header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules["amdsmi"] = MagicMock()\n'
            txt = header + txt
        # Force arch detection
        if 'def _get_gcn_arch() -> str:\n    return "gfx1151"' not in txt:
            txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
            txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
            txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        p_rocm_plat.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (MagicMock amdsmi + forced gfx1151)")

    # Patch 2: _aiter_ops.py (Enable AITER on gfx1x, disable FP8 linear)
    p_aiter = Path('vllm/_aiter_ops.py')
    if p_aiter.exists():
        txt = p_aiter.read_text()

        # Ensure on_gfx1x is available globally for our patches below
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms import current_platform",
                              "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x")

        # Extend is_aiter_found_and_supported
        if "or on_gfx1x()" not in txt:
            txt = txt.replace("import on_mi3xx", "import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")

        # Disable FP8 linear
        if "is_linear_fp8_enabled" in txt:
            txt = re.sub(
                r'(def is_linear_fp8_enabled.*?:\n\s+return) (.*?)\n',
                r'\1 False\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER RMSNorm on gfx1x (CUDA Graph hang)
        if "is_rmsnorm_enabled" in txt:
            txt = re.sub(
                r'(def is_rmsnorm_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._RMSNORM_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER Fused MoE on gfx1x (due to hundreds of CDNA-specific dpp_mov assembly conflicts)
        if "is_fused_moe_enabled" in txt:
            txt = re.sub(
                r'(def is_fused_moe_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._FMOE_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        p_aiter.write_text(txt)
        print(" -> Patched vllm/_aiter_ops.py (gfx1x support, FP8 linear empty, MoE disabled)")

    # Patch 3: rocm_aiter_fa.py
    p_fa = Path('vllm/v1/attention/backends/rocm_aiter_fa.py')
    if p_fa.exists():
        txt = p_fa.read_text()
        if "on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms.rocm import on_mi3xx", "from vllm.platforms.rocm import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")
            p_fa.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_aiter_fa.py (gfx1x support)")

    # Patch 3.5: unquantized.py (Hard-block AITER MoE forced override on gfx1x)
    p_unquant = Path('vllm/model_executor/layers/fused_moe/oracle/unquantized.py')
    if p_unquant.exists():
        txt = p_unquant.read_text()
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace(
                'if envs.is_set("VLLM_ROCM_USE_AITER")',
                'from vllm.platforms.rocm import on_gfx1x\n    if envs.is_set("VLLM_ROCM_USE_AITER")'
            )
            txt = txt.replace(
                'if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:',
                'if getattr(on_gfx1x, "__call__", lambda: False)() or not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:'
            )
            p_unquant.write_text(txt)
            print(" -> Patched unquantized.py (Blocked AITER MoE override on gfx1x)")


    # Patch 5: custom_ops RMSNorm block on gfx1x (Full CUDA Graph capture)
    p_rocm = Path('vllm/platforms/rocm.py')
    if p_rocm.exists():
        txt = p_rocm.read_text()

        # Legacy vLLM < 0.19 fallback
        if "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")" in txt:
            txt = txt.replace(
                "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")",
                "if is_aiter_found_and_supported() and not getattr(self, 'on_gfx1x', lambda: False)():\n            custom_ops.append(\"+rms_norm\")"
            )

        # Modern vLLM 0.19+ struct (compilation_config.custom_ops)
        elif "compilation_config.custom_ops.append(\"+rms_norm\")" in txt:
            if "if not getattr(self, \"on_gfx1x\", lambda: False)():" not in txt:
                txt = re.sub(
                    r'(\s+)compilation_config\.custom_ops\.append\("\+rms_norm"\)',
                    r'\1if not getattr(self, "on_gfx1x", lambda: False)():\n\1    compilation_config.custom_ops.append("+rms_norm")',
                    txt
                )

        # Modern vLLM 0.19.2rc1+ IrOpPriorityConfig bypass
        if 'rms_norm = ["aiter"] + default' in txt:
            txt = txt.replace(
                'rms_norm = ["aiter"] + default',
                'rms_norm = ["aiter"] + default if not on_gfx1x() else default'
            )

        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (custom_ops & IrOpPriorityConfig rms_norm bypassed on gfx1x)")

    # Patch 6: vllm/compilation/passes/fusion/rocm_aiter_fusion.py (duplicate pattern bypass)
    p_fusion = Path('vllm/compilation/passes/fusion/rocm_aiter_fusion.py')
    if p_fusion.exists():
        txt = p_fusion.read_text()
        if "skip_duplicates=True" not in txt:
            txt = re.sub(
                r"(pm\.register_replacement\s*\((?:(?!\bpm\.register_replacement\b).)*?)pm_pass(\s*[\),])",
                r"\1pm_pass, skip_duplicates=True\2",
                txt, flags=re.DOTALL
            )
            p_fusion.write_text(txt)
            print(" -> Patched vllm/compilation/passes/fusion/rocm_aiter_fusion.py (skip_duplicates)")

    # Patch 7: Triton backend AttrsDescriptor repr
    for sp in site.getsitepackages():
        triton_compiler = Path(sp) / "triton/backends/compiler.py"
        if triton_compiler.exists():
            txt = triton_compiler.read_text()
            if "def __repr__(self):" not in txt:
                txt = txt.replace(
                    "def to_dict(self):",
                    "def __repr__(self):\n        return f'AttrsDescriptor.from_dict({self.to_dict()!r})'\n\n    def to_dict(self):"
                )
                triton_compiler.write_text(txt)
                print(f" -> Patched {triton_compiler} (AttrsDescriptor repr)")

    # Patch 7: aiter JIT path fix  -  aiter builds .so files into ~/.aiter/jit/
    # but importlib.import_module("aiter.jit.<module>") only looks in the
    # installed package directory. Fix by adding the JIT cache to __path__.
    for sp in site.getsitepackages():
        aiter_jit_init = Path(sp) / "aiter/jit/__init__.py"
        if aiter_jit_init.exists():
            txt = aiter_jit_init.read_text()
            if "# PATCHED: JIT cache path" not in txt:
                jit_path_fix = '''
# PATCHED: JIT cache path for Strix Halo
# aiter's JIT compiles .so modules into ~/.aiter/jit/ but importlib looks
# in the installed package directory. Add the JIT cache to __path__.
import os as _os
_jit_cache = _os.path.join(_os.path.expanduser("~"), ".aiter", "jit")
if _os.path.isdir(_jit_cache) and _jit_cache not in __path__:
    __path__.append(_jit_cache)
'''
                txt += jit_path_fix
                aiter_jit_init.write_text(txt)
                print(f" -> Patched {aiter_jit_init} (JIT cache added to __path__)")

    # Patch 8: flash_attn_interface.py  -  make aiter import soft as safety net.
    # If aiter JIT fails for any reason, flash_attn should still load (TRITON_ATTN works).
    # ROCM_ATTN will also work when aiter JIT succeeds (patch 7 fixes the path).
    hard_import_bare = "from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu"

    def _patch_flash_interface(fa_iface):
        txt = fa_iface.read_text()
        if hard_import_bare not in txt or "except (ImportError" in txt:
            return False
        # Detect indentation of the original import line
        m = re.search(r'^( *)' + re.escape(hard_import_bare), txt, re.MULTILINE)
        if not m:
            return False
        indent = m.group(1)
        original_line = indent + hard_import_bare
        soft_import = (
            f"{indent}try:\n"
            f"{indent}    {hard_import_bare}\n"
            f"{indent}except (ImportError, KeyError, ModuleNotFoundError):\n"
            f"{indent}    flash_attn_gpu = None"
        )
        txt = txt.replace(original_line, soft_import)
        fa_iface.write_text(txt)
        print(f" -> Patched {fa_iface} (aiter import made resilient)")
        return True

    for sp in site.getsitepackages():
        for fa_egg in Path(sp).glob("flash_attn*.egg"):
            fa_iface = fa_egg / "flash_attn/flash_attn_interface.py"
            if fa_iface.exists():
                _patch_flash_interface(fa_iface)
        # Also check non-egg installs
        fa_iface = Path(sp) / "flash_attn/flash_attn_interface.py"
        if fa_iface.exists():
            _patch_flash_interface(fa_iface)

    # Patch 9: Allow Triton MoE kernels on gfx11xx (Strix Halo)
    # vLLM recently capped MXFP4 Triton MoE kernels to < (11, 0) which excludes RDNA3.5 (11.x)
    for p_triton in [
        Path('vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py'),
        Path('vllm/model_executor/layers/fused_moe/oracle/mxfp4.py')
    ]:
        if p_triton.exists():
            txt = p_triton.read_text()
            if "cap.minor) < (11, 0)" in txt:
                txt = txt.replace("cap.minor) < (11, 0)", "cap.minor) < (12, 0)")
            if "capability() < (11, 0)" in txt:
                txt = txt.replace("capability() < (11, 0)", "capability() < (12, 0)")
            p_triton.write_text(txt)
            print(f" -> Patched {p_triton} (Triton MoE on gfx11xx)")


    # Patch 9.5: Clang 23 rejects direct mwaitxintrin.h includes.
    # vLLM HEAD's spinloop.cpp must include x86intrin.h instead.
    p_spinloop = Path('csrc/spinloop.cpp')
    if p_spinloop.exists():
        txt = p_spinloop.read_text()
        if '#include <mwaitxintrin.h>' in txt:
            txt = txt.replace('#include <mwaitxintrin.h>', '#include <x86intrin.h>')
            p_spinloop.write_text(txt)
            print(" -> Patched csrc/spinloop.cpp (mwaitxintrin include via x86intrin)")

    # Patch 10: ROCM-21812 APU VRAM Dynamic Margin Patch
    # Explanation: ROCm nightly builds introduced a 50% APU VRAM clamp to prevent
    # OOM kernel panics on headless hosts. This broke vLLM large model loading.
    # This patch intercepts PyTorch memory bounds and dynamically proxies the
    # real amdgpu hardware GTT limits, minus a strict 8GB OS safety margin.
    # By symmetrically carving the OS margin from the top of the GTT ceiling,
    # vLLM's memory profiler allocates flawlessly while guaranteeing the OS stays alive,
    # regardless of the specific GTT allocation size on the host.
    # Ref: https://github.com/ROCm/rocm-systems/pull/5113
    # TODO: Remove this patch block entirely once PR #5113 merges and is
    # incorporated into the ROCm nightly tarballs used by this toolbox.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        if "_patched_mem_info" not in txt:
            mem_patch = '''
# --- ROCM-21812 VRAM DYNAMIC PATCH ---
import torch
import glob
import os

try:
    _orig_mem_info = torch.cuda.mem_get_info
    _orig_get_dev_prop = torch.cuda.get_device_properties

    class MockCudaDeviceProperties:
        def __init__(self, prop, override_total):
            self._prop = prop
            self.total_memory = override_total
        def __getattr__(self, name):
            return getattr(self._prop, name)
        def __dir__(self):
            return dir(self._prop)

    def _patched_mem_info(device=None):
        free, total = _orig_mem_info(device)
        try:
            # On APUs, ROCm clamps total to 50% limit. We need the real GTT limits.
            if total < 70 * 1024**3:
                drm_cards = glob.glob('/sys/class/drm/card*/device/mem_info_gtt_total')
                if drm_cards:
                    card_dir = os.path.dirname(drm_cards[0])
                    with open(os.path.join(card_dir, 'mem_info_gtt_total'), 'r') as f:
                        gtt_total = int(f.read().strip())
                    with open(os.path.join(card_dir, 'mem_info_gtt_used'), 'r') as f:
                        gtt_used = int(f.read().strip())

                    # Symmetrically carve 8GB off the TOP of the device perfectly.
                    safe_ceiling = gtt_total - (8 * 1024**3)

                    real_total = safe_ceiling
                    real_free = max(0, safe_ceiling - gtt_used)

                    total = max(total, real_total)
                    free = real_free
        except Exception as e:
            pass
        return int(free), int(total)

    def _patched_get_dev_prop(device=None):
        prop = _orig_get_dev_prop(device)
        free, total = _patched_mem_info(device)
        if hasattr(prop, 'total_memory') and prop.total_memory < total:
            return MockCudaDeviceProperties(prop, total)
        return prop

    torch.cuda.mem_get_info = _patched_mem_info
    torch.cuda.get_device_properties = _patched_get_dev_prop
except Exception:
    pass
# ---------------------------
'''
            txt = mem_patch + txt
            p_rocm_plat.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (ROCM-21812 APU VRAM Dynamic Margin)")

    # Patch 11 (local addition): silence hipCtx* deprecation warnings in
    # csrc/cumem_allocator_compat.h. vLLM still uses hipCtxGetCurrent /
    # hipCtxSetCurrent / hipDevicePrimaryCtxRetain for CUDA-compat context
    # management; HIP marked these deprecated but there is no clean
    # replacement for the use case, and upstream vLLM hasn't migrated yet.
    # Suppressing the warning class for that file keeps our build clean.
    p_cumem = Path('csrc/cumem_allocator_compat.h')
    if p_cumem.exists():
        txt = p_cumem.read_text()
        marker = '#pragma clang diagnostic ignored "-Wdeprecated-declarations"'
        if marker not in txt:
            txt = marker + "\n" + txt
            p_cumem.write_text(txt)
            print(" -> Patched csrc/cumem_allocator_compat.h (suppress hipCtx* deprecations)")

    # Patch 12 (local): allow transformers' GGUF parser to accept Qwen 3.5/3.6
    # arch tag. Upstream transformers registers "qwen2", "qwen3", "qwen3_moe"
    # but not "qwen35" (the arch tag Unsloth's Qwen 3.6 GGUFs declare). vLLM
    # has a Qwen3_5ForConditionalGeneration class downstream that handles the
    # actual model correctly; we just need transformers' parser to route the
    # GGUF through as if it were qwen3 so the config loads. Harmless for the
    # AWQ path (no GGUF involved) but kept for parity with the BF16 sibling.
    import site as _site
    for _sp in _site.getsitepackages():
        gguf_utils = Path(_sp) / "transformers/modeling_gguf_pytorch_utils.py"
        if gguf_utils.exists():
            _txt = gguf_utils.read_text()
            _marker = 'elif "minimax-m2" in architecture:'
            _inject = (
                'elif "qwen35" in architecture or "qwen3_5" in architecture:\n'
                '        updated_architecture = "qwen3"\n'
                '    '
            )
            if 'qwen35' not in _txt and _marker in _txt:
                _txt = _txt.replace(_marker, _inject + _marker, 1)
                gguf_utils.write_text(_txt)
                print(f" -> Patched {gguf_utils} (qwen35 -> qwen3 alias for GGUF parser)")

    # Patch 13 (local cherry-pick of vLLM PR #40176): "[ROCm] Support non-causal
    # attention in ROCM_ATTN". Merged to vllm-project/vllm:main on
    # 2026-04-22T03:57Z (merge commit 6d09769700) but NOT cherry-picked into
    # the v0.20.0 release tag (101584af0, cut 2026-04-23). This patch unblocks
    # DFlash speculative decoding on gfx1151 by:
    #   - Adding `RocmAttentionMetadata.causal: bool = True` field
    #   - Threading `causal=common_attn_metadata.causal` through builder.build()
    #   - Declaring `RocmAttentionBackend.supports_non_causal() -> True`
    #   - Adding `causal: bool = True` parameter to chunked_prefill_paged_decode
    #   - Threading the flag into prefix_prefill.context_attention_fwd
    #   - Splitting the Triton _fwd_kernel inner loop with `CAUSAL: tl.constexpr`
    #     to skip the causal mask + extend the K-range to the full padded query
    #     length when CAUSAL=False
    #   - Tightening rocm_aiter_unified_attn (does NOT support non-causal)
    # Diff is 41+ / 13- across 4 files; reference patch saved at
    # .research/vllm-dflash-prs/raw/PR-40176.patch.

    # 13a: rocm_attn.py  -  backend flag + metadata field + builder propagation +
    #      type annotation cleanup + forward() flag pass-through
    p_rocm_attn = Path('vllm/v1/attention/backends/rocm_attn.py')
    if p_rocm_attn.exists():
        txt = p_rocm_attn.read_text()
        applied = False

        # 1. Drop unused FlashAttentionMetadata import
        old_import = "from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata\n"
        if old_import in txt:
            txt = txt.replace(old_import, "")
            applied = True

        # 2. Add causal field to RocmAttentionMetadata (after prefix_scheduler_metadata)
        if "causal: bool = True" not in txt:
            old_field_block = (
                "    scheduler_metadata: torch.Tensor | None = None\n"
                "    prefix_scheduler_metadata: torch.Tensor | None = None\n"
            )
            new_field_block = (
                "    scheduler_metadata: torch.Tensor | None = None\n"
                "    prefix_scheduler_metadata: torch.Tensor | None = None\n"
                "\n"
                "    # DFlash drafting sets this to False via CommonAttentionMetadata.\n"
                "    causal: bool = True\n"
            )
            if old_field_block in txt:
                txt = txt.replace(old_field_block, new_field_block, 1)
                applied = True

        # 3. Builder.build()  -  propagate common_attn_metadata.causal into the dataclass
        if "causal=common_attn_metadata.causal" not in txt:
            old_build_tail = "            prefix_scheduler_metadata=prefix_scheduler_metadata,\n        )\n        return attn_metadata\n"
            new_build_tail = (
                "            prefix_scheduler_metadata=prefix_scheduler_metadata,\n"
                "            causal=common_attn_metadata.causal,\n"
                "        )\n        return attn_metadata\n"
            )
            if old_build_tail in txt:
                txt = txt.replace(old_build_tail, new_build_tail, 1)
                applied = True

        # 4. Backend.supports_non_causal() classmethod returns True (gates DFlash)
        if "def supports_non_causal" not in txt:
            old_sink_block = (
                "        # kernel, which is less efficient than the proper triton backends.\n"
                "        return False\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            new_sink_block = (
                "        # kernel, which is less efficient than the proper triton backends.\n"
                "        return False\n\n"
                "    @classmethod\n"
                "    def supports_non_causal(cls) -> bool:\n"
                "        return True\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            if old_sink_block in txt:
                txt = txt.replace(old_sink_block, new_sink_block, 1)
                applied = True

        # 5. Type-annotation fixups: FlashAttentionMetadata -> RocmAttentionMetadata
        if "attn_metadata: FlashAttentionMetadata" in txt:
            txt = txt.replace(
                "attn_metadata: FlashAttentionMetadata",
                "attn_metadata: RocmAttentionMetadata",
            )
            applied = True

        # 6. forward()  -  pass causal=attn_metadata.causal into chunked_prefill_paged_decode
        if "causal=attn_metadata.causal" not in txt:
            old_call_tail = (
                "            sm_scale=self.scale,\n"
                "            output_scale=output_scale,\n"
                "            sinks=self.sinks,\n"
                "        )\n"
            )
            new_call_tail = (
                "            sm_scale=self.scale,\n"
                "            output_scale=output_scale,\n"
                "            sinks=self.sinks,\n"
                "            causal=attn_metadata.causal,\n"
                "        )\n"
            )
            if old_call_tail in txt:
                txt = txt.replace(old_call_tail, new_call_tail, 1)
                applied = True

        if applied:
            p_rocm_attn.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_attn.py (PR #40176: non-causal support)")

    # 13b: rocm_aiter_unified_attn.py  -  must explicitly opt OUT of non-causal
    p_rocm_aiter_uni = Path('vllm/v1/attention/backends/rocm_aiter_unified_attn.py')
    if p_rocm_aiter_uni.exists():
        txt = p_rocm_aiter_uni.read_text()
        applied = False

        # Switch the metadata import to RocmAttentionMetadata
        old_imports = (
            "from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata\n"
            "from vllm.v1.attention.backends.rocm_attn import (\n"
            "    RocmAttentionBackend,\n"
            "    RocmAttentionImpl,\n"
            "    RocmAttentionMetadataBuilder,\n"
            ")\n"
        )
        new_imports = (
            "from vllm.v1.attention.backends.rocm_attn import (\n"
            "    RocmAttentionBackend,\n"
            "    RocmAttentionImpl,\n"
            "    RocmAttentionMetadata,\n"
            "    RocmAttentionMetadataBuilder,\n"
            ")\n"
        )
        if old_imports in txt:
            txt = txt.replace(old_imports, new_imports, 1)
            applied = True

        # Add explicit supports_non_causal=False (this backend doesn't support it)
        if "def supports_non_causal" not in txt:
            old_sink_block = (
                "    def supports_sink(cls) -> bool:\n"
                "        return True\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            new_sink_block = (
                "    def supports_sink(cls) -> bool:\n"
                "        return True\n\n"
                "    @classmethod\n"
                "    def supports_non_causal(cls) -> bool:\n"
                "        return False\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            if old_sink_block in txt:
                txt = txt.replace(old_sink_block, new_sink_block, 1)
                applied = True

        # Type annotation fixup
        if "attn_metadata: FlashAttentionMetadata" in txt:
            txt = txt.replace(
                "attn_metadata: FlashAttentionMetadata",
                "attn_metadata: RocmAttentionMetadata",
            )
            applied = True

        if applied:
            p_rocm_aiter_uni.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_aiter_unified_attn.py (PR #40176: explicit non-causal=False)")

    # 13c: chunked_prefill_paged_decode.py  -  add causal kwarg + forward to context_attention_fwd
    p_chunked = Path('vllm/v1/attention/ops/chunked_prefill_paged_decode.py')
    if p_chunked.exists():
        txt = p_chunked.read_text()
        applied = False

        # Add causal: bool = True parameter to public function
        old_sig_tail = (
            "    # Optional tensor for sinks\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "):\n"
        )
        new_sig_tail = (
            "    # Optional tensor for sinks\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "    causal: bool = True,\n"
            "):\n"
        )
        if old_sig_tail in txt and "causal: bool = True" not in txt.split("def chunked_prefill_paged_decode")[1].split(":\n", 1)[0]:
            txt = txt.replace(old_sig_tail, new_sig_tail, 1)
            applied = True

        # Forward causal= into context_attention_fwd call
        old_inner_call_tail = (
            "            skip_decode=True,\n"
            "            fp8_out_scale=output_scale,\n"
            "            sinks=sinks,\n"
            "        )\n"
        )
        new_inner_call_tail = (
            "            skip_decode=True,\n"
            "            fp8_out_scale=output_scale,\n"
            "            sinks=sinks,\n"
            "            causal=causal,\n"
            "        )\n"
        )
        if old_inner_call_tail in txt and "sinks=sinks,\n            causal=causal" not in txt:
            txt = txt.replace(old_inner_call_tail, new_inner_call_tail, 1)
            applied = True

        # Add optional ROCm paged-attention fallback diagnostics for long-context profiling.
        if "import json\nimport os\nfrom pathlib import Path\n\nimport torch" not in txt:
            txt = txt.replace(
                "import torch\n",
                "import json\nimport os\nfrom pathlib import Path\n\nimport torch\n",
                1,
            )
            applied = True

        helper_anchor = "logger = init_logger(__name__)\n\n"
        helper_block = (
            "logger = init_logger(__name__)\n\n"
            "_ROCM_PAGED_ATTN_FALLBACK_COUNTS: dict[str, int] = {}\n"
            "_ROCM_PAGED_ATTN_FALLBACK_TOTAL = 0\n\n"
            "def _record_rocm_paged_attention_fallback(**row) -> None:\n"
            "    if os.environ.get(\"AWQ_ROCM_PAGED_ATTN_STATS\", os.environ.get(\"VLLM_ROCM_PAGED_ATTN_STATS\", \"0\")).strip().lower() not in (\"1\", \"true\", \"yes\", \"on\"):\n"
            "        return\n"
            "    global _ROCM_PAGED_ATTN_FALLBACK_TOTAL\n"
            "    key_parts = [f\"{k}={row.get(k)}\" for k in (\n"
            "        \"native_use_custom\", \"is_pow2\", \"block_size\", \"head_size\",\n"
            "        \"gqa_ratio\", \"sliding_window\", \"kv_cache_dtype\", \"has_alibi\", \"has_sinks\")]\n"
            "    key = \"|\".join(key_parts)\n"
            "    _ROCM_PAGED_ATTN_FALLBACK_COUNTS[key] = _ROCM_PAGED_ATTN_FALLBACK_COUNTS.get(key, 0) + 1\n"
            "    _ROCM_PAGED_ATTN_FALLBACK_TOTAL += 1\n"
            "    interval_raw = os.environ.get(\"AWQ_ROCM_PAGED_ATTN_STATS_INTERVAL\", os.environ.get(\"VLLM_ROCM_PAGED_ATTN_STATS_INTERVAL\", \"128\"))\n"
            "    try:\n"
            "        interval = max(1, int(interval_raw))\n"
            "    except ValueError:\n"
            "        interval = 128\n"
            "    if _ROCM_PAGED_ATTN_FALLBACK_TOTAL > 16 and _ROCM_PAGED_ATTN_FALLBACK_TOTAL % interval != 0:\n"
            "        return\n"
            "    payload = {\n"
            "        \"total_calls\": _ROCM_PAGED_ATTN_FALLBACK_TOTAL,\n"
            "        \"last\": row,\n"
            "        \"top\": [\n"
            "            {\"key\": k, \"count\": v}\n"
            "            for k, v in sorted(_ROCM_PAGED_ATTN_FALLBACK_COUNTS.items(), key=lambda item: -item[1])[:32]\n"
            "        ],\n"
            "    }\n"
            "    logger.info(\"AWQ_ROCM_PAGED_ATTN_STATS %s\", json.dumps(payload[\"last\"], separators=(\",\", \":\")))\n"
            "    path = os.environ.get(\"AWQ_ROCM_PAGED_ATTN_STATS_PATH\", os.environ.get(\"VLLM_ROCM_PAGED_ATTN_STATS_PATH\", \"\")).strip()\n"
            "    if path:\n"
            "        try:\n"
            "            out = Path(path)\n"
            "            out.parent.mkdir(parents=True, exist_ok=True)\n"
            "            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + \"\\n\")\n"
            "        except Exception as exc:  # best-effort diagnostics only\n"
            "            logger.warning(\"failed to write AWQ_ROCM_PAGED_ATTN_STATS_PATH=%s: %s\", path, exc)\n\n"
        )
        if "def _record_rocm_paged_attention_fallback" not in txt and helper_anchor in txt:
            txt = txt.replace(helper_anchor, helper_block, 1)
            applied = True

        native_anchor = "    # Triton is only forced when encountering a non-standard block\n"
        if "native_use_custom = use_custom" not in txt and native_anchor in txt:
            txt = txt.replace(native_anchor, "    native_use_custom = use_custom\n" + native_anchor, 1)
            applied = True

        fallback_anchor = (
            "        logger.warning_once(\n"
            "            \"Cannot use ROCm custom paged attention kernel,\"\n"
            "            \" falling back to Triton implementation.\"\n"
            "        )\n"
        )
        fallback_block = (
            "        _record_rocm_paged_attention_fallback(\n"
            "            native_use_custom=bool(native_use_custom),\n"
            "            is_pow2=bool(is_pow2),\n"
            "            block_size=int(block_size),\n"
            "            head_size=int(head_size),\n"
            "            gqa_ratio=int(num_queries_per_kv),\n"
            "            max_seq_len=int(max_seq_len),\n"
            "            sliding_window=str(sliding_window),\n"
            "            kv_cache_dtype=str(kv_cache_dtype),\n"
            "            has_alibi=alibi_slopes is not None,\n"
            "            has_sinks=sinks is not None,\n"
            "            num_seqs=int(num_seqs),\n"
            "            num_query_heads=int(num_query_heads),\n"
            "            num_kv_heads=int(num_kv_heads),\n"
            "            q_dtype=str(query.dtype),\n"
            "            is_block_table_ptr=bool(is_block_table_ptr),\n"
            "        )\n"
        ) + fallback_anchor
        fallback_pos = txt.find(fallback_anchor)
        fallback_window = txt[max(0, fallback_pos - 1600):fallback_pos] if fallback_pos >= 0 else ""
        if "_record_rocm_paged_attention_fallback(" not in fallback_window and fallback_anchor in txt:
            txt = txt.replace(fallback_anchor, fallback_block, 1)
            applied = True

        if applied:
            p_chunked.write_text(txt)
            print(" -> Patched vllm/v1/attention/ops/chunked_prefill_paged_decode.py (PR #40176 + fallback stats)")

    # 13d: prefix_prefill.py  -  Triton _fwd_kernel CAUSAL constexpr + context_attention_fwd causal arg
    p_prefix = Path('vllm/v1/attention/ops/prefix_prefill.py')
    if p_prefix.exists():
        txt = p_prefix.read_text()
        applied = False

        # 1. Add CAUSAL: tl.constexpr to _fwd_kernel signature (before MAX_Q_LEN)
        old_kernel_sig = (
            "    SKIP_DECODE: tl.constexpr,\n"
            "    USE_SINKS: tl.constexpr,\n"
            "    USE_FP8: tl.constexpr,\n"
            "    MAX_Q_LEN: tl.constexpr = 0,\n"
        )
        new_kernel_sig = (
            "    SKIP_DECODE: tl.constexpr,\n"
            "    USE_SINKS: tl.constexpr,\n"
            "    USE_FP8: tl.constexpr,\n"
            "    CAUSAL: tl.constexpr = True,\n"
            "    MAX_Q_LEN: tl.constexpr = 0,\n"
        )
        if "CAUSAL: tl.constexpr" not in txt and old_kernel_sig in txt:
            txt = txt.replace(old_kernel_sig, new_kernel_sig, 1)
            applied = True

        # 2. Replace the inner-loop upper bound to branch on CAUSAL
        old_loop_block = (
            "    # compute query against itself (with causal mask)\n"
            "    for start_n in tl.range(\n"
            "        0,\n"
            "        block_mask * (start_m + 1) * BLOCK_M,\n"
            "        BLOCK_N,\n"
            "        loop_unroll_factor=num_unroll_request,\n"
            "    ):\n"
        )
        new_loop_block = (
            "    # compute query against itself (causal among queries by default;\n"
            "    # CAUSAL=False for bidirectional attention over query tokens, e.g. DFlash.)\n"
            "    if CAUSAL:\n"
            "        key_range_upper = block_mask * (start_m + 1) * BLOCK_M\n"
            "    else:\n"
            "        q_len_pad = (cur_batch_query_len + BLOCK_N - 1) // BLOCK_N * BLOCK_N\n"
            "        key_range_upper = block_mask * q_len_pad\n"
            "\n"
            "    for start_n in tl.range(\n"
            "        0,\n"
            "        key_range_upper,\n"
            "        BLOCK_N,\n"
            "        loop_unroll_factor=num_unroll_request,\n"
            "    ):\n"
        )
        if "key_range_upper" not in txt and old_loop_block in txt:
            txt = txt.replace(old_loop_block, new_loop_block, 1)
            applied = True

        # 3. Replace the qk causal-mask + sliding-window block with conditional logic
        old_mask_block = (
            "        qk *= sm_scale\n"
            "        # apply causal mask\n"
            "        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float(\"-inf\"))\n"
            "        if SLIDING_WINDOW > 0:\n"
            "            qk = tl.where(\n"
            "                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,\n"
            "                qk,\n"
            "                float(\"-inf\"),\n"
            "            )\n"
        )
        new_mask_block = (
            "        qk *= sm_scale\n"
            "\n"
            "        valid_kv = (start_n + offs_n[None, :]) < cur_batch_query_len\n"
            "        if CAUSAL:\n"
            "            attn_mask = valid_kv & (offs_m[:, None] >= (start_n + offs_n[None, :]))\n"
            "        else:\n"
            "            attn_mask = valid_kv\n"
            "        if SLIDING_WINDOW > 0:\n"
            "            attn_mask = attn_mask & (\n"
            "                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW\n"
            "            )\n"
            "        qk = tl.where(attn_mask, qk, float(\"-inf\"))\n"
        )
        if "valid_kv = " not in txt and old_mask_block in txt:
            txt = txt.replace(old_mask_block, new_mask_block, 1)
            applied = True

        # 4. Add causal: bool = True parameter to context_attention_fwd
        old_ctx_sig_tail = (
            "    fp8_out_scale=None,\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "):\n"
        )
        new_ctx_sig_tail = (
            "    fp8_out_scale=None,\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "    causal: bool = True,\n"
            "):\n"
        )
        # The chunked_prefill file might also match this pattern, but in this
        # file it occurs once at context_attention_fwd's signature. Use rfind
        # discipline by checking that we have not already added causal here.
        if old_ctx_sig_tail in txt and txt.count("def context_attention_fwd") == 1:
            # Only add if context_attention_fwd doesn't already have causal:
            ctx_def_start = txt.find("def context_attention_fwd")
            ctx_def_end = txt.find("):\n", ctx_def_start) + 3
            ctx_signature = txt[ctx_def_start:ctx_def_end]
            if "causal: bool" not in ctx_signature:
                txt = txt.replace(old_ctx_sig_tail, new_ctx_sig_tail, 1)
                applied = True

        # 5. Add alibi+non-causal assert
        old_alibi = (
            "    if alibi_slopes is not None:\n"
            "        assert sinks is None, \"Sinks arg is not supported with alibi\"\n"
        )
        new_alibi = (
            "    if alibi_slopes is not None:\n"
            "        assert causal, \"Non-causal prefix attention is not supported with alibi\"\n"
            "        assert sinks is None, \"Sinks arg is not supported with alibi\"\n"
        )
        if old_alibi in txt and "Non-causal prefix attention is not supported with alibi" not in txt:
            txt = txt.replace(old_alibi, new_alibi, 1)
            applied = True

        # 6. Pass CAUSAL=causal into the kernel call
        old_kernel_call_tail = (
            "        num_warps=4,\n"
            "        num_stages=1,\n"
            "        USE_SINKS=sinks is not None,\n"
            "        **extra_kargs,\n"
            "    )\n"
        )
        new_kernel_call_tail = (
            "        num_warps=4,\n"
            "        num_stages=1,\n"
            "        USE_SINKS=sinks is not None,\n"
            "        CAUSAL=causal,\n"
            "        **extra_kargs,\n"
            "    )\n"
        )
        if "CAUSAL=causal" not in txt and old_kernel_call_tail in txt:
            txt = txt.replace(old_kernel_call_tail, new_kernel_call_tail, 1)
            applied = True

        if applied:
            p_prefix.write_text(txt)
            print(" -> Patched vllm/v1/attention/ops/prefix_prefill.py (PR #40176: CAUSAL constexpr in _fwd_kernel + context_attention_fwd)")

    # Patch 14 (local cherry-pick of vLLM PR #40898): "[Spec Decode] Add Sliding
    # Window Attention support to DFlash drafter". OPEN at 2026-04-26 (not yet
    # merged) but author jianc99 (also the DFlash paper author) explicitly
    # recommends installing this PR for vanilla vLLM compatibility with the
    # z-lab/Qwen3.6-27B-DFlash drafter, which has interleaved SWA layers
    # (4x sliding_attention + 1x full_attention per the drafter's config.json).
    # Without this patch:
    #   1. SWA layers in drafter run as full attention -> ~25% lower acceptance
    #      length on long-context inputs (per author's HumanEval bench).
    #   2. target_layer_ids is OFF BY ONE in gpu_model_runner.py (correctness
    #      issue, not just optimization) -> drafter reads wrong target hidden
    #      states and acceptance plummets at any context length.
    # Diff is 156+ / 1- across 5 files (4 production + 1 test). Production-only
    # cherry-pick saved at .research/vllm-dflash-prs/raw/PR-40898.patch.

    # 14a: qwen3_dflash.py  -  multiple structural changes
    p_qwen3_dflash = Path('vllm/model_executor/models/qwen3_dflash.py')
    if p_qwen3_dflash.exists():
        txt = p_qwen3_dflash.read_text()
        applied = False

        # 14a-1: Helper function and frozenset before DFlashQwen3Attention class
        if "_DFLASH_VALID_LAYER_TYPES" not in txt:
            old_anchor = "logger = init_logger(__name__)\n\n\nclass DFlashQwen3Attention"
            new_block = (
                "logger = init_logger(__name__)\n"
                "\n"
                "\n"
                "_DFLASH_VALID_LAYER_TYPES = frozenset({\"full_attention\", \"sliding_attention\"})\n"
                "\n"
                "\n"
                "def _get_dflash_layer_types(config) -> tuple[str, ...]:\n"
                "    layer_types = getattr(config, \"layer_types\", None)\n"
                "    if layer_types is None:\n"
                "        return (\"full_attention\",) * config.num_hidden_layers\n"
                "    if len(layer_types) != config.num_hidden_layers:\n"
                "        raise ValueError(\n"
                "            f\"DFlash layer_types length {len(layer_types)} does not match \"\n"
                "            f\"num_hidden_layers {config.num_hidden_layers}.\"\n"
                "        )\n"
                "    invalid = set(layer_types) - _DFLASH_VALID_LAYER_TYPES\n"
                "    if invalid:\n"
                "        raise ValueError(f\"Invalid DFlash layer_type(s): {sorted(invalid)}.\")\n"
                "    if \"sliding_attention\" in layer_types and not getattr(\n"
                "        config, \"sliding_window\", None\n"
                "    ):\n"
                "        raise ValueError(\n"
                "            \"DFlash sliding_attention layers require `sliding_window` in config.\"\n"
                "        )\n"
                "    return tuple(layer_types)\n"
                "\n"
                "\n"
                "class DFlashQwen3Attention"
            )
            if old_anchor in txt:
                txt = txt.replace(old_anchor, new_block, 1)
                applied = True

        # 14a-2: Add sliding_window param to DFlashQwen3Attention.__init__ signature
        old_attn_sig = (
            "        attention_bias: bool = False,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        prefix: str = \"\",\n"
            "        attn_type: str = AttentionType.DECODER,\n"
            "    ) -> None:\n"
        )
        new_attn_sig = (
            "        attention_bias: bool = False,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        sliding_window: int | None = None,\n"
            "        prefix: str = \"\",\n"
            "        attn_type: str = AttentionType.DECODER,\n"
            "    ) -> None:\n"
        )
        if old_attn_sig in txt:
            # Only one signature matches this exact 6-line tail  -  replace once.
            txt = txt.replace(old_attn_sig, new_attn_sig, 1)
            applied = True

        # 14a-3: Add per_layer_sliding_window to Attention() call + post-call zero-out
        old_attn_call = (
            "            num_kv_heads=self.num_kv_heads,\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            prefix=f\"{prefix}.attn\",\n"
            "            attn_type=attn_type,\n"
            "        )\n"
            "        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)\n"
        )
        new_attn_call = (
            "            num_kv_heads=self.num_kv_heads,\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            per_layer_sliding_window=sliding_window,\n"
            "            prefix=f\"{prefix}.attn\",\n"
            "            attn_type=attn_type,\n"
            "        )\n"
            "        if sliding_window is not None:\n"
            "            # DFlash keeps full KV allocation while using SWA only for compute.\n"
            "            self.attn.sliding_window = None\n"
            "        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)\n"
        )
        if old_attn_call in txt and "per_layer_sliding_window=sliding_window" not in txt:
            txt = txt.replace(old_attn_call, new_attn_call, 1)
            applied = True

        # 14a-4: Add layer_type param + body changes to DFlashQwen3DecoderLayer.__init__
        old_dec_init = (
            "        config: Qwen3Config,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        prefix: str = \"\",\n"
            "    ) -> None:\n"
            "        super().__init__()\n"
            "        self.hidden_size = config.hidden_size\n"
            "        set_default_rope_theta(config, default_theta=1000000)\n"
            "        attn_type = AttentionType.DECODER\n"
            "\n"
            "        self.self_attn = DFlashQwen3Attention(\n"
        )
        new_dec_init = (
            "        config: Qwen3Config,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        layer_type: str = \"full_attention\",\n"
            "        prefix: str = \"\",\n"
            "    ) -> None:\n"
            "        super().__init__()\n"
            "        self.hidden_size = config.hidden_size\n"
            "        self.layer_type = layer_type\n"
            "        set_default_rope_theta(config, default_theta=1000000)\n"
            "        attn_type = AttentionType.DECODER\n"
            "        sliding_window = (\n"
            "            config.sliding_window if layer_type == \"sliding_attention\" else None\n"
            "        )\n"
            "\n"
            "        self.self_attn = DFlashQwen3Attention(\n"
        )
        if old_dec_init in txt and "self.layer_type = layer_type" not in txt:
            txt = txt.replace(old_dec_init, new_dec_init, 1)
            applied = True

        # 14a-5: Pass sliding_window into DFlashQwen3Attention( ) call site
        old_attn_call2 = (
            "            head_dim=getattr(config, \"head_dim\", None),\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            rope_parameters=config.rope_parameters,\n"
        )
        new_attn_call2 = (
            "            head_dim=getattr(config, \"head_dim\", None),\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            sliding_window=sliding_window,\n"
            "            rope_parameters=config.rope_parameters,\n"
        )
        if old_attn_call2 in txt and "sliding_window=sliding_window," not in txt:
            txt = txt.replace(old_attn_call2, new_attn_call2, 1)
            applied = True

        # 14a-6: DFlashQwen3Model.__init__  -  compute layer_types, propagate, build set
        old_layers = (
            "        self.layers = nn.ModuleList(\n"
            "            [\n"
            "                DFlashQwen3DecoderLayer(\n"
            "                    current_vllm_config,\n"
            "                    prefix=maybe_prefix(prefix, f\"layers.{layer_idx + start_layer_id}\"),\n"
            "                    config=self.config,\n"
            "                )\n"
            "                for layer_idx in range(self.config.num_hidden_layers)\n"
            "            ]\n"
            "        )\n"
            "        if self.use_aux_hidden_state:\n"
        )
        new_layers = (
            "        self.layer_types = _get_dflash_layer_types(self.config)\n"
            "        self.layers = nn.ModuleList(\n"
            "            [\n"
            "                DFlashQwen3DecoderLayer(\n"
            "                    current_vllm_config,\n"
            "                    prefix=maybe_prefix(prefix, f\"layers.{layer_idx + start_layer_id}\"),\n"
            "                    config=self.config,\n"
            "                    layer_type=self.layer_types[layer_idx],\n"
            "                )\n"
            "                for layer_idx in range(self.config.num_hidden_layers)\n"
            "            ]\n"
            "        )\n"
            "        self.sliding_attention_layer_names = {\n"
            "            layer.self_attn.attn.layer_name\n"
            "            for layer in self.layers\n"
            "            if layer.layer_type == \"sliding_attention\"\n"
            "        }\n"
            "        if self.use_aux_hidden_state:\n"
        )
        if old_layers in txt and "self.layer_types = _get_dflash_layer_types" not in txt:
            txt = txt.replace(old_layers, new_layers, 1)
            applied = True

        # 14a-7: Add @property sliding_attention_layer_names to DFlashQwen3ForCausalLM
        old_precompute_tail = (
            "        \"\"\"Precompute projected + RoPE'd K/V and write to cache.\"\"\"\n"
            "        self.model.precompute_and_store_context_kv(\n"
            "            context_states, context_positions, context_slot_mapping\n"
            "        )\n"
            "\n"
            "    def combine_hidden_states(\n"
        )
        new_precompute_tail = (
            "        \"\"\"Precompute projected + RoPE'd K/V and write to cache.\"\"\"\n"
            "        self.model.precompute_and_store_context_kv(\n"
            "            context_states, context_positions, context_slot_mapping\n"
            "        )\n"
            "\n"
            "    @property\n"
            "    def sliding_attention_layer_names(self) -> set[str]:\n"
            "        return self.model.sliding_attention_layer_names\n"
            "\n"
            "    def combine_hidden_states(\n"
        )
        if old_precompute_tail in txt and "def sliding_attention_layer_names(self)" not in txt:
            txt = txt.replace(old_precompute_tail, new_precompute_tail, 1)
            applied = True

        # 14a-8: DFlash combine_hidden_states can receive fp32 target hidden
        # states on ROCm while the drafter projection is fp16. Align the input
        # dtype with fc.weight before the projection.
        old_fc = "        result = self.model.fc(hidden_states)\n"
        new_fc = (
            "        fc_weight = self.model.fc.weight\n"
            "        if hidden_states.dtype != fc_weight.dtype:\n"
            "            hidden_states = hidden_states.to(fc_weight.dtype)\n"
            "        result = self.model.fc(hidden_states)\n"
        )
        if old_fc in txt and "fc_weight = self.model.fc.weight" not in txt:
            txt = txt.replace(old_fc, new_fc, 1)
            applied = True

        if applied:
            p_qwen3_dflash.write_text(txt)
            print(" -> Patched vllm/model_executor/models/qwen3_dflash.py (PR #40898: SWA support in DFlash drafter)")

    # 14b: algos.py  -  preserve SWA fields when extracting HF config
    p_algos = Path('vllm/transformers_utils/configs/speculators/algos.py')
    if p_algos.exists():
        txt = p_algos.read_text()
        old_target = (
            "    if config_dict.get(\"target_hidden_size\") is not None:\n"
            "        pre_trained_config[\"target_hidden_size\"] = config_dict[\"target_hidden_size\"]\n"
            "\n"
            "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
        )
        new_target = (
            "    if config_dict.get(\"target_hidden_size\") is not None:\n"
            "        pre_trained_config[\"target_hidden_size\"] = config_dict[\"target_hidden_size\"]\n"
            "    for key in (\n"
            "        \"layer_types\",\n"
            "        \"use_sliding_window\",\n"
            "        \"sliding_window\",\n"
            "        \"max_window_layers\",\n"
            "    ):\n"
            "        if key in config_dict:\n"
            "            pre_trained_config[key] = config_dict[key]\n"
            "\n"
            "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
        )
        if old_target in txt and '"layer_types",' not in txt:
            txt = txt.replace(old_target, new_target, 1)
            p_algos.write_text(txt)
            print(" -> Patched vllm/transformers_utils/configs/speculators/algos.py (PR #40898: preserve SWA config)")

    # 14c: dflash.py  -  SWA branch in build_per_group_and_layer_attn_metadata
    p_dflash_proposer = Path('vllm/v1/spec_decode/dflash.py')
    if p_dflash_proposer.exists():
        txt = p_dflash_proposer.read_text()
        old_block = (
            "        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(\n"
            "            cad, draft_index\n"
            "        )\n"
            "        for layer_name, attn_metadata in per_layer.items():\n"
            "            assert getattr(attn_metadata, \"causal\", None) is False, (\n"
            "                f\"Attention metadata for layer {layer_name} does not have\"\n"
            "                \" non-causal support, which is required for DFlash.\"\n"
            "                \" Consider using a different attention backend, such as FlashAttention.\"\n"
            "            )\n"
            "        return per_group, per_layer\n"
        )
        new_block = (
            "        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(\n"
            "            cad, draft_index\n"
            "        )\n"
            "        sliding_layer_names = getattr(self.model, \"sliding_attention_layer_names\", set())\n"
            "        if sliding_layer_names:\n"
            "            causal_cad = cad.replace(causal=True)\n"
            "            for attn_group in self.draft_attn_groups:\n"
            "                causal_layers = sliding_layer_names & set(attn_group.layer_names)\n"
            "                if not causal_layers:\n"
            "                    continue\n"
            "                attn_metadata = attn_group.get_metadata_builder().build_for_drafting(\n"
            "                    common_attn_metadata=causal_cad, draft_index=draft_index\n"
            "                )\n"
            "                for layer_name in causal_layers:\n"
            "                    per_layer[layer_name] = attn_metadata\n"
            "\n"
            "        for layer_name, attn_metadata in per_layer.items():\n"
            "            if layer_name in sliding_layer_names:\n"
            "                assert getattr(attn_metadata, \"causal\", None) is True, (\n"
            "                    f\"Attention metadata for sliding layer {layer_name} does not have\"\n"
            "                    \" causal support, which is required for DFlash SWA.\"\n"
            "                )\n"
            "                continue\n"
            "            assert getattr(attn_metadata, \"causal\", None) is False, (\n"
            "                f\"Attention metadata for layer {layer_name} does not have\"\n"
            "                \" non-causal support, which is required for DFlash.\"\n"
            "                \" Consider using a different attention backend, such as FlashAttention.\"\n"
            "            )\n"
            "        return per_group, per_layer\n"
        )
        if old_block in txt and "sliding_layer_names" not in txt:
            txt = txt.replace(old_block, new_block, 1)
            p_dflash_proposer.write_text(txt)
            print(" -> Patched vllm/v1/spec_decode/dflash.py (PR #40898: SWA branch + SWA causal assertion)")

    # 14d: gpu_model_runner.py  -  fix target_layer_ids +1 shift for dflash method
    p_gmr = Path('vllm/v1/worker/gpu_model_runner.py')
    if p_gmr.exists():
        txt = p_gmr.read_text()
        old_block = (
            "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
            "\n"
            "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
            "        if not layer_ids:\n"
            "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
            "            if dflash_config and isinstance(dflash_config, dict):\n"
            "                layer_ids = dflash_config.get(\"target_layer_ids\")\n"
            "\n"
            "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
            "            return tuple(layer_ids)\n"
            "\n"
            "        return None\n"
        )
        new_block = (
            "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
            "\n"
            "        is_dflash = self.speculative_config.method == \"dflash\"\n"
            "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
            "        if is_dflash or not layer_ids:\n"
            "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
            "            if dflash_config and isinstance(dflash_config, dict):\n"
            "                layer_ids = dflash_config.get(\"target_layer_ids\")\n"
            "\n"
            "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
            "            if is_dflash:\n"
            "                return tuple(layer_id + 1 for layer_id in layer_ids)\n"
            "            return tuple(layer_ids)\n"
            "\n"
            "        return None\n"
        )
        if old_block in txt and "is_dflash = self.speculative_config.method" not in txt:
            txt = txt.replace(old_block, new_block, 1)
            p_gmr.write_text(txt)
            print(" -> Patched vllm/v1/worker/gpu_model_runner.py (PR #40898: target_layer_ids +1 shift fix for dflash)")

    # Patch 15 (local): thread chat_template_kwargs through /v1/responses.
    #
    # Without this, ResponsesRequest.to_chat_params() builds chat_template_kwargs
    # from a hardcoded dict and never reads the request body's
    # chat_template_kwargs field. Effect on Qwen3.6: clients that send
    # `chat_template_kwargs: {"enable_thinking": false}` get reasoning anyway,
    # while the same kwarg works on /v1/chat/completions (different code path).
    # The chat template ITSELF supports enable_thinking - this gap is purely
    # in vLLM's request-to-template wiring on the responses endpoint.
    #
    # Fix is two changes:
    #   15a: add a chat_template_kwargs field to the ResponsesRequest model
    #   15b: pass it as `defaults` to merge_kwargs() so user-supplied kwargs
    #        live alongside vLLM's hardcoded add_generation_prompt etc.
    #        (vLLM's overrides still win for keys it controls).
    #
    # Worth filing upstream as a vLLM PR; the gap looks accidental.
    p_responses_proto = Path('vllm/entrypoints/openai/responses/protocol.py')
    if p_responses_proto.exists():
        txt = p_responses_proto.read_text()

        # 15a: add chat_template_kwargs field, sandwiched between `user` (last
        # OpenAI-spec field) and `skip_special_tokens` (first vLLM extension).
        field_anchor = "    user: str | None = None\n    skip_special_tokens: bool = True\n"
        field_replacement = (
            "    user: str | None = None\n"
            "    chat_template_kwargs: dict[str, Any] | None = None\n"
            "    skip_special_tokens: bool = True\n"
        )
        if "chat_template_kwargs: dict[str, Any] | None = None" not in txt and field_anchor in txt:
            txt = txt.replace(field_anchor, field_replacement, 1)
            print(" -> Patched protocol.py (15a: ResponsesRequest gains chat_template_kwargs field)")

        # 15b: in to_chat_params(), feed the user kwargs into merge_kwargs as
        # the `defaults` argument. The hardcoded dict stays as `overrides` so
        # vLLM-managed keys (add_generation_prompt, continue_final_message,
        # reasoning_effort) keep precedence, while user-supplied keys
        # (enable_thinking, etc.) flow through to the chat template renderer.
        # Indents: the call sits inside `return ChatParams(` so it's 12 spaces
        # for the kwarg line and 16 spaces for the inner positional args.
        merge_anchor = (
            "            chat_template_kwargs=merge_kwargs(  # To remove unset values\n"
            "                {},\n"
            "                dict(\n"
            "                    add_generation_prompt=not continue_final,\n"
        )
        merge_replacement = (
            "            chat_template_kwargs=merge_kwargs(  # To remove unset values\n"
            "                self.chat_template_kwargs or {},\n"
            "                dict(\n"
            "                    add_generation_prompt=not continue_final,\n"
        )
        if "self.chat_template_kwargs or {}" not in txt and merge_anchor in txt:
            txt = txt.replace(merge_anchor, merge_replacement, 1)
            print(" -> Patched protocol.py (15b: to_chat_params merges user chat_template_kwargs)")

        p_responses_proto.write_text(txt)

    # Patch 16 (local): register the AWQ-INT4 MMQ HIP custom op into vLLM's
    # mixed-precision kernel dispatcher so it's picked ahead of TritonW4A16
    # for the W4A16 g32 path on gfx1151. The .so is built from
    # /workspace/csrc/awq_mmq_gfx1151/ (host-mounted at /root/csrc/) and
    # imports lazily at module-load time.
    #
    # Implementation: append a registration block to the dispatcher's
    # __init__.py. On load the block adds the package dir to sys.path,
    # imports our RocmMmqQ4LinearKernel, and inserts it at position 0 of
    # _POSSIBLE_KERNELS[ROCM]. If the import fails (e.g. .so not built yet),
    # the kernel list is left untouched and TritonW4A16 keeps its slot.
    #
    # apply_weights internally dispatches: M >= 32 (prefill) -> our HIP
    # kernel, M < 32 (decode) -> TritonW4A16 fallback. Both paths use the
    # same layer's weight tensors via the dual-storage process_weights step.
    # See .research/mmq-q4-gfx1151-port/FINDINGS.md.
    p_dispatch = Path('vllm/model_executor/kernels/linear/__init__.py')
    if p_dispatch.exists():
        txt = p_dispatch.read_text()
        if "Patch 16" not in txt:
            injection = (
                "\n\n# --- Patch 16: AWQ-INT4 MMQ HIP custom op for gfx1151 (Strix Halo) ---\n"
                "import sys as _sys\n"
                "import os as _os\n"
                "_AWQ_MMQ_DIR = '/root/csrc/awq_mmq_gfx1151'\n"
                "if _os.path.exists(_AWQ_MMQ_DIR) and _AWQ_MMQ_DIR not in _sys.path:\n"
                "    _sys.path.insert(0, _AWQ_MMQ_DIR)\n"
                "try:\n"
                "    from awq_mmq_gfx1151.vllm_kernel import RocmMmqQ4LinearKernel as _RocmMmqQ4\n"
                "    if _RocmMmqQ4 not in _POSSIBLE_KERNELS.get(PlatformEnum.ROCM, []):\n"
                "        _POSSIBLE_KERNELS[PlatformEnum.ROCM].insert(0, _RocmMmqQ4)\n"
                "        logger.info('Patch 16: RocmMmqQ4LinearKernel registered at _POSSIBLE_KERNELS[ROCM][0]')\n"
                "except Exception as _e:\n"
                "    logger.warning('Patch 16: failed to register RocmMmqQ4LinearKernel: %s', _e)\n"
                "# --- end Patch 16 ---\n"
            )
            txt += injection
            p_dispatch.write_text(txt)
            print(" -> Patched vllm/model_executor/kernels/linear/__init__.py (Patch 16: AWQ-INT4 MMQ HIP)")

    # Patch 17 (local): drop vLLM's half/half2 atomicAdd polyfills on ROCm.
    #
    # csrc/quantization/gptq/compat.cuh ships polyfills
    #   __device__ void atomicAdd(half*  address, half  val)
    #   __device__ void atomicAdd(half2* address, half2 val)
    # gated on `#if defined(__CUDA_ARCH__) || defined(USE_ROCM)`. ROCm 7.13
    # nightlies (post 7.13.0a20260426) added builtins
    #   __device__ __half  atomicAdd(__half*  const, const __half)   @ amd_hip_fp16.h:869
    #   __device__ __half2 atomicAdd(__half2* const, const __half2)  @ amd_hip_fp16.h:875
    # With both the polyfill and the builtin visible, clang reports
    # "call to 'atomicAdd' is ambiguous" in q_gemm.hip (10 sites).
    #
    # Fix: change the outermost guard to drop the entire ROCm path through
    # this overload region. The polyfills are now CUDA-only; ROCm uses the
    # HIP builtins exclusively. The named helpers atomicAdd_half /
    # atomicAdd_half2 (defined above the guard) are untouched in case any
    # other vLLM source calls them by name.
    p_compat = Path('csrc/quantization/gptq/compat.cuh')
    if p_compat.exists():
        txt = p_compat.read_text()
        old_guard = "#if defined(__CUDA_ARCH__) || defined(USE_ROCM)\n"
        new_guard = "#if defined(__CUDA_ARCH__)\n"
        if old_guard in txt:
            txt = txt.replace(old_guard, new_guard, 1)
            p_compat.write_text(txt)
            print(" -> Patched csrc/quantization/gptq/compat.cuh (Patch 17: drop atomicAdd half/half2 polyfills on ROCm)")

    # Patch 18 (local): restore HIP_FOUND for newer PyTorch builds.
    #
    # PyTorch's cmake/public/LoadHIP.cmake used to call
    #   find_package(HIP 1.0 MODULE)
    # which set the cmake variable HIP_FOUND (uppercase) as a side effect.
    # Somewhere between PyTorch v2.10.0 and main, that MODULE-mode finder
    # was replaced with
    #   find_package_and_print_version(hip REQUIRED CONFIG)
    # which only sets PYTORCH_FOUND_HIP and hip_FOUND (lowercase). The
    # uppercase HIP_FOUND is no longer exported.
    #
    # vLLM v0.20.0 CMakeLists.txt:125-148 still gates GPU-language
    # detection on `elseif(HIP_FOUND)`, so against newer torch wheels
    # (e.g. 2.13.0a0+rocm7.13.0a20260510 from rocm.nightlies v2-staging)
    # the build dies with
    #   CMake Error at CMakeLists.txt:147 (message):
    #     Can't find CUDA or HIP installation.
    # Trigger: rocm.nightlies index rolled torch 2.10 -> 2.13 in 14 days.
    #
    # Fix: alias HIP_FOUND from PYTORCH_FOUND_HIP / hip_FOUND right after
    # find_package(Torch REQUIRED) so the legacy uppercase variable is
    # populated regardless of which LoadHIP.cmake variant we got.
    p_cmake = Path('CMakeLists.txt')
    if p_cmake.exists():
        txt = p_cmake.read_text()
        marker = "find_package(Torch REQUIRED)\n"
        shim = (
            "find_package(Torch REQUIRED)\n"
            "\n"
            "# Patch 18 (Strix): newer PyTorch (>=~2.11) LoadHIP.cmake drops the legacy\n"
            "# find_package(HIP MODULE) call and only sets PYTORCH_FOUND_HIP / hip_FOUND.\n"
            "# Re-export the uppercase HIP_FOUND that the rest of this file (and vLLM\n"
            "# v0.20.0 in general) still reads.\n"
            "if(NOT HIP_FOUND AND (PYTORCH_FOUND_HIP OR hip_FOUND))\n"
            "  set(HIP_FOUND TRUE)\n"
            "endif()\n"
        )
        if "Patch 18" not in txt and marker in txt:
            txt = txt.replace(marker, shim, 1)
            p_cmake.write_text(txt)
            print(" -> Patched CMakeLists.txt (Patch 18: alias HIP_FOUND from PYTORCH_FOUND_HIP / hip_FOUND)")

    # Patch 19 (local): bump Triton fallback iteration tile from 32 to 64 on the
    # non-power-of-2 paged-attention path.
    #
    # Why: ROCm's native paged_attention_rocm rejects non-pow2 cache block
    # sizes (Qwen3 / DFlash hybrid produce 816 / 832 / 848 depending on
    # num_speculative_tokens). vLLM falls back to a Triton kernel that
    # iterates over the KV cache with TRITON_BLOCK_SIZE=32 and Q/K tiles of
    # BLOCK_M=BLOCK_N=32. At 32K context that's ~1000 iterations of a tiny
    # tile per decode step -- two orders of magnitude below UMA bandwidth
    # ceiling. Bumping the tile to 64 reduces iterations 2x and matches
    # gfx1151 wave32 + AMD WMMA 16x16x16 well without risking VGPR spill
    # (a 128 tile of bf16 K+V at head_dim 128 stresses the 1536 VGPR / SIMD
    # budget; 64 is safe).
    #
    # The kernel already decouples the iteration tile (BLOCK_SIZE constexpr)
    # from the physical cache block (PHYSICAL_BLOCK_SIZE constexpr) via
    # per-token block-table lookup, so the tile is free to be any value --
    # this fix is size-agnostic and will work for any future non-pow2
    # PHYSICAL_BLOCK_SIZE that DFlash / hybrid layouts produce.
    #
    # No allocator / cache-geometry change. The standard pow2 path is
    # untouched and continues to use the ROCm custom paged attention kernel.
    p_cppd = Path('vllm/v1/attention/ops/chunked_prefill_paged_decode.py')
    if p_cppd.exists():
        txt = p_cppd.read_text()
        if "Patch 19" not in txt:
            old = "TRITON_BLOCK_SIZE = block_size if is_pow2 else 32"
            new = "TRITON_BLOCK_SIZE = block_size if is_pow2 else 64  # Patch 19: bump non-pow2 iteration tile"
            if old in txt:
                txt = txt.replace(old, new, 1)
                p_cppd.write_text(txt)
                print(" -> Patched vllm/v1/attention/ops/chunked_prefill_paged_decode.py (Patch 19: bump non-pow2 iteration tile to 64)")

    p_pp = Path('vllm/v1/attention/ops/prefix_prefill.py')
    if p_pp.exists():
        txt = p_pp.read_text()
        if "Patch 19" not in txt:
            old_block = (
                "    if is_pow2:\n"
                "        BLOCK_M = 128\n"
                "        BLOCK_N = 64\n"
                "    else:\n"
                "        BLOCK_M = 32\n"
                "        BLOCK_N = 32\n"
                "\n"
                "    # TRITON_BLOCK_SIZE is kept at 32 to ensure\n"
                "    # correct alignment logic when the kernel handles\n"
                "    # non-standard sizes (such as 544).\n"
                "    TRITON_BLOCK_SIZE = 32"
            )
            new_block = (
                "    if is_pow2:\n"
                "        BLOCK_M = 128\n"
                "        BLOCK_N = 64\n"
                "    else:\n"
                "        # Patch 19: bump non-pow2 tile from 32 to 64 for long-context decode throughput\n"
                "        BLOCK_M = 64\n"
                "        BLOCK_N = 64\n"
                "\n"
                "    # Patch 19: iteration tile bumped 32 -> 64. Per-token block-table\n"
                "    # lookup makes the tile size independent of PHYSICAL_BLOCK_SIZE,\n"
                "    # so this is size-agnostic for any future non-standard cache size.\n"
                "    TRITON_BLOCK_SIZE = 64"
            )
            if old_block in txt:
                txt = txt.replace(old_block, new_block, 1)
                p_pp.write_text(txt)
                print(" -> Patched vllm/v1/attention/ops/prefix_prefill.py (Patch 19: bump non-pow2 iteration tile to 64)")

    # Patch 20 (local): Add non-causal attention support to TRITON_ATTN's
    # unified-attention kernel. Mirrors PR #40176's ROCm-only non-causal
    # support but applies it to vllm/v1/attention/ops/triton_unified_attention.py
    # and vllm/v1/attention/backends/triton_attn.py.
    #
    # Why: TRITON_ATTN's kernel_unified_attention already has a 3D split-K
    # (Flash-Decoding) path that fully utilizes gfx1151's 40 CUs. ROCM_ATTN's
    # kernel_paged_attention_2d is 2D-only, single-CU per (seq, kv_head), and
    # serially scans 512 tiles at 32K context (3.45 t/s ceiling).
    # The only blocker to running TRITON_ATTN under DFlash is one assert:
    #   assert causal, "Only causal attention is supported"
    # DFlash verify needs non-causal (drafted tokens can't see future drafts
    # in the same step). This patch threads a CAUSAL constexpr through the
    # kernel + helpers + backend wrapper, exactly like Patch 13 did for ROCm.
    #
    # Combined with Patch 21 (relax the max_seqlen_q>1 gate that blocks the
    # 3D path for DFlash verify's N+1 query tokens), this is expected to
    # 2-3x decode at 32K (3.45 -> 7-10 t/s) and apply SWA loop-bound
    # tightening on drafter layers (compute_tile_loop_bounds in
    # triton_attention_helpers.py already prunes SWA tiles correctly).
    #
    # 20a: triton_unified_attention.py - relax assert, thread CAUSAL through
    p_tua = Path('vllm/v1/attention/ops/triton_unified_attention.py')
    if p_tua.exists():
        txt = p_tua.read_text()
        applied = False

        if "Patch 20" not in txt:
            # 1. Drop the causal-only assert. compute_kv_seq_mask now branches
            #    on CAUSAL and composes correctly with SWA / chunked attention
            #    (PR #40176's prefix_prefill non-causal+SWA pattern carries
            #    through here via the same AND-of-masks structure).
            old_assert = 'assert causal, "Only causal attention is supported"'
            new_assert = (
                '# Patch 20: causal toggle threaded through compute_kv_seq_mask.\n'
                '    # SWA + non-causal is well-defined: bidirectional within window.\n'
                '    pass'
            )
            if old_assert in txt:
                txt = txt.replace(old_assert, new_assert, 1)
                applied = True

            # 2. Add CAUSAL constexpr to kernel_unified_attention signature
            #    (insert right after IS_3D: tl.constexpr,).
            old_sig = (
                "    IS_3D: tl.constexpr,\n"
                "    # KV cache quantization mode handled inside this kernel via constexpr"
            )
            new_sig = (
                "    IS_3D: tl.constexpr,\n"
                "    # Patch 20: causal toggle. False=bidirectional within seq bounds\n"
                "    # (DFlash verify); True=standard causal. SWA + non-causal is\n"
                "    # disallowed in the launcher assert above.\n"
                "    CAUSAL: tl.constexpr = True,\n"
                "    # KV cache quantization mode handled inside this kernel via constexpr"
            )
            if old_sig in txt:
                txt = txt.replace(old_sig, new_sig, 1)
                applied = True

            # 3. Pass CAUSAL into compute_kv_seq_mask call.
            old_call = (
                "        seq_mask = compute_kv_seq_mask(\n"
                "            query_abs_pos,\n"
                "            seq_offset,\n"
                "            seq_idx,\n"
                "            mm_prefix_range_ptr,\n"
                "            SLIDING_WINDOW,\n"
                "            USE_MM_PREFIX,\n"
                "            MAX_MM_RANGES,\n"
                "            CHUNK_LOOKBACK,\n"
                "            CHUNK_SIZE,\n"
                "        )"
            )
            new_call = (
                "        seq_mask = compute_kv_seq_mask(\n"
                "            query_abs_pos,\n"
                "            seq_offset,\n"
                "            seq_idx,\n"
                "            mm_prefix_range_ptr,\n"
                "            SLIDING_WINDOW,\n"
                "            USE_MM_PREFIX,\n"
                "            MAX_MM_RANGES,\n"
                "            CHUNK_LOOKBACK,\n"
                "            CHUNK_SIZE,\n"
                "            CAUSAL,  # Patch 20\n"
                "        )"
            )
            if old_call in txt:
                txt = txt.replace(old_call, new_call, 1)
                applied = True

            # 4. Pass CAUSAL=causal in kernel launch (insert after IS_3D=use_3d,).
            old_launch = "        IS_3D=use_3d,\n        KV_QUANT_MODE=kv_quant_mode,"
            new_launch = (
                "        IS_3D=use_3d,\n"
                "        CAUSAL=causal,  # Patch 20: thread non-causal through to kernel\n"
                "        KV_QUANT_MODE=kv_quant_mode,"
            )
            if old_launch in txt:
                txt = txt.replace(old_launch, new_launch, 1)
                applied = True

        if applied:
            p_tua.write_text(txt)
            print(" -> Patched vllm/v1/attention/ops/triton_unified_attention.py (Patch 20: non-causal support)")

    # 20b: triton_attention_helpers.py - CAUSAL branch in compute_kv_seq_mask
    p_tah = Path('vllm/v1/attention/ops/triton_attention_helpers.py')
    if p_tah.exists():
        txt = p_tah.read_text()
        applied = False

        if "Patch 20" not in txt:
            # 1. Add CAUSAL constexpr to compute_kv_seq_mask signature.
            old_sig = (
                "def compute_kv_seq_mask(\n"
                "    query_abs_pos,\n"
                "    seq_offset,\n"
                "    seq_idx,\n"
                "    mm_prefix_range_ptr,\n"
                "    SLIDING_WINDOW: tl.constexpr,\n"
                "    USE_MM_PREFIX: tl.constexpr,\n"
                "    MAX_MM_RANGES: tl.constexpr,\n"
                "    CHUNK_LOOKBACK: tl.constexpr = -1,\n"
                "    CHUNK_SIZE: tl.constexpr = -1,\n"
                "):"
            )
            new_sig = (
                "def compute_kv_seq_mask(\n"
                "    query_abs_pos,\n"
                "    seq_offset,\n"
                "    seq_idx,\n"
                "    mm_prefix_range_ptr,\n"
                "    SLIDING_WINDOW: tl.constexpr,\n"
                "    USE_MM_PREFIX: tl.constexpr,\n"
                "    MAX_MM_RANGES: tl.constexpr,\n"
                "    CHUNK_LOOKBACK: tl.constexpr = -1,\n"
                "    CHUNK_SIZE: tl.constexpr = -1,\n"
                "    CAUSAL: tl.constexpr = True,  # Patch 20\n"
                "):"
            )
            if old_sig in txt:
                txt = txt.replace(old_sig, new_sig, 1)
                applied = True

            # 2. Branch on CAUSAL for the base mask. Non-causal allows the full
            #    prefix (DFlash verify still needs the per-query position to
            #    bound; tile_mask handles that via max_seq_prefix_len).
            old_mask = "    # Compute attention mask: causal by default (key <= query)\n    seq_mask = seq_offset[None, :] <= query_abs_pos"
            new_mask = (
                "    # Compute attention mask: causal by default (key <= query).\n"
                "    # Patch 20: CAUSAL=False relaxes the causal constraint so drafted\n"
                "    # tokens (DFlash verify) can see each other and the prefix equally.\n"
                "    # tile_mask in the caller bounds seq_offset to < max_seq_prefix_len,\n"
                "    # so OOB keys are zeroed by the V/K masked load + this mask leaves\n"
                "    # softmax to ignore those rows.\n"
                "    if CAUSAL:\n"
                "        seq_mask = seq_offset[None, :] <= query_abs_pos\n"
                "    else:\n"
                "        # All-True (BLOCK_M, TILE_SIZE) via broadcast of always-true\n"
                "        # row vector with implicit broadcast against query_abs_pos.\n"
                "        seq_mask = (seq_offset[None, :] >= 0) | (query_abs_pos < 0)"
            )
            if old_mask in txt:
                txt = txt.replace(old_mask, new_mask, 1)
                applied = True

        if applied:
            p_tah.write_text(txt)
            print(" -> Patched vllm/v1/attention/ops/triton_attention_helpers.py (Patch 20: non-causal mask branch)")

    # 20c: triton_attn.py - metadata field + supports_non_causal + thread causal
    p_triton_attn = Path('vllm/v1/attention/backends/triton_attn.py')
    if p_triton_attn.exists():
        txt = p_triton_attn.read_text()
        applied = False

        if "Patch 20" not in txt:
            # 1. Add causal field to TritonAttentionMetadata.
            old_field_block = (
                "    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None\n"
                "    mm_prefix_range_tensor: torch.Tensor | None = None\n"
            )
            new_field_block = (
                "    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None\n"
                "    mm_prefix_range_tensor: torch.Tensor | None = None\n"
                "\n"
                "    # Patch 20: DFlash drafting sets this to False via\n"
                "    # CommonAttentionMetadata.causal.\n"
                "    causal: bool = True\n"
            )
            if old_field_block in txt:
                txt = txt.replace(old_field_block, new_field_block, 1)
                applied = True

            # 2. Backend.supports_non_causal() classmethod returns True.
            old_sink_block = (
                "    @classmethod\n"
                "    def supports_sink(cls) -> bool:\n"
                "        return True\n"
            )
            new_sink_block = (
                "    @classmethod\n"
                "    def supports_sink(cls) -> bool:\n"
                "        return True\n"
                "\n"
                "    @classmethod\n"
                "    def supports_non_causal(cls) -> bool:\n"
                "        # Patch 20: kernel_unified_attention now supports CAUSAL=False\n"
                "        # (DFlash drafter verify uses this).\n"
                "        return True\n"
            )
            if old_sink_block in txt and "def supports_non_causal" not in txt:
                txt = txt.replace(old_sink_block, new_sink_block, 1)
                applied = True

            # 3. Builder.build() - propagate common_attn_metadata.causal.
            old_build_tail = (
                "            softmax_segm_max=self.softmax_segm_max,\n"
                "            softmax_segm_expsum=self.softmax_segm_expsum,\n"
                "        )\n"
                "        return attn_metadata\n"
            )
            new_build_tail = (
                "            softmax_segm_max=self.softmax_segm_max,\n"
                "            softmax_segm_expsum=self.softmax_segm_expsum,\n"
                "            causal=common_attn_metadata.causal,  # Patch 20\n"
                "        )\n"
                "        return attn_metadata\n"
            )
            if old_build_tail in txt and "causal=common_attn_metadata.causal" not in txt:
                txt = txt.replace(old_build_tail, new_build_tail, 1)
                applied = True

            # 4. forward() - replace hardcoded causal=True with metadata flag.
            old_forward_call = (
                "            softmax_scale=self.scale,\n"
                "            causal=True,\n"
            )
            new_forward_call = (
                "            softmax_scale=self.scale,\n"
                "            causal=attn_metadata.causal,  # Patch 20\n"
            )
            if old_forward_call in txt:
                txt = txt.replace(old_forward_call, new_forward_call, 1)
                applied = True

        if applied:
            p_triton_attn.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/triton_attn.py (Patch 20: non-causal support)")

    # Patch 21 (local): Relax the 3D-launch gate so DFlash verify (max_seqlen_q
    # = N+1) can use Flash-Decoding split-K.
    #
    # Why: kernel_unified_attention's 3D path splits the KV dimension into
    # NUM_SEGMENTS_PER_SEQ workgroups via `tl.program_id(2)`. On gfx1151 with
    # 40 CUs and 8 KV heads at bs=1, the 2D path launches only 8 workgroups
    # (20% utilization) while 3D launches 32 (full utilization). The math
    # works under associativity of online softmax regardless of max_seqlen_q;
    # the existing gate `max_seqlen_q > 1` was a conservatism, not a correctness
    # requirement. The reduce_segments pass handles per-query merging.
    #
    # DFlash verify uses max_seqlen_q = num_speculative_tokens + 1 = 5 (N=4)
    # or 9 (N=8). Bumping the threshold to 16 keeps the gate honest while
    # admitting our verify path.
    p_tua2 = Path('vllm/v1/attention/ops/triton_unified_attention.py')
    if p_tua2.exists():
        txt = p_tua2.read_text()
        if "Patch 21" not in txt:
            old_gate = "        or max_seqlen_q > 1\n"
            new_gate = "        or max_seqlen_q > 16  # Patch 21: admit DFlash verify (N+1 queries)\n"
            if old_gate in txt:
                txt = txt.replace(old_gate, new_gate, 1)
                p_tua2.write_text(txt)
                print(" -> Patched vllm/v1/attention/ops/triton_unified_attention.py (Patch 21: relax 3D gate for DFlash verify)")

    # Patch 22 candidate (NUM_PAR_SOFTMAX_SEGMENTS 16 -> 32) was tested 2026-06-04
    # and reverted: mid-ctx regression (-1.5% / -3.9% at 8K/16K) outweighed the
    # tiny 32K gain (+2.2%). Default 16 keeps 128 workgroups across the 40 CUs
    # of gfx1151 (already saturated at ~3 waves of overcommit) — reduce-segments
    # cost at 32 dominates the shorter per-segment serial chain.
    # See .research/patch22-num-par-segments/FINDINGS.md.

    # Patch 24 (local cherry-pick of vLLM PR #42102, closed unmerged 2026-05-15):
    # "Allow DFlash drafter to coexist with quantized target KV via independent
    # KV groups + dtype override".
    #
    # Why: With VLLM_KV_CACHE_DTYPE=fp8, engine init crashes at
    #   unify_kv_cache_spec_page_size  ->  AssertionError
    # because DFlash drafter Attention layers inherit the target's quantized
    # cache_config and end up with FP8 pages computed against different
    # num_kv_heads/head_size than target layers (drafter is its own ~2B
    # transformer; weights are BF16 but page sizes don't align with target's
    # FP8-padded Mamba page). The unify pass then tries to scale the smaller
    # layer's block_size, which is a no-op for MambaSpec (page_size_bytes is
    # independent of block_size), and the post-scale assert fires.
    #
    # PR #42102 fixes this by:
    #   (a) [Patch 24a] Partitioning DFlash drafter layers out of the target
    #       group BEFORE the unify pass runs. Drafter and target then each
    #       resolve their own uniform page size and pool, with no cross-group
    #       page-size constraint. Layer indices >= target_num_layers are
    #       drafter; uses regex on the layer name.
    #   (b) [Patch 24b] Overriding the drafter's cache_dtype to "auto"
    #       (i.e. BF16) so the drafter does not inherit FP8 from target's
    #       cache_config. Drafter weights are BF16; quantizing drafter KV
    #       adds no benefit and increases per-token dequant overhead.
    #
    # PR #42102 also touched flash_attn.py to thread per-spec kv_quant_mode
    # through the metadata scheduler. We use TRITON_ATTN; TritonAttentionImpl
    # captures kv_cache_dtype per-layer at __init__ time (triton_attn.py:491,
    # self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)) so per-layer
    # dtype routing already works — no analogous patch needed for our backend.
    #
    # See .research/fp8-kv-cache/FINDINGS.md for the failure trace and the
    # upstream investigation (PR #40128 closed, PR #42102 closed, issue #43626
    # still open) that led to porting this locally.
    #
    # 24a: kv_cache_utils.py - partition drafter from target before unify.
    p_kvu = Path('vllm/v1/core/kv_cache_utils.py')
    if p_kvu.exists():
        txt = p_kvu.read_text()
        applied = False

        # Per-sub-patch idempotency. Do NOT guard the whole block by a single
        # "Patch 24a" marker - sub-patch 7 was added after sub-patches 1-6 had
        # already been baked into the live file, so a single outer guard would
        # skip 7 on every subsequent run.
        if True:
            # 1. Add `import re` to the stdlib imports block. Used by the
            #    layer-index regex.
            old_imp = "import math\nimport os\nfrom collections import defaultdict\n"
            new_imp = "import math\nimport os\nimport re  # Patch 24a: DFlash layer-index regex\nfrom collections import defaultdict\n"
            if old_imp in txt and "import re  # Patch 24a" not in txt:
                txt = txt.replace(old_imp, new_imp, 1)
                applied = True

            # 2. Add the layer-index regex constant right after logger init.
            old_logger = (
                "logger = init_logger(__name__)\n\n"
                "# The hash seed for the first block of any prefix block sequence.\n"
            )
            new_logger = (
                "logger = init_logger(__name__)\n\n"
                "# Patch 24a (PR #42102): DFlash drafter layer-index pattern.\n"
                "# Layer indices >= target_num_layers are drafter layers; we\n"
                "# partition them into an isolated KV cache group before unify\n"
                "# so drafter (BF16) and target (FP8) page sizes don't collide.\n"
                "_LAYER_INDEX_RE = re.compile(r\"(?:^|[.])layers[.](\\d+)(?:[.]|$)\")\n\n"
                "# The hash seed for the first block of any prefix block sequence.\n"
            )
            if old_logger in txt and "_LAYER_INDEX_RE" not in txt:
                txt = txt.replace(old_logger, new_logger, 1)
                applied = True

            # 3. Insert helper functions immediately before unify_kv_cache_spec_page_size.
            old_unify_def = (
                "def unify_kv_cache_spec_page_size(\n"
                "    kv_cache_spec: dict[str, KVCacheSpec],\n"
                ") -> dict[str, KVCacheSpec]:\n"
            )
            helpers_block = (
                "# Patch 24a (PR #42102): DFlash drafter partitioning helpers.\n"
                "def _get_dflash_isolated_layer_names(\n"
                "    vllm_config: VllmConfig,\n"
                "    layer_names: Iterable[str],\n"
                ") -> set[str]:\n"
                "    spec_config = vllm_config.speculative_config\n"
                "    if spec_config is None or getattr(spec_config, \"method\", None) != \"dflash\":\n"
                "        return set()\n"
                "    try:\n"
                "        target_num_layers = vllm_config.model_config.get_num_layers(\n"
                "            vllm_config.parallel_config\n"
                "        )\n"
                "    except Exception:\n"
                "        # Be conservative: if we cannot determine target layer count,\n"
                "        # do not isolate (falls back to the legacy unify behaviour).\n"
                "        return set()\n"
                "    isolated: set[str] = set()\n"
                "    for layer_name in layer_names:\n"
                "        m = _LAYER_INDEX_RE.search(layer_name)\n"
                "        if m is not None and int(m.group(1)) >= target_num_layers:\n"
                "            isolated.add(layer_name)\n"
                "    return isolated\n\n"
                "def _partition_dflash_isolated_specs(\n"
                "    vllm_config: VllmConfig,\n"
                "    kv_cache_spec: dict[str, KVCacheSpec],\n"
                ") -> tuple[dict[str, KVCacheSpec], dict[str, KVCacheSpec]]:\n"
                "    isolated_names = _get_dflash_isolated_layer_names(\n"
                "        vllm_config, kv_cache_spec.keys()\n"
                "    )\n"
                "    if not isolated_names:\n"
                "        return kv_cache_spec, {}\n"
                "    shared_specs = {n: s for n, s in kv_cache_spec.items() if n not in isolated_names}\n"
                "    isolated_specs = {n: s for n, s in kv_cache_spec.items() if n in isolated_names}\n"
                "    if not shared_specs or not isolated_specs:\n"
                "        return kv_cache_spec, {}\n"
                "    return shared_specs, isolated_specs\n\n"
                "def _get_dflash_isolated_group_ids(\n"
                "    vllm_config: VllmConfig,\n"
                "    kv_cache_groups: list[KVCacheGroupSpec],\n"
                ") -> set[int]:\n"
                "    isolated_names = _get_dflash_isolated_layer_names(\n"
                "        vllm_config,\n"
                "        (n for g in kv_cache_groups for n in g.layer_names),\n"
                "    )\n"
                "    if not isolated_names:\n"
                "        return set()\n"
                "    out: set[int] = set()\n"
                "    for gid, g in enumerate(kv_cache_groups):\n"
                "        if g.layer_names and all(n in isolated_names for n in g.layer_names):\n"
                "            out.add(gid)\n"
                "    return out\n\n"
                "def _get_layer_spec_from_group(\n"
                "    group: KVCacheGroupSpec,\n"
                "    layer_name: str,\n"
                ") -> KVCacheSpec:\n"
                "    gs = group.kv_cache_spec\n"
                "    if isinstance(gs, UniformTypeKVCacheSpecs):\n"
                "        return gs.kv_cache_specs[layer_name]\n"
                "    return gs\n\n"
                "def _get_dflash_isolated_layers(\n"
                "    vllm_config: VllmConfig,\n"
                "    kv_cache_groups: list[KVCacheGroupSpec],\n"
                ") -> list[tuple[str, KVCacheSpec]]:\n"
                "    gids = _get_dflash_isolated_group_ids(vllm_config, kv_cache_groups)\n"
                "    return [\n"
                "        (n, _get_layer_spec_from_group(kv_cache_groups[gid], n))\n"
                "        for gid in sorted(gids)\n"
                "        for n in kv_cache_groups[gid].layer_names\n"
                "    ]\n\n"
                "def _get_shared_kv_cache_groups(\n"
                "    vllm_config: VllmConfig,\n"
                "    kv_cache_groups: list[KVCacheGroupSpec],\n"
                ") -> list[KVCacheGroupSpec]:\n"
                "    gids = _get_dflash_isolated_group_ids(vllm_config, kv_cache_groups)\n"
                "    return [g for gid, g in enumerate(kv_cache_groups) if gid not in gids]\n\n\n"
            )
            new_unify_def = helpers_block + old_unify_def
            if old_unify_def in txt and "_get_dflash_isolated_layer_names" not in txt:
                txt = txt.replace(old_unify_def, new_unify_def, 1)
                applied = True

            # 4. Wrap get_kv_cache_groups: rename existing -> _get_kv_cache_groups,
            #    then add a new outer get_kv_cache_groups that partitions first.
            old_outer_def = "def get_kv_cache_groups(\n    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]\n) -> list[KVCacheGroupSpec]:\n"
            new_outer_def = "def _get_kv_cache_groups(\n    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]\n) -> list[KVCacheGroupSpec]:  # Patch 24a: renamed; outer wrapper below partitions DFlash drafter\n"
            if old_outer_def in txt and "def _get_kv_cache_groups(" not in txt:
                txt = txt.replace(old_outer_def, new_outer_def, 1)
                applied = True

            # 5. Append the new outer get_kv_cache_groups wrapper right after
            #    the renamed function's final `return` line.
            old_return_line = (
                "    # As KVCacheManager can only allocate memory of one size, we need to unify\n"
                "    # the page size of the layers. For cases cannot be unified, this function\n"
                "    # will raise an error.\n"
                "    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)\n"
                "    # Model contains multiple attention types, but KV cache of all layers\n"
                "    # have the same physical memory per block per layer. Split the layers\n"
                "    # into groups with the same number of layers, and thus same total page\n"
                "    # size.\n"
                "    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)\n"
            )
            wrapper_block = (
                "    # As KVCacheManager can only allocate memory of one size, we need to unify\n"
                "    # the page size of the layers. For cases cannot be unified, this function\n"
                "    # will raise an error.\n"
                "    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)\n"
                "    # Model contains multiple attention types, but KV cache of all layers\n"
                "    # have the same physical memory per block per layer. Split the layers\n"
                "    # into groups with the same number of layers, and thus same total page\n"
                "    # size.\n"
                "    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)\n\n\n"
                "def get_kv_cache_groups(\n"
                "    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]\n"
                ") -> list[KVCacheGroupSpec]:\n"
                "    \"\"\"Patch 24a (PR #42102): partition DFlash drafter layers into their\n"
                "    own KV group before the regular grouping/unify pipeline runs. Drafter\n"
                "    has its own page size (BF16, possibly different num_kv_heads/head_size\n"
                "    than target), so sharing the target's pool causes unify_kv_cache_spec_page_size\n"
                "    to assert when target is FP8.\n"
                "    \"\"\"\n"
                "    shared_specs, isolated_specs = _partition_dflash_isolated_specs(\n"
                "        vllm_config, kv_cache_spec\n"
                "    )\n"
                "    if isolated_specs:\n"
                "        return [\n"
                "            *_get_kv_cache_groups(vllm_config, shared_specs),\n"
                "            *_get_kv_cache_groups(vllm_config, isolated_specs),\n"
                "        ]\n"
                "    return _get_kv_cache_groups(vllm_config, kv_cache_spec)\n"
            )
            if old_return_line in txt and "def get_kv_cache_groups(" not in txt.split("def _get_kv_cache_groups(")[-1]:
                txt = txt.replace(old_return_line, wrapper_block, 1)
                applied = True

            # 6. get_kv_cache_config_from_groups general case: handle isolated
            #    layers by treating them as a per-layer pool appended to the
            #    shared pool.
            old_general = (
                "        # General case:\n"
                "        # We will have group_size memory pools, each is shared by one layer from\n"
                "        # each group. As layers of different groups have different block table,\n"
                "        # they will use different parts of the shared Tensor.\n"
                "        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),\n"
                "        # (sw.1, padding) will be: (group_size = 2)\n"
                "        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2\n"
                "        # full.1, sw.2: share another Tensor with size=available_memory//2\n"
                "        group_size = max(len(group.layer_names) for group in kv_cache_groups)\n\n"
                "        page_size = get_uniform_page_size(\n"
                "            [group.kv_cache_spec for group in kv_cache_groups]\n"
                "        )\n"
                "        assert group_size > 0, \"group_size must be greater than 0\"\n"
                "        num_blocks = get_num_blocks(\n"
                "            vllm_config,\n"
                "            group_size,\n"
                "            available_memory,\n"
                "            page_size,\n"
                "            suppress_log=suppress_log,\n"
                "        )\n"
                "        kv_cache_tensors = []\n"
                "        for i in range(group_size):\n"
                "            shared_by = []\n"
                "            for j in range(len(kv_cache_groups)):\n"
                "                if i < len(kv_cache_groups[j].layer_names):\n"
                "                    shared_by.append(kv_cache_groups[j].layer_names[i])\n"
                "            kv_cache_tensors.append(\n"
                "                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)\n"
                "            )\n"
            )
            new_general = (
                "        # General case (Patch 24a / PR #42102): split shared and DFlash-\n"
                "        # isolated groups. Shared groups follow the legacy layout (one pool\n"
                "        # per slot, shared by one layer from each group). Isolated drafter\n"
                "        # layers each get their own pool sized by that layer's page_size_bytes.\n"
                "        isolated_layers = _get_dflash_isolated_layers(vllm_config, kv_cache_groups)\n"
                "        shared_groups = (\n"
                "            _get_shared_kv_cache_groups(vllm_config, kv_cache_groups)\n"
                "            if isolated_layers else kv_cache_groups\n"
                "        )\n"
                "        shared_group_size = (\n"
                "            max(len(g.layer_names) for g in shared_groups) if shared_groups else 0\n"
                "        )\n"
                "        page_size = (\n"
                "            get_uniform_page_size([g.kv_cache_spec for g in shared_groups])\n"
                "            if shared_groups else 0\n"
                "        )\n"
                "        bytes_per_block = page_size * shared_group_size + sum(\n"
                "            spec.page_size_bytes for _, spec in isolated_layers\n"
                "        )\n"
                "        assert bytes_per_block > 0, \"bytes_per_block must be greater than 0\"\n"
                "        num_blocks = int(available_memory // bytes_per_block)\n"
                "        num_blocks = max(num_blocks, 0)\n"
                "        num_blocks = may_override_num_blocks(\n"
                "            vllm_config, num_blocks, suppress_log=suppress_log\n"
                "        )\n"
                "        kv_cache_tensors = []\n"
                "        for i in range(shared_group_size):\n"
                "            shared_by = []\n"
                "            for g in shared_groups:\n"
                "                if i < len(g.layer_names):\n"
                "                    shared_by.append(g.layer_names[i])\n"
                "            kv_cache_tensors.append(\n"
                "                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)\n"
                "            )\n"
                "        for layer_name, layer_spec in isolated_layers:\n"
                "            kv_cache_tensors.append(\n"
                "                KVCacheTensor(\n"
                "                    size=layer_spec.page_size_bytes * num_blocks,\n"
                "                    shared_by=[layer_name],\n"
                "                )\n"
                "            )\n"
            )
            if old_general in txt and "_get_dflash_isolated_layers(vllm_config, kv_cache_groups)" not in txt:
                txt = txt.replace(old_general, new_general, 1)
                applied = True

            # 7. _max_memory_usage_bytes_from_groups general case: also split
            #    shared vs isolated. This function feeds _check_enough_kv_cache_memory
            #    and the binary-search auto-fit; in v0.20.0 it predates the
            #    _pool_bytes_per_block refactor that PR #42102 touched in the
            #    newer vllm tree, so we keep the same split here.
            old_max_mem_general = (
                "    # General case: group_size pools, each shared by one layer per group\n"
                "    # Memory = group_size * page_size * blocks_for_max_len\n"
                "    group_size = max(len(group.layer_names) for group in kv_cache_groups)\n"
                "    page_size = get_uniform_page_size(\n"
                "        [group.kv_cache_spec for group in kv_cache_groups]\n"
                "    )\n"
                "    blocks_needed = sum(\n"
                "        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)\n"
                "        for group in kv_cache_groups\n"
                "    )\n\n"
                "    return group_size * page_size * blocks_needed\n"
            )
            new_max_mem_general = (
                "    # General case (Patch 24a / PR #42102): split shared vs DFlash-\n"
                "    # isolated. Shared follows the legacy group_size pool layout;\n"
                "    # isolated layers each contribute their own per-layer memory bound.\n"
                "    isolated_layers = _get_dflash_isolated_layers(vllm_config, kv_cache_groups)\n"
                "    shared_groups = (\n"
                "        _get_shared_kv_cache_groups(vllm_config, kv_cache_groups)\n"
                "        if isolated_layers else kv_cache_groups\n"
                "    )\n"
                "    shared_bytes = 0\n"
                "    if shared_groups:\n"
                "        group_size = max(len(group.layer_names) for group in shared_groups)\n"
                "        page_size = get_uniform_page_size(\n"
                "            [group.kv_cache_spec for group in shared_groups]\n"
                "        )\n"
                "        blocks_needed = sum(\n"
                "            cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)\n"
                "            for group in shared_groups\n"
                "        )\n"
                "        shared_bytes = group_size * page_size * blocks_needed\n"
                "    isolated_bytes = sum(\n"
                "        spec.max_memory_usage_bytes(vllm_config) for _, spec in isolated_layers\n"
                "    )\n"
                "    return shared_bytes + isolated_bytes\n"
            )
            if old_max_mem_general in txt and "isolated_bytes = sum(" not in txt:
                txt = txt.replace(old_max_mem_general, new_max_mem_general, 1)
                applied = True

        if applied:
            p_kvu.write_text(txt)
            print(" -> Patched vllm/v1/core/kv_cache_utils.py (Patch 24a: DFlash KV partition for FP8 coexistence)")

    # 24b: qwen3_dflash.py - override drafter cache_dtype to BF16 ("auto") so
    # drafter does not inherit FP8 from the target's cache_config.
    p_qd = Path('vllm/model_executor/models/qwen3_dflash.py')
    if p_qd.exists():
        txt = p_qd.read_text()
        applied = False

        if "Patch 24b" not in txt:
            # 1. Add `from dataclasses import replace` and is_quantized_kv_cache import.
            old_imports = (
                "from collections.abc import Iterable\n"
                "\n"
                "import torch\n"
            )
            new_imports = (
                "from collections.abc import Iterable\n"
                "from dataclasses import replace  # Patch 24b\n"
                "\n"
                "import torch\n"
            )
            if old_imports in txt and "from dataclasses import replace  # Patch 24b" not in txt:
                txt = txt.replace(old_imports, new_imports, 1)
                applied = True

            old_util_import = "from vllm.transformers_utils.config import set_default_rope_theta\n"
            new_util_import = (
                "from vllm.transformers_utils.config import set_default_rope_theta\n"
                "from vllm.utils.torch_utils import is_quantized_kv_cache  # Patch 24b\n"
            )
            if old_util_import in txt and "is_quantized_kv_cache  # Patch 24b" not in txt:
                txt = txt.replace(old_util_import, new_util_import, 1)
                applied = True

            # 2. Override draft_cache_config and pass it instead of cache_config.
            old_attn_init = (
                "        self.attn = Attention(\n"
                "            self.num_heads,\n"
                "            self.head_dim,\n"
                "            self.scaling,\n"
                "            num_kv_heads=self.num_kv_heads,\n"
                "            cache_config=cache_config,\n"
            )
            new_attn_init = (
                "        # Patch 24b (PR #42102): DFlash drafter uses its own KV cache pool\n"
                "        # (see kv_cache_utils._partition_dflash_isolated_specs). Drafter\n"
                "        # weights are BF16; forcing BF16 KV avoids unnecessary FP8 dequant\n"
                "        # on the drafter's hot path. Target attention KV remains quantized.\n"
                "        draft_cache_config = cache_config\n"
                "        if draft_cache_config is not None and is_quantized_kv_cache(\n"
                "            draft_cache_config.cache_dtype\n"
                "        ):\n"
                "            draft_cache_config = replace(draft_cache_config, cache_dtype=\"auto\")\n"
                "        self.attn = Attention(\n"
                "            self.num_heads,\n"
                "            self.head_dim,\n"
                "            self.scaling,\n"
                "            num_kv_heads=self.num_kv_heads,\n"
                "            cache_config=draft_cache_config,\n"
            )
            if old_attn_init in txt and "draft_cache_config" not in txt:
                txt = txt.replace(old_attn_init, new_attn_init, 1)
                applied = True

        if applied:
            p_qd.write_text(txt)
            print(" -> Patched vllm/model_executor/models/qwen3_dflash.py (Patch 24b: drafter BF16 KV override)")

    # Patch 24c (PR #42102 follow-up): with FP8 target + BF16 drafter, the
    # KVBlockZeroer's init_meta asserts uniform PAGE_SIZE_EL across all
    # FullAttentionSpec layers - but target pages are half-size (FP8) vs
    # drafter (BF16), so the assert fires with "Non-uniform page sizes:
    # 827392 vs 1654784". Mirror the encoder-only escape hatch: add drafter
    # layer names to runner_only_attn_layers before init_meta, so the
    # zeroer skips them. Drafter blocks come zeroed from the CuMem pool
    # allocator and the per-position seq mask prevents reads of unwritten
    # positions, so skipping explicit zero-fills is benign.
    p_gmr = Path('vllm/v1/worker/gpu_model_runner.py')
    if p_gmr.exists():
        txt = p_gmr.read_text()

        if "Patch 24c" not in txt:
            old_block = (
                "    def _init_kv_zero_meta(self) -> None:\n"
                "        \"\"\"One-time precomputation for _zero_block_ids.\n"
                "\n"
                "        Delegates to KVBlockZeroer.init_meta with the runner's state.\n"
                "        Called from gpu_worker.py outside the CuMem pool context.\n"
                "        \"\"\"\n"
                "        self._kv_block_zeroer = KVBlockZeroer(self.device, self.pin_memory)\n"
                "        self._kv_block_zeroer.init_meta(\n"
            )
            new_block = (
                "    def _init_kv_zero_meta(self) -> None:\n"
                "        \"\"\"One-time precomputation for _zero_block_ids.\n"
                "\n"
                "        Delegates to KVBlockZeroer.init_meta with the runner's state.\n"
                "        Called from gpu_worker.py outside the CuMem pool context.\n"
                "        \"\"\"\n"
                "        # Patch 24c: skip DFlash drafter layers in the zeroer.\n"
                "        # Target FP8 / drafter BF16 -> page sizes differ 2:1, so\n"
                "        # they cannot share a single PAGE_SIZE_EL kernel constant.\n"
                "        try:\n"
                "            from vllm.v1.core.kv_cache_utils import _get_dflash_isolated_layer_names\n"
                "            drafter_layers = _get_dflash_isolated_layer_names(\n"
                "                self.vllm_config,\n"
                "                self.compilation_config.static_forward_context.keys(),\n"
                "            )\n"
                "            self.runner_only_attn_layers |= drafter_layers\n"
                "        except Exception:\n"
                "            pass\n"
                "        self._kv_block_zeroer = KVBlockZeroer(self.device, self.pin_memory)\n"
                "        self._kv_block_zeroer.init_meta(\n"
            )
            if old_block in txt:
                txt = txt.replace(old_block, new_block, 1)
                p_gmr.write_text(txt)
                print(" -> Patched vllm/v1/worker/gpu_model_runner.py (Patch 24c: skip drafter in KVBlockZeroer)")

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
