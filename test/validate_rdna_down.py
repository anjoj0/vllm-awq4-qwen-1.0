"""GPU check of the optional down-projection layout, boundaries and bias.

Uses the actual vLLM loader and dispatch, plus an independent FP32 oracle.
This is a per-layer numerical check, not whole-model quality qualification.
"""
import json
from unittest.mock import patch

import torch
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import RDNAHybridW4A16LinearKernel
from vllm.model_executor.parameter import PackedvLLMParameter, GroupQuantScaleParameter
from vllm.scalar_type import scalar_types
from rdna_qwen27b.linear import Qwen27BDownProjectionKernel

torch.set_num_threads(1)
torch.manual_seed(20261027)
rows = []
with torch.inference_mode(), \
        patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank', return_value=0), \
        patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size', return_value=1):
    k, n, group = 17408, 5120, 32
    q = torch.randint(0, 16, (n, k), dtype=torch.int32, device='cuda')
    scale = torch.rand(n, k // group, device='cuda', dtype=torch.float16) * 0.02 + 0.001
    shifts = torch.arange(8, device='cuda', dtype=torch.int32) * 4
    packed = torch.sum(q.view(n, k // 8, 8) << shifts, dim=2, dtype=torch.int32)
    zeros = torch.randint(0, 16, (n, k // group), dtype=torch.int32, device='cuda')
    zp = torch.sum(zeros.view(n // 8, 8, k // group) << shifts[None, :, None],
                   dim=1, dtype=torch.int32)
    wref = (q - zeros.repeat_interleave(group, dim=1)).half() * scale.repeat_interleave(group, dim=1)
    config = MPLinearLayerConfig(
        full_weight_shape=(k, n), partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4, act_type=torch.float16,
        group_size=group, zero_points=True)
    assert Qwen27BDownProjectionKernel.can_implement(config)[0]

    def layer():
        module = torch.nn.Module()
        loader = lambda *args: None
        module.register_parameter('q', PackedvLLMParameter(
            data=packed.clone(), input_dim=1, output_dim=0, packed_dim=1,
            packed_factor=8, weight_loader=loader))
        module.register_parameter('s', GroupQuantScaleParameter(
            data=scale.clone(), input_dim=1, output_dim=0, weight_loader=loader))
        module.register_parameter('z', PackedvLLMParameter(
            data=zp.clone(), input_dim=1, output_dim=0, packed_dim=0,
            packed_factor=8, weight_loader=loader))
        return module

    candidate = Qwen27BDownProjectionKernel(config, 'q', 's', 'z')
    native = RDNAHybridW4A16LinearKernel(config, 'q', 's', 'z')
    lc, ln = layer(), layer()
    candidate.process_weights_after_loading(lc)
    native.process_weights_after_loading(ln)
    for m in (1, 2, 3, 5, 9, 10, 129):
        x = torch.randn(m, k, device='cuda', dtype=torch.float16)
        bias = torch.randn(n, device='cuda', dtype=torch.float16) * 0.01
        if m == 9:
            x = x.view(3, 3, k)
        y = candidate.apply_weights(lc, x, bias)
        yn = native.apply_weights(ln, x, bias)
        reference = (x.float().reshape(m, k) @ wref.float().t()).half()
        reference.add_(bias)
        reference = reference.reshape(y.shape)
        error = (y.float() - reference.float()).norm() / reference.float().norm()
        diff = (y.float() - yn.float()).norm() / yn.float().norm()
        assert torch.isfinite(y).all() and error < 0.003 and diff < 0.003, (m, error, diff)
        if not 3 <= m <= 9:
            assert torch.equal(y, yn), (m, 'native fallback changed')
        rows.append({'K': k, 'N': n, 'M': m,
                     'relative_error_reference': error.item(),
                     'relative_error_native': diff.item(),
                     'native_bitwise_equal': torch.equal(y, yn), 'passed': True})
        print(json.dumps(rows[-1]), flush=True)
torch.cuda.synchronize()
print('RESULT_JSON=' + json.dumps({'passed': True,
      'scope': 'Per-layer numerics, not whole-model quality', 'rows': rows}), flush=True)
