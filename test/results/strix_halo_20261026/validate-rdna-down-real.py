"""GPU check of the optional down-projection layout, boundaries and bias.

Uses the actual vLLM loader and dispatch, plus an independent FP32 oracle.
This is a per-layer numerical check, not whole-model quality qualification.
"""
import json
from pathlib import Path
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.quant_utils import unpack_quantized_values_into_int32
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
    for layer_index in (0,31,63):
        k, n, group = 17408, 5120, 32
        model_root = Path('/models/hub/models--cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/46d3c4aca0cfa475e99283abe9fb1254df1eef03')
        index = json.loads((model_root/'model.safetensors.index.json').read_text())['weight_map']
        def read_weight(name):
            key=f'model.language_model.layers.{layer_index}.mlp.down_proj.'+name
            with safe_open(str(model_root/index[key]),framework='pt',device='cuda') as f:
                return f.get_tensor(key)
        packed,scale,zp = (read_weight(name) for name in ('weight_packed','weight_scale','weight_zero_point'))
        q = unpack_quantized_values_into_int32(packed,scalar_types.uint4,packed_dim=1)
        zeros = unpack_quantized_values_into_int32(zp,scalar_types.uint4,packed_dim=0)
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
        for m in (3, 5, 9):
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
            rows.append({'layer':layer_index, 'K': k, 'N': n, 'M': m,
                         'relative_error_reference': error.item(),
                         'relative_error_native': diff.item(),
                         'native_bitwise_equal': torch.equal(y, yn), 'passed': True})
            print(json.dumps(rows[-1]), flush=True)
        x=torch.randn(5,k,device='cuda',dtype=torch.float16)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): candidate.apply_weights(lc,x)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_y=candidate.apply_weights(lc,x)
        graph.replay()
        expected=candidate.apply_weights(lc,x)
        torch.cuda.synchronize()
        assert torch.equal(expected,graph_y)
        print('GRAPH_PASS='+str(layer_index),flush=True)
        del graph,graph_y,expected,x,candidate,native,lc,ln,q,scale,packed,zp,zeros,wref
        torch.cuda.empty_cache()
torch.cuda.synchronize()
print('RESULT_JSON=' + json.dumps({'passed': True,
      'scope': 'Actual target layers 0/31/63, per-layer numerics and graph replay, not whole-model quality', 'rows': rows}), flush=True)
