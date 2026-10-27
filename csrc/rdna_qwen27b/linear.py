"""Use K-major storage only where the native HIP skinny kernel cannot run."""
import torch
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    RDNAHybridW4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import unpack_quantized_values_into_int32
from vllm.platforms.rocm import on_gfx1151
from .decode import down_gemm

logger = init_logger(__name__)


class Qwen27BDownProjectionKernel(RDNAHybridW4A16LinearKernel):
    """Retain native prefill/skinny, accelerate 3..9 row down projections.

    At K=17408, M>=3 exceeds the native HIP kernel's checked LDS limit.
    The second layout is limited to these 64 target layers (about 3.1 GiB),
    instead of duplicating every quantized weight in the model.
    """

    @classmethod
    def can_implement(cls, config):
        supported, reason = super().can_implement(config)
        if not supported:
            return supported, reason
        if not (on_gfx1151() and config.group_size == 32
                and config.act_type == torch.float16 and config.zero_points
                and tuple(config.partition_weight_shape) == (17408, 5120)):
            return False, "Requires gfx1151 FP16 asymmetric group32 Qwen27B down projection"
        return True, None

    def process_weights_after_loading(self, layer):
        raw = getattr(layer, self.w_q_name)
        unpacked = unpack_quantized_values_into_int32(
            raw.data, self.config.weight_type, packed_dim=raw.packed_dim)
        if getattr(raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()
        n, k = unpacked.shape
        shifts = torch.arange(8, device=unpacked.device, dtype=torch.int32) * 4
        kn = unpacked.t().contiguous().view(k, n//8, 8)
        packed = torch.sum((kn & 15) << shifts, dim=2, dtype=torch.int32).contiguous()
        del unpacked, kn
        super().process_weights_after_loading(layer)
        _, scales, zeros = self._get_weight_params(layer)
        layer.register_buffer("_qwen_kmajor_weight", packed, persistent=False)
        layer.register_buffer("_qwen_kmajor_scale", scales.data.t().contiguous(), persistent=False)
        layer.register_buffer("_qwen_kmajor_zero", zeros.data.t().contiguous(), persistent=False)
        logger.info_once("Qwen27B down projection: K-major verify M=3..9; other shapes use native RDNA")

    def apply_weights(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        if not 3 <= x2.shape[0] <= 9:
            return super().apply_weights(layer, x, bias)
        result = down_gemm(x2.contiguous(), layer._qwen_kmajor_weight,
                           layer._qwen_kmajor_scale, layer._qwen_kmajor_zero)
        if bias is not None:
            result.add_(bias)
        return result.reshape(x.shape[:-1] + (self.config.partition_weight_shape[1],))
