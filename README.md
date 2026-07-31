# Qwen3.6-27B AWQ4 + DFlash on AMD RDNA 3/3.5

本仓库研究 Qwen3.6-27B AWQ4 在 AMD RDNA 3/3.5 上的端到端推理优化。系统从 Strix Halo 的统一内存单机路线扩展到 8× Radeon PRO W7900 离散显存节点，包含 W4A16 HIP 算子、DFlash 投机解码、Triton unified attention 调优、多卡部署和科研长文质量门禁。

项目并不将一种配置推广到所有负载，而是根据上下文长度、并发和显存约束进行路由：

- **Strix Halo / gfx1151**：AWQ4 压缩权重，利用 128 GB UMA 支持本地长上下文；DFlash 主要加速短上下文解码。
- **W7900 / gfx1100，8–16K**：RDNA3 W4A16 HIP 内核与 tile=16 attention 提升 prefill。
- **W7900 / 100K+**：切换到配套 BF16 TP=8 路线，避免 AWQ4 TP=4 长文退化。
- **W7900 / 并发服务**：模型能由 4 卡容纳时，可用双 TP=4 实例提高聚合吞吐。

## 核心技术

### 1. RDNA W4A16 HIP 内核

`csrc/` 保留 gfx1151 的 AWQ4/MMQ 与 3D Split-K 实现；`w7900_optimization/csrc/awq_mmq_gfx1100/` 提供 gfx1100 版本、正确性测试、数值验证和 microbenchmark。内核在计算时完成 INT4 解包与反量化，避免物化完整 BF16 权重。

W7900 实测表明 dispatcher 必须感知 shape 和上下文：HIP 相对 Triton 在 8K/16K prefill 分别达到 1.816×/2.220×，到 66K TP=4 时收益缩小为 1.284×。这支持“短中 prefill 用 HIP、超长文回到 BF16 多卡”的分段策略，而不是无条件替换所有 large-M。

### 2. gfx1100 unified attention 重调优

`w7900_optimization/patch_w7900.py` 将 prefill tile、2D/3D launch threshold 等参数改为环境变量，并检查源码中是否残留强制 gfx1151 的目标。tile 从 32 调整为 16 后：

| 指标 | tile=32 | tile=16 | 变化 |
|---|---:|---:|---:|
| VGPR | 224 | 176 | -21.4% |
| standalone kernel | 1021.95 ms | 998.00 ms | -2.3% |
| AWQ4 24K 服务 wall | baseline | -11.5% | 端到端改善 |

较小 tile 降低寄存器压力并改善并发 wave 条件；端到端收益还包含请求 shape、调度和其他 kernel 的联动，不能只由 standalone 时间推导。

### 3. DFlash 上下文感知路由

DFlash 用小型 drafter 产生候选 token，由目标模型并行验证。W7900 上 DFlash 在 8K 快 33.6%，12K 快 4.4%，16K 慢 6.6%，说明候选接受收益会随 verify 成本和上下文长度改变。推荐将其作为有边界的短上下文 profile，而不是全局开关。

### 4. 离散显存与多卡策略

W7900 的 KV cache 不再能依赖 UMA 容量。系统联合选择 TP、`gpu_memory_utilization`、最大模型长度和 KV dtype。FP8 KV 可用容量约为 auto KV 的 1.99×，但单请求更慢，故只在容量不足时启用。100K+ 单请求优先 BF16 TP=8；可分流的短请求优先多实例。

## 关键结果

### Strix Halo

| 配置 | 结果 |
|---|---:|
| BF16 decode | 4.3 token/s |
| AWQ4，无 DFlash | 5.6 token/s |
| AWQ4 + DFlash N=8 | 28.3 token/s peak，约 20 token/s mean |
| 三并发 | 41 token/s peak aggregate |
| gfx1151 HIP W4A16，4K/32K prefill | 约 3.4× / 2.8× |
| 3D Split-K，32K decode | 3.4 → 6.7 token/s |

### W7900

| 实验 | Triton/对照 | 优化配置 | 结果 |
|---|---:|---:|---:|
| AWQ4 8K prefill | 20.671 s | RDNA3 HIP 11.383 s | 1.816× |
| AWQ4 16K prefill | 58.451 s | RDNA3 HIP 26.332 s | 2.220× |
| AWQ4 66K TP=4 | 136.66 s | RDNA3 HIP 106.42 s | 1.284× |
| 102.9K TP 扩展 | TP=2 261.947 s | TP=8 67.977 s | 3.85× |
| 短文并发 | 单 TP=8 129.66 tok/s | 双 TP=4 159.66 tok/s | +23.1% |

在 102,994 与 128,769 tokens 上，BF16 TP=8 分别比 AWQ4 TP=4 快 5.662× 和 6.204×。该结果限定了 AWQ4 的最佳工作区间，也避免把权重压缩等同于所有场景的吞吐提升。

### 科研长文质量

| 配置 | 32K QA / wall | 64K QA / Needle / wall |
|---|---:|---:|
| BF16 TP=8 | 94.79% / 10.933 s | 96.67% / 100% / 22.674 s |
| AWQ4 TP=4 | 96.88% / 26.523 s | 88.33% / 75% / 63.304 s |

64K 门禁区分输出预算截断与真正的证据复制错误；详细案例见 [W7900 质量与 profiler 说明](docs/W7900_QUALITY_AND_ROCPROF.md)。

## 目录

```text
.
├── csrc/                         # Strix Halo gfx1151 HIP 算子
├── scripts/                      # Strix Halo 容器与服务脚本
├── test/                         # API、性能、正确性和精度测评
├── w7900_optimization/
│   ├── patch_w7900.py            # vLLM 回移与 gfx1100 参数化补丁
│   ├── csrc/awq_mmq_gfx1100/     # gfx1100 W4A16 HIP 源码与测试
│   ├── longdoc_sanity/            # 科研长文数据集、评分与回归门禁
│   └── scripts/                   # 构建、启动、benchmark、RCCL/功耗工具
└── docs/
    ├── EXPERIMENT_RESULTS.md
    ├── W7900_FULL_EXPERIMENT_REPORT.md
    ├── W7900_QUALITY_AND_ROCPROF.md
    └── assets/                    # 精选实验图
```

## 快速开始

### Strix Halo

```bash
cp .env.template .env
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/health
```

完整参数和兼容版本见 [原始 Strix Halo README](docs/STRIX_HALO_ORIGINAL_README.md)。

### W7900

实测环境是 AMD ROCm vLLM 0.23 容器、8× W7900/gfx1100。模型和 DFlash 权重不在仓库内。

```bash
cd w7900_optimization
cp .env.w7900.template .env.w7900
set -a; source .env.w7900; set +a

bash scripts/check_w7900_node.sh
bash scripts/prepare_local_vllm.sh
bash scripts/build_local_vllm.sh
```

`prepare_local_vllm.sh` 从容器 `/app/vllm` 建立可修改工作树，`patch_w7900.py` 以可重复方式应用：

- vLLM PR #45207（commit `55da232d`）Mamba page-padding 修复回移；
- Triton unified attention 的 gfx1100 参数化；
- 2D/3D launch threshold 参数化；
- gfx1151 强制目标检查。

构建 gfx1100 W4A16 扩展：

```bash
cd csrc/awq_mmq_gfx1100
PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
python test_correctness.py
python validate_prefill_numerics_gfx1100.py
```

启动参数、profile 与测评入口见 [W7900 README](w7900_optimization/README.md)。

## 实验复现原则

每次实验至少保存：代码 commit、ROCm/PyTorch/Triton/vLLM 版本、GPU/NUMA 拓扑、模型精度、KV dtype、TP、DFlash N、输入/输出 token 数、冷/热状态、wall time、吞吐和峰值显存。不要将模型权重、cache、原始大 trace 或凭据提交到 Git。

## 已知限制

- 当前容器同时存在 `_rocm_sdk_devel` 与 `_rocm_sdk_core` 两套 profiler SDK；真实 vLLM attach 会因 `librocprofiler-sdk.so.1` 路径冲突触发 signal 6。
- 已验证单进程 PID 分片和 `torchrun` TP=2 worker 的 kernel/RCCL trace，但尚未得到可信的真实请求内 W4A16、attention、RCCL 时间百分比。
- standalone kernel 与 RCCL microbenchmark 是解释性辅助数据，不能替代端到端 A/B。
- gfx1100 结果不应直接外推到 gfx1151；每个平台必须重新编译并重新选择 tile/dispatcher 边界。

## 上游与许可

项目基于 [vLLM](https://github.com/vllm-project/vllm)、AMD ROCm、PyTorch、Triton 和 RCCL，并参考/回移上游已经合并的修复。仓库自身代码按 [LICENSE](LICENSE) 发布，模型及上游组件遵循各自许可证。
