# Qwen3.6-27B AWQ4 + DFlash on AMD RDNA 3/3.5

本仓库研究 Qwen3.6-27B AWQ4 在 AMD RDNA 3/3.5 上的端到端推理优化。系统从 Strix Halo 的统一内存单机路线扩展到 8× Radeon PRO W7900 离散显存节点，包含 W4A16 HIP 算子、DFlash 投机解码、Triton unified attention 调优、多卡部署和科研长文质量门禁。

项目并不将一种配置推广到所有负载，而是根据上下文长度、并发和显存约束进行路由：

- **Strix Halo / gfx1151**：AWQ4 压缩权重，利用 128 GB UMA 支持本地长上下文；DFlash 主要加速短上下文解码。
- **W7900 / gfx1100，短中上下文**：RDNA3 W4A16 HIP、shape-aware attention 与 DFlash/target-only 路由协同降低时延。
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

DFlash 用小型 drafter 产生候选 token，由目标模型并行验证。恢复 checkpoint 的 `4 x SWA(2048) + 1 x full` 语义后，相对五层 full attention，8K、16K、32K wall time 分别下降 11.6%、18.2%、21.0%。在同一优化后端上，N=4 相对 target-only 在 8K 快 27.3%、12K 快 7.7%，到 16K 已变慢，因此系统使用 14K 作为保守交叉阈值：单请求且 `<=14K` 进入 DFlash N=4，长请求或 batch 进入 target-only。

DFlash 的 `N+1` small-query/长 KV attention 与普通长 prefill 属于不同 shape 域。前者在 gfx1100 上选择 `tile=32, warps=4`，后者保留 `tile=16`。D-Cut 和 full-layer recent-window 压缩均已实现并通过正确性门禁，但端到端分别退化约 1%–2% 和 6.7%–17.7%，因此作为研究开关保留、默认关闭。

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
| DFlash drafter 32K | 5× full 59.93 s | 4×SWA+1×full 47.35 s | -21.0% |
| 8K 自适应路由 | target-only 13.19 s | DFlash N=4 9.59 s | -27.3% |

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
├── expo_assistant/               # 多模态 Agent 工作台、项目工具、搜索/天气与阶段测速
├── w7900_optimization/
│   ├── patch_w7900.py            # vLLM 回移与 gfx1100 参数化补丁
│   ├── adaptive_dflash_router.py  # 双 TP=4 上下文感知路由
│   ├── patches/                   # 固定 vLLM 基线的完整 DFlash 补丁
│   ├── csrc/awq_mmq_gfx1100/     # gfx1100 W4A16 HIP 源码与测试
│   ├── longdoc_sanity/            # 科研长文数据集、评分与回归门禁
│   ├── results/                   # 五路线报告、CSV、图和原始日志归档
│   └── scripts/                   # 构建、启动、benchmark、RCCL/功耗工具
└── docs/
    ├── EXPERIMENT_RESULTS.md
    ├── W7900_FULL_EXPERIMENT_REPORT.md
    ├── W7900_DFLASH_MULTICARD_PREFIX_CACHE_PD_DISAGGREGATION.md
    ├── W7900_QUALITY_AND_ROCPROF.md
    └── assets/                    # 精选实验图
```

DSpark / DFlash2 的候选权重、运行时兼容性、双后端设计与 AMD 验证计划见 [专项评估](docs/DSpark与DFlash2_替换DFlash可行性_双后端与开源方案评估_20261022.md)。

## 快速开始

### Strix Halo

当前这台机器的开机启动、模型验收、工作台启动、停止和排错顺序见 [本机启动指南](启动指南.md)。

默认构建已固定为官方 `vllm/vllm-openai-rocm:v0.32.0`，使用其配套依赖；新参数、DSpark 切换、可选项目 HIP MMQ 和回退入口见 [v0.32.0 迁移与验收](docs/STRIX_HALO_VLLM_0320_MIGRATION.md)。2026-10-26 已完成本机 target-only、DFlash、DSpark full/49k、DFlash2 的接口验收及真实展示页接入；前述 Strix Halo 性能表仍为旧版实测。

本机 `/home/amd/hf-cache` 的权重审计、固定 snapshot 配置、可选 DFlash2、实测选择与启动入口见 [Strix Halo 本地部署说明](docs/STRIX_HALO_LOCAL_DEPLOYMENT.md)。

2026-10-27 的形状适配已将同一 Qwen3.6 AWQ4 的无缓存 4K / 8K prefill 提高到 **372 / 363 token/s**；DFlash N4 的短输入代码 / 中文 decode 达到 **27.5 / 17.4 token/s**，8K 编程请求的 decode 从 **15.8 提高到 22.6 token/s**。这些是最终默认配置的固定请求实测，完整条件与历史对照见 [本轮记录](test/results/strix_halo_20261026/README.md)。DFlash2 的 FP16 溢出已在本机定位，BF16 恢复有限值与接受能力；不能再将此前近零接受率当作算法排名。

```bash
cp .env.template .env
docker compose build
docker compose run --rm --no-deps vllm --check-env
docker compose up -d
curl http://127.0.0.1:8000/health
```

旧版参数和历史测试见 [原始 Strix Halo README](docs/STRIX_HALO_ORIGINAL_README.md)；旧版构建使用 `Dockerfile.legacy-v020` 和 `docker-compose.legacy-v020.yml`。

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

如需构建独立 Docker 镜像，请使用仓库根目录的 `Dockerfile.w7900` 和 `w7900_optimization/docker-compose.w7900.build.yml`。它直接继承实测的 ROCm 7.14/vLLM 0.23 镜像，并以 `gfx1100` 重编译；根目录的 `Dockerfile` 和 `docker-compose.yml` 仍然是 Strix Halo/gfx1151 路线。

构建 gfx1100 W4A16 扩展：

```bash
cd csrc/awq_mmq_gfx1100
PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
python test_correctness.py
python validate_prefill_numerics_gfx1100.py
```

启动参数、profile 与测评入口见 [W7900 README](w7900_optimization/README.md)。

### 本地 Agent 工作台

模型服务启动后，可以运行 [Qwen 本地工作台](expo_assistant/README.md)。页面复用已有推理 API，支持项目文件读写、图片与文档上传、工具循环、联网搜索、城市天气，以及会话保存与性能统计。左侧有四个带素材和校验点的体验案例。Python 3.11+；依赖安装在独立环境，前端无需构建。

```bash
bash expo_assistant/run.sh --upstream http://127.0.0.1:8000 --model Qwen3.6-27B-AWQ4
```

浏览器打开 `http://127.0.0.1:8088`。性能面板的阶段速率对应最后一次模型调用，总耗时包含整个任务；数据缺失或其他请求混入时显示空值。`bash expo_assistant/run.sh --demo` 可预览界面，模拟模式不显示测速。本机真实模型已通过文件读写、图片与文档联合理解、搜索引用与天气工具验收；使用范围与启动说明见工作台文档。

## 实验复现原则

每次实验至少保存：代码 commit、ROCm/PyTorch/Triton/vLLM 版本、GPU/NUMA 拓扑、模型精度、KV dtype、TP、DFlash N、输入/输出 token 数、冷/热状态、wall time、吞吐和峰值显存。不要将模型权重、cache、原始大 trace 或凭据提交到 Git。

## 已知限制

- 旧研究容器存在两套 profiler SDK 的路径冲突；新版本机 Torch profiler 仅获得 CPU/API 事件，rocprof attach 也未成功。此次改用请求内 HIP events 测量目标模块范围，不能将其解读为逐 GPU kernel 的完整归因。
- W7900 已验证单进程 PID 分片和 `torchrun` TP=2 worker 的 kernel/RCCL trace，其结论与本机单卡模块计时分别记录。
- standalone kernel 与 RCCL microbenchmark 是解释性辅助数据，不能替代端到端 A/B。
- gfx1100 结果不应直接外推到 gfx1151；每个平台必须重新编译并重新选择 tile/dispatcher 边界。

## 上游与许可

项目基于 [vLLM](https://github.com/vllm-project/vllm)、AMD ROCm、PyTorch、Triton 和 RCCL，并参考/回移上游已经合并的修复。仓库自身代码按 [LICENSE](LICENSE) 发布，模型及上游组件遵循各自许可证。
