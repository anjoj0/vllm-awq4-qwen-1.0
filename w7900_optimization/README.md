# W7900 AWQ4 optimization

该目录是 Qwen3.6-27B AWQ4 + DFlash 在 8× Radeon PRO W7900（gfx1100）上的代码、启动 profile 与测评工具。它对应的是离散显存多卡系统，不复用 Strix Halo 的 UMA 参数假设。

## 已实现

- `patch_w7900.py`：对 vLLM 0.23 工作树做幂等补丁，回移 PR #45207 / commit `55da232d`，开放 unified attention tile 和 2D/3D launch 参数，并检查 gfx1151 硬编码。
- `../Dockerfile.w7900`：继承实测 ROCm 7.14/vLLM 0.23 AMD 镜像，以 `gfx1100` 重建工作树；根目录 `Dockerfile` 仍专用于 Strix Halo。
- `vllm_overrides/rdna3_w4a16.py`：整模型 `RDNA3W4A16LinearKernel` 的非对称 `compressed-tensors` dispatcher override；它不同于下方 standalone HIP MMQ 研究扩展。
- `csrc/awq_mmq_gfx1100/`：面向 gfx1100 的 W4A16 HIP 内核、Python binding、正确性测试与 prefill benchmark。
- `longdoc_sanity/`：Nowcast3D 主题科研长文数据集，包含证据、数字、needle、引用和拒答测评。
- `scripts/`：本地 vLLM 构建、单/多卡服务、长文与并发 harness、RCCL 和功耗辅助工具。

## 推荐路由

| 负载 | 推荐路线 | 依据 |
|---|---|---|
| 8–16K AWQ4 prefill | gfx1100 HIP W4A16 | 相对 Triton 1.816×–2.220× |
| 约 66K AWQ4 TP=4 | HIP 可选，必须 A/B | 收益缩小到 1.284× |
| 100K+ 单请求 | BF16 TP=8 | 比 AWQ4 TP=4 快 5.662×–6.204× |
| 8K DFlash | 启用 | wall time 降 33.6% |
| 12K DFlash | 视延迟目标启用 | 仅快 4.4% |
| 16K+ DFlash | 默认关闭 | 16K 实测慢 6.6% |
| KV 容量不足 | FP8 KV | 容量约 1.99×，但延迟更高 |

## 构建

```bash
cp .env.w7900.template .env.w7900
set -a; source .env.w7900; set +a
bash scripts/check_w7900_node.sh
bash scripts/prepare_local_vllm.sh
bash scripts/build_local_vllm.sh
```

若要从 Dockerfile 构建独立 W7900 镜像，使用 `.env.w7900.docker.template` 和仓库根目录的 `Dockerfile.w7900`，不要使用旧的 Strix Halo 根 Dockerfile。

当前容器流程从 `/app/vllm` 复制到 `/workspace/vllm-w7900-023` 后修改；独立 Docker 镜像流程复制到 `/opt/vllm-w7900-023`。两条路径都避免直接污染基础源码，且所有架构目标均固定为 `gfx1100`。

单独构建 gfx1100 算子：

```bash
cd csrc/awq_mmq_gfx1100
PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
python test_correctness.py
python validate_prefill_numerics_gfx1100.py
python benchmark_prefill_gfx1100.py
```

## 启动与测评

通用 AWQ4 服务使用：

```bash
bash scripts/start_local_vllm.sh
```

容量 profile：

```bash
bash scripts/start_awq4_tp8_capacity_w7900.sh
```

长文质量门禁：

```bash
cd longdoc_sanity
python validate_suite.py
python run_longdoc_sanity.py --help
python score_longdoc_sanity.py --help
```

正式实验应先 health check、warmup，再至少重复三次热态请求。保存 tokenizer 后的实际输入长度，不以源文件字节数代替 token 数。

## 结果索引

- [凝练实验结果](../docs/EXPERIMENT_RESULTS.md)
- [W7900 第一阶段完整报告](../docs/W7900_FULL_EXPERIMENT_REPORT.md)
- [科研长文质量与 rocprof 边界](../docs/W7900_QUALITY_AND_ROCPROF.md)
- [图表](../docs/assets/)

## Profiler 边界

单进程 PID 分片与 `torchrun` TP=2 kernel/RCCL trace 已成功。当前容器两套 profiler SDK 的动态库冲突使真实 vLLM attach 触发 signal 6，因此现有报告没有宣称真实请求内 kernel 百分比；只使用 standalone、microbenchmark 和端到端 A/B 形成相互约束的证据。
