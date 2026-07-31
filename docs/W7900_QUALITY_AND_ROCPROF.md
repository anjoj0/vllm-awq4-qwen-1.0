# W7900 跨精度科研长文质量与 rocprof 多进程验证记录

## 1. 实验目的

本轮只回答两个会影响技术报告结论的问题：

1. RDNA3 W4A16 后端接入后，AWQ4 TP=4 在 32K–64K 科研长文中是否仍保持可接受的回答质量，以及它是否适合作为长文主速度路线；
2. 当前 ROCm 7.14 容器能否用 `rocprofv3` 获取多进程 kernel/RCCL trace，并进一步覆盖真实 vLLM 请求。

质量实验使用同一 Nowcast3D evidence/needle/numeric/abstention harness。BF16 固定 TP=8、auto KV、`TRITON_ATTN`；AWQ4 固定 TP=4、auto KV、`TRITON_ATTN`、`enforce-eager`，四个 worker 日志均确认：

```text
Using RDNA3W4A16LinearKernel for CompressedTensorsWNA16
```

## 2. 32K 跨精度门禁

| 配置 | 完成 | QA | JSON | Citation | Evidence | Source | Needle | Mean wall | Mean TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 TP=8 | 8/8 | 94.79% | 100% | 100% | 100% | 85.71% | 100% | 10.933 s | 8.371 s |
| AWQ4 RDNA3 TP=4 | 8/8 | 96.88% | 100% | 100% | 100% | 85.71% | 100% | 26.523 s | 19.938 s |

32K 小样本中没有观察到 AWQ4 的质量下降，但 AWQ4 wall time 为 BF16 的 `2.43×`。这说明低比特路径在少卡显存占用方面有价值，却不能仅凭权重压缩推导出长 prefill 更快。

## 3. 64K 冻结协议结果

两组均使用 15-case core suite 和 256-token 输出上限。

| 配置 | 完成 | QA | JSON | Citation | Evidence | Source | Needle | Mean wall | Mean TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 TP=8 | 15/15 | 96.67% | 100% | 96.15% | 96.15% | 92.31% | 100% | 22.674 s | 19.253 s |
| AWQ4 RDNA3 TP=4 | 15/15 | 88.33% | 93.33% | 88.46% | 88.46% | 84.62% | 75% | 63.304 s | 56.440 s |

AWQ4 原始协议的 QA 保持率为 `91.38%`，wall time 为 BF16 的 `2.79×`。因此，32K 的“质量持平”不能直接外推到 64K；同样，AWQ4 的容量优势也不能外推成单请求时延优势。

## 4. 失败归因与定向复测

原始 64K 结果同时包含评测协议因素和真实低比特边界，不能用单一总分解释：

- `n3d_fact_006_two_stages` 与 `n3d_synthesis_014_training_protocol` 在 BF16、AWQ4 中均有严格关键词或逐字 evidence 评分损失，属于共享评分边界；
- `n3d_fact_010_metrics` 的 AWQ4 回答因 256-token 上限被截断，`finish_reason=length`，不构成内容能力失败；
- `needle_64k_late` 位于约 90% 位置，AWQ4 两次都把正确编号 `ROCM-RADAR-327138` 写成 `ROCM-RADAR-32713`，属于可重复的精确字符串退化。

将指标题输出上限单独提高到 512 token 后，该题恢复为 QA `4/4`，JSON、Citation、Evidence、Source 均为 `100%`。但 64K needle 整组复测仍为 `75%`，90% 位置的编号错误再次出现。因此正式结论同时保留：

1. 冻结 256-token 协议的原始结果，用于可复现实验比较；
2. 512-token 定向复测，用于剥离输出截断因素；
3. 重复 needle 失败，用于标记 AWQ4 在科研编号、标识符和引用键上的真实精确性边界。

这一结果进一步支持：100K+ 科研长文的主速度和高精确性路线仍为 BF16 TP=8；AWQ4 TP=2/4 更适合少卡容量、部署密度和中短上下文，而不是替代全部长文 profile。

## 5. rocprof PID 分片验证

旧脚本让多个 worker 共享固定 `--output-file` 和 summary 文件，存在并发写入争用。本轮改为：

```text
默认 %hostname%/%pid% 输出
--process-sync false
CSV 输出
每个进程独立 kernel/RCCL trace
```

结果如下：

- 单进程 launch 成功生成非空 `kernel_trace.csv` 与 `kernel_stats.csv`；
- `torchrun` TP=2 成功为两个 worker PID `94373`、`94374` 分别生成非空 kernel trace、kernel stats、RCCL API trace 和 RCCL API stats；
- TP=2 microbenchmark 中，1 MiB All-Reduce 平均 `0.128 ms`、16 MiB 平均 `0.844 ms`，算法带宽分别为 `8.19 GB/s` 和 `19.87 GB/s`；
- rank 0 的受采样 kernel 时间中 RCCL device kernel 占 `96.73%`。该百分比只描述 All-Reduce microbenchmark，不能解释为真实 vLLM 请求的通信占比。

这证明当前发行版的多进程落盘问题已经解决，PID 分片方法可以用于普通 PyTorch/RCCL 程序。

## 6. 真实 vLLM trace 的兼容边界

完整 vLLM launch tracing 在模型加载后停在 KV cache 初始化附近；即使设置延迟采集窗口，EngineCore 仍长期高 CPU、GPU 空闲，服务无法就绪。对照的非 profiler 服务约 87 秒正常就绪，表明阻塞来自 profiler 注入而不是模型或 W4A16 后端。

attach 诊断得到更具体的原因：

1. 不设置 `ROCP_TOOL_ATTACH=1` 时，目标没有 `rocp-bg-attach` 线程，attach 明确拒绝；
2. 设置该变量后，attach 能进入注册阶段，但容器内 `_rocm_sdk_devel` 与 `_rocm_sdk_core` 各有一套 `librocprofiler-sdk.so.1`；
3. EngineCore 已注册 `devel` 路径，而注入端尝试使用 `core` 路径，注册器报错后目标线程收到 signal 6，vLLM 随 EngineCore 退出；
4. 显式设置 attach 进程的 `ROCPROFILER_REGISTER_LIBRARY` 仍不能改变注入端解析到 `core` 路径的行为。

关键日志为：

```text
ROCPROFILER_REGISTER_LIBRARY is already set to
.../_rocm_sdk_devel/lib/librocprofiler-sdk.so.1,
not overriding with
.../_rocm_sdk_core/lib/librocprofiler-sdk.so.1

process ... killed by signal 6
rocprofiler_register_attach ... status: 6
```

该现象与 `rocm-systems` PR #8361 所修复的目标 ELF/注册库解析问题一致。当前结论是：通用 PID 分片 launch 已可用，但本镜像不能可靠获得真实 vLLM 请求内 W4A16、attention 与 RCCL 时间占比。报告继续采用 standalone 内核证据和端到端 A/B，且不把 microbenchmark 百分比冒充服务占比。

## 7. 复现实验材料

本地归档：

```text
w7900_incremental_quality_rocprof_20260730.tgz
w7900_incremental_quality_rocprof_20260730/
```

其中包含 4 组主质量结果、2 组定向复测、BF16/AWQ4 服务日志、单进程与 TP=2 PID 分片 trace，以及所有 vLLM launch/attach 失败诊断。用于重试的脚本位于：

```text
.remote_patch/profile_awq4_tp1_pid_split_driver.sh
.remote_patch/run_rocprof_awq4_tp1_pid_split.sh
.remote_patch/run_rocprof_attach_engine.sh
```

实验结束时 8 张 GPU 的 VRAM 占用均为 `0%`，容器保持运行，未执行 `docker stop`。
