# W7900 非对称 Prefill/Decode 资源划分实验

## 1. 实验动机

此前 Prefill/Decode 解耦只验证了物理 `4/4` 划分，即 Prefill TP=4、Decode TP=4。该配置能够隔离长 Prefill 对已进入 Decode 请求的干扰，但无法回答一个更一般的问题：8 张 W7900 应如何根据输入长度、输出长度与并发度，在两个阶段之间分配资源？

Qwen3.6-27B 的 attention/KV head 与线性层维度不适合 TP=6，因此物理 `2/6` 不能直接写成 Prefill TP=2、Decode TP=6。本实验把所有对照统一为 TP=2 副本，保持模型精度、TP 宽度和单卡显存布局不变，只改变 Prefill 与 Decode 的副本数：

| Profile | Prefill | Decode | 物理卡数 |
|---|---|---|---:|
| `p2_d6` | `1 x TP2` | `3 x TP2` | 2 + 6 |
| `p4_d4` | `2 x TP2` | `2 x TP2` | 4 + 4 |
| `p6_d2` | `3 x TP2` | `1 x TP2` | 6 + 2 |

这种设计避免将“副本数收益”和“TP 宽度变化”混在同一组对照中。请求由 vLLM 官方 toy proxy 在 Prefill 与 Decode 副本之间分别轮询；状态迁移使用 NIXL 1.4 与本项目 `W7900_HIP_IPC` backend。

## 2. 统一配置

| 项目 | 配置 |
|---|---|
| GPU | 8 x Radeon PRO W7900 48 GiB，gfx1100 |
| 模型 | Qwen3.6-27B BF16 |
| vLLM | 0.23.1.dev1，V1 engine |
| Attention | `TRITON_ATTN`，HND KV layout |
| TP | 每个副本均为 TP=2 |
| KV connector | `NixlConnector` + `W7900_HIP_IPC` |
| Block size | 128 |
| Prefix caching | 关闭 |
| 执行模式 | eager，custom all-reduce 关闭 |
| 64K KV dtype | BF16 (`auto`) |
| 128K KV dtype | FP8 |

每个 engine 使用独立 HTTP、distributed-init 和 NIXL side-channel 端口。启动器逐组加载模型并进行健康检查；某个 TP2 group 若在 RCCL rendezvous 阶段超时，只回收并重启该组。正式结果不包含模型加载、JIT warmup 和启动重试时间。

## 3. 64K 长输入、短输出

负载为 6 个并发请求，每个请求 64,000 输入 token、32 输出 token。三个 profile 的输入、采样参数和 TP 宽度完全一致。

| Profile | Batch wall | 聚合输入吞吐 | 聚合输出吞吐 | Mean TTFT | P95 TTFT | 单请求 Decode |
|---|---:|---:|---:|---:|---:|---:|
| `p2_d6` | 642.43 s | 597.73 tok/s | 0.299 tok/s | 373.22 s | 613.95 s | 18.24-18.52 tok/s |
| `p4_d4` | 321.48 s | 1194.47 tok/s | 0.597 tok/s | 212.34 s | 319.19 s | 18.37-18.53 tok/s |
| `p6_d2` | **218.66 s** | **1756.12 tok/s** | **0.878 tok/s** | **160.54 s** | **216.21 s** | 16.93-18.55 tok/s |

结果呈现近似离散时隙结构。`p2_d6` 只有一个 Prefill 副本，六个请求约以 106.7 秒为间隔依次获得首 token；`p4_d4` 有两个 Prefill 副本，wall 几乎严格减半；`p6_d2` 有三个 Prefill 副本，六个请求分两轮完成。

相对 `p2_d6`，`p4_d4` 的 batch wall 降低 50.0%，`p6_d2` 降低 66.0%。与此同时，各配置的单请求稳定 Decode 速度接近，说明 32-token 输出尚不足以使 Decode 副本数成为主瓶颈。该工作负载下，把 6 张卡分给 Decode 会造成显著资源失衡；`6/2` 是三种划分中的最优点。

## 4. 8K 高并发、长输出

为提高 Decode 压力，使用 12 个并发请求、8,000 输入 token，并把输出从 512 token 增加到 2,048 token。

### 4.1 512-token 输出

| Profile | Batch wall | 聚合输入吞吐 | 聚合输出吞吐 | Mean TTFT | P95 TTFT | 单请求 Decode |
|---|---:|---:|---:|---:|---:|---:|
| `p2_d6` | 102.88 s | 933.12 tok/s | 59.72 tok/s | 56.85 s | 75.02 s | 18.01-19.35 tok/s |
| `p6_d2` | **93.78 s** | **1023.70 tok/s** | **65.52 tok/s** | **29.21 s** | **58.47 s** | 11.30-17.24 tok/s |

即使输出增长到 512 token，`p2_d6` 仍未胜出。单 Prefill 的排队抵消了三个 Decode 副本的收益；另一方面，`p6_d2` 的单 Decode 能够对多请求进行批量生成，虽然每请求速度下降，但节点聚合吞吐仍更高。这说明“增加 Decode 副本”不能仅依据并发数决定，还需要比较 Prefill 排队时间和批量 Decode 效率。

### 4.2 2,048-token 输出

| Profile | Batch wall | 聚合输入吞吐 | 聚合输出吞吐 | Mean TTFT | P95 TTFT | 单请求 Decode |
|---|---:|---:|---:|---:|---:|---:|
| `p2_d6` | **188.31 s** | **509.81 tok/s** | **130.51 tok/s** | 56.37 s | 75.38 s | **18.14-18.71 tok/s** |
| `p6_d2` | 212.49 s | 451.79 tok/s | 115.66 tok/s | **17.71 s** | **24.83 s** | 10.79-11.23 tok/s |

输出增加到 2,048 token 后，资源最优点发生反转：`p2_d6` 相对 `p6_d2` 将 batch wall 降低 11.4%，聚合输出吞吐提高 12.8%。三个 Decode 副本各自保持约 18 tok/s，而 `p6_d2` 的单个 Decode 副本需要同时批处理 12 个长生成请求，单请求速度降至约 11 tok/s。另一方面，`p6_d2` 仍具有更低 TTFT，因为三个 Prefill 副本消除了输入排队。因此，`2/6` 优化的是长生成阶段的节点完工时间和总吞吐，`6/2` 优化的是首 token 延迟；两者对应不同服务目标。

## 5. 128K 容量边界

TP2 每卡权重约 25.7 GiB。128K profile 使用 FP8 KV，并分别验证了 activation workspace、chunk 大小与 worker watchdog 的联合边界。

| 配置 | 结果 | 原因 |
|---|---|---|
| `max_model_len=131072`，batch budget=131072 | 初始化失败 | 可用 KV cache 为 `-0.43 GiB`，一次性 prefill workspace 过大 |
| batch budget=32768，RPC timeout=300 s | 请求失败 | 处理到 65,536 token 后，下一块 32K 的单次 model RPC 超过 300 s |
| batch budget=16384，RPC timeout=600 s | **成功** | 单块 attention 工作量受控，8 个 chunk 完成 128K |

最终成功点为 `p6_d2`、3 并发、128,000 输入 token、32 输出 token：

| Batch wall | 聚合输入吞吐 | Mean TTFT | P95 TTFT | 单请求 Decode |
|---:|---:|---:|---:|---:|
| 898.29 s | 427.48 tok/s | 892.29 s | 895.62 s | 12.66-13.30 tok/s |

该结果证明 TP2+FP8 KV 具备 128K 容量，但约 15 分钟 TTFT 表明它不是低时延配置。极长上下文不能只用“模型和 KV 是否放得下”判断可行性；chunk budget 必须同时满足显存 workspace 和单次 kernel/watchdog 上限。

## 6. HIP IPC 与多 Decode fan-out

`p2_d6` 的 8K 三并发功能门禁中，一个 Prefill producer 同时与三个 Decode consumer 建立连接。三个 Decode 均通过 TP2 compatibility hash，分别接收一个请求，输出文本一致。每个请求每 rank 平均迁移约 325.4 MB，三组实测 HIP IPC 吞吐约 15.6-16.4 GB/s，传输与通知失败为 0。

backend 的最终源码重编译后，独立门禁复用同一 1 GiB prepared handle 三次，冷态/热态带宽为 14.77/25.19/25.24 GiB/s。活动请求的立即 repost 返回 `NIXL_ERR_REPOST_ACTIVE`，且原请求仍正确完成；故意制造超大通知失败后，故障轮 payload 正确，同一 handle 能重建 stream/event 并恢复成功。这些结果表明非对称矩阵中的瓶颈来自阶段算力比例，而不是 fan-out 数据面错误。

## 7. 结论

W7900 八卡节点不存在固定最优的 `4/4` 划分。资源比例应由 Prefill 服务时间、Decode 总 token 数和 Decode 批处理效率共同决定：

- 64K 输入、32-token 短输出明显由 Prefill 主导，`6/2` 相对 `2/6` 将 wall 降低 66.0%。
- 8K、12 并发、512-token 输出仍由 Prefill 排队与批量 Decode共同决定，三个 Decode 副本没有带来节点吞吐优势。
- 输出增加到 2,048 token 后，最优资源比例从 `6/2` 反转为 `2/6`：batch wall 降低 11.4%，聚合输出吞吐提高 12.8%，证明长生成负载可以摊薄单 Prefill 排队并从 Decode 副本扩展中获益；若目标是最低 TTFT，`6/2` 仍更优。
- 128K 在 TP2+FP8 KV 下容量可行，但必须使用较小 chunk 和更长 worker watchdog；其 TTFT 约 15 分钟，应视为容量模式而非默认服务模式。

原始 JSON、worker/proxy 日志、启动 manifest 与失败日志位于 `20260804_asymmetric_pd_raw/`。
