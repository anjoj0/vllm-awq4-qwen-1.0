# W7900 HIP IPC 与原生 NIXL backend 实验报告

## 1. 实验动机

Qwen3.6-27B 是 Attention 与 Mamba/GDN 混合模型。Prefill/Decode 解耦不仅要迁移 Attention KV，还要迁移 Mamba convolution state 与 SSM state。原始 vLLM `NixlConnector` 已能正确管理 scheduler handshake、TP rank 映射、KV block lease 和通知，但在 Radeon PRO W7900/Navi31 上，NIXL/UCX 的大块 GPU RMA 被选择为 `software emulation | tcp/bond0`。64K 输入时每个 TP rank 约有 1.02 GiB 状态需要迁移，TCP 回退使单 rank 平均传输耗时接近 3.9 s。

本实验的目标不是重写 P/D 调度，而是替换同节点 GPU payload 数据面，同时保持 vLLM 原生 `NixlConnector` 的控制语义和可观测性。

## 2. 实现

实现按三层门禁推进：

1. 裸 HIP IPC：验证 W7900 间的 `hipIpcGetMemHandle`、`hipIpcOpenMemHandle` 和双向 D2D copy。
2. vLLM 专用 facade：保留 NIXL 控制面，只替换 prepared GPU READ，用于快速验证真实混合 KV/SSM payload。
3. NIXL 动态 backend：实现 `W7900_HIP_IPC` plugin，通过原生 `NixlConnector` 配置加载，移除 Python facade。

正式 backend 只声明同 Linux 主机的 `VRAM_SEG`、`READ` 和 `WRITE`。它使用 `hipMemGetAddressRange` 将切片地址回溯到 PyTorch caching allocator 的底层 allocation；远端 handle 采用共享缓存；连续 descriptor 自动合并；HIP event 记录 GPU copy 时间；Unix domain datagram 承载 bundled notification。

```text
vLLM NixlConnector
  ├─ scheduler / metadata / TP mapping / block lease / metrics
  └─ W7900_HIP_IPC backend
       ├─ HIP IPC: Attention KV + Mamba conv/SSM payload
       └─ Unix datagram: completion notification
```

## 3. 环境与口径

| 项目 | 配置 |
|---|---|
| GPU | 8 × Radeon PRO W7900 48 GiB，`gfx1100` |
| 模型 | Qwen3.6-27B BF16 |
| 软件 | ROCm 7.14、PyTorch 2.11、vLLM 0.23.1.dev1、NIXL 1.4.0 |
| Prefill | GPU 0-3，TP=4 |
| Decode | GPU 4-7，TP=4 |
| Attention | Triton，tile=16 |
| KV | BF16/auto，block size 128 |
| 服务模式 | eager，prefix cache 关闭 |
| 长文负载 | 64,000 input tokens，32 output tokens |

UCX/TCP、facade 和原生 plugin 使用相同模型、TP、KV、attention 和服务参数。单请求比较采用热态数据；并发 4 比较采用同一 benchmark harness。

## 4. 分层正确性与带宽

| 门禁 | 大小 | 结果 | 带宽 |
|---|---:|---|---:|
| 裸 HIP IPC GET | 1 GiB | `valid=true` | 27.16 GB/s |
| 裸 HIP IPC PUT | 1 GiB | `valid=true` | 27.30 GB/s |
| 跨进程 HIP IPC | 1 GiB | `valid=true` | 25.12 GiB/s |
| 原生 NIXL plugin | 64 MiB | `valid=true` | 12.72 GiB/s |
| 原生 NIXL plugin | 1 GiB | `valid=true` | 25.12 GiB/s |

NIXL plugin 的 1 GiB 带宽与裸跨进程 HIP IPC 相同，表明 NIXL backend 抽象没有引入明显的大块吞吐损失。8K vLLM P/D 短请求与直接 Decode 的 greedy 输出逐字一致；4 个 Prefill worker 和 4 个 Decode worker 均确认加载 `W7900_HIP_IPC`。

## 5. 端到端结果

### 5.1 8K 单请求

| 数据面 | TTFT | Wall | Decode |
|---|---:|---:|---:|
| UCX/TCP | 4.15-4.21 s | 5.85-5.91 s | 约 19-20 tok/s |
| HIP IPC facade | 3.504-3.515 s | 5.117-5.123 s | 约 19-20 tok/s |
| 原生 NIXL plugin r1 | 3.508 s | 5.156 s | 19.42 tok/s |
| 原生 NIXL plugin r2 | 3.508 s | 5.145 s | 19.55 tok/s |
| 原生 NIXL plugin r3 | 3.510 s | 5.134 s | 19.71 tok/s |

原生 plugin 与 facade 的 TTFT 基本重合，并均明显优于 UCX/TCP。

### 5.2 64K 单请求

| 路径/轮次 | TTFT | Wall | Decode |
|---|---:|---:|---:|
| UCX/TCP 热态基线 | 59.931 s | 61.524 s | 20.09 tok/s |
| facade r2/r3 均值 | 55.736 s | 57.319 s | 20.21 tok/s |
| plugin r1 | 56.583 s | 58.250 s | 19.20 tok/s |
| plugin r2 | 55.667 s | 57.334 s | 19.19 tok/s |
| plugin r3 | 55.598 s | 57.249 s | 19.39 tok/s |
| plugin r2/r3 均值 | **55.633 s** | **57.292 s** | **19.29 tok/s** |

相对 UCX/TCP，原生 plugin 热态 TTFT 下降约 `7.2%`，wall 下降约 `6.9%`。plugin 与 facade 的差异小于 0.2%，因此性能收益来自 HIP IPC 数据面，而不是 facade 中的额外调度逻辑。

单轮 64K 的 4 个 rank 共传输 `4,381,802,496 bytes`，NIXL 记录的 HIP event 累计时间为 `0.17347 s`，有效 payload 带宽约 `25.26 GB/s`。832 个原始 descriptor 在 backend 内按连续区间合并后提交。

### 5.3 64K 并发 4

| 数据面 | Batch wall | Mean TTFT |
|---|---:|---:|
| UCX/TCP | 229.934 s | 144.757 s |
| HIP IPC facade | 225.488 s | 139.738 s |
| 原生 NIXL plugin | **225.259 s** | **139.922 s** |

相对 UCX/TCP，原生 plugin 的 batch wall 下降约 `2.0%`，mean TTFT 下降约 `3.3%`。收益小于单请求，是因为 4 个 64K Prefill 在该调度口径下主要串行排队；传输后端只消除每个请求约 4 s 的 TCP payload 时间，不能消除 Prefill 计算与队列等待。

三轮单请求和一轮并发 4 合计：

| 指标 | 数值 |
|---|---:|
| Rank transfers | 28 |
| Payload bytes | 30,672,617,472 |
| 原始 descriptors | 5,824 |
| HIP event 时间总和 | 1.445616 s |
| Failed transfers | 0 |
| Failed notifications | 0 |

## 6. 结论

`W7900_HIP_IPC` 证明了 W7900 上的限制来自当前 NIXL/UCX transport selection，而不是 Navi31 的同节点 peer copy 能力。将 HIP IPC 封装成 NIXL backend 后，vLLM 无需修改 P/D scheduler 即可获得约 25 GB/s 的真实 GPU payload 数据面，并在 64K 单请求上把 TTFT 降低约 7.2%。原生 plugin 与过渡 facade 性能重合，完成了从概念门禁到 vLLM 正式接口的闭环。

该结果不意味着 P/D 在所有负载上优于多实例。固定并发、相同 8 卡资源下，dual TP=4 仍有更高聚合吞吐；P/D 的主要价值是 Prefill/Decode 阶段隔离、独立扩缩容和服务尾延迟治理。`W7900_HIP_IPC` 解决的是其中同节点状态迁移的数据面瓶颈。

当前 backend 仅支持同主机 `VRAM_SEG`，request handle 为单次提交语义。跨节点传输和通用 request repost 不在本次能力声明中。

## 7. 证据索引

- 源码与构建：`../hip_ipc_transport/`
- 原生启动 profile：`../pd_disaggregation/start_*_tp4_nixl_hip_ipc.sh`
- 原始 JSON、metrics 与日志：`20260803_hip_ipc_raw.tgz`
- E2E 性能测试插件 SHA-256：`7b25a2f1c4e6708056e38fdf0db3d375016a0c608106aa4b5afdf8b6fcae6155`
- 增加 `releaseReqH()` device 恢复后的最终插件 SHA-256：`ec7d9e434ab17adc956bc5d9adf82c0b3f70becdbb4da163d5c3eff05bbf192c`；最终版 64 MiB 双进程门禁为 `valid=true`、`13.56 GiB/s`
- 原始证据包 SHA-256：`a7265102a94ffcf6e3958b91e6876276da7e40609b249b2b8378802843c25d6f`
