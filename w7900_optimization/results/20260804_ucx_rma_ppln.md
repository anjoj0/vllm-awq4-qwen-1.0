# W7900 UCX RMA pipeline 复核

## 1. 动机

NIXL RFC [#2039](https://github.com/ai-dynamo/nixl/issues/2039) 中，ROCm 贡献者建议在 UCX 1.22 上设置 `UCX_RMA_PPLN_ENABLE=y`，以尝试让大块 RMA 采用 IPC 路径。本实验在同一 W7900、UCX、NIXL 和双进程门禁下对 READ/WRITE 分别做匹配 A/B，回答两个问题：该变量是否提高吞吐，以及 `UCX_PROTO_INFO` 是否确认 payload 进入直接 `rocm_ipc` lane。

## 2. 配置与方法

| 项目 | 配置 |
|---|---|
| GPU | Radeon PRO W7900，GPU0 到 GPU1 |
| ROCm | 7.14 |
| UCX | 1.22，commit `95865d6365b08b5c2b437be37375eac55f31533f` |
| NIXL | 1.4，UCX backend |
| Transport | `rocm_ipc,rocm_copy,self,tcp` |
| 采样 | 1 次 warmup + 3 次计时 |
| Payload | 64 MiB、1 GiB |
| 校验 | READ/WRITE 均在 initiator 与 target 校验首 4 KiB 和末字节 |

`nixl_rocm_two_gpu_bench.py` 新增 `--operation READ|WRITE`，确保测试的 PUT 是实际 payload WRITE，而不是 metadata 或 notification 中出现的内部控制操作。

## 3. 结果

| NIXL 操作 | Payload | 默认 UCX | `UCX_RMA_PPLN_ENABLE=y` | 提升 |
|---|---:|---:|---:|---:|
| READ | 64 MiB | 0.297 GB/s | 1.425 GB/s | 4.80x |
| READ | 1 GiB | 0.297 GB/s | 1.060 GB/s | 3.57x |
| WRITE | 64 MiB | 0.285 GB/s | 1.387 GB/s | 4.87x |
| WRITE | 1 GiB | 0.233 GB/s | 1.056 GB/s | 4.54x |

八个配置全部通过 payload 校验。该变量是有效的 UCX 缓解项：它将同节点大块 ROCm RMA 从约 0.23-0.30 GB/s 提高到约 1.06-1.43 GB/s。

## 4. 协议证据

1 GiB WRITE 的实际协议为：

```text
remote memory write by ucp_put*(multi) from rocm/GPU1 to rocm/dev[0]
rndv using pipeline rocm_copy, fenced write to remote, frag host,
rocm_copy, frag host | tcp/bond0
```

1 GiB READ 呈现相同的 host-staged pipeline。target 侧同时显示 `rendezvous data send/fetch` 经 `rocm_copy` 在 GPU 与 host fragment 之间搬运，远端段仍为 `tcp/bond0`。因此该变量启用的是分块流水和 GPU/host copy 重叠，而不是 W7900 上的直接 `rocm_ipc` 数据 lane。

## 5. 结论

- `UCX_RMA_PPLN_ENABLE=y` 应作为 W7900 UCX fallback profile 的推荐参数，它能带来 3.57-4.87 倍的稳定提升。
- 该结果不能描述为“UCX 已使用 HIP/ROCm IPC”。`UCX_PROTO_INFO` 明确保留 host fragments 和 `tcp/bond0`。
- 1 GiB pipeline 约为 1.06 GB/s；原生 HIP IPC 热态约 25.2 GiB/s（约 27.1 GB/s），数据面仍相差约 25 倍。
- 该优化降低了 UCX fallback 的严重程度，但没有消除独立 NIXL HIP IPC backend 的价值。
- 该结果已同步回复 [NIXL #2039](https://github.com/ai-dynamo/nixl/issues/2039#issuecomment-5180268003) 和 [ROCm/ucx #35](https://github.com/ROCm/ucx/issues/35#issuecomment-5180268737)。

完整 stdout/stderr 日志位于 `ucx_rma_ppln_20260804.tgz`，结构化数据位于 `20260804_ucx_rma_ppln_results.json`。
