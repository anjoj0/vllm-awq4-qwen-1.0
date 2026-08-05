# NIXL #2039 `sm` 与原生 UCX ROCm IPC 复核

## 1. 实验动机

NIXL RFC [#2039](https://github.com/ai-dynamo/nixl/issues/2039) 的维护者提出两项关键判断：

1. `UCX_TLS=rocm_ipc,rocm_copy,self,tcp` 缺少同节点握手需要的共享内存 transport，应改为 `UCX_TLS=sm,rocm,tcp,self`；
2. ROCm IPC 已由 UCX 实现，应优先修复 UCX 或 NIXL 的现有路径，而不是增加新的 NIXL transport plugin。

本次实验依次回答：加入 `sm` 后 NIXL 是否选择直接 `rocm_ipc`，以及同一 UCX 构建绕过 NIXL 后能否在两张 W7900 之间使用 `rocm_ipc`。

## 2. 环境与方法

| 项目 | 配置 |
|---|---|
| GPU | 8× Radeon PRO W7900 48 GiB，实验使用物理 GPU0 与 GPU1 |
| ROCm | 7.14 |
| UCX | 1.22，commit `95865d6365b08b5c2b437be37375eac55f31533f` |
| NIXL | 1.4，UCX backend |
| TLS | `sm,rocm,tcp,self` |
| NIXL 负载 | 1 GiB，1 次 warmup + 3 次计时，READ/WRITE 双端数值校验 |
| UCX 负载 | `ucx_perftest` UCP GET/PUT，64 MiB×10 与 1 GiB×3 |

纯 UCX 实验使用以下 HSA 可见顺序：

```text
server: ROCR_VISIBLE_DEVICES=0,1
client: ROCR_VISIBLE_DEVICES=1,0
```

UCX ROCm allocator 从可见列表的第一个 GPU 分配，因此两端分别使用物理 GPU0 和 GPU1；同时保留对端 GPU 可见，使 HSA IPC attach 仍然有效。

## 3. NIXL 加入 `sm` 的结果

| NIXL 操作 | Pipeline | 1 GiB 吞吐 | `UCX_PROTO_INFO` 数据路径 |
|---|---:|---:|---|
| READ | 关闭 | 0.246 GB/s | software emulation over `tcp/bond0` |
| WRITE | 关闭 | 0.279 GB/s | software emulation over `tcp/bond0` |
| READ | 开启 | 3.844 GB/s | `rocm_copy` + host fragments + `cma/memory` |
| WRITE | 开启 | 3.475 GB/s | `rocm_copy` + host fragments + `cma/memory` |

四组实验均通过 initiator 和 target payload 校验。与此前不含 `sm` 的 pipeline 结果相比，1 GiB READ 从 1.060 提高到 3.844 GB/s，WRITE 从 1.056 提高到 3.475 GB/s。`sm` 将同节点 host relay 从 TCP 改为 CMA，显著改善 fallback，但没有让 NIXL payload 进入直接 `rocm_ipc`。

## 4. 纯 UCX 跨卡对照

| UCP 操作 | Payload | 吞吐 | `Config` |
|---|---:|---:|---|
| GET | 64 MiB | 18.016 GB/s | `rocm_ipc/rocm_ipc` |
| PUT | 64 MiB | 13.530 GB/s | `rocm_ipc/rocm_ipc` |
| GET | 1 GiB | 18.243 GB/s | `rocm_ipc/rocm_ipc` |
| PUT | 1 GiB | 11.261 GB/s | `rocm_ipc/rocm_ipc` |

GET 的协议区间为：

```text
remote memory read by ucp_get*(multi) into rocm/GPU0 from rocm/dev[0]
128..inf | zero-copy | rocm_ipc/rocm_ipc
```

PUT 在 `UCX_RMA_PPLN_ENABLE=y` 下同样选择：

```text
remote memory write by ucp_put*(multi) from rocm/GPU0 to rocm/dev[0]
128..inf | rndv using zero-copy read from remote | rocm_ipc/rocm_ipc
```

因此，UCX 1.22 的 ROCm IPC transport 在 W7900 跨卡环境中可以工作。此前的缺口不能再归因于 UCX 缺少 ROCm IPC 实现。

## 5. NIXL 路径的初步定位

`UCX_LOG_LEVEL=data` 显示 NIXL 进程已经完成以下步骤：

```text
register rocm memory on: rocm_cpy, rocm_ipc
created interface using rocm_ipc/rocm_ipc
```

日志中的 `same process` 拒绝发生在 UCX 自动创建的 memtype/self endpoint，属于正常行为。真正的跨进程 `intra-node cfg#2` 只包含：

```text
cma/memory  : rma_bw
tcp/bond0   : am, keepalive
```

该 endpoint 的远端地址没有形成可配对的 `rocm_ipc` lane。因此当前根因范围已经从“W7900/UCX 不支持 IPC”收缩到“NIXL UCX worker address 交换或 endpoint 构建没有保留 ROCm IPC lane”。精确代码位置仍需结合 NIXL 维护者对 worker-address 生命周期的判断。

## 6. 对上游建议的处理结论

- 接受 `sm` 建议：UCX fallback profile 改为 `sm,rocm,tcp,self`，并在 UCX 1.22 上启用 `UCX_RMA_PPLN_ENABLE=y`。
- 接受避免新 plugin 的建议：不再把独立 `HIP_IPC` plugin 作为 NIXL 上游首选方案。
- 将 `W7900_HIP_IPC` 保留为性能上界、数据正确性和生命周期回归原型。
- 将后续上游工作聚焦于 NIXL UCX backend 的 worker-address/endpoint 集成。
- 接受对 NIXL PR #1536 的澄清：该 PR 只强化 `ucx_vram_memtype_hint` 的注册语义，与 UCP RMA/rendezvous 协议选择相互独立。

复现实验脚本为：

- `pd_disaggregation/run_ucx_sm_matrix.sh`
- `pd_disaggregation/run_ucx_perftest_rocm.sh`

原始协议日志归档为 `ucx_sm_followup_20260805.tgz`。
