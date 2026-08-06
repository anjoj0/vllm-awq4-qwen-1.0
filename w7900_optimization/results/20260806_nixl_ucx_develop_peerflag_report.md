# ROCm/UCX `develop` 分支上的 W7900 IPC capability 验证

## 摘要

本轮实验把此前在 UCX 1.22 上得到的单行 capability 修复移植到当前 ROCm/ucx `develop` 分支，并在同一台 8 卡 Radeon PRO W7900 节点上重新完成性能、拓扑和故障回归。补丁仍然只有：让 `rocm_ipc` 声明 `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE`。在 NIXL 默认 `peer` error mode 下，GPU0--1 和 GPU0--4 的 1 GiB GPU RMA 均恢复为 `rocm_ipc/rocm_ipc`，READ 约 `27.38--27.41 GB/s`，WRITE 约 `23.53--23.55 GB/s`，双端 payload 全部校验通过。

与原 UCX `peer` 路径相比，READ/WRITE 加速约 `5.27x/4.75x`（同 NUMA）和 `5.73x/5.76x`（跨 NUMA）。这说明修复依赖的是 UCP endpoint capability 筛选语义，而非某一对 PCIe 设备的偶然行为。

## 1. 实验对象与构建

| 项目 | 值 |
|---|---|
| GPU | 8 x AMD Radeon PRO W7900，48 GiB/GPU |
| 拓扑 | GPU0--3 属于 NUMA0，GPU4--7 属于 NUMA1 |
| ROCm | 7.14 |
| UCX 源码 | ROCm/ucx `develop`，基线 `c2567cbe5075f3aabab13731aabdc164959004ec` |
| 补丁提交 | `61f01ab`，`UCT/ROCM: advertise peer failure support for IPC` |
| 构建产物 | UCX 1.23.0，ROCm + CMA + GTest，独立 prefix |
| NIXL | 1.4 |
| TLS | `sm,rocm,tcp,self` |
| RMA pipeline | `UCX_RMA_PPLN_ENABLE=y` |

构建使用 [build_ucx_rocm.sh](../pd_disaggregation/build_ucx_rocm.sh)，源码、安装 prefix 和日志均位于远端 `/workspace/pd_disagg_20260802/` 下的带 `20260806-v3` 后缀目录。构建日志中的 `ucx_info -v` 报告 API/library version `1.23.0`。

## 2. 补丁与根因

NIXL UCX backend 默认创建 `UCP_ERR_HANDLING_MODE_PEER` endpoint。原始 `rocm_ipc` 只报告 `error handling: none`，于是 UCP 在 endpoint lane 选择阶段排除它，GPU RMA 回退到 ROCm-host-CMA/TCP-ROCm。

补丁只改动 `src/uct/rocm/ipc/rocm_ipc_iface.c`：

```diff
- iface_attr->cap.flags = UCT_IFACE_FLAG_GET_ZCOPY |
+ iface_attr->cap.flags = UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
+                         UCT_IFACE_FLAG_GET_ZCOPY |
```

它不实现新的数据传输、不改变 rkey 生命周期，也不要求 `rocm_ipc` 自己提供 AM/keepalive。控制面仍可由 TCP/CMA lane 承担，IPC lane 只负责 RMA 数据面；这与 CUDA IPC 先例的多 lane 设计一致。

## 3. 最新 `develop` 性能复测

每项使用 1 次 warmup、3 次计时，payload 为 1 GiB，READ/WRITE 均执行 initiator 与 target 双端校验。

| GPU 对 | NUMA | READ (GB/s) | WRITE (GB/s) | 大消息协议 |
|---|---|---:|---:|---|
| 0--1 | 同 NUMA | 27.409 | 23.549 | `rocm_ipc/rocm_ipc` |
| 0--4 | 跨 NUMA | 27.376 | 23.530 | `rocm_ipc/rocm_ipc` |

协议日志显示，小于约 128 B 的控制/短消息仍使用 TCP/CMA；大消息 RMA 使用 `rocm_ipc/rocm_ipc`。跨 NUMA 相对同 NUMA 的差异为 READ `0.12%`、WRITE `0.08%`，在本节点上未观察到可归因于 NUMA 的 IPC 数据面退化。

此前 UCX 1.22 原始 `peer` 对照为：GPU0--1 READ/WRITE `5.203/4.962 GB/s`，GPU0--4 为 `4.775/4.082 GB/s`。因此最新分支上的修复相对原 `peer` 的加速分别为：

| GPU 对 | READ | WRITE |
|---|---:|---:|
| 0--1 | 5.27x | 4.75x |
| 0--4 | 5.73x | 5.76x |

## 4. 跨 NUMA peer-failure 回归

GPU0（target）与 GPU4（initiator）上各场景重复 3 次。测试进程的退出码 `42/43` 是故障注入器主动产生，不是 UCX 崩溃。

| 场景 | 结果 | 时间 |
|---|---|---:|
| 传输前异常退出，1 GiB READ | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | `0.142--0.214 ms` |
| 传输前正常退出，1 GiB READ | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | `0.148--0.221 ms` |
| 提交 8 GiB 后退出，READ | 3/3 `DONE`，3/3 payload verified | `0.402--0.409 s` |
| 提交 8 GiB 后退出，WRITE | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | `0.323--0.329 s` |

这组结果说明 capability flag 不会把跨 NUMA 的 peer 退出变成静默成功，也没有引入等待超时。已经提交且由目标端可完成的 READ 会正常结束；目标端在 WRITE 仍需要存活时退出，则 initiator 得到明确断连。

`stale_registration` 不纳入该 capability 的正确性结论：远端在 rkey 仍可能被访问时注销 registration 违反 UCX rkey 生命周期约束，原始 `none` 路径也会暴露该问题。它应另立错误传播/资源生命周期议题。

## 5. GTest 边界

为了验证 capability 草案，曾运行：

```bash
test/gtest/gtest --gtest_filter='*rocm_ipc_peer_failure_capability*'
```

命令返回 0，报告 8 个实例无失败；但 stdout 明确显示 8 个实例全部因为 `!has_transport("rocm_ipc")` skip。原因是通用 UCT fixture 不能在同一进程内为 `rocm_ipc` 创建可连接 endpoint，因此默认资源集合不枚举该 transport。这个测试不是有效回归，已从上游分支移除，不应在 PR 中声称为测试覆盖。

当前有效证据是 NIXL 双进程 harness：`run_nixl_visibility_matrix.sh`、`run_nixl_peer_failure_matrix.sh` 和 `nixl_rocm_peer_failure_test.py`。上游测试应新增多进程 fixture，或者把 capability 检查放入能直接打开 `rocm_ipc` iface、但不要求同进程 endpoint 连接的 ROCm 专用测试；组合 endpoint 还应覆盖 TCP/CMA AM/keepalive + `rocm_ipc` RMA。

## 6. 复现

```bash
source /workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/activate_pd_env.sh
UCX_ROOT=/workspace/pd_disagg_20260802/deps/ucx-develop-peer-flag-20260806-v3 \
TARGET_DEVICE=0 INITIATOR_DEVICE=4 BYTES=$((1024*1024*1024)) ITERATIONS=3 \
CASE_SET=hip_single_READ \
bash /workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/run_nixl_visibility_matrix.sh

UCX_ROOT=/workspace/pd_disagg_20260802/deps/ucx-develop-peer-flag-20260806-v3 \
TARGET_DEVICE=0 INITIATOR_DEVICE=4 CASE_SET=all \
bash /workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/pd_disaggregation/run_nixl_peer_failure_matrix.sh
```

原始日志与构建日志归档为 [w7900_20260806_ucx_develop_peerflag_raw.tgz](w7900_20260806_ucx_develop_peerflag_raw.tgz)，SHA256：

```text
FEBF233293E6CF183CE7DC9149BAF6380EB15D33E1389401A9726158624943A6
```

精确源码快照为 [ucx-rocm-peer-failure-61f01ab.tgz](ucx-rocm-peer-failure-61f01ab.tgz)，SHA256：

```text
723C5C4100528017E861D16B5C8EAEDF37EAFB1FF8911383CB7371076E61841B
```

## 7. 可提交范围

当前提交适合作为 ROCm/ucx RFC 或 draft PR 的最小实现：一个 capability 声明、W7900/gfx1100 的性能和合法 peer-exit 证据，以及明确的测试空白。不能据此声称 MI300/MI350 已验证，也不能把 `rocm_ipc` 当作跨主机 RDMA 或 UCX 控制面的替代品。下一次上游沟通应请求维护者确认 capability 语义，并讨论多进程/多 lane 回归 fixture，而不是继续扩大单机性能矩阵。
