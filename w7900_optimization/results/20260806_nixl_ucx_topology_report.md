# W7900 UCX ROCm IPC 拓扑与故障回归补充

## 目的

此前的 1 GiB NIXL 基准只覆盖 GPU0 到 GPU1。由于 W7900 节点包含两个 NUMA 域，本轮验证三个问题：

1. `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE` 是否只对单一 PCIe 对有效；
2. 同 NUMA 与跨 NUMA 是否都能选择 `rocm_ipc/rocm_ipc`；
3. 恢复直连路径后，合法的 peer 退出是否仍保持明确的完成或断连语义。

所有测量使用 UCX 1.22、ROCm 7.14、NIXL 1.4，`UCX_TLS=sm,rocm,tcp,self`、`UCX_RMA_PPLN_ENABLE=y`，1 次 warmup + 3 次计时，1 GiB GPU buffer，READ/WRITE 均做双端 payload 校验。容器内拓扑显示 GPU0--3 属于 NUMA0，GPU4--7 属于 NUMA1。

## 带宽结果

| GPU 对 | NUMA 关系 | 原版 `peer` READ/WRITE (GB/s) | 实验 flag `peer` READ/WRITE (GB/s) | 原版 `none` READ/WRITE (GB/s) |
|---|---|---:|---:|---:|
| 0↔1 | 同 NUMA | 5.203 / 4.962 | 27.378 / 23.546 | 27.399 / 23.600（既有结果） |
| 4↔5 | 同 NUMA | 5.065 / 4.910 | 27.386 / 23.535 | 未补测 |
| 0↔4 | 跨 NUMA | 4.775 / 4.082 | 27.394 / 23.528 | 27.400 / 23.576 |

实验 flag 相对原版 `peer` 的加速比分别为：

| GPU 对 | READ | WRITE |
|---|---:|---:|
| 0↔1 | 5.262x | 4.746x |
| 4↔5 | 5.407x | 4.793x |
| 0↔4 | 5.737x | 5.764x |

三种拓扑的实验 flag 结果都落在 READ `27.378--27.394 GB/s`、WRITE `23.528--23.546 GB/s`，拓扑间差异分别小于 0.06% 和 0.08%。跨 NUMA 并没有破坏 ROCm IPC 数据面；在本节点上，性能更接近 PCIe/IPC 数据路径上限，而不是主机 CMA/TCP 回退。

协议日志对三种 GPU 对均显示大消息采用 `rocm_ipc/rocm_ipc`，而控制面仍由 TCP/CMA lane 承担。这正是 capability flag 的预期作用：让 IPC RMA lane 参与 peer-error endpoint 的数据 lane 筛选，不要求 IPC 自身承担 keepalive。

## 跨 NUMA 故障回归

在 GPU0↔GPU4 上以实验 UCX 重复三轮：

| 场景 | 结果 |
|---|---|
| 传输前异常退出 | 3/3 返回 `NIXL_ERR_REMOTE_DISCONNECT`，平均约 0.197 ms |
| 传输前正常退出 | 3/3 返回 `NIXL_ERR_REMOTE_DISCONNECT`，平均约 0.179 ms |
| 提交 8 GiB READ 后退出 | 3/3 `DONE`，3/3 payload verified，平均 0.406 s |
| 提交 8 GiB WRITE 后退出 | 3/3 返回 `NIXL_ERR_REMOTE_DISCONNECT`，平均 0.326 s |

这组结果与 GPU0↔GPU1 的既有故障矩阵一致，说明 flag 没有把跨 GPU 数据 lane 变成不可观测的“静默成功”。同时，stale registration 仍不应放入该 capability 修复：它违反远端 rkey 生命周期约束，原版 `none` 路径也会超时，应单独跟踪为错误传播问题。

## 结论与上游边界

本轮把“一行 capability 修改在 W7900 上有效”的证据从单个 GPU 对扩展到两个 NUMA 域和跨 NUMA GPU 对，并补充了 12 次跨 NUMA 故障注入。当前可以有把握地提交以下主张：

- UCX 现有 `rocm_ipc` transport 在 W7900 上具备跨 NUMA GPU 的 RMA 能力；
- `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE` 使 NIXL 默认 `peer` 模式能够恢复 IPC 数据 lane，性能接近显式 `none`；
- 控制面与数据面可由不同 lane 组成：TCP/CMA 负责 AM/keepalive，ROCm IPC 负责 RMA；
- 结果只代表 W7900/gfx1100 与当前 UCX 1.22 构建，尚不能替代 MI300/MI350 等架构上的 CI 验证。

复现脚本和 JSON 见同目录。原始日志已归档为 `20260806_nixl_ucx_topology_raw.tgz`，SHA256 为 `0832286DBF70A4ADF093BBA3C508F9CE8EBC7AFD39A64061836A2D9D7737AC47`；机器上的原始日志未被覆盖。
