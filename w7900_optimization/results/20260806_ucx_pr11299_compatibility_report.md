# UCX PR #11299 与 ROCm IPC peer-error capability 的兼容性验证

## 摘要

NIXL #2039 的 AMD 维护者指出，OpenUCX PR [#11299](https://github.com/openucx/ucx/pull/11299) 正在修改 `rocm_ipc` 的 handle cache 与 device-initiated PUT，相关工作需要协调。本轮在 #11299 当前 head 上只叠加 `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE`，不修改其 cache、MD/EP 或 GPU kernel，并在 W7900 上完成构建、PR 自带 ROCm GTest、NIXL RMA 性能和跨 NUMA peer-exit 回归。

结果表明两项工作可以机械叠加：#11299 的 ROCm IPC 测试 `92 passed / 40 expected skipped / 0 failed`；NIXL 默认 `peer` error mode 下，同/跨 NUMA 的 1 GiB READ/WRITE 都选择 `rocm_ipc/rocm_ipc`，达到 `27.38/23.52 GB/s`，与未叠加 #11299 的 `develop + flag` 差异不超过 `0.12%`；12 次合法 peer-exit 故障注入全部得到明确完成或断连。

## 1. 代码基线与变更范围

| 项目 | 值 |
|---|---|
| #11299 head | `4dddf15e46735555405bf678be778a23358ec45f` |
| 本地兼容分支提交 | `a33495929` |
| 修改文件 | `src/uct/rocm/ipc/rocm_ipc_iface.c` |
| 修改内容 | 增加 `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE` |
| 未修改 | handle cache、IPC MD/EP、device-initiated PUT、HIP kernel、PR 测试 |
| 构建 | UCX 1.23.0，ROCm + CMA + GTest，独立 prefix |

叠加补丁仍为：

```diff
- iface_attr->cap.flags = UCT_IFACE_FLAG_GET_ZCOPY |
+ iface_attr->cap.flags = UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
+                         UCT_IFACE_FLAG_GET_ZCOPY |
```

#11299 在同一 flags 表中新增 `UCT_IFACE_FLAG_DEVICE_EP`，二者描述不同能力：`DEVICE_EP` 用于 GPU 发起操作；`ERRHANDLE_PEER_FAILURE` 允许 host/UCP 的 peer-error endpoint 把 `rocm_ipc` 选为 RMA lane。两项能力没有共享状态机或 cache 逻辑。

## 2. #11299 ROCm 专用 GTest

运行 PR 新增的 ROCm IPC RMA 与 device PUT 测试：

```bash
test/gtest/gtest \
  --gtest_filter='rocm_ipc/test_rocm_ipc_rma.*:rocm_ipc/test_rocm_ipc_rma_device.*'
```

| 总数 | 通过 | 跳过 | 失败 | 时间 |
|---:|---:|---:|---:|---:|
| 132 | 92 | 40 | 0 | 0.395 s |

4 个基础 RMA/device endpoint 用例全部通过。128 个 device PUT 参数化用例中，thread、合法 warp 和 block 级别全部通过；8 个 `warp nt1` 因 wavefront 至少需要 64 threads 按设计跳过，32 个 grid-level 用例因 PR 尚未支持 grid 按设计跳过。该结果比通用 UCT 单进程 capability 草案更有效，因为它真正实例化了 `rocm_ipc/rocm_ipc`。

## 3. NIXL host-initiated RMA

环境为 ROCm 7.14、NIXL 1.4、`UCX_TLS=sm,rocm,tcp,self`、`UCX_RMA_PPLN_ENABLE=y`、NIXL 默认 `peer` error mode。每项 1 次 warmup、3 次计时，1 GiB payload，initiator 与 target 双端校验。

| GPU 对 | NUMA | READ (GB/s) | WRITE (GB/s) | 协议 |
|---|---|---:|---:|---|
| 0--1 | 同 NUMA | 27.377 | 23.527 | `rocm_ipc/rocm_ipc` |
| 0--4 | 跨 NUMA | 27.391 | 23.522 | `rocm_ipc/rocm_ipc` |

与相同节点上 `ROCm/ucx develop + flag` 的独立构建相比：

| GPU 对 | READ 差异 | WRITE 差异 |
|---|---:|---:|
| 0--1 | -0.117% | -0.093% |
| 0--4 | +0.055% | -0.033% |

这些偏差低于单轮系统噪声，未观察到 #11299 cache/device-PUT 改动对 NIXL host-initiated RMA 的性能回归。

## 4. 跨 NUMA peer-exit 回归

GPU0 target、GPU4 initiator，每个场景重复 3 次：

| 场景 | 结果 | 时间范围 |
|---|---|---:|
| 传输前异常退出，1 GiB READ | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | 0.207--0.219 ms |
| 传输前正常退出，1 GiB READ | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | 0.154--0.231 ms |
| 提交后退出，8 GiB READ | 3/3 `DONE`，payload verified | 0.397--0.406 s |
| 提交后退出，8 GiB WRITE | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | 0.325--0.329 s |

没有出现挂起或静默错误，行为与未叠加 #11299 的 `develop + flag` 一致。`stale_registration` 仍不混入该回归：它违反 rkey 生命周期并涉及 #11299 正在调整的 cache，应由 AMD/OpenUCX 在 cache 设计内单独定义和验证。

## 5. 结论与边界

本轮可以支持三个具体结论：

1. #11299 没有包含 peer-error capability 修复，NIXL lane 排除问题在该分支仍需单独处理。
2. 在 W7900 上，一行 capability flag 可以干净叠加到 #11299，不修改 handle cache，也不影响其 device-initiated PUT 测试。
3. capability flag 恢复的 host-initiated `rocm_ipc` RMA 与 #11299 的 device-initiated 能力是互补关系，不需要建立第二套 NIXL transport。

该证据只覆盖 W7900/gfx1100。MI300/MI350、stale rkey 防御性恢复和 #11299 最终 rebase 仍需要 AMD/OpenUCX CI。合理的上游形态可以是 AMD 在 #11299 或后续提交中吸收该 flag，也可以是依赖 #11299 的独立最小提交；应由 ROCm IPC owner 决定，以免在 #11299 合并前产生重复协调成本。

原始性能、故障、GTest 和构建日志归档为 `w7900_20260806_ucx_pr11299_peerflag_raw.tgz`：

```text
SHA256 07B9B7DCD015B3764D77FDB4610109F53C083D909CA33BF95178BA29E2DEA41F
```

源码快照为 `ucx-pr11299-peerflag-a33495929.tgz`：

```text
SHA256 80C5462D909F6A0C20F51EB53B5E95065CFB20D696251B51880E93C85C279AB3
```
