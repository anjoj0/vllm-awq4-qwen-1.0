# UCX #11299 + ROCm IPC peer-failure flag: stale registration trace

## 结论

本实验在 W7900 上对 OpenUCX #11299 头部叠加
`UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE`，并启用 UCX logging，复现一次
`stale_registration`：远端已经 deregister/释放显存，但 initiator 仍使用已导出的 rkey。
结果是 **initiator 35 s 超时，target 退出码 1**。trace 证明请求确实进入
`rocm_ipc/rocm_ipc`，在 `rma_send.c:310` 提交了 1 GiB `get_nbx`，协议选择为
`get/zcopy`，但之后没有完成回调、错误状态或 peer-disconnect。

这说明 capability flag 没有解决 stale exported-rkey 的生命周期违规；该问题并非
flag 新引入，因为此前 `#11299 + flag/peer` 和原版 `#11299/none` 都超时。但 flag
会让该风险从显式 `none` 路径进入 NIXL 默认 `peer` 数据 lane，因此在合并前必须明确
peer-failure contract，或由后续 ROCm-only cache/error recovery 改动保证失败请求完成
或返回错误。

## 实验配置

| 项目 | 值 |
|---|---|
| GPU | target GPU0 -> initiator GPU4（W7900，跨 NUMA） |
| 传输 | 1 GiB READ |
| 故障 | `stale_registration` |
| NIXL error mode | `peer` |
| UCX source | #11299 head `4dddf15e46735555405bf678be778a23358ec45f` + capability flag |
| UCX prefix | `/workspace/pd_disagg_20260802/deps/ucx-pr11299-peerflag-trace-20260807` |
| logging | `--enable-logging`, `UCX_LOG_LEVEL=data` |
| 超时 | 35 s 进程级硬超时 |
| ROCm | 容器 `rocm/vllm:rocm7.14.0_rdna...` |

## 结果

| 进程 | 返回 | 观察 |
|---|---:|---|
| target | 1 | 注销 registration 后等待 `test_done`，最终被 harness 清理 |
| initiator | 124 | `timeout(35 s)`；没有产生 `DONE`、`ERR` 或 `NIXL_ERR_REMOTE_DISCONNECT` JSON |

关键 initiator trace：

```text
149616 rma_send.c:310  get_nbx ... count 1073741824 ... cb (nil)
149618 proto_select.c:491 ... get(multi) into rocm/GPU4 from rocm/dev[0]
149795 proto_rndv.c:96   selected md rocm_ipc index 3
151047 proto_debug.c:112 rndv using zero-copy read from remote rocm_ipc/rocm_ipc
```

`ucx_info -d` 对 trace prefix 的 capability 输出为：

```text
Memory domain: rocm_ipc
  Transport: rocm_ipc
  get_zcopy: 128..inf
  put_zcopy: 128..inf
  error handling: peer failure
```

因此可以排除“没有选到 IPC”以及“请求回退到 TCP/CMA”。当前 trace 没有观察到
`rocm_ipc` 的错误返回或 UCP completion；请求停留在已经失效的远端 IPC
registration/rkey 对应的数据操作上。

## 与三路 stale 对照的关系

此前在相同 1 GiB、GPU0->GPU4 条件下：

| 配置 | 结果 |
|---|---|
| `#11299 + flag / peer` | initiator 超时，124 |
| `#11299 原版 / none` | initiator 超时，124 |
| `#11299 原版 / peer` | `rocm_copy_cache.c:140` Fatal；initiator 0.257 s 返回 `NIXL_ERR_REMOTE_DISCONNECT` |

trace 版把第一行的“超时”进一步定位为：peer endpoint 的 ROCm IPC RMA 路径已建立
并提交 zcopy，但 invalid rkey 没有被转化为 UCP 错误完成。原版 `peer` 的 host
fallback 虽然较快返回，却是 target abort，不是正确的 stale-rkey 恢复。因此这不是
简单删除 flag 就能解决的问题。

## 上游边界与建议

当前 capability patch 适合作为 W7900 兼容性/性能证据，不应宣称已经实现完整
peer-failure recovery。向维护者沟通时应将两件事分开：

- #11299：handle-cache 与 device-initiated PUT 的改动，可与 capability flag 机械叠加；
- ROCm-only follow-up：定义 stale/invalid IPC registration 的错误传播、请求取消和
  资源回收，确保失败操作在有限时间内返回错误。

在第二项没有实现和跨架构 CI 前，建议保持 RFC/issue 讨论，不提交生产 PR。

## 归档

完整 target（约 755 MiB 原始）和 initiator（约 18 MiB 原始）日志已压缩：

```text
results/nixl_ucx_pr11299_stale_trace_20260807.tgz
SHA256 D6906ADDDF2E2B1AC404A60C85AB1527C4CE7AE551A09CC6FA3886B6CACA8DC3
```

```bash
tar -xzf nixl_ucx_pr11299_stale_trace_20260807.tgz
grep -n 'rma_send.c:310' nixl_ucx_pr11299_stale_trace_20260807/*initiator.log
```
