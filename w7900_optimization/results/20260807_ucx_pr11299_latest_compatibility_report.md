# OpenUCX #11299 最新 head 的 W7900 兼容性复测

## 摘要

OpenUCX PR #11299 从此前验证的 `4dddf15e4` 更新并 squash/rebase 为
`57556bb87`。本轮从该最新 head 建立全新源码、构建目录和安装前缀，
只叠加两项独立改动：

1. `rocm_ipc` 声明 `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE`；
2. OpenUCX #11743 的负 HSA completion signal 错误传播与单测。

组合分支完成 out-of-source 开发构建。ROCm focused GTest 为
`93 passed / 40 expected skipped / 0 failed`，其中 #11299 的原有 132 个
ROCm IPC 用例仍为 `92/40/0`，新增 signal 回归通过。NIXL 默认 `peer`
模式下，同/跨 NUMA 的 1 GiB READ/WRITE 四项均选择
`rocm_ipc/rocm_ipc`，双端 payload 全部校验通过。跨 NUMA 12 次合法
peer-exit 故障注入全部得到明确完成或断连，没有挂起。

## 代码与构建

| 项目 | 值 |
|---|---|
| #11299 最新 head | `57556bb87d6ee3caf8178f1fa28b8cac44b46f4d` |
| capability 提交 | `65b63f850` |
| #11743 组合提交 | `016825a43` |
| 构建 | `contrib/configure-devel`，ROCm、verbs、dm、GTest |
| 安装 | 独立 prefix，未覆盖任何已有 UCX |

#11743 的 `Makefile.am` 变更与 #11299 新增的 HIPCC 测试列表落在同一段，
cherry-pick 时发生文本冲突。解决方式只是同时保留
`test_kernels_uct.hip`、`test_rocm_ipc_device.hip` 和
`test_rocm_signal.cc` 三个测试源；运行时代码没有冲突。

## Focused GTest

```text
filter:
  test_rocm_signal.async_error
  rocm_ipc/test_rocm_ipc_rma.*
  rocm_ipc/test_rocm_ipc_rma_device.*

133 tests from 3 suites
93 passed
40 expected skipped
0 failed
```

40 个 skip 与旧 head 相同：8 个 wavefront 线程数不足的 warp 用例，以及
32 个尚未支持的 grid-level 用例。

## NIXL 1 GiB 性能

环境为 NIXL 1.4、`UCX_TLS=sm,rocm,tcp,self`、
`UCX_RMA_PPLN_ENABLE=y` 和默认 `peer` error mode。每项 1 次 warmup、
3 次计时。

| GPU 对 | NUMA | READ | WRITE | 数据协议 |
|---|---|---:|---:|---|
| 0--1 | 同 NUMA | 27.385 GB/s | 23.518 GB/s | `rocm_ipc/rocm_ipc` |
| 0--4 | 跨 NUMA | 27.598 GB/s | 23.544 GB/s | `rocm_ipc/rocm_ipc` |

四项 initiator/target payload 校验全部通过。跨 NUMA READ 的 3 次计时
离散略高，但最慢一次仍为约 27.36 GB/s；没有出现协议回退或功能错误。

## 跨 NUMA peer-exit 回归

GPU0 target、GPU4 initiator，每个场景重复 3 次：

| 场景 | 结果 | 时间范围 |
|---|---|---:|
| 传输前异常退出，1 GiB READ | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | 0.177--0.250 ms |
| 传输前正常退出，1 GiB READ | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | 0.157--0.233 ms |
| 提交后退出，8 GiB READ | 3/3 `DONE`，payload verified | 0.397--0.400 s |
| 提交后退出，8 GiB WRITE | 3/3 `NIXL_ERR_REMOTE_DISCONNECT` | 0.320--0.324 s |

因此，#11299 最新 head 仍与 capability flag 和 #11743 机械兼容；host
发起 RMA 的性能与合法 peer-exit 行为没有回归。

## 边界

本轮不改变此前对 stale registration 的判断。若 exporter 在 importer
首次 attach 前释放已经发布的 IPC handle，纯 HSA reproducer 可在
`hsa_amd_ipc_memory_attach()` 内永久阻塞；该问题已跟踪在
ROCm/rocm-systems#9827。capability flag 仍不能单独视为完整
peer-failure contract，generation/retirement handshake 仍应由 NIXL 或
上层 connector 防止新传输使用已退役 metadata。

## 归档

- 源码快照：`ucx-pr11299-latest-peerflag-signal-016825a43.tar.gz`
  - SHA256 `B58120978FAFA36FBDCEC8F61ADACAA32BC1ED74C77BAB3D3A58125EC58D2ED3`
- 构建、GTest、性能和故障日志：
  `ucx_pr11299_latest_016825a43_results_20260807.tgz`
  - SHA256 `14EAD5C5E5E26509C610D3736ADC9A95A2B64893BF24B1394286E70EED1F6870`
