# NIXL/UCX ROCm IPC error mode 根因与修复边界

## 摘要

W7900 同节点 GPU RMA 过去在 NIXL 默认配置下只能走 ROCm-GPU 到 host、CMA/TCP、再到 ROCm-GPU 的中转路径。严格单变量实验表明，这不是 GPU visibility、worker address 丢失或 UCX 缺少 ROCm IPC，而是 endpoint error mode 与 transport capability 不匹配：

```text
NIXL default: UCP_ERR_HANDLING_MODE_PEER
UCX rocm_ipc: error handling = none
结果: rocm_ipc lane 在 endpoint 选择阶段被排除
```

将 NIXL backend 的 error mode 改为 `none` 后，1 GiB READ/WRITE 从 `5.203/4.962 GB/s` 提高到 `27.399/23.600 GB/s`。在 UCX `rocm_ipc` 上实验性增加 `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE` 后，NIXL 保持默认 `peer` 也恢复到 `27.382/23.512 GB/s`，相对 `none` 仅差 `0.06%/0.37%`，且 READ/WRITE payload 校验通过。

该 flag 已证明是性能路径的充分条件，但目前仍是研究性补丁。合法的 peer 退出场景未挂死；失效 rkey 场景仍会挂起，不过远端在 descriptor 仍可能被使用时注销 registration 本身违反 UCX 内存句柄生命周期约束，应作为防御性错误恢复缺口单独处理。

## 1. 环境与控制变量

| 项目 | 配置 |
|---|---|
| GPU | 8 x Radeon PRO W7900 48 GiB；实验使用物理 GPU0 与 GPU1 |
| ROCm | 7.14 |
| UCX | 1.22.0，ROCm/CMA enabled |
| NIXL | 1.4，UCX backend |
| TLS | `sm,rocm,tcp,self` |
| RMA pipeline | `UCX_RMA_PPLN_ENABLE=y` |
| 性能方法 | 1 次 warmup + 3 次计时，READ/WRITE 双端 payload 校验 |
| GPU 映射 | target `HIP_VISIBLE_DEVICES=0`，initiator `HIP_VISIBLE_DEVICES=1` |

GPU visibility 另以 `HIP_VISIBLE_DEVICES`、单值 `ROCR_VISIBLE_DEVICES` 和反序双值 `ROCR_VISIBLE_DEVICES` 三种方式交叉验证。默认 `peer` 下三者都不能选择 `rocm_ipc`，排除了 visibility 配置根因。

## 2. 源码与设备能力证据

UCX 1.22 和当前 master 的 `src/uct/rocm/ipc/rocm_ipc_iface.c` 均只声明：

```c
UCT_IFACE_FLAG_GET_ZCOPY |
UCT_IFACE_FLAG_PUT_ZCOPY |
UCT_IFACE_FLAG_PENDING |
UCT_IFACE_FLAG_CONNECT_TO_IFACE
```

未声明：

```c
UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE
```

设备查询与 NIXL 默认值相互印证：

```text
ucx_info -d: rocm_ipc error handling: none
ucx_info -d: cma error handling: peer failure, ep_check
NIXL UCX backend default: peer
```

因此，UCP 为 peer-error endpoint 选择数据 lane 时会排除 `rocm_ipc`。CUDA IPC 已声明该 flag，这也解释了相同 NIXL 配置在 CUDA 与 ROCm 上的路径差异。

### 2.1 CUDA IPC 的上游设计先例

OpenUCX PR [#9751](https://github.com/openucx/ucx/pull/9751) 在 2024 年删除了 `cuda_ipc` 自身的 `EP_CHECK` 实现，但有意保留 `UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE`。该 PR 的设计说明是：UCP 在 lane selection 中只选择一条独立 keepalive lane；IPC RMA 数据 lane 不需要同时提供 keepalive。可作为 keepalive 的 transport 需满足下列条件之一：

1. `CONNECT_TO_EP + EP_CHECK`；
2. `CONNECT_TO_IFACE + AM_BCOPY`；
3. `CONNECT_TO_EP + EP_KEEPALIVE`。

当前 W7900 NIXL endpoint 正好形成相同的多 lane 结构：TCP 负责 AM/keepalive，实验性 `rocm_ipc` 负责 RMA。因此，“ROCm IPC 本身没有 `EP_CHECK`”并不自动否定 peer-failure capability；更合适的回归目标是验证组合 endpoint 在 peer 退出时完成或报错，而不是要求纯 `rocm_ipc` 独立承担控制面 keepalive。

这一先例不能替代 ROCm 的运行时验证，但说明实验补丁与 CUDA IPC 的现有 UCX 语义一致，而不是创造新的 capability 解释。

## 3. 直接 UCP 单变量复现

`ucx_perftest` 的 `-e` 明确定义为“create endpoints with error handling support”。其他变量保持不变，64 MiB 结果如下：

| UCP 操作 | Error mode | 吞吐 | 数据路径 |
|---|---|---:|---|
| GET | none | 17.991 GB/s | `rocm_ipc/rocm_ipc` |
| PUT | none | 13.543 GB/s | `rocm_ipc/rocm_ipc` |
| GET | peer (`-e`) | 0.269 GB/s | `tcp/bond0` |
| PUT | peer (`-e`) | 0.275 GB/s | `tcp/bond0` |

开启 `-e` 后 GET/PUT 分别慢约 `67.0x/49.3x`。该实验完全绕过 NIXL，证明 transport capability 筛选是 UCX 层行为，不是 NIXL metadata 交换缺陷。

需要区分的是，通用 `ucx_perftest` endpoint 还要求 active-message lane；`rocm_ipc` 是纯 RMA transport，没有 `AM_BCOPY`。因此只允许 `rocm_ipc,rocm_copy,self` 时会得到：

```text
no active messages transport:
rocm_ipc/rocm_ipc - no am bcopy
```

真实 NIXL endpoint 会组合 TCP/CMA 控制 lane 与 `rocm_ipc` RMA 数据 lane，不能用强制单 transport 的失败否定该组合路径。

## 4. NIXL error mode A/B

### 4.1 未修改 UCX

| Payload | 操作 | `peer` | `none` | `none/peer` |
|---:|---|---:|---:|---:|
| 1 GiB | READ | 5.203 GB/s | 27.399 GB/s | 5.27x |
| 1 GiB | WRITE | 4.962 GB/s | 23.600 GB/s | 4.76x |

`peer` 使用 ROCm-host-CMA-ROCm pipeline；`none` 的协议表明确出现：

```text
128..inf | zero-copy | rocm_ipc/rocm_ipc
```

64 MiB `none` 的 READ/WRITE 为 `12.650/12.617 GB/s`，均通过数值校验。

### 4.2 实验性 UCX capability flag

补丁只增加一项 capability：

```diff
- iface_attr->cap.flags = UCT_IFACE_FLAG_GET_ZCOPY |
+ iface_attr->cap.flags = UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE |
+                         UCT_IFACE_FLAG_GET_ZCOPY |
```

它安装在独立 prefix，没有覆盖原 UCX。`ucx_info -d` 随后显示 `rocm_ipc error handling: peer failure`。NIXL 保持默认 `peer` 的结果为：

| Payload | 操作 | 实验 UCX `peer` | 原 UCX `none` | 差异 |
|---:|---|---:|---:|---:|
| 64 MiB | READ | 12.653 GB/s | 12.650 GB/s | +0.02% |
| 64 MiB | WRITE | 12.505 GB/s | 12.617 GB/s | -0.89% |
| 1 GiB | READ | 27.382 GB/s | 27.399 GB/s | -0.06% |
| 1 GiB | WRITE | 23.512 GB/s | 23.600 GB/s | -0.37% |

四组均通过 payload 校验，并明确选择 `rocm_ipc/rocm_ipc`。与原 UCX `peer` 相比，1 GiB READ/WRITE 提高约 `5.26x/4.74x`。

## 5. 故障语义

故障注入使用 NIXL `peer`、实验 UCX 与进程级硬超时。结果中的 target 退出码 `42/43` 为测试主动注入。

| 场景 | Payload | target | initiator 结果 | 解释 |
|---|---:|---:|---|---|
| 正常 READ | 1 GiB | 0 | `DONE`, 0.142 s | 冷态正确性基线 |
| 传输前异常退出 | 1 GiB | 42 | `NIXL_ERR_REMOTE_DISCONNECT`, 0.213 ms | 未挂死 |
| 注销后正常退出 | 1 GiB | 0 | `NIXL_ERR_REMOTE_DISCONNECT`, 0.176 ms | 未挂死 |
| 提交后退出，READ | 8 GiB | 43 | `DONE`, 0.405 s，payload verified | 已 attach allocation 支撑在途 READ 完成 |
| 提交后退出，WRITE | 8 GiB | 43 | `NIXL_ERR_REMOTE_DISCONNECT`, 0.322 s | 未挂死 |
| 注销 registration 但进程存活 | 1 GiB | 1 | initiator 外层超时 35 s | stale rkey 无错误传播 |

最后一项的对照结果：

- 原 UCX `none + rocm_ipc` 同样超时，说明不是 capability flag 新引入的回归；
- 原 UCX `peer` 的 host fallback 在 `rocm_copy_cache.c` 触发 Fatal，target `SIGABRT`，initiator 在 0.258 s 得到 remote disconnect；
- 两条路径都缺少理想的防御性恢复，只是表现为“挂起”与“目标崩溃”。

UCX 要求远端 registration/rkey 在可能被访问期间保持有效。因此 stale-rkey case 不等同于合法 peer failure，不能单独作为拒绝 flag 的依据；但它揭示了值得单独修复的 ROCm error propagation 问题。

## 6. 可主张结论与上游边界

可以主张：

1. NIXL 默认 peer error mode 与 UCX `rocm_ipc` capability 不匹配是 W7900 数据 lane 回退的根因。
2. `none` 可恢复直接 ROCm IPC，但放弃 endpoint peer-failure 语义，不适合作为无条件默认值。
3. 实验 flag 在 W7900 NIXL RMA 中恢复 4.74x 至 5.26x 性能，几乎无额外开销。
4. 已测试的合法 peer 退出场景能够完成在途 READ 或返回 remote disconnect，没有挂死。

暂不能主张：

1. 在 MI300/MI350 或其他 ROCm 架构上已验证。
2. stale/invalid rkey 已具备完整错误恢复。
3. 仅增加 capability flag 已达到可无条件合入的生产质量。
4. `ucx_perftest -e` 的所有通用 endpoint 形态都能只依赖 ROCm transport。

建议先向 UCX/ROCm 维护者提交 RFC 或 draft PR：补丁、W7900 性能数据和 peer-exit 测试一并提供，同时请求其他架构 CI。stale-rkey 防御性恢复应独立成 issue，避免把合法 peer failure 与应用违反 rkey 生命周期混成一个问题。

候选回归测试应采用 `TCP/CMA keepalive + rocm_ipc RMA` 的多 lane endpoint，并覆盖传输前退出与在途 READ/WRITE。OpenUCX PR #9751 已明确表明不应要求 IPC transport 自身成为 keepalive lane。

## 7. 复现与归档

主要脚本：

- `pd_disaggregation/nixl_rocm_two_gpu_bench.py`
- `pd_disaggregation/run_nixl_visibility_matrix.sh`
- `pd_disaggregation/run_ucx_perftest_rocm.sh`
- `pd_disaggregation/nixl_rocm_peer_failure_test.py`
- `pd_disaggregation/run_nixl_peer_failure_matrix.sh`
- `pd_disaggregation/ucx_rocm_ipc_peer_failure_experiment.patch`

原始日志：`results/nixl_ucx_rootcause_20260805.tgz`

```text
SHA256 c9f18e9f7b15265bae1c15cc1cac3f5246b11b2e32ae91bd6de2c03a64c48690
```

该归档包含原/实验 UCX 的 64 MiB 与 1 GiB A/B、协议表、build/install 日志、visibility matrix 和全部故障注入日志。
