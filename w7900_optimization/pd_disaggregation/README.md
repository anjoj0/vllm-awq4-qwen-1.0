# W7900 Prefill/Decode 解耦实验环境

该目录在 8 张 Radeon PRO W7900 上实现 Qwen3.6-27B BF16 的 TP=4 Prefill 与 TP=4 Decode 解耦。正式路径使用 vLLM 原生 `NixlConnector`、NIXL 1.4.0 和本项目的 `W7900_HIP_IPC` backend：NIXL 保留调度与生命周期管理，HIP IPC 在同一节点内迁移 Attention KV、Mamba convolution state 和 SSM state。UCX/TCP 路径保留为兼容基线，不依赖 LMCache。

## 兼容性结论

- 基础镜像为 Python 3.14.6，LMCache 0.3.6 要求 Python `<3.14`，不能直接安装。
- PyPI 的 `nixl` 元包只识别 CUDA wheel；ROCm 版本必须从 NIXL 源码构建为 `nixl_rocm`。
- Qwen3.6 是 Attention 与 Mamba/GDN 混合模型。P/D 两侧必须设置 `VLLM_SSM_CONV_STATE_LAYOUT=DS`，否则 3-read Mamba conv state 迁移会在初始化时终止。
- W7900/Navi31 上的 UCX 大块 GPU RMA 会回退到 `tcp/bond0`；`W7900_HIP_IPC` backend 已绕过该数据面限制，在原生 NIXL API 下达到约 25 GB/s 的 64K payload 带宽。

## 固定版本

| 组件 | 版本或 commit |
|---|---|
| ROCm / PyTorch / vLLM | 7.14 / 2.11.0 / 0.23.1.dev1 |
| vLLM 工作树 | `/workspace/vllm-main-20260801` |
| UCX | 1.22.x，`95865d6365b08b5c2b437be37375eac55f31533f` |
| NIXL | 1.4.0，`ad661a7212170db72e7c2505b4f9faaada1dc533` |
| BF16 模型 | `/models/Qwen3.6-27B` |

## 构建

UCX 需要显式启用 ROCm：

```bash
./autogen.sh
./configure \
  --prefix=/workspace/pd_disagg_20260802/deps/ucx-rocm \
  --with-rocm="$ROCM_HOME" \
  --without-cuda \
  --enable-shared \
  --enable-cma \
  --enable-devel-headers \
  --with-verbs
make -j64
make install
```

验证 `ucx_info -d` 同时出现 `rocm_cpy`、`rocm_copy`、`rocm_ipc`：

```bash
LD_LIBRARY_PATH=/workspace/pd_disagg_20260802/deps/ucx-rocm/lib \
  /workspace/pd_disagg_20260802/deps/ucx-rocm/bin/ucx_info -d
```

NIXL 使用独立 prefix 构建：

```bash
python -m venv /workspace/pd_disagg_20260802/venv
source /workspace/pd_disagg_20260802/venv/bin/activate

meson setup build-rocm \
  --buildtype=release \
  --prefix=/workspace/pd_disagg_20260802/venv \
  -Dwheel_variant=rocm \
  -Drocm_path="$ROCM_HOME" \
  -Ducx_path=/workspace/pd_disagg_20260802/deps/ucx-rocm \
  -Denable_plugins=UCX \
  -Dbuild_tests=false \
  -Dbuild_examples=false \
  -Dbuild_nixl_ep=false \
  -Dwith_trace=false \
  -Dinstall_headers=false
meson compile -C build-rocm -j64
meson install -C build-rocm
```

由于 `/opt/python` 是 relocatable Python，启动服务时必须使用 [activate_pd_env.sh](activate_pd_env.sh) 补齐 `PYTHONPATH`、NIXL/UCX 动态库和 plugin 路径。

## 启动

先启动 Prefill，健康后再启动 Decode。顺序冷启动可以避免两个 TP group 同时初始化 RCCL 时偶发停滞。

```bash
MAX_MODEL_LEN=65536 MAX_NUM_BATCHED_TOKENS=65536 \
  bash start_prefill_tp4.sh > prefill.log 2>&1 &
curl --retry 60 --retry-delay 2 http://127.0.0.1:8100/health

MAX_MODEL_LEN=65536 MAX_NUM_BATCHED_TOKENS=65536 \
  bash start_decode_tp4.sh > decode.log 2>&1 &
curl --retry 60 --retry-delay 2 http://127.0.0.1:8200/health

bash start_proxy.sh > proxy.log 2>&1 &
curl http://127.0.0.1:8192/healthcheck
```

服务分配如下：

```text
GPU 0-3  Prefill TP=4, :8100, NixlConnector producer
GPU 4-7  Decode  TP=4, :8200, NixlConnector consumer
CPU      official toy proxy, :8192
```

上述 `start_prefill_tp4.sh` / `start_decode_tp4.sh` 是 UCX/TCP 基线。正式 HIP IPC profile 使用：

```bash
MAX_MODEL_LEN=65536 MAX_NUM_BATCHED_TOKENS=65536 \
  bash start_prefill_tp4_nixl_hip_ipc.sh > prefill_hip_ipc.log 2>&1 &
curl --retry 60 --retry-delay 2 http://127.0.0.1:8100/health

MAX_MODEL_LEN=65536 MAX_NUM_BATCHED_TOKENS=65536 \
  bash start_decode_tp4_nixl_hip_ipc.sh > decode_hip_ipc.log 2>&1 &
curl --retry 60 --retry-delay 2 http://127.0.0.1:8200/health
```

插件必须先安装到 `${VIRTUAL_ENV}/lib/x86_64-linux-gnu/plugins/`。构建、门禁和实现边界见 [HIP IPC transport](../hip_ipc_transport/README.md)。

## 正确性与性能复现

短请求固定输出 sanity：

```bash
curl -sS -H 'Content-Type: application/json' \
  --data-binary @sanity_request.json \
  http://127.0.0.1:8192/v1/completions
```

精确 64K token、固定输出 32 token：

```bash
python benchmark_pd.py \
  --url http://127.0.0.1:8192/v1/completions \
  --source /workspace/bench_data/combined_papers_for_llm.txt \
  --prompt-tokens 64000 \
  --max-tokens 32 \
  --concurrency 4 \
  --output pd_64k_c4.json
```

公平的 8 卡 dual TP=4 基线使用 [benchmark_dual_tp4_64k.sh](benchmark_dual_tp4_64k.sh)。完整结果与分析见 [20260802_pd_disaggregation.md](../results/20260802_pd_disaggregation.md)。

64K、32-token、并发 1 的原生 HIP IPC plugin 热态结果为 TTFT `55.633 s`、wall `57.292 s`，相对 UCX/TCP 分别下降约 `7.2%` 和 `6.9%`。并发 4 的 batch wall 从 `229.934 s` 降至 `225.259 s`，mean TTFT 从 `144.757 s` 降至 `139.922 s`。短请求与直接 Decode 输出逐字一致，累计 28 次 rank transfer 无传输或通知失败。完整数据见 [20260803 HIP IPC 报告](../results/20260803_w7900_hip_ipc_transport.md)。

## 非对称 8 卡资源划分

Qwen3.6-27B 的 head 和线性层维度不适合 TP=6，因此物理 `2/6` 不直接使用 TP2 + TP6。为隔离资源比例与 TP 宽度的影响，三种 profile 均由 TP2 副本组成：

| Profile | Prefill | Decode | 物理卡比例 |
|---|---|---|---:|
| `p2_d6` | 1 x TP2 | 3 x TP2 | 2 + 6 |
| `p4_d4` | 2 x TP2 | 2 x TP2 | 4 + 4 |
| `p6_d2` | 3 x TP2 | 1 x TP2 | 6 + 2 |

启动器逐组加载模型并健康检查；某个 TP2 group 若在 RCCL rendezvous 阶段超时，只回收并重试该组。每个 engine 使用独立 HTTP、distributed-init 和 NIXL side-channel 端口，run 目录保存 PID、worker 映射、manifest、health 和日志。停止脚本只结束该 run 的服务，不停止容器：

```bash
RUN_ID=p6_d2_example \
MAX_MODEL_LEN=65536 MAX_NUM_BATCHED_TOKENS=65536 \
  bash start_asymmetric_pd.sh p6_d2

python benchmark_pd.py \
  --url http://127.0.0.1:8192/v1/completions \
  --source /workspace/bench_data/combined_papers_for_llm.txt \
  --prompt-tokens 64000 --max-tokens 32 --concurrency 6 \
  --output /workspace/pd_disagg_20260802/runs/p6_d2_example/result.json

bash stop_asymmetric_pd.sh \
  /workspace/pd_disagg_20260802/runs/p6_d2_example
```

决定性结果如下：64K 输入、32-token 输出、并发 6 时，`p6_d2` 的 218.66 s wall 相对 `p2_d6` 的 642.43 s 降低 66.0%；8K 输入、2,048-token 输出、并发 12 时，`p2_d6` 的 188.31 s wall 相对 `p6_d2` 的 212.49 s 降低 11.4%。前者由 Prefill 主导，后者已进入 Decode 副本扩展能够覆盖 Prefill 排队的区间。128K TP2+FP8 KV 在 16K chunk 和 600 s worker watchdog 下容量可行，但 Mean TTFT 为 892.29 s，应视为容量 profile。完整配置、失败边界和原始证据见 [非对称 P/D 实验报告](../results/20260804_asymmetric_pd_matrix.md)。

## ROCm 内存类型提示实验

[nixl_ad661_rocm_hint.patch](nixl_ad661_rocm_hint.patch) 将 NIXL PR #1536 的核心 memtype hint 前移植到带 ROCm wheel 支持的 1.4.0 快照，并通过 `NIXL_UCX_VRAM_MEMTYPE_HINT=rocm` 启用。该补丁只强化 `ucp_mem_map` 的 ROCm 内存类型注册语义，不参与 UCP RMA/rendezvous 协议选择；二者应独立评价。该版本编译和数值校验均通过，当时使用的不完整 TLS 配置仍显示：

```text
remote memory read by ucp_get*(multi) into rocm/GPU1 from rocm/dev[0]
software emulation | tcp/bond0
```

因此不能将该结果解释为 PR #1536 修复失败，也不能用它判断 `rocm_ipc` 能否工作。`UCX_TLS` 需要同时包含 `sm` 和控制面 transport；当前推荐值为 `sm,rocm,tcp,self`。

### UCX RMA pipeline 缓解项

UCX 1.22 设置 `UCX_RMA_PPLN_ENABLE=y` 后，W7900 双进程 READ/WRITE 均能通过数值校验。仅加入 pipeline、未加入 `sm` 时，1 GiB READ/WRITE 为 `1.060/1.056 GB/s`；使用完整的 `sm,rocm,tcp,self` 后进一步提高到 `3.844/3.475 GB/s`。NIXL 的 `UCX_PROTO_INFO` 仍显示 `rocm_copy`、host fragments 和 `cma/memory`，不是直接 `rocm_ipc` lane。

纯 UCX 1.22 跨物理 GPU0/GPU1 对照可以直接选择 `rocm_ipc/rocm_ipc`：1 GiB GET/PUT 分别达到 `18.243/11.261 GB/s`。这说明 UCX transport 本身可用，剩余问题位于 NIXL UCX worker-address/endpoint 集成路径。上游方向因此调整为修复现有 NIXL UCX backend；`W7900_HIP_IPC` 保留为性能与生命周期回归原型，不再作为新增 NIXL plugin 的首选提案。完整证据见 [NIXL #2039 `sm` 与原生 UCX ROCm IPC 复核](../results/20260805_ucx_sm_upstream_followup.md)。

### UCX error mode 根因与实验修复

后续单变量实验推翻了“worker address 丢失 lane”的初步推断。NIXL UCX backend 默认使用 `UCP_ERR_HANDLING_MODE_PEER`，而 UCX 1.22 的 `rocm_ipc` 只声明 `error handling: none`，所以 UCP 在 endpoint capability 筛选时排除了该数据 lane。

基准脚本通过 `--ucx-error-handling peer|none` 控制该变量：

```bash
ERROR_HANDLING=none CASE_SET=hip_single_READ \
  bash run_nixl_visibility_matrix.sh

ERROR_HANDLING=peer CASE_SET=hip_single_READ \
  bash run_nixl_visibility_matrix.sh
```

原 UCX 在 1 GiB READ/WRITE 下，`none` 的 `27.399/23.600 GB/s` 相对默认 `peer` 的 `5.203/4.962 GB/s` 提高 `5.27x/4.76x`。实验性 UCX flag 在保留 `peer` 时达到 `27.382/23.512 GB/s`，并恢复 `rocm_ipc/rocm_ipc`。

故障矩阵使用：

```bash
UCX_ROOT=/path/to/experimental-ucx \
  bash run_nixl_peer_failure_matrix.sh
```

传输前退出和 8 GiB 在途 READ/WRITE 均未挂死；stale registration 会暴露 ROCm IPC 既有的失效 rkey 错误传播缺口。该 flag 因而仍标记为实验性，不应在缺少额外架构 CI 时直接作为默认生产补丁。完整根因、性能表、故障语义和日志哈希见 [UCX error mode 根因报告](../results/20260805_nixl_ucx_error_mode_root_cause.md)。

OpenUCX PR [#9751](https://github.com/openucx/ucx/pull/9751) 是该设计的重要先例：上游删除了 `cuda_ipc` 自身的 `EP_CHECK`，但保留 `ERRHANDLE_PEER_FAILURE`，由独立 AM transport 承担 endpoint keepalive。W7900 的候选结构同样是 TCP AM/keepalive 与 `rocm_ipc` RMA 数据 lane 的组合，后续回归测试应验证该多 lane endpoint，而不是要求纯 ROCm IPC transport 独立建立控制面。

### W7900 topology and cross-NUMA regression

`run_nixl_visibility_matrix.sh` accepts `TARGET_DEVICE` and `INITIATOR_DEVICE`, so the same harness can test arbitrary GPU pairs:

```bash
UCX_ROOT=/path/to/ucx-peer-flag \
TARGET_DEVICE=0 INITIATOR_DEVICE=4 \
BYTES=$((1024*1024*1024)) ITERATIONS=3 \
CASE_SET=hip_single_READ bash run_nixl_visibility_matrix.sh
```

On this node GPU0--3 are in NUMA0 and GPU4--7 are in NUMA1. The 2026-08-06 `0-1`, `4-5` (same NUMA), and `0-4` (cross NUMA) runs all selected `rocm_ipc/rocm_ipc`. The experimental flag reached `27.378--27.394/23.528--23.546 GB/s` for 1 GiB READ/WRITE, while original `peer` reached only about `5.065--5.203/4.082--4.962 GB/s`, a `4.75--5.76x` gain. Three repetitions of legal cross-NUMA peer-exit cases all returned an explicit `DONE` or `NIXL_ERR_REMOTE_DISCONNECT`. See [the topology report](../results/20260806_nixl_ucx_topology_report.md) for methods, structured data, and raw-log hash.

This evidence covers the current W7900/gfx1100 and UCX 1.22 build only; it does not replace MI300/MI350 CI. The capability change remains subject to UCX transport-owner review and cross-architecture regression before merge.

The same one-line change was rebuilt from ROCm/ucx `develop` on 2026-08-06 (UCX 1.23.0) and reproduced at `27.376--27.409/23.530--23.549 GB/s` for 1 GiB READ/WRITE on same- and cross-NUMA pairs. The develop-branch report, JSON manifest and raw-log archive are [here](../results/20260806_nixl_ucx_develop_peerflag_report.md). The reproducible build entry point is [build_ucx_rocm.sh](build_ucx_rocm.sh); override `SRC`, `PREFIX`, `ROCM_DEVEL` and `JOBS` for a different checkout. The generic UCT GTest capability draft was removed after observing that all eight instances skip before enumerating `rocm_ipc`; the valid regression remains the dual-process NIXL harness, and an upstream multi-process fixture is still needed.

`build_ucx_rocm.sh` keeps release builds logging-disabled by default. Set
`ENABLE_LOGGING=y` to produce an independent diagnostic build with UCX trace
support; the script records the exact configure command in `configure_command.log`.
The 2026-08-07 stale-registration trace is documented in
[20260807_ucx_pr11299_stale_trace_report.md](../results/20260807_ucx_pr11299_stale_trace_report.md)
and archived with SHA256 alongside it.

The first repair follow-up handles negative HSA async-copy completion signals and
maps them to `UCS_ERR_IO_ERROR`. It preserves the 92-pass ROCm IPC GTest result and
the 27.386/23.565 GB/s NIXL READ/pipeline-WRITE paths, but W7900 ROCr leaves the
exporter-invalidated stale-rkey signal pending rather than negative. The patch,
validation, and the required registration-retirement handshake are documented in
[20260807_ucx_rocm_async_signal_fix.md](../results/20260807_ucx_rocm_async_signal_fix.md).

OpenUCX PR [#11299](https://github.com/openucx/ucx/pull/11299) changes the ROCm IPC handle cache and adds device-initiated PUT. A separate W7900 build of #11299 head `4dddf15e4` plus the peer-failure flag passed `92` ROCm IPC tests with `40` expected skips and no failures. NIXL same/cross-NUMA READ/WRITE remained at `27.377--27.391/23.522--23.527 GB/s`, and 12 legal cross-NUMA peer-exit injections all completed or returned remote disconnect. This demonstrates mechanical compatibility without modifying #11299's cache. See the [#11299 compatibility report](../results/20260806_ucx_pr11299_compatibility_report.md); the unposted maintainer follow-up is [kept separately](ucx_pr11299_followup_draft.md).
