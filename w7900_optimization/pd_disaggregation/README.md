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

## ROCm IPC 前移植实验

[nixl_ad661_rocm_hint.patch](nixl_ad661_rocm_hint.patch) 将 NIXL PR #1536 的核心 memtype hint 前移植到带 ROCm wheel 支持的 1.4.0 快照，并通过 `NIXL_UCX_VRAM_MEMTYPE_HINT=rocm` 启用。该版本编译和数值校验均通过，但 `UCX_PROTO_INFO=y` 仍显示：

```text
remote memory read by ucp_get*(multi) into rocm/GPU1 from rocm/dev[0]
software emulation | tcp/bond0
```

因此该实验补丁不进入默认环境。`UCX_TLS` 仍需包含 `tcp`，因为 `rocm_ipc`/`rocm_copy` 不提供 NIXL 所需的 active-message 控制通道。正式路径改为动态加载 `W7900_HIP_IPC` backend，由 HIP IPC 处理 GPU payload、Unix datagram 处理 notification，同时继续使用 NIXL 原生 scheduler、metadata 和 metrics。
