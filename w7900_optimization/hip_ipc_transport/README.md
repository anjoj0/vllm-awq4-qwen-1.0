# W7900 HIP IPC transport

该目录实现同一 Linux 节点内面向 Radeon PRO W7900 的 GPU 直连传输，用于消除 vLLM Prefill/Decode 解耦中 NIXL/UCX 将 ROCm VRAM RMA 回退到 `tcp/bond0` 的瓶颈。实现分两步验证：先用 vLLM 专用 connector facade 替换 payload 数据面，再实现 NIXL 动态 backend `W7900_HIP_IPC`。正式 profile 使用后者，不依赖 Python facade。

## 数据与控制路径

```text
Prefill TP=4 (GPU 0-3)                 Decode TP=4 (GPU 4-7)
  export VRAM allocation                 import HIP IPC allocation
  hipIpcGetMemHandle  ── metadata ──>   hipIpcOpenMemHandle
          │                                      │
          └──── Attention KV / Mamba state ──────┘
                       hipMemcpyAsync

NIXL: scheduler、agent metadata、TP rank mapping、block lease、metrics
Unix datagram: bundled transfer notification
HIP IPC: 同节点大块 GPU payload
```

关键实现如下：

- `hipMemGetAddressRange` 将 vLLM/PyTorch 的切片地址回溯到 caching allocator 的底层 allocation，HIP IPC handle 因而对应合法的 allocation base。
- `HIP_VISIBLE_DEVICES=0,1,2,3` 与 `4,5,6,7` 可形成互不重叠的进程视图；metadata 同时携带导出 allocation 的全局地址、大小和 handle。
- 相邻且源、目的地址均连续的 descriptor 在 backend 内合并，减少 `hipMemcpyAsync` 提交次数。
- 远端 allocation 使用 `weak_ptr/shared_ptr` 缓存，同一 handle 只打开一次，并在最后一个 metadata 释放后关闭。
- HIP event 记录真实 GPU copy 时间；NIXL 原生 metrics 继续记录字节、descriptor 和失败计数。
- backend 只声明 `VRAM_SEG` 和同主机 `READ/WRITE`，不声称跨节点能力。

## 目录

| 文件 | 作用 |
|---|---|
| `hip_ipc_bench.cpp` | 不经过 vLLM/NIXL 的裸 HIP IPC GET/PUT 门禁 |
| `w7900_hip_ipc.cpp` | vLLM facade 使用的 C ABI transport |
| `w7900_hip_ipc_connector.py` | 保留 NIXL 控制面、替换 GPU payload 的过渡 connector |
| `test_cross_visibility.py` | 两个不重叠 GPU 可见域的跨进程正确性/带宽门禁 |
| `nixl_plugin/` | 原生 NIXL backend、plugin entry、构建和双进程测试 |

## 构建与安装 NIXL plugin

在 ROCm vLLM 容器内：

```bash
cd /workspace/pd_disagg_20260802/hip_ipc_transport/nixl_plugin

NIXL_SRC=/workspace/pd_disagg_20260802/src/nixl-main \
NIXL_PREFIX=/workspace/pd_disagg_20260802/venv \
INSTALL_DIR=/workspace/pd_disagg_20260802/venv/lib/x86_64-linux-gnu/plugins \
  bash build_plugin.sh
```

vLLM worker 是多进程，插件必须安装到 `nixl_rocm` 的标准 plugin 目录，不能只依赖父进程的临时搜索路径。探测应同时显示：

```text
plugins = ['UCX', 'W7900_HIP_IPC']
mems = ['VRAM_SEG']
```

## 分层门禁

### 裸 HIP IPC

```bash
hipcc -O3 -std=c++17 hip_ipc_bench.cpp -o hip_ipc_bench
./hip_ipc_bench --src 0 --dst 4 --direction get --bytes 1073741824
./hip_ipc_bench --src 0 --dst 4 --direction put --bytes 1073741824
```

实测 GET/PUT 均通过 head/tail 数值校验，带宽分别为 `27.16 GB/s` 和 `27.30 GB/s`。跨进程 1 GiB 门禁为 `25.12 GiB/s`。

### 原生 NIXL backend

```bash
source /workspace/pd_disagg_20260802/scripts/activate_pd_env.sh
python nixl_plugin/test_nixl_plugin.py --bytes 1073741824
```

双进程 `HIP_VISIBLE_DEVICES=0,1,2,3` 与 `4,5,6,7` 下，64 MiB 和 1 GiB 均 `valid=true`；1 GiB NIXL 端到端带宽为 `25.12 GiB/s`，与裸 HIP IPC 一致。

### vLLM 原生 `NixlConnector`

正式启动脚本位于 `../pd_disaggregation/`：

```bash
MAX_MODEL_LEN=65536 MAX_NUM_BATCHED_TOKENS=65536 \
  bash ../pd_disaggregation/start_prefill_tp4_nixl_hip_ipc.sh

MAX_MODEL_LEN=65536 MAX_NUM_BATCHED_TOKENS=65536 \
  bash ../pd_disaggregation/start_decode_tp4_nixl_hip_ipc.sh
```

两侧使用：

```json
{
  "kv_connector": "NixlConnector",
  "kv_connector_extra_config": {"backends": ["W7900_HIP_IPC"]}
}
```

日志中的 8 个 worker 都必须出现 `Backend W7900_HIP_IPC was instantiated`。8K 短请求与直接 Decode 的 greedy 输出逐字一致。

## 性能结果

配置为 Qwen3.6-27B BF16、Prefill TP=4 + Decode TP=4、Triton attention tile=16、64K 输入、32-token 输出、热态服务。

| 数据面 | 并发 | TTFT / mean TTFT | Batch wall |
|---|---:|---:|---:|
| UCX/TCP | 1 | 59.931 s | 61.524 s |
| HIP IPC facade | 1 | 55.736 s | 57.319 s |
| 原生 NIXL plugin | 1 | **55.633 s** | **57.292 s** |
| UCX/TCP | 4 | 144.757 s | 229.934 s |
| HIP IPC facade | 4 | 139.738 s | 225.488 s |
| 原生 NIXL plugin | 4 | **139.922 s** | **225.259 s** |

并发 1 的 plugin 数字取第 2、3 轮热态均值。相对 UCX/TCP，TTFT 下降约 `7.2%`，wall 下降约 `6.9%`；并发 4 的 batch wall 下降约 `2.0%`，mean TTFT 下降约 `3.3%`。plugin 与 facade 基本重合，说明正式 NIXL 抽象没有引入可测回退。

单轮 64K 请求的 4 个 rank 共迁移 `4,381,802,496 bytes`，HIP event 累计 `0.17347 s`，有效 payload 带宽约 `25.26 GB/s`。三轮单请求加一轮并发 4 共完成 28 次 rank transfer、`30,672,617,472 bytes` 和 5824 个原始 descriptor，传输失败与通知失败均为 0。

## 当前边界

- 仅支持同一 Linux 主机内的 HIP IPC；跨主机应回退 UCX/RDMA 等 backend。
- 当前 request handle 是单次提交语义；`postXfer()` 后不能 repost 同一 handle。vLLM NIXL connector 符合该生命周期。若作为通用上游 backend，需要补 event reset/repost 语义与更完整的错误恢复。
- Unix datagram 只承载小型通知，不用于 payload；socket path 依赖同一 mount namespace 可见的 `/tmp`。
- 64K 并发 4 的主要时间仍是串行/排队的 Prefill，替换数据面只能消除约 4 s 的 TCP 迁移，不能替代调度层的并发优化。

原始 JSON、Prometheus metrics 和服务日志见 `../results/20260803_hip_ipc_raw.tgz`。
