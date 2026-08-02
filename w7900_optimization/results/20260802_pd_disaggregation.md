# W7900 Prefill/Decode 解耦实验

## 1. 实验动机

超长上下文服务中，Prefill 会长时间占用计算资源。若 Prefill 与 Decode 位于同一 TP group，新到达的长 Prefill 可能使已经开始输出的请求出现明显抖动。Prefill/Decode 解耦把两阶段放到独立 GPU 组，并迁移中间状态，目标是隔离阶段干扰，而不是直接缩短单请求计算量。

Qwen3.6-27B 并非纯 Transformer。它同时包含 Attention KV、Mamba convolution state 和 SSM state，因此仅传 Attention KV 不足以保证远程 Decode 正确。本实验首先验证完整混合状态迁移，再比较单请求、并发请求和同等 8 卡资源下的双副本基线。

## 2. 环境与实现

| 项目 | 配置 |
|---|---|
| GPU | 8× Radeon PRO W7900，`gfx1100`，48 GiB/卡，PCIe-only |
| 拓扑 | GPU 0-3 属于 NUMA 0，GPU 4-7 属于 NUMA 1 |
| 模型 | Qwen3.6-27B BF16 |
| vLLM | 0.23.1.dev1，`/workspace/vllm-main-20260801` |
| Prefill | GPU 0-3，TP=4，端口 8100 |
| Decode | GPU 4-7，TP=4，端口 8200 |
| KV connector | vLLM 原生 `NixlConnector` producer/consumer |
| 数据层 | NIXL 1.4.0 + UCX 1.22 ROCm |
| Attention | `TRITON_ATTN`，HND KV layout，eager |
| 混合状态 | `VLLM_SSM_CONV_STATE_LAYOUT=DS` |

LMCache 0.3.6 要求 Python `<3.14`，与容器 Python 3.14.6 不兼容。因此实验直接使用 vLLM 原生 NIXL connector。系统 UCX 1.16 未启用 ROCm，另行构建 UCX 1.22 后确认存在 `rocm_cpy`、`rocm_copy` 和 `rocm_ipc` memory domain/transport。

初始化日志确认每个 TP worker 均完成以下步骤：

- NIXL UCX backend 实例化；
- Attention KV cache 注册；
- Hybrid SSM 注册，`num_regions=16`、`num_descs=29024`；
- TP=4 到 TP=4 compatibility hash 检查；
- `num_kv_heads=4` 的一对一 transfer topology 建立。

## 3. 正确性门禁

短 prompt 下，P/D 与 Decode 本地计算的 16-token greedy 输出逐字一致。8K prompt 强制输出 32 token 后，两条路径的完整文本也逐字一致，证明 Attention KV、Mamba conv state 与 SSM state 均可被远程 Decode 使用。

未设置 `ignore_eos` 时曾观察到 P/D 与直接路径的 EOS 终止行为不同；强制相同输出长度后文本一致。64K 冷态首轮也出现过路径间分叉，但第二次 direct 自身已产生不同文本；热态 direct 与 P/D 再次逐字一致。因此本文只把固定长度热态结果用于性能比较，不把一次冷态 greedy 分叉归因于 NIXL。

## 4. 单请求结果

所有请求使用真实 tokenizer token 数，`temperature=0`、固定 seed、`ignore_eos=true`，输出固定为 32 token。

| Prompt | 路径 | wall | TTFT | 热态输出速度 |
|---:|---|---:|---:|---:|
| 8K | 直接 TP=4 | 5.070 s | 3.434 s | 19.55 tok/s |
| 8K | P/D TP=4+4 | 5.843 s | 4.171 s | 19.14 tok/s |
| 64K | 直接 TP=4 | 57.826 s | 56.268 s | 20.54 tok/s |
| 64K | P/D TP=4+4 | 61.524 s | 59.931 s | 20.09 tok/s |

8K P/D wall 增加 15.2%，64K 增加 6.4%。Decode 的稳态 token 速度基本不变，差异主要发生在首 token 之前，符合“增加状态迁移、计算量不变”的机制。

64K 时每个 TP rank 平均迁移 1,044.7 MB，平均传输 3.915 s，吞吐 266.8 MB/s。P/D 相对 direct 多出的 3.70 s 与传输时间接近，说明当前单请求损失主要来自数据面，而不是 proxy 或 Decode 调度。

## 5. 四请求阶段隔离

4 个 64K 请求同时提交到一个直接 TP=4 实例时，TTFT 依次约为 56.3、113.4、170.8 和 226.7 s。更关键的是，前三个请求虽已开始输出，仍被后续长 Prefill 干扰，四个请求都到约 228 s 才完成。前三个请求从首 token 到完成的表观速度仅为 0.19、0.28 和 0.56 tok/s。

P/D 路径中，请求完成时间排序为 61.8、118.4、175.3 和 229.9 s；四个请求的 Decode 均保持约 20 tok/s。阶段隔离没有提高整批吞吐，却避免了已进入 Decode 的请求继续被 Prefill 队列阻塞。

| 指标 | 单 TP=4 直接服务 | P/D TP=4+4 | 变化 |
|---|---:|---:|---:|
| batch wall | 228.36 s | 229.93 s | P/D 慢 0.7% |
| 聚合输出吞吐 | 0.561 tok/s | 0.557 tok/s | P/D 低 0.7% |
| 平均 TTFT | 141.80 s | 144.76 s | P/D 高 2.1% |
| 平均请求完成时间 | 228.27 s | 146.34 s | P/D 低 35.9% |

该对比证明了解耦的隔离机制，但资源并不等价：直接服务只使用 4 张卡，P/D 使用 8 张卡。

## 6. 同等 8 卡公平基线

公平基线把 8 张卡划分为两个独立 TP=4 副本，每个副本同时处理 2 个 64K 请求。两个副本并行启动，整批 wall 为 120.06 s。

| 指标 | dual TP=4 replicas | P/D TP=4+4 | P/D 相对变化 |
|---|---:|---:|---:|
| batch wall | 120.06 s | 229.93 s | 慢 91.5% |
| 聚合输出吞吐 | 1.066 tok/s | 0.557 tok/s | 低 47.8% |
| 平均 TTFT | 84.15 s | 144.76 s | 高 72.0% |
| 平均请求完成时间 | 113.94 s | 146.34 s | 高 28.4% |

W7900 节点的 Prefill 与 Decode 使用相同模型、相同精度和对称 TP 资源，且独立长文之间没有可复用前缀。在这种工作负载下，双副本同时执行两组 Prefill 的收益明显高于 P/D 的阶段流水。当前推荐仍是 dual TP=4 或 BF16 TP=8 compile/HIP Graph，而不是 P/D。

## 7. ROCm 数据通道分析

64 MiB 双进程 GPU buffer 基准中：

| GPU 对 | 平均吞吐 | 数据校验 |
|---|---:|---|
| GPU 0→4，跨 NUMA | 0.255 GB/s | 通过 |
| GPU 0→1，同 NUMA | 0.256 GB/s | 通过 |

同 NUMA 与跨 NUMA 几乎一致，且远低于 PCIe GPU peer copy 的合理水平。`UCX_PROTO_INFO=y` 明确显示：

```text
remote memory read by ucp_get*(multi) into rocm/GPU1 from rocm/dev[0]
software emulation | tcp/bond0
```

为排除 VRAM 类型识别问题，实验将 NIXL PR #1536 的 `ucx_vram_memtype_hint=rocm` 核心逻辑前移植到包含 ROCm wheel 支持的 NIXL 1.4.0。前移植版 64 MiB 基准为 0.297 GB/s，但协议仍为 `tcp/bond0`，不能解释为 ROCm IPC 生效。

仅保留 `rocm_ipc,rocm_copy,self` 会导致 UCX endpoint 初始化失败，因为这些 transport 不提供 active-message bcopy；TCP 控制通道不能被简单删除。当前限制位于 NIXL/UCX UCP 对 W7900 ROCm IPC RMA lane 的选择或支持，而不是 HIP peer access、NUMA 分组或模型状态注册。

## 8. 结论

本实验把“容器未安装、无法测试”推进为完整闭环：源码构建 NIXL/UCX，打通 Qwen3.6 混合状态迁移，完成正确性门禁、单请求、并发隔离和同资源公平基线。

P/D 在单 TP=4 对照中能将 4×64K 的平均请求完成时间降低 35.9%，说明 Prefill/Decode 阶段隔离有效；但相同 8 卡资源下，dual TP=4 的 batch wall、吞吐、TTFT 和平均完成时间均更优。当前 W7900 的结论不是“P/D 无法运行”，而是“功能成立，数据面与对称资源配置使其暂不具备部署优势”。

原始 JSON、服务日志、NIXL build/trace 位于 `20260802_pd_disaggregation_raw/`。
