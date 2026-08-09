# 2026-07-31 之后 DFlash、多卡并行、前缀缓存与 P/D 分离工作总结

**统计截止：2026-08-08**
**主要验证环境：`xdhpc` 容器，8 x AMD Radeon PRO W7900（gfx1100，48 GiB/卡）**

本文整理 2026-07-31 之后完成的 DFlash、长上下文多卡并行、前缀缓存和 Prefill/Decode（P/D）分离工作，并单独说明这些工作对 vLLM、NIXL、UCX、ROCm/ROCr 开源社区的普适性价值。文中把“实验观察”“可复现结论”“开源提交/问题”分开记录。

## 1. 先给结论

1. **DFlash 的有效范围被重新标定。** 在 W7900 RDNA3 W4A16 路径上，DFlash 的收益集中在短上下文；测得的在线路由交叉点约为 14K tokens。超过约 16K 后，target-only 通常更快。恢复 checkpoint 中的混合 Sliding Window Attention（SWA）语义后，8K/16K/32K wall time 相对全上下文 drafter 下降约 11.6%/18.2%/21.0%，是最稳定的 DFlash 路线。
2. **Prefix Caching 是科研长文工作负载中收益最大的系统优化。** 在同一 261K 文档上连续提问，784-token page profile 的首问 wall 约 153.95 s，命中后的问题 TTFT 约 1.43 s；四个共享前缀的并发问题从无缓存 616.82 s 降到 13.09 s，约 47.1x 端到端加速。该收益只适用于共享长前缀，不能外推到完全不同的文档。
3. **BF16 长文的默认执行路线应是 TP=8 + `block_size=784` + torch.compile/HIP Graph。** 相同 4 x 约 63.5K 请求中，compile/graph 将 wall 从 121.99 s 降到 78.18 s，聚合输出吞吐从 2.10 提升到 3.27 tok/s。`TP=4 x PP=2` 增加约 94% KV 容量，但 4 请求 wall 为 TP=8 的 2.69x，不是低时延方案。
4. **P/D 分离功能闭环已经成立，但 W7900 初始数据面限制使它不适合作为当前默认 profile。** 混合 Qwen3.6 的 Attention KV、Mamba convolution state 和 SSM state 均能迁移，8K/64K 固定长度热态输出逐字一致。原生 NIXL/UCX 走 `tcp/bond0` 时，单请求 64K P/D 比 direct 慢约 6.4%；相同 8 卡资源下，双 TP=4 副本仍明显优于 P/D。
5. **我们把 P/D 研究中的通用故障模式转化成了可上游讨论的接口问题。** 包括 UCX ROCm IPC lane 选择、负 HSA completion signal 的错误传播、ROCr 无效 IPC handle 阻塞、NIXL VRAM memtype hint、in-flight transfer 生命周期，以及 vLLM/NIXL 的 generation fencing。部分已经进入 PR，部分以 issue/RFC 形式保留。

## 2. 统一环境与证据口径

| 项目 | 主要版本/配置 |
|---|---|
| GPU | 8 x Radeon PRO W7900，gfx1100，PCIe-only；GPU 0-3/4-7 分属两个 NUMA 域 |
| ROCm/PyTorch | ROCm 7.14，PyTorch 2.11.0 |
| BF16 服务 | Qwen3.6-27B BF16；vLLM `0.23.1.dev1`；Triton unified attention；TP=8 或 P/D 两侧 TP=4 |
| DFlash 服务 | Qwen3.6-27B-AWQ4 target + 5-layer Qwen3.6-27B-DFlash drafter；target TP=4、draft TP=1、FP8 KV、eager |
| P/D 数据层 | NIXL 1.4.0；源码构建 UCX 1.22 ROCm；vLLM 原生 `NixlConnector` |
| 长文输入 | 8K、16K、32K、64K、约 261K/256K；科研长文与 Nowcast3D 固定质量题 |
| 质量门禁 | 固定 token 数、temperature=0、固定 seed；Nowcast3D 质量题常用 `QA=4/4`、JSON/citation/evidence 100% |

证据优先级为：容器内原始 JSON/日志和可复现脚本 > 汇总报告 > GitHub 页面描述。不同工作树、不同 runner 或不同 page size 的结果只用于解释趋势，不能直接计算跨版本加速比。

主要容器证据目录：

- `/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results/dflash_five_routes_20260801/`
- `/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results/dflash_longctx_diag_20260801/`
- `/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results/20260802_multicard_frontier.md`
- `/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results/20260802_pd_disaggregation.md`
- `/workspace/pd_disagg_20260802/hip_ipc_transport/`
- `/workspace/pd_disagg_20260802/hsa_ipc_lifetime_repro/`

## 3. 时间线：从性能探索到开源问题

| 日期 | 工作 | 产出/意义 |
|---|---|---|
| 2026-07-31 | 开始把 DFlash 长上下文失效拆成 SWA、small-query kernel、自适应 N、D-Cut、drafter context compression 五条路线 | 从“DFlash 越长越慢”推进到可解释的成本/接受率模型 |
| 2026-08-01 | 完成长上下文失效机理诊断；完成 DFlash 五路线第一轮质量与热态 A/B；完成 TP/PP、DCP、SP、custom all-reduce、prefix cache 候选检查 | 形成 14K 路由阈值、SWA 最高优先级、ROCm backend 能力缺口清单 |
| 2026-08-02 | 完成 8 卡 BF16 多卡 frontier；prefix cache、compile/HIP Graph、TP=4 x PP=2 与公平双副本对比；完成 P/D 初版闭环报告 | 将“支持 256K”转为“长前缀复用 + 请求路由 + 阶段隔离”的工作负载结论 |
| 2026-08-03 | 实现并验证同主机 W7900 HIP IPC 数据面和原生 NIXL `W7900_HIP_IPC` backend | 绕过 UCX/TCP GPU RMA 回退；1 GiB 跨进程约 25.12 GiB/s，64K P/D wall 下降约 6.9% |
| 2026-08-05 | 发布 ROCm/ROCm #6566、ROCm/ucx #35；提交/跟踪 NIXL PR #1536 | 将 W7900 ROCm IPC lane 选择问题公开化，并提供复现信息 |
| 2026-08-06 | vLLM #50047 收到维护者对“由路由器显式通知 P 崩溃”的 review；NIXL #2044 讨论 in-flight transfer release 生命周期 | P/D 故障处理从隐式推断转向显式控制面契约 |
| 2026-08-07 | OpenUCX #11299 合并；ROCm/rocm-systems #9827 发布 invalidated pre-attach IPC handle 复现；NIXL #2039 发布原生 HIP IPC RFC | UCX 新 ROCm transport 能力进入主线，ROCr/NIXL/UCX 的跨项目边界被明确记录 |
| 2026-08-08 | OpenUCX #11743 rebase 到 #11299 当前主线，补 AUTHORS、修 CI 并推送新头 `b3b800cb…`；xdhpc ROCm 定向测试通过 | 异步负 HSA signal 错误不再卡在 pending queue；PR 已重新进入 CI |

## 4. DFlash：长上下文失效、五条路线与普适结论

### 4.1 失效机理

DFlash 的成本不是每轮重新对整个 prompt 做 K/V projection，而是 5 个 draft layer 的 `N+1` query 反复扫描已缓存的长 KV，随后 target 对候选块做精确验证。可用下式判断是否值得启用：

```text
T_spec_per_token = (T_draft(N,L) + T_verify(N+1,L) + T_sync(N)) / E[A(N,L)]
```

只有当该值低于 target-only 的单 token 成本时才有收益。上下文变长会增加 draft/verify 的长 KV 扫描；候选后段接受率下降则会把更多计算变成无效工作。W7900 RDNA3 W4A16 又显著加快了 target-only，使 DFlash 更早跨过收益交叉点。

16K 同后端热态诊断（旧 AWQ 路径）为：target-only 19.159 s，DFlash N=4 22.104 s，N=8 22.094 s，draft TP=4 22.125 s；DFlash 比 target-only 慢约 15.3%，主要损失发生在 TTFT 后的 draft/verification 循环。

### 4.2 五路线结果

| 路线 | 关键结果 | 决策 |
|---|---|---|
| 混合 SWA 语义 | DFlash checkpoint 为 4 层 2K sliding + 1 层 full；恢复后相对 5 层 full：8K 10.93→9.66 s，16K 23.73→19.40 s，32K 59.93→47.35 s | **最高优先级/推荐主线** |
| gfx1100 small-query attention | DFlash/target verification 的 shape-scoped 最优为 tile=32、warps=4；普通长 prefill 仍以 tile=16 为主 | 形状域专门调度，不全局替换 |
| 自适应 N | target-only/DFlash 交叉点约 12K-16K；短文 N=4，超过约 14K 路由 target-only；N=4 与 N=8差异小于约 1.2% | **上线轻量路由器** |
| D-Cut | 移植上游 #47131，并补 V2 logits confidence、`dcut_keep_lens` 回传与 scheduler 截断；keep ratio 0.778，但单请求/并发 4 慢约 1%-2.1% | 默认关闭，保留研究分支 |
| Drafter context compression | 只压缩唯一 full-attention draft layer；16K 最近 8K 比 full 慢 17.7%，32K 最近 8K/16K 慢 7.3%/6.7%，质量仍 QA=4/4 | 不进入默认配置 |

额外多卡结果：5 层 drafter 的 draft TP=4 比 TP=1 慢约 0.5%-0.6%；复制 drafter 不会自动提升单请求速度。DCP 在当前 ROCm FlashAttention/Triton 后端无法提供 paged-KV/LSE 合并契约，未伪造性能结论。

### 4.3 DFlash 的开源关联

- vLLM [#40898](https://github.com/vllm-project/vllm/pull/40898)：DFlash drafter Sliding Window Attention；已关闭，未合并。它解释了为什么不能只手工补一个 `sliding_window` 参数，KV group/block table/slot mapping 需要成套迁移。
- vLLM [#47914](https://github.com/vllm-project/vllm/pull/47914)：hybrid SWA + full-attention DFlash 支持；已关闭。其思路与本阶段恢复 checkpoint layer types 的工作一致。
- vLLM [#48113](https://github.com/vllm-project/vllm/pull/48113)：修 DFlash draft/target layer-count mismatch；已关闭。它是混合层数正确性的必要背景。
- vLLM [#47131](https://github.com/vllm-project/vllm/pull/47131)：D-Cut 自适应 verification depth pruning；仍 open。本阶段完成 V2 方向的移植验证，但量化结果显示当前 W7900 端到端开销仍略大于收益。
- vLLM [#41703](https://github.com/vllm-project/vllm/pull/41703)：DFlash batched verification 参考；仍 open。

普适价值是：DFlash 不能以固定 `N=8` 或单一 kernel benchmark 评价，必须同时考虑 checkpoint 的 attention 语义、target/draft 后端、上下文长度、接受率、batch 和 warmup/JIT。这个成本模型可迁移到其他 speculative decoder 和其他 GPU 架构。

## 5. 多卡并行与前缀缓存

### 5.1 Prefix Caching：面向真实科研工作流的状态复用

Qwen3.6 同时包含 Attention 与 Mamba/GDN 状态。prefix caching 会触发混合缓存的 `align` 模式；默认约 400-token 页在新工作树上使 261K 首问 wall 仍约 294-296 s。历史优化树使用 V1 runner + `mamba_cache_mode=align` + 显式 `block_size=784` 后，首问 TTFT 150.72 s、wall 153.95 s，后续共享前缀问题 TTFT 约 1.43 s。

四个共享 261K 前缀问题的同树 A/B：

| | 无缓存 | 命中缓存 |
|---|---:|---:|
| 4 请求 wall | 616.82 s | **13.09 s** |
| 平均 TTFT | 383.28 s | **6.15 s** |
| 聚合吞吐 | 0.415 tok/s | **19.56 tok/s** |

该结果消除了约 104 万个重复 prompt token 的主 prefill。它对同一论文、代码库、病例或长对话的多轮/多用户问答具有直接普适性，但对四个完全不同的 261K 文档没有收益。

### 5.2 Compile/HIP Graph、PP 与未形成结论的路线

- 同树 4 x 63.5K 请求：eager wall 121.99 s，compile + HIP Graph 78.18 s；平均 TTFT 74.90→48.31 s，吞吐 2.10→3.27 tok/s。
- TP=4 x PP=2：KV 容量约 398 万 token，对比 TP=8 的约 205 万；但 4 请求 wall 295.61 s 对 109.71 s，吞吐仅 37.1%。它是容量 profile，不是速度 profile。
- DCP：需要各卡返回局部 softmax LSE；当前 ROCm FlashAttention 不满足 FP8/paged-KV 接口，Triton/ROCM_ATTN 也没有完整合并契约，未运行不可靠的性能测试。
- PCP：当前实现只对 MLA 开放，Qwen3.6 是非 MLA hybrid attention/Mamba，配置层明确拒绝。
- Custom All-Reduce：最终仍选择 PYNCCL；W7900 PCIe-only 拓扑不满足当前 custom all-reduce 全连接要求。
- Sequence Parallel/GEMM-communication fusion：参数能进入 V1 config，但 AOT 编译因 `AsyncTPPass` 缺失失败，未形成性能结论。

多卡研究的普适价值在于形成了 workload router：共享前缀优先 prefix cache；不同长文优先 TP=8 compile/graph；短上下文多租户用双 TP=4；不要把 PP=2 的容量收益误报为时延收益。

## 6. P/D 分离：从机制验证到可用数据面

### 6.1 初版 NIXL/UCX P/D 闭环

GPU 0-3 运行 Prefill TP=4，GPU 4-7 运行 Decode TP=4，vLLM 两侧使用原生 `NixlConnector`。设置 `VLLM_SSM_CONV_STATE_LAYOUT=DS` 后，以下状态均完成注册与迁移：

- Attention KV cache；
- Mamba convolution state；
- SSM state。

8K/64K 固定长度、temperature=0 的热态 greedy 输出与本地 Decode 逐字一致。单请求 64K：direct TP=4 wall 57.826 s/TTFT 56.268 s；P/D wall 61.524 s/TTFT 59.931 s。P/D 多出的约 3.70 s 与每个 TP rank 平均迁移 1,044.7 MB、平均传输 3.915 s 相符。

四个 64K 请求的阶段隔离对比：direct 单 TP=4 的平均请求完成时间 228.27 s，P/D 为 146.34 s；但这不是等资源比较，因为 direct 只使用 4 张卡。相同 8 卡下，双 TP=4 副本 batch wall 120.06 s，P/D 229.93 s，P/D 平均 TTFT 144.76 s 对双副本 84.15 s。因此当前节点不推荐以 P/D 替代双副本。

### 6.2 UCX 数据面问题与 HIP IPC backend

`UCX_PROTO_INFO=y` 显示 W7900 大块 GPU RMA 走 `software emulation | tcp/bond0`，而不是 `rocm_ipc`。将 NIXL PR #1536 的 `ucx_vram_memtype_hint=rocm` 前移植到 NIXL 1.4.0 后，64 MiB 仍显示 TCP，说明问题不是简单 VRAM 类型识别。

为绕过该限制，实现了同主机 `W7900_HIP_IPC` NIXL backend：

```text
NIXL scheduler/metadata/metrics -> 保留
Unix datagram                 -> 小型完成通知
HIP IPC + hipMemcpyAsync      -> GPU payload 数据面
```

实现使用 `hipMemGetAddressRange` 找到 caching allocator allocation base，避免把 tensor interior pointer 当作 IPC handle；远端映射按 handle 缓存，最后一个 metadata 释放后关闭；只声明同主机 `VRAM_SEG` 和 READ/WRITE，不声称跨主机能力。

裸 HIP IPC GET/PUT 约 27.16/27.30 GB/s，跨进程 1 GiB 门禁 25.12 GiB/s；原生 NIXL plugin 64 MiB/1 GiB 校验均通过。64K P/D 热态：

| 数据面 | 并发 1 TTFT | 并发 1 wall | 并发 4 mean TTFT | 并发 4 wall |
|---|---:|---:|---:|---:|
| UCX/TCP | 59.931 s | 61.524 s | 144.757 s | 229.934 s |
| HIP IPC facade | 55.736 s | 57.319 s | 139.738 s | 225.488 s |
| 原生 NIXL plugin | **55.633 s** | **57.292 s** | **139.922 s** | **225.259 s** |

原生 plugin 相对 UCX/TCP 单请求 wall 下降约 6.9%，并发 4 wall 下降约 2.0%；三轮单请求加一轮并发 4 共完成 28 次 rank transfer、约 30.67 GB payload，传输和通知失败均为 0。该数据面并没有消除 Prefill/Decode 的调度成本，说明 P/D 的系统收益必须按“隔离价值、迁移成本、并发负载、数据面能力”四项一起评估。

### 6.3 P/D 的普适性问题

P/D 研究阶段产生的普适性问题不是“W7900 专属 benchmark”，而是跨实现都存在的控制面契约：

1. **状态完整性：** 混合 Attention/Mamba 模型不能只迁移 Attention KV。
2. **故障可见性：** Decode 不能从一次 transfer 失败猜测 Prefill 是否死亡，路由器应显式通知 P 已下线。
3. **生命周期：** release 必须等待 in-flight GPU transfer；通知失败不能重复提交 payload。
4. **代际安全：** Prefill 重启后旧 generation 的 metadata/lease 不能污染新 generation。
5. **传输能力声明：** same-host HIP IPC、host/RDMA、跨主机 fallback 必须由 backend/scheduler 明确区分。

## 7. 已产生的开源 PR、Issue 与 RFC

### 7.1 UCX/ROCm

| 页面 | 状态（2026-08-08） | 我们贡献/获得的普适价值 |
|---|---|---|
| [openucx/ucx#11299](https://github.com/openucx/ucx/pull/11299) | 已于 2026-08-07 合并；当前主线 parent `0957aaea9…` | UCT ROCm device-initiated IPC put 的基础能力进入主线；为后续 signal/error 语义提供共同基线 |
| [openucx/ucx#11743](https://github.com/openucx/ucx/pull/11743) | Open；已 rebase，head `b3b800cb…`；`mergeable=true`，PR 汇总 `mergeable_state=unstable`。2026-08-08 API 快照中 30 个已报告 check-run 均为 completed/success | `uct_rocm_base_progress()` 将 `signal<=0` 视为 terminal；负 signal 映射 `UCS_ERR_IO_ERROR`、回调带错误状态；新增真实 HSA signal gtest；补 AUTHORS |
| [ROCm/ucx#35](https://github.com/ROCm/ucx/issues/35) | Open；无维护者新回复 | 记录 W7900 UCP GET 错选 `tcp/bond0` 而非 `rocm_ipc`，给出 `UCX_PROTO_INFO` 证据和 NIXL 复现路径 |

PR #11743 容器内验证：完整 UCX/ROCm 构建通过；`test_rocm_signal.async_error` 1 passed；#11299 相关 `rocm_ipc` 基础测试 4 passed；device PUT 88 passed、40 skipped。

### 7.2 NIXL/vLLM P/D 控制面

| 页面 | 状态/讨论 | 普适性 |
|---|---|---|
| [ai-dynamo/nixl#1536](https://github.com/ai-dynamo/nixl/pull/1536) | Open；`NIXL/UCX: enforce VRAM memtype hint query behavior and add tests` | 公开 VRAM memtype hint 查询契约；前移植验证显示 hint 本身不能保证 UCX 选择 rocm_ipc lane |
| [ai-dynamo/nixl#2039](https://github.com/ai-dynamo/nixl/issues/2039) | Open；RFC：Native same-host HIP IPC backend；18 条评论均截至我们更新，无维护者结论 | 把 same-host ROCm VRAM transport 的 metadata、handle、notification、backend capability 和 upstream review gates 结构化公开 |
| [ai-dynamo/nixl#2044](https://github.com/ai-dynamo/nixl/pull/2044) | Open；wait for in-flight transfers before `release()` returns | 将 GPU 异步操作的 release 生命周期从实现细节提升为 NIXL backend contract |
| [vllm-project/vllm#50047](https://github.com/vllm-project/vllm/pull/50047) | Open；`CHANGES_REQUESTED`；维护者建议由 router 显式通知 P 死亡，而非让 D 通过一次 xfer 推断 | P/D 故障恢复的正确抽象应位于 router/control plane，避免把偶发传输失败误判为进程死亡 |
| [vllm-project/vllm#49238](https://github.com/vllm-project/vllm/issues/49238) | Open；Decode 在 P 重启后 `loadRemoteMD` segfault；无 2026-08-01 后新维护者结论 | 记录 metadata generation/retirement 不完整时的真实崩溃面 |

我们本地的 generation-fencing 原型提交为 `b42cdd9f2df4869850e67367e15ded1be7e32a4b`，在 `xdhpc` 完成 `415 passed, 2 skipped, 3 deselected`。它尚未作为独立 PR 推送，本轮只把设计边界和上游 #50047 的 router-owned failure notification 讨论记录下来。

### 7.3 ROCm/ROCr

| 页面 | 状态 | 普适性 |
|---|---|---|
| [ROCm/ROCm#6566](https://github.com/ROCm/ROCm/issues/6566) | Open；无 AMD 新回复 | 公开 W7900 同主机 HIP IPC transport 需求，包含裸 HIP/NIXL backend 对比和 25 GiB/s 级门禁 |
| [ROCm/rocm-systems#9827](https://github.com/ROCm/rocm-systems/issues/9827) | Open；无回复 | 复现 `hsa_amd_ipc_memory_attach` 对 pre-attach 后失效 handle 的阻塞；将 P/D stale registration 风险定位到 ROCr 生命周期语义 |

### 7.4 DFlash 相关 vLLM 页面

| 页面 | 当前状态 | 与本阶段关系 |
|---|---|---|
| [#40898](https://github.com/vllm-project/vllm/pull/40898) | Closed，未合并 | DFlash SWA 基础实现；本阶段不能只抽取一个参数，需连同 KV group/block table 一起回移 |
| [#47914](https://github.com/vllm-project/vllm/pull/47914) | Closed | hybrid SWA + full-attention DFlash；与 checkpoint layer types 恢复直接相关 |
| [#48113](https://github.com/vllm-project/vllm/pull/48113) | Closed | 修 draft/target layer-count mismatch；是混合 DFlash 正确性的必要背景 |
| [#47131](https://github.com/vllm-project/vllm/pull/47131) | Open | D-Cut verification depth pruning；本阶段完成 V2 方向功能验证，但当前 W7900 端到端略慢 |
| [#41703](https://github.com/vllm-project/vllm/pull/41703) | Open | batched verification 参考实现 |
| [#48459](https://github.com/vllm-project/vllm/pull/48459) | Open | prefix-cache lookup 对 draft KV group 的排除；说明 speculative draft cache 与 core prefix cache 仍需分层处理 |

## 8. P/D 分离对开源社区的普适性价值

### 8.1 从“能传 KV”到“能传完整模型状态”

混合模型把 P/D 的状态边界从传统 Attention KV 扩展为 KV + convolution state + SSM state。`VLLM_SSM_CONV_STATE_LAYOUT=DS`、`num_regions=16`、`num_descs=29024` 等验证记录，给 vLLM/NIXL connector 实现提供了可复现的状态注册清单。

### 8.2 从“传输失败”到分层故障模型

HIP IPC backend 把 data completion 与 notification completion 分开：短暂的 `EAGAIN/EWOULDBLOCK/ENOBUFS` 应返回 in-progress 并重试通知，不能重传 payload；invalid descriptor、oversized notification、peer socket removed、HIP failure 分别映射为不同 NIXL error。这个模型适用于任何异步 GPU transport，而不仅是 W7900。

### 8.3 从隐式存活推断到显式 router control plane

一次 xfer 失败不等价于 P 进程死亡。vLLM #50047 的 review 已明确倾向“router 通知 D P 已死亡”的接口。固定 engine id、P 扩缩容、P 重启和 D 侧 stale metadata 都应由 router 统一知道并广播 generation/retirement 事件。

### 8.4 从实现可行到上游可审查

NIXL #2039 的 upstreaming notes 固定了 backend 的四项边界：same-host only、`VRAM_SEG`、READ/WRITE、Unix datagram notification；同时要求非 W7900 ROCm 架构、旧/新 ROCm 版本、READ/WRITE、repost、active-repost、recovery、ASAN/UBSAN 和 fork/exec cleanup 门禁。这为将实验代码变成通用插件提供了审查清单。

## 9. 当前推荐路由

| 工作负载 | 推荐配置 | 原因 |
|---|---|---|
| 同一论文/代码库/病例的连续或并发问答 | BF16 TP=8、V1、tile=16、page=784、prefix cache、compile/HIP Graph | 共享前缀收益远高于微调 speculative N |
| 不同的 100K-261K 文档、单请求优先 | BF16 TP=8、compile/HIP Graph；不启用 DFlash | 长文 target-only 更稳定，避免重复 draft/verify 扫描 |
| 不同的约 64K 文档并发 | BF16 TP=8、compile/HIP Graph；按队列调度 | 图执行降低 launch/同步开销 |
| 短上下文多租户 | 两个独立 TP=4 服务 | 比 PP=2 更适合并行处理独立请求 |
| 8K-14K、接受率高的短请求 | AWQ4 target TP=4 + DFlash N=4、draft TP=1、SWA、Triton small-query tile=32/warps=4 | 当前实测 DFlash 净收益区间 |
| P/D 研究/跨节点实验 | NIXL connector + 明确 generation/retirement；同主机优先 HIP IPC，跨主机使用 UCX/RDMA fallback | 将阶段隔离与数据面能力分开评估 |

## 10. 尚未完成与下一步

1. 将 DFlash SWA/hybrid 变更按当前 vLLM 主线重新 rebase，做 target-only、full-attention DFlash、SWA DFlash 三方 8K/16K/32K A/B。
2. 将 DFlash dispatcher 从固定 14K 阈值升级为基于 `T_draft`、`T_verify`、最近接受长度、batch 和输出长度的 cost-aware policy。
3. 把 Prefix Cache 的 page-size/`align` 语义整理为稳定的 Qwen hybrid cache contract，并补 V1/V2 runner 的明确错误信息。
4. 继续推进 NIXL HIP IPC backend 的非 W7900 架构、跨 NUMA、repost/recovery、ASAN/UBSAN 和 fork/exec 测试。
5. 在 router 层落地 P death/generation retirement 事件，避免 D 侧通过 transfer 失败推断进程状态。
6. 等 ROCm/ROCr 对 invalidated IPC handle 和 UCX `rocm_ipc` lane 选择有结论后，再重新评估 P/D 是否能超过双副本。

## 11. 证据索引

- DFlash 五路线：`/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results/dflash_five_routes_20260801/`。
- DFlash 机理诊断：`/workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/results/dflash_longctx_diag_20260801/`。
- 多卡/Prefix Cache：`/workspace/multicard_frontier_20260802/`，汇总见 `20260802_multicard_frontier.md`。
- P/D 原始结果：`/workspace/pd_disagg_20260802/results/`、`logs/`，汇总见 `20260802_pd_disaggregation.md`。
- HIP IPC backend：`/workspace/pd_disagg_20260802/hip_ipc_transport/README.md`、`nixl_plugin/UPSTREAM.md`。
- ROCr 生命周期复现：`/workspace/pd_disagg_20260802/hsa_ipc_lifetime_repro/`。
- vLLM generation fencing 原型：本地提交 `b42cdd9f2df4869850e67367e15ded1be7e32a4b`，`415 passed, 2 skipped, 3 deselected`。

## 12. 口径声明

本文不是完整论文，也不把单个 W7900 节点的绝对时延外推为所有 GPU 的硬件结论。所有性能数字均应与其工作树、runner、page size、KV dtype、prompt 长度和 warmup 条件一起阅读。对于 DCP、PCP、Sequence Parallel、custom all-reduce 等未通过完整后端/数值门禁的路线，本文只记录能力缺口，不给出“收益为零”的结论。
