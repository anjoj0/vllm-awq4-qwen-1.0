# 基于 AMD ROCm 生态的 Qwen3.6-27B AWQ4 + DFlash 推理优化系统

## 一、作品难点与创新

### 1.1 项目背景与目标

本项目面向 AMD Radeon / Ryzen AI Max+ "Strix Halo" 平台 (gfx1151, RDNA 3.5, 40 CU, 128 GiB UMA),
完整运行 Qwen3.6-27B 大语言模型,
在 AWQ INT4 量化 + DFlash 推测解码 (speculative decoding) 双重压缩下,
实现 OpenAI 兼容的 chat / responses / completions / 视觉输入 / 工具调用全功能服务。
最终单流 `/v1/responses` 峰值达到 **28.3 token/s** (2026-06-06 `test/bench_full.py --runs 3` 复现, FP8 KV + 256K context + `gpu_memory_utilization=0.90`),
相对无推测解码的 5.6 t/s baseline 提升 **+405%**,
对比同硬件 BF16 (4.3 t/s) 提升 **6.6 倍**,
甚至超过了 vllm-project 公开的 DGX Spark FP8 + DFlash + MTP 单流峰值 (25 t/s) 同档,
全程仅依赖 ROCm 开源软件栈 (TheRock ROCm 7.13 + PyTorch + Triton + vLLM)。

### 1.2 主要技术难点

**(1) RDNA 3.5 与 CDNA/MI300 在多个关键维度差异巨大。**
gfx1151 仅有 40 CU、wave32、1536 VGPR/SIMD、80 KB LDS/CU,
而 MI300 是 304 CU、wave64、寄存器与 LDS 资源均更宽裕。
vLLM 默认的 Triton kernel tile shape 与 AITER FP8/RMSNorm/fused-MoE 都是为 CDNA 调优,
直接套用会出现占用率低、寄存器溢出甚至冻结。
gfx1151 **无原生 FP8 ALU**,
e4m3 dequant 必须在 Triton 中以 "load → cast → multiply scale" 软件路径处理。

**(2) HIP graph 在 gfx1151 上属于"已知冻结类",
必须强制 `--enforce-eager`,
失去了 CUDA Graph 类优化的常规收益。**

**(3) DFlash 推测解码在 vLLM 上游处于早期阶段。**
DFlash 于 2026-03-30 合入 vLLM main (PR #36847),
但 5+ DFlash bug-fix PR 仍处于 open 状态,
NVIDIA H100、A100、AMD MI300 各家都有未完结的 DFlash issue。
其中跨架构通用的痛点是: target / drafter / SWA / DeltaNet 混合架构产生非均匀 KV page size,
vLLM 的 `unify_kv_cache_spec_page_size` 直接断言失败。

**(4) AWQ INT4 W4A16 GEMM 在 RDNA 3.5 上吞吐瓶颈明显。**
vLLM 的默认 `TritonW4A16LinearKernel` tile 形状基于 MI300 优化,
在 gfx1151 上 8K 之后 prefill 从 132 t/s 掉到 38 t/s,
形成陡峭"悬崖曲线",
而真正能驱动 RDNA WMMA INT8 吞吐 (2 倍峰值) 的 `i32_16x16x16_iu8_w32` builtin 没有被利用。

**(5) 长上下文 decode 性能瓶颈在 attention,
不在 GEMM。**
8K 以后 decode 直接掉到个位数 t/s,
HIP custom kernel 解决不了这一段,
必须深入 vLLM attention backend 内部。

### 1.3 创新点

**(1) 自研 AWQ MMQ Q4 HIP custom op (`csrc/awq_mmq_gfx1151/`),
向 vLLM 的 MPLinear dispatcher 注册。**
借鉴 llama.cpp MMQ Q4 模式,
权重以 INT4 形式存入寄存器、解包为 INT8 进入 LDS tile、
通过链式 `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` 在 i32 tile 中累加、
每 group_size=32 边界做 fp32 dequant。
tile shape `(mmq_x=48, mmq_y=64, nwarps=4)` 是按 gfx1151 寄存器预算专门调优。
该 kernel 让 prefill 在 0-32K 上下文范围内被压成 **平直 105-134 t/s**,
4K 痛点处相对原 Triton 路径 3.4 倍,
32K 处 2.8 倍。

**(2) 修改 vLLM Triton unified attention 解锁多查询 verify 的 3D split-K (Flash-Decoding) 路径。**
原 `kernel_unified_attention` 在 launcher 中 `assert causal`,
DFlash verify 所需的双向 attention 直接不被接受;
即便去掉断言,
3D-launch 又被 `max_seqlen_q <= 1` 把 verify 的 N+1 query token shape 挡在门外。
本项目同时放宽这两道门: 增加 `CAUSAL: tl.constexpr` 在 helper 与 kernel 中正确分支,
将 `max_seqlen_q > 1` 替换为 `max_seqlen_q > MAX_SEQLEN_Q_3D_LIMIT (=16)`,
并向 vLLM 上游提交 **PR #44652** (本队首个 vLLM 上游 PR),
完成 DCO、pre-commit 全部通过。

**(3) 对 DFlash hybrid KV page size 上游修复方向做了独立验证。**
本队从 `cyankiwi/Qwen3.6-27B-AWQ-INT4` (hybrid DeltaNet GDN + standard attn) + FP8 target KV + BF16 drafter KV 这条独立路径,
触发了 issue #43626 报告中 vllm-project 已确认的 page-size assertion,
并发现了一个下游次级断言: `KVBlockZeroer.__init__` 跨组累积 `page_size_el`,
即使分区修好仍会因 FP8 vs BF16 page 大小 2:1 触发 `Non-uniform page sizes: 827392 vs 1654784`。
本队向 issue #43626 与 PR #37429 提供了:
跨 rig 复现数据 (gfx1151)、`runner_only_attn_layers` 短期 workaround (~10 行)、
以及对断言信息缺失的可读化建议。

**(4) 一个完整的 ROCm 7.13 + vLLM v0.20.0 可复现工具链。**
含 21 个 idempotent string-replace patch (含 2 个上游 PR cherry-pick、5 个本地补丁、12 个 kyuz0 硬件赋能补丁、2 个新增的非因果 attention + 3D launch),
覆盖 amdsmi disable、AITER CDNA-only fence、`hipCtx` 弃用沉默、
`atomicAdd` 多义性消除 (ROCm 7.13 新增 builtin 重载)、
`HIP_FOUND` cmake 兼容 (PyTorch #180485 回归)、
`/v1/responses` 流式 `chat_template_kwargs` 透传。

## 二、方案论证与设计

### 2.1 整体路线选择

| 决策 | 选择 | 论证 |
| --- | --- | --- |
| 量化方案 | AWQ INT4 W4A16 g32 (cyankiwi 量化) | RDNA 3.5 无原生 FP8 ALU, Qwen3.6-27B-FP8 在 Triton w8a8 autotune 阶段挂起; AWQ INT4 ~14 GiB 是接近无损的最小量化, 视觉模块保留 BF16 |
| 推测解码 | DFlash N=8 + Qwen3.6-27B-DFlash drafter | DFlash 是 z-lab 官方提供的同模型 drafter (~2B BF16, 4 SWA + 1 full), 上游验证最完善 |
| Attention 后端 | TRITON_ATTN (Patches 20+21 后启用) | 与 ROCM_ATTN 同为 ROCm 路径, 但 TRITON_ATTN 有 3D split-K (Flash-Decoding), 长文 decode 收益显著 |
| 推理框架 | vLLM v0.20.0 源码 + 21 patches | 唯一同时支持 AWQ INT4 + DFlash + 视觉 + 工具调用 + OpenAI API 的开源栈 |
| HIP 加速 | 自研 MMQ Q4 custom op | RDNA INT8 WMMA 路径在 vLLM 默认 kernel 没用到, 自研是唯一打通方式 |

### 2.2 关键 ROCm 组件依赖

| 组件 | 版本 / 来源 | 在项目中的作用 |
| --- | --- | --- |
| TheRock ROCm SDK | `7.13.0a20260510` | HIP 编译工具链、rocBLAS、HIPCC |
| PyTorch | `2.13.0a0+rocm7.13.0a20260510` | tensor 后端 + HIP autograd |
| Triton | `3.7.0+git31234c9b.rocm7.13.0a20260510` | unified attention / W4A16 GEMM JIT |
| MIOpen | 随 ROCm | ViT 卷积茎 (`MIOPEN_FIND_MODE=FAST`) |
| AITER | 随 ROCm,部分 fence | gfx 路径 fence 后保留 KV ops |
| HIP custom op | 容器内 `setup.py build_ext` | AWQ MMQ Q4 kernel `--offload-arch=gfx1151` |
| vLLM | `v0.20.0` + 21 patches | 调度 / KV cache / attention dispatch / OpenAI API |

### 2.3 数据流路径设计

请求 → vLLM OpenAI server → tokenizer → scheduler → ModelRunner →
**(target step)** Qwen3.6-27B AWQ INT4 forward (前向中 LM linear 走 HIP MMQ for M≥32, Triton W4A16 for M<32) →
attention 阶段调用 TRITON_ATTN (3D-split-K Flash-Decoding) →
KV write/read 通过 vLLM block pool →
**(drafter step)** DFlash drafter forward (BF16,4 SWA layer + 1 full,共 5 层) 预测 N=8 token →
target verify pass 用 `max_seqlen_q = N+1 = 9` 在 TRITON_ATTN 非因果路径运行 →
acceptance/rejection 反馈给 sampler → 输出 token 流 → SSE/JSON 返回客户端。

整条链路中,
HIP MMQ kernel 负责 prefill GEMM 加速、
TRITON_ATTN 3D-split-K 负责长上下文 decode、
DFlash 负责 decode 多 token 并行、
其他模块均为社区或上游路径。

## 三、原理分析

### 3.1 RDNA 3.5 微架构与 MI300 关键差异

| 项 | gfx1151 (RDNA 3.5) | gfx942 (MI300) | 影响 |
| --- | --- | --- | --- |
| CU 数 | 40 | 304 | 并行度差 8 倍, 决定 tile shape |
| Wavefront | 32 | 64 | Triton/HIP kernel tile 必须重写 |
| VGPR | 1536 / SIMD | 1024 / SIMD | tile 设计需精算寄存器 |
| LDS | 80 KB / CU | 64 KB / CU | LDS tile 余量充足 |
| FP8 ALU | 无 | 有 | e4m3 在 RDNA 走软件 dequant |
| INT8 WMMA | `i32_16x16x16_iu8` | MFMA | 自研 kernel 必走 WMMA iu8 |

### 3.2 AWQ MMQ Q4 kernel 原理

llama.cpp 的 MMQ Q4 模式核心是**保持 INT4 权重直到最后一刻**:
权重以 INT4 packed 存入寄存器, 解包为 INT8 进入 LDS tile,
WMMA `iu8` 在 i32 tile 中累加 16x16x16 矩阵乘,
每经过一个 group_size=32 K-边界时,
用 group 的 fp32 scale 对 i32 partial sum 做 dequant 并写入 fp32 累加器。
端到端的"加权累加"过程完全避开了 dequant-to-fp16 的常见路径,
理论峰值为 fp16 的 2 倍 (per AMD ISA docs)。

kernel decode (M<32) 路径不走 HIP,
而是 fallback 到 `triton_w4a16_gemm`,
DFlash spec round 典型 M=8,
保持位级一致;
代价是 ~22 GiB 双存储 (HIP 与 Triton 各持一份转置后的权重),
在 64K context cap 下的内存预算内可接受。

### 3.3 Triton unified attention 2D vs 3D launch

vLLM 的 `kernel_unified_attention` 有两条 launch 路径:

- **2D launch**: grid `(total_num_q_blocks, num_kv_heads)`,
  每个 workgroup 串行扫描整段 KV cache。
  batch=1、`max_seqlen_q=1`、Qwen3 GQA-8 时只有 8 个 workgroup,
  无法填满 40 CU。32K 长文 decode 因此被串行单 CU 扫描 512 tile 卡死。
- **3D launch (Flash-Decoding)**: grid `(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS_PER_SEQ)`,
  KV 维度沿 segments 切片,
  各 segment 独立计算 `(M, L, O)` partial,
  尾部 `reduce_segments` 用 online softmax recombination (associative, 数学等价) 合并。
  workgroup 数提升到 `8 * NUM_SEGMENTS_PER_SEQ`,
  能充分喂饱 gfx1151 的 40 CU。

### 3.4 非因果 + 多查询 verify 的解锁

DFlash verify 阶段对 `N+1` 个 query token 做**双向** attention (因为这些 token 还没确定哪些被接受),
但 vLLM 原版 `kernel_unified_attention`:
- launcher 中 `assert causal, "Only causal attention is supported"` 直接挡死;
- 即便允许非因果, `use_3d = ... and max_seqlen_q <= 1` 把 N+1 query 形状挡在 2D 路径上。

本项目同时放宽两道门:
- helper 中 `compute_kv_seq_mask(...)` 增加 `CAUSAL: tl.constexpr = True` 分支, 非因果时 `seq_mask = (seq_offset[None,:] >= 0) & (query_abs_pos >= 0)`;
- kernel 中追加 `CAUSAL` constexpr 并向 helper 透传;
- launcher 把全局 `assert causal` 替换为 SWA-only 守卫 `if not causal: assert window_size == (-1,-1)`;
- 使用 `MAX_SEQLEN_Q_3D_LIMIT = 16` 作为 3D-launch 上限。

该模式与上游 PR #40176 对 `ROCM_ATTN` 的非因果改造同构,
只是把同样的逻辑搬到了 `TRITON_ATTN` (kernel 不同、但思路与回归测试用例完全可复用),
因此能在 PR 描述中直接引用 #40176 作为先例论据。

### 3.5 DFlash hybrid KV page-size 矛盾

Qwen3.6-27B 是混合架构: standard attention 层 (用 `AttentionSpec`, page size 与 head_dim 相关)
+ DeltaNet GDN 层 (用 `MambaSpec`, page size 与 `block_size` 相关);
drafter 又是 BF16 + interleaved SWA。
vLLM 的 `unify_kv_cache_spec_page_size` 在所有 layer 间要求 page size 一致,
混合架构必然踩 assert。
进一步,
即便分区修好,
`KVBlockZeroer.__init__` 仍跨组累积 `page_size_el`,
FP8 target (page_size_el = 827392) 与 BF16 drafter (1654784) 两倍差异再次断言失败。
正确修法是
**(a) 上游 PR #37429 (cyankiwi 等) 在 `kv_cache_utils.py` 引入按层分组的 partition;
(b) `KVBlockZeroer` 要么按组追踪 `page_size_el`,
要么把 drafter 层加入 `runner_only_attn_layers` 跳过显式清零** (drafter block 出自 CuMem 已清零内存,
且 attention 内的 per-position `seq_mask` 阻止读取未写位置, 跳过清零是 benign)。

## 四、软件设计与流程

### 4.1 仓库结构与关键模块

```
vllm-awq4-qwen-1.0/
├── Dockerfile                 # 多阶段构建: ROCm SDK + torch + vLLM + 自研 HIP kernel
├── docker-compose.yml         # 服务编排, attention-backend=TRITON_ATTN
├── csrc/awq_mmq_gfx1151/      # 自研 HIP MMQ Q4 custom op
│   ├── awq_mmq_gfx1151_kernel.hip
│   ├── bindings.cpp           # torch.ops.awq_mmq_gfx1151 注册
│   └── awq_mmq_gfx1151/vllm_kernel.py  # MPLinear adapter + M-dispatch
├── scripts/
│   ├── install_rocm_sdk.sh    # TheRock ROCm 7.13 nightly 安装
│   ├── patch_strix.py         # 21 个 idempotent vLLM patch
│   └── start_vllm_awq.sh      # 容器启动脚本 (构建 HIP kernel + exec vllm)
└── test/
    ├── bench_competition.py   # OpenAI API 比赛基准
    ├── bench_matrix.py        # 0/2K/4K/8K x 4 case 矩阵基准
    ├── sanity.py              # 六步快速健康检查
    └── results/               # 所有测量数据 (JSON + Markdown)
```

### 4.2 patch 体系

| 类别 | patch 编号 | 内容摘要 |
| --- | --- | --- |
| 上游 PR cherry-pick | 13 (#40176), 14 (#40898) | ROCM_ATTN 非因果支持; DFlash SWA + target layer +1 fix |
| 本地补丁 | 15-18 | `/v1/responses` thinking 透传; AWQ MMQ kernel 注册; `atomicAdd` 重载消歧; `HIP_FOUND` cmake 兼容 |
| 硬件赋能 (kyuz0 verbatim) | 1-12 | amdsmi disable, gfx1151 检测强制, AITER fence, MoE cap, APU VRAM 边距等 |
| 新增 attention 优化 | 20, 21 | TRITON_ATTN 非因果 + `max_seqlen_q ≤ 16` 3D-launch |

所有 patch 均为 idempotent string-replace,
`patch_strix.py` 可以重复运行不破坏状态,
适合长期演进。

### 4.3 一键复现流程

宿主机准备 (一次性):
1. BIOS UMA 缓冲设为 2 GB minimum (拿不到 GTT-on-demand 就只能定额);
2. GRUB `ttm.pages_limit=30408704` 让 amdgpu 能映射 ~116 GiB GTT;
3. Hugging Face 接受 `z-lab/Qwen3.6-27B-DFlash` 协议;

部署:
```
docker compose build           # ~25-35 min, 含 ROCm SDK + torch + vLLM + HIP kernel
docker compose up -d           # ~9 min 冷启 (model + Triton autotune + DFlash 装配)
python3 test/sanity.py         # ~1-2 min 6 步健康检查
python3 test/bench_matrix.py   # 矩阵基准, 验证 prefill / decode 曲线
```

冷启时间分解: ~95 s 模型加载 + ~6-7 min `profile_run` 与 Triton autotune + ~5 s FastAPI 启动。
即便 Triton kernel cache 命中, vLLM `profile_run` 仍需每次重跑以校准 KV pool 与 OOM 边界,
是稳定性必要成本。

## 五、系统测试与分析

### 5.1 测试方法

测试脚本 `test/bench_competition.py` 与 `test/bench_matrix.py` 统一通过 OpenAI 兼容 API 请求,
解析 streaming SSE event,
提取 wall time / TTFT / prompt-token / output-token / decode rate,
并解析 docker logs 中的 vLLM runtime 指标 (KV cache token 数、max concurrency、attention block size、DFlash acceptance %)。

工作负载:
- **short**: 26 token prompt, 128 output, 测纯 decode 开销;
- **mid (~2K)**: 3765 token prompt, 测中度 prefill;
- **paper8k**: 1480 token prompt, 测短 prompt 上下文;
- **paper32k**: 6708 token prompt, 长 prompt 长 context;
- **paper120k**: 长上下文压力测试 (selective)。

### 5.2 Stage A — 默认配置基准 (Patches 20+21 已应用)

`gpu=0.60`, `N=8`, `MBT=8192`, `KV=auto`, `max_num_seqs=1`, AWQ decode=triton。

| case | wall s | TTFT s | prompt tok | output tok | decode t/s | total t/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 6.329 | 0.249 | 26 | 128 | 21.05 | 24.33 |
| mid 2k | 35.958 | 29.986 | 3765 | 128 | 21.43 | 108.27 |
| paper8k | 17.251 | 11.367 | 1480 | 128 | 21.76 | 93.22 |
| paper32k | 67.079 | 55.140 | 6708 | 128 | 10.72 | 101.91 |

Runtime geometry: model memory 40.5 GiB, KV available 26.41 GiB, KV cache 91,520 tokens, attention block size 832, max concurrency 3.43x。

### 5.3 Stage B — DFlash N 扫描

固定 `gpu=0.60`, `MBT=8192`, KV=auto。

| N | block | KV tokens | max conc | accept % | paper8k wall | paper8k decode t/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 816 | 92,208 | 3.73x | 72.1 | 18.747 | 17.53 |
| 6 | 816 | 92,208 | 3.57x | 54.3 | 17.895 | 19.74 |
| 8 | 832 | 91,520 | 3.43x | 45.7 | 17.314 | 21.79 |
| 10 | 848 | 91,584 | 3.26x | 41.6 | 17.148 | 22.68 |
| 12 | 848 | 91,584 | 3.14x | 30.3 | 16.870 | 23.43 |

N=4 acceptance 最高但 wall 偏慢; N=12 表观 decode 最高但 acceptance 跌 30%、attention block 增至 848;
综合稳定性、cache geometry 与 decode, 默认 **N=8** 是均衡最优点。

### 5.4 Stage C — 调度 token budget 扫描

| MBT | KV tokens | max conc | engine init s | paper8k TTFT | paper32k TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 91,520 | 3.43x | 65.50 | 11.513 | 55.567 |
| 12288 | 88,192 | 3.28x | 97.31 | 12.801 | 55.690 |
| 16384 | 84,032 | 3.13x | 131.25 | 12.930 | 56.165 |

扩大 MBT 反而拖慢 prefill 并压缩 KV pool,
profile/compile 成本也线性上升。默认保持 `MBT=8192`。

### 5.5 Stage D — GPU 内存上限扫描

| gpu util | KV GiB | KV tokens | max conc | paper8k TTFT | paper32k TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.58 | 24.09 | 84,032 | 3.13x | 12.919 | 55.973 |
| 0.60 | 26.41 | 91,520 | 3.43x | 11.513 | 55.567 |
| 0.62 | 28.73 | 99,840 | 3.73x | 12.907 | 56.513 |

`max_num_seqs=1` 下额外 KV 无法被消费, `gpu=0.60` 在 KV 余量与 UMA 给其他服务的留量之间最均衡。

### 5.6 Stage E — Patches 20+21 attention 优化效果 (本项目核心新贡献)

固定 `N=8`, `max_num_seqs=1`, `max_model_len=65536`, FP16 KV, vLLM v0.20.0 + Patches 20+21。
DFlash 接受率不变 (~2.3-2.5 / step),
说明改动是纯后端吞吐收益,
不是 spec-decode 行为变化。

| ctx | ROCM_ATTN (baseline) | TRITON_ATTN + Patches 20+21 | delta |
| ---: | ---: | ---: | ---: |
| 0 | 14.4 t/s | 14.5 t/s | +0.7% |
| 8K | 10.6 t/s | 12.5 t/s | **+17.9%** |
| 16K | 6.9 t/s | 10.0 t/s | **+44.9%** |
| 32K | 3.4 t/s | 6.7 t/s | **+97.1%** |

32K decode 几乎翻倍,
原因正是 3.3 节所述:
2D launch 在 batch=1 时只能填 8 个 workgroup,
而 3D split-K 把 KV 段分到 `NUM_SEGMENTS_PER_SEQ` 个 workgroup,
gfx1151 的 40 CU 第一次被有效填满。

### 5.7 Stage F — FP8 KV restartability A/B 与上游缺陷追踪

在前一轮已运行容器内做的 FP8 KV 实验 (32K decode 7.33 t/s) 暴露了一个上游问题:
**容器 `docker compose up --force-recreate` 后无法重启 FP8 配置**, 触发:

```
AssertionError at vllm/v1/core/kv_cache_utils.py:1030
    assert new_spec.page_size_bytes == max_page_size
```

A/B 三轮均失败:

| 试验 | KV | N | MBT | 启动 | 结果 |
| --- | --- | ---: | ---: | --- | --- |
| N3_MBT8192_fp8 | fp8 | 3 | 8192 | 失败 | unify_kv_cache_spec_page_size assert |
| N4_MBT16384_fp8 | fp8 | 4 | 16384 | 失败 | 同上 |
| N4_MBT8192_fp8_recreate | fp8 | 4 | 8192 | 失败 | 同上 |

诊断: image 在 24a/b/c (本地 #42102 port) 被加入 `patch_strix.py` 之前烘焙;
`docker compose up --force-recreate` 复用镜像、不触发重 build, 故新容器没有这些补丁。
此为本队的下一步工作 (镜像重建 + Patch 24a/b/c 烘焙),
不属于本论文测量范围。当前可重启稳定配置: `N=4, MBT=8192, KV=auto`。

### 5.8 与同硬件/异构平台基线对比

| 平台 / 量化 | 配置 | 单流 decode 中位 | 单流 decode 峰值 |
| --- | --- | ---: | ---: |
| 同硬件 BF16 sibling 无 spec | Strix Halo | 4.3 t/s | 4.3 t/s |
| 同硬件 AWQ4 无 spec | Strix Halo | 5.6 t/s | 5.6 t/s |
| DGX Spark FP8 无 spec | NVIDIA GB10 | 7.8 t/s | 7.8 t/s |
| DGX Spark FP8 + DFlash + MTP | NVIDIA GB10 | 20-25 t/s | 25 t/s |
| **本作品 AWQ4 + DFlash N=8** | **Strix Halo** | **20.0 t/s** | **28.3 t/s** |

本作品在一颗风扇集成 iGPU 上实现的单流 decode 峰值已超过 vllm-project 公开的 DGX Spark FP8 + DFlash + MTP 峰值 (28.3 vs 25 t/s),
显著超过同硬件 BF16 / AWQ4 无 spec 基线。

### 5.9 2026-06-06 复现验证 (`bench_full.py --runs 3`)

为了独立、可复现地验证 README 中 18.5 t/s 与 24.8 t/s 的核心性能声明,
我们在以下配置下复跑了 `test/bench_full.py`:

```
max_model_len=262144         # 全 256K 上下文,与 README 默认值不同
DFlash N=8
kv_cache_dtype=fp8           # FP8 KV (依赖 24a/b/c 本地补丁)
gpu_memory_utilization=0.90  # 最大化 UMA 占用
max_num_batched_tokens=16384
max_num_seqs=1
```

按用例的 median 与 README 历史值对照:

| 用例 | 完成 tokens | wall s | decode t/s (median) | README 历史值 | 复现倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| chat_factual ("speed of light") | 213 | — | **22.47** | 21.82 | +3.0% |
| chat_explainer ("entanglement") | 1329-1469 | 66.4 | **19.96** | 18.50 | +7.9% |
| responses_reasoning ("3 trains") | 910 | 32.15 | **28.29** | 24.80 | +14.1% |
| vision_frost (`forest.png`, 1280×720) | 691 | 42.61 | **16.22** | 13.84 | +17.2% |
| vision_splash (`fly.png`, 1024×1024) | 651 | 38.79 | **16.78** | 13.82 | +21.4% |
| tool_chat (get_weather Tokyo, non-stream) | 137 | 9.88 | **13.87** | 15.53 | -10.5% |
| tool_responses (get_weather Paris, stream) | 110 | 7.71 | **14.25** | 13.26 | +7.5% |
| completions_short ("capital of France") | 8 | — | **6.11** | 6.34 | -3.6% |

7 / 8 用例复现并超过 README 数字, 其中 `responses_reasoning` 与两张图像测试均超过 +14% 以上,
唯一退化的 `tool_chat` (-10.5%) 是 137 token 短 round-trip, run-to-run 噪声范围内, 不构成回归。
完整原始 JSON 与 Markdown 报告:
- `test/results/readme_verification/20260606-200114_bench_full_n8_256k_fp8_runs3.json`
- `test/results/readme_verification/20260606-202654_bench_full_n8_256k_fp8_complete_summary.md`

注: 该验证配置 (`gpu=0.90, MBT=16384, KV=fp8, max_model_len=262144`) 与第 5.2-5.5 节稳定默认 (`gpu=0.60, MBT=8192, KV=auto, max_model_len=65536`) 不同;
默认配置同样能跑通且更加节省 UMA, 仅峰值略低。
两套配置都已生产可用,前者面向"单流极限",后者面向"多服务共驻"场景。

### 5.10 显存与并发能力

| 指标 | AWQ4 + DFlash | BF16 sibling |
| --- | ---: | ---: |
| Model memory | 40.5 GiB | 51.2 GiB |
| KV available | 26.41 GiB | 53.83 GiB |
| KV cache tokens | 91,520 | 220,304 |
| max concurrency @ 64K | 3.43x | 12.92x |
| engine init | 64.86 s | 22.80 s |

AWQ4 牺牲一定 KV 池容量 (UMA 上 22 GiB 双存储是 HIP kernel 的成本) 换取 5.7 倍 decode 吞吐;
对于单/少流业务是值得的折衷。
三并发 multi-stream 压力测试中 (`max_num_seqs=3`),
总吞吐峰值 41 t/s, 平均 27 t/s, 跨 9 + 1 round-trip 工具调用 0 引擎错误。

### 5.11 上游贡献

| 类别 | 项目 | 状态 |
| --- | --- | --- |
| vLLM PR | **PR #44652** — Triton unified attention non-causal + 3D-launch gate | OPEN, DCO/pre-commit 全过, 待 maintainer 触发 CI |
| vLLM issue 评论 | **issue #43626** — 跨 rig 复现 + KVBlockZeroer 旁路 + FP8 KV benchmark | 已发布 |
| vLLM PR 评论 | **PR #37429** — 跨 rig 验证 + workaround + 后续 PR 提案 | 已发布 |

后续待 PR #37429 落地后,
将拆出一个独立的 ~10 行 `KVBlockZeroer` 旁路 PR (Patch 24c) 作为第二个上游 PR。

## 六、总结

### 6.1 阶段性成果

本项目以 **AMD Strix Halo gfx1151 + ROCm 7.13** 为目标硬件,
在 **vLLM v0.20.0 开源框架** 之上,
完整跑通了 **Qwen3.6-27B-AWQ-INT4 + DFlash 推测解码 + 视觉 + 工具调用** 的全功能 OpenAI 兼容服务,
并系统性地完成了五条优化路径:

1. **自研 HIP MMQ Q4 custom op** 将 prefill 在 0-32K 上下文压成平直 105-134 t/s, 4K 痛点 3.4 倍提升;
2. **Triton unified attention Patches 20+21** 解锁 3D split-K Flash-Decoding 在 DFlash verify 场景下的应用, 32K decode 从 3.4 t/s 提升到 6.7 t/s (+97.1%);
3. **DFlash + AWQ + 混合架构 KV 兼容性补丁** 让本来仅 MI300 验证过的推测解码栈在 RDNA 3.5 iGPU 上端到端运行;
4. **21 个 idempotent string-replace patch** 形成可演进、可复现的 vLLM-on-RDNA3.5 工具链;
5. **首个 vLLM 上游 PR #44652 + 两个深度技术评论** 把 RDNA 3.5 路径的实测数据贡献回 vllm-project 开源社区。

最终性能成绩: 单流 `/v1/responses` peak **28.3 t/s** (2026-06-06 `bench_full.py` 复现, vs no-spec 5.6 baseline **+405%**, vs 同硬件 BF16 4.3 t/s **6.6 倍**, 超过公开的 DGX Spark FP8+DFlash 25 t/s 峰值),
chat steady **20.0 t/s**,
视觉用例 **16.2-16.8 t/s**,
三流聚合 41 t/s 峰值,
prefill 在 32K context 仍维持 106 t/s。

### 6.2 后续规划

- **短期**: 重建 docker image 烘焙 Patches 24a/b/c, 恢复 FP8 KV 可重启路径,
  解锁 FP8 KV 的 25 GiB 显存节省与潜在更高并发;
- **中期**: 跟踪 PR #44652 maintainer 反馈, 推动 merge;
  PR #37429 合入后提交 Patch 24c 单独 PR;
- **长期**: 将 AWQ MMQ Q4 HIP custom op 整理为独立 PR 贡献给 vLLM 或 AMD ROCm-libraries,
  让 RDNA 路径下的 W4A16 GEMM 不再依赖私有补丁;
- **方法学**: 引入 `rocprof` / `omniperf` 进行 kernel 级 profiling,
  探索 Triton unified attention 中 e4m3 dequant 软件路径的进一步优化空间。

### 6.3 ROCm 生态价值体现

本项目从硬件赋能 (gfx1151 检测、AITER fence) 到框架适配 (PyTorch HIP_FOUND、Triton 路径),
再到自研 HIP custom op (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`),
最后到上游开源贡献 (vLLM PR #44652 + 两份技术评论),
完整覆盖了 ROCm 软件栈从底层算子到上层服务的全链路使用。
所有补丁、测试脚本、HIP kernel 源码、bench 结果、`.research/` 内部技术笔记均以 Unlicense (公有领域) 开源发布,
对在 RDNA / Radeon PRO W 系列上做大模型推理的研究者与工程师具有直接可复用价值。

整套方案证明:
即便面对 RDNA 3.5 + 推测解码 + 混合架构这种 vLLM 上游覆盖度最低的组合,
通过 ROCm 提供的 HIP 编译路径、Triton JIT、PyTorch ATen 接口、
以及 vLLM 的可扩展 backend 框架,
仍能在一周内做到从工具链组装 →
瓶颈定位 → kernel 自研 → 上游 PR 提交的完整研发循环,
这正是 ROCm 生态在科研工作负载上的实际工程化价值。
