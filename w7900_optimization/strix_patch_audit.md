# Strix Halo 补丁到 Radeon PRO W7900 的适用性审计

基线：8× Radeon PRO W7900D，gfx1100，ROCm 7.14，vLLM 0.23.1.dev1。
原则：不执行 `scripts/patch_strix.py`；仅移植有明确正确性需求或在 gfx1100 A/B 中证明收益的改动。

## 补丁矩阵

| Patch | Strix Halo 目的 | W7900 处置 | 依据 |
|---|---|---|---|
| 1 / 1.5 | 禁用 amdsmi、强制 ROCm/gfx1151 | 删除 | W7900 的 amd-smi、rocminfo 和 gfx1100 探测均正常；强制 gfx1151 会生成错误 ISA |
| 2 / 3 / 3.5 / 5 | 在 gfx1x 强开 AITER，同时绕开不稳定 RMSNorm/MoE | 默认不应用 | 当前容器无可用 AITER，原生/Triton 路径已运行；后续只能以独立 AITER 实验验证 |
| 6 / 7 / 8 | 旧版 fusion、Triton/AITER/flash-attn 兼容 | 不应用 | vLLM 0.23 + Triton 3.7 已构建和运行，无对应错误 |
| 9 | 放开 gfx11 Triton MoE | 当前无关 | 当前 27B target 不走该 MoE 路径 |
| 9.5 | Clang 23 spinloop include 修复 | 已上游等价解决 | 当前源码 `spinloop` 已编译成功 |
| 10 | APU GTT/UMA 动态显存上限伪装 | 必须删除 | W7900 是独立 48 GiB VRAM；伪造 `mem_get_info` 会破坏 KV 容量计算 |
| 11 | 屏蔽 hipCtx deprecated 警告 | 可选、无性能作用 | 当前构建仅告警，未失败 |
| 12 | GGUF qwen35 标签 | 默认不应用 | 当前使用 safetensors；仅 GGUF 实验需要 |
| 13 / 14 | ROCm non-causal attention、SWA Draft | 不移植旧实现 | 当前 vLLM 0.23 的 DFlash + 4 SWA/1 full Draft 已启动并推理，接受率可测 |
| 15 | Responses API chat-template 参数 | 与 W7900 无关 | 仅 API 功能补丁 |
| 16 | 注册 gfx1151 AWQ4 HIP MMQ | 重新实现为 gfx1100 实验后端 | 不能直接复用旧注册和路由；见下节 |
| 17 / 18 | ROCm atomicAdd / HIP_FOUND 编译兼容 | 不应用 | 当前 ROCm 7.14 完整源码构建成功 |
| 19 | Triton attention iteration tile 32→64 | 仅作为 A/B 候选 | 这是 Strix 实测参数，不应直接成为 W7900 默认值 |
| 20 / 21 | Triton non-causal、DFlash verify 3D gate | 不移植旧实现 | 当前 DFlash N=2/4 已真实生成，说明 v0.23 路径已有等价能力 |
| 22 | softmax segments 16→32 | 不采用 | 原项目已标注为候选/负结果 |
| 24a/b/c | DFlash 独立 KV group、Draft BF16 KV、zeroer 规避 | 研究分支，不进基线 | #42102 已关闭未合并；当前 v0.23 在 FP8 target KV 下已启动并推理，应先做 dtype/显存/接受率 A/B |

已纳入基线的唯一额外源码修复是上游已合并 #45207：对 Mamba page 使用 `page_size_padded`，保持 `block_size` 不变。

## AWQ4 HIP MMQ 的 gfx1100 迁移判断

内核主体没有 `__gfx1151__` 架构条件，使用 gfx11 wave32 的 fp16 WMMA intrinsic；硬绑定集中在：

1. `setup.py` 的 `TARGET_ARCH = "gfx1151"`；
2. 包、namespace、torch op 和 launcher 的 gfx1151 命名；
3. vLLM v0.20 的 MPLinear 注册接口；
4. 基于 Strix Halo 测量得出的 tile、路由阈值和双权重布局策略。

现有候选：

| 版本 | Tile | Waves/CTA | 说明 |
|---|---|---:|---|
| v5 | M16×N32×K32 | 2 | 小 M、较低 LDS/并行度 |
| v6 | M16×N32×K32 | 2 | 增加 scale/zero metadata LDS staging |
| v7 | M16×N64×K32 | 4 | Strix 最佳通用候选，A/metadata 摊销更好 |
| v8 | M8×N128×K32 | 8 | Strix 短请求略优，但中长请求回退 |
| v9 | M16×N64×K32 | 4 | B LDS XOR swizzle；Strix 全档回退 |

W7900 首轮应同时编译 v5/v6/v7/v8/v9，以 TritonW4A16 为基线，分别测试 M=1/2/4/8/9/16/32/64/128 及真实模型 K/N 形状。不得预设 v7 在 W7900 仍最优。

旧 adapter 默认 `M<32` 走 Triton、`M>=32` 走 HIP，并为两条路径保留 native 与 K-major 两份权重。W7900 需要重新评估：

- 是否只为命中的 K/N 层保存第二布局；
- Decode 与 DFlash verify 是否应按 M 和上下文长度动态路由；
- 双布局显存成本是否挤压 128K/256K KV Cache；
- Tile 对 LDS bank conflict、occupancy 和显存带宽的影响；
- TP=2/4 下算子节省是否被 RCCL all-reduce 抵消。

## 运行参数审计

| 参数 | 当前用途 | W7900 下一步 |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | Strix 兼容伪装 | 禁止使用；W7900 原生报告 gfx1100 |
| `HSA_NO_SCRATCH_RECLAIM=1` | Strix AWQ 稳定性 | A/B 测试后决定，不默认视为最优 |
| `VLLM_ROCM_USE_AITER=0` | 当前稳定基线 | 保持，AITER 单独实验 |
| `AWQ_MMQ_ENABLE=0` | 尚未有 gfx1100 内核 | 保持，直到 correctness + microbench + E2E 均通过 |
| `--enforce-eager` | Bring-up 降低失败面 | 不是最终性能配置；后续测试 compile/CUDAGraph |
| `--skip-mm-profiling` | 纯文本长上下文 | 纯文本保留；多模态赛项单独配置 |

## 多卡与 KV Cache 实验顺序

1. TP=2 同 NUMA：Target-only、DFlash Draft TP=1/2、N=2/4/8。
2. TP=4 同 NUMA：关注 RCCL 占比、每卡权重、128K/256K KV 容量。
3. TP=8 跨 NUMA：只作为扩展性上限；与 4×TP2 多实例吞吐比较。
4. KV dtype：`auto` 与 `fp8`，同时记录容量、接受率和准确率；不能只报显存节省。
5. 上下文感知 DFlash：当前 24K prompt 下 Target-only 优于 N=2/N=4，需建立启停阈值。
