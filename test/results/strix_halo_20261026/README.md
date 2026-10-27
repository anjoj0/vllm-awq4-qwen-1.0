# 2026-10-26～27 Strix Halo 部署与优化实测

这是本机受限范围的工程选择记录，后续模型适配研究以这组数据为基线。它不构成 DSpark、DFlash2 与 DFlash 的普遍算法排名。

## 最终部署复核

默认 `optimized-v0320`（v6）、DFlash N4、FP16、单请求、前缀缓存开启。以下性能请求均 fresh salt、实际缓存命中 0；完整条件与每行对照文件见 [汇总 JSON](performance-final.json)。

| 指标 | 对应优化前记录 | 最终配置 | 提升 |
| --- | ---: | ---: | ---: |
| 4K prefill token/s | 190.70 | **372.39** | 95.3% |
| 8K prefill token/s | 234.31 | **363.07** | 55.0% |
| 短输入代码 decode token/s | 19.15 | **27.54** | 43.8% |
| 短输入中文 decode token/s | 12.10 | **17.42** | 44.0% |
| 8K 编程输入 decode token/s | 15.78 | **22.60** | 43.2% |

各行不是同一个初始配置的统一扫描：8K prefill 对照已有大 tile+down，8K decode 对照 v4，短输入 decode 对照原生 N4/block64。Prefill 取两次中的第二次，decode 为两次中位数；短输入输出 cap512，8K 编程 cap256。最终 4K/8K TTFT 分别 11.03/22.57 秒。前一 v4 的 4K 375.04 也保留在过程记录中，当前默认成绩用最终复核值。

最终六项 API 冒烟通过；展示页 Responses SSE、服务端指标、取消/恢复通过；二分查找修复代码经审阅，通过生成的 3 个 assert 和额外 5 个边界测试。另做了一次前缀状态正确性检查，缓存 token 从 0 到 2496，输出不变；未将缓存命中作为上述性能提升。见 `final-functional-check.json`、`final-demo-check.json`、`final-demo-code-check.json`。[部署配置](selected-config.json)保存镜像 ID、固定权重、全部优化开关和回退镜像。

## 后续补测：2048-token 输入

沿用相同 `optimized-v0320` 镜像、DFlash N4、单请求、greedy、thinking off，未修改服务配置。Prefill 使用固定原始 token ID、输出 8 token；decode 使用补齐至 2048 token 的 LRU 编程 chat、输出 256 token。每项两次，每次 fresh salt，实际输入均为 2048、cached_tokens 均为 0。

| 指标 | 第一次 | 第二次 | 采用值 |
| --- | ---: | ---: | ---: |
| Prefill token/s | 263.64 | 290.09 | **290.09**，第二次稳态 |
| Prefill 请求客户端 TTFT | 7.77 s | 7.06 s | **7.06 s** |
| 代码 decode token/s | 23.32 | 23.90 | **23.61**，两次中位数 |
| 代码请求 TTFT | 7.52 s | 7.12 s | 分别记录 |

代码请求的平均接受长度均为 3.52、proposal 接受率均为 63.01%。Decode 仍以 `(output_tokens−1)/server generation_time` 计算，不包含 prefill；它与上述 prefill 项使用不同的固定提示词。2K prefill 的本次结果低于既有 4K/8K 吞吐，按实际结果记录，没有按输入长度线性外推。见 [2K 汇总](performance-2k.json)、[prefill 原始记录](prefill-final-v6-2048.json)和 [decode 原始记录](decode-long-dflash-final-v6-2k.json)。

## 环境和口径

Ryzen AI Max+ 395 / gfx1151，系统 RAM 约 122 GiB，ROCm 可见 61.34 GiB。vLLM 0.32.0+rocm723、Torch 2.13.0+git733fca1、HIP 7.2.53211、V2 runner、eager、单请求、32K 上限、MBT8192、auto KV、prefix caching 开启。初始草稿比较使用同一 AWQ target 和上游 RDNA hybrid 目标算子；后续专用算子按版本单列。基础镜像 digest、各权重 revision 与配置见 [环境记录](acceptance-environment.json)。

每种配置两个固定短输入任务，各两次、最多 512 输出 token；temperature=0、thinking=false、新 cache_salt、cached_tokens=0。Decode = (output_tokens−1)/server generation_time。表为两次中位数；完整请求、输出、每次耗时与草稿计数见相邻 JSON。DFlash2 只完成第一条代码样本就停止追加。后端顺序测试，未做交错多轮，故小差距不宜过度解释。输出受 cap 截断，不作为完整 LRU 代码质量验收。

## 草稿与验证成本

| 配置 | 代码 decode token/s | 中文解释 decode token/s |
| --- | ---: | ---: |
| Target-only | 10.88 | 10.88 |
| DFlash N=8 | 11.89 | 6.38 |
| DSpark full N=8 | 11.28 | 6.17 |
| DSpark 49k N=8 | 11.89 | 4.75 |
| DFlash2 N=7（单条止损） | 2.48 | 未追加 |
| DFlash N=4 / auto block | 19.61 | 12.38 |
| DFlash N=4 / block minimum 64（原生） | 19.15 | 12.10 |
| DSpark full N=4 / block minimum 64（后续定向复核） | 18.43 | 11.85 |

上表初始对照的 512-token 样本均与 target-only 的同题文本一致。平均接受长度按 vLLM 的 `1+accepted/draft_rounds` 计数约定记录；请求尾部可能裁掉多余 proposal，不能直接把它当作实际提交长度。

原版 DFlash 与 DSpark full 在 N=8 的平均接受长度接近，速度差距更应追查执行成本。49k 中文接受长度约 1.86，而 full 与 DFlash 约 2.58。DFlash2 提议 3,563 个 token，只接受 2 个；其 config 经实际解析确认 feature layers=[5,19,33,47,61]、N=7、SWA2048、独立 RoPE，运行精度随 target 为 FP16。原因尚未定位，不能从该异常推出算法不如 DFlash。模型卡中的 BF16 训练目标和本机 AWQ/FP16 条件不同。

[v0.32 RDNA hybrid 源码](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py)的 skinny 路径要求 M≤5 及 K×M 上限；N=8 的典型 M=9 不满足。受这个新分界启发，只补测了 DFlash N=4，没有重扫旧 N 网格。N=4 降低每轮成本，虽每轮接受长度较短，decode 反而更快。未经请求内 profiler 分解，不把全部收益都归因于某一个 GEMM。

## Prefill 的独立选择

第一轮相同 N=8 的最小迁移对照：上游 RDNA hybrid 为 182.28 token/s，项目 HIP v1 为 124.32 token/s；均为第二条 4096-input/8-output 请求、无缓存命中。MMQ 模型占用 35.76 GiB，上游为 22.45 GiB。旧 HIP 编译和两项随机数值检查通过，但未启用为默认，也没有重新扫描 v2–v9。

N=4 的 auto block 为 816，4K prefill 下降至 148.89 token/s。随后使用保存了完整 token ID 的同一脚本、同一 4096-token 输入复核：

| 配置 | 物理 block | 第二条 prefill token/s | 客户端 TTFT 秒 |
| --- | ---: | ---: | ---: |
| DFlash N=4 / auto | 816 | 148.89 | 27.53 |
| DFlash N=8 / auto 复核 | 832 | 213.18 | 19.33 |
| DFlash N=4 / minimum 64 | 832 | 196.94 | 20.92 |

最小块对齐将 N=4 的 prefill 提高约 32%，没有通过缓存命中放大吞吐。当前配置仍比本轮 N=8 的 prefill 慢约 8%，但两个 decode 任务明显更快。最早的 182 与后来的 213 不是同一轮稳态数字，不据此宣称额外 17% 提升。

## 当前配置与展示验收

最初验收的原生配置是 DFlash N4/block64、模型 22.45 GiB、KV 预算 8.56 GiB。后续优化仍为 DFlash N4/物理 block832，但启用下文经过验证的投影与 attention 适配，模型 25.63 GiB、KV 预算约 5.61 GiB。`optimized-v0320` 用作展示默认，旧 `stable-v0320` 保留；[最终配置](selected-config.json) 与根目录 `.env` / helper profile 对齐。

六项后端接口检查覆盖 health、models、算术、Chat SSE、Responses SSE、强制函数调用；[通过记录](functional-checks.json)仅代表对应接口，不是知识评测。流式的 Strix Halo 问答曾给出错误事实，因此没有把非空输出记作语义质量通过。

展示页 http://127.0.0.1:8088、模型 API http://127.0.0.1:8000/v1。真实 Responses 流、请求归属明确的阶段指标、取消和恢复均通过。二分查找修复代码经审阅，通过模型给出的 3 条断言和额外 5 个边界用例。见 `harness-*.json`。展示页是编程问答界面，没有文件读写或命令执行的 Agent 循环。

## 10-27：模型适配与公开 prefill 研究

后续 DSpark full N=4 / block minimum 64 四条请求均与 target-only 完全一致。代码中位数 18.43、中文 11.85 token/s；相对该草稿 N=8 分别提高约 63% 和 92%。接受长度分别约 3.77 和 2.44。这是验证宽度与新版目标算子的适配收益，不能单独证明模型重新训练必要或不必要；与 DFlash N=4 仅有小差距，未做交错多轮，不给算法排名。

公开的模型契约错误确实存在：[satgeze DSpark 模型卡](https://huggingface.co/satgeze/Qwen3.6-27B-DSpark)报告旧导出误继承 target 的 partial RoPE，使 vLLM 接受率降至约 1%，修正为 full-dimension RoPE 后恢复。当前 DFlash2 是 BF16 non-thinking 目标上的早期适配 checkpoint，而本机为 AWQ/FP16；作者明确尚未做 serving benchmark，见[固定模型卡](https://huggingface.co/vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1/blob/a233ec768abb2ed9a2a97a294b5808d0ad10932e/README.md)。本机配置检查未发现错误特征层或 target RoPE 继承，但 dtype、卷积/selector 中间值和 target 特征数值尚待排查。

[vLLM #58894](https://github.com/vllm-project/vllm/issues/58894)报告 prefix hit 后 DFlash2 接受率归零，作者后续确认 #55601 有效；该 PR 已合并。本机首次 fresh-salt/cached=0 请求就异常，因此没有把该 issue 直接当成本机根因。当前安装源码已使用 DFlash2 decoder subclass 和 candidate selector，不能将早期 decoder class 被忽略的旧问题当作已确认缺陷。

以下是本次检索找到的可追溯公开参考，不是全网最高纪录证明，也不是同权重同输入的性能排名。均为单台 Strix Halo、dense 27B；大于千 token/s 的 MoE 数字未混入。

| 公开路线 | Prefill token/s | 重要条件 |
| --- | ---: | --- |
| [ROCmFPX 历史作者对照](https://github.com/charlie12345/ROCmFPX/blob/fb08d7cdb67082ddda15a25e64dc7ec413afc41c/README.md#qwen36-27b-vanilla-no-mtp) | 376.98 | Qwen3.6-27B ROCmFP4 STRIX_LEAN，Vulkan，pp512，无投机；同表 ROCm 为 328.03 |
| [Gufo](https://github.com/gufo-org/gufo/blob/1c62f0a9f96aec33c8128bf652eff312bdf56f90/docs/models/qwen3.8-27b/BENCHMARKS.md) | 634.59–648.89 | Qwen3.8-27B Q4_K_XL，HTTP，pp2048，空历史 AR；两个不同日期的记录，不是误差区间；开启 DFlash2 的同口径约 580–586 |
| [Halogen 作者报告](https://github.com/peonist-ai/halogen-server/blob/5a0f952c9e07794a0de0a617a377bae6a63ca56a/README.md#prefill) | 710 / 566 | Qwen3.8-27B，HTTP pp2048 / 完整 pp32768；专用权重与运行时，prefill 使用 W4A4 |

Halogen 的 [量化说明](https://github.com/peonist-ai/halogen-server/blob/5a0f952c9e07794a0de0a617a377bae6a63ca56a/docs/QUANT.md)区分 decode 约 6.32 bpw 与 prefill 的 INT4 activation；不能将其速度归为本项目 AWQ W4A16 的无损等价实现。Gufo 的原 target/GGUF 转换质量也未全部资格验证，见其 [质量边界](https://github.com/gufo-org/gufo/blob/1c62f0a9f96aec33c8128bf652eff312bdf56f90/docs/models/qwen3.8-27b/QUALITY.md)。这些成绩足以提示优化空间，不能直接承诺本机 4K 达到同一数字。

可复用机制：Gufo 的 [FP16 prefill 内核](https://github.com/gufo-org/gufo/blob/1c62f0a9f96aec33c8128bf652eff312bdf56f90/src/models/qwen/hip/kernels/prefill_fp16.hip)采用大矩阵分块、片上解码量化权重、WMMA 和融合 epilogue；[执行链](https://github.com/gufo-org/gufo/blob/1c62f0a9f96aec33c8128bf652eff312bdf56f90/src/models/qwen/hip/prefill_chunk.cpp)还减少重复 KV 转换和末层无用输出。它也有 A8 路由，不能把所有路径都描述成同一种精度。[Frontier Lab 的 Vulkan 消融](https://thefrontierlab.ai/frontier-watch-builds/)显示连续化 KV 可显著改善深上下文 prefill；其 pp@32K 是已有 32K 历史后追加 512 token，不等于完整冷 32K 输入。

据此只对本机两种主要 FFN 形状补了小范围 tile probe，复用上游 group32 算术：4096×5120→34816 的 256×256/32-warps tile 为原路径约 2.66×，无编译器报告的 spill；17408→5120 的 down 投影仅约 6%，暂不改。随机非对称权重、分散输出列 FP32 oracle、前后同源计时对照见 [probe](prefill-tile-probe.json)。这些只是单层结果，整模型收益须另测。

## 10-27：本机形状适配过程

以下均继续使用同一 target revision、FP16、DFlash N=4、minimum block64、fresh salt。这里保留优化过程中的对照，最终选择以 `selected-config.json` 为准。

| 配置 | 4K 无缓存 prefill token/s | 代码 decode token/s | 中文 decode token/s |
| --- | ---: | ---: | ---: |
| 同轮关闭 tile 的原生路径 | 190.70 | — | — |
| 仅 gate/up 大 tile | 231.62 | — | — |
| 四类投影大 tile | 294.45 | — | — |
| 大 tile + down 第二布局 | 294.03 | 26.27 | 16.72 |
| 关闭 down 布局的复核 | — | 19.48 | 12.31 |
| 上述布局 + decode full graphs | — | 26.18 | 16.69 |
| 再加入大投影逐层反量化 + rocBLAS | 311.54 | — | — |

Prefill 为每组第二条相同 4096-input/8-output 请求；所有 4K 输入 ID 和输出 hash 相同、cached_tokens=0。相对同轮 190.70，扩展 tile 提高约 54%，rocBLAS 路径约 63%。大 tile+down 在首次 8K 实测为 234.31 token/s，不将 4K 的提升比例外推到长文。

Decode 为各两条 512-token 样本的中位数。down-on 和 graphs 两组都是前三条文本与 target-only 一致，第二条中文样本在后段出现措辞差异；down-off 四条均一致。不能宣称全部整模型输出逐位一致，差异尚未严格归因。随机权重检查和目标第 0/31/63 层的真实权重检查中新旧 down 算子输出均逐位相同；这只是局部数值证据。见 `shape-tuned-output-comparison.json`、`down-real-validation.json` 和各完整请求 JSON。

修改范围：

- 上游 tile 针对其他模型/group size 的默认值未覆盖本模型。只在 gfx1151、FP16、非对称 group32、M=1536..8192 及四种实际投影形状启用 256×256 / 32-warps tile。真实 4K prefill 由于 Mamba 对齐被拆为 2496+1600，因此仅改 M≥2048 的第一版遗漏了后一段。
- K=17408 的 down 投影在 M≥3 时越过原 HIP LDS 限制；为 64 层单独保存 K-major 权重，在 M=3..9 使用已有 Triton 算子，其余尺寸走原路径。模型内存 22.45→25.63 GiB，KV 预算约 5.61 GiB、可分配约 54K token，仍覆盖单请求 32K。未复制全模型第二布局。
- 大投影可逐层暂时解码为 FP16，再调用 rocBLAS；无 activation INT4 量化，无全模型 FP16 常驻副本。单层 probe 包含每次反量化成本，实际服务另测。小投影仍保留融合路径。
- 新 ROCm/vLLM 栈的单请求 target 与 DFlash full graph 捕获及接口 smoke 通过，但此次 decode 没有获得明显收益，不能套用旧版“必然卡住”的结论，也不把 graphs 列为已证实的提速来源。未验收 adaptive verification。

GPU 数值检查见 `prefill-expanded-validation.json`、`rocblas-validation.json`、`down-layout-validation.json`、`down-real-validation.json`。32 个 prefill shape/boundary 检查均通过 FP32 oracle，新旧结果逐位相同；真实权重 down 的小图回放也通过。微基准见 `prefill-actual-shapes-probe.json`、`verify-layout-probe.json`、`dequant-matmul-probe.json`。

## 10-27：DFlash2 FP16 故障已在本机捕获

新增定位依据是 [vLLM #55250](https://github.com/vllm-project/vllm/issues/55250)：报告者逐层追踪出 DFlash2 的 FP16 overflow，另有贡献者在 CUDA 上复现。[对应 #55294](https://github.com/vllm-project/vllm/pull/55294)仍为 Open，提议对这类配置直接报错，而非已经合入的数值修复。

对本地 Qwen3.6 DFlash2 v0.1 的真实首个请求，在不改变算术的诊断插桩中观察到：第一层 MLP 输出最大绝对值 62784，经过 `mlp_conv.finish` 后 40960 个元素中有 7 个变成非有限值；进入下一层后 NaN 扩散，后续层失效。128-token 诊断请求 889 个 proposal、0 个接受。见 `df2-fp16-stage-finiteness.json` 与 `df2-fp16-finite-diagnostic.json`；同步插桩耗时不能用于正式速度比较。

BF16 复核时，同时将 target 与 draft 改为 BF16；首两次 draft forward 的所有记录均为有限值。第一层 `mlp_conv.finish` 的幅值达到 233472，解释了 FP16 的范围不足。同题 128-token 诊断得到 62/455 个 proposal 接受，接受率 13.63%、平均接受长度 1.95。见 `df2-bf16-stage-finiteness.json` 和 `df2-bf16-finite-diagnostic.json`。这是恢复工作能力的证据，包含插桩和新 dtype，不能作为与 DFlash 的正式速度 A/B，也未证明该早期检查点有替换价值。生成的 DFlash2 profile 现使用 BF16，避免再次默认落入已确认的 FP16 故障。

## 10-27：Prefill 请求内分项计时

Torch profiler 在此镜像只返回 CPU/API 事件，未获得 GPU kernel 事件；rocprof attach 也未能直接附加，因此未根据它们编造 GPU 百分比。随后在独立 eager 研究容器中，用 HIP events 包围实际目标模块。嵌套范围不相加，插桩请求不计入正式吞吐。

| 输入 | 目标前向总计 | 全部矩阵投影 | 完整 attention 核心 | GDN chunk 核心 |
| --- | ---: | ---: | ---: | ---: |
| 4096（2496+1600） | 12.96 s | 7.74 s | 2.36 s | 1.12 s |
| 8192（6656+1536） | 30.92 s | 14.46 s | 9.33 s | 3.21 s |

范围包含各模块排入当前 stream 的工作以及可能的 launch 空隙，不包含草稿、外层 scheduler 和 sampling；其余差额包含归一化、RoPE、gating、conv、残差与框架间隙。见 `stage-prefill4k.json`、`stage-prefill8k.json`。完整 attention 随输入增长成为值得优先处理的瓶颈。

## 10-27：Attention 形状适配与旧 split-K 优化迁移

本模型目标 attention 为 24 个 query heads / 4 个 KV heads / head_dim256。上游默认 BLOCK_M16 只覆盖 2 个完整 query 位置，默认编译参数还产生大量寄存器 spill。小范围 probe 将 prefill 改为 BLOCK_M128、KV tile32、8 warps、1 stage；保留 KV 累加顺序，并屏蔽不能组成完整 GQA 组的补齐行。gfx1151、FP16、单请求、物理 block832、causal full attention、256≤query≤8192 和总长≤32768 才启用，其余路径保持上游参数。

14 项检查覆盖真实 4K/8K 分块、32K 历史、非连续 query 行、BF16/其他 shape/SWA 回退。与独立 FP32 softmax 的相对 L2 误差均在设定范围内，回退路径使用同一编译内核。上游相同内核重复执行也观察到极小的浮点差异，因此没有将“完全相等”作为所有 attention 路径的结论。见 `attention-validation.json` 与 `test/validate_rdna_attention.py`。

| 完整服务，DFlash N4 | 4K prefill token/s | 8K prefill token/s | 4K / 8K TTFT |
| --- | ---: | ---: | ---: |
| 原生同轮 4K / 大 tile+down 的 8K | 190.70 | 234.31 | 21.60 / 34.97 s |
| 大投影 rocBLAS + prefill attention（v4） | **375.04** | **362.63** | **10.93 / 22.60 s** |

上表第二条请求作为稳态样本，实际 cached_tokens=0，完整输入 ID 与既有对照相同，对应 8-token 输出 hash 相同。4K 相对同轮原生约 +96.7%；8K 相对已有大 tile+down 配置约 +54.8%，后者包含 rocBLAS 和 attention 两项变化。不是单独 attention 的整机贡献，也不是广泛工作负载平均值。证据为 `prefill-dflash-attention-v4-4096.json`、`prefill-dflash-attention-v4-8192.json`。

旧 Patch21 的 split-K 思路值得迁入新版：上游仍以 max_query>1 排除投机验证，而已有 16-segment 算法支持每个 query 的归并。新的独立开关只允许 2..9 query、上述目标形状和足够的 scratch 容量；没有重新扫已否定的 32-segment 配置。20 项检查覆盖 M=2/5/9、256..32768 上下文、scratch 不足/不同 segment 数的安全回退，全部通过。见 `attention-verify-validation.json`。M5 长上下文单层速度可显著提高，但服务收益单独测量，不直接套用内核加速倍数。

原始 probes 及 `.patch` 位于本目录；`.patch` 的输入是基础镜像中 SHA256 `a1d13e015969ce0b80a1024ba9e9703bfa50481cd9aa7d8f893f6cc955e657a1` 的 `triton_unified_attention.py`。它们是实验源码差异，正式安装入口为 `scripts/patch_rdna_attention.py`，带版本/完整源码哈希检查。

整机 split-K 对照已经完成：同一 8192-token 编程输入、greedy、256-token 输出、两次 fresh-salt 请求，decode 中位数 **15.78→22.06 token/s，+39.8%**；每轮平均接受长度同为 3.70，完整输出 hash 一致。这是验证执行变快的证据，而不是靠接受率上升。短输入代码/中文为 **26.59 / 16.92 token/s**；v5 的四条 512-token 文本均与既有 target-only 相同，但不据此否定此前 v3/graphs 样本出现过的差异。见 `decode-long-dflash-attn-v4-8k.json`、`decode-long-dflash-attn-v5-8k.json`、`decode-long-dflash-attn-v5.json`。

继续缩小 down 的 M tile 至 16、N tile64、K tile32、4 warps/1 stage，复用上游原算术。M5 单层中位数约 0.60→0.48 ms；7 个随机形状/边界和三个真实层的 9 个用例均与原生输出逐位一致。v6 完整服务代码/中文中位数 **27.54 / 17.42 token/s**，四条输出与 target-only 相同。见 `down-tile-probe.json`、`down-v6-validation-0.json`、`down-v6-validation-1.json` 和 `decode-long-dflash-attn-v6.json`。这些 tile 参数只作用于已限定的 down 投影，不修改其他模型。

同一 v6 底座，DSpark full N4 + Top-K128 Markov 头路径的针对性复核为代码 **26.75**、中文 **17.33 token/s**；两条文本与 target-only 相同。每题仅一条样本，与 DFlash 的小差距不构成稳定算法排名。相对最初 N8 的差距已大幅缩小，收益包含宽度、目标投影/attention 和 head 路径的共同变化，没有将全部提升记为 Top-K 单项贡献。见 `decode-long-dspark-v6-topk128.json`。

DSpark 49k 在同底座 N4/Top-K128 的单条代码请求达到 **28.67 token/s**，中文为 **13.44 token/s**；文本也均与 target-only 相同。代码的平均接受长度 3.77，中文只有 1.77，说明缩小输出头的收益不足以抵消此中文任务的接受率下降。小样本结果支持保留它作为研究选项，不足以取代展示默认的 DFlash。DSpark 两个生成 profile 均采用 N4/block64/Top-K128，可将 `VLLM_DSPARK_DRAFT_TOPK` 设为空恢复完整 Markov 投影。

## 复现与本轮未做的工作

在仓库根目录：

```bash
bash scripts/strix_halo.sh up dflash
python3 test/bench_strix_decode.py --label new-dflash-check
python3 test/bench_strix_prefill.py --label new-dflash-check
```

每次仅一个后端、无其他推理客户端。脚本拒绝覆盖已有同名结果。切换使用 `up target-only`、`up dspark`、`up dspark-49k`、`up dflash2`；DFlash2 需已准备可选权重/profile。N=8 原始对照可用 `VLLM_SPEC_TOKENS=8 VLLM_BLOCK_SIZE= bash scripts/strix_halo.sh up dflash`。

复用了旧 MBT8192、HIP decode 不胜 Triton、全模型双布局额外约 13.10 GiB 和五项精度评测记录。没有开展缓存命中矩阵、旧 HIP 版本扫描或完整五项质量重跑。未验收 32K 全长生成、多模态、随机采样分布、并发或 DSpark adaptive/full graph。早期全模型 K-major 原型没有启用；后续只给 down 投影增加布局，范围及实测见上节。
