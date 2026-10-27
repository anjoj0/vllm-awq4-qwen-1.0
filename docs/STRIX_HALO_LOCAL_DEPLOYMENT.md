# 当前 Strix Halo 的部署与验收

部署日期：2026-10-26；后续研究延续至 10-27（Asia/Shanghai）。工作目录 `/home/amd/demo01/vllm-awq4-qwen-1.0`，仓库基线 `6f7d50a`。完整请求、输出、配置和数值记录在 [本机结果目录](../test/results/strix_halo_20261026/README.md)。

## 已部署配置

模型 API 为 `http://127.0.0.1:8000/v1`，展示页为 `http://127.0.0.1:8088`。展示配置为 **Qwen3.6-27B AWQ4 + DFlash N=4**，镜像 `vllm-awq4-qwen:optimized-v0320`。在上游 RDNA 算术上加入本模型的投影 tile/rocBLAS、down 专用布局和 attention 适配。`--block-size 64` 被混合模型增至实际物理块 832。开启前缀缓存和 chunked prefill，eager，单请求，32K 上下文上限，MBT8192，auto KV，GPU 利用预算 0.55。模型占 25.63 GiB，KV 预算约 5.61 GiB；旧 `stable-v0320` 镜像保留用于回退。

Ryzen AI Max+ 395 / Radeon 8060S / gfx1151，Ubuntu 26.04.1、内核 7.0.0-30-generic；系统内存约 122 GiB，**ROCm 实际可见 61.34 GiB**。没有修改内核、驱动或 GTT 上限。功率配置为 performance。

实际运行时：vLLM `0.32.0+rocm723`、Torch `2.13.0+git733fca1`、HIP `7.2.53211`、V2 runner。基础镜像固定 digest：

```text
vllm/vllm-openai-rocm:v0.32.0@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905
```

Target-only、DFlash、DSpark full、DSpark 49k、DFlash2 的初始六项接口冒烟均通过。随后已定位 DFlash2 的 FP16 grouped-conv 溢出：改用 BF16 后中间值有限，诊断请求接受率从 0 恢复至 13.63%。这不是公平速度 A/B，也不足以将早期 DFlash2 checkpoint 设为默认。选择 DFlash 不代表给三种算法作普遍排名。

最终默认配置复核：无缓存 4K/8K prefill **372.39/363.07 token/s**，TTFT **11.03/22.57 秒**；短输入代码/中文 decode **27.54/17.42 token/s**，8K 编程 decode **22.60 token/s**。下面保留分阶段对照，最终汇总见 [performance-final.json](../test/results/strix_halo_20261026/performance-final.json)。

## 离线权重

缓存 `/home/amd/hf-cache` 以只读 `/models` 挂载。已核对配置、分片索引、tensor shape/offset、文件长度和 tokenizer 资源；除下述 DFlash2 外，没有对所有完整权重载荷重新做 SHA256。

| 用途 | 固定 snapshot revision | 文件 GiB |
| --- | --- | ---: |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | `46d3c4aca0cfa475e99283abe9fb1254df1eef03` | 19.040 |
| z-lab/Qwen3.6-27B-DFlash | `0919688658996800f86b895034249700e9481106` | 3.223 |
| abstract-extraordinary/Qwen3.6-27B-DSpark | `36ccb76a7bf87a7f33cc3b85260e952685a8a7d0` | 8.196 |
| abstract-extraordinary/Qwen3.6-27B-DSpark-49k | `a3d00995272f95093311c638cb5913db7be3c033` | 6.202 |
| vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1 | `a233ec768abb2ed9a2a97a294b5808d0ad10932e` | 3.584 |

Target 实际为 **compressed-tensors INT4、非对称 group32、FP16 激活**，不要额外强制 `--quantization awq`。文件大小不等于额外运行内存；共享 embedding/head、KV 和 workspace 都影响后者。

DFlash 为 4×SWA2048+1×full；DSpark 为 5×full；DFlash2 为 5×SWA2048，特征层 `[5,19,33,47,61]`，不是旧草稿的特征层。49k 映射按 `target_id=draft_id+d2t[draft_id]` 检查。DFlash2 权重 SHA256 为 `52d968b9a9f7cc077580327074119a3815f74f48296c5f05b1aa1e0713f7b1f8`；实际解析的 N=7、特征层、SWA、独立 RoPE 和中间激活诊断见本机结果目录。

## 已完成的性能选择

Decode 使用相同两个短输入任务，每题两次、512 输出 token 上限、greedy、thinking off、新 cache salt、实际 cached tokens=0。以 `(output_tokens-1)/server generation_time` 计速。v6 的四条输出与原 target-only 同题文本一致；早期 v3/graphs 组曾有一条中文后段措辞不同。局部数值检查和有限文本回归不能替代完整质量评测。

| 配置 | 代码 decode token/s | 中文解释 decode token/s |
| --- | ---: | ---: |
| Target-only | 10.88 | 10.88 |
| DFlash N=8 | 11.89 | 6.38 |
| DSpark full N=8 | 11.28 | 6.17 |
| DSpark 49k N=8 | 11.89 | 4.75 |
| DFlash2 N=7，单条止损 | 2.48 | 未追加 |
| DFlash N=4 / block minimum 64，原生 | 19.15 | 12.10 |
| DSpark full N=4 / block minimum 64，原生 | 18.43 | 11.85 |
| DFlash N=4 / down 布局与 attention，v5 | 26.59 | 16.92 |
| DFlash N=4 / 再优化 down tile，v6 | **27.54** | **17.42** |

DSpark full N=4 相比其 N=8 在代码/中文分别提高约 63%/92%，说明此前差距包含运行时形状适配问题。49k 的中文接受长度较低。DFlash2 FP16 的早期 2/3563 个 proposal 不能用于算法排名，原因已有数值证据。后续优化底座上的 DSpark 复核单列在结果目录。

v6 底座 + Top-K128 的 DSpark full 单次代码/中文为 26.75/17.33，49k 为 28.67/13.44 token/s，四条文本与 target-only 一致。它们已接近或在某个代码点超过 DFlash，但单次小差距不构成算法排名；默认保留语言任务间表现更均衡的 DFlash。

v0.32 原生 HIP skinny 分支有 M≤5 和 K×M 的 LDS 限制。N=4 的常见验证 M=5 可以加速多数投影，但 K=17408 的 down 仍越界；只为这些层保留 K-major 布局，并使用经过验证的小矩阵 tile，额外约 3.18 GiB。没有复制全模型第二布局或强行放宽 C++ 内存限制。请求内 HIP event 分项计时、随机/真实权重数值检查均在结果目录；不能将全部收益归到单一 GEMM。

长上下文另用同一 8192-token 编程输入、256-token 输出、两次无缓存请求验证。仅迁入验证阶段的 16-segment split-K attention，decode 从 **15.78→22.06 token/s**；平均接受长度同为 3.70、文本相同。该数字来自 v4/v5 对照，不将短输入的 27.54 外推到长输入。

Prefill 单独测量。使用相同完整 token ID 的 4096-token 输入，8-token 输出，每个请求新 salt、缓存命中 0，以下为第二次调用：

| 配置 | 实际物理 block | Prefill token/s | 客户端 TTFT 秒 |
| --- | ---: | ---: | ---: |
| DFlash N=4 / auto | 816 | 148.89 | 27.53 |
| DFlash N=8 / auto 复核 | 832 | 213.18 | 19.33 |
| DFlash N=4 / block minimum 64 | 832 | **196.94** | **20.92** |
| 原生路径同轮关闭优化复核 | 832 | 190.70 | 21.60 |
| 大投影 + prefill attention，v4 | 832 | **375.04** | **10.93** |

4K 从同轮原生 190.70 提高至 375.04，约 +96.7%，不是缓存命中收益。相同输入 ID 和对应输出 hash 保持一致。8K 从大 tile+down 版本的 234.31 提高到 **362.63 token/s**，TTFT 34.97→22.60 秒；这一差值包含 rocBLAS 与 attention 两项变化。大投影逐层临时反量化后调用 rocBLAS，没有 activation INT4 量化或全模型 FP16 常驻副本。attention 只在限定 gfx1151/FP16/GQA6/head256/block832 形状启用，其余路径回退。

首次迁移中旧 HIP v1 的 4K prefill 为 124.32，上游为 182.28，因此没有恢复旧 MMQ 为默认。新栈 full decode graph 已成功运行，但此前 26.18/16.69 与 eager 的 26.27/16.72 接近，所以仍用 eager。公开的 635–710 token/s 参考使用不同 target/量化/输入长度；最高引用值涉及 W4A4，不能当作本机 AWQ W4A16 的等价目标。来源和条件见结果目录。

## 启动与切换

```bash
cd /home/amd/demo01/vllm-awq4-qwen-1.0

# 审计缓存，生成固定 revision 配置；默认不覆盖已有手动配置。
bash scripts/strix_halo.sh prepare --cache /home/amd/hf-cache
python3 scripts/prepare_strix_halo.py --backend dflash2

# 镜像已构建；首次构建才需要 build。
bash scripts/strix_halo.sh check dflash
bash scripts/strix_halo.sh up dflash

# 研究时逐个切换，同一时刻只有一个模型服务。
bash scripts/strix_halo.sh up dspark
bash scripts/strix_halo.sh up dspark-49k
bash scripts/strix_halo.sh up dflash2
bash scripts/strix_halo.sh up target-only

# 展示前恢复基线。
bash scripts/strix_halo.sh up dflash
```

`up` 重建同名容器并等待健康检查，再运行六项冒烟；它不是请求内热切换。配置为 `.runtime/strix-halo/<backend>.env`，DFlash/DSpark N=4、block64、FP16，DSpark 另设 Top-K128；DFlash2 N=7、BF16、关闭 FP16 专用优化。研究参数可由 shell 临时覆盖，例如：

```bash
# 清空 Top-K，恢复完整 Markov 投影用于研究对照。
VLLM_DSPARK_DRAFT_TOPK= bash scripts/strix_halo.sh up dspark
python3 test/bench_strix_decode.py --label dspark-n4-new-check
```

基线恢复命令不带 shell 覆盖。根目录 `.env` 与默认 DFlash 配置对齐。启动器被复制进镜像，修改代码后需要 rebuild；修改 profile 参数只需重建容器。

回退到本轮原生 DFlash N4/block64 时，保留新 profile 的固定权重，切换镜像并关闭五个优化开关：

```bash
VLLM_IMAGE=vllm-awq4-qwen:stable-v0320 \
VLLM_QWEN27B_PREFILL_TILES=0 VLLM_QWEN27B_PREFILL_ROCBLAS=0 \
VLLM_QWEN27B_DOWN_KMAJOR=0 VLLM_QWEN27B_PREFILL_ATTN=0 \
VLLM_QWEN27B_VERIFY_ATTN=0 bash scripts/strix_halo.sh up dflash
```

```bash
# 仅在服务空闲时运行；脚本拒绝覆盖同名结果。
python3 test/bench_strix_decode.py --label new-dflash-check
python3 test/bench_strix_prefill.py --label new-dflash-check
bash scripts/strix_halo.sh logs dflash
```

## 展示层

现有 `expo_assistant/` 已完成真实后端验收：静态资源、Responses SSE、归属明确的请求指标、取消与后续请求恢复；二分查找修复示例通过模型给出的 3 个断言和额外 5 个边界用例。记录在结果目录的 `harness-*.json`。

最终优化镜像已重新通过六项 API 冒烟及展示页生成/取消/恢复；修复代码的 8 项检查也再次通过。前缀状态复用从 0 到 2496 个 cached tokens，输出不变。这是状态正确性检查，上述性能请求仍全部为缓存命中 0。见 `final-functional-check.json`、`final-demo-check.json`、`final-demo-code-check.json`。

2026-10-27 展示层已升级为多模态 Agent 工作台，以持久化用户 systemd 服务运行：

```bash
systemctl --user status qwen-strix-expo.service
# 若该服务没有运行，也可以前台启动：
bash scripts/strix_halo.sh harness dflash
```

它没有重复加载模型，使用 :8000 上实际运行的后端。unit 位于 `~/.config/systemd/user/qwen-strix-expo.service`，已启用用户登录时启动；模型容器仍按原流程启动。工作台的独立 Python 环境位于 `.runtime/harness-venv/`，会话、附件和文件变更备份位于 `.runtime/harness/`。

新增项目文件浏览/读写、图片和 PDF/Office/文本上传、自动多轮工具调用、网页搜索/正文读取/来源引用、城市天气查询。界面采用浅色工作台布局，支持附件预览、文件编辑器、工具状态、天气卡片、会话保存和移动端。左侧有四个可直接载入素材的测试案例。当前仍不提供任意终端执行。模式、限制与运行说明见 [工作台文档](../expo_assistant/README.md)。

本机真实模型已完成以下功能冒烟：读取 JSON 并创建结果、多轮读取/精确修改；图片与文档联合理解（识别图形、读取文档暗号）；样例柱状图读取五个值并合计 250、与目标 260 比较；搜索失败换词、从首页筛选正确链接并读取官方文档引用；当前天气与三天预报。浏览器另验证了真实 PDF 上传预览、手动保存和下载、取消/继续及刷新恢复。自动测试覆盖 28 项协议/状态/文件/网络边界；这些不是通用 Agent 质量或新速度基准。

## 历史实验与验收边界

已阅读旧 MBT 扫描、N 扫描、HIP decode v2–v9/hybrid、双布局内存、五项精度评测和 2026-10-22 DSpark/DFlash2 静态研究。保留 MBT8192，不重新开展缓存矩阵、旧 HIP 全参数扫描或五项完整精度评测。缺失的 `.research/` 不在这台机器，未将其当成已读证据。

本轮已测 8K decode 和内核级 32K attention；尚未完成 32K 全长生成、随机采样分布、并发 2/4、thinking 或大规模多模态/Agent 质量评测。多模态和自动多轮工具任务已在 2026-10-27 完成功能冒烟，具体范围见上节。功能冒烟不是通用正确率证明。DFlash2 BF16 诊断恢复接受能力，但没有通过插桩数字宣称速度领先；DSpark adaptive 也未验收。

CPU 检查：11 项 launcher、5 项部署检查、13 项既有展示测试通过。GPU 检查覆盖投影边界/真实权重、14 项 prefill attention 和 20 项 verify attention。复现入口：

```bash
python3 -m unittest discover -s test -p 'test_*launcher.py' -v
python3 -m unittest discover -s test -p 'test_strix_deployment.py' -v
bash scripts/strix_halo.sh config dflash
```
