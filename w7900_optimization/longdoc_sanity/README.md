# Nowcast3D 科研长文质量 Sanity

本目录是一套可直接对 OpenAI-compatible vLLM 服务运行的固定小样本质量回归测试。它用于判断 BF16 auto KV、BF16 FP8 KV、AWQ4 和 AWQ4+DFlash 等优化配置是否造成明显的科研长文质量退化，不替代 LongBench、RULER 或完整学术准确率评测。

## 1. 测试内容

- 20 道冻结的 Nowcast3D 自然问题。
- 每题包含标准答案要点、原始文档名和逐字原文证据。
- 4 个动态 needle，分别插入上下文约 10%、50%、90%、99% 位置。
- 自然问题覆盖事实检索、数值、跨段综合、方法比较、数据/代码可用性和不可回答拒答。
- 自动检查 JSON、答案要点、文档来源、逐字引用、needle 和拒答。
- 保存 TTFT、wall time、API usage、原始响应、配置 manifest 和语料/题集 SHA256。

数据集：`data/nowcast3d_cases.jsonl`
上下文档位：`config/profiles.json`

## 2. 两种上下文模式

### `evidence`

将每道题的原文证据作为一个 Nowcast3D source packet，插入由其他论文组成的 6K/24K/64K/103K/近 256K 背景中。适合严格、可自动评分的位置检索回归。

### `full-paper`

插入完整 Nowcast3D 论文，适合综合理解。完整论文本身不能放入 6K 档位；建议使用 64K 或更长 profile。

报告中必须分别称为“证据片段检索”和“完整论文理解”，不要混淆。

## 3. 环境要求

容器现有 vLLM 环境即可，无额外 pip 依赖：

- Python 3.10+
- `transformers`（vLLM 镜像已经包含）
- 本地 Qwen tokenizer/model
- 已启动的 `/v1/chat/completions` 服务
- `/workspace/bench_data/combined_papers_for_llm_L.txt`

执行器使用本地 tokenizer 的 `apply_chat_template` 计算完整 prompt token 数，包含 system、问题和 generation prompt，避免字符数估算超出模型上限。

## 4. 上传后先验证

```bash
cd /workspace/vllm-awq4-qwen-1.0-main/w7900_optimization/longdoc_sanity

python3 validate_suite.py \
  --corpus /workspace/bench_data/combined_papers_for_llm_L.txt

python3 -m unittest discover -s tests -v
```

预期：所有 20 道题的证据逐字存在于 Nowcast3D 原文，单元测试全部通过。

## 5. 先做 prompt 干跑

不访问 API，只验证 tokenizer、chat template、token 预算和证据位置：

```bash
python3 run_longdoc_sanity.py \
  --tokenizer /models/Qwen3.6-27B \
  --corpus /workspace/bench_data/combined_papers_for_llm_L.txt \
  --profile 103k \
  --suite smoke \
  --context-mode evidence \
  --config-label dryrun_bf16 \
  --dry-run
```

干跑输出目录的 `dry_run_summary.json` 中不应有错误，`prompt_tokens` 应接近 103,000。

## 6. 单次运行

假设 BF16 auto KV 服务运行在 8030：

```bash
python3 run_longdoc_sanity.py \
  --base-url http://127.0.0.1:8030 \
  --model Qwen3.6-27B-BF16 \
  --tokenizer /models/Qwen3.6-27B \
  --corpus /workspace/bench_data/combined_papers_for_llm_L.txt \
  --profile 103k \
  --suite core \
  --context-mode evidence \
  --config-label bf16_tp8_auto
```

默认关闭 thinking，以降低输出格式波动。若要测 thinking 路径，显式添加 `--enable-thinking` 并使用独立配置标签。

Suite 范围：

| Suite | 内容 | 建议用途 |
|---|---|---|
| `smoke` | 4 道自然题，默认再加 4 needles | 新服务快速检查 |
| `core` | 11 道代表性自然题，默认再加 4 needles | 每个正式配置 |
| `full` | 20 道自然题，默认再加 4 needles | 最终候选配置 |
| `needle` | 仅 4 个位置 needle | 近 256K 极限 |

`--no-needles` 可关闭自动 needle；`--case-id ID` 可重复指定单题。

## 7. 推荐的当前服务完整流程

服务启动后执行：

```bash
CONFIG_LABEL=bf16_tp8_auto \
BASE_URL=http://127.0.0.1:8030 \
MODEL=Qwen3.6-27B-BF16 \
TOKENIZER=/models/Qwen3.6-27B \
bash run_recommended_current_service.sh
```

脚本依次运行：

1. 103K evidence core + 4 needles。
2. 103K 完整 Nowcast3D 论文 smoke。
3. 近 256K 的 4 个位置 needles。

切换服务配置后更改 `CONFIG_LABEL` 再运行：

```text
bf16_tp8_auto
bf16_tp8_fp8
awq4_tp8_fp8
awq4_tp4_dflash_n8
```

AWQ4+DFlash 只建议另外跑 6K/8K/12K 候选区间，不需要把它作为 24K+ 性能路线。6K 例子：

```bash
python3 run_longdoc_sanity.py \
  --base-url http://127.0.0.1:8044 \
  --model Qwen3.6-27B-AWQ4 \
  --tokenizer /workspace/cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/master \
  --corpus /workspace/bench_data/combined_papers_for_llm_L.txt \
  --profile 6k --suite core --context-mode evidence \
  --config-label awq4_tp4_dflash_n8
```

## 8. 输出文件

每轮位于 `results/<run_id>/`：

```text
manifest.json       固定配置、文件哈希、case IDs
results.jsonl       每题原始答案、时间、引用检查和自动评分
summary.json        机器可读汇总
summary.md          人类可读汇总
prompts/            仅在 --save-prompts 时保存完整 prompt
```

如需人工复核开放式答案：

```bash
python3 export_manual_review.py results/<run_id>
```

生成 Excel 可直接打开的 `manual_review.csv`。自动关键词分数用于快速回归，开放式综合题的正式结论应以人工复核为准。

## 9. 跨配置比较

只比较相同 profile 和相同 suite/context mode 的结果：

```bash
python3 compare_longdoc_sanity.py \
  results/<bf16-auto-run> \
  results/<bf16-fp8-run> \
  results/<awq4-run> \
  --baseline-label bf16_tp8_auto \
  --output comparison_103k.md
```

输出 Markdown 和 CSV，包含 QA、相对基线保持率、needle EM、引用有效率、拒答率、JSON 成功率、平均 wall 和 TTFT。

## 10. 正式报告口径

可以写：

> 在冻结的 Nowcast3D 小样本科研长文回归集上，某配置相对 BF16 auto 基线保持了 X% 的自动 QA 得分，证据引用有效率为 Y%，在 103K/近 256K 四个位置的 needle 命中率为 Z%。

不能写：

- “通过 20 道题证明完全无损”。
- 把 evidence packet 测试称为完整论文理解。
- 只报告回答正确率而隐藏无效引用或编造来源。
- 在看完结果后删除失败题或修改准入门槛。

建议准入线在测试前冻结：JSON 成功率不低于 95%，引用有效率不低于 90%，目标配置相对 BF16 auto 的 QA 保持率不低于 95%，needle EM 相对基线下降不超过 5 个百分点。
