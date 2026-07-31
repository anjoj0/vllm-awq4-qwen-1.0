#!/usr/bin/env python3
import argparse

from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/models/Qwen3.6-27B")
parser.add_argument("--file", required=True)
parser.add_argument("--chars", nargs="+", type=int, required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
with open(args.file, encoding="utf-8") as stream:
    text = stream.read()

instruction = (
    "请阅读下面的论文资料，给出结构化摘要，并列出与长上下文推理优化相关的关键技术。\n\n"
)

for char_count in args.chars:
    token_ids = tokenizer(
        instruction + text[:char_count],
        add_special_tokens=True,
    ).input_ids
    print(f"{char_count}\t{len(token_ids)}")
