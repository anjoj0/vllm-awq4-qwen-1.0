#!/usr/bin/env python3
"""Shared helpers for the Nowcast3D long-document quality sanity suite."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DOCUMENT_MARKER = "DOCUMENT_START:"
SOURCE_DOCUMENT = "##未开源nowcast3d.pdf"
REFUSAL_MARKERS = (
    "资料未提供",
    "文中未提供",
    "没有提供",
    "未报告",
    "无法从资料",
    "not provided",
    "not reported",
    "cannot be determined",
)
POSITION_FRACTIONS = {
    "early": 0.10,
    "middle": 0.50,
    "late": 0.90,
    "very_late": 0.99,
}
SYSTEM_PROMPT = """你是科研文献核验助手。只能依据用户提供的资料回答。
每个事实必须给出来源文档名和资料中的原文短证据。
如果资料中没有答案，必须回答“资料未提供”，不得使用外部知识补全。
严格输出一个合法 JSON 对象，不要使用 Markdown 代码块，不要输出额外解释。
JSON 格式：
{"answer":"...","source_documents":["..."],"evidence_quotes":["..."],"confidence":0.0}
其中 evidence_quotes 必须是输入资料中逐字存在的短引文。"""


@dataclass
class BuiltPrompt:
    messages: list[dict[str, str]]
    prompt_tokens: int
    context_sha256: str
    context_chars: int
    inserted_fraction: float
    source_document: str
    expected_evidence: list[str]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(value: Any) -> str:
    return re.sub(r"[^\w%./:+-]+", "", normalize_text(value), flags=re.UNICODE)


def split_documents(corpus: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^DOCUMENT_START:\s*(.+?)\s*$", corpus))
    documents: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(corpus)
        documents.append((match.group(1).strip(), corpus[match.start():end].strip()))
    if not documents:
        raise ValueError("No DOCUMENT_START markers were found in the corpus")
    return documents


def extract_source_and_filler(corpus: str, source_document: str = SOURCE_DOCUMENT) -> tuple[str, str, list[str]]:
    documents = split_documents(corpus)
    source_matches = [body for name, body in documents if name == source_document]
    if len(source_matches) != 1:
        raise ValueError(
            f"Expected exactly one {source_document!r} document, found {len(source_matches)}"
        )
    filler_documents = [body for name, body in documents if name != source_document]
    return source_matches[0], "\n\n".join(filler_documents), [name for name, _ in documents]


def make_evidence_packet(case: dict[str, Any], fallback_source: str = "") -> str:
    evidence = case.get("gold_evidence", [])
    if not evidence:
        # Use real source prose for unanswerable cases. Explicitly saying that
        # information is absent would leak the expected refusal to the model.
        evidence_text = fallback_source[:4000].strip()
        if not evidence_text:
            raise ValueError("Unanswerable evidence mode requires fallback source text")
    else:
        evidence_text = "\n\n".join(evidence)
    return (
        f"\n\n[RETRIEVAL_SOURCE_BEGIN]\n{DOCUMENT_MARKER} "
        f"{case['source_document']}\n{evidence_text}\n[RETRIEVAL_SOURCE_END]\n\n"
    )


def make_needle_cases(profile_name: str, seed: int) -> list[dict[str, Any]]:
    buckets = ("early", "middle", "late", "very_late")
    cases: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        # Deterministic per run, but unrelated to facts in the papers.
        number = (seed * 7919 + (index + 1) * 104729) % 900000 + 100000
        code = f"ROCM-RADAR-{number}"
        group = f"Cedar-{(seed * 17 + index * 23) % 90 + 10}"
        sentence = (
            f"本次封闭测试的校验编号为 {code}，负责复核的虚构工作组为 {group}。"
        )
        cases.append(
            {
                "id": f"needle_{profile_name}_{bucket}",
                "type": "needle",
                "question": "根据资料，本次封闭测试的校验编号和工作组分别是什么？",
                "answerable": True,
                "source_document": "SANITY_NEEDLE",
                "position_bucket": bucket,
                "required_answer_groups": [[code], [group]],
                "gold_evidence": [sentence],
            }
        )
    return cases


def render_user_prompt(context: str, question: str) -> str:
    return (
        "以下是科研论文资料。资料中可能包含多篇论文，文档边界由 "
        "DOCUMENT_START 标识。\n\n"
        "[LONG_DOCUMENT_BEGIN]\n"
        f"{context}\n"
        "[LONG_DOCUMENT_END]\n\n"
        f"问题：{question}\n"
        "请严格按 system 消息规定的 JSON 格式回答。"
    )


def chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(tokenized, "get") and tokenized.get("input_ids") is not None:
        token_ids = tokenized["input_ids"]
    else:
        token_ids = tokenized
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def _decode_slice(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_prompt(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    filler_text: str,
    source_text: str,
    target_prompt_tokens: int,
    context_mode: str,
) -> BuiltPrompt:
    if context_mode == "evidence":
        source_packet = make_evidence_packet(case, fallback_source=source_text)
    elif context_mode == "full-paper":
        if case["source_document"] == "SANITY_NEEDLE":
            source_packet = make_evidence_packet(case, fallback_source=source_text)
        else:
            source_packet = f"\n\n[RETRIEVAL_SOURCE_BEGIN]\n{source_text}\n[RETRIEVAL_SOURCE_END]\n\n"
    else:
        raise ValueError(f"Unsupported context mode: {context_mode}")

    base_context = source_packet
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(base_context, case["question"])},
    ]
    base_tokens = chat_token_count(tokenizer, base_messages)
    if base_tokens >= target_prompt_tokens:
        raise ValueError(
            f"Case {case['id']} source packet already uses {base_tokens} tokens, "
            f"which does not fit target {target_prompt_tokens}. Use a larger profile "
            "or --context-mode evidence."
        )

    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    fraction = POSITION_FRACTIONS[case["position_bucket"]]
    wanted_filler = target_prompt_tokens - base_tokens
    if len(filler_ids) < wanted_filler + 32:
        raise ValueError(
            f"Corpus has only {len(filler_ids)} filler tokens; about {wanted_filler} are needed"
        )

    # Token allocation is only approximate after decoding and applying the chat
    # template. Use a bounded search so very large corpora are clipped to the
    # requested budget instead of relying on a few linear correction steps.
    def render_with_budget(filler_budget: int) -> tuple[list[dict[str, str]], str, int, int]:
        before_count = int(filler_budget * fraction)
        after_count = max(0, filler_budget - before_count)
        before = _decode_slice(tokenizer, filler_ids[:before_count])
        after = _decode_slice(
            tokenizer,
            filler_ids[before_count:before_count + after_count],
        )
        candidate_context = before + source_packet + after
        candidate_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(candidate_context, case["question"])},
        ]
        return (
            candidate_messages,
            candidate_context,
            chat_token_count(tokenizer, candidate_messages),
            before_count,
        )

    high = min(len(filler_ids), max(1, wanted_filler + 4096))
    high_messages, high_context, high_tokens, high_before = render_with_budget(high)
    while high_tokens < target_prompt_tokens - 2 and high < len(filler_ids):
        high = min(len(filler_ids), int(high * 1.5) + 1024)
        high_messages, high_context, high_tokens, high_before = render_with_budget(high)
    if high_tokens < target_prompt_tokens - 2:
        raise ValueError(
            f"Corpus has only enough filler for {high_tokens} prompt tokens; "
            f"{target_prompt_tokens} were requested"
        )

    low = 0
    best_messages, best_context, best_tokens, best_before, best_budget = (
        base_messages,
        base_context,
        base_tokens,
        0,
        0,
    )
    while low <= high:
        mid = (low + high) // 2
        candidate_messages, candidate_context, candidate_tokens, candidate_before = render_with_budget(mid)
        if candidate_tokens <= target_prompt_tokens + 2:
            best_messages = candidate_messages
            best_context = candidate_context
            best_tokens = candidate_tokens
            best_before = candidate_before
            best_budget = mid
            low = mid + 1
        else:
            high = mid - 1

    messages = best_messages
    context = best_context
    prompt_tokens = best_tokens

    if prompt_tokens > target_prompt_tokens + 2:
        raise ValueError(
            f"Constructed prompt is too large: {prompt_tokens} > {target_prompt_tokens}"
        )
    inserted_fraction = best_before / max(1, best_budget)
    return BuiltPrompt(
        messages=messages,
        prompt_tokens=prompt_tokens,
        context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
        context_chars=len(context),
        inserted_fraction=inserted_fraction,
        source_document=case["source_document"],
        expected_evidence=list(case.get("gold_evidence", [])),
    )


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
        return (value, None) if isinstance(value, dict) else (None, "JSON root is not an object")
    except json.JSONDecodeError as first_error:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(candidate[start:end + 1])
                return (value, None) if isinstance(value, dict) else (None, "JSON root is not an object")
            except json.JSONDecodeError:
                pass
        return None, str(first_error)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value)]
    return []


def check_citations(parsed: dict[str, Any] | None, context: str) -> dict[str, Any]:
    if not parsed:
        return {"sources": [], "evidence": [], "source_valid_fraction": 0.0,
                "evidence_valid_fraction": 0.0}
    documents = {name for name, _ in split_documents(context)}
    if "SANITY_NEEDLE" in context:
        documents.add("SANITY_NEEDLE")
    sources = _as_string_list(parsed.get("source_documents"))
    evidence = _as_string_list(parsed.get("evidence_quotes"))
    normalized_context = normalize_text(context)
    source_rows = [
        {"value": item, "valid": item in documents}
        for item in sources
    ]
    evidence_rows = [
        {"value": item, "valid": normalize_text(item) in normalized_context}
        for item in evidence
    ]
    source_fraction = sum(row["valid"] for row in source_rows) / max(1, len(source_rows))
    evidence_fraction = sum(row["valid"] for row in evidence_rows) / max(1, len(evidence_rows))
    return {
        "sources": source_rows,
        "evidence": evidence_rows,
        "source_valid_fraction": source_fraction,
        "evidence_valid_fraction": evidence_fraction,
    }


def score_case(
    case: dict[str, Any],
    response_text: str,
    citation_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed, parse_error = parse_json_object(response_text)
    checks = citation_checks or {
        "source_valid_fraction": 0.0,
        "evidence_valid_fraction": 0.0,
        "sources": [],
        "evidence": [],
    }
    answer = "" if not parsed else str(parsed.get("answer", ""))
    normalized_answer = compact_text(answer)

    if case["answerable"]:
        group_rows = []
        for alternatives in case.get("required_answer_groups", []):
            matched = [alt for alt in alternatives if compact_text(alt) in normalized_answer]
            group_rows.append({"alternatives": alternatives, "matched": matched, "pass": bool(matched)})
        answer_fraction = sum(row["pass"] for row in group_rows) / max(1, len(group_rows))
        expected_source = case["source_document"]
        source_values = [row["value"] for row in checks.get("sources", []) if row.get("valid")]
        source_correct = expected_source in source_values
        evidence_valid = checks.get("evidence_valid_fraction", 0.0)
        gold_evidence = [compact_text(item) for item in case.get("gold_evidence", [])]
        evidence_support_rows = []
        for row in checks.get("evidence", []):
            quote = compact_text(row["value"])
            supports = bool(
                row.get("valid")
                and len(quote) >= 12
                and any(quote in gold or gold in quote for gold in gold_evidence)
            )
            evidence_support_rows.append({"value": row["value"], "supports_gold": supports})
        evidence_support = sum(
            row["supports_gold"] for row in evidence_support_rows
        ) / max(1, len(evidence_support_rows))
        score = 2.0 * answer_fraction + float(source_correct) + evidence_support
        return {
            "json_valid": parsed is not None,
            "json_error": parse_error,
            "answer_group_fraction": answer_fraction,
            "answer_groups": group_rows,
            "source_correct": source_correct,
            "citation_validity": evidence_valid,
            "evidence_support": evidence_support,
            "evidence_support_rows": evidence_support_rows,
            "refusal_correct": None,
            "score": round(score, 4),
            "max_score": 4.0,
        }

    refusal = any(marker in normalize_text(answer) for marker in REFUSAL_MARKERS)
    suspicious_number = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:kwh|kg\s*co2e?|hours?|小时|张)\b", normalize_text(answer))
    )
    forbidden = [
        term for term in case.get("forbidden_claim_terms", [])
        if compact_text(term) in normalized_answer and not refusal
    ]
    no_fabricated_claim = not suspicious_number and not forbidden
    no_invalid_citation = all(row.get("valid") for row in checks.get("sources", [])) and all(
        row.get("valid") for row in checks.get("evidence", [])
    )
    score = 2.0 * float(refusal) + float(no_fabricated_claim) + float(no_invalid_citation)
    return {
        "json_valid": parsed is not None,
        "json_error": parse_error,
        "answer_group_fraction": None,
        "answer_groups": [],
        "source_correct": None,
        "citation_validity": checks.get("evidence_valid_fraction", 0.0),
        "evidence_support": None,
        "refusal_correct": refusal,
        "no_fabricated_claim": no_fabricated_claim,
        "forbidden_matches": forbidden,
        "no_invalid_citation": no_invalid_citation,
        "score": round(score, 4),
        "max_score": 4.0,
    }


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    completed = [row for row in values if row.get("status") == "ok"]
    scored = [row for row in completed if row.get("score")]
    total_score = sum(row["score"]["score"] for row in scored)
    max_score = sum(row["score"]["max_score"] for row in scored)
    answerable = [row for row in scored if row.get("case", {}).get("answerable")]
    unanswerable = [row for row in scored if not row.get("case", {}).get("answerable")]
    needles = [row for row in answerable if row.get("case", {}).get("type") == "needle"]
    citations = [row["score"]["citation_validity"] for row in answerable]
    support = [row["score"]["evidence_support"] for row in answerable]
    source_accuracy = [row["score"]["source_correct"] for row in answerable]
    numeric = [row for row in answerable if row.get("case", {}).get("type") == "numeric_fact"]
    return {
        "requested": len(values),
        "completed": len(completed),
        "failed": len(values) - len(completed),
        "qa_score": total_score,
        "qa_max_score": max_score,
        "qa_score_fraction": total_score / max_score if max_score else 0.0,
        "json_success_rate": sum(row["score"]["json_valid"] for row in scored) / max(1, len(scored)),
        "citation_validity": sum(citations) / max(1, len(citations)),
        "evidence_support": sum(support) / max(1, len(support)),
        "source_accuracy": sum(source_accuracy) / max(1, len(source_accuracy)),
        "numeric_exact_match": sum(
            row["score"]["answer_group_fraction"] == 1.0 for row in numeric
        ) / max(1, len(numeric)),
        "needle_exact_match": sum(
            row["score"]["answer_group_fraction"] == 1.0 for row in needles
        ) / max(1, len(needles)),
        "abstention_accuracy": sum(
            row["score"]["refusal_correct"] is True for row in unanswerable
        ) / max(1, len(unanswerable)),
        "mean_wall_s": sum(row["timing"]["wall_s"] for row in completed) / max(1, len(completed)),
        "mean_ttft_s": sum(
            row["timing"]["ttft_s"] for row in completed
            if row["timing"].get("ttft_s") is not None
        ) / max(1, sum(row["timing"].get("ttft_s") is not None for row in completed)),
    }


def summary_markdown(metadata: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Nowcast3D Long-Document Sanity Result",
        "",
        f"- Run ID: `{metadata.get('run_id', '')}`",
        f"- Config label: `{metadata.get('config_label', '')}`",
        f"- Model: `{metadata.get('model', '')}`",
        f"- Profile: `{metadata.get('profile', '')}`",
        f"- Context mode: `{metadata.get('context_mode', '')}`",
        f"- Completed: {summary['completed']}/{summary['requested']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| QA score | {summary['qa_score']:.2f}/{summary['qa_max_score']:.2f} |",
        f"| QA score fraction | {summary['qa_score_fraction']:.2%} |",
        f"| JSON success | {summary['json_success_rate']:.2%} |",
        f"| Citation validity | {summary['citation_validity']:.2%} |",
        f"| Evidence support | {summary['evidence_support']:.2%} |",
        f"| Source accuracy | {summary['source_accuracy']:.2%} |",
        f"| Numeric exact match | {summary['numeric_exact_match']:.2%} |",
        f"| Needle exact match | {summary['needle_exact_match']:.2%} |",
        f"| Abstention accuracy | {summary['abstention_accuracy']:.2%} |",
        f"| Mean wall time | {summary['mean_wall_s']:.3f} s |",
        f"| Mean TTFT | {summary['mean_ttft_s']:.3f} s |",
        "",
        "## Cases",
        "",
        "| Case | Type | Prompt tokens | Position | Score | JSON | Citation | Wall (s) |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('case', {}).get('id', '?')} | error | - | - | - | no | - | - |"
            )
            continue
        score = row["score"]
        lines.append(
            f"| {row['case']['id']} | {row['case']['type']} | "
            f"{row['context']['prompt_tokens']} | {row['context']['inserted_fraction']:.1%} | "
            f"{score['score']:.2f}/{score['max_score']:.0f} | "
            f"{'yes' if score['json_valid'] else 'no'} | "
            f"{score['citation_validity']:.0%} | {row['timing']['wall_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a fixed small-sample engineering regression suite. It is not a replacement for a full academic long-context benchmark.",
            "",
        ]
    )
    return "\n".join(lines)
