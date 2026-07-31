#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from longdoc_sanity_lib import (  # noqa: E402
    build_prompt,
    check_citations,
    make_needle_cases,
    parse_json_object,
    score_case,
    split_documents,
    summarize_results,
)
from run_longdoc_sanity import post_stream  # noqa: E402


class FakeTokenizer:
    """Character tokenizer sufficient to test exact prompt-budget assembly."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(item) for item in token_ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered)


class LongdocSanityTest(unittest.TestCase):
    def test_split_documents(self) -> None:
        corpus = "DOCUMENT_START: a.pdf\nalpha\nDOCUMENT_START: b.pdf\nbeta"
        self.assertEqual([name for name, _ in split_documents(corpus)], ["a.pdf", "b.pdf"])

    def test_json_fence_recovery(self) -> None:
        parsed, error = parse_json_object("```json\n{\"answer\":\"ok\"}\n```")
        self.assertIsNone(error)
        self.assertEqual(parsed, {"answer": "ok"})

    def test_answerable_score(self) -> None:
        case = {
            "answerable": True,
            "source_document": "paper.pdf",
            "required_answer_groups": [["160"], ["57%"], ["51%"]],
            "gold_evidence": ["The result was 160, 57%, and 51%."],
        }
        context = "DOCUMENT_START: paper.pdf\nThe result was 160, 57%, and 51%."
        response = json.dumps({
            "answer": "160, 57%, 51%",
            "source_documents": ["paper.pdf"],
            "evidence_quotes": ["The result was 160, 57%, and 51%."],
        })
        parsed, _ = parse_json_object(response)
        citations = check_citations(parsed, context)
        score = score_case(case, response, citations)
        self.assertEqual(score["score"], 4.0)

    def test_irrelevant_valid_quote_gets_no_evidence_point(self) -> None:
        case = {
            "answerable": True,
            "source_document": "paper.pdf",
            "required_answer_groups": [["160"]],
            "gold_evidence": ["The result was 160."],
        }
        context = "DOCUMENT_START: paper.pdf\nThe result was 160. Another true sentence."
        response = json.dumps({
            "answer": "160",
            "source_documents": ["paper.pdf"],
            "evidence_quotes": ["Another true sentence."],
        })
        parsed, _ = parse_json_object(response)
        score = score_case(case, response, check_citations(parsed, context))
        self.assertEqual(score["citation_validity"], 1.0)
        self.assertEqual(score["evidence_support"], 0.0)
        self.assertEqual(score["score"], 3.0)

    def test_unanswerable_score(self) -> None:
        case = {
            "answerable": False,
            "source_document": "paper.pdf",
            "required_answer_groups": [],
            "forbidden_claim_terms": ["A100"],
        }
        context = "DOCUMENT_START: paper.pdf\nNo hardware details."
        response = json.dumps({
            "answer": "资料未提供。",
            "source_documents": ["paper.pdf"],
            "evidence_quotes": [],
        }, ensure_ascii=False)
        parsed, _ = parse_json_object(response)
        score = score_case(case, response, check_citations(parsed, context))
        self.assertEqual(score["score"], 4.0)

    def test_needles_are_deterministic_and_distinct(self) -> None:
        first = make_needle_cases("103k", 7)
        second = make_needle_cases("103k", 7)
        self.assertEqual(first, second)
        values = {case["gold_evidence"][0] for case in first}
        self.assertEqual(len(values), 4)

    def test_prompt_builder_hits_budget_and_position(self) -> None:
        case = {
            "id": "test",
            "question": "value?",
            "source_document": "paper.pdf",
            "position_bucket": "late",
            "gold_evidence": ["unique evidence"],
        }
        result = build_prompt(
            tokenizer=FakeTokenizer(),
            case=case,
            filler_text="filler " * 2000,
            source_text="DOCUMENT_START: paper.pdf\nreal source",
            target_prompt_tokens=3000,
            context_mode="evidence",
        )
        self.assertLessEqual(abs(result.prompt_tokens - 3000), 2)
        self.assertAlmostEqual(result.inserted_fraction, 0.9, places=2)

    def test_streaming_api_parser(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", "0"))
                json.loads(self.rfile.read(content_length))
                chunks = [
                    {"choices": [{"delta": {"content": "{\"answer\":"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "\"ok\"}"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
                ]
                body = "".join(
                    f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
                ) + "data: [DONE]\n\n"
                payload = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            text, meta, timing = post_stream(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                {"model": "mock", "messages": [], "stream": True},
                5.0,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
        self.assertEqual(text, '{"answer":"ok"}')
        self.assertEqual(meta["usage"]["prompt_tokens"], 10)
        self.assertEqual(meta["finish_reason"], "stop")
        self.assertIsNotNone(timing["ttft_s"])

    def test_summary_metrics(self) -> None:
        row = {
            "status": "ok",
            "case": {"answerable": True, "type": "numeric_fact"},
            "score": {
                "score": 4.0,
                "max_score": 4.0,
                "json_valid": True,
                "citation_validity": 1.0,
                "evidence_support": 1.0,
                "source_correct": True,
                "answer_group_fraction": 1.0,
                "refusal_correct": None,
            },
            "timing": {"wall_s": 2.0, "ttft_s": 1.0},
        }
        summary = summarize_results([row])
        self.assertEqual(summary["qa_score_fraction"], 1.0)
        self.assertEqual(summary["numeric_exact_match"], 1.0)
        self.assertEqual(summary["evidence_support"], 1.0)


if __name__ == "__main__":
    unittest.main()
