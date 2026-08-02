#!/usr/bin/env python3
"""Context-aware OpenAI-compatible router for dual TP=4 W7900 services."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask


def select_backend(
    prompt_tokens: int,
    batch_size: int,
    threshold_tokens: int,
) -> str:
    """Return the evidence-backed speculative mode for a request."""
    if batch_size == 1 and prompt_tokens <= threshold_tokens:
        return "dflash_n4"
    return "target_only"


def _encoded_length(encoded: Any) -> int:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    shape = getattr(encoded, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if encoded and isinstance(encoded[0], list):
        return len(encoded[0])
    return len(encoded)


def count_prompt_tokens(tokenizer: Any, payload: dict[str, Any]) -> tuple[int, int]:
    messages = payload.get("messages")
    if messages is not None:
        batches = messages if messages and isinstance(messages[0], list) else [messages]
        lengths = []
        for item in batches:
            encoded = tokenizer.apply_chat_template(
                item,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=payload.get("chat_template_kwargs", {}).get(
                    "enable_thinking", False
                ),
            )
            lengths.append(_encoded_length(encoded))
        return max(lengths, default=0), len(batches)

    prompt = payload.get("prompt", "")
    prompts = prompt if isinstance(prompt, list) else [prompt]
    lengths = [
        _encoded_length(tokenizer.encode(item, add_special_tokens=True))
        for item in prompts
    ]
    return max(lengths, default=0), len(prompts)


def create_app(
    tokenizer: Any,
    dflash_url: str,
    target_url: str,
    threshold_tokens: int,
) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> JSONResponse:
        async with httpx.AsyncClient(timeout=5.0) as client:
            checks = await asyncio.gather(
                client.get(f"{dflash_url}/health"),
                client.get(f"{target_url}/health"),
                return_exceptions=True,
            )
        ready = all(
            isinstance(item, httpx.Response) and item.is_success for item in checks
        )
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request) -> Response:
        body = await request.body()
        payload: dict[str, Any] = {}
        prompt_tokens = 0
        batch_size = 1
        route = "target_only"
        if request.method == "POST" and body:
            payload = await request.json()
            prompt_tokens, batch_size = count_prompt_tokens(tokenizer, payload)
            route = select_backend(prompt_tokens, batch_size, threshold_tokens)

        base_url = dflash_url if route == "dflash_n4" else target_url
        logging.getLogger("uvicorn.error").info(
            "adaptive route=%s prompt_tokens=%d batch_size=%d",
            route,
            prompt_tokens,
            batch_size,
        )
        client = httpx.AsyncClient(timeout=None)
        upstream = await client.send(
            client.build_request(
                request.method,
                f"{base_url}/{path}",
                content=body,
                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() not in {"host", "content-length"}
                },
            ),
            stream=True,
        )
        headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        headers["X-DFlash-Route"] = route
        headers["X-Prompt-Tokens"] = str(prompt_tokens)

        async def stream() -> AsyncIterator[bytes]:
            async for chunk in upstream.aiter_raw():
                yield chunk

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=headers,
            background=BackgroundTask(client.aclose),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dflash-url", default="http://127.0.0.1:8061")
    parser.add_argument("--target-url", default="http://127.0.0.1:8062")
    parser.add_argument("--threshold-tokens", type=int, default=14000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8060)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    uvicorn.run(
        create_app(
            tokenizer,
            args.dflash_url.rstrip("/"),
            args.target_url.rstrip("/"),
            args.threshold_tokens,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
