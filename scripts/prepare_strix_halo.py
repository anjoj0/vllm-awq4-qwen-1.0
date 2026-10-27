#!/usr/bin/env python3
"""Audit offline checkpoints and generate revision-pinned Strix Halo profiles.

Uses only the standard library. Reads safetensors headers and the small d2t
mapping, not full model payloads. This is not a checksum or GPU correctness test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "target-only": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "dflash": "z-lab/Qwen3.6-27B-DFlash",
    "dspark": "abstract-extraordinary/Qwen3.6-27B-DSpark",
    "dspark-49k": "abstract-extraordinary/Qwen3.6-27B-DSpark-49k",
    "dflash2": "vvsotnikov/Qwen3.6-27B-nothink-DFlash2-v0.1",
}
DEFAULT_BACKENDS = ("target-only", "dflash", "dspark", "dspark-49k")
PINNED_REVISIONS = {
    "target-only": "46d3c4aca0cfa475e99283abe9fb1254df1eef03",
    "dflash": "0919688658996800f86b895034249700e9481106",
    "dspark": "36ccb76a7bf87a7f33cc3b85260e952685a8a7d0",
    "dspark-49k": "a3d00995272f95093311c638cb5913db7be3c033",
    "dflash2": "a233ec768abb2ed9a2a97a294b5808d0ad10932e",
}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1,
               "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_header(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        require(len(raw) == 8, f"{path}: missing safetensors header")
        length = struct.unpack("<Q", raw)[0]
        require(0 < length <= 32 * 1024**2, f"{path}: invalid header length")
        raw = stream.read(length)
        require(len(raw) == length, f"{path}: truncated header")
        header = json.loads(raw)
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    require(bool(tensors), f"{path}: no tensors")
    cursor = 0
    for name, value in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"]):
        start, end = value["data_offsets"]
        require(start == cursor and end >= start, f"{path}: invalid offsets for {name}")
        require(value["dtype"] in DTYPE_BYTES, f"{path}: unsupported dtype for {name}")
        shape = value["shape"]
        require(all(isinstance(n, int) and n >= 0 for n in shape), f"{path}: invalid shape")
        expected = math.prod(shape) * DTYPE_BYTES[value["dtype"]]
        require(end - start == expected, f"{path}: size/shape mismatch for {name}")
        cursor = end
    require(path.stat().st_size == 8 + length + cursor, f"{path}: truncated/trailing payload")
    return tensors, 8 + length


def inspect_checkpoint(cache, model_id, revision=None):
    repo = cache / "hub" / ("models--" + model_id.replace("/", "--"))
    revision = revision or (repo / "refs/main").read_text().strip()
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), f"{model_id}: invalid refs/main")
    snapshot = repo / "snapshots" / revision
    config = json.loads((snapshot / "config.json").read_text())
    tensors, locations, shards = {}, {}, []
    for path in sorted(snapshot.glob("*.safetensors")):
        header, offset = read_header(path)
        require(not (tensors.keys() & header.keys()), f"{model_id}: duplicate tensors")
        tensors.update(header)
        locations.update({name: (path, offset) for name in header})
        shards.append({"name": path.name, "bytes": path.stat().st_size, "tensors": len(header)})
    require(bool(shards), f"{model_id}: no safetensors files")
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        require(set(weight_map) == set(tensors), f"{model_id}: index/tensor mismatch")
        require(all(locations[k][0].name == v for k, v in weight_map.items()),
                f"{model_id}: index points to the wrong shard")
    elif len(shards) != 1:
        raise ValueError(f"{model_id}: multiple shards without an index")
    text = config.get("text_config", config)
    report = {
        "model_id": model_id, "revision": revision, "snapshot": str(snapshot),
        "container_path": "/models/" + snapshot.relative_to(cache).as_posix(),
        "architectures": config.get("architectures"), "dtype": text.get("dtype"),
        "hidden_size": text.get("hidden_size"), "vocab_size": text.get("vocab_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "layer_types": text.get("layer_types"), "shards": shards,
        "weight_GiB": round(sum(s["bytes"] for s in shards) / 2**30, 3),
        "structural_check": "passed", "payload_checksums_verified": False,
    }
    return config, tensors, locations, report


def validate_target(config, report):
    require(config.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
            "Unexpected target architecture")
    text = config["text_config"]
    require((text["hidden_size"], text["num_hidden_layers"], text["vocab_size"])
            == (5120, 64, 248320), "Unexpected target dimensions")
    quant = config["quantization_config"]
    require(quant["quant_method"] == "compressed-tensors", "Expected compressed-tensors")
    groups = quant["config_groups"].values()
    require(all(g["weights"]["num_bits"] == 4 and g["weights"]["group_size"] == 32
                for g in groups), "Expected INT4 group_size=32")
    snapshot = Path(report["snapshot"])
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json"):
        require((snapshot / name).is_file(), f"Missing target asset: {name}")


def validate_draft(kind, config, tensors, locations, target):
    text = target["text_config"]
    require(config["hidden_size"] == text["hidden_size"], "Draft hidden_size mismatch")
    require(config["vocab_size"] == text["vocab_size"], "Draft input vocabulary mismatch")
    require(config["num_target_layers"] == text["num_hidden_layers"], "Target layer mismatch")
    layers = config.get("target_layer_ids", config.get("dflash_config", {}).get("target_layer_ids"))
    expected_layers = [5, 19, 33, 47, 61] if kind == "dflash2" else [1, 16, 31, 46, 61]
    require(layers == expected_layers, "Unexpected target feature layers")
    if kind == "dflash2":
        require(config["architectures"] == ["DFlash2DraftModel"], "Expected DFlash2")
        draft = config["dflash_config"]
        require(draft["block_size"] == 8, "DFlash2 expects anchor + 7 proposals")
        require(config["layer_types"] == ["sliding_attention"] * 5
                and config["sliding_window"] == 2048, "Expected DFlash2 5 SWA2048")
        require((draft["conv_kernel_size"], draft["conv_group_size"],
                 draft["selector_rank"], draft["selector_top_k"]) == (2, 16, 256, 16),
                "Unexpected DFlash2 convolution/selector configuration")
        for name, shape in {
            "candidate_selector.hidden_projection.weight": [256, 5120],
            "candidate_selector.predecessor_codebook": [248320, 256],
            "candidate_selector.successor_codebook": [248320, 256],
        }.items():
            require(name in tensors and tensors[name]["shape"] == shape,
                    f"Missing/incorrect DFlash2 tensor: {name}")
        return
    if kind == "dflash":
        require(config["architectures"] == ["DFlashDraftModel"], "Expected a DFlash checkpoint")
        require(config["block_size"] > 8, "DFlash block must fit anchor + 8 proposals")
        require(config["layer_types"] == ["sliding_attention"] * 4 + ["full_attention"],
                "Expected original DFlash 4 SWA + 1 full attention")
        require(config["sliding_window"] == 2048, "Unexpected DFlash sliding window")
        return
    require(config["architectures"] == ["Qwen3DSparkModel"], "Expected a DSpark checkpoint")
    require(config["block_size"] == 8, "Expected DSpark block8")
    require(config["layer_types"] == ["full_attention"] * 5, "Expected DSpark 5 full layers")
    require(config["rope_parameters"]["partial_rotary_factor"] == 1.0,
            "DSpark requires its own full-dimension RoPE")
    require(config["enable_confidence_head"], "Expected a trained confidence head")
    vocab, hidden, rank = text["vocab_size"], text["hidden_size"], config["markov_rank"]
    draft_vocab = config.get("draft_vocab_size", vocab)
    require(draft_vocab == (49167 if kind == "dspark-49k" else vocab), "Wrong DSpark vocabulary")
    for name, shape in {
        "embed_tokens.weight": [vocab, hidden], "lm_head.weight": [draft_vocab, hidden],
        "markov_head.markov_w1.weight": [vocab, rank],
        "markov_head.markov_w2.weight": [draft_vocab, rank],
        "confidence_head.proj.weight": [1, hidden + rank],
    }.items():
        require(name in tensors and tensors[name]["shape"] == shape,
                f"Missing/incorrect DSpark tensor: {name}")
    if draft_vocab < vocab:
        require("d2t" in tensors and tensors["d2t"]["dtype"] == "I64"
                and tensors["d2t"]["shape"] == [draft_vocab], "Missing/invalid d2t mapping")
        path, offset = locations["d2t"]
        start, end = tensors["d2t"]["data_offsets"]
        with path.open("rb") as stream:
            stream.seek(offset + start)
            offsets = [value[0] for value in struct.iter_unpack("<q", stream.read(end - start))]
        # vLLM maps draft ids via draft_id + d2t[draft_id]; d2t stores offsets.
        mapped = [index + value for index, value in enumerate(offsets)]
        require(len(set(mapped)) == draft_vocab and min(mapped) >= 0 and max(mapped) < vocab,
                "d2t contains duplicate/out-of-range target IDs")


def profile(cache, kind, reports):
    values = {
        "VLLM_HOST_MODELS_DIR": str(cache),
        "VLLM_UPSTREAM_IMAGE": "vllm/vllm-openai-rocm:v0.32.0@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905",
        "VLLM_IMAGE": "vllm-awq4-qwen:optimized-v0320",
        "VLLM_HOST_PORT": "8000",
        "VLLM_MODEL_ID": reports["target-only"]["container_path"],
        "VLLM_SERVED_MODEL_NAME": "Qwen3.6-27B-AWQ4",
        "VLLM_SPEC_METHOD": ("none" if kind == "target-only" else
                             "dflash" if kind == "dflash2" else kind.split("-")[0]),
        "VLLM_DRAFT_MODEL_ID": "" if kind == "target-only" else reports[kind]["container_path"],
        # This DFlash2 checkpoint overflows its grouped-conv/residual stream
        # in FP16 on this machine. BF16 restored finite states and acceptance.
        "VLLM_DTYPE": "bfloat16" if kind == "dflash2" else "half",
        "VLLM_DSPARK_DRAFT_TOPK": "128" if kind.startswith("dspark") else "",
        # v0.32 RDNA hybrid has a fast M<=5 path. On this machine DFlash N=4
        # beat N=8 in the fixed 512-token code and Chinese decode checks.
        "VLLM_SPEC_TOKENS": "7" if kind == "dflash2" else "4",
        "VLLM_DISABLE_DFLASH": "0",
        "VLLM_ENABLE_PREFIX_CACHING": "1",
        "VLLM_ENFORCE_EAGER": "1",
        "VLLM_GPU_MEMORY_UTIL": "0.65" if kind.startswith("dspark") else "0.55",
        "VLLM_MAX_NUM_SEQS": "1", "VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
        "VLLM_MAX_MODEL_LEN": "32768", "VLLM_KV_CACHE_DTYPE": "auto",
        # Auto at N=4 yields 816; a 64-token minimum aligns the physical block
        # to 832 and recovers prefill throughput on this Qwen hybrid model.
        "VLLM_BLOCK_SIZE": "64" if kind != "dflash2" else "",
        "VLLM_QWEN27B_PREFILL_TILES": "0" if kind == "dflash2" else "1",
        "VLLM_QWEN27B_PREFILL_ROCBLAS": "0" if kind == "dflash2" else "1",
        "VLLM_QWEN27B_DOWN_KMAJOR": "0" if kind == "dflash2" else "1",
        "VLLM_QWEN27B_PREFILL_ATTN": "0" if kind == "dflash2" else "1",
        "VLLM_QWEN27B_VERIFY_ATTN": "0" if kind == "dflash2" else "1",
        "VLLM_DECODE_GRAPHS": "0",
        "VLLM_ENABLE_PROJECT_MMQ": "0", "AWQ_MMQ_DECODE_BACKEND": "triton",
        "AWQ_MMQ_SMALL_M_THRESHOLD": "32",
    }
    lines = ["# Generated from locally audited snapshots. GPU acceptance is still required."]
    for key, value in values.items():
        require("\n" not in value and "\r" not in value, "Newlines are not supported in dotenv values")
        lines.append(key + "='" + value.replace("'", "\\'") + "'")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/home/amd/hf-cache"))
    parser.add_argument("--backend", choices=["all", *MODELS], default="all",
                        help="all audits the four original caches; dflash2 is optional")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".runtime/strix-halo")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace previously generated profiles")
    parser.add_argument("--write-default-env", action="store_true", help="Create .env for DFlash, only if absent")
    args = parser.parse_args()
    cache = args.cache.resolve()
    kinds = list(DEFAULT_BACKENDS) if args.backend == "all" else list(dict.fromkeys(["target-only", args.backend]))
    reports = {}
    target = None
    for kind in kinds:
        config, tensors, locations, report = inspect_checkpoint(cache, MODELS[kind],
                                                              PINNED_REVISIONS.get(kind))
        if kind == "target-only":
            validate_target(config, report)
            target = config
        else:
            validate_draft(kind, config, tensors, locations, target)
        reports[kind] = report
        print(f"PASS {kind}: {report['weight_GiB']} GiB, revision {report['revision']}")
    print("Checked headers, shapes, shard index, required assets and draft compatibility; no full-payload checksums or GPU inference.")
    if args.check_only:
        return
    contents = {args.output_dir / (kind + ".env"): profile(cache, kind, reports) for kind in kinds}
    if args.write_default_env:
        require("dflash" in reports, "--write-default-env requires DFlash inspection")
        default = ROOT / ".env"
        require(not default.exists(), ".env already exists; it will not be overwritten")
        contents[default] = profile(cache, "dflash", reports)
    for path, content in contents.items():
        require(args.force or not path.exists() or path.read_text() == content,
                f"{path} was changed; use --force explicitly to replace generated profiles")
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "cache": str(cache),
        "gpu_inference_verified": False, "models": reports,
    }
    path = args.output_dir / ("model-manifest.json" if args.backend == "all"
                             else f"model-manifest-{args.backend}.json")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Profiles and model manifest: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
