"""Fetch public DFlash2 metadata and bounded JSON tensor headers, never tensors."""
import json
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener()

def read_json(url):
    with OPENER.open(url, timeout=60) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Metadata JSON too large")
    return json.loads(raw)

search = read_json("https://huggingface.co/api/models?" + urllib.parse.urlencode({"search": "dflash2", "limit": "1000"}))
output = {"search": {"query": "dflash2", "limit": 1000, "count": len(search), "ids": [item["id"] for item in search]}, "models": {}}
print("search_count", len(search))
for item in search:
    if "qwen3.6" in item["id"].lower():
        print("candidate", item["id"])

repos = sys.argv[1:] or ["z-lab/Qwen3.6-27B-DFlash2", "z-lab/Qwen3.8-27B-DFlash2", "incoai/Qwen3.8-27B-DFlash2"]
for repo in repos:
    try:
        api = read_json("https://huggingface.co/api/models/" + repo + "?blobs=true")
    except urllib.error.HTTPError as err:
        output["models"][repo] = {"http_status": err.code}
        print(repo, "HTTP", err.code)
        continue
    rev = api["sha"]
    config = read_json("https://huggingface.co/" + repo + "/resolve/" + rev + "/config.json")
    request = urllib.request.Request("https://huggingface.co/" + repo + "/resolve/" + rev + "/model.safetensors", headers={"Range": "bytes=0-262143"})
    with OPENER.open(request, timeout=60) as response:
        head_len = struct.unpack("<Q", response.read(8))[0]
        if head_len > 262136:
            raise ValueError("Header too large")
        raw = response.read(head_len)
        if len(raw) != head_len:
            raise ValueError("Incomplete header")
        header = json.loads(raw)
    output["models"][repo] = {"revision": rev, "lastModified": api.get("lastModified"), "siblings": api.get("siblings"), "config": config, "header_bytes_read": 8 + head_len, "tensors": header}
    print(repo, "revision", rev, "header_bytes", 8 + head_len, "tensor_count", len(header) - int("__metadata__" in header))
    for key, val in header.items():
        if any(term in key.lower() for term in ["selector", "conv", "lm_head", "embed_tokens", "confidence", "markov", "d2t"]):
            print(key, json.dumps(val))
target = ROOT / ("hf_dflash2_qwen36_metadata.json" if sys.argv[1:] else "hf_dflash2_metadata.json")
target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved", target)
