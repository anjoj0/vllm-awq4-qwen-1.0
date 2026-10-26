"""Read only public model JSON and a bounded safetensors JSON header.

Never imports remote code, downloads tensor payloads, or uses API credentials.
"""
import json
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener()
REPOS = ["abstract-extraordinary/Qwen3.6-27B-DSpark", "abstract-extraordinary/Qwen3.6-27B-DSpark-49k"]

def read_json(url):
    with OPENER.open(url, timeout=45) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("JSON exceeded bounded metadata read")
    return json.loads(raw)

output = {}
for repo in REPOS:
    api = read_json("https://huggingface.co/api/models/" + repo)
    revision = api["sha"]
    config = read_json("https://huggingface.co/" + repo + "/resolve/" + revision + "/config.json")
    url = "https://huggingface.co/" + repo + "/resolve/" + revision + "/model.safetensors"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-262143"})
    with OPENER.open(request, timeout=60) as response:
        first = response.read(8)
        header_size = struct.unpack("<Q", first)[0]
        if header_size > 262136:
            raise ValueError("Safetensors metadata header exceeded bounded read")
        header_raw = response.read(header_size)
        if len(header_raw) != header_size:
            raise ValueError("Incomplete metadata header")
        header = json.loads(header_raw)
    output[repo] = {"sha": revision, "lastModified": api.get("lastModified"), "config": config,
                    "header_bytes_read": 8 + header_size, "tensors": header}
    print(repo, "revision", revision, "header_bytes", 8 + header_size, "tensor_count", len(header) - int("__metadata__" in header))
    for key, value in header.items():
        if any(term in key.lower() for term in ["markov", "confidence", "d2t", "t2d", "lm_head", "embed_tokens"]):
            print(key, json.dumps(value))
target = ROOT / "hf_dspark_metadata.json"
target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved", target)
