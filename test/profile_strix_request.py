"""Capture a bounded diagnostic trace, separately from throughput benchmarks."""
import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from bench_strix_decode import PROMPTS
from bench_strix_prefill import PASSAGE, token_ids
from smoke_strix import Client, ROOT, require
from server import App, Config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", required=True)
parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
args = parser.parse_args()
require(re.fullmatch(r"[A-Za-z0-9_-]+", args.label), "Invalid label")
path = ROOT / ".runtime/strix-halo" / ("diagnostic-"+args.label+".json")
require(not path.exists(), "Refusing to overwrite a diagnostic")
client=Client("http://127.0.0.1:8000", "Qwen3.6-27B-AWQ4", 600)
metrics=App(Config(upstream=client.host, model=client.model)).metrics()
require(metrics.ok and not any(metrics.values.get(k,0) for k in
        ("vllm:num_requests_running","vllm:num_requests_waiting")), "Service busy")
if args.phase=="prefill":
    prompt=token_ids(client,PASSAGE*150)[:4096]
    require(len(prompt)==4096,"Not enough input tokens")
    route="/v1/completions"
    payload={"model":client.model,"prompt":prompt,"add_special_tokens":False,
             "max_tokens":8,"temperature":0,"ignore_eos":True,"cache_salt":uuid.uuid4().hex}
    bounds={"delay_iterations":0,"max_iterations":0}
else:
    route="/v1/chat/completions"
    payload=client.chat(PROMPTS["code"])
    payload.update(max_tokens=128,cache_salt=uuid.uuid4().hex)
    bounds={"delay_iterations":1,"max_iterations":16}
with client.request("/start_profile",{"profile_prefix":args.label,**bounds}) as response:
    require(response.status==200,"Profiler did not start")
try:
    result=client.json(route,payload)
finally:
    with client.request("/stop_profile",{}) as response:
        require(response.status==200,"Profiler did not stop")
require(result["usage"]["prompt_tokens_details"]["cached_tokens"]==0,"Unexpected cache hit")
path.write_text(json.dumps({"label":args.label,"phase":args.phase,
    "created_at":datetime.now(timezone.utc).isoformat(),
    "scope":"Instrumented diagnostic; never use this request as headline speed",
    "profile_bounds":bounds,"request":payload,"response":result},ensure_ascii=False,indent=2)+"\n")
print("Diagnostic:",path)
