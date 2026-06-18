#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-vllm-awq4-qwen:w7900-gfx1100}"
VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

echo "== Host kernel =="
uname -a

echo
echo "== Host ROCm tools =="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | grep -E 'Name:.*gfx|Marketing Name' || true
else
  echo "rocminfo not found on host"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname --showbus --showdriverversion || true
else
  echo "rocm-smi not found on host"
fi

echo
echo "== Container torch smoke =="
"${DOCKER_BIN}" run --rm \
  --entrypoint /bin/bash \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HIP_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  "${IMAGE}" \
  -lc 'python - <<PY
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
count = torch.cuda.device_count()
print("device_count", count)
for i in range(count):
    name = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(i)
    print(f"device[{i}]", name, "total_memory_gib", round(props.total_memory / 1024**3, 2))
    x = torch.zeros((1024, 1024), device=f"cuda:{i}")
    torch.cuda.synchronize(i)
    print(f"alloc_ok[{i}]", tuple(x.shape), x.dtype)
PY'
