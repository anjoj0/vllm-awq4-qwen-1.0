# Build And Run On W7900

The root project Dockerfile is Strix Halo (`gfx1151`) first. For W7900, use the
files in this directory.

## 1. Prepare Environment

From the repo root:

```bash
cp w7900_optimization/.env.w7900.template .env
```

Edit `.env`:

- set `VLLM_HOST_MODELS_DIR` to the host HuggingFace cache directory;
- set proxy variables only if needed;
- keep `W7900_GFX=gfx1100`;
- start with `W7900_VISIBLE_DEVICES=0,1` and `VLLM_TENSOR_PARALLEL_SIZE=2`.

The model cache should contain:

- `cyankiwi/Qwen3.6-27B-AWQ-INT4`
- `z-lab/Qwen3.6-27B-DFlash`

## 2. Build

```bash
sudo docker compose -f w7900_optimization/docker-compose.w7900.build.yml build
```

If the AMD nightly wheel index does not contain the pinned gfx1100 wheels for
`TORCH_ROCM_DATE=20260510`, update `TORCH_ROCM_DATE` only after verifying the
matching TheRock tarball exists and the build completes. Do not float versions
silently.

## 3. Hardware Smoke

```bash
DOCKER_BIN="sudo docker" \
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash w7900_optimization/scripts/check_w7900_node.sh
```

Expected:

- PyTorch sees 8 devices;
- each device allocates a small CUDA/HIP tensor;
- device name should be Radeon PRO W7900 or equivalent;
- vLLM ROCm platform should resolve to `gfx1100`.

## 4. Run TP=2

```bash
sudo docker compose -f w7900_optimization/docker-compose.w7900.build.yml up -d
```

Check:

```bash
sudo docker logs -f vllm-awq4-qwen-w7900
```

Health check:

```bash
python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10).status)'
```

## 5. Sweep

Print suggested commands:

```bash
bash w7900_optimization/scripts/print_w7900_sweep_commands.sh
```

Recommended order:

1. single GPU, 64K, fp8 KV;
2. TP=2, 128K, fp8 KV;
3. TP=4, 256K, fp8 KV;
4. TP=8 only if TP=4 cannot meet context/concurrency goals.

## Current Caveat

This W7900 build path has been prepared for clone-and-build, but it has not yet
been validated on the physical W7900 node from this workspace. Treat the first
build as bring-up, not as a guaranteed final performance image.
