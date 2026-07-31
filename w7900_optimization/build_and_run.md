# W7900 构建与运行

仓库根目录的 `Dockerfile` 是 Strix Halo/gfx1151 版本；W7900 必须使用根目录的 `Dockerfile.w7900`。该文件直接继承已验证的 AMD 镜像：

```text
rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0
```

镜像构建阶段会：

1. 从基础镜像 `/app/vllm` 复制源码到独立的 `/opt/vllm-w7900-023`；
2. 应用 `patch_w7900.py`，包含 PR #45207 回移和 gfx1100 attention 参数化；
3. 覆盖已验证的 `RDNA3W4A16LinearKernel` 非对称 `compressed-tensors` 实现；
4. 使用 `gfx1100` 重新编译 vLLM ROCm 扩展；
5. 通过 `start_container_w7900.sh` 启动 AWQ4、DFlash、TP 和 KV profile。

这条路径不使用根 Strix Halo Dockerfile、不调用 `patch_strix.py`，也不下载 ROCm 7.13 nightly wheel。`w7900_optimization/csrc/awq_mmq_gfx1100/` 是 standalone HIP MMQ 研究算子，和整模型 dispatcher 的 `RDNA3W4A16LinearKernel` 分开记录。

## A. 当前 AMD 容器内构建（推荐）

如果已经进入 AMD 提供的 ROCm 7.14/vLLM 0.23 容器：

```bash
cp w7900_optimization/.env.w7900.template w7900_optimization/.env
bash w7900_optimization/scripts/check_w7900_node.sh
bash w7900_optimization/scripts/prepare_local_vllm.sh
bash w7900_optimization/scripts/build_local_vllm.sh
bash w7900_optimization/scripts/start_local_vllm.sh
```

该流程使用容器已有的 `/app/vllm`、PyTorch、Triton 和 ROCm，不重新安装另一套 SDK。`build_local_vllm.sh` 中的目标固定为 `gfx1100`，工作树默认是 `/workspace/vllm-w7900-023`。

## B. 从 W7900 Dockerfile 构建

在有 Docker 和 W7900 设备的主机上：

```bash
cp w7900_optimization/.env.w7900.docker.template w7900_optimization/.env
# 编辑 VLLM_HOST_MODELS_DIR，使其包含 target 与 DFlash 模型目录

docker compose -f w7900_optimization/docker-compose.w7900.build.yml build
docker compose -f w7900_optimization/docker-compose.w7900.build.yml up -d
docker logs -f vllm-awq4-qwen-w7900
```

默认使用 GPU 0–1、TP=2 进行 bring-up。确认健康后再设置：

```text
W7900_VISIBLE_DEVICES=0,1,2,3
VLLM_TENSOR_PARALLEL_SIZE=4
```

或：

```text
W7900_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
VLLM_TENSOR_PARALLEL_SIZE=8
```

模型目录必须包含：

```text
VLLM_TARGET_MODEL=/models/Qwen3.6-27B-AWQ-INT4
VLLM_DRAFT_MODEL=/models/Qwen3.6-27B-DFlash
```

两者都应存在 `config.json`。模型权重不包含在 GitHub 仓库中。

## C. 架构检查

构建日志中应出现：

```text
PYTORCH_ROCM_ARCH=gfx1100
GPU_TARGETS=gfx1100
HIP_ARCHITECTURES=gfx1100
W7900 gfx1100: GPUs=...
```

容器内可检查：

```bash
python3 - <<'PY'
import torch, vllm
print("vllm", vllm.__version__)
print("torch", torch.__version__, "hip", torch.version.hip)
print("devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

整模型 RDNA3 后端命中时，服务日志应包含：

```text
Using RDNA3W4A16LinearKernel for CompressedTensorsWNA16
```

如果没有该日志，服务可能回退到 Triton W4A16；此时不能把结果写成 HIP 后端结果。

## D. 运行时 profile

| 目标 | 关键配置 |
|---|---|
| 8K AWQ4/DFlash | TP=1/2/4，DFlash N=8，tile=16 |
| 64K AWQ4 容量 | TP=2/4，DFlash 关闭，auto/FP8 KV A/B |
| 100K+ 长文 | 使用配套 BF16 TP=8 profile |
| 短请求多租户 | 两个同 NUMA 的 BF16 TP=4 服务 |

W7900 上 DFlash 不是全局开关：实测 8K 快 33.6%，12K 快 4.4%，16K 慢 6.6%。
