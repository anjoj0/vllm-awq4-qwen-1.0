# Strix Halo：升级至 vLLM v0.32.0

核查日期：2026-10-26。设备：Ryzen AI Max+ 395 / Radeon 8060S / gfx1151。宿主机已知为 Ubuntu 26.04.1、7.0.0-30-generic，系统可见内存约 122 GiB。

## 固定版本与验证边界

截至核查日，[最新非预发布版本为 v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)，于 2026-10-18 发布，对应源码 `5df63a7203cdb9328d824a27b682e77c7b9928be`。默认 Docker 构建继承 `vllm/vllm-openai-rocm:v0.32.0`，沿用上游镜像配套的 ROCm、PyTorch、Triton 和运行时依赖，构建时核对实际安装的 vLLM 版本。

[v0.32.0 GPU 安装说明](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/)列出 gfx1151，并推荐官方 ROCm 镜像。发布说明中 ROCm 默认启用 Model Runner V2；固定标签源码已包含 DFlash、DSpark、RDNA W4A16 和非因果 Triton attention。它们是选用该版本的依据，不能替代这台笔记本的验收。

最初的迁移准备在 Windows 上只完成静态检查。2026-10-26 已在这台 Linux/gfx1151 机器构建上游与项目 MMQ 镜像，完成 target-only、DFlash、DSpark full/49k、DFlash2 的接口验收，并接通真实展示页。实际版本、固定权重、本机性能与验收边界见 [本地部署记录](STRIX_HALO_LOCAL_DEPLOYMENT.md)。原有 28.3 token/s 属于旧环境历史结果，不作为新默认构建的性能保证。

## 新旧构建

- 默认：`Dockerfile`、`docker-compose.yml`、`.env.template`，固定 v0.32.0。
- 回退：`Dockerfile.legacy-v020`、`docker-compose.legacy-v020.yml`、`.env.legacy-v020.template`，保留旧版 TheRock SDK 与 v0.20.0 补丁构建。
- 新默认不执行旧 `patch_strix.py`，也不在新镜像上重新安装旧版 Torch/Triton wheel。
- 历史 `VLLM_COMMIT` 仅供 legacy 构建使用；新默认通过 `VLLM_UPSTREAM_IMAGE` 选择基础镜像。
- 新版复用上游 RDNA 算术，并提供经过本机验证的 Qwen27B 投影布局、prefill tile/rocBLAS 和 attention 启动参数适配。各开关与最新实测见本地部署记录；项目旧 HIP MMQ 仍为单独的构建开关。

## 雷神上首次部署

在项目目录执行。以下复制命令用于首次配置；已有自己的 `.env` 时，按模板补齐新字段。

```bash
cp .env.template .env
sed -i 's|^VLLM_HOST_MODELS_DIR=.*|VLLM_HOST_MODELS_DIR=/home/amd/hf-cache|' .env

docker compose config --quiet
docker compose build --pull
docker compose run --rm --no-deps vllm --print-command
docker compose run --rm --no-deps vllm --check-env
```

`--print-command` 只显示参数；`--check-env` 会核对 vLLM 版本、ROCm PyTorch、GPU 可用性和实际 `gfx1151` 架构，并打印 GPU 可见容量，不加载目标权重。

上面通过后启动服务：

```bash
docker compose up -d --force-recreate
docker compose logs -f --tail=100 vllm
```

模板使用单流、32K、FP16 target/auto KV、eager 和 DFlash N4。DFlash2 的本机 profile 单独使用 BF16，避免已确认的 FP16 grouped-conv 溢出。宿主机的 TTM/GTT、`/dev/shm` 与模型 KV 预算仍需按实际检测结果处理。

本项目不再写死 `127.0.0.1:7890` 构建代理；需要时设置对应代理环境变量。镜像拉取经过 Docker daemon，其代理配置需单独生效。

## 验收顺序

先把 `.env` 中 `VLLM_SPEC_METHOD=none`，验证 target-only；再分别测试 DFlash 与 DSpark。每次修改配置后执行 `docker compose up -d --force-recreate`。

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-AWQ4","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

再覆盖长输出、8K/16K/32K、图像、工具与 Responses API。`test/sanity.py` 的旧性能区间可能不适用于新版，应分别判断内容正确性和吞吐。

### DFlash

```dotenv
VLLM_SPEC_METHOD=dflash
VLLM_DRAFT_MODEL_ID=z-lab/Qwen3.6-27B-DFlash
VLLM_SPEC_TOKENS=4
```

本机 v0.32 固定长输出对照选择 N=4；旧 v0.20 的 N=8 结论不直接移植为新版最优值。详见 [本地实测](STRIX_HALO_LOCAL_DEPLOYMENT.md)。

### DSpark 完整词表

```dotenv
VLLM_SPEC_METHOD=dspark
VLLM_DRAFT_MODEL_ID=abstract-extraordinary/Qwen3.6-27B-DSpark
VLLM_SPEC_TOKENS=4
VLLM_BLOCK_SIZE=64
VLLM_DSPARK_DRAFT_TOPK=128
```

49k 对照只更换草稿 ID：

```dotenv
VLLM_DRAFT_MODEL_ID=abstract-extraordinary/Qwen3.6-27B-DSpark-49k
```

两份 DSpark 都是社区候选；作者在 NVFP4/GB10 上的成绩不能直接归因到当前 AWQ4/RDNA 环境。启动、正确性、接受长度和峰值内存均需验证。完整原理与权重核验见 [DSpark/DFlash2 评估](DSpark与DFlash2_替换DFlash可行性_双后端与开源方案评估_20261022.md)。

## 可选项目 HIP MMQ

本机旧 MMQ 已慢于选定配置，以下只作为研究入口。需要使用独立镜像标签，并关闭本轮目标算子适配，避免与 down dispatcher 冲突。在单独的 `.env` 副本设置后再构建：

```dotenv
VLLM_IMAGE=vllm-awq4-qwen:mmq-v0320
VLLM_ENABLE_PROJECT_MMQ=1
VLLM_QWEN27B_PREFILL_TILES=0
VLLM_QWEN27B_PREFILL_ROCBLAS=0
VLLM_QWEN27B_DOWN_KMAJOR=0
VLLM_QWEN27B_PREFILL_ATTN=0
VLLM_QWEN27B_VERIFY_ATTN=0
```

```bash
docker compose build
docker compose run --rm --no-deps vllm --check-env
docker compose run --rm --no-deps \
  --entrypoint python3 vllm \
  /opt/amd-kernels/awq_mmq_gfx1151/test_correctness.py
docker compose up -d --force-recreate
```

镜像会使用自己的 PyTorch/HIP 编译项目扩展，注册脚本限定 v0.32.0。适配器兼容 v0.20 四元组和 v0.32 三元组权重接口，并处理新版缺少 `has_g_idx` 字段的情况。这里没有改变 HIP 数值内核。2026-10-26 已完成 gfx1151 编译、两项数值检查及整模型实际路由验证；本次 4K prefill 上游 RDNA hybrid 快于旧 HIP v1，所以没有把 MMQ 设为默认，详见本地记录。双权重布局带来的额外占用也仍存在。

## 回退入口

保留当前 `.env`，另建旧环境配置：

```bash
cp .env.legacy-v020.template .env.legacy-v020
# 编辑缓存路径，并检查 legacy Compose 内的历史代理配置。
docker compose -f docker-compose.legacy-v020.yml \
  --env-file .env.legacy-v020 build
docker compose -f docker-compose.legacy-v020.yml \
  --env-file .env.legacy-v020 up -d --force-recreate
```

新旧入口使用同一服务/容器名，按需选择一套；Triton 与 vLLM 缓存默认分目录，避免跨版本复用。
