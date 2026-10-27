# Strix Halo stable profile. The upstream release owns the matched ROCm stack.
# Legacy source build and patches: Dockerfile.legacy-v020.
ARG VLLM_UPSTREAM_IMAGE=vllm/vllm-openai-rocm:v0.32.0
FROM ${VLLM_UPSTREAM_IMAGE}

USER root
WORKDIR /opt/amd-project
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG VLLM_ENABLE_PROJECT_MMQ=0
ENV VLLM_EXPECTED_VERSION=0.32.0 \
    VLLM_ENABLE_PROJECT_MMQ=${VLLM_ENABLE_PROJECT_MMQ} \
    VLLM_ROCM_USE_AITER=0 \
    VLLM_USE_TRITON_AWQ=1 \
    PYTORCH_ROCM_ARCH=gfx1151 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    MIOPEN_FIND_MODE=FAST

COPY scripts/start_vllm_stable.py scripts/register_mmq_stable.py scripts/patch_rdna_prefill.py scripts/register_rdna_down.py scripts/patch_rdna_attention.py /opt/amd-project/
COPY csrc/awq_mmq_gfx1151 /opt/amd-kernels/awq_mmq_gfx1151
COPY csrc/rdna_qwen27b /opt/amd-kernels/rdna_qwen27b

# Validate the actual installed package; changing an old VLLM_COMMIT variable
# cannot silently select another engine in this profile.
RUN python3 -c "from importlib.metadata import version; v=version('vllm'); assert v.split('+', 1)[0] == '0.32.0', v"
RUN python3 /opt/amd-project/patch_rdna_prefill.py
RUN python3 /opt/amd-project/patch_rdna_attention.py
RUN python3 /opt/amd-project/register_rdna_down.py
RUN if [[ "$VLLM_ENABLE_PROJECT_MMQ" == "1" ]]; then \
      if ! (command -v c++ && command -v ninja && command -v pkg-config) >/dev/null; then \
        apt-get update && apt-get install -y --no-install-recommends \
          build-essential ninja-build pkg-config; \
      fi && \
      cd /opt/amd-kernels/awq_mmq_gfx1151 && \
      python3 setup.py build_ext --inplace && \
      python3 /opt/amd-project/register_mmq_stable.py && \
      rm -rf /var/lib/apt/lists/*; \
    elif [[ "$VLLM_ENABLE_PROJECT_MMQ" != "0" ]]; then \
      echo "VLLM_ENABLE_PROJECT_MMQ must be 0 or 1" >&2; exit 2; \
    fi

RUN python3 /opt/amd-project/start_vllm_stable.py --print-command > /opt/amd-project/default-launch.json
LABEL org.opencontainers.image.base.name="vllm/vllm-openai-rocm:v0.32.0"
ENTRYPOINT ["python3", "/opt/amd-project/start_vllm_stable.py"]
CMD []
