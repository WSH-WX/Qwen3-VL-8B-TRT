#!/usr/bin/env bash
set -euo pipefail

# x86 / RTX 4090 主机：启动 TensorRT Edge-LLM ONNX 导出容器
# 本实验使用 NVIDIA PyTorch 26.05 容器。

docker stop edgellm-export 2>/dev/null || true
docker rm edgellm-export 2>/dev/null || true

docker run --gpus '"device=0"' -it \
  --name edgellm-export \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt/data/pc/edgellm_x86/workspace:/workspace \
  -v /mnt/data/pc/edgellm_x86/hf_cache:/root/.cache/huggingface \
  -v /mnt/data/pc/edgellm_x86/logs:/logs \
  -v /mnt/data/pc/edgellm_x86/models:/models \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.05-py3 \
  bash
