#!/usr/bin/env bash
set -euo pipefail

# 容器内执行：安装 TensorRT Edge-LLM Python CLI

cd /workspace/TensorRT-Edge-LLM

python3 -m venv --system-site-packages venv
source venv/bin/activate

# 避免 pip 依赖解析卡住，先安装本项目本体，再手动补依赖。
pip3 install -e . --no-deps --no-build-isolation -v \
  2>&1 | tee /logs/edgellm_pip_install_nodeps_clean.log

pip3 install transformers==5.9.0 --no-deps

pip3 install \
  "huggingface-hub>=1.5,<2.0" \
  "tokenizers>=0.22,<0.23" \
  regex \
  pyyaml \
  requests \
  tqdm \
  filelock \
  datasets \
  accelerate \
  safetensors \
  sentencepiece \
  protobuf

which tensorrt-edgellm-export
which tensorrt-edgellm-quantize
