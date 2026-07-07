#!/usr/bin/env bash
set -euo pipefail

# 容器内执行。
# 目标：Qwen3-VL-8B-Instruct -> INT4 AWQ 量化 checkpoint。
#
# 注意：
# 不同 TensorRT Edge-LLM 版本的 CLI 参数可能有差异。
# 如果报参数错误，请先运行：
#   tensorrt-edgellm-quantize llm --help
# 然后按当前版本调整参数名。

cd /workspace/TensorRT-Edge-LLM
source venv/bin/activate

MODEL_DIR=${MODEL_DIR:-/models/Qwen3-VL-8B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-/models/Qwen3-VL-8B-Instruct-EdgeLLM/quantized}
CALIB_FILE=${CALIB_FILE:-/tmp/calib_robo2vlm/train.jsonl}
LOG_FILE=${LOG_FILE:-/logs/qwen3vl8b_int4_awq_quantize.log}

mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tensorrt-edgellm-quantize llm \
  --model "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --quantization int4_awq \
  --calib-dataset "$CALIB_FILE" \
  2>&1 | tee "$LOG_FILE"
