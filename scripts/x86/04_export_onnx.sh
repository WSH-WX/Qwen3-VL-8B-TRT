#!/usr/bin/env bash
set -euo pipefail

# 容器内执行。
# 目标：将量化后的 Qwen3-VL checkpoint 导出为 TensorRT Edge-LLM ONNX 中间格式。
#
# 注意：
# 不同 TensorRT Edge-LLM 版本的 export CLI 参数可能有差异。
# 如遇参数错误，请先运行：
#   tensorrt-edgellm-export --help

cd /workspace/TensorRT-Edge-LLM
source venv/bin/activate

QUANTIZED_DIR=${QUANTIZED_DIR:-/models/Qwen3-VL-8B-Instruct-EdgeLLM/quantized}
OUTPUT_DIR=${OUTPUT_DIR:-/models/Qwen3-VL-8B-Instruct-EdgeLLM/onnx}
LOG_FILE=${LOG_FILE:-/logs/qwen3vl8b_export_onnx.log}

mkdir -p "$OUTPUT_DIR"

tensorrt-edgellm-export \
  "$QUANTIZED_DIR" \
  "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo "[INFO] ONNX files:"
find "$OUTPUT_DIR" -maxdepth 3 -type f | sort | head -n 100
