#!/usr/bin/env bash
set -euo pipefail

# x86 宿主机执行。
# 打包 ONNX 目录并生成 sha256。
# 注意：ONNX 压缩包很大，不上传 GitHub。

ARTIFACT_DIR=${ARTIFACT_DIR:-/mnt/data/pc/edgellm_x86/artifacts}
ONNX_PARENT=${ONNX_PARENT:-/mnt/data/pc/edgellm_x86/models/Qwen3-VL-8B-Instruct-EdgeLLM}
ARCHIVE_NAME=${ARCHIVE_NAME:-Qwen3-VL-8B-Instruct-INT4AWQ-ONNX.tar.gz}

mkdir -p "$ARTIFACT_DIR"

tar -C "$ONNX_PARENT" \
  -czf "$ARTIFACT_DIR/$ARCHIVE_NAME" \
  onnx

cd "$ARTIFACT_DIR"

sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"

ls -lh "$ARCHIVE_NAME" "$ARCHIVE_NAME.sha256"
cat "$ARCHIVE_NAME.sha256"

echo "[INFO] Archive preview:"
tar -tzf "$ARCHIVE_NAME" | head -n 50
