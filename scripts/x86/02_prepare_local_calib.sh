#!/usr/bin/env bash
set -euo pipefail

# x86 宿主机执行。
# 作用：
# 将仓库中的完整 JSONL 校准集复制到 TensorRT Edge-LLM 导出容器中。
#
# 默认读取：
#   data/calib_robo2vlm/train.jsonl
#
# 默认复制到容器内：
#   /tmp/calib_robo2vlm/train.jsonl
#
# 如需覆盖默认路径，可通过环境变量指定：
#   CONTAINER_NAME=edgellm-export
#   REPO_ROOT=/path/to/Qwen3-VL-8B-TRT
#   HOST_CALIB_DIR=/path/to/calib_robo2vlm
#   CONTAINER_CALIB_DIR=/tmp/calib_robo2vlm

CONTAINER_NAME=${CONTAINER_NAME:-edgellm-export}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-$DEFAULT_REPO_ROOT}

HOST_CALIB_DIR=${HOST_CALIB_DIR:-$REPO_ROOT/data/calib_robo2vlm}
CONTAINER_CALIB_DIR=${CONTAINER_CALIB_DIR:-/tmp/calib_robo2vlm}

echo "[INFO] Container name: $CONTAINER_NAME"
echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Host calibration dir: $HOST_CALIB_DIR"
echo "[INFO] Container calibration dir: $CONTAINER_CALIB_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker command not found. Please run this script on the x86 host with Docker installed."
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "[ERROR] Container '$CONTAINER_NAME' is not running."
  echo "[HINT] Start the export container first, for example:"
  echo "       bash scripts/x86/00_start_export_container.sh"
  exit 1
fi

if [ ! -f "$HOST_CALIB_DIR/train.jsonl" ]; then
  echo "[ERROR] Missing calibration file: $HOST_CALIB_DIR/train.jsonl"
  echo "[HINT] Expected file structure:"
  echo "       data/calib_robo2vlm/train.jsonl"
  echo "       data/calib_robo2vlm/README_calib.json"
  echo "       data/calib_robo2vlm/calib_distribution_report.txt"
  exit 1
fi

docker exec "$CONTAINER_NAME" mkdir -p "$CONTAINER_CALIB_DIR"

cat "$HOST_CALIB_DIR/train.jsonl" | \
  docker exec -i "$CONTAINER_NAME" sh -c "cat > $CONTAINER_CALIB_DIR/train.jsonl"

if [ -f "$HOST_CALIB_DIR/README_calib.json" ]; then
  cat "$HOST_CALIB_DIR/README_calib.json" | \
    docker exec -i "$CONTAINER_NAME" sh -c "cat > $CONTAINER_CALIB_DIR/README_calib.json"
fi

if [ -f "$HOST_CALIB_DIR/calib_distribution_report.txt" ]; then
  cat "$HOST_CALIB_DIR/calib_distribution_report.txt" | \
    docker exec -i "$CONTAINER_NAME" sh -c "cat > $CONTAINER_CALIB_DIR/calib_distribution_report.txt"
fi

echo "[INFO] Calibration files in container:"
docker exec "$CONTAINER_NAME" sh -c "ls -lh $CONTAINER_CALIB_DIR"

echo "[INFO] Calibration sample count:"
docker exec "$CONTAINER_NAME" wc -l "$CONTAINER_CALIB_DIR/train.jsonl"

echo "[INFO] Done."