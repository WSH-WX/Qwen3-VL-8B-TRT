#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "Usage:"
  echo "  $0 <task_name> <input_json> <output_json> <profile_json> <runtime_log> <tegrastats_log> <max_generate_length>"
  exit 1
fi

TASK_NAME="$1"
INPUT_FILE="$2"
OUTPUT_FILE="$3"
PROFILE_FILE="$4"
RUNTIME_LOG="$5"
TEGRA_LOG="$6"
MAX_GEN="$7"

EDGELLM_ROOT="/mnt/ssd/TensorRT-Edge-LLM"
MODEL_ROOT="/mnt/ssd/edgellm_models/Qwen3-VL-8B-Instruct-INT4AWQ"
ENGINE_DIR="$MODEL_ROOT/engine/llm_unified_4096_8192"
MM_ENGINE_DIR="$MODEL_ROOT/engine"
PLUGIN_PATH="$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so"
LLM_BIN="$EDGELLM_ROOT/build/examples/llm/llm_inference"

mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$PROFILE_FILE")" "$(dirname "$RUNTIME_LOG")" "$(dirname "$TEGRA_LOG")"

{
  echo "============================================================"
  echo "TASK_NAME=$TASK_NAME"
  echo "INPUT_FILE=$INPUT_FILE"
  echo "OUTPUT_FILE=$OUTPUT_FILE"
  echo "PROFILE_FILE=$PROFILE_FILE"
  echo "RUNTIME_LOG=$RUNTIME_LOG"
  echo "TEGRA_LOG=$TEGRA_LOG"
  echo "MAX_GEN=$MAX_GEN"
  echo "ENGINE_DIR=$ENGINE_DIR"
  echo "MM_ENGINE_DIR=$MM_ENGINE_DIR"
  echo "PLUGIN_PATH=$PLUGIN_PATH"
  echo "LLM_BIN=$LLM_BIN"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
} | tee "$RUNTIME_LOG"

test -x "$LLM_BIN"
test -f "$PLUGIN_PATH"
test -d "$ENGINE_DIR"
test -d "$MM_ENGINE_DIR"
test -f "$INPUT_FILE"

cd "$EDGELLM_ROOT"
export EDGELLM_PLUGIN_PATH="$PLUGIN_PATH"

HELP="$("$LLM_BIN" --help 2>&1 || true)"

CMD=(
  "$LLM_BIN"
  --engineDir "$ENGINE_DIR"
  --multimodalEngineDir "$MM_ENGINE_DIR"
  --inputFile "$INPUT_FILE"
  --outputFile "$OUTPUT_FILE"
  --dumpOutput
  --maxGenerateLength "$MAX_GEN"
)

if echo "$HELP" | grep -q -- "--dumpProfile"; then
  CMD+=(--dumpProfile)
fi

if echo "$HELP" | grep -q -- "--profileOutputFile"; then
  CMD+=(--profileOutputFile "$PROFILE_FILE")
fi

if echo "$HELP" | grep -q -- "--batchSize"; then
  CMD+=(--batchSize 1)
fi

if echo "$HELP" | grep -q -- "--warmup"; then
  CMD+=(--warmup 1)
fi

echo "Command:" | tee -a "$RUNTIME_LOG"
printf '%q ' "${CMD[@]}" | tee -a "$RUNTIME_LOG"
echo | tee -a "$RUNTIME_LOG"

tegrastats --interval 1000 --logfile "$TEGRA_LOG" &
TEGRA_PID=$!

cleanup() {
  kill "$TEGRA_PID" 2>/dev/null || true
  wait "$TEGRA_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 3

if [ -x /usr/bin/time ]; then
  /usr/bin/time -v "${CMD[@]}" 2>&1 | tee -a "$RUNTIME_LOG"
else
  echo "[WARN] /usr/bin/time not found; running without GNU time -v." | tee -a "$RUNTIME_LOG"
  echo "[WARN] Process-level time -v metrics will be unavailable, but tegrastats and Edge-LLM logs will still be recorded." | tee -a "$RUNTIME_LOG"
  "${CMD[@]}" 2>&1 | tee -a "$RUNTIME_LOG"
fi

sleep 3

echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RUNTIME_LOG"
echo "Done: $TASK_NAME" | tee -a "$RUNTIME_LOG"
