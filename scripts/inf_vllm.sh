#!/usr/bin/env bash

# Resolve project root (parent directory of this script)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Ensure project root is on PYTHONPATH so that `src` package can be imported
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Wrapper script to run src.inference.inference_vllm without relying on YAML
# Adjust the variables below as needed.

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
FUNCTION_DOCS="../src/graph/tool_docs.json"
DATASET_DIR="../data"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/inf_vllm"
MAX_TOKENS=8196
TP_SIZE=1  # tensor_parallel_size

# Create output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

uv run python -m src.inference.inference_vllm --model_name "$MODEL_NAME" \
  --function_docs "$FUNCTION_DOCS" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" --max_tokens $MAX_TOKENS \
  --tensor_parallel_size $TP_SIZE 