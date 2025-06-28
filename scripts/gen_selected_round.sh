#!/usr/bin/env bash

# Resolve project root (parent directory of this script)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Ensure project root is on PYTHONPATH so that `src` package can be imported
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

AGENT_NUM_LIST=(2 3 4)
DOMAIN_LIST=("Persuassion_Deliberation_and_Negotiation" "Inquiry_and_Information_Seeking" "Eristic")

DATASET_NUM=50
OUTPUT_DIR="${PROJECT_ROOT}/outputs/selected_round"
MAX_TURNS=20
SELECTED_ROUND=4
OUTPUT_FILE="$OUTPUT_DIR/round_$SELECTED_ROUND.json"
mkdir -p "$OUTPUT_DIR"

TASK_TYPE="multi_round"
if [[ $SELECTED_ROUND -eq 1 ]]; then
  TASK_TYPE="single_round"
fi

if [[ ! -f "$OUTPUT_FILE" ]]; then
  echo "[]" > "$OUTPUT_FILE"
fi

for agent_num in "${AGENT_NUM_LIST[@]}"; do
  for domain in "${DOMAIN_LIST[@]}"; do
    echo "=========="
    echo "▶ Generating data: round=$SELECTED_ROUND, agent_num=$agent_num, domain=$domain, task=$TASK_TYPE"
    echo "Appending data to: $OUTPUT_FILE"
    echo "=========="

    uv run python -m src.main --agent_num $agent_num --rounds_num $SELECTED_ROUND \
        --fewshot "" --domain "$domain" --output_path "$OUTPUT_FILE" \
        --task "$TASK_TYPE" --dataset_num $DATASET_NUM --max_turns $MAX_TURNS

  done
done

echo "All done for round $SELECTED_ROUND!"