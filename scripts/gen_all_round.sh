#!/usr/bin/env bash

# Resolve project root (parent directory of this script)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Ensure project root is on PYTHONPATH so that `src` package can be imported
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

AGENT_NUM_LIST=(2 3 4)
DOMAIN_LIST=("Persuassion" "Negotiation" "Inquiry" "Deliberation" "Information-Seeking" "Eristic")
ROUND_LIST=(1 2 3 4)

DATASET_NUM=1
OUTPUT_DIR="${PROJECT_ROOT}/outputs/all_rounds"
MAX_TURNS=10
TASK_TYPE="multi_round"
mkdir -p "$OUTPUT_DIR"

for agent_num in "${AGENT_NUM_LIST[@]}"; do
  for domain in "${DOMAIN_LIST[@]}"; do
    for round_num in "${ROUND_LIST[@]}"; do
      for repeat in 1 2; do
        echo "=========="
        echo "▶ agent_num=$agent_num, domain=$domain, round_num=$round_num, repeat=$repeat"
        echo "=========="
        
        # Determine task type based on round number (single vs multi)
        if [[ $round_num -eq 1 ]]; then
          TASK_TYPE="single_round"
        else
          TASK_TYPE="multi_round"
        fi

        OUTPUT_FILE="$OUTPUT_DIR/round_${round_num}.json"

        # Run src.main as a module so absolute imports work
        uv run python -m src.main --agent_num $agent_num --rounds_num $round_num \
            --fewshot "" --domain "$domain" --output_path "$OUTPUT_FILE" \
            --task "$TASK_TYPE" --dataset_num $DATASET_NUM --max_turns $MAX_TURNS
      done
    done
  done
done