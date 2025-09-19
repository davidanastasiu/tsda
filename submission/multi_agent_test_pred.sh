#!/bin/bash

TEST_PRED_DIR="../vqa_results_test"
BASE_VARIANT="mid_all_e3"
FILE_NAME="veh_augmented.jsonl"
BEST_JSON="../multi_agent_selection/best_QA_agents_vqa.json"
OUTPUT_JSON="$TEST_PRED_DIR/multi_agent_pred/multi_agent_test_pred.jsonl"
VQA_JSON="data/final_test_data/SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json"


echo "[INFO] Running multi-agent routing + postprocess pipeline..."
export PYTHONPATH=$(realpath ..)

python multi_agent_test_pred.py \
  --test_predictions "$TEST_PRED_DIR" \
  --base_variant "$BASE_VARIANT" \
  --file_name "$FILE_NAME" \
  --best "$BEST_JSON" \
  --output "$OUTPUT_JSON" \
  --vqa_json "$VQA_JSON" \
  --postprocess

echo "Done."
