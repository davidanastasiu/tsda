#!/bin/bash
set -e

# Move to TSDA root (one level up from evaluation/)
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
DATA_DIR="data"
GT_ROOT="data/WTS"
VAL_RESULTS_DIR="vqa_results_val"

# Model predictions are inside vqa_results_val
MODELS="$VAL_RESULTS_DIR/mid_bdd.json,\
$VAL_RESULTS_DIR/mid_all_group_qas.json,\
$VAL_RESULTS_DIR/even_bdd.json,\
$VAL_RESULTS_DIR/even_wts.json,\
$VAL_RESULTS_DIR/mid_all_e2_5.json,\
$VAL_RESULTS_DIR/mid_all_e2.json,\
$VAL_RESULTS_DIR/mid_all_e3.json"

python multi_agent_selection/select_top_qa_agents_val.py \
  --models "$MODELS" \
  --gt_dirs $GT_ROOT/vqa/val $GT_ROOT/BDD_anno/vqa/val \
  --questions_file $DATA_DIR/final_test_data/SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json \
  --output_path multi_agent_selection/multi_agent_results_vqa.json \
  --best_agents_path multi_agent_selection/best_QA_agents_vqa.json
