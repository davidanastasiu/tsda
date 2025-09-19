#!/bin/bash
set -e

# Always work from project root (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"

POSTPROCESS_SCRIPT="evaluation/postprocess.py"
INPUT_DIR="vqa_predictions_val"
GT_ROOT="data/WTS"
# Relative GT dirs (adjust if different layout in repo)
GT_DIRS="$GT_ROOT/vqa/val $GT_ROOT/BDD_anno/vqa/val"

# Directory to collect all converted.json files
OUTPUT_DIR="vqa_results_val"
mkdir -p "$OUTPUT_DIR"

# Experiment paths (relative)
EXPERIMENT_PATHS=(
  "$INPUT_DIR/mid_bdd"
  "$INPUT_DIR/mid_all_e2"
  "$INPUT_DIR/mid_all_e2_5"
  "$INPUT_DIR/mid_all_e3"
  "$INPUT_DIR/mid_all_group_qas"
  "$INPUT_DIR/even_bdd"
  "$INPUT_DIR/even_wts"
)

for EXP_PATH in "${EXPERIMENT_PATHS[@]}"; do
  EXP_NAME=$(basename "$EXP_PATH")
  OUTPUT_FILE="$OUTPUT_DIR/${EXP_NAME}.json"

  echo "Processing $EXP_NAME → $OUTPUT_FILE"

  python "$POSTPROCESS_SCRIPT" \
    --base_path "$EXP_PATH" \
    --num_ranks 8 \
    --gt_dirs $GT_DIRS \
    --converted_json "$OUTPUT_FILE" \
    --merge --trim --convert

  echo "Done: $OUTPUT_FILE"
  echo "----------------------------------------"
done

echo "All converted files saved in $OUTPUT_DIR"
