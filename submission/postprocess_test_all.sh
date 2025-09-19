# !/bin/bash

cd "$(dirname "$0")/.."

# Path Configuration:
# All paths are relative to the TSDA project root.

DATA_DIR="data"

VQA_JSON="$DATA_DIR/final_test_data/SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json"
POSTPROCESS_SCRIPT="submission/postprocess_test.py"
INPUT_BASE_DIR="vqa_predictions_test"
OUTPUT_DIR="vqa_results_test"


echo "Creating output directory at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "--------------------------------------------"
.
for EXP_PATH in "$INPUT_BASE_DIR"/*/; do
  EXP_PATH=${EXP_PATH%/}
  NAME=$(basename "$EXP_PATH")
  

  OUT_PATH="$OUTPUT_DIR/$NAME"

  echo "🔧 Processing experiment: $NAME"
  
export PYTHONPATH=$(pwd)

  python "$POSTPROCESS_SCRIPT" \
    --base_path "$EXP_PATH" \
    --vqa_json "$VQA_JSON" \
    --output_dir "$OUT_PATH" \
    --num_ranks 8

  echo "Done. Output saved to: $OUT_PATH"
  echo "--------------------------------------------"
done

echo "All individual agent predictions have been processed and saved in $OUTPUT_DIR"
