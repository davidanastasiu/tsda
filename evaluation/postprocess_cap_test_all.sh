#!/usr/bin/env bash


OUT_ROOT="cap_results_test"
INPUT_ROOT="cap_predictions_test"
# Final outputs (separate files)
FINAL_PED_VEH="$OUT_ROOT/ped_veh_cap.json"
FINAL_EVEN="$OUT_ROOT/even_qa.json"
FINAL_MID="$OUT_ROOT/mid_facts.json"


PED_QA_DIRS="$INPUT_ROOT/ped_qa"
VEH_QA_DIRS="$INPUT_ROOT/veh_qa"
EVEN_QA_DIRS="$INPUT_ROOT/even_qa"
NEW_CAPTIONS_DIRS="$INPUT_ROOT/mid_facts"

mkdir -p "$OUT_ROOT"

# Initialize empty JSONs
echo "{}" > "$FINAL_PED_VEH"
echo "{}" > "$FINAL_EVEN"
echo "{}" > "$FINAL_MID"

# Run ped and veh (merged into same file)
python postprocess_cap_agents.py --model_dirs $PED_QA_DIRS --out_dir "$OUT_ROOT" \
  --mode ped_qa_caption --final_output "$FINAL_PED_VEH"

python postprocess_cap_agents.py --model_dirs $VEH_QA_DIRS --out_dir "$OUT_ROOT" \
  --mode veh_qa_caption --final_output "$FINAL_PED_VEH"

# Run even
python postprocess_cap_agents.py --model_dirs $EVEN_QA_DIRS --out_dir "$OUT_ROOT" \
  --mode even_qa_caption --final_output "$FINAL_EVEN"

# Run mid-facts
python postprocess_cap_agents.py --model_dirs $NEW_CAPTIONS_DIRS --out_dir "$OUT_ROOT" \
  --mode mid_facts --final_output "$FINAL_MID"

echo "[DONE] Outputs:"
echo "  Ped+Veh : $FINAL_PED_VEH"
echo "  Even    : $FINAL_EVEN"
echo "  Mid     : $FINAL_MID"
