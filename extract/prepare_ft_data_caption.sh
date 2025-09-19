#!/bin/bash

# Change to the project's root directory (one level up from this script)
cd "$(dirname "$0")/.."

# Path Configuration :

# Base input and output directories
DATA_DIR="data"
OUTPUT_DIR="Internvl3_ft_data_caption"

mkdir -p "$OUTPUT_DIR"
# Input files and directories
WTS_MID_FRAMES_DIR="$DATA_DIR/WTS_mid_frames"
BDD_MID_FRAMES_DIR="$DATA_DIR/BDD_mid_frames"
WTS_EVENLY_FRAMES_DIR="$DATA_DIR/WTS_even_frames"
BDD_EVENLY_FRAMES_DIR="$DATA_DIR/BDD_even_frames"
VIEW_CSV="$DATA_DIR/WTS/view_used_as_main_reference_for_multiview_scenario.csv"

# Output directories
OUT_MID_FACTS_DIR="$OUTPUT_DIR/mid_facts"
OUT_MID_PED_QA_DIR="$OUTPUT_DIR/mid_ped_QA"
OUT_MID_VEH_QA_DIR="$OUTPUT_DIR/mid_veh_QA"
OUT_EVEN_QA_DIR="$OUTPUT_DIR/even_QA"

mkdir -p "$OUT_MID_FACTS_DIR" "$OUT_MID_PED_QA_DIR" "$OUT_MID_VEH_QA_DIR" "$OUT_EVEN_QA_DIR"
# --- Data Preparation ---

# echo "--- Preparing Mid-frame Caption Facts ---"

python -m extract.prepare_data_cap_facts \
    --wts-json "$WTS_MID_FRAMES_DIR/wts_train_frames.json" \
    --bdd-json "$BDD_MID_FRAMES_DIR/bdd_train_frames.json" \
    --view-csv "$VIEW_CSV" \
    --out-jsonl "$OUT_MID_FACTS_DIR/mid_cap_facts_train.jsonl" 

python -m extract.prepare_data_cap_facts \
    --wts-json "$WTS_MID_FRAMES_DIR/wts_val_frames.json" \
    --bdd-json "$BDD_MID_FRAMES_DIR/bdd_val_frames.json" \
    --view-csv "$VIEW_CSV" \
    --out-jsonl "$OUT_MID_FACTS_DIR/mid_cap_facts_val.jsonl" 

echo "--- Preparing Mid-frame Pedestrian and Vehicle QA ---"

python -m extract.prepare_data_ped_veh_QA \
    --wts-json "$WTS_MID_FRAMES_DIR/wts_train_frames.json" \
    --bdd-json "$BDD_MID_FRAMES_DIR/bdd_train_frames.json" \
    --view-csv "$VIEW_CSV" \
    --out-ped-qa-jsonl "$OUT_MID_PED_QA_DIR/mid_ped_qa_train.jsonl" \
    --out-veh-qa-jsonl "$OUT_MID_VEH_QA_DIR/mid_veh_qa_train.jsonl"

python -m extract.prepare_data_ped_veh_QA \
    --wts-json "$WTS_MID_FRAMES_DIR/wts_val_frames.json" \
    --bdd-json "$BDD_MID_FRAMES_DIR/bdd_val_frames.json" \
    --view-csv "$VIEW_CSV" \
    --out-ped-qa-jsonl "$OUT_MID_PED_QA_DIR/mid_ped_qa_val.jsonl" \
    --out-veh-qa-jsonl "$OUT_MID_VEH_QA_DIR/mid_veh_qa_val.jsonl"


echo "--- Preparing Evenly-spaced frame Caption QA ---"

python -m extract.prepare_data_even_cap_QA \
    --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_train_frames.json" \
    --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_train_frames.json" \
    --out-jsonl "$OUT_EVEN_QA_DIR/even_cap_qa_train.jsonl"

python -m extract.prepare_data_even_cap_QA \
    --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_val_frames.json" \
    --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_val_frames.json" \
    --out-jsonl "$OUT_EVEN_QA_DIR/even_cap_qa_val.jsonl"

echo "Data processing finished for all Caption Agents."