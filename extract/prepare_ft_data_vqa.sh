#!/bin/bash

# Change to the project's root directory (one level up from this script)
cd "$(dirname "$0")/.."

# Path Configuration

DATA_DIR="data"
OUTPUT_DIR="InternVL3_ft_data_vqa"

VIEW_CSV="$DATA_DIR/WTS/view_used_as_main_reference_for_multiview_scenario.csv"

# Input directories for processed frame data
WTS_MID_FRAMES_DIR="WTS_mid_frames"
BDD_MID_FRAMES_DIR="BDD_mid_frames"
WTS_EVENLY_FRAMES_DIR="WTS_even_frames"
BDD_EVENLY_FRAMES_DIR="BDD_even_frames"

mkdir -p "$OUTPUT_DIR"
# Output directories for different processing runs
OUT_ALL_MID_DIR="$OUTPUT_DIR/mid_frames_all"
OUT_WTS_MID_DIR="$OUTPUT_DIR/mid_frames_wts"
OUT_BDD_MID_DIR="$OUTPUT_DIR/mid_frames_bdd"
OUT_ALL_MID_GROUPQAs_DIR="$OUTPUT_DIR/mid_frames_groupQAs"
OUT_BDD_EVENLY_DIR="$OUTPUT_DIR/even_frames_bdd"
OUT_WTS_EVENLY_DIR="$OUTPUT_DIR/even_frames_wts"

# Create output directories if they don't exist
mkdir -p "$OUT_ALL_MID_DIR" "$OUT_WTS_MID_DIR" "$OUT_BDD_MID_DIR" "$OUT_BDD_EVENLY_DIR" "$OUT_WTS_EVENLY_DIR" "$OUT_ALL_MID_GROUPQAs_DIR"


# --- Data Preparation ---

Data for agents using Mid k-spaced frames strategy

echo "Processing All Mid frames Train Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_MID_FRAMES_DIR/wts_train_frames.json" \
  --bdd-json "$BDD_MID_FRAMES_DIR/bdd_train_frames.json" \
  --view-csv "$VIEW_CSV" \
  --json-out "$OUT_ALL_MID_DIR/mid_train_all.json" \
  --jsonl-out "$OUT_ALL_MID_DIR/mid_train_all.jsonl" \
  --pretty-json-out "$OUT_ALL_MID_DIR/mid_train_all_pr.json" \
  --log-dir "$OUT_ALL_MID_DIR/logs/train" \
  --dataset-type all

echo "Processing All Mid frames Val Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_MID_FRAMES_DIR/wts_val_frames.json" \
  --bdd-json "$BDD_MID_FRAMES_DIR/bdd_val_frames.json" \
  --view-csv "$VIEW_CSV" \
  --json-out "$OUT_ALL_MID_DIR/mid_val_all.json" \
  --jsonl-out "$OUT_ALL_MID_DIR/mid_val_all.jsonl" \
  --pretty-json-out "$OUT_ALL_MID_DIR/mid_val_all_pr.json" \
  --log-dir "$OUT_ALL_MID_DIR/logs/val" \
  --dataset-type all

echo "Processing All Mid frames Test Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_MID_FRAMES_DIR/wts_test_frames.json" \
  --bdd-json "$BDD_MID_FRAMES_DIR/bdd_test_frames.json" \
  --view-csv "$VIEW_CSV" \
  --json-out "$OUT_ALL_MID_DIR/mid_test_all.json" \
  --jsonl-out "$OUT_ALL_MID_DIR/mid_test_all.jsonl" \
  --pretty-json-out "$OUT_ALL_MID_DIR/mid_test_all_pr.json" \
  --log-dir "$OUT_ALL_MID_DIR/logs/test" \
  --mode test \
  --dataset-type all


echo "Processing BDD Mid frames Train Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_MID_FRAMES_DIR/wts_train_frames.json" \
  --bdd-json "$BDD_MID_FRAMES_DIR/bdd_train_frames.json" \
  --view-csv "$VIEW_CSV" \
  --json-out "$OUT_BDD_MID_DIR/mid_train_bdd.json" \
  --jsonl-out "$OUT_BDD_MID_DIR/mid_train_bdd.jsonl" \
  --pretty-json-out "$OUT_BDD_MID_DIR/mid_train_bdd_pr.json" \
  --log-dir "$OUT_BDD_MID_DIR/logs/train" \
  --dataset-type bdd

echo "Processing BDD Mid frames Val Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_MID_FRAMES_DIR/wts_val_frames.json" \
  --bdd-json "$BDD_MID_FRAMES_DIR/bdd_val_frames.json" \
  --view-csv "$VIEW_CSV" \
  --json-out "$OUT_BDD_MID_DIR/mid_val_bdd.json" \
  --jsonl-out "$OUT_BDD_MID_DIR/mid_val_bdd.jsonl" \
  --pretty-json-out "$OUT_BDD_MID_DIR/mid_val_bdd_pr.json" \
  --log-dir "$OUT_BDD_MID_DIR/logs/val" \
  --dataset-type bdd

echo "Processing BDD Mid frames Test Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_MID_FRAMES_DIR/wts_test_frames.json" \
  --bdd-json "$BDD_MID_FRAMES_DIR/bdd_test_frames.json" \
  --view-csv "$VIEW_CSV" \
  --json-out "$OUT_BDD_MID_DIR/mid_test_bdd.json" \
  --jsonl-out "$OUT_BDD_MID_DIR/mid_test_bdd.jsonl" \
  --pretty-json-out "$OUT_BDD_MID_DIR/mid_test_bdd_pr.json" \
  --log-dir "$OUT_BDD_MID_DIR/logs/test" \
  --mode test \
  --dataset-type bdd

echo "Processing Mid Frames Group QAs Train Data"

python prepare_data_groupQAs.py "$OUT_ALL_MID_DIR/mid_train_all.jsonl" \
      "$OUT_ALL_MID_GROUPQAs_DIR/mid_train_groupQAs.jsonl"

echo "Processing Mid Frames Group QAs Val Data"

python prepare_data_groupQAs.py "$OUT_ALL_MID_DIR/mid_val_all.jsonl" \
      "$OUT_ALL_MID_GROUPQAs_DIR/mid_val_groupQAs.jsonl"

echo "Processing Mid Frames Group QAs Test Data"

python -m extract.prepare_data_groupQAs "$OUT_ALL_MID_DIR/mid_test_all.jsonl" \
      "$OUT_ALL_MID_GROUPQAs_DIR/mid_test_groupQAs.jsonl"

# Data for agents using Evenly spaced frames strategy (Dataset split)

echo "Processing BDD evenly frames Train Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_train_frames.json" \
  --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_train_frames.json" \
  --view-csv "$VIEW_CSV" \
  --jsonl-out "$OUT_BDD_EVENLY_DIR/evenly_train_bdd.jsonl" \
  --pretty-json-out "$OUT_BDD_EVENLY_DIR/even_train_bdd_pr.json" \
  --log-dir "$OUT_BDD_EVENLY_DIR/logs/train" \
  --dataset-type bdd \
  --processing-mode evenly

echo "Processing BDD evenly frames val Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_val_frames.json" \
  --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_val_frames.json" \
  --view-csv "$VIEW_CSV" \
  --jsonl-out "$OUT_BDD_EVENLY_DIR/evenly_val_bdd.jsonl" \
  --pretty-json-out "$OUT_BDD_EVENLY_DIR/even_val_bdd_pr.json" \
  --log-dir "$OUT_BDD_EVENLY_DIR/logs/val" \
  --dataset-type bdd \
  --processing-mode evenly

echo "Processing BDD evenly frames Test Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_test_frames.json" \
  --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_test_frames.json" \
  --view-csv "$VIEW_CSV" \
  --jsonl-out "$OUT_BDD_EVENLY_DIR/evenly_test_bdd.jsonl" \
  --pretty-json-out "$OUT_BDD_EVENLY_DIR/even_test_bdd_pr.json" \
  --log-dir "$OUT_BDD_EVENLY_DIR/logs/test" \
  --dataset-type bdd \
  --is-test-set \
  --processing-mode evenly

echo "Processing WTS evenly frames Train Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_train_frames.json" \
  --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_train_frames.json" \
  --view-csv "$VIEW_CSV" \
  --jsonl-out "$OUT_WTS_EVENLY_DIR/even_train_wts.jsonl" \
  --pretty-json-out "$OUT_WTS_EVENLY_DIR/even_train_wts_pr.json" \
  --log-dir "$OUT_WTS_EVENLY_DIR/logs/train" \
  --dataset-type wts \
  --processing-mode evenly

echo "Processing WTS evenly frames val Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_val_frames.json" \
  --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_val_frames.json" \
  --view-csv "$VIEW_CSV" \
  --jsonl-out "$OUT_WTS_EVENLY_DIR/even_val_wts.jsonl" \
  --pretty-json-out "$OUT_WTS_EVENLY_DIR/even_val_wts_pr.json" \
  --log-dir "$OUT_WTS_EVENLY_DIR/logs/val" \
  --dataset-type wts \
  --processing-mode evenly

echo "Processing WTS evenly frames test Data"
python -m extract.prepare_data_vqa \
  --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_test_frames.json" \
  --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_test_frames.json" \
  --view-csv "$VIEW_CSV" \
  --jsonl-out "$OUT_WTS_EVENLY_DIR/even_test_wts.jsonl" \
  --pretty-json-out "$OUT_WTS_EVENLY_DIR/even_test_wts_pr.json" \
  --log-dir "$OUT_WTS_EVENLY_DIR/logs/test" \
  --dataset-type wts \
  --is-test-set \
  --processing-mode evenly

echo "Data processing finished for all VQA Agents."