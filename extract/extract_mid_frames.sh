#!/bin/bash

# Change to the project's root directory (one level up from this script)

cd "$(dirname "$0")/.."

# Path Configuration:

# For general purposes, set it to your local directory
DATA_DIR="data"
PROJECT_ROOT="$(pwd)"
# WTS dataset paths
WTS_ANNO_DIR="$DATA_DIR/WTS/final_annotations_WTS"
WTS_VIDEO_DIR="$DATA_DIR/WTS/videos"

# BDD dataset paths
BDD_VIDEO_DIR="$DATA_DIR/BDD_PC_5k/videos"
BDD_ANNO_DIR="$DATA_DIR/WTS/final_annotations_WTS/BDD_anno"

# Test data paths
TEST_DATA_ROOT="$DATA_DIR/final_test_data"
WTS_TEST_DATA_DIR="$TEST_DATA_ROOT/SubTask1-Caption/WTS_DATASET_PUBLIC_TEST"
BDD_TEST_DATA_DIR="$WTS_TEST_DATA_DIR/external/BDD_PC_5K"
VQA_TEST_JSON="$TEST_DATA_ROOT/SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json"

# Output directories
WTS_OUTPUT_DIR="$PROJECT_ROOT/WTS_mid_frames"
BDD_OUTPUT_DIR="$PROJECT_ROOT/BDD_mid_frames"

mkdir -p "$WTS_OUTPUT_DIR" "$BDD_OUTPUT_DIR"
# --- Mid Frame Extraction ---

echo "Starting Frame Extraction - WTS Train"
python -m extract.preprocess_mid \
  --dataset WTS \
  --split train \
  --video-dir "$WTS_VIDEO_DIR" \
  --caption-dir "$WTS_ANNO_DIR/caption" \
  --bbox-dir "$WTS_ANNO_DIR/bbox_generated" \
  --gt-dir "$WTS_ANNO_DIR/vqa" \
  --output-dir "$WTS_OUTPUT_DIR" \
  --save

echo "Starting Frame Extraction - WTS Val"
python -m extract.preprocess_mid \
  --dataset WTS \
  --split val \
  --video-dir "$WTS_VIDEO_DIR" \
  --caption-dir "$WTS_ANNO_DIR/caption" \
  --bbox-dir "$WTS_ANNO_DIR/bbox_generated" \
  --gt-dir "$WTS_ANNO_DIR/vqa" \
  --output-dir "$WTS_OUTPUT_DIR" \
  --save

echo "Starting Frame Extraction - WTS Test"
python -m extract.preprocess_mid \
  --dataset WTS \
  --split test \
  --video-dir "$WTS_TEST_DATA_DIR/videos" \
  --caption-dir "$WTS_TEST_DATA_DIR/annotations/caption" \
  --bbox-dir "$WTS_TEST_DATA_DIR/annotations/bbox_generated" \
  --vqa-json "$VQA_TEST_JSON" \
  --output-dir "$WTS_OUTPUT_DIR" \
  --save


echo "Starting Frame Extraction - BDD Train"
python -m extract.preprocess_mid \
  --dataset BDD \
  --split train \
  --video-dir "$BDD_VIDEO_DIR" \
  --caption-dir "$BDD_ANNO_DIR/caption" \
  --bbox-dir "$BDD_ANNO_DIR/bbox_generated" \
  --gt-dir "$BDD_ANNO_DIR/vqa" \
  --output-dir "$BDD_OUTPUT_DIR" \
  --save


echo "Starting Frame Extraction - BDD Val"
python -m extract.preprocess_mid \
  --dataset BDD \
  --split val \
  --video-dir "$BDD_VIDEO_DIR" \
  --caption-dir "$BDD_ANNO_DIR/caption" \
  --bbox-dir "$BDD_ANNO_DIR/bbox_generated" \
  --gt-dir "$BDD_ANNO_DIR/vqa" \
  --output-dir "$BDD_OUTPUT_DIR" \
  --save

echo "Starting Frame Extraction - BDD Test"
python -m extract.preprocess_mid \
  --dataset BDD \
  --split test \
  --video-dir "$BDD_TEST_DATA_DIR/videos" \
  --caption-dir "$BDD_TEST_DATA_DIR/annotations/caption" \
  --bbox-dir "$BDD_TEST_DATA_DIR/annotations/bbox_generated" \
  --vqa-json "$VQA_TEST_JSON" \
  --output-dir "$BDD_OUTPUT_DIR" \
  --save

echo "All frame extraction complete."