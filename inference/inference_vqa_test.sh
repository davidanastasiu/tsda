#!/bin/bash

# Navigate to the project's root directory.
# Assuming the script script is located in a subdirectory like '.../TSDA/inference/'.
cd "$(dirname "$0")/.."

# Path Configuration :

# Base directories
INPUT_DATA_DIR="InternVL3_ft_data_vqa"
OUTPUT_PRED_DIR="vqa_predictions_test"
MODEL_BASE_DIR="InternVL/internvl_chat/work_dirs/internvl_chat_v3"
mkdir -p "$OUTPUT_PRED_DIR"
# Specific model checkpoint paths
MODEL_MID_E2="$MODEL_BASE_DIR/mid_all_vqa/checkpoint-572"
MODEL_MID_E2_5="$MODEL_BASE_DIR/mid_all_vqa/checkpoint-715"
MODEL_MID_E3="$MODEL_BASE_DIR/mid_all_vqa/checkpoint-855"
MODEL_MID_BDD_ONLY="$MODEL_BASE_DIR/mid_bdd_vqa/checkpoint-795"
MODEL_MID_GROUPQA="$MODEL_BASE_DIR/mid_groupQAs/checkpoint-834"
MODEL_EVEN_BDD_ONLY="$MODEL_BASE_DIR/even_bdd_vqa/checkpoint-518"
MODEL_EVEN_WTS_ONLY="$MODEL_BASE_DIR/even_wts_vqa/checkpoint-24"

mkdir -p "$OUTPUT_PRED_DIR"/{even_bdd,mid_all_e2_5,mid_all_e3,mid_all_e2,mid_bdd,mid_all_groupQAs,even_wts}
# --- Run Inference on all VQA Agents ---

## Mid Frames model on both dataset at epoch 2
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_all/mid_test_all.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_all_e2/mid_all_e2_test.jsonl" \
    --model-path "$MODEL_MID_E2" \
    --img-size 448 \
    --max-total-tokens 6144

# Mid Frames model on both dataset at epoch 2.5
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_all/mid_test_all.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_all_e2_5/mid_all_e2_5_test.jsonl" \
    --model-path "$MODEL_MID_E2_5" \
    --img-size 448 \
    --max-total-tokens 8024

## Mid Frames model on both dataset at epoch 3
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_all/mid_test_all.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_all_e3/mid_all_e3_test.jsonl" \
    --model-path "$MODEL_MID_E3" \
    --img-size 448 \
    --max-total-tokens 8024

## Mid Frames model on BDD only dataset
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_bdd/mid_test_bdd.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_bdd/mid_bdd_test.jsonl" \
    --model-path "$MODEL_MID_BDD_ONLY" \
    --img-size 448 \
    --max-total-tokens 6144

## Mid Frames model on selected group of QAs
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_groupQAs/mid_test_groupQAs.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_all_group_qas/mid_test_groupQAs.jsonl" \
    --model-path "$MODEL_MID_GROUPQA" \
    --img-size 336 \
    --max-total-tokens 6144

## Evenly Spaced Frames model on BDD only dataset
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/even_frames_bdd/evenly_test_bdd.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/even_bdd/even_bdd_test.jsonl" \
    --model-path "$MODEL_EVEN_BDD_ONLY" \
    --img-size 336 \
    --max-total-tokens 6144

## Evenly Spaced Frames model on WTS only dataset
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/even_frames_wts/evenly_test_wts.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/even_wts/even_wts_test.jsonl" \
    --model-path "$MODEL_EVEN_WTS_ONLY" \
    --img-size 336 \
    --max-total-tokens 6144

echo "Inference script finished."