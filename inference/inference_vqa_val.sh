#!/bin/bash

# Navigate to the project's root directory.
# Assuming the script script is located in a subdirectory like '.../TSDA/inference/'.
cd "$(dirname "$0")/.."

# Path Configuration :

INPUT_DATA_DIR="InternVL3_ft_data_vqa"
OUTPUT_PRED_DIR="vqa_predictions_val"
MODEL_BASE_DIR="InternVL/internvl_chat/work_dirs/internvl_chat_v3"
mkdir -p "$OUTPUT_PRED_DIR"
# Specific model checkpoint paths
MODEL_MID_E2="$MODEL_BASE_DIR/mid_all_vqa/checkpoint-572"
MODEL_MID_E2_5="$MODEL_BASE_DIR/mid_all_vqa/checkpoint-715"
MODEL_MID_E3="$MODEL_BASE_DIR/mid_all_vqa/checkpoint-855"
MODEL_MID_BDD_ONLY="$MODEL_BASE_DIR/mid_bdd/checkpoint-795"
MODEL_MID_GROUPQA="$MODEL_BASE_DIR/mid_groupQAs/checkpoint-834"
MODEL_EVEN_BDD_ONLY="$MODEL_BASE_DIR/even_bdd/checkpoint-518"
MODEL_EVEN_WTS_ONLY="$MODEL_BASE_DIR/even_wts/checkpoint-24"

mkdir -p "$OUTPUT_PRED_DIR"/{even_bdd,mid_all_e2_5,mid_all_e3,mid_all_e2,mid_bdd,mid_all_group_qas,even_wts}
# --- Run Inference on all VQA Agents ---

## Mid Frames model on both dataset at epoch 2
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_all/mid_val_all.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_e2/mid_e2_val.jsonl" \
    --model-path "$MODEL_MID_E2" \
    --img-size 448 \
    --max-total-tokens 6144

# Mid Frames model on both dataset at epoch 2.5
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_all/mid_val_all.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_e2_5/mid_e2_5_val.jsonl" \
    --model-path "$MODEL_MID_E2_5" \
    --img-size 448 \
    --max-total-tokens 8024

## Mid Frames model on both dataset at epoch 3
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_all/mid_val_all.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_e3/mid_e3_val.jsonl" \
    --model-path "$MODEL_MID_E3" \
    --img-size 448 \
    --max-total-tokens 8024

## Mid Frames model on BDD only dataset
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_bdd/mid_val_bdd.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_bdd/mid_bdd_val.jsonl" \
    --model-path "$MODEL_MID_BDD_ONLY" \
    --img-size 448 \
    --max-total-tokens 6144

## Mid Frames model on selected group of QAs
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/mid_frames_groupQAs/mid_val_groupQAs.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_all_group_qas/mid_groupQAs_val.jsonl" \
    --model-path "$MODEL_MID_GROUPQA" \
    --img-size 336 \
    --max-total-tokens 6144

## Evenly Spaced Frames model on BDD only dataset
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/even_bdd_val/evenly_bdd_val.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/even_bdd/even_bdd_val.jsonl" \
    --model-path "$MODEL_EVEN_BDD_ONLY" \
    --img-size 336 \
    --max-total-tokens 6144

## Evenly Spaced Frames model on WTS only dataset
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_vqa.py \
    --input-jsonl "$INPUT_DATA_DIR/even_wts_val/evenly_wts_val.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/even_wts/even_wts_val.jsonl" \
    --model-path "$MODEL_EVEN_WTS_ONLY" \
    --img-size 336 \
    --max-total-tokens 6144

echo "Inference script finished."