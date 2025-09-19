#!/bin/bash

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

INPUT_BASE=Internvl3_ft_data_caption
OUTPUT_BASE=ma_cap_val
GT_ROOT="data" # Path to data directory root

mkdir -p ${OUTPUT_BASE}
# Run commands with relative paths

python multi_agent_selection/convert_to_descriptor_format.py \
    --best multi_agent_selection/multi_agent_results_vqa.json \
    --gt_dirs "$GT_ROOT/vqa/val" \
              "$GT_ROOT/BDD_anno/vqa/val" \
    --output multi_agent_selection/ma_descriptor_res.jsonl

python multi_agent_selection/prepare_val_inf_cap.py \
    --input-val-jsonl ${INPUT_BASE}/even_QA/even_cap_qa_val.jsonl \
    --best-answers-json multi_agent_selection/ma_descriptor_res.jsonl \
    --output-jsonl ${OUTPUT_BASE}/even_qa_val_inf.jsonl \
    --format even_qa

python multi_agent_selection/prepare_val_inf_cap.py \
    --input-val-jsonl ${INPUT_BASE}/mid_ped_QA/mid_ped_qa_val.jsonl \
    --best-answers-json multi_agent_selection/ma_descriptor_res.jsonl \
    --output-jsonl ${OUTPUT_BASE}/mid_ped_qa_val_inf.jsonl \
    --format selective_qa

python multi_agent_selection/prepare_val_inf_cap.py \
    --input-val-jsonl ${INPUT_BASE}/mid_veh_QA/mid_veh_qa_val.jsonl \
    --best-answers-json multi_agent_selection/ma_descriptor_res.jsonl \
    --output-jsonl ${OUTPUT_BASE}/mid_veh_qa_val_inf.jsonl \
    --format selective_qa

python multi_agent_selection/prepare_val_inf_cap.py \
    --input-val-jsonl ${INPUT_BASE}/mid_facts/mid_cap_facts_val.jsonl \
    --best-answers-json multi_agent_selection/ma_descriptor_res.jsonl \
    --output-jsonl ${OUTPUT_BASE}/mid_cap_facts_val_inf.jsonl \
    --format caption_facts
