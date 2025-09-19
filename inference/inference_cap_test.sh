cd "$(dirname "$0")/.."

# Path Configuration :
INPUT_DATA_DIR="ma_test_cap_inf"
OUTPUT_PRED_DIR="cap_predictions_test"
MODEL_BASE_DIR="InternVL/internvl_chat/work_dirs/internvl_chat_v3"
mkdir -p "$OUTPUT_PRED_DIR"
# Specific model checkpoint paths
MODEL_EVEN_QA="$MODEL_BASE_DIR/even_cap_qa/checkpoint-556/"
MODEL_PED_QA="$MODEL_BASE_DIR/mid_ped_qa/checkpoint-570"
MODEL_VEH_QA="$MODEL_BASE_DIR/mid_veh_qa/checkpoint-570"
MODEL_MID_FACTS="$MODEL_BASE_DIR/mid_facts_cap/checkpoint-570"

mkdir -p "$OUTPUT_PRED_DIR"/{even_qa,ped_qa,veh_qa,mid_facts}
# --- Run Inference on all Caption Agents ---

## Mid Frames facts caption model
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_caption.py --input-jsonl "$INPUT_DATA_DIR/ma_test_caption_facts.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/mid_facts/mid_facts_test.jsonl" \
    --model-path "$MODEL_MID_FACTS" \
    --model-type frames_facts \
    --max-new-tokens 512 \
    --use-flash-attn

# Mid Frames Veh QA caption model
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_caption.py --input-jsonl "$INPUT_DATA_DIR/ma_test_veh_qa.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/veh_qa/veh_qa_test.jsonl" \
    --model-path "$MODEL_VEH_QA" \
    --max-new-tokens 512 \
    --model-type veh \
    --use-flash-attn

# Mid Frames Ped QA caption model
torchrun --nproc_per_node=8 --nnodes=1 inference/internvl3_inf_caption.py --input-jsonl "$INPUT_DATA_DIR/ma_test_ped_qa.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/ped_qa/ped_qa_test.jsonl" \
    --model-path "$MODEL_PED_QA" \
    --max-new-tokens 512 \
    --model-type ped \
    --use-flash-attn

## Even Frames QA caption model

torchrun --nproc_per_node=7 --nnodes=1 inference/internvl3_inf_caption.py --input-jsonl "$INPUT_DATA_DIR/ma_test_even_qa.jsonl" \
    --output-jsonl "$OUTPUT_PRED_DIR/even_qa/even_qa_test.jsonl" \
    --model-path "$MODEL_EVEN_QA" \
    --max-new-tokens 512 \
    --model-type even_qa \
    --use-flash-attn

echo "Inference script finished."