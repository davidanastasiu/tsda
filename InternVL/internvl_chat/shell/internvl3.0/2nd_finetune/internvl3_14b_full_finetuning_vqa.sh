set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
# Calculate accumulation steps safely
GRADIENT_ACC=$((BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * GPUS)))



export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch


OUTPUT_DIR='internvl3_checkpoints/mid_frames_vqa'           # Change this to your desired output directory

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "TSDA/agent/base_model/InternVL3-14B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "TSDA/agent/InternVL/internvl_chat/shell/data/mid_frames_vqa.json" \
 --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 4 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 143 \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8024 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage2_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/mid_frames_training_log.txt"    # Training logs saved in current directory