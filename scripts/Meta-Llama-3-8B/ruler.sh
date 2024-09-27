#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

export MASTER_PORT=$(echo $METIS_WORKER_0_PORT | cut -d',' -f1)

LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=3
VANILLA=False

MODEL_NAME_OR_PATH=/data1/HF-Models/meta-llama/Meta-Llama-3-8B
echo "Finetune from: ${MODEL_NAME_OR_PATH}"
MODEL=${MODEL_NAME_OR_PATH##*/}

TEMPLATE=custom
echo "Finetune data template: ${TEMPLATE}"

DATA_PATH=../datasets/ruler_training_dataset.jsonl
echo "Finetune data path: ${DATA_PATH}"

MODEL_MAX_LENGTH=2048
echo "Model max length: ${MODEL_MAX_LENGTH}"

BATCH_SIZE=4
echo "Per device train batch size: ${BATCH_SIZE}"

GRAD_ACCUM=8
echo "Gradient accumulation steps: ${GRAD_ACCUM}"

OUTPUT_DIR="../outputs/checkpoints/ruler_${MODEL}_bs_${BATCH_SIZE}_ga_${GRAD_ACCUM}_lr_${LEARNING_RATE}_eps_${NUM_TRAIN_EPOCHS}"
LOG_DIR=../logs

deepspeed finetuning/finetune.py \
    --vanilla $VANILLA \
    --deepspeed ../configs/ds_config_zero3.json \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --template $TEMPLATE\
    --model_max_length $MODEL_MAX_LENGTH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.05 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --evaluation_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    2>&1 | tee ${LOG_DIR}/output_ruler_${MODEL}.log