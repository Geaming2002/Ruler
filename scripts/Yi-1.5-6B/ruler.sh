#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

find_free_port() {
    while :
    do
        PORT=$(( ( RANDOM % 64512 ) + 1024 ))
        (echo >/dev/tcp/localhost/$PORT) >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo $PORT
            return
        fi
    done
}

export MASTER_PORT=$(find_free_port)

LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=3
VANILLA=False

MODEL_NAME_OR_PATH=/data1/HF-Models/01-ai/Yi-1.5-6B
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

deepspeed --master_port=$MASTER_PORT finetuning/finetune.py \
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