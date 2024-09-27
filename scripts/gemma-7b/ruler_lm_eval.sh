set -ex
export CUDA_VISIBLE_DEVICES=2
export NUMEXPR_MAX_THREADS=128

MODEL=vllm
MODEL_NAME=ruler_gemma-7b
MODEL_NAME_OR_PATH=/home/lijiaming/workspace/Seed/Seed-Ruler/outputs/checkpoints/ruler_gemma-7b_bs_4_ga_8_lr_2e-5_eps_3/checkpoint-2841
OUTPUT_PATH=../outputs/other_tasks/${MODEL_NAME}
TOKENIZER_MODE=auto
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=0.8

mkdir -p $OUTPUT_PATH

# lm_eval --model $MODEL \
#     --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
#     --tasks leaderboard \
#     --device cuda \
#     --output_path ${OUTPUT_PATH}/${MODEL}_eval_leaderboard \
#     --batch_size 1 \
#     --write_out \
#     2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_leaderboard.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks ai2_arc \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_ai2_arc \
    --batch_size 1 \
    --num_fewshot 25 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_ai2_arc.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks hellaswag \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_hellaswag \
    --batch_size 1 \
    --num_fewshot 10 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_hellaswag.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks truthfulqa \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_truthfulqa \
    --batch_size 1 \
    --num_fewshot 0 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_truthfulqa.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks mmlu \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_mmlu \
    --batch_size 1 \
    --num_fewshot 5 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_mmlu.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks winogrande \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_winogrande \
    --batch_size 1 \
    --num_fewshot 5 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_winogrande.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks gsm8k \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_gsm8k \
    --batch_size 1 \
    --num_fewshot 5 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_gsm8k.log