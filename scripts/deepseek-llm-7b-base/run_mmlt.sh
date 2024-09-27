python exp/run_exp.py\
    --dataset_path ../datasets/multi_mlt.jsonl\
    --gpus 1\
    --template custom\
    --model_name_or_path ../outputs/checkpoints/ruler_deepseek-llm-7b-base_bs_4_ga_8_lr_2e-5_eps_3/checkpoint-2841\
    --output_path ../outputs/multi_mlt/mmlt_ruler_deepseek-llm-7b-base.jsonl
