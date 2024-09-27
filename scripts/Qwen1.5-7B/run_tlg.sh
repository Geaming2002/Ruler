python exp/run_exp.py\
    --dataset_path ../datasets/tlg_dataset.jsonl\
    --gpus 1\
    --template custom\
    --model_name_or_path ../outputs/checkpoints/ruler_Qwen1.5-7B_bs_4_ga_8_lr_2e-5_eps_3/checkpoint-2841\
    --output_path ../outputs/tlg/tlg_ot_ruler_Qwen1.5-7B.jsonl

python exp/run_exp.py\
    --dataset_path ../datasets/tlg_dataset.jsonl\
    --gpus 1\
    --template custom\
    --model_name_or_path ../outputs/checkpoints/vanilla_Qwen1.5-7B_bs_4_ga_8_lr_2e-5_eps_3/checkpoint-2841\
    --output_path ../outputs/tlg/tlg_ot_vanilla_Qwen1.5-7B.jsonl
