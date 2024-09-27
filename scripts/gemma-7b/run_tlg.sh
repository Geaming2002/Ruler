python exp/run_exp.py\
    --dataset_path ../datasets/tlg_dataset.jsonl\
    --gpus 1\
    --template custom\
    --model_name_or_path ../outputs/checkpoints/ruler_gemma-7b_bs_4_ga_8_lr_2e-5_eps_3/checkpoint-2841\
    --output_path ../outputs/tlg/tlg_ot_ruler_gemma-7b.jsonl

python exp/run_exp.py\
    --dataset_path ../datasets/tlg_dataset.jsonl\
    --gpus 1\
    --template custom\
    --model_name_or_path ../outputs/checkpoints/vanilla_gemma-7b_bs_4_ga_8_lr_2e-5_eps_3/checkpoint-2841\
    --output_path ../outputs/tlg/tlg_ot_vanilla_gemma-7b.jsonl
