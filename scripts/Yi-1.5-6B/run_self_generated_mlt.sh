python exp/run_exp.py\
    --dataset_path ../datasets/self_generated_mlt.jsonl\
    --gpus 1\
    --template custom\
    --model_name_or_path ../outputs/checkpoints/ruler_Yi-1.5-6B_bs_4_ga_8_lr_2e-5_eps_3/checkpoint-2841\
    --output_path ../outputs/self_generated_mlt/sgm_ruler_Yi-1.5-6B.jsonl
