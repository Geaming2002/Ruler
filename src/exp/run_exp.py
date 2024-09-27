import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_jsonl, save_jsonl
from utils.templates import TemplatesMapping


def main(args):
    # raw data load
    df = load_jsonl(args.dataset_path)
    # load tokenizer and llm
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.gpus,
    )
    template = TemplatesMapping[args.template]
    if args.template == "default" in args.dataset_path:
        terminators = TemplatesMapping["default"].get_stop_tokens(
            args.model_name_or_path
        )
    elif "self_generated_mlt.jsonl" in args.dataset_path:
        terminators = ["<|end_of_text|>", "<|eot_id|>"]
    else:
        terminators = template.STOP_TOKENS
    print(f"> STOP_TOKENS:{terminators}")
    terminators = tokenizer.convert_tokens_to_ids(terminators)
    skip_sepcial_tokens = False if "self_generated_mlt.jsonl" in args.dataset_path else True
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2048,
        stop_token_ids=terminators,
        skip_special_tokens=skip_sepcial_tokens,
    )
    for idx in tqdm(range(len(df))):
        instruction = df[idx]["Instruction"]
        targetlength = df[idx]["TargetLength"] if "TargetLength" in df[idx] else ""
        if args.template == "default":
            prompts = [
                template.apply_template_for_generation(
                    instruction, targetlength, tokenizer
                )
            ]
        elif args.template == "custom":
            if "vanilla" in args.model_name_or_path:
                prompts = [
                    template.apply_template_for_generation_vanilla(instruction, targetlength)
                ]
            else:
                prompts = [
                    template.apply_template_for_generation(instruction, targetlength)
                ]
        else:
            prompts = [
                template.apply_template_for_generation(instruction, targetlength)
            ]
        df[idx]["prompt"] = prompts[0]
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            df[idx]["output"] = generated_text
        if idx % 100 == 0:
            save_jsonl(args.output_path, df)
    # save to output_path
    save_jsonl(args.output_path, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--template", type=str, default="default")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
