import random
import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer
from dataclasses import dataclass, field

from dataset import load_custom_dataset, DataCollatorForSupervisedDataset
from utils.config import MetaLengthToken
from utils.templates import TemplatesMapping


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="",
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    template: str = field(default="", metadata={"help": "The template used to train"})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    vanilla: bool = field(
        default=False,
        metadata={"help": "Vanilla finetuning or Ruler finetuning, defaulty is False."},
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={"help": "gradient checkpointing kwargs"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args.vanilla)
    if training_args.local_rank == 0:
        print("=" * 100)
        print(training_args)

    if training_args.local_rank == 0:
        print("> Loading tokenizer from {}".format(model_args.model_name_or_path))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        truncation_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    template = TemplatesMapping[model_args.template]
    # add special tokens
    if training_args.vanilla:
        special_tokens = {"additional_special_tokens": [t for t in template.SPECIAL_TOKENS]}
    elif model_args.template == 'custom':
        special_tokens = {"additional_special_tokens": [t for t in template.SPECIAL_TOKENS + [m[0]for m in MetaLengthToken]]}
    else:
        special_tokens = {"additional_special_tokens": [t[0] for t in MetaLengthToken]}
    print(f"> New special tokens: {special_tokens}")
    tokenizer.add_special_tokens(special_tokens)
    for st in special_tokens["additional_special_tokens"]:
        print(f"{st}:{tokenizer.convert_tokens_to_ids(st)}")

    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    if training_args.local_rank == 0:
        print("> PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
        print("> BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
        print("> EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("> Loading model from {}".format(model_args.model_name_or_path))

    if "glm-4" in model_args.model_name_or_path:  # glm-4 not support flash attention 2s
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model.resize_token_embeddings(len(tokenizer))
    train_dataset = load_custom_dataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        template=template,
        vanilla=training_args.vanilla,
    )
    
    if training_args.local_rank == 0:
        print("> Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print("=" * 100)
            print(
                f"Sample {index} of the training set:\n{tokenizer.decode(list(train_dataset[index]['input_ids']))}"
            )
            print(f"{train_dataset[index]['input_ids']}")
            print(f"{train_dataset[index]['labels']}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
