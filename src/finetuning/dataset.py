import torch
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from utils.config import MetaLengthToken
from typing import Dict, Sequence

IGNORE_INDEX = -100


class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def preprocess_template(instruction, mlt, output, tokenizer, template,vanilla):
    if vanilla:
        mlt = ''
    prompts = template.apply_template(instruction, mlt, output)
    input_ids = tokenizer.encode(
        prompts, truncation=True, max_length=tokenizer.model_max_length
    )  # truncation
    split_token_idx = None
    for i in MetaLengthToken:
        i_id = tokenizer.convert_tokens_to_ids(i[0])
        if i_id in input_ids:
            split_token_idx = input_ids.index(i_id)
    if split_token_idx is None:
        labels = [IGNORE_INDEX for _ in range(len(input_ids))]
    else:
        labels = [
            input_ids[i] if i >= split_token_idx else IGNORE_INDEX
            for i in range(len(input_ids))
        ]
    # vanilla
    if vanilla:
        instruction_prompts = template.apply_template_for_instruction(instruction)
        instruction_ids = tokenizer.encode(
        instruction_prompts, truncation=True, max_length=tokenizer.model_max_length
        )
        labels = [
            input_ids[i] if i >= len(instruction_ids) else IGNORE_INDEX
            for i in range(len(input_ids))
        ]

    return input_ids, labels


def preprocess(examples, tokenizer, template, vanilla):
    processed_input_ids, processed_labels = [], []

    instructions, mlts, outputs = (
        examples["Instruction"],
        examples["mlt"],
        examples["output"],
    )
    for instruction, mlt, output in zip(instructions, mlts, outputs):
        input_ids, labels = preprocess_template(
            instruction, mlt, output, tokenizer, template, vanilla
        )

        processed_input_ids.append(input_ids)
        processed_labels.append(labels)

    return {"input_ids": processed_input_ids, "labels": processed_labels}


def load_custom_dataset(tokenizer: PreTrainedTokenizer, data_path: str, template, vanilla):
    train_datasets = load_dataset("json", data_files=data_path, split="train")

    train_dataset = train_datasets.map(
        preprocess,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=train_datasets.column_names,
        keep_in_memory=True,
        load_from_cache_file=False,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer, "template": template, "vanilla":vanilla},
    )

    torch.distributed.barrier()

    return train_dataset
