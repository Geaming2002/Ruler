import os
import nlp
import random
import argparse
import pandas as pd
from utils import load_jsonl, save_jsonl
from utils.config import TARGET_LENGTH


def main(args):
    df = load_jsonl(args.dataset_path)
    data = []
    id = 0
    for d in df:
        data.append({"id": id, "Instruction": d["turns"][0]["content"]})
        id += 1
    if args.num is not None:
        random.seed(args.random_seed)
        random.shuffle(data)
        data = data[: args.num]
        data = [
            {**d, "TargetLength": tl} for d in data for tl in TARGET_LENGTH
        ]
    # save to output_path
    save_jsonl(args.output_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
