import random
import argparse
from utils import load_jsonl, save_jsonl
from utils.config import TARGET_LENGTH


def main(args):
    # random seed
    random.seed(args.random_seed)
    # raw data load
    df = load_jsonl(args.dataset_path)
    # random sample
    random.shuffle(df)
    df = df[: args.num]
    # add target length
    data = []
    target_lengths = [random.choice(TARGET_LENGTH) for _ in range(args.num)]
    for idx in range(len(df)):
        d = {}
        d['id'] = idx
        d["Instruction"] = df[idx]["conversations"][0]["value"]
        d["TargetLength"] = target_lengths[idx]
        data.append(d)
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
