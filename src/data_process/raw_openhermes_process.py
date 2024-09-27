import argparse
from utils import load_json, save_jsonl


def main(args):
    # raw data load
    df = load_json(args.dataset_path)
    data = []
    for d in df:
        if len(d["conversations"]) == 2:
            data.append(d)
    # save to output_path
    save_jsonl(args.output_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
