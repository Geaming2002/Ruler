import os
import nlp
import random
import argparse
import pandas as pd
from utils import load_jsonl, save_jsonl
from utils.config import MetaLengthToken, SAMPLE
from utils.count import count_words




def list_files(directory):
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def add_MLT(instruction: str):
    result = None
    word_count = count_words(instruction)
    for mlt in MetaLengthToken:
        if word_count > mlt[1][0] and word_count <= mlt[1][1]:
            result = mlt[0]
    return result


def process_OpenHermes(dataset_path, random_seed, num):
    # set random seed
    random.seed(random_seed)
    df = load_jsonl(dataset_path)
    random.shuffle(df)
    print(f"{'='*10}First data in TLG Dataset{'='*10}")
    print(df[0]["conversations"][0]["value"])
    print(f"{'='*10}Last data in TLG Dataset{'='*10}")
    print(df[num - 1]["conversations"][0]["value"])
    print("=" * 20)
    df = df[num:]  # cut off the FLCG exp dataset
    # sampled data
    sampled_data = {key[0]: [] for key in MetaLengthToken}
    for idx in range(len(df)):
        d = {}
        d["Instruction"] = df[idx]["conversations"][0]["value"]
        d["word_count"] = len(df[idx]["conversations"][1]["value"].split())
        d["output"] = df[idx]["conversations"][1]["value"]
        d["mlt"] = add_MLT(df[idx]["conversations"][1]["value"])
        if d["mlt"] is not None:
            sampled_data[d["mlt"]].append(d)
    return sampled_data


def process_longform(dir_path):
    # sampled data
    sampled_data = {key[0]: [] for key in MetaLengthToken}
    longform_files = list_files(dir_path)
    for file in longform_files:
        df = pd.read_parquet(f"{dir_path}/{file}")
        for idx in range(df.shape[0]):
            d = {}
            d["Instruction"] = df.iloc[idx]["input"]
            d["word_count"] = len(df.iloc[idx]["output"].split())
            d["output"] = df.iloc[idx]["output"]
            d["mlt"] = add_MLT(df.iloc[idx]["output"])
            if d["mlt"] is not None:
                sampled_data[d["mlt"]].append(d)
    return sampled_data


def process_eli5():
    # sampled data
    sampled_data = {key[0]: [] for key in MetaLengthToken}
    eli5 = nlp.load_dataset("eli5")
    files = ["train_eli5", "test_eli5", "validation_eli5"]
    for file in files:
        for data in eli5[file]:
            d = {}
            d["Instruction"] = data["title"]
            answer = ""
            for i in data["answers"]["text"]:
                if len(i.split()) > len(answer.split()):
                    answer = i
            d["word_count"] = len(answer.split())
            d["output"] = answer
            d["mlt"] = add_MLT(answer)
            if d["mlt"] is not None:
                sampled_data[d["mlt"]].append(d)
    return sampled_data


def main(args):
    sampled_data = {key[0]: [] for key in MetaLengthToken}
    # OpenHermes2.5
    openhermes_data = process_OpenHermes(args.dataset_path, args.random_seed, args.num)
    print(f"{'='*10}OpenHermes2.5 dataset{'='*10}")
    for key in openhermes_data:
        random.shuffle(openhermes_data[key])
        data_num = min(len(openhermes_data[key]), SAMPLE[key] - len(sampled_data[key]))
        sampled_data[key] += openhermes_data[key][:data_num]
        print(f"{key}-{len(openhermes_data[key])}-take {data_num}.")
    # Long Form
    longform_data = process_longform(args.longform_dir)
    print(f"{'='*10}LongForm dataset{'='*10}")
    for key in longform_data:
        random.shuffle(longform_data[key])
        data_num = min(len(longform_data[key]), SAMPLE[key] - len(sampled_data[key]))
        sampled_data[key] += longform_data[key][:data_num]
        print(f"{key}-{len(longform_data[key])}-take {data_num}")
    # ELI5
    eli5_data = process_eli5()
    print(f"{'='*10}ELI5 dataset{'='*10}")
    for key in eli5_data:
        random.shuffle(eli5_data[key])
        data_num = min(len(eli5_data[key]), SAMPLE[key] - len(sampled_data[key]))
        sampled_data[key] += eli5_data[key][:data_num]
        print(f"{key}-{len(eli5_data[key])}-take {data_num}")
    print(f"{'='*10}FINAL{'='*10}")
    data = []
    for key in sampled_data:
        data += sampled_data[key]
        print(f"{key}-{len(sampled_data[key])}")
    random.shuffle(data)
    global_id = 0
    for d in data:
        d["id"] = global_id
        global_id += 1
    print(f"Total:{global_id}")
    # save to output_path
    save_jsonl(args.output_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--longform_dir", type=str, default=None)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
