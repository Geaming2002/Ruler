import re
import argparse
from utils.config import MetaLengthToken, RANGE
from utils import load_jsonl
from rich.table import Table
from rich.console import Console
from utils.count import count_words


def main(args):
    # raw data load
    df = load_jsonl(args.dataset_path)
    # draw table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim", width=15)
    for mlt in MetaLengthToken:
        table.add_column(mlt[0].split(":")[-1][:-1], justify="right")
    table.add_column("FM", justify="right")
    table.add_column("Avg", justify="right")
    count = {mlt[0]: 0 for mlt in MetaLengthToken}
    for d in df:
        if d["output"].count("MLT") != 1:
            d["output"] = "[MLT:" + d["output"].split("[MLT:")[1]
        for mlt in MetaLengthToken:
            if mlt[0] in d["output"]:
                count[mlt[0]] += 1
    hit = 0
    all_wc = 0
    for d in df:
        cleaned_text = re.sub(r"\[MLT:\d+\]", "", d["output"])  # clean MLT
        wc = count_words(cleaned_text)
        mlt = d["output"].split("]")[0] + "]"
        if (wc > RANGE[mlt.split(":")[-1][:-1]]["FM"][0]) and (
            wc <= RANGE[mlt.split(":")[-1][:-1]]["FM"][1]
        ):
            hit += 1
        all_wc += wc
    table.add_row(
        args.dataset_path.split("/")[-1].split("tl_")[-1][:15],
        f"{count['[MLT:10]']}",
        f"{count['[MLT:30]']}",
        f"{count['[MLT:50]']}",
        f"{count['[MLT:80]']}",
        f"{count['[MLT:150]']}",
        f"{count['[MLT:300]']}",
        f"{count['[MLT:500]']}",
        f"{count['[MLT:700]']}",
        f"{count['[MLT:>800]']}",
        f"{hit/len(df)*100:.2f}",
        f"{all_wc/len(df):.0f}",
    )
    console = Console()
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
