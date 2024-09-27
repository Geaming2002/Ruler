import argparse
from utils.config import RANGE, LEVEL0, LEVEL1, LEVEL2
from utils import load_jsonl
from rich.table import Table
from rich.console import Console
from utils.count import count_words


def metric_targetlength(df, LEVEL):
    result = {
        targetlength: {"PM": {"in": 0, "out": 0}, "FM": {"in": 0, "out": 0}}
        for targetlength in LEVEL
    }
    for d in df:
        length = count_words(d["output"])
        if d["TargetLength"] in result:
            # PM
            if (
                length > RANGE[d["TargetLength"]]["PM"][0]
                and length <= RANGE[d["TargetLength"]]["PM"][1]
            ):
                result[d["TargetLength"]]["PM"]["in"] += 1
            else:
                result[d["TargetLength"]]["PM"]["out"] += 1
            # FM
            if (
                length > RANGE[d["TargetLength"]]["FM"][0]
                and length <= RANGE[d["TargetLength"]]["FM"][1]
            ):
                result[d["TargetLength"]]["FM"]["in"] += 1
            else:
                result[d["TargetLength"]]["FM"]["out"] += 1
    # draw table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("TargetLength", style="dim", width=12)
    table.add_column("PM_in", justify="right")
    table.add_column("PM_out", justify="right")
    table.add_column("PM", justify="right")
    table.add_column("FM_in", justify="right")
    table.add_column("FM_out", justify="right")
    table.add_column("FM", justify="right")
    # latex_str = ""
    for key in result:
        table.add_row(
            key,
            f"{result[key]['PM']['in']}",
            f"{result[key]['PM']['out']}",
            f"{result[key]['PM']['in'] / (result[key]['PM']['in'] + result[key]['PM']['out'])*100:.2f}",
            f"{result[key]['FM']['in']}",
            f"{result[key]['FM']['out']}",
            f"{result[key]['FM']['in'] / (result[key]['FM']['in'] + result[key]['FM']['out'])*100:.2f}",
        )
        # latex_str = (
        #     latex_str
        #     + "&"
        #     + f"{result[key]['PM']['in'] / (result[key]['PM']['in'] + result[key]['PM']['out'])*100:.2f}"
        #     + "&"
        #     + f"{result[key]['FM']['in'] / (result[key]['FM']['in'] + result[key]['FM']['out'])*100:.2f}"
        # )
    table.add_row(
        "Total",
        f"{sum([result[key]['PM']['in']for key in result])}",
        f"{sum([result[key]['PM']['out']for key in result])}",
        f"{sum([result[key]['PM']['in']for key in result]) / (sum([result[key]['PM']['in']for key in result]) + sum([result[key]['PM']['out']for key in result]))*100:.2f}",
        f"{sum([result[key]['FM']['in']for key in result])}",
        f"{sum([result[key]['FM']['out']for key in result])}",
        f"{sum([result[key]['FM']['in']for key in result]) / (sum([result[key]['FM']['in']for key in result]) + sum([result[key]['FM']['out']for key in result]))*100:.2f}",
    )
    console = Console()
    console.print(table)
    # print(latex_str)


def main(args):
    # raw data load
    df = load_jsonl(args.dataset_path)
    print(f"> LEVEL0{'='*20}")
    metric_targetlength(df, LEVEL0)
    print(f"> LEVEL1{'='*20}")
    metric_targetlength(df, LEVEL1)
    print(f"> LEVEL2{'='*20}")
    metric_targetlength(df, LEVEL2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
