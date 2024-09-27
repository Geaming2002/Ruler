import argparse
from utils import load_jsonl
from rich.table import Table
from rich.console import Console
from utils.count import count_words
import scipy.stats as stats


def main(args):
    # raw data load
    vanilla_df = load_jsonl(args.vanilla_dataset_path)
    ruler_dataset_path = args.vanilla_dataset_path.replace("tlg_", "tlg_Ruler_")
    ruler_df = load_jsonl(ruler_dataset_path)
    print(ruler_dataset_path)
    vanilla_lengths, ruler_lengths = [], []
    for idx in range(len(vanilla_df)):
        vanilla_lengths.append(count_words(vanilla_df[idx]["output"]))
        ruler_lengths.append(count_words(ruler_df[idx]["output"]))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim", width=12)
    table.add_column("t", justify="right")
    table.add_column("p", justify="right")
    t_statistic, p_value = stats.ttest_ind(ruler_lengths,vanilla_lengths)
    table.add_row(
        args.vanilla_dataset_path.split("/")[-1][4:],
        f"{t_statistic:.4f}",
        f"{p_value:.4f}",
    )
    console = Console()
    console.print(table)
    # print(f"{t_statistic:.4f}|{p_value:.4f}|")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla_dataset_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
