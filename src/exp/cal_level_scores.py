import argparse
from utils.config import RANGE, LEVEL0, LEVEL1, LEVEL2
from utils import load_jsonl
from rich.table import Table
from rich.console import Console
from utils.count import count_words


def main(args):
    result = RANGE
    for key in result:
        (
            result[key]["PM_in"],
            result[key]["PM_out"],
            result[key]["FM_in"],
            result[key]["FM_out"],
        ) = 0, 0, 0, 0
    # raw data load
    df = load_jsonl(args.dataset_path)
    # calculate metric
    for d in df:
        length = count_words(d["output"])
        # PM
        if (
            length > result[d["TargetLength"]]["PM"][0]
            and length <= result[d["TargetLength"]]["PM"][1]
        ):
            result[d["TargetLength"]]["PM_in"] += 1
        else:
            result[d["TargetLength"]]["PM_out"] += 1
        # FM
        if (
            length > result[d["TargetLength"]]["FM"][0]
            and length <= result[d["TargetLength"]]["FM"][1]
        ):
            result[d["TargetLength"]]["FM_in"] += 1
        else:
            result[d["TargetLength"]]["FM_out"] += 1
    # level 0
    levle0_pm_in, levle0_pm_out, levle0_fm_in, levle0_fm_out = 0, 0, 0, 0
    # level 1
    levle1_pm_in, levle1_pm_out, levle1_fm_in, levle1_fm_out = 0, 0, 0, 0
    # level 2
    levle2_pm_in, levle2_pm_out, levle2_fm_in, levle2_fm_out = 0, 0, 0, 0
    for key in result:
        if key in LEVEL0:
            levle0_pm_in += result[key]["PM_in"]
            levle0_pm_out += result[key]["PM_out"]
            levle0_fm_in += result[key]["FM_in"]
            levle0_fm_out += result[key]["FM_out"]
        elif key in LEVEL1:
            levle1_pm_in += result[key]["PM_in"]
            levle1_pm_out += result[key]["PM_out"]
            levle1_fm_in += result[key]["FM_in"]
            levle1_fm_out += result[key]["FM_out"]
        elif key in LEVEL2:
            levle2_pm_in += result[key]["PM_in"]
            levle2_pm_out += result[key]["PM_out"]
            levle2_fm_in += result[key]["FM_in"]
            levle2_fm_out += result[key]["FM_out"]
    # draw table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Level", style="dim", width=12)
    table.add_column("PM_in", justify="right")
    table.add_column("PM_out", justify="right")
    table.add_column("PM", justify="right")
    table.add_column("FM_in", justify="right")
    table.add_column("FM_out", justify="right")
    table.add_column("FM", justify="right")
    table.add_row(
        "Level:0",
        f"{levle0_pm_in}",
        f"{levle0_pm_out}",
        f"{levle0_pm_in/(levle0_pm_in + levle0_pm_out)*100:.2f}",
        f"{levle0_fm_in}",
        f"{levle0_fm_out}",
        f"{levle0_fm_in/(levle0_fm_in + levle0_fm_out)*100:.2f}",
    )
    table.add_row(
        "Level:1",
        f"{levle1_pm_in}",
        f"{levle1_pm_out}",
        f"{levle1_pm_in/(levle1_pm_in + levle1_pm_out)*100:.2f}",
        f"{levle1_fm_in}",
        f"{levle1_fm_out}",
        f"{levle1_fm_in/(levle1_fm_in + levle1_fm_out)*100:.2f}",
    )
    table.add_row(
        "Level:2",
        f"{levle2_pm_in}",
        f"{levle2_pm_out}",
        f"{levle2_pm_in/(levle2_pm_in + levle2_pm_out)*100:.2f}",
        f"{levle2_fm_in}",
        f"{levle2_fm_out}",
        f"{levle2_fm_in/(levle2_fm_in + levle2_fm_out)*100:.2f}",
    )
    table.add_row(
        "All Level",
        f"{levle0_pm_in +levle1_pm_in + levle2_pm_in}",
        f"{levle0_pm_out+ levle1_pm_out + levle2_pm_out}",
        f"{(levle0_pm_in +levle1_pm_in + levle2_pm_in)/(levle0_pm_in +levle1_pm_in + levle2_pm_in + levle0_pm_out+ levle1_pm_out + levle2_pm_out)*100:.2f}",
        f"{levle0_fm_in +levle1_fm_in + levle2_fm_in}",
        f"{levle0_fm_out+ levle1_fm_out + levle2_fm_out}",
        f"{(levle0_fm_in +levle1_fm_in + levle2_fm_in)/(levle0_fm_in +levle1_fm_in + levle2_fm_in + levle0_fm_out+ levle1_fm_out + levle2_fm_out)*100:.2f}",
    )
    console = Console()
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
