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
        table.add_column(mlt[0].split(':')[-1][:-1], justify="right")
    table.add_column('Acc', justify="right")
    count = {mlt[0]:0 for mlt in MetaLengthToken}
    hit = {mlt[0]:0 for mlt in MetaLengthToken}
    for d in df:
        count[f"[MLT:{d['TargetLength']}]"] += 1
    for d in df:
        wc = count_words(d['output'])
        mlt = f"[MLT:{d['TargetLength']}]"
        if (wc > RANGE[mlt.split(':')[-1][:-1]]['FM'][0]) and (wc <= RANGE[mlt.split(':')[-1][:-1]]['FM'][1]):
            hit[mlt] += 1
    # print(hit)
    # print(count)
    table.add_row(
        args.dataset_path.split('/')[-1].split('tl_')[-1][:15],
        f"{hit['[MLT:10]']/count['[MLT:10]']*100:.2f}",
        f"{hit['[MLT:30]']/count['[MLT:30]']*100:.2f}",
        f"{hit['[MLT:50]']/count['[MLT:50]']*100:.2f}",
        f"{hit['[MLT:80]']/count['[MLT:80]']*100:.2f}",
        f"{hit['[MLT:150]']/count['[MLT:150]']*100:.2f}",
        f"{hit['[MLT:300]']/count['[MLT:300]']*100:.2f}",
        f"{hit['[MLT:500]']/count['[MLT:500]']*100:.2f}",
        f"{hit['[MLT:700]']/count['[MLT:700]']*100:.2f}",
        f"{hit['[MLT:>800]']/count['[MLT:>800]']*100:.2f}",
        f"{sum(hit.values())/sum(count.values())*100:.2f}",
    )
    console = Console()
    console.print(table)
    latex = [
        f"{hit['[MLT:10]']/count['[MLT:10]']*100:.1f}",
        f"{hit['[MLT:30]']/count['[MLT:30]']*100:.1f}",
        f"{hit['[MLT:50]']/count['[MLT:50]']*100:.1f}",
        f"{hit['[MLT:80]']/count['[MLT:80]']*100:.1f}",
        f"{hit['[MLT:150]']/count['[MLT:150]']*100:.1f}",
        f"{hit['[MLT:300]']/count['[MLT:300]']*100:.1f}",
        f"{hit['[MLT:500]']/count['[MLT:500]']*100:.1f}",
        f"{hit['[MLT:700]']/count['[MLT:700]']*100:.1f}",
        f"{hit['[MLT:>800]']/count['[MLT:>800]']*100:.1f}",
        f"{sum(hit.values())/sum(count.values())*100:.2f}",
    ]
    print('&'.join(latex) + '\\\\')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
