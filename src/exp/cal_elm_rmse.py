import argparse
from utils.config import LEVEL0, LEVEL1, LEVEL2
from utils import load_jsonl
from utils.count import count_words
from rich.table import Table
from rich.console import Console
from sklearn.metrics import root_mean_squared_error


def calculate_rmse(actual, predicted):
    """
    Calculate the Root Mean Square Error between two arrays using scikit-learn.

    Parameters:
    actual (array-like): The array of actual values.
    predicted (array-like): The array of predicted values.

    Returns:
    float: The calculated RMSE value.
    """
    # Calculate the RMSE
    rmse = root_mean_squared_error(actual, predicted)

    return rmse


def elm(list1, list2):
    """
    Count the number of elements that are the same in both lists at the same positions.

    Parameters:
    list1 (list): The first list.
    list2 (list): The second list.

    Returns:
    int: The count of elements that are the same at the same positions.
    """
    # Use zip to pair the elements and then check for equality
    same_position_count = sum(1 for a, b in zip(list1, list2) if a == b)

    return same_position_count


def main(args):
    # raw data load
    df = load_jsonl(args.dataset_path)
    # calculate metric
    predicted_lengths = []
    target_lengths = []
    predicted_lengths_0, predicted_lengths_1, predicted_lengths_2 = [], [], []
    target_lengths_0, target_lengths_1, target_lengths_2 = [], [], []
    for d in df:
        length = count_words(d["output"])
        if d["TargetLength"] != ">800":
            predicted_lengths.append(length)
            target_lengths.append(int(d["TargetLength"]))
        if d["TargetLength"] in LEVEL0:
            predicted_lengths_0.append(length)
            target_lengths_0.append(int(d["TargetLength"]))
        elif d["TargetLength"] in LEVEL1:
            predicted_lengths_1.append(length)
            target_lengths_1.append(int(d["TargetLength"]))
        elif d["TargetLength"] in LEVEL2 and d["TargetLength"] != ">800":
            predicted_lengths_2.append(length)
            target_lengths_2.append(int(d["TargetLength"]))
        else:
            if d["TargetLength"] != ">800":
                raise KeyError
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim", width=12)
    table.add_column("Level 0_elm", justify="right")
    table.add_column("Level 0_rmse", justify="right")
    table.add_column("Level 1_elm", justify="right")
    table.add_column("Level 1_rmse", justify="right")
    table.add_column("Level 2_elm", justify="right")
    table.add_column("Level 2_rmse", justify="right")
    table.add_column("All Level_elm", justify="right")
    table.add_column("All Level 0_rmse", justify="right")
    table.add_row(
        args.dataset_path.split("/")[-1][4:],
        f"{elm(target_lengths_0,predicted_lengths_0)/len(predicted_lengths_0)*100:.2f}",
        f"{calculate_rmse(predicted_lengths_0,target_lengths_0):.2f}",
        f"{elm(target_lengths_1,predicted_lengths_1)/len(predicted_lengths_1)*100:.2f}",
        f"{calculate_rmse(predicted_lengths_1,target_lengths_1):.2f}",
        f"{elm(target_lengths_2,predicted_lengths_2)/len(predicted_lengths_2)*100:.2f}",
        f"{calculate_rmse(predicted_lengths_2,target_lengths_2):.2f}",
        f"{elm(target_lengths,predicted_lengths)/len(predicted_lengths)*100:.2f}",
        f"{calculate_rmse(predicted_lengths,target_lengths):.2f}",
    )
    console = Console()
    console.print(table)
    print(f"{elm(target_lengths_0,predicted_lengths_0)/len(predicted_lengths_0)*100:.2f}/{calculate_rmse(predicted_lengths_0,target_lengths_0):.2f}|{elm(target_lengths_1,predicted_lengths_1)/len(predicted_lengths_1)*100:.2f}/{calculate_rmse(predicted_lengths_1,target_lengths_1):.2f}|{elm(target_lengths_2,predicted_lengths_2)/len(predicted_lengths_2)*100:.2f}/{calculate_rmse(predicted_lengths_2,target_lengths_2):.2f}|{elm(target_lengths,predicted_lengths)/len(predicted_lengths)*100:.2f}/{calculate_rmse(predicted_lengths,target_lengths):.2f}|")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
