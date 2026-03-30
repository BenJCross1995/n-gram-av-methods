#!/usr/bin/env python3
"""
Create and save adjusted, like-for-like, and like-for-like-adjusted datasets.

Pipeline
--------
1. Read an input dataframe from .xlsx or .rds
2. Create an adjusted version of the original data
3. Create a like-for-like version of the original data
4. Create an adjusted version of the like-for-like data
5. Save all derived outputs
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from from_root import from_root

# Make local src/ imports available
sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import write_rds, read_rds


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for input/output locations and pipeline settings.
    """
    ap = argparse.ArgumentParser(
        description="Create adjusted, like-for-like, and like-for-like-adjusted datasets"
    )

    ap.add_argument("--input_loc", required=True)
    ap.add_argument("--adjusted_save_loc", required=True)
    ap.add_argument("--like_for_like_save_loc", required=True)
    ap.add_argument("--like_for_like_adjusted_save_loc", required=True)

    ap.add_argument("--sheet_name", default=0)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument(
        "--group_cols",
        nargs="+",
        default=[
            "data_type",
            "corpus",
            "scoring_model",
            "max_context_tokens",
            "min_token_size",
        ],
    )

    ap.add_argument(
        "--base_cols",
        nargs="+",
        default=[
            "data_type",
            "corpus",
            "scoring_model",
            "max_context_tokens",
        ],
    )

    ap.add_argument("--problem_col", default="problem")

    return ap.parse_args()


def get_file_type(path: str) -> str:
    """
    Get the lowercase file extension for a supported input/output path.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        Supported extension, either '.xlsx' or '.rds'.

    Raises
    ------
    ValueError
        If the extension is unsupported.
    """
    ext = Path(path).suffix.lower()
    if ext not in [".xlsx", ".rds"]:
        raise ValueError(
            f"Unsupported file extension for {path}. "
            "Only .xlsx and .rds are supported."
        )
    return ext


def ensure_parent_dir(path: str) -> None:
    """
    Create the parent directory for a file if it does not already exist.

    Parameters
    ----------
    path : str
        File path whose parent directory should be created if needed.
    """
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def read_input_file(input_loc: str, sheet_name=0) -> pd.DataFrame:
    """
    Read an input dataframe from .xlsx or .rds.

    Parameters
    ----------
    input_loc : str
        Input file path.
    sheet_name : int | str, default=0
        Sheet name or sheet index to read if the input is an Excel file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    file_type = get_file_type(input_loc)

    if file_type == ".xlsx":
        return pd.read_excel(input_loc, sheet_name=sheet_name)

    return read_rds(input_loc)


def write_output(df: pd.DataFrame, save_loc: str, overwrite: bool = False) -> None:
    """
    Write a dataframe to .xlsx or .rds depending on the output extension.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to write.
    save_loc : str
        Output file path.
    overwrite : bool, default=False
        Whether to overwrite an existing file.
    """
    if os.path.exists(save_loc) and not overwrite:
        print(f"Output already exists, skipping write: {save_loc}")
        return

    ensure_parent_dir(save_loc)
    file_type = get_file_type(save_loc)

    if file_type == ".xlsx":
        df.to_excel(save_loc, index=False)
    else:
        write_rds(df, save_loc)


def validate_required_columns(
    df: pd.DataFrame,
    group_cols: list[str],
    base_cols: list[str],
    problem_col: str,
) -> None:
    """
    Validate that all required columns are present in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        Full grouping columns including min_token_size.
    base_cols : list[str]
        Grouping columns excluding min_token_size.
    problem_col : str
        Problem identifier column name.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    required_cols = set(group_cols) | set(base_cols) | {problem_col}
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
def get_no_context_group_cols(
    group_cols: list[str],
    max_context_col: str = "max_context_tokens",
) -> list[str]:
    """
    Remove the max-context column from grouping columns for no-context checks/appends.
    """
    return [col for col in group_cols if col != max_context_col]


def get_valid_no_context_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    no_context_col: str = "no_context_sum_log_probs",
) -> pd.DataFrame:
    """
    Return only the groups where the no-context score is constant within group.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        Columns defining each group.
    no_context_col : str, default='no_context_sum_log_probs'
        Column whose within-group constancy is checked.

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per valid group.
    """
    if no_context_col not in df.columns:
        raise ValueError(f"Missing required column: {no_context_col}")

    summary = (
        df.groupby(group_cols, dropna=False)[no_context_col]
        .agg(n_unique="nunique")
        .reset_index()
    )

    valid_groups = summary[summary["n_unique"] == 1].drop(columns="n_unique").copy()
    return valid_groups


def append_no_context_partition_rows(
    df: pd.DataFrame,
    group_cols: list[str],
    no_context_col: str = "no_context_sum_log_probs",
    max_context_col: str = "max_context_tokens",
    new_max_context_value: int | float = 0,
    copy_to_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Append one synthetic max-context row per valid group using the no-context score.

    A valid group is one where `no_context_col` is constant within the group.

    For each valid group:
    - take the first row in original order
    - set max_context_tokens to 0
    - copy the no-context value into the requested target columns
    - append the row back to the dataframe

    Groups where `no_context_col` is not constant are ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        Columns defining each group.
    no_context_col : str, default='no_context_sum_log_probs'
        Source no-context score column.
    max_context_col : str, default='max_context_tokens'
        Column to overwrite with 0 for the synthetic rows.
    new_max_context_value : int | float, default=0
        Value to assign to max_context_col in appended rows.
    copy_to_cols : list[str] | None, default=None
        Columns that should receive the no_context_col value.

    Returns
    -------
    pd.DataFrame
        Original dataframe with appended synthetic rows for valid groups only.
    """
    if copy_to_cols is None:
        copy_to_cols = [no_context_col]

    required_cols = set(group_cols) | {no_context_col, max_context_col} | set(copy_to_cols)
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns for no-context row append: {missing_cols}")

    valid_groups = get_valid_no_context_groups(
        df=df,
        group_cols=group_cols,
        no_context_col=no_context_col,
    )

    if valid_groups.empty:
        return df.copy()

    # Keep only rows belonging to valid groups, then take first row per group
    valid_df = df.merge(valid_groups, on=group_cols, how="inner")

    first_rows = (
        valid_df.groupby(group_cols, dropna=False, sort=False)
        .head(1)
        .copy()
    )

    synthetic_rows = first_rows.copy()
    synthetic_rows[max_context_col] = new_max_context_value

    for col in copy_to_cols:
        synthetic_rows[col] = synthetic_rows[no_context_col]

    out = pd.concat([df, synthetic_rows], ignore_index=True)
    return out

def get_distinct_problems(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Count rows at each grouping level.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        Grouping columns used to define distinct levels.

    Returns
    -------
    pd.DataFrame
        One row per distinct group with a problem_count column.
    """
    distinct_levels = (
        df.groupby(group_cols)
        .size()
        .reset_index(name="problem_count")
    )

    return distinct_levels


def add_adjusted_token_size(
    df: pd.DataFrame,
    distinct_problems: pd.DataFrame,
    base_cols: list[str],
    problem_col: str = "problem",
) -> pd.DataFrame:
    """
    Add adjusted token-size rows by carrying problems forward to lower thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    distinct_problems : pd.DataFrame
        Distinct level table produced by get_distinct_problems().
    base_cols : list[str]
        Base grouping columns excluding min_token_size.
    problem_col : str, default='problem'
        Problem identifier column.

    Returns
    -------
    pd.DataFrame
        Adjusted dataframe with min_token_size reassigned to the adjusted threshold.
    """
    levels = distinct_problems[base_cols + ["min_token_size"]].rename(
        columns={"min_token_size": "min_token_size_adjusted"}
    )

    tmp = df.merge(levels, on=base_cols, how="inner")
    tmp = tmp[tmp["min_token_size"] >= tmp["min_token_size_adjusted"]]

    tmp = tmp.sort_values(
        base_cols + ["min_token_size_adjusted", problem_col, "min_token_size"],
        kind="mergesort",
    )

    tmp["row_number"] = (
        tmp.groupby(base_cols + ["min_token_size_adjusted", problem_col])
        .cumcount()
        .add(1)
    )

    out = tmp[tmp["row_number"] == 1].drop(columns=["row_number"]).copy()
    out = out.drop(columns=["min_token_size"])
    out = out.rename(columns={"min_token_size_adjusted": "min_token_size"})

    return out


def adjusted_token_size_pipeline(
    df: pd.DataFrame,
    group_cols: list[str],
    base_cols: list[str],
    problem_col: str = "problem",
) -> pd.DataFrame:
    """
    Run the adjusted token-size pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        Grouping columns including min_token_size.
    base_cols : list[str]
        Base grouping columns excluding min_token_size.
    problem_col : str, default='problem'
        Problem identifier column.

    Returns
    -------
    pd.DataFrame
        Adjusted dataframe.
    """
    distinct_levels = get_distinct_problems(df, group_cols=group_cols)

    adjusted_df = add_adjusted_token_size(
        df=df,
        distinct_problems=distinct_levels,
        base_cols=base_cols,
        problem_col=problem_col,
    )

    return adjusted_df


def get_like_for_like_problems(
    df: pd.DataFrame,
    base_cols: list[str],
    problem_col: str = "problem",
) -> pd.DataFrame:
    """
    Create a like-for-like dataframe across min_token_size levels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    base_cols : list[str]
        Base grouping columns excluding min_token_size.
    problem_col : str, default='problem'
        Problem identifier column.

    Returns
    -------
    pd.DataFrame
        Like-for-like dataframe.
    """
    anchors = (
        df[base_cols + [problem_col, "min_token_size"]]
        .drop_duplicates()
        .rename(columns={"min_token_size": "anchor_min_token_size"})
    )

    out_like_for_like = df.merge(
        anchors,
        on=base_cols + [problem_col],
        how="inner",
    )

    out_like_for_like = out_like_for_like[
        out_like_for_like["min_token_size"] <= out_like_for_like["anchor_min_token_size"]
    ].copy()

    return out_like_for_like


def main():
    """
    Run the full pipeline for one input file.

    Steps
    -----
    1. Read input data
    2. Create and save adjusted data from the original input
    3. Create and save like-for-like data from the original input
    4. Create and save adjusted data from the like-for-like input
    """
    args = parse_args()

    if not os.path.exists(args.input_loc):
        print(f"Input file does not exist, skipping: {args.input_loc}")
        sys.exit(0)

    print(f"Reading: {args.input_loc}")
    df = read_input_file(args.input_loc, sheet_name=args.sheet_name)

    validate_required_columns(
        df=df,
        group_cols=args.group_cols,
        base_cols=args.base_cols,
        problem_col=args.problem_col,
    )

    print(f"Input rows: {len(df)}")

    no_context_group_cols = get_no_context_group_cols(
        args.group_cols,
        max_context_col="max_context_tokens",
    )

    df = append_no_context_partition_rows(
        df=df,
        group_cols=no_context_group_cols,
        no_context_col="no_context_sum_log_probs",
        max_context_col="max_context_tokens",
        new_max_context_value=0,
        copy_to_cols=["unknown_sum_log_probs", "no_context_sum_log_probs"],
    )
    
    print(f"Input rows: {len(df)}")
    
    adjusted_df = adjusted_token_size_pipeline(
        df=df,
        group_cols=args.group_cols,
        base_cols=args.base_cols,
        problem_col=args.problem_col,
    )

    print(f"Adjusted rows: {len(adjusted_df)}")
    print(f"Writing: {args.adjusted_save_loc}")
    write_output(
        df=adjusted_df,
        save_loc=args.adjusted_save_loc,
        overwrite=args.overwrite,
    )

    like_for_like_df = get_like_for_like_problems(
        df=df,
        base_cols=args.base_cols,
        problem_col=args.problem_col,
    )

    print(f"Like-for-like rows: {len(like_for_like_df)}")
    print(f"Writing: {args.like_for_like_save_loc}")
    write_output(
        df=like_for_like_df,
        save_loc=args.like_for_like_save_loc,
        overwrite=args.overwrite,
    )

    like_for_like_adjusted_df = adjusted_token_size_pipeline(
        df=like_for_like_df,
        group_cols=args.group_cols,
        base_cols=args.base_cols,
        problem_col=args.problem_col,
    )

    print(f"Like-for-like adjusted rows: {len(like_for_like_adjusted_df)}")
    print(f"Writing: {args.like_for_like_adjusted_save_loc}")
    write_output(
        df=like_for_like_adjusted_df,
        save_loc=args.like_for_like_adjusted_save_loc,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()