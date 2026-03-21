#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd

from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_rds, write_rds


def get_completed_problems(df):
    """
    Keep only rows marked as completed.

    Also drops bookkeeping columns that are only needed before completion
    filtering.
    """
    completed_df = (
        df[df["completed"] == True]
        .drop(columns=["num_problems", "ngrams_found", "completed"], errors="ignore")
        .copy()
    )

    return completed_df


def get_distinct_problems(
    df,
    group_cols=None
):
    """
    Count rows within each grouping combination.

    In your current workflow this is used to identify the available
    min_token_size levels within each grouping structure.
    """
    if group_cols is None:
        group_cols = [
            "data_type", "corpus", "scoring_model",
            "max_context_tokens", "min_token_size"
        ]

    distinct_levels = (
        df
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="problem_count")
    )

    return distinct_levels


def like_for_like(df, base_cols, compare_cols, id_col="problem"):
    """
    Keep only IDs that appear in all compare-column combinations within
    each base group.

    Example
    -------
    If:
        base_cols    = ["data_type", "corpus", "scoring_model", "min_token_size"]
        compare_cols = ["weight", "alpha", "base"]
        id_col       = "problem"

    then, within each data_type/corpus/scoring_model/min_token_size group,
    a problem is only kept if it exists for every distinct combination of
    weight/alpha/base present in that group.
    """
    work = df.copy()
    all_group_cols = base_cols + compare_cols

    # Fill missing values temporarily so groupby/merge logic treats them
    # consistently. This is especially important for columns like alpha/base
    # where some methods naturally leave them missing.
    for col in all_group_cols:
        work[col] = work[col].astype("object").where(work[col].notna(), "__MISSING__")

    # Number of compare combinations that should exist within each base group
    required = (
        work[base_cols + compare_cols]
        .drop_duplicates()
        .groupby(base_cols, dropna=False)
        .size()
        .reset_index(name="required_n")
    )

    # Number of compare combinations actually observed for each ID
    observed = (
        work[base_cols + compare_cols + [id_col]]
        .drop_duplicates()
        .groupby(base_cols + [id_col], dropna=False)
        .size()
        .reset_index(name="observed_n")
    )

    # Keep IDs whose observed combination count matches the required count
    keep = (
        observed
        .merge(required, on=base_cols, how="left")
        .loc[lambda x: x["observed_n"] == x["required_n"], base_cols + [id_col]]
        .drop_duplicates()
    )

    # Filter original rows down to only the retained IDs
    out = work.merge(keep, on=base_cols + [id_col], how="inner")

    # Put missing values back
    for col in all_group_cols:
        out[col] = out[col].replace("__MISSING__", np.nan)

    return out


def add_adjusted_token_size(
    df,
    distinct_problems,
    base_cols=None,
    token_col="min_token_size"
):
    """
    Create adjusted token-size rows.

    Logic
    -----
    Suppose a problem exists at token size 4, and token size 3 also exists
    as a valid grouping level for the same base group. Then that problem
    should also be included in the adjusted token-size 3 bucket.

    The function:
    1. takes all available token thresholds from `distinct_problems`
    2. joins them back onto the completed data within each base group
    3. keeps rows where original token_col >= adjusted threshold
    4. within each base group + adjusted threshold + ID, keeps the first
       eligible row (smallest original token size after sorting)
    5. replaces the original token column with the adjusted one
    """
    if base_cols is None:
        base_cols = ["data_type", "corpus", "scoring_model", "max_context_tokens"]

    adjusted_col = f"{token_col}_adjusted"

    # Pull out the available token thresholds for each base group
    levels = distinct_problems[base_cols + [token_col]].rename(
        columns={token_col: adjusted_col}
    )

    # Join every available threshold onto each row in the same base group
    tmp = df.merge(levels, on=base_cols, how="inner")

    # Keep rows where the original token size is at least the adjusted threshold
    tmp = tmp[tmp[token_col] >= tmp[adjusted_col]]

    # For each base group + adjusted threshold + ID, keep the smallest
    # original token size that satisfies the threshold
    tmp = tmp.sort_values(
        base_cols + [adjusted_col, "problem", token_col],
        kind="mergesort"
    )

    tmp["row_number"] = (
        tmp.groupby(base_cols + [adjusted_col, "problem"], dropna=False)
        .cumcount()
        .add(1)
    )

    out = tmp[tmp["row_number"] == 1].drop(columns=["row_number"]).copy()

    # Replace original token column with adjusted token column
    out = out.drop(columns=[token_col])
    out = out.rename(columns={adjusted_col: token_col})

    return out


def adjusted_token_size_pipeline(
    df,
    group_cols=None,
    base_cols=None,
    token_col="min_token_size"
):
    """
    Full adjusted-token-size pipeline:
    1. keep completed problems
    2. get distinct grouping levels
    3. build the adjusted token-size dataframe
    """
    if group_cols is None:
        group_cols = [
            "data_type", "corpus", "scoring_model",
            "min_token_size", "weight", "alpha", "base"
        ]

    if base_cols is None:
        base_cols = [
            "data_type", "corpus", "scoring_model",
            "weight", "alpha", "base"
        ]

    completed_df = get_completed_problems(df)

    distinct_levels = get_distinct_problems(
        completed_df,
        group_cols=group_cols
    )

    adjusted_token_size_df = add_adjusted_token_size(
        completed_df,
        distinct_levels,
        base_cols=base_cols,
        token_col=token_col
    )

    return adjusted_token_size_df


def parse_args():
    ap = argparse.ArgumentParser(
        description="Create adjusted-token-size and like-for-like datasets"
    )

    # File paths
    ap.add_argument("--input_loc", required=True, help="Path to input .rds file")
    ap.add_argument("--adjusted_save_loc", required=True, help="Path to save adjusted dataframe .rds")
    ap.add_argument("--like_save_loc", required=True, help="Path to save like-for-like dataframe .rds")

    # Generic column controls
    ap.add_argument("--id_col", default="problem", help="ID column used for like-for-like matching")
    ap.add_argument("--token_col", default="min_token_size", help="Token threshold column to adjust")

    # Grouping used to identify available levels for adjustment
    ap.add_argument(
        "--group_cols",
        nargs="+",
        default=["data_type", "corpus", "scoring_model", "min_token_size", "weight", "alpha", "base"],
        help="Columns used in get_distinct_problems"
    )

    # Base columns used when creating adjusted token-size rows
    ap.add_argument(
        "--adjust_base_cols",
        nargs="+",
        default=["data_type", "corpus", "scoring_model", "weight", "alpha", "base"],
        help="Base columns used in add_adjusted_token_size"
    )

    # Base columns used in like-for-like matching
    ap.add_argument(
        "--like_base_cols",
        nargs="+",
        default=["data_type", "corpus", "scoring_model", "min_token_size"],
        help="Base columns used in like_for_like"
    )

    # Compare columns used in like-for-like matching
    ap.add_argument(
        "--like_compare_cols",
        nargs="+",
        default=["weight", "alpha", "base"],
        help="Compare columns used in like_for_like"
    )

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Reading input file: {args.input_loc}")
    df = read_rds(args.input_loc)
    print(f"Input shape: {df.shape}")

    print("\nRunning adjusted token size pipeline")
    print(f"group_cols: {args.group_cols}")
    print(f"adjust_base_cols: {args.adjust_base_cols}")
    print(f"token_col: {args.token_col}")

    adjusted_df = adjusted_token_size_pipeline(
        df,
        group_cols=args.group_cols,
        base_cols=args.adjust_base_cols,
        token_col=args.token_col
    )
    print(f"adjusted_df shape: {adjusted_df.shape}")

    print("\nRunning like-for-like filter")
    print(f"like_base_cols: {args.like_base_cols}")
    print(f"like_compare_cols: {args.like_compare_cols}")
    print(f"id_col: {args.id_col}")

    like_df = like_for_like(
        adjusted_df,
        base_cols=args.like_base_cols,
        compare_cols=args.like_compare_cols,
        id_col=args.id_col
    )
    print(f"like_df shape: {like_df.shape}")

    # Create output directories if needed
    adjusted_dir = os.path.dirname(args.adjusted_save_loc)
    like_dir = os.path.dirname(args.like_save_loc)

    if adjusted_dir:
        os.makedirs(adjusted_dir, exist_ok=True)
    if like_dir:
        os.makedirs(like_dir, exist_ok=True)

    print(f"\nSaving adjusted_df to: {args.adjusted_save_loc}")
    write_rds(adjusted_df, args.adjusted_save_loc)

    print(f"Saving like_df to: {args.like_save_loc}")
    write_rds(like_df, args.like_save_loc)

    print("\nDone.")


if __name__ == "__main__":
    main()