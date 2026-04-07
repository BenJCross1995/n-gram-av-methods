#!/usr/bin/env python3
"""
Add additional logprob-derived metrics to a phrase/occurrence-level score dataframe,
aggregate to phrase level, then aggregate by minimum token size.

Input:
    per-occurrence RDS file

Output:
    per-problem, per-min_token_size summary RDS file
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_rds, write_rds


def llr_unknown_vs_nocontext(df: pd.DataFrame) -> pd.DataFrame:
    """
    positive -> unknown-context score is more probable
    negative -> no-context score is more probable
    zero     -> equally probable
    """
    out = df.copy()
    out["llr_unknown_vs_nocontext"] = (
        out["unknown_sum_log_probs"] - out["no_context_sum_log_probs"]
    )
    return out


def create_per_phrase_table(
    per_occurrence_table: pd.DataFrame,
    include_sums: bool = False
) -> pd.DataFrame:
    """
    Create an aggregated version of the table containing one row per phrase.

    By default, numeric non-key columns are averaged over occurrences.
    If include_sums=True, summed versions of those same numeric columns are also added
    with a '_sum' suffix.
    """

    key_cols = [
        "data_type", "corpus", "scoring_model", "max_context_tokens", "problem",
        "known_author", "unknown_author", "target", "completed",
        "phrase_num", "phrase", "tokens", "num_tokens"
    ]
    skip_cols = ["phrase_occurrence"]

    group_keys = [
        "data_type", "corpus", "scoring_model", "max_context_tokens", "problem",
        "known_author", "unknown_author", "target", "completed",
        "phrase_num", "phrase", "num_tokens"
    ]

    agg_cols = [
        c for c in per_occurrence_table.columns
        if c not in set(key_cols + skip_cols)
    ]

    numeric_agg_cols = [
        c for c in agg_cols
        if pd.api.types.is_numeric_dtype(per_occurrence_table[c])
    ]

    if include_sums:
        per_phrase_table = (
            per_occurrence_table
            .groupby(group_keys, as_index=False)
            .agg(**{
                **{c: (c, "mean") for c in numeric_agg_cols},
                **{f"{c}_sum": (c, "sum") for c in numeric_agg_cols},
            })
        )
    else:
        per_phrase_table = (
            per_occurrence_table
            .groupby(group_keys, as_index=False)[numeric_agg_cols]
            .mean()
        )

    tokens_map = (
        per_occurrence_table[
            [
                "data_type", "corpus", "scoring_model", "max_context_tokens",
                "problem", "known_author", "unknown_author", "target", "completed",
                "phrase_num", "phrase", "tokens"
            ]
        ]
        .drop_duplicates(
            subset=[
                "data_type", "corpus", "scoring_model", "max_context_tokens",
                "problem", "known_author", "unknown_author", "target", "completed",
                "phrase_num", "phrase"
            ]
        )
    )

    per_phrase_table = per_phrase_table.merge(
        tokens_map,
        on=[
            "data_type", "corpus", "scoring_model", "max_context_tokens",
            "problem", "known_author", "unknown_author", "target", "completed",
            "phrase_num", "phrase"
        ],
        how="left"
    )

    ordered_cols = key_cols + numeric_agg_cols
    if include_sums:
        ordered_cols += [f"{c}_sum" for c in numeric_agg_cols]

    per_phrase_table = per_phrase_table[ordered_cols]

    return per_phrase_table


def create_summary_by_token_num(per_phrase_table: pd.DataFrame) -> pd.DataFrame:
    """
    For each problem/metadata group and each token threshold t (2, 3, ...),
    compute column-wise sums over rows where num_tokens >= t.

    Returns one row per problem per threshold.
    """

    key_cols = [
        "data_type", "corpus", "scoring_model", "max_context_tokens", "problem",
        "known_author", "unknown_author", "target", "completed"
    ]
    skip_cols = ["phrase_num", "phrase", "tokens", "num_tokens"]

    if "num_tokens" not in per_phrase_table.columns:
        raise ValueError("per_phrase_table must contain a 'num_tokens' column.")

    sum_cols = [
        c for c in per_phrase_table.columns
        if c not in set(skip_cols + key_cols)
        and pd.api.types.is_numeric_dtype(per_phrase_table[c])
    ]

    rows = []

    for group_values, group_df in per_phrase_table.groupby(key_cols, dropna=False):
        token_thresholds = sorted(group_df["num_tokens"].dropna().unique())

        for t in token_thresholds:
            filt = group_df[group_df["num_tokens"] >= t]
            sums = filt[sum_cols].sum(numeric_only=True)

            row = dict(zip(key_cols, group_values))
            row["min_token_size"] = int(t)
            row["n_rows"] = int(len(filt))
            row.update(sums.to_dict())
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        key_cols + ["min_token_size"]
    ).reset_index(drop=True)

    ordered_cols = (
        key_cols
        + ["min_token_size", "n_rows"]
        + [c for c in sum_cols if c in out.columns]
    )

    return out[ordered_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a per-occurrence score RDS, add LLR metrics, aggregate to phrase "
            "level, then aggregate by min token size and save to RDS."
        )
    )
    parser.add_argument(
        "--data_loc",
        required=True,
        help="Path to the input per-occurrence .rds file."
    )
    parser.add_argument(
        "--save_loc",
        required=True,
        help="Path to the output .rds file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists."
    )
    parser.add_argument(
        "--include_sums",
        action="store_true",
        help="Also include per-phrase summed numeric columns before token-threshold aggregation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_loc)
    save_path = Path(args.save_loc)

    if not data_path.exists():
        print(f"Input does not exist, skipping: {data_path}")
        return

    if save_path.exists() and not args.overwrite:
        print(f"Output already exists and overwrite is False, skipping: {save_path}")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {data_path}")
    df = read_rds(str(data_path))

    print("Adding llr_unknown_vs_nocontext")
    df_with_scores = llr_unknown_vs_nocontext(df)

    print("Creating per-phrase table")
    phrase_scores = create_per_phrase_table(
        per_occurrence_table=df_with_scores,
        include_sums=args.include_sums
    )

    print("Creating summary by token number")
    score_by_tokens = create_summary_by_token_num(phrase_scores)

    print(f"Saving: {save_path}")
    write_rds(score_by_tokens, str(save_path))

    print("Done")


if __name__ == "__main__":
    main()