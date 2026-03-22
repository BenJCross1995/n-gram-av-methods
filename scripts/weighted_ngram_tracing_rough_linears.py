#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_excel_sheets
from utils import list_xlsx_files


def calculate_rough_weighted_scores(weighted_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rough token-length-weighted Simpson and Jaccard scores for each
    min_token_size threshold.

    For each distinct token_level t:
        - keep rows where token_level >= t
        - rough_simpson_score = sum(simpson * token_level) / sum(token_level)
        - rough_jaccard_score = sum(jaccard * token_level) / sum(token_level)

    Parameters
    ----------
    weighted_raw : pd.DataFrame
        DataFrame containing at least:
        - token_level
        - simpson
        - jaccard

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - min_token_size
        - simpson_score
        - jaccard_score
    """
    required_cols = {"token_level", "simpson", "jaccard"}
    missing = required_cols - set(weighted_raw.columns)
    if missing:
        raise ValueError(f"weighted_raw is missing required columns: {sorted(missing)}")

    thresholds = sorted(weighted_raw["token_level"].dropna().unique())

    results = []
    for t in thresholds:
        sub = weighted_raw[weighted_raw["token_level"] >= t].copy()
        weight_sum = sub["token_level"].sum()

        simpson_score = (
            (sub["simpson"] * sub["token_level"]).sum() / weight_sum
            if weight_sum > 0 else 0.0
        )
        jaccard_score = (
            (sub["jaccard"] * sub["token_level"]).sum() / weight_sum
            if weight_sum > 0 else 0.0
        )

        results.append({
            "min_token_size": t,
            "simpson_score": simpson_score,
            "jaccard_score": jaccard_score,
        })

    return pd.DataFrame(results)


def build_rough_score_metadata(metadata: pd.DataFrame, weighted_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final metadata-style output using the original metadata sheet
    and rough weighted scores from the weighted raw sheet.
    """
    if metadata.empty:
        raise ValueError("metadata sheet is empty")

    rough_scores = calculate_rough_weighted_scores(weighted_raw)

    metadata = metadata.copy()

    if "weight" in metadata.columns:
        metadata["weight"] = "linear_rough"

    metadata = metadata.drop(
        columns=["min_token_size", "simpson_score", "jaccard_score"],
        errors="ignore"
    )

    metadata_row = metadata.iloc[[0]].copy()

    metadata_out = pd.concat(
        [metadata_row] * len(rough_scores),
        ignore_index=True
    )

    metadata_out["min_token_size"] = rough_scores["min_token_size"].values
    metadata_out["simpson_score"] = rough_scores["simpson_score"].values
    metadata_out["jaccard_score"] = rough_scores["jaccard_score"].values

    return metadata_out


def process_file(file_loc: str | Path, save_loc: str | Path) -> None:
    """
    Read one Excel file, keep the original sheets, replace the metadata sheet
    with the rough-score metadata output, and save to a new file.
    """
    sheet_names = ["ngrams", "weighted raw", "metadata"]
    sheets = read_excel_sheets(file_loc, sheet_names=sheet_names)

    ngrams = sheets["ngrams"].copy()
    weighted_raw = sheets["weighted raw"].copy()
    metadata = sheets["metadata"].copy()

    rough_metadata = build_rough_score_metadata(metadata=metadata, weighted_raw=weighted_raw)

    save_loc = Path(save_loc)
    save_loc.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(save_loc, engine="openpyxl") as writer:
        ngrams.to_excel(writer, sheet_name="ngrams", index=False)
        weighted_raw.to_excel(writer, sheet_name="weighted raw", index=False)
        rough_metadata.to_excel(writer, sheet_name="metadata", index=False)

def process_directory(input_dir: str | Path, save_dir: str | Path) -> None:
    """
    Loop through all .xlsx files in input_dir, process each one, and save
    the output to save_dir using the same filename.

    If the input directory does not exist, skip it.
    If the output file already exists, skip it.
    """
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)

    if not input_dir.exists():
        print(f"Skipping missing input directory: {input_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    xlsx_files = list_xlsx_files(input_dir)

    if len(xlsx_files) == 0:
        print(f"No .xlsx files found in: {input_dir}")
        return

    for file_loc in xlsx_files:
        save_loc = save_dir / file_loc.name

        if save_loc.exists():
            print(f"Skipping existing file: {save_loc}")
            continue

        try:
            print(f"Processing: {file_loc}")
            process_file(file_loc=file_loc, save_loc=save_loc)
            print(f"Saved: {save_loc}")
        except Exception as e:
            print(f"Failed: {file_loc}")
            print(f"Error: {e}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Create rough weighted score Excel files for all .xlsx files in a directory"
    )
    ap.add_argument("--input_dir", required=True, help="Directory containing input .xlsx files")
    ap.add_argument("--save_dir", required=True, help="Directory to save output .xlsx files")
    return ap.parse_args()


def main():
    args = parse_args()
    process_directory(args.input_dir, args.save_dir)


if __name__ == "__main__":
    main()