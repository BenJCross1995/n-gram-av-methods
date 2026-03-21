#!/usr/bin/env python3
import argparse
import sys
import os

import pandas as pd

from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_excel_sheets, write_rds, read_rds
from utils import list_xlsx_files


def parse_args():
    ap = argparse.ArgumentParser(description="Combine xlsx files or RDS files")

    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["combine_xlsx", "combine_rds"]
    )

    ap.add_argument("--combine_level", type=str, default=None)
    ap.add_argument("--data_loc", default=None)
    ap.add_argument("--save_loc", required=True)
    ap.add_argument("--input_files", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")

    return ap.parse_args()


def log_header(mode: str, combine_level: str | None, save_loc: str) -> None:
    print("==================================================")
    print(f"Mode          : {mode}")
    if combine_level is not None:
        print(f"Combine level : {combine_level}")
    print(f"Save location : {save_loc}")
    print("==================================================")


def should_skip_output(save_loc: str, overwrite: bool) -> bool:
    if os.path.exists(save_loc):
        if overwrite:
            print(f"Path {save_loc} already exists but --overwrite set. Rebuilding.")
            return False
        else:
            print(f"Path {save_loc} already exists. Exiting.")
            return True
    return False


def combine_xlsx_files(
    data_loc: str,
    save_loc: str,
    overwrite: bool = False,
    combine_level: str | None = None
) -> None:
    log_header("combine_xlsx", combine_level, save_loc)

    if should_skip_output(save_loc, overwrite):
        sys.exit(0)

    if not os.path.isdir(data_loc):
        print(f"Input directory does not exist: {data_loc}. Skipping.")
        sys.exit(0)

    file_list = list_xlsx_files(data_loc)

    if not file_list:
        print(f"No .xlsx files found in: {data_loc}. Skipping.")
        sys.exit(0)

    print(f"Found {len(file_list)} xlsx files in {data_loc}")

    result_list = []
    for file in file_list:
        print(f"Reading xlsx: {file}")
        metadata = read_excel_sheets(file, sheet_names=["metadata"])["metadata"]
        result_list.append(metadata)

    result_df = pd.concat(result_list, ignore_index=True, sort=False)

    print(f"Final combined shape: {result_df.shape}")
    print(f"Writing combined xlsx-derived RDS to: {save_loc}")
    write_rds(result_df, save_loc)


def combine_rds_files(
    input_files: list[str],
    save_loc: str,
    overwrite: bool = False,
    combine_level: str | None = None
) -> None:
    log_header("combine_rds", combine_level, save_loc)

    if should_skip_output(save_loc, overwrite):
        sys.exit(0)

    if not input_files:
        print("No input RDS files provided. Skipping.")
        sys.exit(0)

    existing_files = []
    for file in input_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"Missing RDS file, skipping: {file}")

    if not existing_files:
        print("No input RDS files exist. Skipping.")
        sys.exit(0)

    print(f"Found {len(existing_files)} existing RDS files")

    result_list = []
    for file in existing_files:
        print(f"Reading RDS: {file}")
        df = read_rds(file)
        print(f"  Shape: {df.shape}")
        result_list.append(df)

    result_df = pd.concat(result_list, ignore_index=True, sort=False)

    print(f"Final combined shape: {result_df.shape}")
    print(f"Writing combined RDS to: {save_loc}")
    write_rds(result_df, save_loc)


def main():
    args = parse_args()

    if args.mode == "combine_xlsx":
        if args.data_loc is None:
            raise ValueError("--data_loc is required when --mode combine_xlsx")
        combine_xlsx_files(
            data_loc=args.data_loc,
            save_loc=args.save_loc,
            overwrite=args.overwrite,
            combine_level=args.combine_level
        )

    elif args.mode == "combine_rds":
        combine_rds_files(
            input_files=args.input_files,
            save_loc=args.save_loc,
            overwrite=args.overwrite,
            combine_level=args.combine_level
        )


if __name__ == "__main__":
    main()