#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

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
    ap.add_argument("--sheet_names", nargs="+", default=["metadata"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--complete_only", action="store_true")

    return ap.parse_args()


def log_header(
    mode: str,
    combine_level: str | None,
    save_loc: str,
    complete_only: bool = False
) -> None:
    print("==================================================")
    print(f"Mode          : {mode}")
    if combine_level is not None:
        print(f"Combine level : {combine_level}")
    print(f"Save location : {save_loc}")
    print(f"Complete only : {complete_only}")
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


def get_file_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in [".xlsx", ".rds"]:
        raise ValueError(
            f"Unsupported file extension for {path}. "
            "Only .xlsx and .rds are supported."
        )
    return ext


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def write_excel(data: dict[str, pd.DataFrame], save_loc: str) -> None:
    ensure_parent_dir(save_loc)
    with pd.ExcelWriter(save_loc, engine="openpyxl") as writer:
        for sheet_name, df in data.items():
            safe_sheet_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)


def write_output(
    result,
    save_loc: str,
    default_sheet_name: str = "metadata"
) -> None:
    file_type = get_file_type(save_loc)
    ensure_parent_dir(save_loc)

    if file_type == ".rds":
        if isinstance(result, dict):
            if len(result) != 1:
                raise ValueError(
                    "Cannot write multiple sheets to a single .rds file. "
                    "Please either save to .xlsx or pass a single sheet name."
                )
            result_df = next(iter(result.values()))
            write_rds(result_df, save_loc)
        else:
            write_rds(result, save_loc)

    elif file_type == ".xlsx":
        if isinstance(result, dict):
            write_excel(result, save_loc)
        else:
            write_excel({default_sheet_name: result}, save_loc)


def filter_complete_df(df: pd.DataFrame, complete_only: bool) -> pd.DataFrame:
    """
    If complete_only is True and a 'completed' column exists,
    keep only rows where completed == True.
    Otherwise return df unchanged.
    """
    if not complete_only:
        return df

    if "completed" not in df.columns:
        print("  No 'completed' column found. Leaving data unchanged.")
        return df

    original_n = len(df)
    df = df[df["completed"] == True].copy()
    filtered_n = len(df)

    print(f"  Filtered to completed == True: {original_n} -> {filtered_n}")
    return df


def filter_complete_result(result, complete_only: bool):
    """
    Apply completed filtering to either:
    - a single dataframe
    - a dict of sheet_name -> dataframe
    """
    if not complete_only:
        return result

    if isinstance(result, dict):
        filtered_result = {}
        for sheet_name, df in result.items():
            print(f"Applying complete_only filter to sheet '{sheet_name}'")
            filtered_result[sheet_name] = filter_complete_df(df, complete_only=True)
        return filtered_result

    return filter_complete_df(result, complete_only=True)


def combine_xlsx_files(
    data_loc: str,
    save_loc: str,
    sheet_names: list[str],
    overwrite: bool = False,
    combine_level: str | None = None,
    complete_only: bool = False
) -> None:
    log_header("combine_xlsx", combine_level, save_loc, complete_only=complete_only)

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
    print(f"Reading sheets: {sheet_names}")

    result_dict = {sheet_name: [] for sheet_name in sheet_names}

    for file in file_list:
        print(f"Reading xlsx: {file}")
        file_data = read_excel_sheets(file, sheet_names=sheet_names)

        for sheet_name in sheet_names:
            if sheet_name not in file_data:
                print(f"  Sheet '{sheet_name}' not found in {file}. Skipping that sheet.")
                continue

            df = file_data[sheet_name]
            print(f"  Sheet '{sheet_name}' shape: {df.shape}")
            result_dict[sheet_name].append(df)

    combined_dict = {}
    for sheet_name, df_list in result_dict.items():
        if not df_list:
            print(f"No data found for sheet '{sheet_name}'.")
            continue

        combined_df = pd.concat(df_list, ignore_index=True, sort=False)
        print(f"Combined sheet '{sheet_name}' shape before filtering: {combined_df.shape}")
        combined_dict[sheet_name] = combined_df

    if not combined_dict:
        print("No sheets were combined. Skipping.")
        sys.exit(0)

    combined_dict = filter_complete_result(combined_dict, complete_only=complete_only)

    for sheet_name, df in combined_dict.items():
        print(f"Combined sheet '{sheet_name}' shape after filtering: {df.shape}")

    print(f"Writing combined output to: {save_loc}")
    write_output(
        result=combined_dict,
        save_loc=save_loc,
        default_sheet_name=sheet_names[0]
    )


def combine_rds_files(
    input_files: list[str],
    save_loc: str,
    sheet_names: list[str],
    overwrite: bool = False,
    combine_level: str | None = None,
    complete_only: bool = False
) -> None:
    log_header("combine_rds", combine_level, save_loc, complete_only=complete_only)

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

    print(f"Final combined shape before filtering: {result_df.shape}")

    result_df = filter_complete_result(result_df, complete_only=complete_only)

    print(f"Final combined shape after filtering: {result_df.shape}")
    print(f"Writing combined output to: {save_loc}")

    output_sheet_name = sheet_names[0] if sheet_names else "metadata"

    write_output(
        result=result_df,
        save_loc=save_loc,
        default_sheet_name=output_sheet_name
    )


def main():
    args = parse_args()

    if args.mode == "combine_xlsx":
        if args.data_loc is None:
            raise ValueError("--data_loc is required when --mode combine_xlsx")

        combine_xlsx_files(
            data_loc=args.data_loc,
            save_loc=args.save_loc,
            sheet_names=args.sheet_names,
            overwrite=args.overwrite,
            combine_level=args.combine_level,
            complete_only=args.complete_only
        )

    elif args.mode == "combine_rds":
        combine_rds_files(
            input_files=args.input_files,
            save_loc=args.save_loc,
            sheet_names=args.sheet_names,
            overwrite=args.overwrite,
            combine_level=args.combine_level,
            complete_only=args.complete_only
        )


if __name__ == "__main__":
    main()