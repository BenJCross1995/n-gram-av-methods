#!/usr/bin/env python3
"""
Combine Excel or RDS files into a single output file.

This script supports two modes:

1. combine_xlsx
   - Reads all .xlsx files in a directory
   - Reads one or more requested sheet names from each file
   - Concatenates matching sheets across files
   - Writes the result to either .xlsx or .rds depending on save_loc

2. combine_rds
   - Reads a provided list of .rds files
   - Concatenates them into a single dataframe
   - Writes the result to either .xlsx or .rds depending on save_loc

Optional filtering is available through --complete_only:
- If enabled, and a dataframe contains a 'completed' column,
  only rows where completed == True are kept.
"""

import argparse
import sys
import os
from pathlib import Path

import pandas as pd

from from_root import from_root

# Ensure local src/ imports are available
sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_excel_sheets, write_rds, read_rds
from utils import list_xlsx_files


def parse_args():
    """
    Parse command-line arguments for the combine script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
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
    """
    Print a short one-line summary of the current run.

    Parameters
    ----------
    mode : str
        The combine mode, either 'combine_xlsx' or 'combine_rds'.
    combine_level : str | None
        Optional label describing the current combine stage.
    save_loc : str
        Output path for the combined result.
    complete_only : bool, default=False
        Whether completed-only filtering is enabled.
    """
    level_text = f"{combine_level}" if combine_level is not None else "None"
    print(
        f"Running mode={mode}, combine_level={level_text}, "
        f"complete_only={complete_only}, save_loc={save_loc}"
    )


def should_skip_output(save_loc: str, overwrite: bool) -> bool:
    """
    Decide whether to skip writing because the output already exists.

    Parameters
    ----------
    save_loc : str
        Output file path.
    overwrite : bool
        If True, existing output will be replaced.

    Returns
    -------
    bool
        True if the script should skip this output, otherwise False.
    """
    if os.path.exists(save_loc):
        if overwrite:
            print(f"Overwriting existing output: {save_loc}")
            return False
        print(f"Output already exists, skipping: {save_loc}")
        return True
    return False


def get_file_type(path: str) -> str:
    """
    Determine the supported output file type from a path extension.

    Parameters
    ----------
    path : str
        File path whose extension should be checked.

    Returns
    -------
    str
        The lowercase extension, either '.xlsx' or '.rds'.

    Raises
    ------
    ValueError
        If the extension is not supported.
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
    Create the parent directory for a file path if it does not exist.

    Parameters
    ----------
    path : str
        File path whose parent directory should be ensured.
    """
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def write_excel(data: dict[str, pd.DataFrame], save_loc: str) -> None:
    """
    Write one or more dataframes to an Excel workbook.

    Each dictionary key becomes a sheet name and each value is written
    as a dataframe. Sheet names are truncated to 31 characters to satisfy
    Excel's sheet-name limit.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping of sheet name to dataframe.
    save_loc : str
        Output Excel file path.
    """
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
    """
    Write a combined result to either .rds or .xlsx based on save_loc.

    Parameters
    ----------
    result : pd.DataFrame or dict[str, pd.DataFrame]
        Result to write. A dataframe is written directly. A dictionary is
        treated as sheet_name -> dataframe.
    save_loc : str
        Output path. Must end in .rds or .xlsx.
    default_sheet_name : str, default='metadata'
        Sheet name to use when writing a single dataframe to Excel.

    Raises
    ------
    ValueError
        If attempting to write multiple sheets to a single .rds file.
    """
    file_type = get_file_type(save_loc)
    ensure_parent_dir(save_loc)

    if file_type == ".rds":
        # RDS can only store a single dataframe in this workflow
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
        # Excel can store either a single sheet or multiple sheets
        if isinstance(result, dict):
            write_excel(result, save_loc)
        else:
            write_excel({default_sheet_name: result}, save_loc)


def filter_complete_df(df: pd.DataFrame, complete_only: bool) -> pd.DataFrame:
    """
    Optionally filter a dataframe to rows where completed == True.

    Filtering is only applied when:
    - complete_only is True, and
    - the dataframe contains a 'completed' column

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    complete_only : bool
        Whether completed-only filtering should be applied.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe if applicable, otherwise the original dataframe.
    """
    if not complete_only:
        return df

    if "completed" not in df.columns:
        return df

    return df[df["completed"] == True].copy()


def filter_complete_result(result, complete_only: bool):
    """
    Apply completed-only filtering to a dataframe or dict of dataframes.

    Parameters
    ----------
    result : pd.DataFrame or dict[str, pd.DataFrame]
        The result object to filter.
    complete_only : bool
        Whether completed-only filtering should be applied.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
        Filtered result in the same structure as the input.
    """
    if not complete_only:
        return result

    if isinstance(result, dict):
        return {
            sheet_name: filter_complete_df(df, complete_only=True)
            for sheet_name, df in result.items()
        }

    return filter_complete_df(result, complete_only=True)


def combine_xlsx_files(
    data_loc: str,
    save_loc: str,
    sheet_names: list[str],
    overwrite: bool = False,
    combine_level: str | None = None,
    complete_only: bool = False
) -> None:
    """
    Combine requested sheets from all .xlsx files in a directory.

    For each .xlsx file in data_loc:
    - read the requested sheet_names
    - collect matching sheets across files
    - concatenate each sheet independently
    - optionally filter to completed rows only
    - write the combined result to save_loc

    Parameters
    ----------
    data_loc : str
        Directory containing input .xlsx files.
    save_loc : str
        Output file path (.xlsx or .rds).
    sheet_names : list[str]
        Sheet names to read from each Excel file.
    overwrite : bool, default=False
        If True, overwrite existing output.
    combine_level : str | None, default=None
        Optional label describing the combine stage.
    complete_only : bool, default=False
        If True, and a dataframe contains a 'completed' column,
        keep only rows where completed == True.
    """
    log_header("combine_xlsx", combine_level, save_loc, complete_only=complete_only)

    if should_skip_output(save_loc, overwrite):
        sys.exit(0)

    if not os.path.isdir(data_loc):
        print(f"Input directory does not exist, skipping: {data_loc}")
        sys.exit(0)

    # Discover input Excel files
    file_list = list_xlsx_files(data_loc)

    if not file_list:
        print(f"No .xlsx files found, skipping: {data_loc}")
        sys.exit(0)

    # Store dataframes separately for each requested sheet
    result_dict = {sheet_name: [] for sheet_name in sheet_names}

    for file in file_list:
        file_data = read_excel_sheets(file, sheet_names=sheet_names)

        # Append sheet data only when that sheet exists in the current file
        for sheet_name in sheet_names:
            if sheet_name in file_data:
                result_dict[sheet_name].append(file_data[sheet_name])

    # Concatenate each collected sheet independently
    combined_dict = {}
    for sheet_name, df_list in result_dict.items():
        if df_list:
            combined_dict[sheet_name] = pd.concat(
                df_list,
                ignore_index=True,
                sort=False
            )

    if not combined_dict:
        print(f"No requested sheets found, skipping: {data_loc}")
        sys.exit(0)

    # Optionally keep only completed rows
    combined_dict = filter_complete_result(combined_dict, complete_only=complete_only)

    # Build a short summary for logging
    summary_parts = [
        f"{sheet_name}={df.shape[0]} rows"
        for sheet_name, df in combined_dict.items()
    ]
    print(f"Combined {len(file_list)} xlsx files -> {', '.join(summary_parts)}")
    print(f"Writing: {save_loc}")

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
    """
    Combine a list of .rds files into a single output.

    The input RDS files are read, concatenated row-wise, optionally filtered
    to completed rows only, and then written to save_loc.

    Parameters
    ----------
    input_files : list[str]
        List of candidate input .rds files.
    save_loc : str
        Output file path (.xlsx or .rds).
    sheet_names : list[str]
        Used only when writing to Excel; the first sheet name is used
        as the output sheet name for a single dataframe.
    overwrite : bool, default=False
        If True, overwrite existing output.
    combine_level : str | None, default=None
        Optional label describing the combine stage.
    complete_only : bool, default=False
        If True, and the dataframe contains a 'completed' column,
        keep only rows where completed == True.
    """
    log_header("combine_rds", combine_level, save_loc, complete_only=complete_only)

    if should_skip_output(save_loc, overwrite):
        sys.exit(0)

    if not input_files:
        print("No input RDS files provided, skipping.")
        sys.exit(0)

    # Keep only input files that actually exist
    existing_files = [file for file in input_files if os.path.exists(file)]

    if not existing_files:
        print("No input RDS files exist, skipping.")
        sys.exit(0)

    # Read and concatenate all existing RDS files
    result_list = [read_rds(file) for file in existing_files]
    result_df = pd.concat(result_list, ignore_index=True, sort=False)

    # Optionally keep only completed rows
    result_df = filter_complete_result(result_df, complete_only=complete_only)

    print(f"Combined {len(existing_files)} rds files -> {result_df.shape[0]} rows")
    print(f"Writing: {save_loc}")

    output_sheet_name = sheet_names[0] if sheet_names else "metadata"

    write_output(
        result=result_df,
        save_loc=save_loc,
        default_sheet_name=output_sheet_name
    )


def main():
    """
    Entry point for the combine script.

    Dispatches to either Excel-combine mode or RDS-combine mode
    based on the --mode argument.
    """
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