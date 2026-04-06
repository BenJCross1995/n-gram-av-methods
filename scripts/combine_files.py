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
  
Additional optional Excel functionality is available through:
- --metadata_columns
- --metadata_sheet_name

When --metadata_columns is supplied in combine_xlsx mode and the requested
sheets include both the metadata sheet and one or more other sheets:
- the specified metadata columns are taken from the metadata sheet
- deduplicated to a single row per file
- replicated to match the number of rows in each non-metadata sheet
- prepended to the left of that non-metadata sheet
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

    # New optional metadata-merge functionality for combine_xlsx
    ap.add_argument("--metadata_sheet_name", type=str, default="metadata")
    ap.add_argument("--metadata_columns", nargs="*", default=None)
    
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

def should_skip_file_from_metadata(
    metadata_df: pd.DataFrame | None,
    complete_only: bool,
    file_path: str
) -> bool:
    """
    Decide whether an Excel file should be skipped based on the metadata sheet.

    If complete_only is True and the metadata sheet contains a 'completed'
    column, the whole file is skipped when there are no completed rows.

    Parameters
    ----------
    metadata_df : pd.DataFrame | None
        Metadata dataframe for the current file.
    complete_only : bool
        Whether completed-only filtering is enabled.
    file_path : str
        Current file path, used for logging.

    Returns
    -------
    bool
        True if the file should be skipped, otherwise False.
    """
    if not complete_only:
        return False

    if metadata_df is None:
        return False

    if "completed" not in metadata_df.columns:
        return False

    completed_metadata = metadata_df[metadata_df["completed"] == True]
    if completed_metadata.empty:
        print(f"Skipping file because metadata.completed != True: {file_path}")
        return True

    return False

def extract_metadata_row(
    metadata_df: pd.DataFrame,
    metadata_columns: list[str],
    file_path: str
) -> pd.DataFrame:
    """
    Extract a single deduplicated metadata row from the metadata sheet.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata dataframe for one Excel file.
    metadata_columns : list[str]
        Columns to select from metadata.
    file_path : str
        Current file path, used for clearer error messages.

    Returns
    -------
    pd.DataFrame
        A one-row dataframe containing the requested metadata columns.

    Raises
    ------
    KeyError
        If any requested metadata columns are missing.
    ValueError
        If no metadata rows remain after selection/deduplication.
    """
    missing_cols = [col for col in metadata_columns if col not in metadata_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing metadata columns in {file_path}: {missing_cols}"
        )

    metadata_row = metadata_df[metadata_columns].drop_duplicates().reset_index(drop=True)

    if metadata_row.empty:
        raise ValueError(
            f"No metadata rows available after deduplication in {file_path}"
        )

    if metadata_row.shape[0] > 1:
        print(
            f"Warning: metadata deduplication in {file_path} produced "
            f"{metadata_row.shape[0]} rows. Using the first row."
        )

    return metadata_row.iloc[[0]].copy()

def prepend_metadata_to_sheet(
    sheet_df: pd.DataFrame,
    metadata_row: pd.DataFrame
) -> pd.DataFrame:
    """
    Replicate a single metadata row to match a sheet and prepend it.

    If the target sheet already contains any of the metadata columns,
    those columns are removed from the target sheet before concatenation
    so the metadata columns appear only once on the left.

    Parameters
    ----------
    sheet_df : pd.DataFrame
        Target non-metadata sheet dataframe.
    metadata_row : pd.DataFrame
        One-row metadata dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with replicated metadata columns prepended.
    """
    if sheet_df.empty:
        # preserve column order even for empty frames
        target_df = sheet_df.drop(columns=metadata_row.columns, errors="ignore").copy()
        empty_metadata = pd.DataFrame(columns=metadata_row.columns)
        return pd.concat(
            [empty_metadata.reset_index(drop=True), target_df.reset_index(drop=True)],
            axis=1
        )

    target_df = sheet_df.drop(columns=metadata_row.columns, errors="ignore").reset_index(drop=True)

    metadata_repeated = pd.DataFrame({
        col: [metadata_row.iloc[0][col]] * len(target_df)
        for col in metadata_row.columns
    })

    return pd.concat(
        [metadata_repeated.reset_index(drop=True), target_df],
        axis=1
    )

def combine_xlsx_files(
    data_loc: str,
    save_loc: str,
    sheet_names: list[str],
    overwrite: bool = False,
    combine_level: str | None = None,
    complete_only: bool = False,
    metadata_sheet_name: str = "metadata",
    metadata_columns: list[str] | None = None
) -> None:
    """
    Combine requested sheets from all .xlsx files in a directory.

    For each .xlsx file in data_loc:
    - read the requested sheet_names
    - collect matching sheets across files
    - concatenate each sheet independently
    - optionally filter to completed rows only
    - write the combined result to save_loc

    Optional metadata expansion:
    - if metadata_columns is supplied and the metadata sheet is present,
      each non-metadata requested sheet gets those metadata columns
      prepended from the metadata sheet of the same file

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
    metadata_sheet_name : str, default='metadata'
        Name of the metadata sheet.
    metadata_columns : list[str] | None, default=None
        Metadata columns to extract, deduplicate, replicate, and prepend
        to each requested non-metadata sheet.
    """
    log_header("combine_xlsx", combine_level, save_loc, complete_only=complete_only)

    if should_skip_output(save_loc, overwrite):
        sys.exit(0)

    if not os.path.isdir(data_loc):
        print(f"Input directory does not exist, skipping: {data_loc}")
        sys.exit(0)

    if metadata_columns and metadata_sheet_name not in sheet_names:
        raise ValueError(
            "--metadata_columns was provided but the metadata sheet "
            f"'{metadata_sheet_name}' is not included in --sheet_names."
        )

    # Discover input Excel files
    file_list = list_xlsx_files(data_loc)

    if not file_list:
        print(f"No .xlsx files found, skipping: {data_loc}")
        sys.exit(0)

    # Store dataframes separately for each requested sheet
    result_dict = {sheet_name: [] for sheet_name in sheet_names}

    for file in file_list:
        file_data = read_excel_sheets(file, sheet_names=sheet_names)

        if not file_data:
            continue

        metadata_df = file_data.get(metadata_sheet_name)

        # File-level skip based on metadata.completed
        if should_skip_file_from_metadata(
            metadata_df=metadata_df,
            complete_only=complete_only,
            file_path=file
        ):
            continue

        metadata_row = None
        if metadata_columns:
            if metadata_df is None:
                print(
                    f"Warning: metadata sheet '{metadata_sheet_name}' not found "
                    f"in {file}. Skipping metadata prepend for this file."
                )
            else:
                metadata_row = extract_metadata_row(
                    metadata_df=metadata_df,
                    metadata_columns=metadata_columns,
                    file_path=file
                )

        # Append sheet data
        for sheet_name in sheet_names:
            if sheet_name not in file_data:
                continue

            current_df = file_data[sheet_name]

            # Keep metadata sheet as-is
            if sheet_name == metadata_sheet_name:
                result_dict[sheet_name].append(current_df)
                continue

            # Optionally prepend replicated metadata
            if metadata_row is not None:
                current_df = prepend_metadata_to_sheet(
                    sheet_df=current_df,
                    metadata_row=metadata_row
                )

            result_dict[sheet_name].append(current_df)

    # Concatenate each collected sheet independently
    combined_dict = {}
    for sheet_name, df_list in result_dict.items():
        if df_list:
            combined_dict[sheet_name] = pd.concat(
                df_list,
                ignore_index=True,
                sort=False
            )
            
    # If metadata was only used to expand other sheets and the output is .rds,
    # do not try to write the metadata sheet as well.
    output_file_type = get_file_type(save_loc)

    if (
        output_file_type == ".rds"
        and metadata_columns
        and metadata_sheet_name in combined_dict
        and len(combined_dict) > 1
    ):
        combined_dict = {
            sheet_name: df
            for sheet_name, df in combined_dict.items()
            if sheet_name != metadata_sheet_name
        }
        
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
            complete_only=args.complete_only,
            metadata_sheet_name=args.metadata_sheet_name,
            metadata_columns=args.metadata_columns
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