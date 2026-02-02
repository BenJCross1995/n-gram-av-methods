# -*- coding: utf-8 -*-
"""
Data reading and writing utility functions.

"""
import json
import pyreadr

import pandas as pd

from pathlib import Path
from typing import Iterable, Dict, Optional

def read_jsonl(file_path):
    """Reads a JSONL file and converts it into a pandas DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parsed_line = json.loads(line)
            if isinstance(parsed_line, list) and len(parsed_line) == 1:
                data.append(parsed_line[0])
            else:
                data.append(parsed_line)
                
    return pd.DataFrame(data)

def _strip_rownames(x):
    """Remove pyreadr/R rownames artifacts from a loaded object."""
    if not isinstance(x, pd.DataFrame):
        return x

    df = x

    # Case 1: rownames came in as a real column
    if "rownames" in df.columns:
        df = df.drop(columns=["rownames"])

    # Case 2: rownames are stored in the index
    if getattr(df.index, "name", None) == "rownames":
        df = df.reset_index(drop=True)

    # Even if index isn't named, you may still want a clean RangeIndex
    # (safe + consistent for your pipeline)
    df.index.name = None
    df = df.reset_index(drop=True)

    return df

def read_rds(file_path):
    """Reads an RDS file and converts it into pandas objects, stripping rownames."""
    objects = pyreadr.read_r(file_path)  # dict-like: {object_name: object}

    if len(objects) == 1:
        return _strip_rownames(next(iter(objects.values())))

    return {k: _strip_rownames(v) for k, v in objects.items()}


def read_txt(file_path):
    """Read a .txt file and return its contents as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def write_jsonl(data, output_file_path):
    """Writes a pandas DataFrame to a JSONL file."""
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for _, row in data.iterrows():
            json.dump(row.to_dict(), file)
            file.write('\n')

def write_rds(obj, file_path):
    """
    Write an object to an RDS file using pyreadr.

    Notes
    -----
    - .rds files store a single R object.
    - If you pass a dict of objects, this function writes one .rds per key by
      appending the key to the filename.
    """
    file_path = Path(file_path)

    # If a dict is provided, write each object to its own .rds file
    if isinstance(obj, dict):
        out_paths = {}
        stem = file_path.stem
        suffix = file_path.suffix or ".rds"
        parent = file_path.parent

        for name, val in obj.items():
            out_fp = parent / f"{stem}_{name}{suffix}"
            pyreadr.write_rds(str(out_fp), val)
            out_paths[name] = str(out_fp)

        return out_paths  # helpful if you want to know where each file went

    # Single object -> single .rds
    pyreadr.write_rds(str(file_path), obj)
          
def read_excel_sheets(
    file_path: str | Path,
    sheet_names: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Read sheets from an Excel file into a dict of DataFrames.

    Parameters:
    - file_path: Path to the Excel file.
    - sheet_names: Names of the sheets you want to load. If None or empty, all sheets are read.

    Returns:
    - Dictionary where each key is a sheet name and each value is the corresponding DataFrame.
    """
    # If no sheet names provided -> load all sheets
    if sheet_names is None:
        return pd.read_excel(file_path, sheet_name=None)

    # Convert to list so we can check if it's empty and reuse it
    sheet_names = list(sheet_names)
    if len(sheet_names) == 0:
        return pd.read_excel(file_path, sheet_name=None)

    # Load workbook structure once
    xls = pd.ExcelFile(file_path)

    missing = [s for s in sheet_names if s not in xls.sheet_names]
    if missing:
        raise ValueError(
            f"These sheets were not found in {file_path}: {missing}. "
            f"Available sheets: {xls.sheet_names}"
        )

    return {
        name: pd.read_excel(xls, sheet_name=name)
        for name in sheet_names
    }