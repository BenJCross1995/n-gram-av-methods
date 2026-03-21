#!/usr/bin/env python3
import argparse
import sys
import os

import pandas as pd

from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_excel_sheets, write_rds
from utils import list_xlsx_files

def parse_args():
    ap = argparse.ArgumentParser(description="Script to combine all data in location")
    ap.add_argument("--data_loc")
    ap.add_argument("--save_loc")
    return ap.parse_args()

def main():
    
    args=parse_args()
    
    # Skip if output already exists
    if os.path.exists(args.save_loc):
        print(f"Path {args.save_loc} already exists. Exiting.")
        sys.exit(0)

    # Skip if input directory does not exist
    if not os.path.isdir(args.data_loc):
        print(f"Input directory does not exist: {args.data_loc}. Skipping.")
        sys.exit(0)
        
    # Get all of the excel files in the data location
    file_list = list_xlsx_files(args.data_loc)

    # Skip if no files found
    if not file_list:
        print(f"No .xlsx files found in: {args.data_loc}. Skipping.")
        sys.exit(0)
        
    # Loop through files, combine and write
    result_list = []
    for file in file_list:
        print(f"Reading: {file}")
        metadata = read_excel_sheets(file, sheet_names=["metadata"])["metadata"]
        result_list.append(metadata)

    result_df = pd.concat(result_list, ignore_index=True)

    print(f"Writing combined file to: {args.save_loc}")
    write_rds(result_df, args.save_loc)

if __name__ == "__main__":
    main()