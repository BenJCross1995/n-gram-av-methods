# -*- coding: utf-8 -*-
"""
Data reading and writing utility functions.

"""
import json
import pyreadr

import pandas as pd

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

def read_rds(file_path):
    """Reads an RDS file and converts it into a pandas DataFrame."""
    objects = pyreadr.read_r(file_path)  # dict-like: {object_name: object}

    # If the file contains a single object, just return that object
    if len(objects) == 1:
        return next(iter(objects.values()))

    # Otherwise, return the whole dict (could be multiple data frames, models, etc.)
    return dict(objects)

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