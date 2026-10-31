import re
import platform
import unicodedata
import pandas as pd

from pathlib import Path
from typing import Union, List

def get_base_location():
    """
    Returns the base location of my NAS drive depending on the current operating system.
    Useful for if i want to run code on both laptops.
    """

    system = platform.system().lower()

    if system == "windows":
        base_loc = "//bc_nas_storage/BCross"
    elif system == "darwin":  # macOS reports as 'Darwin'
        base_loc = "/Volumes/BCross"
    elif system == "linux":
        base_loc = "/mnt/BCross"
    else:
        raise ValueError(f"Unsupported operating system: {system}")

    return base_loc

def create_temp_doc_id(input_text):
    """
    Create a new doc id by preprocessing the current id.

    If [...] exists, use what's inside; otherwise use the full string.

    Unicode is normalised to NFC so accented characters have a
    consistent representation.
    """

    if input_text is None or (
        isinstance(input_text, float)
        and pd.isna(input_text)
    ):
        return None

    # --------------------------------------------------------
    # Convert to string and normalise Unicode
    # --------------------------------------------------------

    s = unicodedata.normalize(
        "NFC",
        str(input_text)
    )

    # --------------------------------------------------------
    # Extract text inside square brackets if present
    # --------------------------------------------------------

    match = re.search(
        r'\[(.*?)\]',
        s
    )

    extracted_text = (
        match.group(1)
        if match
        else s
    )

    # --------------------------------------------------------
    # Clean document ID
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r'[^\w]+',
        '_',
        extracted_text.strip()
    )

    cleaned_text = re.sub(
        r'_{2,}',
        '_',
        cleaned_text
    )

    cleaned_text = re.sub(
        r'^_+|_+$',
        '',
        cleaned_text
    )

    # --------------------------------------------------------
    # Normalise final ID as well
    # --------------------------------------------------------

    cleaned_text = unicodedata.normalize(
        "NFC",
        cleaned_text
    )

    return (
        cleaned_text.lower()
        if cleaned_text
        else None
    )


def apply_temp_doc_id(df):
    """
    Apply the temporary doc ID function to the dataframe safely.
    """

    df = df.copy()

    # --------------------------------------------------------
    # If both already exist, normalise the existing doc_id
    # --------------------------------------------------------

    if (
        'doc_id' in df.columns
        and 'orig_doc_id' in df.columns
    ):

        df['doc_id'] = (
            df['doc_id']
            .apply(
                lambda x:
                unicodedata.normalize("NFC", str(x))
                if pd.notna(x)
                else None
            )
        )

        return df

    # --------------------------------------------------------
    # Determine source column for original ID
    # --------------------------------------------------------

    if 'orig_doc_id' in df.columns:

        src = 'orig_doc_id'

    elif 'doc_id' in df.columns:

        df = df.rename(
            columns={
                'doc_id': 'orig_doc_id'
            }
        )

        src = 'orig_doc_id'

    else:

        raise KeyError(
            "Expected a 'doc_id' or 'orig_doc_id' column in df."
        )

    # --------------------------------------------------------
    # Create temporary document ID
    # --------------------------------------------------------

    df['doc_id'] = (
        df[src]
        .apply(create_temp_doc_id)
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    front = (
        ['doc_id', 'orig_doc_id']
        if 'orig_doc_id' in df.columns
        else ['doc_id', src]
    )

    text_cols = (
        ['text']
        if 'text' in df.columns
        else []
    )

    rest = [
        c
        for c in df.columns
        if c not in set(
            front + text_cols
        )
    ]

    cols = (
        front
        + rest
        + text_cols
    )

    return df[cols]

def build_metadata_df(filtered_metadata: pd.DataFrame,
                      known_df: pd.DataFrame,
                      unknown_df: pd.DataFrame) -> pd.DataFrame:
    """
    From filtered_metadata (with columns problem, corpus, known_author, unknown_author)
    and known_df (with columns author, doc_id), build a metadata table exploded so that
    each known_doc_id gets its own row, and assign a running sample_id.
    """
    # Step 1: build the initial DataFrame with a list-column
    records = []
    for _, met in filtered_metadata.iterrows():
        problem        = met['problem']
        corpus         = met['corpus']
        known_author   = met['known_author']
        unknown_author = met['unknown_author']

        # collect all doc_ids for this author
        doc_ids = known_df.loc[
            known_df['author'] == known_author,
            'doc_id'
        ].unique().tolist()

        unknown_doc_id = unknown_df.loc[
            unknown_df['author'] == unknown_author,
            'doc_id'
        ].iloc[0]
        
        records.append({
            'problem':        problem,
            'corpus':         corpus,
            'known_author':   known_author,
            'unknown_author': unknown_author,
            'unknown_doc_id': unknown_doc_id,
            'known_doc_ids':  doc_ids
        })

    meta = pd.DataFrame(records)

    # Step 2: explode the list-column into individual rows
    exploded = (
        meta
        .explode('known_doc_ids')
        .rename(columns={'known_doc_ids': 'known_doc_id'})
        .reset_index(drop=True)
    )

    # Step 3: add sample_id starting at 1
    exploded.insert(0, 'sample_id', pd.RangeIndex(start=1, stop=len(exploded) + 1))

    return exploded

def list_xlsx_files(
    directory: Union[str, Path],
    *,
    recursive: bool = False,
    include_temp: bool = False,
    sort: bool = True,
) -> List[Path]:
    """
    Return all .xlsx files in a directory as a list of Paths.

    Parameters
    ----------
    directory : str | Path
        Directory to search.
    recursive : bool
        If True, search subdirectories too.
    include_temp : bool
        If True, include Excel temp files like "~$something.xlsx".
    sort : bool
        If True, sort results by path.

    Returns
    -------
    List[Path]
        Paths to .xlsx files.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    files = list(directory.glob(pattern))

    if not include_temp:
        files = [p for p in files if not p.name.startswith("~$")]

    if sort:
        files = sorted(files)

    return files

def _get_corpus_expected_files():
    """Function which returns the expected number of files by corpus and data type"""
    
    expected_by_corpus = {
        ("ACL", "training"): 186,
        ("ACL", "test"): 280,
        # NOTE - Got no value for these
        ("All-the-news", "training"): 0,
        ("All-the-news", "test"): 0,
        ("Amazon", "training"): 6400,
        ("Amazon", "test"): 9600,
        ("Enron", "training"): 224,
        ("Enron", "test"): 340,
        ("Enron Cleaned", "training"): 204,
        ("Enron Cleaned", "test"): 318,
        # NOTE - Need value for these
        ("IMDB", "training"): 0,
        ("IMDB", "test"): 0,
        ("Koppel's Blogs", "training"): 3600,
        ("Koppel's Blogs", "test"): 5400,
        ("Perverted Justice", "training"): 380,
        ("Perverted Justice", "test"): 574,
        ("Reddit", "training"): 2400,
        ("Reddit", "test"): 3600,
        ("StackExchange", "training"): 150,
        ("StackExchange", "test"): 228,
        ("The Apricity", "training"): 900,
        ("The Apricity", "test"): 1322,
        ("The Telegraph", "training"): 440,
        ("The Telegraph", "test"): 664,
        ("TripAdvisor", "training"): 120,
        ("TripAdvisor", "test"): 480,
        ("Wiki", "training"): 450,
        ("Wiki", "test"): 672,
        ("Yelp", "training"): 1600,
        ("Yelp", "test"): 2400
    }
    
    return expected_by_corpus