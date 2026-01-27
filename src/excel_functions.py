import pandas as pd

from pathlib import Path

def create_per_occurrence_table(known, unknown, no_context):
    """Create a base score summary table for the data"""
    
    # Get the base LLR table
    cols = ['phrase_num', 'phrase_occurrence', 'phrase', 'tokens', 'num_tokens']
    key_cols = ['phrase_num', 'phrase_occurrence', 'phrase']
    base_score_table = (
        pd.concat([known[cols], unknown[cols]], ignore_index=True)
        .drop_duplicates(subset=key_cols, keep='first')
        .sort_values(cols, ascending=[True, True, True, True, True])
        .reset_index(drop=True)
    )
    
    # ---- Add: bring in logprob cols from known + unknown with prefixes ----
    join_cols = ['phrase_num', 'phrase_occurrence']
    keep_cols = ['sum_log_probs']

    known_small = known[join_cols + keep_cols].rename(columns={c: f"known_{c}" for c in keep_cols})
    unknown_small = unknown[join_cols + keep_cols].rename(columns={c: f"unknown_{c}" for c in keep_cols})
    no_context_small = no_context[['phrase_num'] + keep_cols].rename(columns={c: f"no_context_{c}" for c in keep_cols})

    base_score_table = (
        base_score_table
        .merge(no_context_small, on='phrase_num', how='left')
        .merge(known_small, on=join_cols, how='left')
        .merge(unknown_small, on=join_cols, how='left')
    )

    return base_score_table

def create_per_phrase_table(per_occurrence_table):
    """Create an aggregated version of the table which contains the value per phrase occurrence"""
    
    key_cols = ['phrase_num', 'phrase', 'tokens', 'num_tokens']
    skip_cols = ['phrase_occurrence']
    
    # average everything that's not a key or skipped (only numeric columns will actually be averaged)
    avg_cols = [c for c in per_occurrence_table.columns if c not in set(key_cols + skip_cols)]

    per_phrase_table = (
        per_occurrence_table
        .groupby(key_cols, as_index=False)[avg_cols]
        .mean(numeric_only=True)
    )

    return per_phrase_table

def create_summary_by_token_num(per_phrase_table: pd.DataFrame) -> pd.DataFrame:
    """
    For each token threshold t (2,3,...) compute column-wise sums over rows where
    num_tokens >= t, for all numeric columns not in skip_cols.
    Returns one row per threshold with a summed score for each column.
    """
    key_col   = 'num_tokens'
    skip_cols = ['phrase_num', 'phrase', 'tokens']

    # numeric cols to sum (exclude identifiers)
    sum_cols = [
        c for c in per_phrase_table.columns
        if c not in set(skip_cols + [key_col])
        and pd.api.types.is_numeric_dtype(per_phrase_table[c])
    ]

    token_thresholds = sorted(per_phrase_table[key_col].dropna().unique())

    rows = []
    for t in token_thresholds:
        filt = per_phrase_table[per_phrase_table[key_col] >= t]
        sums = filt[sum_cols].sum(numeric_only=True)

        row = {'min_token_size': int(t), 'n_rows': int(len(filt))}
        row.update(sums.to_dict())
        rows.append(row)

    return pd.DataFrame(rows).sort_values('min_token_size').reset_index(drop=True)

def create_excel_template(
    known: pd.DataFrame,
    unknown: pd.DataFrame,
    no_context: pd.DataFrame,
    metadata: pd.DataFrame,
    docs: pd.DataFrame,
    path: str | Path = "template.xlsx",
    known_sheet: str = "known",
    unknown_sheet: str = "unknown",
    nc_sheet: str = "no context",
    metadata_sheet: str = "metadata",
    docs_sheet: str = "docs",
    phrase_score_sheet: str = "scores"
) -> None:
    """
    Writes all sheets, builds a distinct phrases 'LLR' table, adds include_phrase lookups
    to Known & Unknown, and then adds your LLR formulas (D..H).
    """
    path = Path(path)

    # Create base scoring table of scores per phrase and occurrence
    per_occurrence_df = create_per_occurrence_table(known, unknown, no_context)
    
    # Now aggregate to per phrase level
    per_phrase_df = create_per_phrase_table(per_occurrence_df)
    
    # Now summarise at the num tokens level
    summary_by_num_tokens_df = create_summary_by_token_num(per_phrase_df)
    
    # Choose writer mode safely
    writer_mode = "a" if path.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": writer_mode}
    if writer_mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"  # only valid in append mode
        
    with pd.ExcelWriter(path, **writer_kwargs) as writer:
        # Write sheets
        docs.to_excel(writer, index=False, sheet_name=docs_sheet)
        known.to_excel(writer, index=False, sheet_name=known_sheet)
        unknown.to_excel(writer, index=False, sheet_name=unknown_sheet)
        no_context.to_excel(writer, index=False, sheet_name=nc_sheet)
        distinct_phrases.to_excel(writer, index=False, sheet_name=phrase_score_sheet)
        metadata.to_excel(writer, index=False, sheet_name=metadata_sheet)