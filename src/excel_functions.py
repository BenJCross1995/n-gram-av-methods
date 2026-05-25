import pandas as pd

from pathlib import Path


def _existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return columns from cols that exist in df."""
    return [c for c in cols if c in df.columns]


def create_per_occurrence_table(known=None, unknown=None, no_context=None):
    """Create a base score summary table for the data.

    known and unknown are optional, but at least one of them must be provided.
    no_context is also optional.

    If known/unknown contain score_time_seconds, these are also carried into
    the phrase occurrence table as known_score_time_seconds and/or
    unknown_score_time_seconds. They are not carried for no_context because
    the main timing use case is contextual scoring.
    """

    cols = ['phrase_num', 'phrase_occurrence', 'phrase', 'tokens', 'num_tokens']
    key_cols = ['phrase_num', 'phrase_occurrence', 'phrase']
    join_cols = ['phrase_num', 'phrase_occurrence']
    score_keep_cols = ['sum_log_probs']
    contextual_keep_cols = ['sum_log_probs', 'score_time_seconds']

    # Build base table from whichever of known/unknown are provided
    base_parts = []
    if known is not None:
        base_parts.append(known[cols])
    if unknown is not None:
        base_parts.append(unknown[cols])

    if not base_parts:
        raise ValueError("At least one of known or unknown must be provided.")

    base_score_table = (
        pd.concat(base_parts, ignore_index=True)
        .drop_duplicates(subset=key_cols, keep='first')
        .sort_values(key_cols, ascending=[True, True, True])
        .reset_index(drop=True)
    )

    # Merge optional score columns
    if no_context is not None:
        keep_cols = _existing_cols(no_context, score_keep_cols)
        if keep_cols:
            no_context_small = no_context[['phrase_num'] + keep_cols].rename(
                columns={c: f"no_context_{c}" for c in keep_cols}
            )
            base_score_table = base_score_table.merge(no_context_small, on='phrase_num', how='left')

    if known is not None:
        keep_cols = _existing_cols(known, contextual_keep_cols)
        if keep_cols:
            known_small = known[join_cols + keep_cols].rename(
                columns={c: f"known_{c}" for c in keep_cols}
            )
            base_score_table = base_score_table.merge(known_small, on=join_cols, how='left')

    if unknown is not None:
        keep_cols = _existing_cols(unknown, contextual_keep_cols)
        if keep_cols:
            unknown_small = unknown[join_cols + keep_cols].rename(
                columns={c: f"unknown_{c}" for c in keep_cols}
            )
            base_score_table = base_score_table.merge(unknown_small, on=join_cols, how='left')

    return base_score_table


def create_per_phrase_table(per_occurrence_table: pd.DataFrame) -> pd.DataFrame:
    """Create an aggregated version of the table which contains the value per phrase.

    Numeric values are averaged over occurrences at phrase level. Timing columns
    are kept here for inspection only; threshold-level timing summaries are
    calculated separately from the occurrence-level known/unknown data.
    """

    # Desired output key column order (same as your original function)
    key_cols = ['phrase_num', 'phrase', 'tokens', 'num_tokens']
    skip_cols = ['phrase_occurrence']

    # Group keys must exclude list-typed 'tokens'
    group_keys = ['phrase_num', 'phrase', 'num_tokens']

    # Average everything that's not a key or skipped (preserve original column order)
    avg_cols = [
        c for c in per_occurrence_table.columns
        if c not in set(key_cols + skip_cols)
    ]

    # 1) aggregate without tokens
    per_phrase_table = (
        per_occurrence_table
        .groupby(group_keys, as_index=False)[avg_cols]
        .mean(numeric_only=True)
    )

    # 2) bring tokens back (guaranteed 1:1 by your assumption)
    tokens_map = (
        per_occurrence_table[['phrase_num', 'phrase', 'tokens']]
        .drop_duplicates(subset=['phrase_num', 'phrase'])
    )

    per_phrase_table = per_phrase_table.merge(tokens_map, on=['phrase_num', 'phrase'], how='left')

    # 3) reorder columns to match your original output order:
    # key_cols first, then averaged columns (in original order)
    per_phrase_table = per_phrase_table[key_cols + avg_cols]

    return per_phrase_table


def create_summary_by_token_num(per_phrase_table: pd.DataFrame) -> pd.DataFrame:
    """
    For each token threshold t (2,3,...) compute column-wise sums over rows where
    num_tokens >= t, for all numeric score columns not in skip_cols.

    Timing columns are deliberately excluded here because they need sum/mean/median
    aggregation and are calculated separately by create_timing_summary_by_token_num().
    """
    key_col = 'num_tokens'
    skip_cols = ['phrase_num', 'phrase', 'tokens']

    # numeric cols to sum (exclude identifiers and timing columns)
    sum_cols = [
        c for c in per_phrase_table.columns
        if c not in set(skip_cols + [key_col])
        and not c.endswith('score_time_seconds')
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


def create_timing_summary_by_token_num(
    known: pd.DataFrame | None = None,
    unknown: pd.DataFrame | None = None,
    token_thresholds: list[int] | None = None,
    time_col: str = 'score_time_seconds',
) -> pd.DataFrame:
    """
    For each token threshold t, calculate timing summaries over occurrence-level
    rows where num_tokens >= t.

    Output columns follow the same prefix style as the score columns, e.g.:
        known_sum_score_time_seconds
        known_mean_score_time_seconds
        known_median_score_time_seconds
        unknown_sum_score_time_seconds
        unknown_mean_score_time_seconds
        unknown_median_score_time_seconds

    Only known/unknown tables that exist and contain score_time_seconds are used.
    """
    sources = []
    if known is not None and {'num_tokens', time_col}.issubset(known.columns):
        sources.append(('known', known))
    if unknown is not None and {'num_tokens', time_col}.issubset(unknown.columns):
        sources.append(('unknown', unknown))

    if not sources:
        return pd.DataFrame()

    if token_thresholds is None:
        thresholds = sorted(
            set().union(*[
                set(df['num_tokens'].dropna().astype(int).tolist())
                for _, df in sources
            ])
        )
    else:
        thresholds = [int(t) for t in token_thresholds]

    rows = []
    for t in thresholds:
        row = {'min_token_size': int(t)}

        for prefix, df in sources:
            filt = df[df['num_tokens'] >= t]
            vals = pd.to_numeric(filt[time_col], errors='coerce').dropna()

            row[f'{prefix}_sum_{time_col}'] = float(vals.sum()) if len(vals) else 0.0
            row[f'{prefix}_mean_{time_col}'] = float(vals.mean()) if len(vals) else 0.0
            row[f'{prefix}_median_{time_col}'] = float(vals.median()) if len(vals) else 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values('min_token_size').reset_index(drop=True)


def add_timing_summary_by_token_num(
    score_by_token_num: pd.DataFrame,
    known: pd.DataFrame | None = None,
    unknown: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add known/unknown timing summaries to the score-by-token table when timing
    columns are available.
    """
    if score_by_token_num.empty or 'min_token_size' not in score_by_token_num.columns:
        return score_by_token_num

    timing_summary = create_timing_summary_by_token_num(
        known=known,
        unknown=unknown,
        token_thresholds=score_by_token_num['min_token_size'].dropna().astype(int).tolist(),
    )

    if timing_summary.empty:
        return score_by_token_num

    return score_by_token_num.merge(timing_summary, on='min_token_size', how='left')


def create_final_metadata(metadata: pd.DataFrame, score_by_token_num: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate each metadata row the same number of times as there are rows in score_by_token_num,
    and attach the score_by_token_num rows to each replicated block.
    """
    n = len(score_by_token_num)
    if n == 0:
        # nothing to attach; return empty with combined columns
        return metadata.iloc[0:0].assign(**{c: pd.Series(dtype=score_by_token_num[c].dtype)
                                           for c in score_by_token_num.columns})

    # repeat metadata rows (each row becomes a block of n rows)
    meta_rep = metadata.loc[metadata.index.repeat(n)].reset_index(drop=True)

    # tile score rows to match metadata repetition
    score_rep = pd.concat([score_by_token_num] * len(metadata), ignore_index=True)

    return pd.concat([meta_rep, score_rep.reset_index(drop=True)], axis=1)


def create_excel_template(
    known: pd.DataFrame | None = None,
    unknown: pd.DataFrame | None = None,
    no_context: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    docs: pd.DataFrame | None = None,
    path: str | Path = "template.xlsx",
) -> None:
    """
    Writes all sheets, builds phrase/occurrence and phrase-level score tables,
    adds threshold-level score summaries, and appends contextual timing summaries
    when known/unknown contain score_time_seconds.

    known and unknown are optional, but at least one must be provided.
    """
    path = Path(path)

    if known is None and unknown is None:
        raise ValueError("At least one of known or unknown must be provided.")

    # Create base scoring table of scores per phrase and occurrence
    per_occurrence_df = create_per_occurrence_table(known, unknown, no_context)

    # Now aggregate to per phrase level
    per_phrase_df = create_per_phrase_table(per_occurrence_df)

    # Now summarise at the num tokens level
    summary_by_num_tokens_df = create_summary_by_token_num(per_phrase_df)

    # Add timing summaries to the score-by-token table if score_time_seconds exists
    summary_by_num_tokens_df = add_timing_summary_by_token_num(
        summary_by_num_tokens_df,
        known=known,
        unknown=unknown,
    )

    # Now create the final metadata table which is a replication of metadata with token level scores
    final_metadata = (
        create_final_metadata(metadata, summary_by_num_tokens_df)
        if metadata is not None
        else summary_by_num_tokens_df.copy()
    )

    # Choose writer mode safely
    writer_mode = "a" if path.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": writer_mode}
    if writer_mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(path, **writer_kwargs) as writer:
        # Write sheets only when provided / available
        if docs is not None:
            docs.to_excel(writer, index=False, sheet_name="docs")
        if known is not None:
            known.to_excel(writer, index=False, sheet_name="known")
        if unknown is not None:
            unknown.to_excel(writer, index=False, sheet_name="unknown")
        if no_context is not None:
            no_context.to_excel(writer, index=False, sheet_name="no context")

        per_occurrence_df.to_excel(writer, index=False, sheet_name="phrase occurrence score")
        per_phrase_df.to_excel(writer, index=False, sheet_name="phrase score")
        summary_by_num_tokens_df.to_excel(writer, index=False, sheet_name="score by tokens")
        final_metadata.to_excel(writer, index=False, sheet_name="metadata")
