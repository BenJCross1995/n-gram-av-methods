#!/usr/bin/env python3
"""Score common and non-common token n-grams for one AV problem.

Outputs two RDS data frames (common / non_common), each containing scored
unknown-document occurrences, all known-document occurrences, and context-free
scores. The `occurrence_aware` flag is a membership indicator, not a filter.

Requires the project's `src` modules and the `common_ngrams` package.
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from common_ngrams import (
    combine_ngrams,
    extract_ngrams,
    filter_len_common_ngrams,
    find_common_ngrams_in_collections,
    find_occurrence_aware_common_ngrams,
    subtract_ngrams,
)
from common_ngrams.preprocessing import preprocess_text
from model_loading import load_model
from n_gram_scoring import score_ngrams_to_df
from read_and_write_docs import read_jsonl, read_rds, write_rds
from utils import apply_temp_doc_id


PREPROCESS_OPTIONS = (
    "lowercase",
    "remove_punctuation",
    "remove_symbols",
    "remove_numeric",
    "remove_url",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score shared and unshared n-grams, saving two RDS tables."
    )
    parser.add_argument("--known_loc", required=True)
    parser.add_argument("--unknown_loc", required=True)
    parser.add_argument("--metadata_loc", required=True)
    parser.add_argument("--model_loc", required=True)
    parser.add_argument("--save_loc", required=True, help="Output directory")
    parser.add_argument("--completed_loc", default=None)
    parser.add_argument("--error_loc", default=None)

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--data_type", default="test")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument(
        "--num_tokens", type=int, default=10,
        help="Maximum context tokens for document scoring (0 allowed)",
    )
    parser.add_argument("--overwrite", action="store_true")

    # BooleanOptionalAction supports --flag and --no-flag.
    for name in PREPROCESS_OPTIONS:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            "--" + name,
            dest=name,
            action=argparse.BooleanOptionalAction,
            default=True,
        )

    args = parser.parse_args()
    args.problem = args.problem.strip().strip('"').strip("'")
    if args.min_len is not None and args.min_len < 1:
        parser.error("--min_len must be >= 1")
    if args.max_len is not None and args.max_len < args.min_len:
        parser.error("--max_len must be >= --min_len")
    if args.num_tokens < 0:
        parser.error("--num_tokens must be >= 0")
    return args


def add_metadata(scored_df, metadata):
    """Attach document metadata as leading columns (one value per row)."""
    scored_df = scored_df.copy()
    for column, value in reversed(list(metadata.items())):
        scored_df.insert(0, column, value)
    return scored_df


def score_set(ngrams, *, model, tokenizer, full_text, num_tokens,
              occurrence_aware_set, metadata):
    """Score one n-gram list against one document (or without context)."""
    if ngrams:
        options = {
            "full_text": full_text,
            "use_bos": True,
        }
        if full_text is not None:
            options.update(num_tokens=num_tokens, greatest_common=False)
        scored = score_ngrams_to_df(ngrams, model, tokenizer, **options)
        scored["occurrence_aware"] = scored["tokens"].map(
            lambda tokens: tuple(tokens) in occurrence_aware_set
        )
    else:
        scored = pd.DataFrame({"occurrence_aware": pd.Series(dtype=bool)})
    return add_metadata(scored, metadata)


def score_group(known_ngrams, unknown_ngrams, context_free_ngrams, *,
                selected_known, unknown_text, base_metadata, unknown_author,
                unknown_doc_id, model, tokenizer, num_tokens,
                occurrence_aware_set, label):
    """Combine scored unknown, known, and no-context rows for one group."""
    print(f"Scoring {label}: unknown", flush=True)
    unknown_metadata = dict(
        base_metadata,
        max_context_tokens=num_tokens,
        author=unknown_author,
        texttype="unknown",
        doc_id=unknown_doc_id,
    )
    pieces = [
        score_set(
            unknown_ngrams,
            model=model, tokenizer=tokenizer, full_text=unknown_text,
            num_tokens=num_tokens, occurrence_aware_set=occurrence_aware_set,
            metadata=unknown_metadata,
        )
    ]

    known_dfs = []
    for i, (_, row) in enumerate(selected_known.iterrows(), start=1):
        print(f"Scoring {label}: known {i}/{len(selected_known)}", flush=True)
        known_metadata = dict(
            base_metadata,
            max_context_tokens=num_tokens,
            author=row["author"],
            texttype="known",
            doc_id=row["doc_id"],
        )
        known_dfs.append(
            score_set(
                known_ngrams,
                model=model, tokenizer=tokenizer, full_text=row["text"],
                num_tokens=num_tokens, occurrence_aware_set=occurrence_aware_set,
                metadata=known_metadata,
            )
        )

    known_df = pd.concat(known_dfs, ignore_index=True)
    sort_cols = ["phrase_num", "doc_id", "phrase_occurrence"]
    if all(col in known_df.columns for col in sort_cols):
        known_df = known_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    pieces.append(known_df)

    print(f"Scoring {label}: no context", flush=True)
    no_context_metadata = dict(
        base_metadata,
        max_context_tokens=0,
        author=None,
        texttype="no_context",
        doc_id=None,
    )
    pieces.append(
        score_set(
            context_free_ngrams,
            model=model, tokenizer=tokenizer, full_text=None,
            num_tokens=0, occurrence_aware_set=occurrence_aware_set,
            metadata=no_context_metadata,
        )
    )
    return pd.concat(pieces, ignore_index=True)


def prepare_for_rds(df):
    """Convert nested Python values (notably token lists) to JSON strings.

    Ordinary R data frames cannot directly store arbitrary Python list objects
    as cells through the project's RDS writer. JSON retains token boundaries
    and can be decoded in Python/R when needed.
    """
    result = df.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        nested = result[column].map(
            lambda value: isinstance(value, (list, tuple, dict, set))
        )
        if nested.any():
            result[column] = result[column].map(
                lambda value: json.dumps(
                    sorted(value) if isinstance(value, set) else value,
                    ensure_ascii=False,
                ) if isinstance(value, (list, tuple, dict, set)) else value
            )
    return result


def run_pipeline(args):
    preprocessing = {key: getattr(args, key) for key in PREPROCESS_OPTIONS}
    output_dir = Path(args.save_loc)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "common": output_dir / f"{args.problem}_common.rds",
        "non_common": output_dir / f"{args.problem}_non_common.rds",
    }
    if args.completed_loc and all(
        (Path(args.completed_loc) / p.name).exists() for p in paths.values()
    ):
        print("Both RDS files already exist in completed_loc; skipping.")
        return
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists: " + ", ".join(map(str, existing))
            + ". Use --overwrite to regenerate both tables."
        )

    metadata = read_rds(args.metadata_loc)
    subset = metadata.loc[
        (metadata["corpus"] == args.corpus)
        & (metadata["problem"] == args.problem)
    ]
    if subset.empty:
        raise ValueError(f"Problem not found: {args.corpus!r} / {args.problem!r}")
    known_author = subset["known_author"].iloc[0]
    unknown_author = subset["unknown_author"].iloc[0]
    known = apply_temp_doc_id(read_jsonl(args.known_loc))
    unknown = apply_temp_doc_id(read_jsonl(args.unknown_loc))
    selected_known = known.loc[known["author"] == known_author].copy()
    selected_unknown = unknown.loc[unknown["author"] == unknown_author].copy()
    if selected_known.empty or selected_unknown.empty:
        raise ValueError("No known or unknown documents matched the problem authors.")
    if len(selected_unknown) > 1:
        print(
            f"WARNING: {len(selected_unknown)} unknown documents match author "
            f"{unknown_author!r}; using the first, as in the notebook.",
            flush=True,
        )

    unknown_row = selected_unknown.iloc[0]
    unknown_text = preprocess_text(unknown_row["text"], **preprocessing)
    selected_known["text"] = selected_known["text"].map(
        lambda value: preprocess_text(value, **preprocessing)
    )
    known_texts = selected_known["text"].tolist()
    print(f"Problem: {args.problem}; known texts: {len(known_texts)}", flush=True)

    print("Loading model", flush=True)
    tokenizer, model = load_model(args.model_loc)
    model_name = Path(args.model_loc).name

    # Input texts have already been preprocessed consistently. These helpers
    # normally preprocess again; disabling their defaults avoids silently
    # lowercasing texts when --no-lowercase is selected.
    import inspect

    def no_extra_preprocessing(fn):
        params = inspect.signature(fn).parameters
        return {key: False for key in PREPROCESS_OPTIONS if key in params}

    print("Finding common n-grams", flush=True)
    common_ngrams = find_common_ngrams_in_collections(
        reference_texts=known_texts,
        comparison_texts=unknown_text,
        tokenizer=tokenizer,
        include_subgrams=True,
        **no_extra_preprocessing(find_common_ngrams_in_collections),
    )
    occurrence_aware_ngrams = find_occurrence_aware_common_ngrams(
        reference_texts=known_texts,
        comparison_texts=unknown_text,
        tokenizer=tokenizer,
        **no_extra_preprocessing(find_occurrence_aware_common_ngrams),
    )
    occurrence_aware_set = {tuple(gram) for gram in occurrence_aware_ngrams}
    filtered_common = filter_len_common_ngrams(
        common_ngrams, min_len=args.min_len, max_len=args.max_len
    )
    distinct_lengths = sorted({len(gram) for gram in filtered_common})
    print(
        f"Common n-grams: {len(filtered_common)}; "
        f"occurrence-aware candidates: {len(occurrence_aware_set)}; "
        f"lengths: {distinct_lengths}", flush=True,
    )
    if not distinct_lengths:
        raise ValueError("No common n-grams at the selected lengths; no lengths to compare.")

    # Unshared n-grams have exactly the same lengths as the common candidates.
    print("Finding non-common n-grams", flush=True)
    known_all = combine_ngrams(
        extract_ngrams(
            known_texts, tokenizer=tokenizer, n=distinct_lengths,
            **no_extra_preprocessing(extract_ngrams),
        )
    )
    unknown_all = combine_ngrams(
        extract_ngrams(
            unknown_text, tokenizer=tokenizer, n=distinct_lengths,
            **no_extra_preprocessing(extract_ngrams),
        )
    )
    remaining_known = subtract_ngrams(whole=known_all, common=common_ngrams)
    remaining_unknown = subtract_ngrams(whole=unknown_all, common=common_ngrams)
    remaining_all = [
        list(gram)
        for gram in dict.fromkeys(
            tuple(gram) for gram in remaining_unknown + remaining_known
        )
    ]
    overlap = {tuple(g) for g in remaining_known} & {tuple(g) for g in remaining_unknown}
    if overlap:
        raise ValueError(
            f"{len(overlap)} n-grams remain on both sides after subtraction. "
            "Check that the installed common_ngrams package includes all "
            "contiguous subgrams and that preprocessing/tokenisation agree. "
            f"Example: {list(next(iter(overlap)))}"
        )
    print(
        f"Remaining known: {len(remaining_known)}, "
        f"remaining unknown: {len(remaining_unknown)}, "
        f"remaining distinct: {len(remaining_all)}", flush=True,
    )

    base_metadata = {
        "data_type": args.data_type,
        "corpus": args.corpus,
        "scoring_model": model_name,
        "problem": args.problem,
        "known_author": known_author,
        "unknown_author": unknown_author,
        "target": bool(known_author == unknown_author),
        **preprocessing,
    }

    # Group 1: each shared n-gram in each document + one context-free score.
    common_df = score_group(
        filtered_common, filtered_common, filtered_common,
        selected_known=selected_known,
        unknown_text=unknown_text,
        base_metadata=base_metadata,
        unknown_author=unknown_author,
        unknown_doc_id=unknown_row["doc_id"],
        model=model, tokenizer=tokenizer, num_tokens=args.num_tokens,
        occurrence_aware_set=occurrence_aware_set, label="common",
    )

    # Group 2: side-specific unshared n-grams; pooled for no-context scoring.
    non_common_df = score_group(
        remaining_known, remaining_unknown, remaining_all,
        selected_known=selected_known,
        unknown_text=unknown_text,
        base_metadata=base_metadata,
        unknown_author=unknown_author,
        unknown_doc_id=unknown_row["doc_id"],
        model=model, tokenizer=tokenizer, num_tokens=args.num_tokens,
        occurrence_aware_set=occurrence_aware_set, label="non-common",
    )

    for key, frame in (("common", common_df), ("non_common", non_common_df)):
        path = paths[key]
        write_rds(prepare_for_rds(frame), str(path))
        print(f"Saved {len(frame):,} rows to {path}", flush=True)


def main():
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"ERROR processing {args.problem}:\n{tb}", file=sys.stderr)
        if args.error_loc:
            error_dir = Path(args.error_loc)
            error_dir.mkdir(parents=True, exist_ok=True)
            error_path = error_dir / f"{args.problem}.rds"
            new_error = pd.DataFrame([{
                "error_sent_datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
                "data_type": args.data_type,
                "corpus": args.corpus,
                "scoring_model": Path(args.model_loc).name,
                "max_context_tokens": args.num_tokens,
                "problem": args.problem,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
            }])
            if error_path.exists():
                new_error = pd.concat([read_rds(str(error_path)), new_error], ignore_index=True)
            write_rds(new_error, str(error_path))
            print(f"Error record saved to {error_path}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
