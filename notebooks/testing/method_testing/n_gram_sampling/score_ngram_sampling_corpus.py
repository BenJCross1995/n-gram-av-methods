#!/usr/bin/env python3
# REFACTORED: one --problem per invocation; writes completed/<problem>.rds or errors/<problem>.rds.
"""Run sampled token n-gram tracing for one author-verification problem."""

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, jensenshannon
from scipy.stats import entropy, kendalltau, pearsonr, spearmanr

from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from model_loading import load_model
from n_gram_scoring import score_ngrams
from n_gram_tracing import (
    common_ngrams,
    filter_len_common_ngrams,
    ngram_occurrence_stats,
    tokenize_to_tokens,
)
from read_and_write_docs import read_jsonl, read_rds, write_rds
from utils import apply_temp_doc_id


RESULT_METADATA_COLUMNS = [
    "data_type",
    "corpus",
    "problem",
    "scoring_model",
]

ERROR_COLUMNS = [
    "data_type",
    "corpus",
    "problem",
    "scoring_model",
    "error_type",
    "error_reason",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run sampled token n-gram tracing for one problem and save either "
            "a completed RDS result or a one-row RDS error record."
        )
    )

    # Input/output paths
    ap.add_argument("--known_loc", required=True)
    ap.add_argument("--unknown_loc", required=True)
    ap.add_argument("--metadata_loc", required=True)
    ap.add_argument("--model_loc", required=True)
    ap.add_argument(
        "--save_loc",
        required=True,
        help="Directory for successfully completed problem-level RDS files.",
    )
    ap.add_argument(
        "--error_save_loc",
        required=True,
        help="Directory for problem-level RDS error records.",
    )

    # Dataset/problem metadata
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--data_type", required=True)
    ap.add_argument("--problem", required=True)

    # N-gram collection
    ap.add_argument("--min_len", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--lowercase", dest="lowercase", action="store_true")
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")

    ap.add_argument(
        "--greatest_common",
        dest="greatest_common",
        action="store_true",
        help="Use the global second-pass greatest-common n-gram filtering.",
    )
    ap.add_argument(
        "--no-greatest_common",
        dest="greatest_common",
        action="store_false",
        help="Use the deduplicated union of pairwise common n-grams instead.",
    )

    # Matched random sampling and scoring
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--max_attempts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score_col", default="sum_log_probs")
    ap.add_argument("--use_bos", dest="use_bos", action="store_true")
    ap.add_argument("--no-use_bos", dest="use_bos", action="store_false")

    # Final minimum-token-size sweep
    ap.add_argument(
        "--use_min_token_size",
        dest="use_min_token_size",
        action="store_true",
    )
    ap.add_argument(
        "--no-use_min_token_size",
        dest="use_min_token_size",
        action="store_false",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate a problem even when its completed RDS already exists.",
    )

    ap.set_defaults(
        lowercase=True,
        greatest_common=True,
        use_bos=True,
        use_min_token_size=True,
    )

    args = ap.parse_args()

    if args.n_samples < 1:
        ap.error("--n_samples must be at least 1")
    if args.max_attempts < 1:
        ap.error("--max_attempts must be at least 1")
    if args.min_len is not None and args.min_len < 1:
        ap.error("--min_len must be at least 1")
    if args.max_len is not None and args.max_len < 1:
        ap.error("--max_len must be at least 1")
    if (
        args.min_len is not None
        and args.max_len is not None
        and args.min_len > args.max_len
    ):
        ap.error("--min_len cannot be greater than --max_len")

    return args


def safe_filename(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def resolve_rds_path(location: str, default_filename: str) -> str:
    """Allow an output argument to be either a complete .rds path or a directory."""
    if location.lower().endswith(".rds"):
        output_path = location
    else:
        output_path = os.path.join(location, default_filename)

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    return output_path


def derive_error_path(result_path: str) -> str:
    stem, extension = os.path.splitext(result_path)
    extension = extension or ".rds"
    return f"{stem}_errors{extension}"


def dedupe_ngrams(ngrams: Sequence[Sequence[str]]) -> List[List[str]]:
    """Deduplicate n-grams while preserving their first-seen order."""
    return [list(x) for x in dict.fromkeys(tuple(g) for g in ngrams)]


def sort_ngrams(ngrams: Sequence[Sequence[str]]) -> List[List[str]]:
    """Sort by token count and then by total token character length."""
    return sorted(
        (list(ngram) for ngram in ngrams),
        key=lambda x: (len(x), sum(len(str(token)) for token in x)),
    )


def pairwise_common_union(
    known_texts: Sequence[str],
    unknown_text: str,
    tokenizer,
    *,
    lowercase: bool = True,
) -> List[List[str]]:
    """Return the deduplicated union of pairwise maximal common n-grams."""
    all_common: List[Sequence[str]] = []

    for known_text in known_texts:
        all_common.extend(
            common_ngrams(
                text1=known_text,
                text2=unknown_text,
                tokenizer=tokenizer,
                include_subgrams=False,
                lowercase=lowercase,
            )
        )

    return sort_ngrams(dedupe_ngrams(all_common))


def global_second_pass_greatest_common(
    known_texts: Sequence[str],
    unknown_text: str,
    tokenizer,
    *,
    lowercase: bool = True,
) -> List[List[str]]:
    """Collect pairwise candidates, then apply problem-level occurrence filtering."""
    all_common: List[Sequence[str]] = []

    for known_text in known_texts:
        pair_common = common_ngrams(
            text1=known_text,
            text2=unknown_text,
            tokenizer=tokenizer,
            include_subgrams=True,
            lowercase=lowercase,
        )
        all_common.extend(pair_common)

    global_common = sort_ngrams(dedupe_ngrams(all_common))

    if not global_common:
        return []

    unknown_stats = ngram_occurrence_stats(
        ngrams=global_common,
        text=unknown_text,
        tokenizer=tokenizer,
        lowercase=lowercase,
    )

    known_stats_list = [
        ngram_occurrence_stats(
            ngrams=global_common,
            text=known_text,
            tokenizer=tokenizer,
            lowercase=lowercase,
        )
        for known_text in known_texts
    ]

    kept: List[List[str]] = []

    for ngram in dict.fromkeys(tuple(x) for x in global_common):
        unknown_keep = unknown_stats.get(ngram, {}).get("keep", False)
        known_keep = any(
            stats.get(ngram, {}).get("keep", False)
            for stats in known_stats_list
        )

        if unknown_keep and known_keep:
            kept.append(list(ngram))

    return sort_ngrams(kept)


def ngram_in_tokens(ngram: Tuple[str, ...], tokens: Sequence[str]) -> bool:
    """Check whether an exact n-gram occurs contiguously in a token sequence."""
    n = len(ngram)
    if n == 0 or n > len(tokens):
        return False

    return any(
        tuple(tokens[start : start + n]) == ngram
        for start in range(len(tokens) - n + 1)
    )


def sample_matching_ngrams(
    full_tokens: List[str],
    common_ngram_list: Sequence[Sequence[str]],
    n_samples: int = 1,
    max_attempts: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample non-common n-grams matched to each common n-gram's token length."""
    rng = random.Random(seed)
    common_set = {tuple(ngram) for ngram in common_ngram_list}

    pairs = []
    all_sampled = []
    processed = []
    skipped = []

    for ngram in common_ngram_list:
        ngram_tuple = tuple(ngram)
        ngram_length = len(ngram_tuple)

        if (
            ngram_length == 0
            or len(full_tokens) < ngram_length
            or not ngram_in_tokens(ngram_tuple, full_tokens)
        ):
            skipped.append(ngram_tuple)
            continue

        max_start = len(full_tokens) - ngram_length
        samples = []
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            start = rng.randint(0, max_start)
            candidate = tuple(full_tokens[start : start + ngram_length])
            attempts += 1

            if candidate in common_set or candidate in samples:
                continue

            samples.append(candidate)

        processed.append(ngram_tuple)
        pairs.append({"original": ngram_tuple, "sample": samples})
        all_sampled.extend(samples)

    return {
        "common_ngrams": processed,
        "skipped_ngrams": skipped,
        "sampled_ngrams": all_sampled,
        "pairs": pairs,
    }


def score_pairs_to_rows(
    pairs: Sequence[Dict[str, Any]],
    model,
    tokenizer,
    doc_label: str,
    lowercase: bool = True,
    use_bos: bool = True,
) -> List[Dict[str, Any]]:
    """Score every original and sampled n-gram in a document."""
    rows: List[Dict[str, Any]] = []

    for pair_index, pair in enumerate(pairs):
        original = pair["original"]
        samples = pair["sample"]

        original_scores = score_ngrams(
            original,
            model=model,
            tokenizer=tokenizer,
            lowercase=lowercase,
            use_bos=use_bos,
        )
        rows.append(
            {
                "doc_label": doc_label,
                "pair_index": pair_index,
                "kind": "original",
                "ngram": tuple(original),
                "ngram_len": len(original),
                **original_scores,
            }
        )

        for sample_index, sample in enumerate(samples):
            sample_scores = score_ngrams(
                sample,
                model=model,
                tokenizer=tokenizer,
                lowercase=lowercase,
                use_bos=use_bos,
            )
            rows.append(
                {
                    "doc_label": doc_label,
                    "pair_index": pair_index,
                    "kind": "sample",
                    "sample_index": sample_index,
                    "ngram": tuple(sample),
                    "ngram_len": len(sample),
                    "paired_original": tuple(original),
                    **sample_scores,
                }
            )

    return rows


def build_score_dataframes(
    known_texts: Sequence[str],
    unknown_text: str,
    common_tokens: Sequence[Sequence[str]],
    model,
    tokenizer,
    n_samples: int = 10,
    max_attempts: int = 1000,
    seed: int = 42,
    lowercase: bool = True,
    use_bos: bool = True,
    known_labels: Optional[Sequence[str]] = None,
    unknown_label: str = "unknown",
) -> Dict[str, pd.DataFrame]:
    """Sample and score matched n-grams for the known and unknown documents."""
    if known_labels is None:
        known_labels = [f"known_{i + 1}" for i in range(len(known_texts))]

    if len(known_labels) != len(known_texts):
        raise ValueError("known_labels must contain one label per known text")

    unknown_tokens = tokenize_to_tokens(
        unknown_text,
        tokenizer,
        lowercase=lowercase,
    )
    unknown_samples = sample_matching_ngrams(
        unknown_tokens,
        common_tokens,
        n_samples=n_samples,
        max_attempts=max_attempts,
        seed=seed,
    )
    unknown_rows = score_pairs_to_rows(
        unknown_samples["pairs"],
        model=model,
        tokenizer=tokenizer,
        doc_label=unknown_label,
        lowercase=lowercase,
        use_bos=use_bos,
    )

    known_rows: List[Dict[str, Any]] = []
    for known_index, (text, label) in enumerate(zip(known_texts, known_labels)):
        known_tokens = tokenize_to_tokens(text, tokenizer, lowercase=lowercase)
        known_samples = sample_matching_ngrams(
            known_tokens,
            common_tokens,
            n_samples=n_samples,
            max_attempts=max_attempts,
            seed=seed + known_index,
        )
        known_rows.extend(
            score_pairs_to_rows(
                known_samples["pairs"],
                model=model,
                tokenizer=tokenizer,
                doc_label=str(label),
                lowercase=lowercase,
                use_bos=use_bos,
            )
        )

    return {
        "known_df": pd.DataFrame(known_rows),
        "unknown_df": pd.DataFrame(unknown_rows),
    }


def empty_ngram_score_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ngram",
            "ngram_len",
            "mean_llr",
            "n_docs",
            "docs",
            "mean_original_score",
            "mean_sample_score",
        ]
    )


def compute_llr_scores(
    df: pd.DataFrame,
    score_col: str = "sum_log_probs",
) -> Dict[str, pd.DataFrame]:
    """Calculate original-minus-mean-sample LLRs and aggregate by n-gram."""
    if df.empty:
        return {
            "pair_df": pd.DataFrame(),
            "ngram_df": empty_ngram_score_df(),
        }

    required_columns = {
        "doc_label",
        "pair_index",
        "kind",
        "ngram",
        "ngram_len",
        score_col,
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(
            "Scored n-gram table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    working_df = df.copy()
    working_df["ngram"] = working_df["ngram"].apply(tuple)

    originals = (
        working_df[working_df["kind"] == "original"]
        .loc[:, ["doc_label", "pair_index", "ngram", "ngram_len", score_col]]
        .rename(columns={score_col: "original_score"})
    )

    samples = (
        working_df[working_df["kind"] == "sample"]
        .groupby(["doc_label", "pair_index"], as_index=False)
        .agg(
            sample_mean_score=(score_col, "mean"),
            n_samples=(score_col, "count"),
        )
    )

    pair_df = originals.merge(
        samples,
        on=["doc_label", "pair_index"],
        how="left",
    )
    pair_df["llr"] = pair_df["original_score"] - pair_df["sample_mean_score"]

    valid_pair_df = pair_df.dropna(subset=["llr"]).copy()
    if valid_pair_df.empty:
        return {
            "pair_df": pair_df,
            "ngram_df": empty_ngram_score_df(),
        }

    ngram_df = (
        valid_pair_df.groupby("ngram", as_index=False)
        .agg(
            ngram_len=("ngram_len", "first"),
            mean_llr=("llr", "mean"),
            n_docs=("doc_label", "nunique"),
            docs=("doc_label", lambda values: sorted(set(values))),
            mean_original_score=("original_score", "mean"),
            mean_sample_score=("sample_mean_score", "mean"),
        )
        .sort_values("mean_llr", ascending=False)
        .reset_index(drop=True)
    )

    return {"pair_df": pair_df, "ngram_df": ngram_df}


def compare_ngram_scores(
    unknown_ngram_df: pd.DataFrame,
    known_ngram_df: pd.DataFrame,
    ngram_col: str = "ngram",
    score_col: str = "mean_llr",
    ngram_len_col: str = "ngram_len",
    use_min_token_size: bool = False,
):
    """Compare known and unknown n-gram score vectors."""

    metric_columns = [
        "n_ngrams",
        "cosine_similarity",
        "euclidean_distance",
        "manhattan_distance",
        "rmse",
        "mae",
        "dot_product",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "kendall_tau",
        "kendall_p",
        "kl_div_unk_to_kno",
        "kl_div_kno_to_unk",
        "symmetric_kl",
        "js_divergence",
        "composite_similarity",
    ]

    def empty_metrics() -> Dict[str, Any]:
        return {
            column: (0 if column == "n_ngrams" else np.nan)
            for column in metric_columns
        }

    def calculate_metrics(
        unknown_df: pd.DataFrame,
        known_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        merged = (
            pd.merge(
                unknown_df[[ngram_col, score_col]].rename(
                    columns={score_col: "unk"}
                ),
                known_df[[ngram_col, score_col]].rename(
                    columns={score_col: "kno"}
                ),
                on=ngram_col,
                how="inner",
            )
            .dropna(subset=["unk", "kno"])
            .sort_values(ngram_col)
            .reset_index(drop=True)
        )

        if merged.empty:
            raise ValueError("No valid overlapping n-gram scores remain")

        unknown_values = merged["unk"].to_numpy(dtype=float)
        known_values = merged["kno"].to_numpy(dtype=float)

        results: Dict[str, Any] = {"n_ngrams": len(merged)}

        results["cosine_similarity"] = (
            1 - cosine(unknown_values, known_values)
            if np.any(unknown_values) and np.any(known_values)
            else np.nan
        )
        results["euclidean_distance"] = float(
            euclidean(unknown_values, known_values)
        )
        results["manhattan_distance"] = float(
            np.sum(np.abs(unknown_values - known_values))
        )
        results["rmse"] = float(
            np.sqrt(np.mean((unknown_values - known_values) ** 2))
        )
        results["mae"] = float(
            np.mean(np.abs(unknown_values - known_values))
        )
        results["dot_product"] = float(
            np.dot(unknown_values, known_values)
        )

        if (
            len(unknown_values) >= 2
            and np.std(unknown_values) > 0
            and np.std(known_values) > 0
        ):
            results["pearson_r"], results["pearson_p"] = pearsonr(
                unknown_values,
                known_values,
            )
            results["spearman_r"], results["spearman_p"] = spearmanr(
                unknown_values,
                known_values,
            )
            results["kendall_tau"], results["kendall_p"] = kendalltau(
                unknown_values,
                known_values,
            )
        else:
            results["pearson_r"] = np.nan
            results["pearson_p"] = np.nan
            results["spearman_r"] = np.nan
            results["spearman_p"] = np.nan
            results["kendall_tau"] = np.nan
            results["kendall_p"] = np.nan

        epsilon = 1e-12
        shift = min(unknown_values.min(), known_values.min())
        unknown_positive = unknown_values - shift + epsilon
        known_positive = known_values - shift + epsilon
        unknown_distribution = unknown_positive / unknown_positive.sum()
        known_distribution = known_positive / known_positive.sum()

        results["kl_div_unk_to_kno"] = float(
            entropy(unknown_distribution, known_distribution)
        )
        results["kl_div_kno_to_unk"] = float(
            entropy(known_distribution, unknown_distribution)
        )
        results["symmetric_kl"] = float(
            0.5
            * (
                results["kl_div_unk_to_kno"]
                + results["kl_div_kno_to_unk"]
            )
        )
        results["js_divergence"] = float(
            jensenshannon(unknown_distribution, known_distribution) ** 2
        )

        js_similarity = 1 - results["js_divergence"]
        cosine_similarity = results["cosine_similarity"]
        cosine_rescaled = (
            (cosine_similarity + 1) / 2
            if not np.isnan(cosine_similarity)
            else 0
        )
        pearson_value = results["pearson_r"]
        pearson_rescaled = (
            (pearson_value + 1) / 2
            if not np.isnan(pearson_value)
            else 0
        )
        results["composite_similarity"] = float(
            np.mean([cosine_rescaled, js_similarity, pearson_rescaled])
        )

        return results

    required_unknown = {ngram_col, score_col}
    required_known = {ngram_col, score_col}
    if use_min_token_size:
        required_unknown.add(ngram_len_col)
        required_known.add(ngram_len_col)

    missing_unknown = required_unknown.difference(unknown_ngram_df.columns)
    missing_known = required_known.difference(known_ngram_df.columns)
    if missing_unknown:
        raise KeyError(
            "unknown_ngram_df is missing columns: "
            + ", ".join(sorted(missing_unknown))
        )
    if missing_known:
        raise KeyError(
            "known_ngram_df is missing columns: "
            + ", ".join(sorted(missing_known))
        )

    if not use_min_token_size:
        return calculate_metrics(unknown_ngram_df, known_ngram_df)

    min_token_sizes = sorted(
        set(unknown_ngram_df[ngram_len_col].dropna().unique())
        | set(known_ngram_df[ngram_len_col].dropna().unique())
    )

    if not min_token_sizes:
        raise ValueError("No n-gram lengths are available for the token-size sweep")

    comparison_rows = []

    for min_token_size in min_token_sizes:
        filtered_unknown = unknown_ngram_df[
            unknown_ngram_df[ngram_len_col] >= min_token_size
        ]
        filtered_known = known_ngram_df[
            known_ngram_df[ngram_len_col] >= min_token_size
        ]

        try:
            metrics = calculate_metrics(filtered_unknown, filtered_known)
        except ValueError:
            metrics = empty_metrics()

        comparison_rows.append(
            {
                "min_token_size": min_token_size,
                **metrics,
            }
        )

    return pd.DataFrame(comparison_rows)


def get_single_metadata_value(
    problem_metadata: pd.DataFrame,
    column: str,
    problem: Any,
) -> Any:
    values = problem_metadata[column].dropna().drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one {column!r} value for problem {problem!r}; "
            f"found {len(values)}"
        )
    return values[0]


def process_problem(
    problem: Any,
    corpus_metadata: pd.DataFrame,
    known: pd.DataFrame,
    unknown: pd.DataFrame,
    tokenizer,
    model,
    model_name: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run the complete sampled n-gram pipeline for one problem."""
    problem_metadata = corpus_metadata[
        corpus_metadata["problem"] == problem
    ].copy()

    if problem_metadata.empty:
        raise ValueError("No metadata rows were found for the problem")

    known_author = get_single_metadata_value(
        problem_metadata,
        "known_author",
        problem,
    )
    unknown_author = get_single_metadata_value(
        problem_metadata,
        "unknown_author",
        problem,
    )

    selected_known = known[known["author"] == known_author].copy()
    selected_unknown = unknown[unknown["author"] == unknown_author].copy()

    if selected_known.empty:
        raise ValueError(f"No known documents found for author {known_author!r}")
    if selected_unknown.empty:
        raise ValueError(f"No unknown document found for author {unknown_author!r}")

    known_texts = selected_known["text"].dropna().tolist()
    if not known_texts:
        raise ValueError("The problem has no non-null known texts")

    unknown_text_values = selected_unknown["text"].dropna().tolist()
    if not unknown_text_values:
        raise ValueError("The problem has no non-null unknown text")
    unknown_text = unknown_text_values[0]

    if args.greatest_common:
        common_tokens = global_second_pass_greatest_common(
            known_texts,
            unknown_text,
            tokenizer,
            lowercase=args.lowercase,
        )
    else:
        common_tokens = pairwise_common_union(
            known_texts,
            unknown_text,
            tokenizer,
            lowercase=args.lowercase,
        )

    common_tokens = filter_len_common_ngrams(
        common_tokens,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    if not common_tokens:
        raise ValueError("No common n-grams remained after collection and filtering")

    if "doc_id" in selected_known.columns:
        known_labels = selected_known.loc[
            selected_known["text"].notna(),
            "doc_id",
        ].astype(str).tolist()
    else:
        known_labels = [f"known_{i + 1}" for i in range(len(known_texts))]

    score_dataframes = build_score_dataframes(
        known_texts=known_texts,
        unknown_text=unknown_text,
        common_tokens=common_tokens,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        max_attempts=args.max_attempts,
        seed=args.seed,
        lowercase=args.lowercase,
        use_bos=args.use_bos,
        known_labels=known_labels,
        unknown_label="unknown",
    )

    known_df = score_dataframes["known_df"]
    unknown_df = score_dataframes["unknown_df"]

    if known_df.empty:
        raise ValueError("No known-document n-grams could be sampled and scored")
    if unknown_df.empty:
        raise ValueError("No unknown-document n-grams could be sampled and scored")

    known_scores = compute_llr_scores(known_df, score_col=args.score_col)
    unknown_scores = compute_llr_scores(unknown_df, score_col=args.score_col)

    known_ngram_df = known_scores["ngram_df"]
    unknown_ngram_df = unknown_scores["ngram_df"]

    if known_ngram_df.empty:
        raise ValueError("No valid known-document LLR scores were produced")
    if unknown_ngram_df.empty:
        raise ValueError("No valid unknown-document LLR scores were produced")

    comparison = compare_ngram_scores(
        unknown_ngram_df=unknown_ngram_df,
        known_ngram_df=known_ngram_df,
        ngram_col="ngram",
        score_col="mean_llr",
        ngram_len_col="ngram_len",
        use_min_token_size=args.use_min_token_size,
    )

    if isinstance(comparison, dict):
        comparison_df = pd.DataFrame([comparison])
    else:
        comparison_df = comparison.copy()

    if comparison_df.empty:
        raise ValueError("The comparison stage returned no result rows")

    comparison_df.insert(0, "data_type", args.data_type)
    comparison_df.insert(1, "corpus", args.corpus)
    comparison_df.insert(2, "problem", str(problem))
    comparison_df.insert(3, "scoring_model", model_name)

    return comparison_df


def main() -> None:
    args = parse_args()

    selected_problem = args.problem.strip().strip('"').strip("'")
    if not selected_problem:
        raise ValueError("--problem cannot be empty")

    model_name = os.path.basename(os.path.normpath(args.model_loc))
    problem_filename = f"{safe_filename(selected_problem)}.rds"

    result_path = resolve_rds_path(args.save_loc, problem_filename)
    error_path = resolve_rds_path(args.error_save_loc, problem_filename)

    if os.path.exists(result_path) and not args.overwrite:
        print(f"Completed result already exists; skipping: {result_path}")
        return

    if args.overwrite and os.path.exists(result_path):
        os.remove(result_path)

    print("============================================================")
    print(f"DATA_TYPE    : {args.data_type}")
    print(f"CORPUS       : {args.corpus}")
    print(f"PROBLEM      : {selected_problem}")
    print(f"SCORING_MODEL: {model_name}")
    print(f"RESULT_PATH  : {result_path}")
    print(f"ERROR_PATH   : {error_path}")
    print("============================================================")

    try:
        print("Loading model")
        tokenizer, model = load_model(args.model_loc)

        print("Loading known, unknown, and metadata tables")
        known = apply_temp_doc_id(read_jsonl(args.known_loc))
        unknown = apply_temp_doc_id(read_jsonl(args.unknown_loc))
        metadata = read_rds(args.metadata_loc)

        required_metadata_columns = {
            "corpus",
            "problem",
            "known_author",
            "unknown_author",
        }
        missing_metadata_columns = required_metadata_columns.difference(
            metadata.columns
        )
        if missing_metadata_columns:
            raise KeyError(
                "Metadata is missing columns: "
                + ", ".join(sorted(missing_metadata_columns))
            )

        for table_name, table in (("known", known), ("unknown", unknown)):
            missing_columns = {"author", "text"}.difference(table.columns)
            if missing_columns:
                raise KeyError(
                    f"{table_name} data is missing columns: "
                    + ", ".join(sorted(missing_columns))
                )

        corpus_metadata = metadata[metadata["corpus"] == args.corpus].copy()
        if corpus_metadata.empty:
            raise ValueError(
                f"No metadata rows were found for corpus {args.corpus!r}"
            )

        problem_results = process_problem(
            problem=selected_problem,
            corpus_metadata=corpus_metadata,
            known=known,
            unknown=unknown,
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            args=args,
        )

        sort_columns = RESULT_METADATA_COLUMNS.copy()
        if "min_token_size" in problem_results.columns:
            sort_columns.append("min_token_size")
        problem_results = (
            problem_results
            .sort_values(sort_columns)
            .reset_index(drop=True)
        )

        write_rds(problem_results, result_path)

        # Remove a stale error record when a previously failing problem succeeds.
        if os.path.exists(error_path):
            os.remove(error_path)

        print(
            f"Completed {selected_problem}: "
            f"saved {len(problem_results)} row(s) to {result_path}"
        )

    except Exception as exc:
        reason = str(exc).strip() or repr(exc)
        error_df = pd.DataFrame(
            [
                {
                    "data_type": args.data_type,
                    "corpus": args.corpus,
                    "problem": selected_problem,
                    "error_type": type(exc).__name__,
                    "error_reason": reason,
                }
            ],
            columns=ERROR_COLUMNS,
        )

        write_rds(error_df, error_path)

        print(
            f"Failed {selected_problem}: "
            f"{type(exc).__name__}: {reason}"
        )
        print(f"Saved error record to: {error_path}")



if __name__ == "__main__":
    main()
