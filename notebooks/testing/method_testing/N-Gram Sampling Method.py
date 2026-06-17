# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
!pip install pyreadr
dbutils.library.restartPython()

# COMMAND ----------

# import sys

# from from_root import from_root

# sys.path.insert(0, str(from_root("src")))

# from model_loading import load_model
# from read_and_write_docs import read_txt
# from n_gram_tracing import (
#     common_ngrams,
#     filter_len_common_ngrams,
#     tokens_to_text,
# )

# COMMAND ----------

import sys

# Point this to wherever your repo lives in Databricks
SRC_PATH = "/Workspace/Users/ben.cross@coop.co.uk/n-gram-av-methods/src"
# Or if using Databricks Repos: "/Workspace/Repos/<email>/<repo-name>/src"

sys.path.insert(0, SRC_PATH)

from model_loading import load_model
from read_and_write_docs import read_txt
from n_gram_tracing import (
    common_ngrams,
    filter_len_common_ngrams,
    tokens_to_text,
    ngram_occurrence_stats,
    tokenize_to_tokens
)
from n_gram_scoring import score_ngrams

# COMMAND ----------

def dedupe_ngrams(ngrams):
    """
    Deduplicate n-grams while preserving order.
    """
    return [list(x) for x in dict.fromkeys(tuple(g) for g in ngrams)]


def sort_ngrams(ngrams):
    """
    Sort first by number of tokens, then by total character length.
    """
    return sorted(
        ngrams,
        key=lambda x: (len(x), sum(len(str(token)) for token in x))
    )
    
def global_second_pass_greatest_common(
    known_texts,
    unknown_text,
    tokenizer,
    *,
    lowercase=True,
):
    all_common = []

    # First pass: collect all pairwise common n-grams
    for known_text in known_texts:
        pair_common = common_ngrams(
            text1=known_text,
            text2=unknown_text,
            tokenizer=tokenizer,
            include_subgrams=True,
            lowercase=lowercase,
        )

        all_common.extend(pair_common)

    # Problem-level candidate list
    global_common = dedupe_ngrams(all_common)
    global_common = sort_ngrams(global_common)

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

    kept = []

    for g in dict.fromkeys(tuple(x) for x in global_common):
        unknown_keep = unknown_stats.get(g, {}).get("keep", False)

        known_keep = any(
            stats.get(g, {}).get("keep", False)
            for stats in known_stats_list
        )

        if unknown_keep and known_keep:
            kept.append(list(g))

    kept = sort_ngrams(kept)
    
    return kept

# COMMAND ----------

tokenizer, model = load_model("openai-community/gpt2")

# COMMAND ----------

known_1 = read_txt("../../../data/hodja_nasreddin_text_1.txt")
known_2 = read_txt("../../../data/hodja_nasreddin_text_10.txt")

known_texts = [known_1, known_2]

unknown_text = read_txt("../../../data/honestopl_text_3.txt")

# COMMAND ----------

unknown_text

# COMMAND ----------

common_tokens = global_second_pass_greatest_common(
    known_texts,
    unknown_text,
    tokenizer,
    lowercase=True,
)

# COMMAND ----------

import random
from typing import List, Dict, Tuple, Any


def _ngram_in_tokens(ngram: Tuple[str, ...], tokens: List[str]) -> bool:
    """Check if the exact n-gram sequence appears contiguously in tokens."""
    n = len(ngram)
    if n == 0 or n > len(tokens):
        return False
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i : i + n]) == ngram:
            return True
    return False


def sample_matching_ngrams(
    full_tokens: List[str],
    common_ngrams: List[Tuple[str, ...]],
    n_samples: int = 1,
    max_attempts: int = 1000,
    seed: int = None,
) -> Dict[str, Any]:
    """
    For each n-gram in `common_ngrams` that actually occurs as a contiguous
    sequence in `full_tokens`, sample `n_samples` random n-grams of the same
    length from `full_tokens` that do NOT appear in `common_ngrams`.

    N-grams not present (in order) in `full_tokens` are skipped — useful when
    `common_ngrams` is derived from multiple documents.

    Args:
        full_tokens: Complete list of tokens from the document, in order.
        common_ngrams: List of n-grams (each a tuple/list of tokens).
        n_samples: Number of random n-grams to draw per common n-gram.
        max_attempts: Safety cap on retries when searching for a valid sample.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary with:
            - "common_ngrams": n-grams found as contiguous sequences in the doc
            - "skipped_ngrams": n-grams not found in the doc
            - "sampled_ngrams": flat list of all sampled n-grams
            - "pairs": list of {"original": ..., "sample": [...]} dicts
    """
    if seed is not None:
        random.seed(seed)

    common_set = {tuple(ng) for ng in common_ngrams}

    pairs = []
    all_sampled = []
    processed = []
    skipped = []

    for ngram in common_ngrams:
        ngram_t = tuple(ngram)
        n = len(ngram_t)

        # Skip if document too short
        if len(full_tokens) < n or n == 0:
            skipped.append(ngram_t)
            continue

        # Skip if exact sequence not present in this document
        if not _ngram_in_tokens(ngram_t, full_tokens):
            skipped.append(ngram_t)
            continue

        max_start = len(full_tokens) - n
        samples = []
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            start = random.randint(0, max_start)
            candidate = tuple(full_tokens[start : start + n])
            attempts += 1

            if candidate in common_set:
                continue
            if candidate in samples:
                continue

            samples.append(candidate)

        processed.append(ngram_t)
        pairs.append({"original": ngram_t, "sample": samples})
        all_sampled.extend(samples)

    return {
        "common_ngrams": processed,
        "skipped_ngrams": skipped,
        "sampled_ngrams": all_sampled,
        "pairs": pairs,
    }

# COMMAND ----------

unknown_tokens = tokenize_to_tokens(unknown_text, tokenizer, lowercase=True)

known_1_tokens = tokenize_to_tokens(known_1, tokenizer, lowercase=True)
known_2_tokens = tokenize_to_tokens(known_2, tokenizer, lowercase=True)

# COMMAND ----------

import pandas as pd
from typing import List, Dict, Any


def _score_pairs_to_rows(
    pairs: List[Dict[str, Any]],
    model,
    tokenizer,
    doc_label: str,
    lowercase: bool = True,
    use_bos: bool = True,
) -> List[Dict[str, Any]]:
    """
    Score every original + sampled n-gram in `pairs` and return a list of
    flat row dicts ready for a DataFrame.
    """
    rows = []

    for pair_idx, pair in enumerate(pairs):
        original = pair["original"]
        samples = pair["sample"]

        # Score the original
        orig_scores = score_ngrams(
            original,
            model=model,
            tokenizer=tokenizer,
            lowercase=lowercase,
            use_bos=use_bos,
        )
        rows.append({
            "doc_label": doc_label,
            "pair_index": pair_idx,
            "kind": "original",
            "ngram": tuple(original),
            "ngram_len": len(original),
            **orig_scores,
        })

        # Score each sample tied back to that original
        for s_idx, sample in enumerate(samples):
            samp_scores = score_ngrams(
                sample,
                model=model,
                tokenizer=tokenizer,
                lowercase=lowercase,
                use_bos=use_bos,
            )
            rows.append({
                "doc_label": doc_label,
                "pair_index": pair_idx,
                "kind": "sample",
                "sample_index": s_idx,
                "ngram": tuple(sample),
                "ngram_len": len(sample),
                "paired_original": tuple(original),
                **samp_scores,
            })

    return rows


def build_score_dataframes(
    known_texts: List[str],
    unknown_text: str,
    common_tokens,
    model,
    tokenizer,
    n_samples: int = 10,
    max_attempts: int = 1000,
    seed: int = 42,
    lowercase: bool = True,
    use_bos: bool = True,
    known_labels: List[str] = None,
    unknown_label: str = "unknown",
) -> Dict[str, pd.DataFrame]:
    """
    Tokenize the known and unknown texts, sample matched n-grams against
    `common_tokens`, score them, and return two dataframes:

        - known_df:   union of all known docs (with a `doc_label` column)
        - unknown_df: scores for the unknown doc

    Each row contains the n-gram, whether it's an original or sample, which
    document it came from, and the score fields returned by `score_ngrams`.
    """
    if known_labels is None:
        known_labels = [f"known_{i+1}" for i in range(len(known_texts))]
    assert len(known_labels) == len(known_texts), \
        "known_labels must match known_texts length"

    # ---- Unknown ----
    unknown_tokens = tokenize_to_tokens(unknown_text, tokenizer, lowercase=lowercase)
    unknown_dict = sample_matching_ngrams(
        unknown_tokens,
        common_tokens,
        n_samples=n_samples,
        max_attempts=max_attempts,
        seed=seed,
    )
    unknown_rows = _score_pairs_to_rows(
        unknown_dict["pairs"],
        model=model,
        tokenizer=tokenizer,
        doc_label=unknown_label,
        lowercase=lowercase,
        use_bos=use_bos,
    )
    unknown_df = pd.DataFrame(unknown_rows)

    # ---- Known (union) ----
    known_rows_all = []
    for text, label in zip(known_texts, known_labels):
        known_tokens = tokenize_to_tokens(text, tokenizer, lowercase=lowercase)
        known_dict = sample_matching_ngrams(
            known_tokens,
            common_tokens,
            n_samples=n_samples,
            max_attempts=max_attempts,
            seed=seed,
        )
        known_rows_all.extend(
            _score_pairs_to_rows(
                known_dict["pairs"],
                model=model,
                tokenizer=tokenizer,
                doc_label=label,
                lowercase=lowercase,
                use_bos=use_bos,
            )
        )

    known_df = pd.DataFrame(known_rows_all)

    return {"known_df": known_df, "unknown_df": unknown_df}


# COMMAND ----------

import pandas as pd
import numpy as np


def compute_llr_scores(df: pd.DataFrame, score_col: str = "sum_log_probs") -> dict:
    """
    Given a long-format dataframe (originals + samples) produced by
    build_score_dataframes, compute:

        - pair_df:   one row per (doc_label, pair_index) with the LLR
                     (original score - mean sample score)
        - ngram_df:  one row per ngram, averaging the per-pair LLRs across
                     the documents in which it appeared

    Args:
        df: known_df or unknown_df from build_score_dataframes.
        score_col: which score column to use (default "sum_log_probs").

    Returns:
        {"pair_df": DataFrame, "ngram_df": DataFrame}
    """
    if df.empty:
        return {"pair_df": df.copy(), "ngram_df": df.copy()}

    # Tuples are hashable so they group cleanly
    df = df.copy()
    df["ngram"] = df["ngram"].apply(tuple)

    # ---- Originals: one row per (doc, pair) ----
    originals = (
        df[df["kind"] == "original"]
        .loc[:, ["doc_label", "pair_index", "ngram", "ngram_len", score_col]]
        .rename(columns={score_col: "original_score"})
    )

    # ---- Samples: average per (doc, pair) ----
    samples = (
        df[df["kind"] == "sample"]
        .groupby(["doc_label", "pair_index"], as_index=False)[score_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "sample_mean_score", "count": "n_samples"})
    )

    # ---- Per-pair LLR ----
    pair_df = originals.merge(samples, on=["doc_label", "pair_index"], how="left")
    pair_df["llr"] = pair_df["original_score"] - pair_df["sample_mean_score"]

    # ---- Per-ngram aggregate across documents ----
    ngram_df = (
        pair_df.groupby("ngram", as_index=False)
        .agg(
            ngram_len=("ngram_len", "first"),
            mean_llr=("llr", "mean"),
            n_docs=("doc_label", "nunique"),
            docs=("doc_label", lambda s: sorted(set(s))),
            mean_original_score=("original_score", "mean"),
            mean_sample_score=("sample_mean_score", "mean"),
        )
        .sort_values("mean_llr", ascending=False)
        .reset_index(drop=True)
    )

    return {"pair_df": pair_df, "ngram_df": ngram_df}

# COMMAND ----------

import pandas as pd
import numpy as np


def compute_problem_score(
    known_ngram_df: pd.DataFrame,
    unknown_ngram_df: pd.DataFrame,
    score_col: str = "mean_llr",
    eps: float = 1e-9,
) -> dict:
    """
    Combine per-ngram LLRs from the known and unknown dataframes into a
    single problem-level score.

    For each n-gram present in BOTH known and unknown:
        - diff   = unknown_llr - known_llr   (recommended, additive)
        - ratio  = unknown_llr / known_llr   (your original spec)

    The final problem score is the cumulative sum of `diff` across all
    shared n-grams, plus a few summary stats.

    Returns:
        {
            "ngram_score_df": per-ngram comparison dataframe (with cumulative score),
            "summary": dict of overall scores and counts,
        }
    """
    k = known_ngram_df[["ngram", "ngram_len", score_col]].rename(
        columns={score_col: "known_llr"}
    )
    u = unknown_ngram_df[["ngram", "ngram_len", score_col]].rename(
        columns={score_col: "unknown_llr"}
    )

    # Inner merge: only n-grams scored in both
    merged = k.merge(u, on=["ngram", "ngram_len"], how="inner")

    # Per-ngram scores
    merged["diff"] = merged["unknown_llr"] - merged["known_llr"]
    merged["ratio"] = merged["unknown_llr"] / merged["known_llr"].replace(0, np.nan)

    # Sort for a stable cumulative trace
    merged = merged.sort_values("ngram").reset_index(drop=True)
    merged["cumulative_diff"] = merged["diff"].cumsum()
    merged["cumulative_ratio"] = merged["ratio"].cumsum()

    summary = {
        "n_shared_ngrams": len(merged),
        "n_known_ngrams": len(k),
        "n_unknown_ngrams": len(u),
        "problem_score_diff_sum": merged["diff"].sum(),
        "problem_score_diff_mean": merged["diff"].mean(),
        "problem_score_ratio_sum": merged["ratio"].sum(),
        "problem_score_ratio_mean": merged["ratio"].mean(),
    }

    return {"ngram_score_df": merged, "summary": summary}

# COMMAND ----------

dfs = build_score_dataframes(
    known_texts=known_texts,
    unknown_text=unknown_text,
    common_tokens=common_tokens,
    model=model,
    tokenizer=tokenizer,
    n_samples=10,
    max_attempts=1000,
    seed=42,
    known_labels=["known_1", "known_2"],   # optional, defaults to known_1, known_2, ...
    unknown_label="unknown",
)

# COMMAND ----------

# From the previous step
known_df   = dfs["known_df"]
unknown_df = dfs["unknown_df"]

known_scores   = compute_llr_scores(known_df)
unknown_scores = compute_llr_scores(unknown_df)

known_pair_df    = known_scores["pair_df"]
known_ngram_df   = known_scores["ngram_df"]
unknown_pair_df  = unknown_scores["pair_df"]
unknown_ngram_df = unknown_scores["ngram_df"]

# COMMAND ----------

result = compute_problem_score(
    known_ngram_df=known_ngram_df,
    unknown_ngram_df=unknown_ngram_df,
)

ngram_score_df = result["ngram_score_df"]
summary = result["summary"]

print(summary)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy.stats import entropy, pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import cosine, euclidean, jensenshannon


def compare_ngram_scores(unknown_ngram_df, known_ngram_df, 
                         ngram_col='ngram', score_col='mean_llr'):
    """
    Compare two vectors of n-gram scores for author verification.
    
    Parameters
    ----------
    unknown_ngram_df : pd.DataFrame
        DataFrame with columns [ngram_col, score_col] for the unknown author.
    known_ngram_df : pd.DataFrame
        DataFrame with columns [ngram_col, score_col] for the known author.
    ngram_col : str
        Name of the n-gram column.
    score_col : str
        Name of the score column (e.g., 'mean_llr').
    
    Returns
    -------
    dict
        Dictionary of similarity / divergence / correlation metrics.
        Higher = more similar for: cosine_similarity, pearson, spearman,
        kendall, dot_product.
        Lower  = more similar for: euclidean, manhattan, kl_div_*, js_div, 
        rmse, mae.
    """
    # --- Align by ngram ---
    merged = pd.merge(
        unknown_ngram_df[[ngram_col, score_col]].rename(columns={score_col: 'unk'}),
        known_ngram_df[[ngram_col, score_col]].rename(columns={score_col: 'kno'}),
        on=ngram_col,
        how='inner'
    ).sort_values(ngram_col).reset_index(drop=True)

    if len(merged) == 0:
        raise ValueError("No overlapping n-grams between the two dataframes.")

    u = merged['unk'].to_numpy(dtype=float)
    k = merged['kno'].to_numpy(dtype=float)

    results = {'n_ngrams': len(merged)}

    # --- Geometric / vector similarity ---
    # cosine similarity in [-1, 1] (1 = identical direction)
    results['cosine_similarity'] = 1 - cosine(u, k) if np.any(u) and np.any(k) else np.nan
    results['euclidean_distance'] = float(euclidean(u, k))
    results['manhattan_distance'] = float(np.sum(np.abs(u - k)))
    results['rmse'] = float(np.sqrt(np.mean((u - k) ** 2)))
    results['mae'] = float(np.mean(np.abs(u - k)))
    results['dot_product'] = float(np.dot(u, k))

    # --- Correlation-based ---
    if np.std(u) > 0 and np.std(k) > 0:
        results['pearson_r'], results['pearson_p'] = pearsonr(u, k)
        results['spearman_r'], results['spearman_p'] = spearmanr(u, k)
        results['kendall_tau'], results['kendall_p'] = kendalltau(u, k)
    else:
        results['pearson_r'] = results['spearman_r'] = results['kendall_tau'] = np.nan

    # --- Distribution-based (need non-negative, normalised) ---
    # Shift to be non-negative, then normalise to a probability distribution
    eps = 1e-12
    shift = min(u.min(), k.min())
    u_pos = u - shift + eps
    k_pos = k - shift + eps
    p = u_pos / u_pos.sum()
    q = k_pos / k_pos.sum()

    results['kl_div_unk_to_kno'] = float(entropy(p, q))   # KL(P||Q)
    results['kl_div_kno_to_unk'] = float(entropy(q, p))   # KL(Q||P)
    results['symmetric_kl']      = 0.5 * (results['kl_div_unk_to_kno'] +
                                          results['kl_div_kno_to_unk'])
    results['js_divergence']     = float(jensenshannon(p, q) ** 2)  # squared JS dist = JS div

    # --- A composite "verification score" ---
    # Higher => more likely same author. Combine cosine + (1 - normalised JS).
    js_sim = 1 - results['js_divergence']            # in [0, 1]
    cos_sim = (results['cosine_similarity'] + 1) / 2 # rescale [-1,1] -> [0,1]
    pear = results['pearson_r'] if not np.isnan(results['pearson_r']) else 0
    pear_sim = (pear + 1) / 2                        # rescale [-1,1] -> [0,1]
    results['composite_similarity'] = float(np.mean([cos_sim, js_sim, pear_sim]))

    return results

# COMMAND ----------

scores = compare_ngram_scores(unknown_ngram_df, known_ngram_df)
for k, v in scores.items():
    print(f"{k:25s}: {v}")

# COMMAND ----------


