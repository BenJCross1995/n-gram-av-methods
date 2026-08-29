#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from from_root import from_root

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

sys.path.insert(
    0,
    str(from_root("src")),
)

from read_and_write_docs import (
    read_rds,
    write_rds,
)


# ============================================================
# Version
# ============================================================

METRICS_VERSION = "t5_paraphrase_metrics_v1"
SENTINEL = "<extra_id_0>"


# ============================================================
# Globals initialised after parsing CLI arguments
# ============================================================

DEVICE = None

SEMANTIC_MODEL_NAME = None
NLI_MODEL_NAME = None
FLUENCY_MODEL_NAME = None
NGRAM_TOKENIZER_NAME = None
T5_TOKENIZER_NAME = None

semantic_model = None

nli_tokenizer = None
nli_model = None
NLI_LABEL_INDICES = None

fluency_tokenizer = None
fluency_model = None

ngram_tokenizer = None
t5_tokenizer = None


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Add semantic, NLI, fluency, phrase-preference, "
            "surface and diagnostic metrics to one T5 paraphrase "
            "candidate RDS dataframe."
        )
    )

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    ap.add_argument(
        "--input_loc",
        required=True,
        help="Input candidate .rds file.",
    )

    ap.add_argument(
        "--save_loc",
        required=True,
        help="Output scored .rds file.",
    )

    ap.add_argument(
        "--error_loc",
        default=None,
        help=(
            "Optional directory in which to save an RDS error record."
        ),
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite save_loc if it already exists.",
    )

    ap.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. If supplied, only the first "
            "N rows are evaluated."
        ),
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    ap.add_argument(
        "--semantic_model_loc",
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "SentenceTransformer model/name for embedding similarity."
        ),
    )

    ap.add_argument(
        "--nli_model_loc",
        default="cross-encoder/nli-deberta-v3-base",
        help="Sequence-classification NLI model/name.",
    )

    ap.add_argument(
        "--fluency_model_loc",
        default="gpt2",
        help=(
            "Causal LM used for local fluency and phrase preference."
        ),
    )

    ap.add_argument(
        "--ngram_tokenizer_loc",
        default="gpt2",
        help=(
            "Tokenizer used to define local context windows. "
            "This should match the tokenizer used for the common n-grams."
        ),
    )

    ap.add_argument(
        "--t5_tokenizer_loc",
        default="t5-large",
        help=(
            "T5 tokenizer used for candidate/source token-length metrics."
        ),
    )

    # --------------------------------------------------------
    # Metric settings
    # --------------------------------------------------------

    ap.add_argument(
        "--semantic_windows",
        nargs="+",
        type=int,
        default=[0, 5, 10],
        help=(
            "Local context windows, in n-gram-tokenizer tokens on each "
            "side, for embedding semantic similarity."
        ),
    )

    ap.add_argument(
        "--nli_windows",
        nargs="+",
        type=int,
        default=[0, 5, 10],
        help=(
            "Local context windows, in n-gram-tokenizer tokens on each "
            "side, for bidirectional NLI."
        ),
    )

    ap.add_argument(
        "--fluency_context_tokens",
        type=int,
        default=10,
        help=(
            "Local context window on each side used for whole-context "
            "causal-LM fluency comparison."
        ),
    )

    ap.add_argument(
        "--phrase_context_tokens",
        type=int,
        default=10,
        help=(
            "Number of preceding n-gram-tokenizer tokens used when "
            "scoring only the original/candidate phrase."
        ),
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--keep_local_text",
        action="store_true",
        help=(
            "Keep original_local_* and candidate_local_* columns "
            "in the final RDS."
        ),
    )

    return ap.parse_args()


# ============================================================
# Device / model loading
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def initialise_models(args):
    global DEVICE

    global SEMANTIC_MODEL_NAME
    global NLI_MODEL_NAME
    global FLUENCY_MODEL_NAME
    global NGRAM_TOKENIZER_NAME
    global T5_TOKENIZER_NAME

    global semantic_model

    global nli_tokenizer
    global nli_model
    global NLI_LABEL_INDICES

    global fluency_tokenizer
    global fluency_model

    global ngram_tokenizer
    global t5_tokenizer

    DEVICE = get_device()

    SEMANTIC_MODEL_NAME = args.semantic_model_loc
    NLI_MODEL_NAME = args.nli_model_loc
    FLUENCY_MODEL_NAME = args.fluency_model_loc
    NGRAM_TOKENIZER_NAME = args.ngram_tokenizer_loc
    T5_TOKENIZER_NAME = args.t5_tokenizer_loc

    print(f"Using device: {DEVICE}")

    print(
        f"Loading semantic model: "
        f"{SEMANTIC_MODEL_NAME}"
    )

    semantic_model = SentenceTransformer(
        SEMANTIC_MODEL_NAME,
        device=str(DEVICE),
    )

    print(
        f"Loading NLI model: "
        f"{NLI_MODEL_NAME}"
    )

    nli_tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL_NAME,
        use_fast=True,
    )

    nli_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            NLI_MODEL_NAME,
        )
        .to(DEVICE)
    )

    nli_model.eval()

    print(
        f"Loading fluency model: "
        f"{FLUENCY_MODEL_NAME}"
    )

    fluency_tokenizer = AutoTokenizer.from_pretrained(
        FLUENCY_MODEL_NAME,
        use_fast=True,
    )

    fluency_model = (
        AutoModelForCausalLM
        .from_pretrained(
            FLUENCY_MODEL_NAME,
        )
        .to(DEVICE)
    )

    fluency_model.eval()

    print(
        f"Loading n-gram tokenizer: "
        f"{NGRAM_TOKENIZER_NAME}"
    )

    ngram_tokenizer = AutoTokenizer.from_pretrained(
        NGRAM_TOKENIZER_NAME,
        use_fast=True,
    )

    print(
        f"Loading T5 tokenizer: "
        f"{T5_TOKENIZER_NAME}"
    )

    t5_tokenizer = AutoTokenizer.from_pretrained(
        T5_TOKENIZER_NAME,
        use_fast=True,
    )

    NLI_LABEL_INDICES = get_nli_label_indices(
        nli_model
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_candidate_df(df):
    required = {
        "masked_text",
        "original_span",
        "candidate",
        "starts_with_space",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Dataframe is missing required columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# TEXT / BOUNDARY HELPERS
# ============================================================

def split_masked_text(masked_text):
    """Recover exact left/right T5 context around <extra_id_0>."""
    masked_text = str(masked_text)

    if SENTINEL not in masked_text:
        raise ValueError(
            f"{SENTINEL!r} not found in masked_text: {masked_text!r}"
        )

    return masked_text.split(SENTINEL, 1)


def get_original_generation_span(row):
    """Use the same casing convention that T5 received."""
    span = str(row["original_span"])

    if bool(row.get("lowercase_input", False)):
        span = span.casefold()

    return span


def get_candidate_text(row):
    return str(row["candidate"])


def candidate_with_boundary(candidate, starts_with_space, left_context):
    """Restore T5 SentencePiece's leading word-boundary decision."""
    candidate = str(candidate)

    if (
        bool(starts_with_space)
        and left_context
        and not left_context[-1].isspace()
        and candidate
        and not candidate[0].isspace()
    ):
        candidate = " " + candidate

    return candidate


# ============================================================
# CHARACTER-PRESERVING TOKEN WINDOWS
# ============================================================

def get_offsets(text, tokenizer):
    if not text:
        return []

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "A fast tokenizer is required for offset-based local windows."
        )

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
    )

    return [
        tuple(offset)
        for offset in encoded["offset_mapping"]
        if tuple(offset) != (0, 0)
    ]


def trim_left_context(text, num_tokens, tokenizer=None):
    """Keep only the final N tokens but preserve exact source characters."""
    if tokenizer is None:
        tokenizer = ngram_tokenizer
    if num_tokens is None:
        return text

    if num_tokens <= 0 or not text:
        return ""

    offsets = get_offsets(text, tokenizer)

    if len(offsets) <= num_tokens:
        return text

    start_char = offsets[-num_tokens][0]
    return text[start_char:]


def trim_right_context(text, num_tokens, tokenizer=None):
    """Keep only the first N tokens but preserve exact source characters."""
    if tokenizer is None:
        tokenizer = ngram_tokenizer
    if num_tokens is None:
        return text

    if num_tokens <= 0 or not text:
        return ""

    offsets = get_offsets(text, tokenizer)

    if len(offsets) <= num_tokens:
        return text

    end_char = offsets[num_tokens - 1][1]
    return text[:end_char]


def build_local_pair(row, context_tokens, tokenizer=None):
    """
    Build original/candidate texts for semantic evaluation.

    context_tokens=0  -> span only
    context_tokens=5  -> ±5 original-tokenizer tokens
    context_tokens=10 -> ±10 original-tokenizer tokens
    """
    if tokenizer is None:
        tokenizer = ngram_tokenizer

    left, right = split_masked_text(row["masked_text"])

    original_span = get_original_generation_span(row)
    candidate = get_candidate_text(row)

    if context_tokens == 0:
        return original_span.strip(), candidate.strip()

    local_left = trim_left_context(
        left,
        context_tokens,
        tokenizer,
    )

    local_right = trim_right_context(
        right,
        context_tokens,
        tokenizer,
    )

    candidate_span = candidate_with_boundary(
        candidate=candidate,
        starts_with_space=row["starts_with_space"],
        left_context=local_left,
    )

    original_text = local_left + original_span + local_right
    candidate_text = local_left + candidate_span + local_right

    return original_text, candidate_text


def add_local_text_columns(df, windows=(0, 5, 10)):
    """Save exact texts used by embedding/NLI metrics for inspection."""
    out = df.copy()

    for window in windows:
        pairs = [
            build_local_pair(
                row,
                window,
                tokenizer=ngram_tokenizer,
            )
            for _, row in out.iterrows()
        ]

        out[f"original_local_{window}"] = [x[0] for x in pairs]
        out[f"candidate_local_{window}"] = [x[1] for x in pairs]

    return out


# ============================================================
# 1. T5 / LENGTH METRICS
# ============================================================

def add_t5_length_metrics(df, tokenizer=None):
    if tokenizer is None:
        tokenizer = t5_tokenizer

    out = df.copy()

    original_lengths = []
    candidate_lengths = []

    for _, row in out.iterrows():
        original_span = get_original_generation_span(row)
        candidate = get_candidate_text(row)

        original_lengths.append(
            len(
                tokenizer(
                    original_span,
                    add_special_tokens=False,
                ).input_ids
            )
        )

        candidate_lengths.append(
            len(
                tokenizer(
                    candidate,
                    add_special_tokens=False,
                ).input_ids
            )
        )

    out["original_span_t5_num_tokens"] = original_lengths
    out["candidate_t5_num_tokens"] = candidate_lengths

    denom = out["original_span_t5_num_tokens"].replace(0, np.nan)

    out["candidate_length_ratio"] = (
        out["candidate_t5_num_tokens"] / denom
    )

    out["candidate_t5_token_difference"] = (
        out["candidate_t5_num_tokens"]
        - out["original_span_t5_num_tokens"]
    )

    return out


# ============================================================
# 2. EMBEDDING SEMANTIC SIMILARITY
# ============================================================

def add_embedding_semantic_metrics(
    df,
    model=None,
    windows=(0, 5, 10),
    batch_size=32,
):
    if model is None:
        model = semantic_model

    out = df.copy()

    for window in windows:
        original_texts = (
            out[f"original_local_{window}"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        candidate_texts = (
            out[f"candidate_local_{window}"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        start = time.perf_counter()

        original_embeddings = model.encode(
            original_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        candidate_embeddings = model.encode(
            candidate_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        similarity = np.sum(
            original_embeddings * candidate_embeddings,
            axis=1,
        )

        elapsed = time.perf_counter() - start

        out[f"semantic_similarity_{window}"] = similarity
        out[f"semantic_similarity_{window}_time_seconds"] = (
            elapsed / len(out) if len(out) else np.nan
        )

    out["semantic_model"] = SEMANTIC_MODEL_NAME

    return out


# ============================================================
# 3. NLI / BIDIRECTIONAL SEMANTIC EQUIVALENCE
# ============================================================

def get_nli_label_indices(model=None):
    """Resolve entailment / neutral / contradiction without assuming order."""
    if model is None:
        model = nli_model
    id2label = getattr(model.config, "id2label", {}) or {}
    label2id = getattr(model.config, "label2id", {}) or {}

    indices = {
        "entailment": None,
        "neutral": None,
        "contradiction": None,
    }

    for idx, label in id2label.items():
        label_cf = str(label).casefold()

        for target in indices:
            if target in label_cf:
                indices[target] = int(idx)

    for label, idx in label2id.items():
        label_cf = str(label).casefold()

        for target in indices:
            if indices[target] is None and target in label_cf:
                indices[target] = int(idx)

    if indices["entailment"] is None:
        raise ValueError(
            "Could not identify NLI entailment class. "
            f"id2label={id2label}, label2id={label2id}"
        )

    return indices



@torch.inference_mode()
def nli_probabilities(
    premises,
    hypotheses,
    batch_size=32,
    tokenizer=None,
    model=None,
):
    """Return all available NLI class probabilities."""
    if tokenizer is None:
        tokenizer = nli_tokenizer

    if model is None:
        model = nli_model
    if len(premises) != len(hypotheses):
        raise ValueError("premises and hypotheses must have the same length")

    output = {
        "entailment": [],
        "neutral": [],
        "contradiction": [],
    }

    for start in range(0, len(premises), batch_size):
        p_batch = premises[start:start + batch_size]
        h_batch = hypotheses[start:start + batch_size]

        encoded = tokenizer(
            p_batch,
            h_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        logits = model(**encoded).logits
        probs = F.softmax(logits, dim=-1)

        for label, index in NLI_LABEL_INDICES.items():
            if index is None:
                output[label].extend([np.nan] * len(p_batch))
            else:
                output[label].extend(
                    probs[:, index].detach().cpu().tolist()
                )

    return {
        key: np.asarray(values, dtype=float)
        for key, values in output.items()
    }


def add_nli_metrics(df, windows=(0, 5, 10), batch_size=32):
    out = df.copy()

    for window in windows:
        originals = (
            out[f"original_local_{window}"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        candidates = (
            out[f"candidate_local_{window}"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        start = time.perf_counter()

        o_to_c = nli_probabilities(
            premises=originals,
            hypotheses=candidates,
            batch_size=batch_size,
        )

        c_to_o = nli_probabilities(
            premises=candidates,
            hypotheses=originals,
            batch_size=batch_size,
        )

        elapsed = time.perf_counter() - start

        for label in ["entailment", "neutral", "contradiction"]:
            out[f"nli_original_to_candidate_{label}_{window}"] = o_to_c[label]
            out[f"nli_candidate_to_original_{label}_{window}"] = c_to_o[label]

        out[f"nli_bidirectional_entailment_mean_{window}"] = (
            o_to_c["entailment"] + c_to_o["entailment"]
        ) / 2.0

        out[f"nli_bidirectional_entailment_min_{window}"] = np.minimum(
            o_to_c["entailment"],
            c_to_o["entailment"],
        )

        out[f"nli_bidirectional_contradiction_mean_{window}"] = (
            o_to_c["contradiction"] + c_to_o["contradiction"]
        ) / 2.0

        out[f"nli_bidirectional_contradiction_max_{window}"] = np.maximum(
            o_to_c["contradiction"],
            c_to_o["contradiction"],
        )

        out[f"nli_{window}_time_seconds"] = (
            elapsed / len(out) if len(out) else np.nan
        )

    out["nli_model"] = NLI_MODEL_NAME

    return out


# ============================================================
# 4. CAUSAL-LM FLUENCY / LOG-PROB HELPERS
# ============================================================

def get_causal_bos_id(tokenizer, model):
    bos_id = getattr(tokenizer, "bos_token_id", None)

    if bos_id is None:
        bos_id = getattr(model.config, "bos_token_id", None)

    # Practical GPT-2 fallback for local standalone scoring.
    if bos_id is None:
        bos_id = getattr(tokenizer, "eos_token_id", None)

    return bos_id


@torch.inference_mode()
def token_log_probs_with_offsets(
    text,
    tokenizer=None,
    model=None,
    use_bos=True,
):
    """
    Return token log probabilities aligned to ordinary input tokens,
    plus their character offsets.
    """
    if tokenizer is None:
        tokenizer = fluency_tokenizer

    if model is None:
        model = fluency_model

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    offsets = [
        tuple(x)
        for x in encoded["offset_mapping"][0].tolist()
    ]

    if input_ids.shape[1] == 0:
        return np.array([], dtype=float), offsets

    bos_id = get_causal_bos_id(tokenizer, model)
    has_bos = bool(use_bos and bos_id is not None)

    if has_bos:
        bos = torch.tensor(
            [[int(bos_id)]],
            dtype=torch.long,
            device=DEVICE,
        )

        ids_for_model = torch.cat([bos, input_ids], dim=1)
    else:
        ids_for_model = input_ids

    if ids_for_model.shape[1] < 2:
        return np.array([np.nan], dtype=float), offsets

    logits = model(input_ids=ids_for_model).logits

    log_probs_vocab = F.log_softmax(
        logits[:, :-1, :],
        dim=-1,
    )

    next_ids = ids_for_model[:, 1:]

    values = (
        log_probs_vocab
        .gather(-1, next_ids.unsqueeze(-1))
        .squeeze(-1)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )

    if has_bos:
        aligned = values
    else:
        aligned = np.concatenate([
            np.array([np.nan], dtype=float),
            values,
        ])

    return aligned, offsets


def sequence_log_prob_metrics(text):
    """Mean/sum token log-probability and perplexity for a text chunk."""
    token_log_probs, _ = token_log_probs_with_offsets(text)

    valid = token_log_probs[~np.isnan(token_log_probs)]

    if len(valid) == 0:
        return {
            "mean_log_prob": np.nan,
            "sum_log_prob": np.nan,
            "perplexity": np.nan,
            "num_scored_tokens": 0,
        }

    mean_lp = float(np.mean(valid))
    sum_lp = float(np.sum(valid))

    return {
        "mean_log_prob": mean_lp,
        "sum_log_prob": sum_lp,
        "perplexity": float(np.exp(-mean_lp)),
        "num_scored_tokens": int(len(valid)),
    }


def span_log_prob_metrics(full_text, span_char_start, span_char_end):
    """Score only tokens overlapping one character span."""
    token_log_probs, offsets = token_log_probs_with_offsets(full_text)

    selected = []

    for index, (token_start, token_end) in enumerate(offsets):
        overlaps = (
            token_start < span_char_end
            and token_end > span_char_start
        )

        if overlaps and index < len(token_log_probs):
            value = token_log_probs[index]

            if not np.isnan(value):
                selected.append(float(value))

    if not selected:
        return {
            "mean_log_prob": np.nan,
            "sum_log_prob": np.nan,
            "num_scored_tokens": 0,
        }

    return {
        "mean_log_prob": float(np.mean(selected)),
        "sum_log_prob": float(np.sum(selected)),
        "num_scored_tokens": int(len(selected)),
    }


def build_phrase_scoring_texts(row, context_tokens=10):
    """
    Build left-context + span for causal scoring of only the realised phrase.
    Right context is excluded because a causal LM cannot condition on it.
    """
    left, _ = split_masked_text(row["masked_text"])

    left = trim_left_context(
        left,
        context_tokens,
        ngram_tokenizer,
    )

    original_span = get_original_generation_span(row)

    candidate_span = candidate_with_boundary(
        candidate=get_candidate_text(row),
        starts_with_space=row["starts_with_space"],
        left_context=left,
    )

    original_full = left + original_span
    candidate_full = left + candidate_span

    return {
        "original_full": original_full,
        "candidate_full": candidate_full,
        "original_span_start": len(left),
        "original_span_end": len(original_full),
        "candidate_span_start": len(left),
        "candidate_span_end": len(candidate_full),
    }


def add_fluency_and_phrase_metrics(
    df,
    fluency_context_tokens=10,
    phrase_context_tokens=10,
):
    """
    Add:
      1. local whole-text fluency
      2. original-vs-candidate phrase preference given left context
    """
    out = df.copy()

    original_context_mean = []
    original_context_sum = []
    original_context_ppl = []

    candidate_context_mean = []
    candidate_context_sum = []
    candidate_context_ppl = []

    original_span_mean = []
    original_span_sum = []
    original_span_n = []

    candidate_span_mean = []
    candidate_span_sum = []
    candidate_span_n = []

    row_times = []

    original_local_col = f"original_local_{fluency_context_tokens}"
    candidate_local_col = f"candidate_local_{fluency_context_tokens}"

    for _, row in out.iterrows():
        start = time.perf_counter()

        original_context_metrics = sequence_log_prob_metrics(
            str(row[original_local_col])
        )

        candidate_context_metrics = sequence_log_prob_metrics(
            str(row[candidate_local_col])
        )

        phrase_text = build_phrase_scoring_texts(
            row,
            context_tokens=phrase_context_tokens,
        )

        original_phrase_metrics = span_log_prob_metrics(
            full_text=phrase_text["original_full"],
            span_char_start=phrase_text["original_span_start"],
            span_char_end=phrase_text["original_span_end"],
        )

        candidate_phrase_metrics = span_log_prob_metrics(
            full_text=phrase_text["candidate_full"],
            span_char_start=phrase_text["candidate_span_start"],
            span_char_end=phrase_text["candidate_span_end"],
        )

        original_context_mean.append(original_context_metrics["mean_log_prob"])
        original_context_sum.append(original_context_metrics["sum_log_prob"])
        original_context_ppl.append(original_context_metrics["perplexity"])

        candidate_context_mean.append(candidate_context_metrics["mean_log_prob"])
        candidate_context_sum.append(candidate_context_metrics["sum_log_prob"])
        candidate_context_ppl.append(candidate_context_metrics["perplexity"])

        original_span_mean.append(original_phrase_metrics["mean_log_prob"])
        original_span_sum.append(original_phrase_metrics["sum_log_prob"])
        original_span_n.append(original_phrase_metrics["num_scored_tokens"])

        candidate_span_mean.append(candidate_phrase_metrics["mean_log_prob"])
        candidate_span_sum.append(candidate_phrase_metrics["sum_log_prob"])
        candidate_span_n.append(candidate_phrase_metrics["num_scored_tokens"])

        row_times.append(time.perf_counter() - start)

    out["original_context_mean_log_prob"] = original_context_mean
    out["candidate_context_mean_log_prob"] = candidate_context_mean
    out["fluency_mean_logprob_difference"] = (
        out["candidate_context_mean_log_prob"]
        - out["original_context_mean_log_prob"]
    )

    out["original_context_sum_log_prob"] = original_context_sum
    out["candidate_context_sum_log_prob"] = candidate_context_sum
    out["fluency_sum_logprob_difference"] = (
        out["candidate_context_sum_log_prob"]
        - out["original_context_sum_log_prob"]
    )

    out["original_context_perplexity"] = original_context_ppl
    out["candidate_context_perplexity"] = candidate_context_ppl
    out["candidate_to_original_perplexity_ratio"] = (
        out["candidate_context_perplexity"]
        / out["original_context_perplexity"].replace(0, np.nan)
    )

    out["original_span_mean_log_prob"] = original_span_mean
    out["candidate_span_mean_log_prob"] = candidate_span_mean

    # Positive => actual/original wording is preferred.
    out["span_mean_logprob_preference"] = (
        out["original_span_mean_log_prob"]
        - out["candidate_span_mean_log_prob"]
    )

    out["original_span_sum_log_prob"] = original_span_sum
    out["candidate_span_sum_log_prob"] = candidate_span_sum

    # Positive => actual/original wording is preferred.
    out["span_sum_logprob_preference"] = (
        out["original_span_sum_log_prob"]
        - out["candidate_span_sum_log_prob"]
    )

    out["original_span_fluency_num_tokens"] = original_span_n
    out["candidate_span_fluency_num_tokens"] = candidate_span_n

    out["fluency_score_time_seconds"] = row_times
    out["fluency_model"] = FLUENCY_MODEL_NAME
    out["fluency_context_tokens"] = fluency_context_tokens
    out["phrase_context_tokens"] = phrase_context_tokens

    return out


# ============================================================
# 5. SURFACE / LEXICAL METRICS
# ============================================================

def levenshtein_distance(seq_a, seq_b):
    """Generic Levenshtein distance for strings or token sequences."""
    seq_a = list(seq_a)
    seq_b = list(seq_b)

    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = list(range(len(seq_b) + 1))

    for i, item_a in enumerate(seq_a, start=1):
        current = [i]

        for j, item_b in enumerate(seq_b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + int(item_a != item_b)

            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]


def normalized_edit_distance(seq_a, seq_b):
    denominator = max(len(seq_a), len(seq_b))

    if denominator == 0:
        return 0.0

    return levenshtein_distance(seq_a, seq_b) / denominator


def surface_tokens(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = ngram_tokenizer

    ids = tokenizer(
        str(text),
        add_special_tokens=False,
    ).input_ids

    return tokenizer.convert_ids_to_tokens(ids)


def token_jaccard_similarity(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    union = set_a | set_b

    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


def normalize_apostrophes(text):
    return (
        str(text)
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )


def remove_whitespace(text):
    return re.sub(r"\s+", "", str(text))


def possible_contraction_expansions(text):
    """
    Generate simple plausible surface expansions.

    's and 'd are ambiguous, so both common alternatives are retained.
    This is only a diagnostic flag, not a linguistic decision rule.
    """
    text = normalize_apostrophes(text).casefold()

    variants = {text}

    replacements = [
        ("n't", [" not"]),
        ("'re", [" are"]),
        ("'ve", [" have"]),
        ("'ll", [" will"]),
        ("'m", [" am"]),
        ("'s", [" is", " has"]),
        ("'d", [" would", " had"]),
    ]

    for contraction, expansions in replacements:
        new_variants = set(variants)

        for variant in variants:
            if contraction in variant:
                for expansion in expansions:
                    new_variants.add(
                        variant.replace(contraction, expansion)
                    )

        variants = new_variants

    return {
        re.sub(r"\s+", " ", variant).strip()
        for variant in variants
    }


def add_surface_metrics(df):
    out = df.copy()

    rows = []

    for _, row in out.iterrows():
        original = str(row["original_span"])
        candidate = str(row["candidate"])

        original_tokens = surface_tokens(original)
        candidate_tokens = surface_tokens(candidate)

        original_apostrophe_norm = normalize_apostrophes(original)
        candidate_apostrophe_norm = normalize_apostrophes(candidate)

        original_normalized = re.sub(
            r"\s+",
            " ",
            original_apostrophe_norm.casefold(),
        ).strip()

        candidate_normalized = re.sub(
            r"\s+",
            " ",
            candidate_apostrophe_norm.casefold(),
        ).strip()

        contraction_variants = possible_contraction_expansions(original)

        rows.append({
            "character_edit_distance": levenshtein_distance(
                original,
                candidate,
            ),
            "normalized_character_edit_distance": normalized_edit_distance(
                original,
                candidate,
            ),
            "token_edit_distance": levenshtein_distance(
                original_tokens,
                candidate_tokens,
            ),
            "normalized_token_edit_distance": normalized_edit_distance(
                original_tokens,
                candidate_tokens,
            ),
            "token_jaccard_similarity": token_jaccard_similarity(
                original_tokens,
                candidate_tokens,
            ),
            "exact_match": original == candidate,
            "case_only_change": (
                original != candidate
                and original.casefold() == candidate.casefold()
            ),
            "apostrophe_only_change": (
                original != candidate
                and original_apostrophe_norm == candidate_apostrophe_norm
            ),
            "whitespace_only_change": (
                original != candidate
                and remove_whitespace(original).casefold()
                == remove_whitespace(candidate).casefold()
            ),
            "possible_contraction_expansion": (
                candidate_normalized in contraction_variants
                and candidate_normalized != original_normalized
            ),
            "candidate_is_substring_original": (
                candidate.casefold() in original.casefold()
            ),
            "original_is_substring_candidate": (
                original.casefold() in candidate.casefold()
            ),
        })

    metric_df = pd.DataFrame(rows, index=out.index)

    return pd.concat([out, metric_df], axis=1)


# ============================================================
# 6. MASTER TEST FUNCTION
# ============================================================

def add_all_paraphrase_metrics(
    df,
    semantic_windows=(0, 5, 10),
    nli_windows=(0, 5, 10),
    batch_size=32,
    fluency_context_tokens=10,
    phrase_context_tokens=10,
    keep_local_text=True,
):
    """
    Add every candidate metric discussed so far.

    IMPORTANT:
    - nothing is filtered
    - no arbitrary thresholds are applied
    - no combined score is created
    """
    validate_candidate_df(df)

    out = df.copy()
    total_start = time.perf_counter()

    all_windows = sorted(
        set(semantic_windows)
        | set(nli_windows)
        | {fluency_context_tokens}
    )

    print("1/6 Building local comparison windows...")
    out = add_local_text_columns(
        out,
        windows=all_windows,
    )

    print("2/6 Adding T5 length metrics...")
    out = add_t5_length_metrics(out)

    print("3/6 Adding embedding semantic similarity...")
    out = add_embedding_semantic_metrics(
        out,
        windows=semantic_windows,
        batch_size=batch_size,
    )

    print("4/6 Adding bidirectional NLI metrics...")
    out = add_nli_metrics(
        out,
        windows=nli_windows,
        batch_size=batch_size,
    )

    print("5/6 Adding causal-LM fluency / phrase preference...")
    out = add_fluency_and_phrase_metrics(
        out,
        fluency_context_tokens=fluency_context_tokens,
        phrase_context_tokens=phrase_context_tokens,
    )

    print("6/6 Adding surface metrics...")
    out = add_surface_metrics(out)

    if "manual_label" not in out.columns:
        out["manual_label"] = None

    if "manual_notes" not in out.columns:
        out["manual_notes"] = None

    if not keep_local_text:
        local_cols = [
            col
            for col in out.columns
            if col.startswith("original_local_")
            or col.startswith("candidate_local_")
        ]

        out = out.drop(columns=local_cols)

    out["paraphrase_metric_total_time_seconds"] = (
        time.perf_counter() - total_start
    )

    return out


# ============================================================
# Production runner
# ============================================================

def run_pipeline(args):
    if not os.path.exists(args.input_loc):
        raise FileNotFoundError(
            f"Input RDS does not exist: {args.input_loc}"
        )

    if os.path.exists(args.save_loc) and not args.overwrite:
        print(
            f"Output already exists: {args.save_loc}"
        )
        print(
            "Exiting. Use --overwrite to replace it."
        )
        return

    save_dir = os.path.dirname(
        os.path.abspath(args.save_loc)
    )

    if save_dir:
        os.makedirs(
            save_dir,
            exist_ok=True,
        )

    print(
        f"Metrics version: {METRICS_VERSION}"
    )

    print(
        f"Reading: {args.input_loc}"
    )

    df = read_rds(
        args.input_loc
    )

    print(
        f"Input rows: {len(df)}"
    )

    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError(
                "--max_rows must be >= 1"
            )

        df = df.head(
            args.max_rows
        ).copy()

        print(
            f"Smoke-test row limit applied: "
            f"{len(df)} rows"
        )

    validate_candidate_df(
        df
    )

    initialise_models(
        args
    )

    total_start = time.perf_counter()

    scored_df = add_all_paraphrase_metrics(
        df=df,
        semantic_windows=args.semantic_windows,
        nli_windows=args.nli_windows,
        batch_size=args.batch_size,
        fluency_context_tokens=args.fluency_context_tokens,
        phrase_context_tokens=args.phrase_context_tokens,
        keep_local_text=args.keep_local_text,
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    scored_df[
        "paraphrase_metrics_version"
    ] = METRICS_VERSION

    scored_df[
        "paraphrase_metrics_input_file"
    ] = os.path.basename(
        args.input_loc
    )

    scored_df[
        "paraphrase_metrics_total_pipeline_time_seconds"
    ] = total_time

    print(
        f"Scored rows: {len(scored_df)}"
    )

    print(
        f"Total metric time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Saving: {args.save_loc}"
    )

    write_rds(
        scored_df,
        args.save_loc,
    )

    print(
        "Saved successfully."
    )


# ============================================================
# Error handling
# ============================================================

def write_error(args, exc, tb):
    if args.error_loc is None:
        return

    os.makedirs(
        args.error_loc,
        exist_ok=True,
    )

    input_name = (
        os.path.splitext(
            os.path.basename(
                args.input_loc
            )
        )[0]
    )

    error_file = os.path.join(
        args.error_loc,
        f"{input_name}.rds",
    )

    new_error_df = pd.DataFrame([
        {
            "error_sent_datetime": (
                datetime.now()
                .astimezone()
                .isoformat(
                    timespec="seconds"
                )
            ),
            "metrics_version": METRICS_VERSION,
            "input_loc": args.input_loc,
            "save_loc": args.save_loc,
            "semantic_model": args.semantic_model_loc,
            "nli_model": args.nli_model_loc,
            "fluency_model": args.fluency_model_loc,
            "ngram_tokenizer": args.ngram_tokenizer_loc,
            "t5_tokenizer": args.t5_tokenizer_loc,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": tb,
        }
    ])

    if os.path.exists(
        error_file
    ):
        existing = read_rds(
            error_file
        )

        error_df = pd.concat(
            [
                existing,
                new_error_df,
            ],
            ignore_index=True,
        )

    else:
        error_df = new_error_df

    write_rds(
        error_df,
        error_file,
    )

    print(
        f"Error information written to: "
        f"{error_file}"
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    try:
        run_pipeline(
            args
        )

    except Exception as exc:
        tb = traceback.format_exc()

        print(
            "ERROR while scoring T5 paraphrase metrics"
        )

        print(
            tb
        )

        write_error(
            args,
            exc,
            tb,
        )

        raise


if __name__ == "__main__":
    main()
