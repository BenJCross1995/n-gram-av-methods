#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from n_gram_tracing import (
    find_all_token_ngram_spans,
    find_independent_token_ngram_spans,
    tokenize_to_tokens,
    tokens_to_text,
)


# ============================================================
# Device / model loading
# ============================================================

def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_t5_paraphrase_model(
    model_loc: str,
    device: Optional[torch.device] = None,
):
    """
    Load T5 for deterministic span-infilling generation.
    """
    if device is None:
        device = get_torch_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_loc,
        use_fast=True,
    )

    model_kwargs = {}

    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = T5ForConditionalGeneration.from_pretrained(
        model_loc,
        **model_kwargs,
    ).to(device)

    model.eval()

    return tokenizer, model, device


# ============================================================
# T5 generation helpers
# ============================================================

def get_quote_variants(text: str) -> List[str]:
    """
    Return straight- and curly-apostrophe variants.
    """
    variants = {
        text,
        text.replace("'", "’"),
        text.replace("’", "'"),
    }

    return list(variants)


def make_bad_words_ids(
    original_span: str,
    tokenizer: Any,
) -> List[List[int]]:
    """
    Build token sequences that T5 is not allowed to reproduce.

    Both leading-space and non-leading-space versions are included because
    SentencePiece tokenisation changes at word boundaries.
    """
    variants = set()

    for quote_variant in get_quote_variants(original_span):
        stripped = quote_variant.strip()

        if not stripped:
            continue

        variants.add(stripped)
        variants.add(" " + stripped)

    bad_words_ids = []

    for text in variants:
        ids = tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids

        if ids:
            bad_words_ids.append(list(ids))

    # Deduplicate token sequences while preserving order.
    unique = []

    for ids in bad_words_ids:
        if ids not in unique:
            unique.append(ids)

    return unique


def _normalise_for_comparison(text: str) -> str:
    """
    Normalisation used only to remove exact reconstructions.
    """
    return (
        text
        .strip()
        .casefold()
        .replace("’", "'")
    )


def extract_t5_fill(
    token_ids: torch.Tensor,
    tokenizer: Any,
    lowercase_output: bool = True,
) -> Tuple[str, bool, List[str]]:
    """
    Extract only the generated fill between:

        <extra_id_0> ... <extra_id_1>

    Returns
    -------
    candidate
        Decoded replacement text.

    starts_with_space
        Whether the first SentencePiece token has the ▁ word-boundary
        marker. This preserves T5's own left-boundary decision when the
        replacement is inserted back into the text.

    fill_tokens
        Raw T5 SentencePiece tokens.
    """
    extra_id_0 = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")

    ids = token_ids.detach().cpu().tolist()

    try:
        start = ids.index(extra_id_0) + 1
    except ValueError:
        # T5 normally begins with <pad><extra_id_0>, but retain a safe
        # fallback for unexpected generations.
        start = 1

    try:
        end = ids.index(extra_id_1, start)
    except ValueError:
        end = len(ids)

    fill_ids = ids[start:end]

    if not fill_ids:
        return "", False, []

    fill_tokens = tokenizer.convert_ids_to_tokens(fill_ids)

    starts_with_space = bool(
        fill_tokens
        and str(fill_tokens[0]).startswith("▁")
    )

    candidate = tokenizer.decode(
        fill_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if lowercase_output:
        candidate = candidate.casefold()

    return candidate, starts_with_space, list(fill_tokens)


@torch.inference_mode()
def generate_t5_candidates(
    masked_text: str,
    original_span: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    num_beams: int = 40,
    num_beam_groups: int = 10,
    num_return_sequences: int = 20,
    diversity_penalty: float = 0.5,
    max_new_tokens: int = 20,
    lowercase_output: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic alternative T5 infills for one masked span.

    The original removed span is explicitly prohibited. Sampling is disabled.
    """
    if num_beams < 1:
        raise ValueError("num_beams must be >= 1")

    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be >= 1")

    if num_return_sequences > num_beams:
        raise ValueError(
            "num_return_sequences cannot exceed num_beams"
        )

    if num_beam_groups < 1:
        raise ValueError("num_beam_groups must be >= 1")

    if num_beams % num_beam_groups != 0:
        raise ValueError(
            "num_beams must be divisible by num_beam_groups"
        )

    inputs = tokenizer(
        masked_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    bad_words_ids = make_bad_words_ids(
        original_span=original_span,
        tokenizer=tokenizer,
    )

    extra_id_1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")

    generation_kwargs = {
        "do_sample": False,
        "trust_remote_code": True,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "bad_words_ids": bad_words_ids,
        # Stop immediately after T5 finishes the first missing span.
        "eos_token_id": extra_id_1,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": max_new_tokens,
        "early_stopping": True,
        "renormalize_logits": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    if num_beam_groups > 1:
        generation_kwargs["num_beam_groups"] = num_beam_groups
        generation_kwargs["diversity_penalty"] = diversity_penalty

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    sequence_scores = getattr(
        outputs,
        "sequences_scores",
        None,
    )

    candidates = []
    seen = set()

    original_norm = _normalise_for_comparison(
        original_span
    )

    candidate_rank = 0

    for generation_rank, sequence in enumerate(
        outputs.sequences,
        start=1,
    ):
        candidate, starts_with_space, fill_tokens = extract_t5_fill(
            token_ids=sequence,
            tokenizer=tokenizer,
            lowercase_output=lowercase_output,
        )

        if not candidate:
            continue

        # Secondary safeguard in case bad_words_ids misses an alternative
        # tokenisation of the original removed phrase.
        if (
            _normalise_for_comparison(candidate)
            == original_norm
        ):
            continue

        dedupe_key = (
            _normalise_for_comparison(candidate),
            starts_with_space,
        )

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        candidate_rank += 1

        generation_score = None

        if sequence_scores is not None:
            generation_score = float(
                sequence_scores[generation_rank - 1]
                .detach()
                .cpu()
            )

        candidates.append({
            # Rank amongst candidates that survive filtering/deduplication.
            "candidate_rank": candidate_rank,

            # Rank in the raw T5 generation output.
            "generation_rank": generation_rank,

            "candidate": candidate,
            "generation_score": generation_score,
            "starts_with_space": starts_with_space,
            "candidate_tokens": fill_tokens,
        })

    return candidates


# ============================================================
# Existing n_gram_tracing integration
# ============================================================

def prepare_source_token_data(
    text: str,
    tokenizer: Any,
    *,
    lowercase: bool = True,
) -> Dict[str, Any]:
    """
    Tokenise once using the canonical n_gram_tracing path and obtain
    character offsets for those same tokens.

    `tokenize_to_tokens()` remains the authoritative tokenisation path.
    Offset mapping is used only to map already-identified token spans back
    onto text for T5 masking/reconstruction.
    """
    tokens = tokenize_to_tokens(
        text,
        tokenizer=tokenizer,
        lowercase=lowercase,
    )

    working_text = (
        text.casefold()
        if lowercase
        else text
    )

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "The n-gram tokenizer must be a fast Hugging Face tokenizer "
            "because offset_mapping is required to map token spans back "
            "onto source text."
        )

    encoded = tokenizer(
        working_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
    )

    offsets = [
        tuple(x)
        for x in encoded["offset_mapping"]
    ]

    encoded_ids = encoded.get("input_ids", [])

    encoded_tokens = tokenizer.convert_ids_to_tokens(
        encoded_ids
    )

    if len(tokens) != len(offsets):
        raise ValueError(
            "Canonical tokenisation and offset tokenisation returned "
            f"different lengths: {len(tokens)} vs {len(offsets)}."
        )

    if list(tokens) != list(encoded_tokens):
        raise ValueError(
            "Canonical tokenisation and offset tokenisation returned "
            "different token sequences."
        )

    # Character offsets were calculated over working_text. In normal English
    # lowercasing/casefolding preserves character positions, so we can retain
    # the original surface form for output. If casefolding changed the string
    # length, use the exact string against which offsets were calculated.
    if len(working_text) == len(text):
        offset_text = text
        offset_text_is_casefolded = False
    else:
        offset_text = working_text
        offset_text_is_casefolded = True

    return {
        "tokens": list(tokens),
        "offsets": offsets,
        "working_text": working_text,
        "offset_text": offset_text,
        "offset_text_is_casefolded": offset_text_is_casefolded,
    }


def find_ngram_occurrence_spans(
    full_tokens: Sequence[Any],
    ngram_tokens: Sequence[Any],
    *,
    all_ngrams: Optional[Sequence[Sequence[Any]]] = None,
    greatest_common: bool = False,
    allow_overlaps: bool = False,
) -> List[Tuple[int, int]]:
    """
    Find unknown-text occurrences using the existing n_gram_tracing helpers.

    Returns token spans as:
        (start, end)
    where end is exclusive.

    This mirrors the occurrence choice already used by score_ngrams_to_df():
      - ordinary n-grams -> find_all_token_ngram_spans()
      - greatest_common -> find_independent_token_ngram_spans()
    """
    if greatest_common:
        if all_ngrams is None:
            raise ValueError(
                "all_ngrams must be supplied when greatest_common=True"
            )

        return find_independent_token_ngram_spans(
            tokens=list(full_tokens),
            ngram_tokens=list(ngram_tokens),
            all_ngrams=all_ngrams,
            start=0,
            allow_overlaps=allow_overlaps,
        )

    return find_all_token_ngram_spans(
        tokens=list(full_tokens),
        ngram_tokens=list(ngram_tokens),
        start=0,
        allow_overlaps=allow_overlaps,
    )


def create_expanded_occurrence_spans(
    source_token_data: Dict[str, Any],
    occurrence_spans: Sequence[Tuple[int, int]],
    *,
    max_left_expansion: int = 2,
    max_right_expansion: int = 2,
) -> List[Dict[str, Any]]:
    """
    Create every containing span obtained by expanding each n-gram
    occurrence left/right in the ORIGINAL n-gram tokenizer space.

    The target n-gram itself therefore remains defined by the same tokenizer
    as the authorship-verification method; T5 does not redefine its boundaries.
    """
    if max_left_expansion < 0:
        raise ValueError(
            "max_left_expansion must be >= 0"
        )

    if max_right_expansion < 0:
        raise ValueError(
            "max_right_expansion must be >= 0"
        )

    tokens = source_token_data["tokens"]
    offsets = source_token_data["offsets"]
    offset_text = source_token_data["offset_text"]

    spans = []

    for occurrence_index, (ngram_start, ngram_end) in enumerate(
        occurrence_spans,
        start=1,
    ):
        seen = set()

        for left_expansion in range(
            max_left_expansion + 1
        ):
            for right_expansion in range(
                max_right_expansion + 1
            ):
                span_start = max(
                    0,
                    ngram_start - left_expansion,
                )

                span_end = min(
                    len(tokens),
                    ngram_end + right_expansion,
                )

                key = (span_start, span_end)

                if key in seen:
                    continue

                seen.add(key)

                char_start = offsets[span_start][0]
                char_end = offsets[span_end - 1][1]

                span_tokens = list(
                    tokens[span_start:span_end]
                )

                spans.append({
                    "occurrence_index": occurrence_index,

                    "ngram_token_start": ngram_start,
                    "ngram_token_end": ngram_end,

                    "span_token_start": span_start,
                    "span_token_end": span_end,

                    "left_expansion": left_expansion,
                    "right_expansion": right_expansion,

                    "char_start": char_start,
                    "char_end": char_end,

                    "span_tokens": span_tokens,

                    # Surface form from source text.
                    "original_span": offset_text[
                        char_start:char_end
                    ],
                })

    return spans


def build_local_t5_context(
    source_token_data: Dict[str, Any],
    span: Dict[str, Any],
    *,
    context_tokens: Optional[int] = 128,
    lowercase_input: bool = True,
) -> Dict[str, Any]:
    """
    Build local text containing context on BOTH sides of an expanded span.

    Context size is measured using the ORIGINAL n-gram tokenizer so that the
    operation is consistent with the existing token-span pipeline.

    If context_tokens=None, the complete document context on both sides is
    retained.

    Unlike get_trimmed_context_before_span(), this is deliberately symmetric
    because T5 span infilling needs both left and right context.
    """
    tokens = source_token_data["tokens"]
    offsets = source_token_data["offsets"]
    offset_text = source_token_data["offset_text"]

    span_start = span["span_token_start"]
    span_end = span["span_token_end"]

    if context_tokens is None:
        context_start = 0
        context_end = len(tokens)
    else:
        if context_tokens < 0:
            raise ValueError(
                "context_tokens must be >= 0 or None"
            )

        context_start = max(
            0,
            span_start - context_tokens,
        )

        context_end = min(
            len(tokens),
            span_end + context_tokens,
        )

    context_char_start = offsets[
        context_start
    ][0]

    context_char_end = offsets[
        context_end - 1
    ][1]

    char_start = span["char_start"]
    char_end = span["char_end"]

    left_context = offset_text[
        context_char_start:char_start
    ]

    original_span = offset_text[
        char_start:char_end
    ]

    right_context = offset_text[
        char_end:context_char_end
    ]

    if lowercase_input:
        generation_left = left_context.casefold()
        generation_span = original_span.casefold()
        generation_right = right_context.casefold()
    else:
        generation_left = left_context
        generation_span = original_span
        generation_right = right_context

    masked_text = (
        generation_left
        + "<extra_id_0>"
        + generation_right
    )

    original_context = (
        left_context
        + original_span
        + right_context
    )

    generation_original_context = (
        generation_left
        + generation_span
        + generation_right
    )

    return {
        "context_token_start": context_start,
        "context_token_end": context_end,
        "context_char_start": context_char_start,
        "context_char_end": context_char_end,

        "left_context": left_context,
        "right_context": right_context,
        "original_context": original_context,

        "generation_left_context": generation_left,
        "generation_right_context": generation_right,
        "generation_original_span": generation_span,
        "generation_original_context": generation_original_context,

        "masked_text": masked_text,
    }


def reconstruct_candidate_context(
    generation_left_context: str,
    generation_right_context: str,
    candidate: str,
    starts_with_space: bool,
    *,
    lowercase_output: bool = True,
) -> str:
    """
    Reconstruct the local T5 input after replacing <extra_id_0>.

    T5 SentencePiece drops a leading whitespace when an isolated generated
    span is decoded. `starts_with_space` restores T5's own boundary decision.
    """
    replacement = candidate

    if (
        starts_with_space
        and generation_left_context
        and not generation_left_context[-1].isspace()
        and replacement
        and not replacement[0].isspace()
    ):
        replacement = " " + replacement

    reconstructed = (
        generation_left_context
        + replacement
        + generation_right_context
    )

    if lowercase_output:
        reconstructed = reconstructed.casefold()

    return reconstructed


def ngram_tokens_to_text(
    ngram_tokens: Sequence[Any],
    tokenizer: Any,
) -> str:
    """
    Convenience wrapper around the existing canonical decoder.
    """
    return tokens_to_text(
        list(ngram_tokens),
        tokenizer,
    )