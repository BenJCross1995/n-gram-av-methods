import re
from typing import Any, List, Optional, Sequence, Tuple, Union, Dict, Set
from collections import defaultdict

# -------------------------------------------------------------------
# SHARED TOKEN HELPERS
# -------------------------------------------------------------------

def tokenize_to_tokens(
    text: str,
    tokenizer: Optional[Any] = None,
    *,
    lowercase: bool = True,
) -> List[Any]:
    """
    Canonical tokenisation path used everywhere.

    If tokenizer is None:
        falls back to regex word tokenisation

    If tokenizer is provided:
        text -> input_ids -> tokens
    """
    text_in = text.casefold() if lowercase else text

    if tokenizer is None:
        return re.findall(r"\w+", text_in)

    enc = tokenizer(
        text_in,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = enc.get("input_ids", [])

    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(input_ids)

    if hasattr(tokenizer, "tokenize"):
        return list(tokenizer.tokenize(text_in))

    return list(input_ids)


def tokens_to_ids(
    tokens: Sequence[Any],
    tokenizer: Any,
) -> List[int]:
    """
    Convert token strings to ids.
    """
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        raise TypeError("tokenizer must have convert_tokens_to_ids()")

    ids = tokenizer.convert_tokens_to_ids(list(tokens))
    if any(i is None for i in ids):
        raise ValueError("Some tokens could not be converted to ids.")

    return list(ids)


def tokens_to_text(tokens: List[str], tokenizer: Any) -> str:
    """
    Convert a list of tokenizer tokens back into normal text using a Hugging Face tokenizer.
    """
    if not tokens:
        return ""

    if not hasattr(tokenizer, "convert_tokens_to_ids") or not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer must have convert_tokens_to_ids() and decode()")

    ids = tokenizer.convert_tokens_to_ids(tokens)

    if any(i is None for i in ids):
        raise ValueError("Some tokens could not be converted to ids by this tokenizer.")

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


# -------------------------------------------------------------------
# COMMON NGRAMS
# -------------------------------------------------------------------

def common_ngrams(
    text1: str,
    text2: str,
    tokenizer: Optional[Any] = None,
    include_subgrams: bool = False,
    lowercase: bool = True,
) -> List[List[Any]]:
    """
    Return shared n-grams between two texts as a list of token lists.

    If include_subgrams is False, removes any shared n-gram that is a contiguous
    subspan of a longer shared n-gram.

    This version does NOT materialize all possible n-grams of both texts.
    Instead, it finds maximal common runs directly across alignment diagonals.
    """

    seq1 = tokenize_to_tokens(text1, tokenizer=tokenizer, lowercase=lowercase)
    seq2 = tokenize_to_tokens(text2, tokenizer=tokenizer, lowercase=lowercase)

    n = len(seq1)
    m = len(seq2)

    if n == 0 or m == 0:
        return []

    # ---------------------------------------------------------
    # Collect maximal equal runs across all diagonals
    # ---------------------------------------------------------
    candidates: set[Tuple[Any, ...]] = set()

    # delta = i - j
    for delta in range(-(m - 1), n):
        i = max(0, delta)
        j = max(0, -delta)

        run_start_i = None
        run_len = 0

        while i < n and j < m:
            if seq1[i] == seq2[j]:
                if run_start_i is None:
                    run_start_i = i
                    run_len = 1
                else:
                    run_len += 1
            else:
                if run_start_i is not None and run_len > 0:
                    candidates.add(tuple(seq1[run_start_i:run_start_i + run_len]))
                    run_start_i = None
                    run_len = 0

            i += 1
            j += 1

        # flush run at end of diagonal
        if run_start_i is not None and run_len > 0:
            candidates.add(tuple(seq1[run_start_i:run_start_i + run_len]))

    if not candidates:
        return []

    # ---------------------------------------------------------
    # Optional global subgram removal
    # Keeps only n-grams that are not contiguous subspans
    # of any longer shared candidate.
    # ---------------------------------------------------------
    if not include_subgrams:
        candidates_sorted = sorted(candidates, key=len, reverse=True)
        kept: List[Tuple[Any, ...]] = []

        for g in candidates_sorted:
            is_subspan = False
            g_len = len(g)

            for h in kept:
                h_len = len(h)
                if h_len <= g_len:
                    continue

                for s in range(0, h_len - g_len + 1):
                    if h[s:s + g_len] == g:
                        is_subspan = True
                        break

                if is_subspan:
                    break

            if not is_subspan:
                kept.append(g)

        candidates = set(kept)

    return [list(g) for g in sorted(candidates, key=lambda x: (len(x), x))]

def filter_len_common_ngrams(
    common_ngram_list: List[List[Any]],
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> List[List[Any]]:
    """
    Filter a list of common n-grams so only those with length between
    min_len and max_len are kept.
    """
    if min_len is not None and min_len < 1:
        raise ValueError("min_len must be >= 1 or None")
    if max_len is not None and max_len < 1:
        raise ValueError("max_len must be >= 1 or None")
    if min_len is not None and max_len is not None and min_len > max_len:
        raise ValueError("min_len cannot be greater than max_len")

    return [
        gram
        for gram in common_ngram_list
        if (min_len is None or len(gram) >= min_len)
        and (max_len is None or len(gram) <= max_len)
    ]

# -------------------------------------------------------------------
# New Functions For Greatest Common N-Grams
# -------------------------------------------------------------------

def find_ngram_starts(
    tokens: Sequence[Any],
    ngram: Sequence[Any],
) -> List[int]:
    """
    Return all start positions where ngram appears in tokens.
    """
    ngram = tuple(ngram)
    n = len(ngram)

    if n == 0:
        return []

    return [
        i
        for i in range(len(tokens) - n + 1)
        if tuple(tokens[i:i + n]) == ngram
    ]


def subgram_offsets(
    child: Sequence[Any],
    parent: Sequence[Any],
) -> List[int]:
    """
    Return offsets where child appears as a contiguous subspan of parent.
    """
    child = tuple(child)
    parent = tuple(parent)

    if len(child) >= len(parent):
        return []

    child_len = len(child)

    return [
        i
        for i in range(len(parent) - child_len + 1)
        if parent[i:i + child_len] == child
    ]


def ngram_occurrence_stats(
    ngrams: Sequence[Sequence[Any]],
    text: str,
    tokenizer: Optional[Any] = None,
    *,
    lowercase: bool = True,
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    """
    For one text, count each n-gram and identify whether its occurrences
    are covered by larger n-grams.

    An n-gram is kept if it has at least one occurrence outside any larger
    n-gram that contains it.
    """
    tokens = tokenize_to_tokens(
        text,
        tokenizer=tokenizer,
        lowercase=lowercase,
    )

    # deduplicate while preserving order
    grams = list(dict.fromkeys(tuple(g) for g in ngrams if len(g) > 0))

    starts_by_ngram = {
        g: find_ngram_starts(tokens, g)
        for g in grams
    }

    covered_starts_by_ngram: Dict[Tuple[Any, ...], Set[int]] = {
        g: set()
        for g in grams
    }

    for child in grams:
        for parent in grams:
            offsets = subgram_offsets(child, parent)

            if not offsets:
                continue

            for parent_start in starts_by_ngram[parent]:
                for offset in offsets:
                    covered_starts_by_ngram[child].add(parent_start + offset)

    stats = {}

    for g in grams:
        starts = set(starts_by_ngram[g])
        covered_starts = starts & covered_starts_by_ngram[g]
        outside_starts = starts - covered_starts

        stats[g] = {
            "ngram": list(g),
            "num_tokens": len(g),
            "total_count": len(starts),
            "covered_count": len(covered_starts),
            "outside_count": len(outside_starts),
            "starts": sorted(starts),
            "covered_starts": sorted(covered_starts),
            "outside_starts": sorted(outside_starts),
            "keep": len(outside_starts) > 0,
        }

    return stats


def filter_ngrams_with_outside_occurrences(
    ngrams: Sequence[Sequence[Any]],
    text: str,
    tokenizer: Optional[Any] = None,
    *,
    lowercase: bool = True,
) -> List[List[Any]]:
    """
    Return only n-grams that occur at least once outside any larger n-gram.
    """
    stats = ngram_occurrence_stats(
        ngrams=ngrams,
        text=text,
        tokenizer=tokenizer,
        lowercase=lowercase,
    )

    return [
        row["ngram"]
        for row in stats.values()
        if row["keep"]
    ]

def filter_ngrams_with_outside_occurrences_in_both_texts(
    ngrams,
    known_text,
    unknown_text,
    tokenizer=None,
    *,
    lowercase=True,
):
    """
    Keep n-grams only if they have at least one occurrence outside longer
    candidate n-grams in BOTH the known text and the unknown text.
    """
    known_stats = ngram_occurrence_stats(
        ngrams=ngrams,
        text=known_text,
        tokenizer=tokenizer,
        lowercase=lowercase,
    )

    unknown_stats = ngram_occurrence_stats(
        ngrams=ngrams,
        text=unknown_text,
        tokenizer=tokenizer,
        lowercase=lowercase,
    )

    kept = []

    for g in dict.fromkeys(tuple(x) for x in ngrams):
        known_keep = known_stats.get(g, {}).get("keep", False)
        unknown_keep = unknown_stats.get(g, {}).get("keep", False)

        if known_keep and unknown_keep:
            kept.append(list(g))

    return kept

# -------------------------------------------------------------------
# TOKEN-BASED MATCHING
# -------------------------------------------------------------------

def find_all_token_ngram_spans(
    tokens: List[Any],
    ngram_tokens: List[Any],
    *,
    start: int = 0,
    allow_overlaps: bool = False,
) -> List[Tuple[int, int]]:
    """
    Find all spans of ngram_tokens inside tokens.

    Returns:
        [(start_idx, end_idx), ...]  # end exclusive
    """
    if not ngram_tokens:
        raise ValueError("ngram_tokens must be non-empty")
    if start < 0:
        raise ValueError("start must be >= 0")

    spans: List[Tuple[int, int]] = []
    n = len(ngram_tokens)
    i = start

    while i <= len(tokens) - n:
        if tokens[i:i + n] == ngram_tokens:
            spans.append((i, i + n))
            i += 1 if allow_overlaps else n
        else:
            i += 1

    return spans


def texts_before_each_token_ngram(
    text: str,
    ngram_tokens: List[Any],
    *,
    tokenizer: Any,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False,
    return_positions: bool = False,
) -> Union[List[str], List[Tuple[str, Tuple[int, int]]]]:
    """
    Return decoded text before each token-ngram occurrence.
    """
    full_tokens = tokenize_to_tokens(text, tokenizer=tokenizer, lowercase=lowercase)
    spans = find_all_token_ngram_spans(
        full_tokens,
        ngram_tokens,
        start=start,
        allow_overlaps=allow_overlaps,
    )

    out = []
    for s, _ in spans:
        prefix_tokens = full_tokens[:s]
        prefix_text = tokens_to_text(prefix_tokens, tokenizer)
        if return_positions:
            out.append((prefix_text, (s, s)))
        else:
            out.append(prefix_text)

    return out


def texts_around_each_token_ngram(
    text: str,
    ngram_tokens: List[Any],
    *,
    tokenizer: Any,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False,
    return_spans: bool = False,
    return_tokenized_text: bool = False,
):
    """
    For each token-ngram match, return the decoded text up to the end of the match.
    """
    full_tokens = tokenize_to_tokens(text, tokenizer=tokenizer, lowercase=lowercase)
    spans = find_all_token_ngram_spans(
        full_tokens,
        ngram_tokens,
        start=start,
        allow_overlaps=allow_overlaps,
    )

    prefix_through_end = [
        tokens_to_text(full_tokens[:end], tokenizer)
        for _, end in spans
    ]

    if return_spans and return_tokenized_text:
        return prefix_through_end, spans, full_tokens
    if return_spans:
        return prefix_through_end, spans
    if return_tokenized_text:
        return prefix_through_end, full_tokens
    return prefix_through_end


def get_trimmed_context_before_span(
    tokens: List[Any],
    token_span: Tuple[int, int],
    max_tokens: Optional[int] = None,
    return_text: bool = False,
    tokenizer: Optional[Any] = None,
):
    """
    Return context + phrase using token spans.
    """
    start, end = token_span
    context = tokens[:start]
    phrase = tokens[start:end]

    if max_tokens is not None and len(context) > max_tokens:
        context = context[-max_tokens:]

    combined = context + phrase

    if not return_text:
        return combined

    if tokenizer is None:
        raise ValueError("tokenizer must be provided when return_text=True")

    return tokens_to_text(combined, tokenizer)
def is_subspan_tokens(
    child: Sequence[Any],
    parent: Sequence[Any],
) -> bool:
    """
    True if child appears as a contiguous subspan inside parent.
    """
    child = tuple(child)
    parent = tuple(parent)

    if len(child) >= len(parent):
        return False

    child_len = len(child)

    return any(
        parent[i:i + child_len] == child
        for i in range(len(parent) - child_len + 1)
    )


def find_independent_token_ngram_spans(
    tokens: List[Any],
    ngram_tokens: List[Any],
    all_ngrams: Sequence[Sequence[Any]],
    *,
    start: int = 0,
    allow_overlaps: bool = False,
) -> List[Tuple[int, int]]:
    """
    Find occurrences of ngram_tokens that are NOT part of any other retained
    longer n-gram.

    So if ["i", "am"] is retained, but also appears inside:
        [".", "i", "see", "therefore", "i", "am"]
    then that occurrence is excluded.

    But an independent occurrence like:
        ["i", "am", ...]
    is kept, unless it is itself covered by a different longer retained n-gram
    such as ["i", "am", "dead"].
    """
    ngram_tuple = tuple(ngram_tokens)

    target_spans = find_all_token_ngram_spans(
        tokens=tokens,
        ngram_tokens=list(ngram_tokens),
        start=start,
        allow_overlaps=allow_overlaps,
    )

    blocking_spans = []

    for other_ngram in all_ngrams:
        other_tuple = tuple(other_ngram)

        # skip itself
        if other_tuple == ngram_tuple:
            continue

        # only longer n-grams can block this n-gram
        if len(other_tuple) <= len(ngram_tuple):
            continue

        # only relevant if target is a subspan of the longer n-gram
        if not is_subspan_tokens(ngram_tuple, other_tuple):
            continue

        blocking_spans.extend(
            find_all_token_ngram_spans(
                tokens=tokens,
                ngram_tokens=list(other_ngram),
                start=start,
                allow_overlaps=allow_overlaps,
            )
        )

    independent_spans = []

    for span in target_spans:
        s, e = span

        is_inside_other_ngram = any(
            block_s <= s and e <= block_e
            for block_s, block_e in blocking_spans
        )

        if not is_inside_other_ngram:
            independent_spans.append(span)

    return independent_spans

def texts_around_each_independent_token_ngram(
    text: str,
    ngram_tokens: List[Any],
    all_ngrams: Sequence[Sequence[Any]],
    *,
    tokenizer: Any,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False,
    return_spans: bool = False,
    return_tokenized_text: bool = False,
):
    full_tokens = tokenize_to_tokens(
        text,
        tokenizer=tokenizer,
        lowercase=lowercase,
    )

    spans = find_independent_token_ngram_spans(
        tokens=full_tokens,
        ngram_tokens=ngram_tokens,
        all_ngrams=all_ngrams,
        start=start,
        allow_overlaps=allow_overlaps,
    )

    prefix_through_end = [
        tokens_to_text(full_tokens[:end], tokenizer)
        for _, end in spans
    ]

    if return_spans and return_tokenized_text:
        return prefix_through_end, spans, full_tokens
    if return_spans:
        return prefix_through_end, spans
    if return_tokenized_text:
        return prefix_through_end, full_tokens
    return prefix_through_end