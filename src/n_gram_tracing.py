import re
from typing import Any, List, Optional, Sequence, Tuple, Union


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

    Output format:
        [[tok1, tok2, ...], [tok1, tok2, ...], ...]
    """

    def _all_ngrams(seq: Sequence[Any]) -> set[tuple[Any, ...]]:
        out: set[tuple[Any, ...]] = set()
        L = len(seq)
        for k in range(1, L + 1):
            for i in range(0, L - k + 1):
                out.add(tuple(seq[i:i + k]))
        return out

    seq1 = tokenize_to_tokens(text1, tokenizer=tokenizer, lowercase=lowercase)
    seq2 = tokenize_to_tokens(text2, tokenizer=tokenizer, lowercase=lowercase)

    common = _all_ngrams(seq1) & _all_ngrams(seq2)
    if not common:
        return []

    if not include_subgrams:
        common_sorted = sorted(common, key=len, reverse=True)
        kept: List[Tuple[Any, ...]] = []

        for g in common_sorted:
            is_subspan = False
            for h in kept:
                if len(h) <= len(g):
                    continue
                for i in range(0, len(h) - len(g) + 1):
                    if h[i:i + len(g)] == g:
                        is_subspan = True
                        break
                if is_subspan:
                    break
            if not is_subspan:
                kept.append(g)

        common = set(kept)

    return [list(g) for g in sorted(common, key=lambda x: (len(x), x))]


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