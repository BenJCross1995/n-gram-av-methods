import re
from typing import Any, List, Sequence, Optional, Tuple

def common_ngrams(
    text1: str,
    text2: str,
    n: int,
    tokenizer: Optional[Any] = None,
    include_subgrams: bool = False,
    lowercase: bool = True,
) -> List[List[Any]]:
    """
    Return shared n-grams (length >= n) between two texts as a list of lists.

    If include_subgrams is False, removes any shared n-gram that is a contiguous
    subspan of a longer shared n-gram.

    Output format: [[tok1, tok2, ...], [tok1, tok2, ...], ...]
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    def _tokens(s: str) -> List[Any]:
        s2 = s.casefold() if lowercase else s
        if tokenizer is None:
            return re.findall(r"\w+", s2)

        # Hugging Face tokenizers
        if hasattr(tokenizer, "tokenize"):
            return list(tokenizer.tokenize(s2))

        enc = tokenizer(
            s2,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            return tokenizer.convert_ids_to_tokens(input_ids)
        return input_ids

    def _all_ngrams(seq: Sequence[Any], min_n: int) -> set[tuple[Any, ...]]:
        out: set[tuple[Any, ...]] = set()
        L = len(seq)
        for k in range(min_n, L + 1):
            for i in range(0, L - k + 1):
                out.add(tuple(seq[i : i + k]))
        return out

    seq1 = _tokens(text1)
    seq2 = _tokens(text2)

    common = _all_ngrams(seq1, n) & _all_ngrams(seq2, n)
    if not common:
        return []

    if not include_subgrams:
        # Keep only maximal n-grams (drop any that is a contiguous subspan of a longer one)
        common_sorted = sorted(common, key=len, reverse=True)
        kept: list[tuple[Any, ...]] = []
        for g in common_sorted:
            # if g appears as a contiguous subspan of any already-kept longer gram, skip it
            is_subspan = False
            for h in kept:  # h is longer or equal (because sorted desc)
                if len(h) <= len(g):
                    continue
                # contiguous subspan check
                for i in range(0, len(h) - len(g) + 1):
                    if h[i : i + len(g)] == g:
                        is_subspan = True
                        break
                if is_subspan:
                    break
            if not is_subspan:
                kept.append(g)
        common = set(kept)

    # Return as list[list]
    return [list(g) for g in sorted(common, key=lambda x: (len(x), x))]

def tokens_to_text(tokens: List[str], tokenizer: Any) -> str:
    """
    Convert a list of tokenizer tokens back into normal text using a Hugging Face tokenizer.

    Works best for BPE/SentencePiece tokenizers that support:
      - convert_tokens_to_ids
      - decode
    """
    if not tokens:
        return ""

    if not hasattr(tokenizer, "convert_tokens_to_ids") or not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer must have convert_tokens_to_ids() and decode()")

    ids = tokenizer.convert_tokens_to_ids(tokens)

    # Some tokenizers may return None or an "unknown" id for weird tokens;
    # fall back to a safer path if that happens.
    if any(i is None for i in ids):
        raise ValueError("Some tokens could not be converted to ids by this tokenizer.")

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# -----FUNCTIONS TO FIND TEXT BEFORE N-GRAM----- #

def find_all_ngram_positions(
    text: str,
    ngram: str,
    *,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False
) -> List[int]:
    """
    Return a list of start indices for every occurrence of `ngram` in `text`.
    """
    if not ngram:
        raise ValueError("ngram must be non-empty")
    if start < 0:
        raise ValueError("start must be >= 0")

    hay = text.casefold() if lowercase else text
    needle = ngram.casefold() if lowercase else ngram

    positions: List[int] = []
    i = start
    step = 1 if allow_overlaps else len(needle)

    while True:
        idx = hay.find(needle, i)
        if idx == -1:
            break
        positions.append(idx)
        i = idx + step

    return positions


def texts_before_each_ngram(
    text: str,
    ngram: str,
    *,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False,
    return_positions: bool = False
) -> List[str] | List[Tuple[str, int]]:
    """
    Return a list of the text-before substring for each occurrence of `ngram`.

    If return_positions=True, returns [(text_before, idx), ...].
    """
    positions = find_all_ngram_positions(
        text, ngram, start=start, lowercase=lowercase, allow_overlaps=allow_overlaps
    )

    if return_positions:
        return [(text[:idx], idx) for idx in positions]
    return [text[:idx] for idx in positions]

# -----FUNCTIONS TO FIND TEXT INCLUDING N-GRAM----- #

def find_all_ngram_spans(
    text: str,
    ngram: str,
    *,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False
) -> List[Tuple[int, int]]:
    """
    Return [(start_idx, end_idx), ...] for every occurrence of `ngram` in `text`.
    end_idx is exclusive (i.e., slice-ready: text[start:end]).
    """
    if not ngram:
        raise ValueError("ngram must be non-empty")
    if start < 0:
        raise ValueError("start must be >= 0")

    hay = text.casefold() if lowercase else text
    needle = ngram.casefold() if lowercase else ngram

    spans: List[Tuple[int, int]] = []
    i = start
    step = 1 if allow_overlaps else len(needle)

    while True:
        s = hay.find(needle, i)
        if s == -1:
            break
        e = s + len(needle)
        spans.append((s, e))
        i = s + step

    return spans

def texts_around_each_ngram(
    text: str,
    ngram: str,
    *,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False,
    return_spans: bool = False
) -> Any:
    """
    For each occurrence of `ngram`, return:
      - prefix_before: text[:start_idx]
      - prefix_through_end: text[:end_idx]

    If return_spans=True, returns (prefix_before_list, prefix_through_end_list, spans)
    else returns (prefix_before_list, prefix_through_end_list)
    """
    spans = find_all_ngram_spans(
        text, ngram, start=start, lowercase=lowercase, allow_overlaps=allow_overlaps
    )

    prefix_through_end = [text[:e] for _, e in spans]

    if return_spans:
        return prefix_through_end, spans
    return prefix_through_end