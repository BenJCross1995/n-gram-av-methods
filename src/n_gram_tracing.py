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

# --- helper: convert char spans -> token spans (best with HF *fast* tokenizer) ---

def _char_spans_to_token_spans(
    text: str,
    char_spans: List[Tuple[int, int]],
    tokenizer: Any,
) -> List[Tuple[int, int]]:
    """
    Convert [(char_start, char_end), ...] -> [(tok_start, tok_end), ...].

    Preferred path uses return_offsets_mapping=True (fast tokenizers).
    Fallback path counts tokens in prefixes (can be approximate for some tokenizers).
    """
    # --- preferred: offsets mapping ---
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        offsets = enc.get("offset_mapping", None)
        if offsets is None:
            raise ValueError("No offset_mapping")

        # batch vs non-batch
        if offsets and isinstance(offsets[0], (list, tuple)) and len(offsets[0]) == 2 and isinstance(offsets[0][0], int):
            token_offsets = offsets  # [(s,e), ...]
        else:
            token_offsets = offsets[0]  # [[(s,e), ...]] -> take first

        token_spans: List[Tuple[int, int]] = []
        for s_char, e_char in char_spans:
            t_start = None
            t_end = None

            for i, (ts, te) in enumerate(token_offsets):
                if ts is None or te is None or ts == te:
                    continue
                if (te > s_char) and (ts < e_char):  # overlaps char span
                    if t_start is None:
                        t_start = i
                    t_end = i + 1  # exclusive

            if t_start is None:
                # no overlap (e.g. whitespace-only match) -> empty span at nearest boundary
                before = 0
                for ts, te in token_offsets:
                    if ts is None or te is None or ts == te:
                        continue
                    if te <= s_char:
                        before += 1
                token_spans.append((before, before))
            else:
                token_spans.append((t_start, t_end))  # type: ignore[arg-type]

        return token_spans

    except Exception:
        # --- fallback: token counts in prefixes (approx for some tokenizers) ---
        token_spans: List[Tuple[int, int]] = []
        for s_char, e_char in char_spans:
            try:
                s_ids = tokenizer(text[:s_char], add_special_tokens=False).get("input_ids")
                e_ids = tokenizer(text[:e_char], add_special_tokens=False).get("input_ids")
                if s_ids is None or e_ids is None:
                    raise TypeError
                if s_ids and isinstance(s_ids[0], (list, tuple)):
                    s_ids = s_ids[0]
                if e_ids and isinstance(e_ids[0], (list, tuple)):
                    e_ids = e_ids[0]
                token_spans.append((len(s_ids), len(e_ids)))
            except Exception:
                # last-ditch for tokenizers without __call__ dict output
                s_ids = tokenizer.encode(text[:s_char], add_special_tokens=False)
                e_ids = tokenizer.encode(text[:e_char], add_special_tokens=False)
                token_spans.append((len(s_ids), len(e_ids)))
        return token_spans


# -----FUNCTIONS TO FIND TEXT INCLUDING N-GRAM----- 

def _tokenize_full_text(text: str, tokenizer: Any) -> List[Any]:
    """
    Tokenise the full text in a way that aligns with token spans (i.e., ids->tokens).
    Returns a list of tokens if possible, otherwise returns ids.
    """
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc.get("input_ids")
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
        if input_ids is None:
            raise TypeError
    except Exception:
        input_ids = tokenizer.encode(text, add_special_tokens=False)

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(input_ids)

    # fallback
    if hasattr(tokenizer, "tokenize"):
        return list(tokenizer.tokenize(text))

    return input_ids

def find_all_ngram_spans(
    text: str,
    ngram: str,
    *,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False,
    tokenizer: Optional[Any] = None,
    return_token_spans: bool = False,
    return_tokenized_text: bool = False,
) -> Any:
    """
    Return spans for every occurrence of `ngram` in `text`.

    Default (unchanged): returns [(char_start, char_end), ...] (end exclusive).

    If return_token_spans=True, requires tokenizer and returns:
      (char_spans, token_spans)

    If return_tokenized_text=True, requires tokenizer and appends:
      tokenized_text

    Return shapes:
      - neither flag: char_spans
      - token only: (char_spans, token_spans)
      - tokenized only: (char_spans, tokenized_text)
      - both: (char_spans, token_spans, tokenized_text)
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

    # old behaviour
    if not return_token_spans and not return_tokenized_text:
        return spans

    if tokenizer is None:
        raise ValueError(
            "tokenizer must be provided when return_token_spans=True or return_tokenized_text=True"
        )

    token_spans = None
    tokenized_text = None

    if return_token_spans:
        token_spans = _char_spans_to_token_spans(text, spans, tokenizer)

    if return_tokenized_text:
        tokenized_text = _tokenize_full_text(text, tokenizer)

    if return_token_spans and return_tokenized_text:
        return spans, token_spans, tokenized_text
    if return_token_spans:
        return spans, token_spans
    # return_tokenized_text only
    return spans, tokenized_text

def texts_around_each_ngram(
    text: str,
    ngram: str,
    *,
    start: int = 0,
    lowercase: bool = True,
    allow_overlaps: bool = False,
    return_spans: bool = False,
    tokenizer: Optional[Any] = None,
    return_token_spans: bool = False,
    return_tokenized_text: bool = False,
) -> Any:
    """
    For each occurrence of `ngram`, return:
      - prefix_through_end: text[:end_idx]   (end_idx is character-based, unchanged)

    Default (unchanged):
      - return_spans=False -> prefix_through_end_list
      - return_spans=True  -> (prefix_through_end_list, char_spans)

    If return_token_spans=True (requires tokenizer):
      - return_spans=False -> (prefix_through_end_list, token_spans)
      - return_spans=True  -> (prefix_through_end_list, char_spans, token_spans)

    If return_tokenized_text=True (requires tokenizer), appends tokenised full text.
    """
    if (return_token_spans or return_tokenized_text) and tokenizer is None:
        raise ValueError(
            "tokenizer must be provided when return_token_spans=True or return_tokenized_text=True"
        )

    spans_out = find_all_ngram_spans(
        text,
        ngram,
        start=start,
        lowercase=lowercase,
        allow_overlaps=allow_overlaps,
        tokenizer=tokenizer,
        return_token_spans=return_token_spans,
        return_tokenized_text=return_tokenized_text,
    )

    # unpack outputs from find_all_ngram_spans
    token_spans = None
    tokenized_text = None

    if return_token_spans and return_tokenized_text:
        char_spans, token_spans, tokenized_text = spans_out
    elif return_token_spans:
        char_spans, token_spans = spans_out
    elif return_tokenized_text:
        char_spans, tokenized_text = spans_out
    else:
        char_spans = spans_out

    prefix_through_end = [text[:e] for _, e in char_spans]

    # Build return values while preserving existing behaviour unless new flags are on
    if return_spans:
        if return_token_spans and return_tokenized_text:
            return prefix_through_end, char_spans, token_spans, tokenized_text
        if return_token_spans:
            return prefix_through_end, char_spans, token_spans
        if return_tokenized_text:
            return prefix_through_end, char_spans, tokenized_text
        return prefix_through_end, char_spans

    # not return_spans
    if return_token_spans and return_tokenized_text:
        return prefix_through_end, token_spans, tokenized_text
    if return_token_spans:
        return prefix_through_end, token_spans
    if return_tokenized_text:
        return prefix_through_end, tokenized_text
    return prefix_through_end

def get_trimmed_context_before_span(
    tokens: List[Any],
    token_span: Tuple[int, int],
    max_tokens: Optional[int] = None,
    return_text: bool = False,
    tokenizer: Optional[Any] = None
):
    """
    Return the context tokens before a span (optionally trimmed to `max_tokens`),
    and the phrase tokens at that span.

    Parameters
    ----------
    tokens : list
        Tokenised full text (e.g. from tokenizer.convert_ids_to_tokens(...))
    token_span : tuple[int, int]
        Token span of the phrase (start, end), end-exclusive.
    max_tokens : int or None, optional
        If set, trims the context before the phrase to the last `max_tokens`.

    Returns
    -------
    context_tokens : list
        Tokens before the span, optionally trimmed
    phrase_tokens : list
        The tokens in the span itself
    """
    start, end = token_span
    context = tokens[:start]
    phrase = tokens[start:end]

    if max_tokens is not None and len(context) > max_tokens:
        context = context[-max_tokens:]  # take last max_tokens only

    if not return_text:
        return context + phrase
    else:
        if not tokenizer:
            raise ValueError(
            "tokenizer must be provided when return_text=True"
            )
        else:
            plain_text = tokens_to_text(context + phrase, tokenizer)
            
        return plain_text