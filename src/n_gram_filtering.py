import ast
import re

import unicodedata

import pandas as pd

from model_loading import load_model, distinct_special_chars

def ensure_tokens_are_lists(df):
    df = df.copy()
    df["tokens"] = df["tokens"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df

def filter_min_length(df, min_tokens: int = 2):
    
    df = df[df['num_tokens'] >= min_tokens]
    
    return df.copy()

def filter_only_special_tokens(df: pd.DataFrame, special_token_list) -> pd.DataFrame:
    """
    REMOVE rows where df['tokens'] contains ONLY special tokens (and at least 1 token).
    Assumes df['tokens'] is a list of token strings per row.
    """
    special = set(special_token_list)
    only_special = df["tokens"].apply(lambda toks: bool(toks) and all(t in special for t in toks))
    return df.loc[~only_special].copy()

def filter_only_numbers_and_special_tokens(df: pd.DataFrame, special_token_list) -> pd.DataFrame:
    """
    REMOVE rows where df['tokens'] contains ONLY numbers plus any strings in special_token_list.
    Example removed: ["<special>19", "<special>97", "112"]  (assuming "<special>" is in special_token_list)
    Assumes df['tokens'] is a list (items can be str/int).
    """
    special = sorted(set(map(str, special_token_list)), key=len, reverse=True)

    def is_only_numbers_plus_special(toks) -> bool:
        s = "".join(map(str, toks))  # combine tokens into one string
        for sp in special:
            s = s.replace(sp, "")
        s = "".join(s.split())       # drop any remaining whitespace
        return bool(s) and bool(re.fullmatch(r"\d+", s))

    mask = df["tokens"].apply(is_only_numbers_plus_special)
    return df.loc[~mask].copy()

def filter_only_punct_and_special_tokens(df: pd.DataFrame, special_token_list) -> pd.DataFrame:
    """
    REMOVE rows where df['tokens'] contains ONLY punctuation and/or special tokens,
    allowing special tokens to appear as substrings inside tokens (e.g. "<special>,").
    Assumes df['tokens'] is a list (items will be cast to str).
    """
    specials = sorted(set(map(str, special_token_list)), key=len, reverse=True)
    special_re = re.compile("|".join(re.escape(s) for s in specials)) if specials else None

    def strip_specials(s: str) -> str:
        s = str(s)
        return special_re.sub("", s) if special_re else s

    def is_punct_or_empty_after_strip(tok) -> bool:
        rem = strip_specials(tok)
        rem = "".join(rem.split())  # drop whitespace
        if rem == "":
            return True
        return all(unicodedata.category(ch).startswith("P") for ch in rem)

    only_punct_or_special = df["tokens"].apply(
        lambda toks: bool(toks) and all(is_punct_or_empty_after_strip(t) for t in toks)
    )

    return df.loc[~only_punct_or_special].copy()

def filter_at_least_n_minus_1_specials(df: pd.DataFrame, special_token_list, n: int | None = None) -> pd.DataFrame:
    """
    REMOVE rows where:
      - if n is set: len(tokens) == n AND (# special-only tokens) >= n-1
      - if n is None: (# special-only tokens) >= len(tokens)-1  (i.e., at most 1 non-special)

    Handles specials embedded inside tokens by stripping special substrings first.
    Assumes df['tokens'] is a list (items cast to str).
    """
    specials = sorted(set(map(str, special_token_list)), key=len, reverse=True)
    special_re = re.compile("|".join(re.escape(s) for s in specials)) if specials else None

    def is_special_only(tok) -> bool:
        s = str(tok)
        if special_re:
            s = special_re.sub("", s)
        s = "".join(s.split())  # drop whitespace
        return s == ""

    def should_remove(toks) -> bool:
        if not toks:
            return False
        L = len(toks)
        if n is not None and L != n:
            return False
        special_count = sum(is_special_only(t) for t in toks)
        threshold = (n - 1) if n is not None else (L - 1)
        return special_count >= threshold

    mask = df["tokens"].apply(should_remove)
    return df.loc[~mask].copy()

def filter_zero_special_tokens(df: pd.DataFrame, special_token_list) -> pd.DataFrame:
    """
    REMOVE rows where df['tokens'] contains ZERO special tokens.
    Treats specials as matching anywhere inside a token (substring match).
    Assumes df['tokens'] is a list (items cast to str).
    """
    specials = sorted(set(map(str, special_token_list)), key=len, reverse=True)
    special_re = re.compile("|".join(re.escape(s) for s in specials)) if specials else None

    def has_any_special(toks) -> bool:
        if not special_re:
            return False
        return any(bool(special_re.search(str(t))) for t in toks)

    zero_special = df["tokens"].apply(lambda toks: not has_any_special(toks))
    return df.loc[~zero_special].copy()


def apply_ngram_filtering(
    df: pd.DataFrame,
    model_loc: str = None,
    token_col: str = 'tokens',
    min_tokens: int = 2
):
    
    tokenizer = load_model(model_loc, load_model=False)
    special_tokens = distinct_special_chars(tokenizer=tokenizer)
    
    raw_tokens = ensure_tokens_are_lists(df)
    print("Original token count:", len(raw_tokens))

    filtered_tokens = filter_min_length(raw_tokens, min_tokens=2)
    print("After filter_min_length:", len(filtered_tokens))

    filtered_tokens = filter_only_special_tokens(filtered_tokens, special_tokens)
    print("After filter_only_special_tokens:", len(filtered_tokens))

    filtered_tokens = filter_only_numbers_and_special_tokens(filtered_tokens, special_tokens)
    print("After filter_only_number_and_special_tokens:", len(filtered_tokens))

    filtered_tokens = filter_only_punct_and_special_tokens(filtered_tokens, special_tokens)
    print("After filter_only_punct_and_special_tokens:", len(filtered_tokens))

    filtered_tokens = filter_at_least_n_minus_1_specials(filtered_tokens, special_tokens)
    print("After filter_at_least_n_minus_1_specials:", len(filtered_tokens))

    filtered_tokens = filter_zero_special_tokens(filtered_tokens, special_tokens)
    print("After filter_zero_special_tokens:", len(filtered_tokens))
    
    return filtered_tokens