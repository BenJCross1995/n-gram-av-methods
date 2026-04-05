import pandas as pd

from collections import defaultdict
from typing import Hashable, Iterable, Sequence, Tuple, Dict, List, Literal
from pathlib import Path

Token = Hashable
Ngram = Tuple[Token, ...]

def get_tokens(
    text: str,
    tokenizer,
    *,
    lowercase: bool = True,
    add_special_tokens: bool = False,
):
    """
    Hugging Face Transformers tokenizer -> list of token strings.
    """
    if lowercase:
        text = text.lower()

    input_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return tokenizer.convert_ids_to_tokens(input_ids)

def _distinct_ngrams(tokens: Sequence[Token], n: int) -> set[Ngram]:
    """Distinct contiguous n-gram *types* (presence/absence)."""
    L = len(tokens)
    if n <= 0 or n > L:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(L - n + 1)}

def weighted_ngram_tracing_df(
    *,
    known_tokens_list: list[Sequence[Token]],
    unknown_tokens: Sequence[Token],
    common_n_grams: Iterable[Sequence[Token]],
    weight: Literal["none", "linear", "power", "exp"] = "linear",
    alpha: float = 1.0,
    base: float = 2.0,
    decimals: int = 3,
    validate_common: bool = True,
) -> pd.DataFrame:
    """
    Per-n table using MULTIPLE known documents.

    known_ngrams_distinct is the union of distinct n-grams across all known docs.

    weight options:
      - "none"   -> no weighting, w = 1
      - "linear" -> w = n
      - "power"  -> w = n ** alpha
      - "exp"    -> w = base ** n
    """
    common_by_n: Dict[int, set[Ngram]] = defaultdict(set)
    for ng in common_n_grams:
        t = tuple(ng)
        if len(t) > 0:
            common_by_n[len(t)].add(t)

    rows: List[dict] = []

    for n in sorted(common_by_n):
        K = set()
        for known_tokens in known_tokens_list:
            K |= _distinct_ngrams(known_tokens, n)

        Q = _distinct_ngrams(unknown_tokens, n)

        overlap_set = set(common_by_n[n])

        if validate_common:
            overlap_set &= (K & Q)

        a = len(overlap_set)
        known_cnt = len(K)
        unknown_cnt = len(Q)
        union_cnt = known_cnt + unknown_cnt - a

        simpson = 0.0 if unknown_cnt == 0 else a / unknown_cnt
        jaccard = 0.0 if union_cnt == 0 else a / union_cnt

        weight_l = weight.lower().strip()
        if weight_l == "none":
            w = 1.0
        elif weight_l == "linear":
            w = float(n)
        elif weight_l == "power":
            w = float(n) ** float(alpha)
        elif weight_l == "exp":
            w = float(base) ** float(n)
        else:
            raise ValueError("weight must be 'none', 'linear', 'power', or 'exp'.")

        rows.append(
            {
                "token_level": n,
                "known_ngrams_distinct": len(K),
                "unknown_ngrams_distinct": unknown_cnt,
                "overlap_ngrams_distinct": a,
                "union_ngrams_distinct": union_cnt,
                "simpson": round(simpson, decimals),
                "jaccard": round(jaccard, decimals),
                "weight_w": w,
                "num_w": w * a,
                "den_simpson_w": w * unknown_cnt,
                "den_jaccard_w": w * union_cnt,
            }
        )

    return pd.DataFrame(rows)

def aggregate_ge(df: pd.DataFrame, t: int, coef: str = "simpson") -> float:
    sub = df[df["token_level"] >= t]
    if sub.empty:
        return 0.0
    num = sub["num_w"].sum()
    if coef == "simpson":
        den = sub["den_simpson_w"].sum()
    elif coef == "jaccard":
        den = sub["den_jaccard_w"].sum()
    else:
        raise ValueError("coef must be 'simpson' or 'jaccard'.")
    return 0.0 if den == 0 else float(num / den)

def aggregate_weighted_df(
    df: pd.DataFrame,
    problem_metadata: pd.DataFrame,
    *,
    weight: Literal["none", "linear", "power", "exp"],
    alpha: float = 1.0,
    base: float = 2.0,
    level_col: str = "token_level",
) -> pd.DataFrame:
    """
    One row per min_token_size (t). Computes BOTH aggregated simpson/jaccard scores
    over token_level >= t. Prepends repeated single-row problem_metadata.

    Weighting metadata columns before min_token_size:
      - always: weight
      - if weight == "power": alpha
      - if weight == "exp": base
    """
    if problem_metadata.shape[0] != 1:
        raise ValueError("problem_metadata must be a single-row dataframe.")

    weight_l = weight.lower().strip()
    if weight_l not in {"none", "linear", "power", "exp"}:
        raise ValueError("weight must be 'none', 'linear', 'power', or 'exp'.")

    levels = sorted(df[level_col].dropna().unique().tolist())

    results = pd.DataFrame({"min_token_size": levels})
    results["simpson_score"] = [aggregate_ge(df, int(t), coef="simpson") for t in levels]
    results["jaccard_score"] = [aggregate_ge(df, int(t), coef="jaccard") for t in levels]

    weight_cols = {"weight": weight_l}
    if weight_l == "power":
        weight_cols["alpha"] = float(alpha)
    elif weight_l == "exp":
        weight_cols["base"] = float(base)

    for col, val in reversed(list(weight_cols.items())):
        results.insert(0, col, val)

    meta_rep = pd.concat([problem_metadata] * len(results), ignore_index=True)
    return pd.concat([meta_rep.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

def create_excel_template(
    weighted_df: pd.DataFrame,
    agg_weighted_df: pd.DataFrame,
    docs: pd.DataFrame = None,
    ngrams: pd.DataFrame = None,
    path: str | Path = "template.xlsx",
) -> None:
    """
    Writes all sheets, builds a distinct phrases 'LLR' table, adds include_phrase lookups
    to Known & Unknown, and then adds your LLR formulas (D..H).
    """
    path = Path(path)

    writer_mode = "a" if path.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": writer_mode}
    if writer_mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"
        
    with pd.ExcelWriter(path, **writer_kwargs) as writer:
        if docs is not None:
            docs.to_excel(writer, index=False, sheet_name="docs")
        if ngrams is not None:
            ngrams.to_excel(writer, index=False, sheet_name="ngrams")
        weighted_df.to_excel(writer, index=False, sheet_name="weighted raw")
        agg_weighted_df.to_excel(writer, index=False, sheet_name="metadata")
        
def create_ngram_df(ngram_list):
    ngram_df = pd.DataFrame({
        "n_gram": ngram_list
    })

    ngram_df.insert(0, "id", range(1, len(ngram_df) + 1))
    ngram_df.insert(2, "num_tokens", ngram_df["n_gram"].apply(len))
    ngram_df["num_chars"] = ngram_df["n_gram"].apply(lambda x: sum(len(str(item)) for item in x))
    
    return ngram_df