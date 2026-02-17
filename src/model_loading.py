# -*- coding: utf-8 -*-
"""
Model loading utilities.

Loads a local Hugging Face causal LM and its tokenizer with sensible defaults
for inference/scoring and optional multi-GPU device mapping.
"""
import string

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Iterable

def load_model(model_loc: str):
    """Load a local AutoModelForCausalLM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    model = AutoModelForCausalLM.from_pretrained(model_loc)
    model.eval()
    return tokenizer, model

# -------------------------------------------------------------- #
# -- FUNCTIONS TO GET SPECIAL WHITESPACE CHARACTERS FOR MODEL -- #
# -------------------------------------------------------------- #

def build_space_probe_strings(include_joiners: bool = True):
    """
    Return an ordered, de-duplicated list of characters/strings that function like
    spaces or separators between words (plus a few invisible format controls that
    can occur between words). No emojis.

    Groups included:
      - ASCII whitespace: ' ', '\\t', '\\n', '\\r', '\\v', '\\f', and '\\r\\n'
      - Unicode Space_Separator (Zs): NBSP, OGHAM SPACE, U+2000..U+200A, NNBSP, MMSP, IDEOGRAPHIC SPACE
      - Line/Paragraph separators: U+2028, U+2029
      - Zero-width & space-like format controls: ZWSP, WORD JOINER, BOM/ZWNBS, MVS
      - (Optional) Joiners: ZWNJ, ZWJ
    """
    probes = []

    # --- ASCII whitespace ---
    probes += [" ", "\t", "\n", "\r", "\v", "\f", "\r\n"]

    # --- Unicode Space_Separator (Zs) ---
    probes += [
        "\u00A0",              # NO-BREAK SPACE
        "\u1680",              # OGHAM SPACE MARK
        *[chr(c) for c in range(0x2000, 0x200B)],  # EN QUAD..HAIR SPACE (U+2000..U+200A)
        "\u202F",              # NARROW NO-BREAK SPACE
        "\u205F",              # MEDIUM MATHEMATICAL SPACE
        "\u3000",              # IDEOGRAPHIC SPACE
    ]

    # --- Line/Paragraph separators ---
    probes += [
        "\u2028",              # LINE SEPARATOR
        "\u2029",              # PARAGRAPH SEPARATOR
    ]

    # --- Zero-width & space-like format controls ---
    probes += [
        "\u200B",              # ZERO WIDTH SPACE
        "\u2060",              # WORD JOINER (non-breaking, zero width)
        "\uFEFF",              # ZERO WIDTH NO-BREAK SPACE (BOM)
        "\u180E",              # MONGOLIAN VOWEL SEPARATOR (historically spacing; now Cf)
    ]

    # --- Optional: invisible joiners that appear between words in some scripts ---
    if include_joiners:
        probes += [
            "\u200C",          # ZERO WIDTH NON-JOINER
            "\u200D",          # ZERO WIDTH JOINER
        ]

    # De-duplicate while preserving order
    seen = set()
    out = []
    for s in probes:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def visible_markers_via_tokenizer(tokenizer, texts: Iterable[str]) -> Dict[str, str]:
    """
    For each string in `texts` (e.g., " ", "\\n", "\\t"), return what the tokenizer
    shows as *visible token strings* (e.g., Ġ, Ċ, ĉ) by round-tripping through the
    tokenizer. If nothing special is shown, the entry is omitted.

    Returns: dict mapping input string -> visible token string
    """
    out = {}
    for t in texts:
        # Encode without adding model special tokens
        ids = tokenizer.encode(t, add_special_tokens=False)
        # Convert to the tokens-as-strings (what you see in vocab.json)
        toks = tokenizer.convert_ids_to_tokens(ids)
        visible = "".join(toks)

        # If the tokenizer reconstructs exactly the same text when decoding,
        # and the visible token string is the same as the raw text, then there's
        # no “special” visible marker worth returning.
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if visible != t or decoded != t:
            # Keep only entries that reveal something special/visible
            out[t] = visible
    return out

def distinct_special_chars(
    marker_map: Optional[Dict[str, str]] = None,
    *,
    tokenizer=None,                          # required if marker_map is None
    drop_if_all_punct: bool = True,
    extra_punctuation: Iterable[str] = (),
    include_joiners: bool = True             # forwarded to build_space_probe_strings
) -> List[str]:
    """
    If marker_map is provided: return a distinct, ordered list of *marker strings*
    (do not split into characters) found in the values, excluding values identical
    to the source and (optionally) values made only of ASCII punctuation.

    If marker_map is None: we will build one automatically by:
      1) building probe strings with build_space_probe_strings(include_joiners),
      2) computing visible markers via visible_markers_via_tokenizer(tokenizer, probes).

    Parameters
    ----------
    marker_map : Optional[Dict[str, str]]
        e.g., {" ": "Ġ", "\\n": "Ċ", "\\u00A0": "ÂƠ"}.
    tokenizer : Any
        A Hugging Face tokenizer instance (only needed if marker_map is None).
    drop_if_all_punct : bool
        If True, skip values composed entirely of ASCII punctuation
        (per string.punctuation plus extra_punctuation).
    extra_punctuation : Iterable[str]
        Additional characters you want to treat as “punctuation-like”.
    include_joiners : bool
        Passed to build_space_probe_strings(); include ZWNJ/ZWJ if True.

    Returns
    -------
    List[str]
        Distinct, ordered list of marker strings (e.g., ['Ġ', 'Ċ'] or ['▁', '▁<unk>']).
    """
    # If no map provided, build it using the two helper functions you already have:
    if marker_map is None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when marker_map is None")
        probes = build_space_probe_strings(include_joiners=include_joiners)
        marker_map = visible_markers_via_tokenizer(tokenizer, probes)

    punct = set(string.punctuation) | set(extra_punctuation)
    seen = set()
    out: List[str] = []
    for src, vis in marker_map.items():
        if not vis or vis == src:
            continue
        if drop_if_all_punct and all(ch in punct for ch in vis):
            continue
        if vis not in seen:
            seen.add(vis)
            out.append(vis)
    return out
