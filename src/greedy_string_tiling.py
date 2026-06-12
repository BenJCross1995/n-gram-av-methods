from dataclasses import dataclass
from typing import Any, Sequence

from n_gram_tracing import tokenize_to_tokens


@dataclass(frozen=True)
class Tile:
    known_start: int
    unknown_start: int
    length: int


def _unmarked_run(marked: list[bool], start: int, length: int) -> bool:
    """
    Return True if the whole span is currently unmarked.
    """
    return start + length <= len(marked) and not any(marked[start:start + length])


def greedy_string_tiling(
    known_tokens: Sequence[Any],
    unknown_tokens: Sequence[Any],
    min_match: int = 3,
) -> tuple[list[Tile], int]:
    """
    Basic Greedy String Tiling over two token sequences.

    Args:
        known_tokens:
            Token sequence from the known text.
        unknown_tokens:
            Token sequence from the unknown/disputed text.
        min_match:
            Minimum tile length in tokens.

    Returns:
        raw_tiles:
            List of Tile objects containing known_start, unknown_start, and length.
        score:
            Total number of tiled tokens.
    """
    if min_match < 1:
        raise ValueError("min_match must be at least 1")

    n_known = len(known_tokens)
    n_unknown = len(unknown_tokens)

    marked_known = [False] * n_known
    marked_unknown = [False] * n_unknown

    raw_tiles: list[Tile] = []
    score = 0

    while True:
        max_match = min_match
        matches: list[Tile] = []

        # Scan all currently unmarked starting positions.
        for i in range(n_known):
            if marked_known[i]:
                continue

            for j in range(n_unknown):
                if marked_unknown[j]:
                    continue

                k = 0

                while (
                    i + k < n_known
                    and j + k < n_unknown
                    and not marked_known[i + k]
                    and not marked_unknown[j + k]
                    and known_tokens[i + k] == unknown_tokens[j + k]
                ):
                    k += 1

                if k == max_match:
                    matches.append(Tile(i, j, k))
                elif k > max_match:
                    max_match = k
                    matches = [Tile(i, j, k)]

        if not matches:
            break

        # Turn non-occluded maximal matches into permanent tiles.
        for match in matches:
            if (
                _unmarked_run(marked_known, match.known_start, match.length)
                and _unmarked_run(marked_unknown, match.unknown_start, match.length)
            ):
                for offset in range(match.length):
                    marked_known[match.known_start + offset] = True
                    marked_unknown[match.unknown_start + offset] = True

                raw_tiles.append(match)
                score += match.length

        # The final pass creates minimum-length tiles, then stops.
        if max_match == min_match:
            break

    raw_tiles.sort(key=lambda t: (t.known_start, t.unknown_start, t.length))
    return raw_tiles, score


def materialise_tiles(
    known_tokens: Sequence[Any],
    unknown_tokens: Sequence[Any],
    raw_tiles: Sequence[Tile],
) -> list[dict[str, Any]]:
    """
    Convert GST tile positions into readable token-level tile information.

    End indices are exclusive, so:
        known_tokens[known_start:known_end]
        unknown_tokens[unknown_start:unknown_end]
    return the matched tile span.
    """
    detailed_tiles = []

    for tile_id, tile in enumerate(raw_tiles, start=1):
        known_start = tile.known_start
        unknown_start = tile.unknown_start
        length = tile.length

        known_tile_tokens = list(known_tokens[known_start:known_start + length])
        unknown_tile_tokens = list(unknown_tokens[unknown_start:unknown_start + length])

        if known_tile_tokens != unknown_tile_tokens:
            raise ValueError(
                f"Tile {tile_id} does not match between sequences. "
                f"known_tile_tokens={known_tile_tokens}, "
                f"unknown_tile_tokens={unknown_tile_tokens}"
            )

        detailed_tiles.append({
            "tile_id": tile_id,
            "known_start": known_start,
            "known_end": known_start + length,
            "unknown_start": unknown_start,
            "unknown_end": unknown_start + length,
            "length": length,
            "tokens": known_tile_tokens,
            "tile_text": " ".join(map(str, known_tile_tokens)),
        })

    return detailed_tiles

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


def get_gst_tile_token_lists(
    result: dict[str, Any],
    *,
    sort: bool = True,
    dedupe: bool = True,
) -> list[list[Any]]:
    """
    Extract GST tiles as a list of token lists.

    By default:
        1. extract tile tokens
        2. sort by token length, then character length
        3. dedupe while preserving sorted order
    """
    tile_token_lists = [
        list(tile["tokens"])
        for tile in result["tiles"]
    ]

    if sort:
        tile_token_lists = sort_ngrams(tile_token_lists)

    if dedupe:
        tile_token_lists = dedupe_ngrams(tile_token_lists)

    return tile_token_lists

def gst_known_unknown_tiles(
    known_text: str,
    unknown_text: str,
    tokenizer=None,
    *,
    min_match: int = 3,
    lowercase: bool = True,
    include_token_lists: bool = True,
) -> dict[str, Any]:
    """
    Tokenize known and unknown texts using tokenize_to_tokens(), then run GST.

    If tokenizer is None, tokenize_to_tokens() should fall back to word tokens.
    If tokenizer is supplied, tokenize_to_tokens() should use that tokenizer.

    Returns:
        Dictionary containing readable GST tiles, raw Tile objects, score, and
        normalised similarity values.
    """
    known_tokens = tokenize_to_tokens(
        known_text,
        tokenizer,
        lowercase=lowercase,
    )

    unknown_tokens = tokenize_to_tokens(
        unknown_text,
        tokenizer,
        lowercase=lowercase,
    )

    raw_tiles, score = greedy_string_tiling(
        known_tokens=known_tokens,
        unknown_tokens=unknown_tokens,
        min_match=min_match,
    )

    tiles = materialise_tiles(
        known_tokens=known_tokens,
        unknown_tokens=unknown_tokens,
        raw_tiles=raw_tiles,
    )

    tile_token_lists = get_gst_tile_token_lists(
        {"tiles": tiles},
        sort=True,
        dedupe=True,
    )
    
    n_known_tokens = len(known_tokens)
    n_unknown_tokens = len(unknown_tokens)

    shorter_len = min(n_known_tokens, n_unknown_tokens)
    total_len = n_known_tokens + n_unknown_tokens

    result = {
        "tiles": tiles,
        "tile_token_lists": tile_token_lists,
        "raw_tiles": raw_tiles,
        "score": score,
        "coverage_of_shorter": score / shorter_len if shorter_len else 0.0,
        "dice_similarity": (2 * score / total_len) if total_len else 0.0,
        "n_known_tokens": n_known_tokens,
        "n_unknown_tokens": n_unknown_tokens,
        "min_match": min_match,
        "lowercase": lowercase,
    }

    if include_token_lists:
        result["known_tokens"] = known_tokens
        result["unknown_tokens"] = unknown_tokens

    return result


def print_gst_tiles(result: dict[str, Any]) -> None:
    """
    Convenience printer for GST tile results.
    """
    print(f"Score: {result['score']}")
    print(f"Coverage of shorter: {result['coverage_of_shorter']:.4f}")
    print(f"Dice similarity: {result['dice_similarity']:.4f}")
    print()

    for tile in result["tiles"]:
        print(f"Tile {tile['tile_id']}")
        print(f"Known span:   {tile['known_start']}:{tile['known_end']}")
        print(f"Unknown span: {tile['unknown_start']}:{tile['unknown_end']}")
        print(f"Length:       {tile['length']}")
        print(f"Tokens:       {tile['tokens']}")
        print(f"Text:         {tile['tile_text']}")
        print()

def gst_many_knowns_vs_unknown_tiles(
    known_texts: list[str],
    unknown_text: str,
    tokenizer=None,
    *,
    min_match: int = 3,
    lowercase: bool = True,
    known_ids: list[Any] | None = None,
    include_token_lists: bool = True,
) -> dict[str, Any]:
    """
    Run GST between multiple known texts and one unknown text.

    Known texts are concatenated with unique separator objects so tiles cannot
    cross known-document boundaries. GST is then run once, meaning unknown
    tokens can only be tiled once globally.

    Args:
        known_texts:
            List of known-author texts.
        unknown_text:
            Disputed/unknown text.
        tokenizer:
            Optional tokenizer passed to tokenize_to_tokens().
        min_match:
            Minimum tile length in tokens.
        lowercase:
            Whether to lowercase before tokenization.
        known_ids:
            Optional labels/IDs for known texts. If None, uses 0, 1, 2, ...
        include_token_lists:
            Whether to include token lists in the returned result.

    Returns:
        Dictionary containing:
            - tiles: tile metadata mapped back to known document indices
            - tile_token_lists: sorted/deduped list of GST tile tokens
            - raw_tiles: raw GST tiles over the concatenated known sequence
            - score and similarity values
    """
    if known_ids is None:
        known_ids = list(range(len(known_texts)))

    if len(known_ids) != len(known_texts):
        raise ValueError("known_ids must have the same length as known_texts")

    known_token_lists = [
        tokenize_to_tokens(
            text,
            tokenizer,
            lowercase=lowercase,
        )
        for text in known_texts
    ]

    unknown_tokens = tokenize_to_tokens(
        unknown_text,
        tokenizer,
        lowercase=lowercase,
    )

    known_concat = []
    known_pos_map = []

    for doc_idx, tokens in enumerate(known_token_lists):
        if doc_idx > 0:
            # Unique separator prevents matches crossing document boundaries.
            known_concat.append(object())
            known_pos_map.append(None)

        for tok_idx, token in enumerate(tokens):
            known_concat.append(token)
            known_pos_map.append({
                "known_doc_idx": doc_idx,
                "known_id": known_ids[doc_idx],
                "known_token_idx": tok_idx,
            })

    raw_tiles, score = greedy_string_tiling(
        known_tokens=known_concat,
        unknown_tokens=unknown_tokens,
        min_match=min_match,
    )

    tiles = []

    for tile_id, tile in enumerate(raw_tiles, start=1):
        concat_start = tile.known_start
        concat_end = tile.known_start + tile.length

        start_info = known_pos_map[concat_start]
        end_info = known_pos_map[concat_end - 1]

        # Should not happen because separators prevent crossing, but keep safe.
        if start_info is None or end_info is None:
            continue

        if start_info["known_doc_idx"] != end_info["known_doc_idx"]:
            continue

        known_doc_idx = start_info["known_doc_idx"]
        known_id = start_info["known_id"]
        known_start = start_info["known_token_idx"]
        known_end = known_start + tile.length

        unknown_start = tile.unknown_start
        unknown_end = unknown_start + tile.length

        known_tile_tokens = list(
            known_token_lists[known_doc_idx][known_start:known_end]
        )

        unknown_tile_tokens = list(
            unknown_tokens[unknown_start:unknown_end]
        )

        if known_tile_tokens != unknown_tile_tokens:
            raise ValueError(
                f"Tile {tile_id} does not match after mapping. "
                f"known_tile_tokens={known_tile_tokens}, "
                f"unknown_tile_tokens={unknown_tile_tokens}"
            )

        tiles.append({
            "tile_id": tile_id,
            "known_doc_idx": known_doc_idx,
            "known_id": known_id,
            "known_start": known_start,
            "known_end": known_end,
            "unknown_start": unknown_start,
            "unknown_end": unknown_end,
            "length": tile.length,
            "tokens": known_tile_tokens,
            "tile_text": " ".join(map(str, known_tile_tokens)),
        })

    tile_token_lists = dedupe_ngrams(
        sort_ngrams([
            list(tile["tokens"])
            for tile in tiles
        ])
    )

    n_known_tokens = sum(len(tokens) for tokens in known_token_lists)
    n_unknown_tokens = len(unknown_tokens)

    shorter_len = min(n_known_tokens, n_unknown_tokens)
    total_len = n_known_tokens + n_unknown_tokens

    result = {
        "tiles": tiles,
        "tile_token_lists": tile_token_lists,
        "raw_tiles": raw_tiles,
        "score": score,
        "coverage_of_shorter": score / shorter_len if shorter_len else 0.0,
        "dice_similarity": (2 * score / total_len) if total_len else 0.0,
        "n_known_tokens": n_known_tokens,
        "n_unknown_tokens": n_unknown_tokens,
        "n_known_texts": len(known_texts),
        "min_match": min_match,
        "lowercase": lowercase,
    }

    if include_token_lists:
        result["known_token_lists"] = known_token_lists
        result["known_concat_tokens"] = known_concat
        result["unknown_tokens"] = unknown_tokens

    return result

from html import escape
from typing import Any


_TILE_COLOURS = [
    "#fff3b0",
    "#c7f9cc",
    "#bde0fe",
    "#ffc8dd",
    "#d0bfff",
    "#ffd6a5",
    "#caffbf",
    "#a0c4ff",
    "#ffadad",
    "#e9edc9",
]


def _html_escape(x: Any) -> str:
    return escape(str(x), quote=True)


def _tile_colour(tile_id: int) -> str:
    return _TILE_COLOURS[(tile_id - 1) % len(_TILE_COLOURS)]


def _render_plain_token_html(
    token: Any,
    token_idx: int,
    *,
    show_token_indices: bool = True,
) -> str:
    if show_token_indices:
        return f"""
        <span class="gst-token" title="token {token_idx}">
            <span class="gst-token-index">{token_idx}</span>
            <span class="gst-token-text">{_html_escape(token)}</span>
        </span>
        """

    return f"""
    <span class="gst-token" title="token {token_idx}">
        <span class="gst-token-text">{_html_escape(token)}</span>
    </span>
    """


def _render_tiled_token_html(
    token: Any,
    token_idx: int,
    *,
    show_token_indices: bool = True,
) -> str:
    if show_token_indices:
        return f"""
        <span class="gst-token gst-token-in-tile" title="token {token_idx}">
            <span class="gst-token-index">{token_idx}</span>
            <span class="gst-token-text">{_html_escape(token)}</span>
        </span>
        """

    return f"""
    <span class="gst-token gst-token-in-tile" title="token {token_idx}">
        <span class="gst-token-text">{_html_escape(token)}</span>
    </span>
    """


def _render_token_stream_html_multi(
    tokens: list[Any],
    tiles: list[dict[str, Any]],
    *,
    side: str,
    known_doc_idx: int | None = None,
    max_tokens: int | None = None,
    show_token_indices: bool = True,
) -> str:
    """
    Render either:
        - one known document token stream, if side="known"
        - the unknown document token stream, if side="unknown"

    For known streams, pass known_doc_idx.
    For unknown stream, all tiles are shown.
    """
    if side not in {"known", "unknown"}:
        raise ValueError("side must be either 'known' or 'unknown'")

    if side == "known":
        if known_doc_idx is None:
            raise ValueError("known_doc_idx must be supplied when side='known'")

        relevant_tiles = [
            tile for tile in tiles
            if tile["known_doc_idx"] == known_doc_idx
        ]

        start_key = "known_start"
        end_key = "known_end"

    else:
        relevant_tiles = tiles
        start_key = "unknown_start"
        end_key = "unknown_end"

    spans_by_start = {
        tile[start_key]: tile
        for tile in relevant_tiles
    }

    n_tokens = len(tokens)
    render_limit = n_tokens if max_tokens is None else min(max_tokens, n_tokens)

    html_parts = []
    i = 0

    while i < render_limit:
        if i in spans_by_start:
            tile = spans_by_start[i]

            tile_id = tile["tile_id"]
            start = tile[start_key]
            end = min(tile[end_key], render_limit)
            colour = _tile_colour(tile_id)

            tile_tokens = tokens[start:end]

            tile_token_html = " ".join(
                _render_tiled_token_html(
                    tok,
                    token_idx=start + offset,
                    show_token_indices=show_token_indices,
                )
                for offset, tok in enumerate(tile_tokens)
            )

            if side == "known":
                title = (
                    f"Tile {tile_id}; known doc {tile['known_doc_idx']}; "
                    f"known span {tile['known_start']}:{tile['known_end']}; "
                    f"unknown span {tile['unknown_start']}:{tile['unknown_end']}; "
                    f"length {tile['length']}"
                )
            else:
                title = (
                    f"Tile {tile_id}; from known doc {tile['known_doc_idx']}; "
                    f"known span {tile['known_start']}:{tile['known_end']}; "
                    f"unknown span {tile['unknown_start']}:{tile['unknown_end']}; "
                    f"length {tile['length']}"
                )

            html_parts.append(
                f"""
                <span class="gst-tile"
                      style="background-color:{colour};"
                      title="{_html_escape(title)}">
                    <span class="gst-tile-label">T{tile_id}</span>
                    {tile_token_html}
                </span>
                """
            )

            i = end

        else:
            html_parts.append(
                _render_plain_token_html(
                    tokens[i],
                    token_idx=i,
                    show_token_indices=show_token_indices,
                )
            )
            i += 1

    if max_tokens is not None and n_tokens > max_tokens:
        html_parts.append(
            f"""
            <span class="gst-omitted">
                ... {n_tokens - max_tokens} tokens omitted
            </span>
            """
        )

    return "\n".join(html_parts)


def build_many_knowns_gst_tiles_html(
    result: dict[str, Any],
    *,
    title: str = "Greedy String Tiling: many known texts vs unknown text",
    max_tokens_per_text: int | None = None,
    show_token_indices: bool = True,
) -> str:
    """
    Build an HTML view for gst_many_knowns_vs_unknown_tiles() results.

    Requires:
        result["known_token_lists"]
        result["unknown_tokens"]
        result["tiles"]

    This version does not show the tile table. It shows:
        - summary pills
        - unknown text panel
        - one known text panel per known document
        - token numbers inside each token span
    """
    if "known_token_lists" not in result or "unknown_tokens" not in result:
        raise ValueError(
            "result must contain 'known_token_lists' and 'unknown_tokens'. "
            "Run gst_many_knowns_vs_unknown_tiles(..., include_token_lists=True)."
        )

    known_token_lists = result["known_token_lists"]
    unknown_tokens = result["unknown_tokens"]
    tiles = result["tiles"]

    unknown_html = _render_token_stream_html_multi(
        unknown_tokens,
        tiles,
        side="unknown",
        max_tokens=max_tokens_per_text,
        show_token_indices=show_token_indices,
    )

    known_panels = []

    for known_doc_idx, known_tokens in enumerate(known_token_lists):
        known_tiles = [
            tile for tile in tiles
            if tile["known_doc_idx"] == known_doc_idx
        ]

        known_id = (
            known_tiles[0].get("known_id", known_doc_idx)
            if known_tiles
            else known_doc_idx
        )

        known_html = _render_token_stream_html_multi(
            known_tokens,
            tiles,
            side="known",
            known_doc_idx=known_doc_idx,
            max_tokens=max_tokens_per_text,
            show_token_indices=show_token_indices,
        )

        known_panels.append(
            f"""
            <div class="gst-panel">
                <div class="gst-panel-title">
                    Known text {known_doc_idx} 
                    <span class="gst-panel-subtitle">
                        ID: {_html_escape(known_id)} · 
                        tokens: {len(known_tokens)} · 
                        tiles: {len(known_tiles)}
                    </span>
                </div>
                <div class="gst-token-stream">
                    {known_html}
                </div>
            </div>
            """
        )

    return f"""
    <style>
        .gst-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            line-height: 1.45;
            color: #222;
        }}

        .gst-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .gst-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 14px;
        }}

        .gst-pill {{
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 13px;
        }}

        .gst-layout {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
        }}

        .gst-panel {{
            border: 1px solid #d1d5db;
            border-radius: 8px;
            overflow: hidden;
            background: white;
        }}

        .gst-panel-title {{
            background: #f9fafb;
            border-bottom: 1px solid #d1d5db;
            padding: 8px 10px;
            font-weight: 700;
        }}

        .gst-panel-subtitle {{
            font-weight: 400;
            color: #6b7280;
            margin-left: 8px;
            font-size: 13px;
        }}

        .gst-token-stream {{
            padding: 10px;
            max-height: 420px;
            overflow: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            line-height: 2.1;
        }}

        .gst-token {{
            display: inline-flex;
            align-items: center;
            gap: 3px;
            margin: 1px 2px;
            padding: 1px 3px;
            border-radius: 4px;
            background: #f3f4f6;
            border: 1px solid transparent;
        }}

        .gst-token-in-tile {{
            background: transparent;
        }}

        .gst-token-index {{
            font-size: 9px;
            color: #6b7280;
            border-right: 1px solid rgba(0, 0, 0, 0.18);
            padding-right: 3px;
        }}

        .gst-token-text {{
            white-space: pre;
        }}

        .gst-tile {{
            display: inline-block;
            margin: 3px 4px;
            padding: 4px 6px;
            border: 1px solid #555;
            border-radius: 7px;
        }}

        .gst-tile-label {{
            display: inline-block;
            font-weight: 700;
            margin-right: 6px;
            color: #111;
        }}

        .gst-omitted {{
            color: #6b7280;
            font-style: italic;
            padding: 4px 6px;
        }}
    </style>

    <div class="gst-container">
        <div class="gst-title">{_html_escape(title)}</div>

        <div class="gst-meta">
            <span class="gst-pill">Score: {result["score"]}</span>
            <span class="gst-pill">Coverage of shorter: {result["coverage_of_shorter"]:.4f}</span>
            <span class="gst-pill">Dice similarity: {result["dice_similarity"]:.4f}</span>
            <span class="gst-pill">Known texts: {result["n_known_texts"]}</span>
            <span class="gst-pill">Known tokens: {result["n_known_tokens"]}</span>
            <span class="gst-pill">Unknown tokens: {result["n_unknown_tokens"]}</span>
            <span class="gst-pill">Min match: {result["min_match"]}</span>
            <span class="gst-pill">Tiles: {len(tiles)}</span>
        </div>

        <div class="gst-layout">
            <div class="gst-panel">
                <div class="gst-panel-title">
                    Unknown text
                    <span class="gst-panel-subtitle">
                        tokens: {len(unknown_tokens)} · tiles: {len(tiles)}
                    </span>
                </div>
                <div class="gst-token-stream">
                    {unknown_html}
                </div>
            </div>

            {''.join(known_panels)}
        </div>
    </div>
    """


def show_many_knowns_gst_tiles_html(
    result: dict[str, Any],
    *,
    title: str = "Greedy String Tiling: many known texts vs unknown text",
    max_tokens_per_text: int | None = None,
    show_token_indices: bool = True,
):
    """
    Display many-known GST results as HTML in a notebook.
    """
    from IPython.display import HTML, display

    html = build_many_knowns_gst_tiles_html(
        result,
        title=title,
        max_tokens_per_text=max_tokens_per_text,
        show_token_indices=show_token_indices,
    )

    display(HTML(html))