from dataclasses import dataclass
from typing import Any, Sequence

from n_gram_tracing import tokenize_to_tokens


@dataclass(frozen=True)
class Tile:
    known_start: int
    unknown_start: int
    length: int


@dataclass(frozen=True)
class MultiTile:
    """A GST tile linking one known document to the unknown document."""

    known_doc_idx: int
    known_start: int
    unknown_start: int
    length: int


def _unmarked_run(marked: list[bool], start: int, length: int) -> bool:
    """
    Return True if the whole span [start, start + length) is currently unmarked.

    Uses an early-exit loop rather than any() on a slice, so it allocates no
    intermediate list and bails out at the first marked token.
    """
    end = start + length

    if end > len(marked):
        return False

    for idx in range(start, end):
        if marked[idx]:
            return False

    return True


def _next_unmarked(marked: list[bool], start: int) -> int:
    """
    Return the next index >= start that is unmarked, or len(marked) if none.

    Used to skip over already-tiled runs in the outer scan loops.
    """
    n = len(marked)
    i = start

    while i < n and marked[i]:
        i += 1

    return i


def greedy_string_tiling(
    known_tokens: Sequence[Any],
    unknown_tokens: Sequence[Any],
    min_match: int = 3,
) -> tuple[list[Tile], int]:
    """
    Greedy String Tiling over two token sequences (Wise, 1993).

    Produces a non-overlapping greedy population of maximal contiguous shared
    spans. Each token in either sequence is claimed by at most one tile.

    Efficiency notes:
        - First pass per round: build a hash map of length-`min_match` token
          tuples from the unknown sequence, then scan the known sequence and
          probe that map. This avoids the O(n*m*L) brute-force scan.
        - Outer loops skip over marked runs.
        - Match candidates are sorted deterministically before marking, so the
          extracted tile set is stable across runs.

    Args:
        known_tokens:
            Token sequence from the known text.
        unknown_tokens:
            Token sequence from the unknown/disputed text.
        min_match:
            Minimum tile length in tokens.

    Returns:
        raw_tiles:
            List of Tile objects containing known_start, unknown_start, and
            length, sorted by (known_start, unknown_start, length).
        score:
            Total number of tiled tokens (sum of tile lengths).
    """
    if min_match < 1:
        raise ValueError("min_match must be at least 1")

    n_known = len(known_tokens)
    n_unknown = len(unknown_tokens)

    marked_known = [False] * n_known
    marked_unknown = [False] * n_unknown

    raw_tiles: list[Tile] = []
    score = 0

    # If either sequence is shorter than min_match, no tiles are possible.
    if n_known < min_match or n_unknown < min_match:
        return raw_tiles, 0

    while True:
        max_match = min_match
        matches: list[Tile] = []

        # ------------------------------------------------------------------
        # Build a hash index of length-min_match token tuples from the
        # currently-unmarked positions of the unknown sequence.
        #
        # Key   : tuple of min_match tokens starting at j
        # Value : list of starting indices j where that tuple occurs
        #
        # We only insert positions whose entire min_match window is unmarked,
        # because any tile must start in an unmarked run.
        # ------------------------------------------------------------------
        unknown_index: dict[tuple, list[int]] = {}

        for j in range(n_unknown - min_match + 1):
            # Reject windows that overlap any marked unknown token.
            if not _unmarked_run(marked_unknown, j, min_match):
                continue

            key = tuple(unknown_tokens[j:j + min_match])
            bucket = unknown_index.get(key)

            if bucket is None:
                unknown_index[key] = [j]
            else:
                bucket.append(j)

        # No length-min_match windows survive; nothing more to do.
        if not unknown_index:
            break

        # ------------------------------------------------------------------
        # Scan the known sequence. For each unmarked length-min_match window
        # in known_tokens, look up matching unknown positions in the hash
        # index, then extend each candidate forward as far as possible.
        # ------------------------------------------------------------------
        i = _next_unmarked(marked_known, 0)

        while i <= n_known - min_match:
            # Ensure the entire min_match window starting at i is unmarked.
            # If not, jump past the offending mark.
            if not _unmarked_run(marked_known, i, min_match):
                i = _next_unmarked(marked_known, i + 1)
                continue

            key = tuple(known_tokens[i:i + min_match])
            candidates = unknown_index.get(key)

            if candidates is None:
                i += 1
                continue

            for j in candidates:
                # Extend the match as far as possible from (i, j), stopping at
                # sequence ends, marked tokens, or token mismatch. We start
                # at offset min_match because the first min_match tokens are
                # already known to match (they came from the hash key).
                k = min_match

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

            i += 1

        if not matches:
            break

        # Sort match candidates deterministically before marking. Without this
        # sort, two equally long matches could be marked in iteration order,
        # which is well-defined here but easier to reason about when explicit.
        matches.sort(key=lambda t: (t.known_start, t.unknown_start))

        # ------------------------------------------------------------------
        # Convert non-occluded maximal matches into permanent tiles. A
        # candidate is occluded if any of its tokens were marked by an
        # earlier candidate in this same pass.
        # ------------------------------------------------------------------
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

        # The pass at max_match == min_match is the final pass; after marking
        # all min_match-length tiles, no further passes can find anything
        # longer, so we stop.
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

    Raises:
        ValueError: if a tile's known and unknown spans do not match. This is
            a defensive check; if it ever fires, GST has a bug.
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

    Each n-gram is converted to a tuple for hashing, then back to a list for
    output. The first occurrence of each n-gram in the input wins.
    """
    return [list(x) for x in dict.fromkeys(tuple(g) for g in ngrams)]


def sort_ngrams(ngrams):
    """
    Sort first by number of tokens, then by total character length.

    Ascending order on both keys: shortest n-grams come first, and within
    a given token count the ones with fewer characters come first.
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

def _assert_valid_many_tiles(
    raw_tiles: Sequence[MultiTile],
    known_token_lists: Sequence[Sequence[Any]],
    n_unknown_tokens: int,
) -> None:
    """
    Validate the multi-known GST associations.

    Several known documents may be associated with the *same complete unknown
    span*. This represents one shared tile occurring in several known texts.
    Partial overlaps between different unknown spans remain forbidden.

    Within each known document, token positions may still belong to at most one
    tile association.
    """
    unknown_owner: list[tuple[int, int] | None] = [None] * n_unknown_tokens
    known_owner: list[list[int | None]] = [
        [None] * len(tokens)
        for tokens in known_token_lists
    ]

    for association_idx, tile in enumerate(raw_tiles):
        unknown_span = (
            tile.unknown_start,
            tile.unknown_start + tile.length,
        )

        for offset in range(tile.length):
            unknown_pos = tile.unknown_start + offset
            known_pos = tile.known_start + offset

            previous_unknown_span = unknown_owner[unknown_pos]

            # Exact reuse of the complete unknown span is permitted because it
            # means that the same tile occurs in another known document.
            if (
                previous_unknown_span is not None
                and previous_unknown_span != unknown_span
            ):
                raise RuntimeError(
                    "GST produced partially overlapping unknown tiles: "
                    f"unknown span {previous_unknown_span} overlaps "
                    f"{unknown_span} at token position {unknown_pos}."
                )

            previous_known_tile = known_owner[tile.known_doc_idx][known_pos]
            if previous_known_tile is not None:
                raise RuntimeError(
                    "GST produced overlapping known spans: "
                    f"associations {previous_known_tile + 1} and "
                    f"{association_idx + 1} both use known document "
                    f"{tile.known_doc_idx}, token position {known_pos}."
                )

            unknown_owner[unknown_pos] = unknown_span
            known_owner[tile.known_doc_idx][known_pos] = association_idx


def greedy_string_tiling_many_knowns(
    known_token_lists: Sequence[Sequence[Any]],
    unknown_tokens: Sequence[Any],
    min_match: int = 3,
) -> tuple[list[MultiTile], int]:
    """
    Run a multi-known extension of GST against one unknown document.

    The unknown document has one global marked-token array. However, when the
    same maximal unknown span occurs in more than one known document, all of
    those known-document occurrences are attached to one shared tile before
    the unknown span is marked.

    This prevents the following undesirable behaviour:

        known 1: [we have been building ...]
        known 2:  we [have] been building ...
        unknown: [we have been building ...]

    Instead, the complete opening span is represented as one unknown tile that
    is associated with both known documents. Unknown coverage is counted once.

    For a repeated matching span within the same known document, the earliest
    available occurrence is used, preserving GST's one-to-one behaviour within
    each individual known/unknown comparison.
    """
    if min_match < 1:
        raise ValueError("min_match must be at least 1")

    marked_known = [
        [False] * len(tokens)
        for tokens in known_token_lists
    ]
    marked_unknown = [False] * len(unknown_tokens)

    raw_tiles: list[MultiTile] = []
    score = 0

    if not known_token_lists or len(unknown_tokens) < min_match:
        return raw_tiles, score

    while True:
        max_match = min_match
        matches: list[MultiTile] = []

        # Index all still-available minimum-length windows in the unknown text.
        unknown_index: dict[tuple, list[int]] = {}

        for unknown_start in range(len(unknown_tokens) - min_match + 1):
            if not _unmarked_run(
                marked_unknown,
                unknown_start,
                min_match,
            ):
                continue

            key = tuple(
                unknown_tokens[
                    unknown_start:unknown_start + min_match
                ]
            )
            unknown_index.setdefault(key, []).append(unknown_start)

        if not unknown_index:
            break

        # Gather the globally longest candidates across every known document.
        for known_doc_idx, known_tokens in enumerate(known_token_lists):
            n_known = len(known_tokens)

            if n_known < min_match:
                continue

            document_marks = marked_known[known_doc_idx]
            known_start = _next_unmarked(document_marks, 0)

            while known_start <= n_known - min_match:
                if not _unmarked_run(
                    document_marks,
                    known_start,
                    min_match,
                ):
                    known_start = _next_unmarked(
                        document_marks,
                        known_start + 1,
                    )
                    continue

                key = tuple(
                    known_tokens[
                        known_start:known_start + min_match
                    ]
                )
                candidate_unknown_starts = unknown_index.get(key)

                if candidate_unknown_starts is None:
                    known_start += 1
                    continue

                for unknown_start in candidate_unknown_starts:
                    length = min_match

                    while (
                        known_start + length < n_known
                        and unknown_start + length < len(unknown_tokens)
                        and not document_marks[known_start + length]
                        and not marked_unknown[unknown_start + length]
                        and known_tokens[known_start + length]
                        == unknown_tokens[unknown_start + length]
                    ):
                        length += 1

                    candidate = MultiTile(
                        known_doc_idx=known_doc_idx,
                        known_start=known_start,
                        unknown_start=unknown_start,
                        length=length,
                    )

                    if length == max_match:
                        matches.append(candidate)
                    elif length > max_match:
                        max_match = length
                        matches = [candidate]

                known_start += 1

        if not matches:
            break

        # Group candidates representing exactly the same unknown tile. Every
        # candidate in a group has the same length because only globally
        # maximal candidates are retained during this round.
        groups: dict[tuple[int, int], list[MultiTile]] = {}

        for match in matches:
            key = (match.unknown_start, match.length)
            groups.setdefault(key, []).append(match)

        # Preserve the previous deterministic GST tie-break between different
        # competing unknown spans. The earliest known-document candidate in a
        # group determines that group's position in the marking order.
        grouped_matches = sorted(
            groups.values(),
            key=lambda group: min(
                (
                    tile.known_doc_idx,
                    tile.known_start,
                    tile.unknown_start,
                )
                for tile in group
            ),
        )

        marked_any = False

        for group in grouped_matches:
            unknown_start = group[0].unknown_start
            length = group[0].length

            # Another equal-length group processed earlier may overlap this
            # unknown span. GST keeps only the first non-occluded group.
            if not _unmarked_run(marked_unknown, unknown_start, length):
                continue

            # Attach at most one available occurrence from each known document
            # to this shared unknown tile.
            accepted: list[MultiTile] = []
            accepted_docs: set[int] = set()

            for match in sorted(
                group,
                key=lambda tile: (
                    tile.known_doc_idx,
                    tile.known_start,
                ),
            ):
                if match.known_doc_idx in accepted_docs:
                    continue

                document_marks = marked_known[match.known_doc_idx]

                if not _unmarked_run(
                    document_marks,
                    match.known_start,
                    match.length,
                ):
                    continue

                accepted.append(match)
                accepted_docs.add(match.known_doc_idx)

            if not accepted:
                continue

            # Mark the unknown tile once.
            for offset in range(length):
                marked_unknown[unknown_start + offset] = True

            # Mark every corresponding known-document occurrence and retain an
            # association for each one. They will share one public tile_id.
            for match in accepted:
                document_marks = marked_known[match.known_doc_idx]

                for offset in range(match.length):
                    document_marks[match.known_start + offset] = True

                raw_tiles.append(match)

            score += length
            marked_any = True

        if not marked_any:
            break

        if max_match == min_match:
            break

    raw_tiles.sort(
        key=lambda tile: (
            tile.unknown_start,
            tile.length,
            tile.known_doc_idx,
            tile.known_start,
        )
    )

    _assert_valid_many_tiles(
        raw_tiles=raw_tiles,
        known_token_lists=known_token_lists,
        n_unknown_tokens=len(unknown_tokens),
    )

    return raw_tiles, score

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
    Run one global GST process over multiple known texts and one unknown text.

    Unlike independent pairwise GST runs, this function uses a single shared
    unknown-token marking state. Therefore, every unknown token can occur in
    at most one returned tile across the complete set of known documents.
    """
    n_texts = len(known_texts)

    if known_ids is None:
        known_ids = list(range(n_texts))

    if len(known_ids) != n_texts:
        raise ValueError("known_ids must have the same length as known_texts")

    known_token_lists = [
        tokenize_to_tokens(text, tokenizer, lowercase=lowercase)
        for text in known_texts
    ]
    unknown_tokens = tokenize_to_tokens(
        unknown_text,
        tokenizer,
        lowercase=lowercase,
    )

    raw_tiles, score = greedy_string_tiling_many_knowns(
        known_token_lists=known_token_lists,
        unknown_tokens=unknown_tokens,
        min_match=min_match,
    )

    tiles: list[dict[str, Any]] = []

    # Associations sharing the same complete unknown span receive the same
    # tile ID and therefore the same colour in the HTML visualisation.
    tile_id_by_unknown_span: dict[tuple[int, int], int] = {}

    for tile in raw_tiles:
        known_end = tile.known_start + tile.length
        unknown_end = tile.unknown_start + tile.length
        unknown_span = (tile.unknown_start, unknown_end)

        tile_id = tile_id_by_unknown_span.setdefault(
            unknown_span,
            len(tile_id_by_unknown_span) + 1,
        )

        known_tile_tokens = list(
            known_token_lists[tile.known_doc_idx][
                tile.known_start:known_end
            ]
        )
        unknown_tile_tokens = list(
            unknown_tokens[tile.unknown_start:unknown_end]
        )

        if known_tile_tokens != unknown_tile_tokens:
            raise ValueError(
                f"Tile {tile_id} does not match after materialisation. "
                f"known_tile_tokens={known_tile_tokens}, "
                f"unknown_tile_tokens={unknown_tile_tokens}"
            )

        tiles.append({
            "tile_id": tile_id,
            "known_doc_idx": tile.known_doc_idx,
            "known_id": known_ids[tile.known_doc_idx],
            "known_start": tile.known_start,
            "known_end": known_end,
            "unknown_start": tile.unknown_start,
            "unknown_end": unknown_end,
            "length": tile.length,
            "tokens": known_tile_tokens,
            "tile_text": " ".join(map(str, known_tile_tokens)),
        })

    # Validate public tile dictionaries. Exact duplicate unknown spans are
    # allowed because they are associations from multiple known documents to
    # one shared tile. Partial overlap between distinct spans is forbidden.
    unknown_span_by_position: dict[int, tuple[int, int]] = {}

    for tile in tiles:
        tile_span = (tile["unknown_start"], tile["unknown_end"])

        for position in range(*tile_span):
            previous_span = unknown_span_by_position.get(position)

            if previous_span is not None and previous_span != tile_span:
                raise RuntimeError(
                    "Returned GST tiles partially overlap in the unknown "
                    f"document: {previous_span} and {tile_span}."
                )

            unknown_span_by_position[position] = tile_span

    if len(unknown_span_by_position) != score:
        raise RuntimeError(
            "GST score does not equal the number of uniquely covered "
            "unknown-token positions."
        )

    tile_token_lists = sort_ngrams(
        dedupe_ngrams([
            list(tile["tokens"])
            for tile in tiles
        ])
    )

    n_known_tokens = sum(len(tokens) for tokens in known_token_lists)
    n_unknown_tokens = len(unknown_tokens)
    shorter_len = min(n_known_tokens, n_unknown_tokens)
    total_len = n_known_tokens + n_unknown_tokens

    result: dict[str, Any] = {
        "tiles": tiles,
        "tile_token_lists": tile_token_lists,
        "raw_tiles": raw_tiles,
        "score": score,
        "coverage_of_shorter": (
            score / shorter_len
            if shorter_len
            else 0.0
        ),
        "dice_similarity": (
            2 * score / total_len
            if total_len
            else 0.0
        ),
        "n_known_tokens": n_known_tokens,
        "n_unknown_tokens": n_unknown_tokens,
        "n_known_texts": n_texts,
        "known_ids": list(known_ids),
        "n_unique_tiles": len(tile_id_by_unknown_span),
        "n_tile_associations": len(tiles),
        "algorithm_variant": "shared multi-known GST tiles",
        "min_match": min_match,
        "lowercase": lowercase,
    }

    if include_token_lists:
        result["known_token_lists"] = known_token_lists
        result["unknown_tokens"] = unknown_tokens

    return result


# Backwards-compatible alias for notebooks that already use the v2 name.
gst_many_knowns_vs_unknown_tiles_v2 = gst_many_knowns_vs_unknown_tiles

# ============================================================================
# Paper-style HTML visualisation for multi-known shared GST tiles
# ============================================================================

import math
import shutil
import uuid
from html import escape
from pathlib import Path

try:
    from n_gram_tracing import tokens_to_text
except ImportError:
    tokens_to_text = None


_GST_TILE_COLOURS = [
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
    "#bae6fd",
    "#fecdd3",
]


def _html_escape(x: Any) -> str:
    return escape(str(x), quote=True)


def _tile_colour(tile_id: int) -> str:
    return _GST_TILE_COLOURS[(tile_id - 1) % len(_GST_TILE_COLOURS)]


def _display_tokens(tokens: Sequence[Any], tokenizer=None) -> str:
    """Convert token objects into readable text for labels and token boxes."""
    token_list = list(tokens)

    if tokenizer is not None and tokens_to_text is not None:
        try:
            return tokens_to_text(token_list, tokenizer)
        except Exception:
            pass

    if len(token_list) == 1:
        token = str(token_list[0])
        return token.replace("Ġ", " ").replace("▁", " ")

    return "".join(
        str(token).replace("Ġ", " ").replace("▁", " ")
        for token in token_list
    )


def _group_unknown_tile_associations(
    tiles: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse known-document associations sharing one unknown tile."""
    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}

    for tile in tiles:
        key = (
            tile["tile_id"],
            tile["unknown_start"],
            tile["unknown_end"],
        )

        if key not in grouped:
            grouped[key] = {
                "tile_id": tile["tile_id"],
                "unknown_start": tile["unknown_start"],
                "unknown_end": tile["unknown_end"],
                "length": tile["length"],
                "tokens": list(tile["tokens"]),
                "associations": [],
            }

        grouped[key]["associations"].append(tile)

    return sorted(
        grouped.values(),
        key=lambda tile: (
            tile["unknown_start"],
            tile["unknown_end"],
            tile["tile_id"],
        ),
    )


def _visual_tiles_for_side(
    tiles: Sequence[dict[str, Any]],
    *,
    side: str,
    known_doc_idx: int | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    if side == "unknown":
        return (
            _group_unknown_tile_associations(tiles),
            "unknown_start",
            "unknown_end",
        )

    if side != "known":
        raise ValueError("side must be 'known' or 'unknown'")

    if known_doc_idx is None:
        raise ValueError("known_doc_idx is required when side='known'")

    known_tiles = [
        tile
        for tile in tiles
        if tile["known_doc_idx"] == known_doc_idx
    ]
    known_tiles.sort(
        key=lambda tile: (
            tile["known_start"],
            tile["known_end"],
            tile["tile_id"],
        )
    )
    return known_tiles, "known_start", "known_end"


def _tile_tooltip(tile: dict[str, Any], *, side: str) -> str:
    tile_id = tile["tile_id"]

    if side == "known":
        return (
            f"Tile {tile_id}; known document {tile['known_doc_idx']}; "
            f"known span {tile['known_start']}:{tile['known_end']}; "
            f"unknown span {tile['unknown_start']}:{tile['unknown_end']}; "
            f"length {tile['length']}"
        )

    association_text = "; ".join(
        (
            f"known document {association['known_doc_idx']} "
            f"({association.get('known_id', association['known_doc_idx'])}), "
            f"span {association['known_start']}:{association['known_end']}"
        )
        for association in tile["associations"]
    )
    return (
        f"Tile {tile_id}; unknown span "
        f"{tile['unknown_start']}:{tile['unknown_end']}; "
        f"length {tile['length']}; associations: {association_text}"
    )


def _render_plain_token_html(
    token_text: str,
    token_idx: int,
    *,
    show_token_indices: bool,
) -> str:
    index_html = (
        f'<span class="gst-token-index">{token_idx}</span>'
        if show_token_indices
        else ""
    )

    return f"""
    <span class="gst-token" title="token {token_idx}">
        {index_html}
        <span class="gst-token-text">{_html_escape(token_text)}</span>
    </span>
    """


def _render_hit_token_html(
    token_text: str,
    token_idx: int,
    tile: dict[str, Any],
    *,
    side: str,
    start_key: str,
    show_token_indices: bool,
    show_start_labels: bool,
) -> str:
    tile_id = tile["tile_id"]
    colour = _tile_colour(tile_id)
    is_start = tile[start_key] == token_idx

    label_html = ""
    if show_start_labels and is_start:
        label_html = f'<span class="gst-hit-label">T{tile_id}</span>'

    index_html = (
        f'<span class="gst-token-index">{token_idx}</span>'
        if show_token_indices
        else ""
    )

    return f"""
    <span class="gst-token gst-token-hit gst-t-{tile_id}"
          style="background:{colour};"
          title="{_html_escape(_tile_tooltip(tile, side=side))}">
        {label_html}
        {index_html}
        <span class="gst-token-text">{_html_escape(token_text)}</span>
    </span>
    """


def _render_token_stream_html_multi(
    tokens: Sequence[Any],
    tiles: Sequence[dict[str, Any]],
    *,
    side: str,
    tokenizer=None,
    known_doc_idx: int | None = None,
    max_tokens: int | None = None,
    show_token_indices: bool = True,
    show_start_labels: bool = True,
) -> str:
    """Render GST tiles with the same per-token styling as the n-gram figure."""
    visual_tiles, start_key, end_key = _visual_tiles_for_side(
        tiles,
        side=side,
        known_doc_idx=known_doc_idx,
    )

    n_tokens = len(tokens)
    render_limit = n_tokens if max_tokens is None else min(max_tokens, n_tokens)
    tile_at_position: list[dict[str, Any] | None] = [None] * render_limit

    for tile in visual_tiles:
        start = max(0, tile[start_key])
        end = min(render_limit, tile[end_key])

        for token_idx in range(start, end):
            if tile_at_position[token_idx] is not None:
                raise RuntimeError(
                    f"Overlapping visual GST tiles at {side} token {token_idx}."
                )
            tile_at_position[token_idx] = tile

    html_parts: list[str] = []

    for token_idx in range(render_limit):
        token_text = _display_tokens([tokens[token_idx]], tokenizer)
        tile = tile_at_position[token_idx]

        if tile is None:
            html_parts.append(
                _render_plain_token_html(
                    token_text,
                    token_idx,
                    show_token_indices=show_token_indices,
                )
            )
        else:
            html_parts.append(
                _render_hit_token_html(
                    token_text,
                    token_idx,
                    tile,
                    side=side,
                    start_key=start_key,
                    show_token_indices=show_token_indices,
                    show_start_labels=show_start_labels,
                )
            )

    if max_tokens is not None and n_tokens > max_tokens:
        html_parts.append(
            f'<span class="gst-omitted">... {n_tokens - max_tokens} tokens omitted</span>'
        )

    return "\n".join(html_parts)


def _build_panel_html(
    *,
    panel_title: str,
    panel_subtitle: str,
    token_stream_html: str,
) -> str:
    return f"""
    <div class="gst-panel">
        <div class="gst-panel-title">
            {_html_escape(panel_title)}
            <span class="gst-panel-subtitle">{panel_subtitle}</span>
        </div>
        <div class="gst-token-stream">
            {token_stream_html}
        </div>
    </div>
    """


def _build_tile_legend_html(
    unique_tiles: Sequence[dict[str, Any]],
    *,
    tokenizer=None,
    viz_uid: str,
    max_legend_items: int | None,
) -> str:
    shown_tiles = (
        list(unique_tiles)
        if max_legend_items is None
        else list(unique_tiles)[:max_legend_items]
    )

    legend_bits = [
        f"""
        <label class="gst-legend-chip gst-legend-clear"
               for="{viz_uid}-none">
            Clear selection
        </label>
        """
    ]

    for tile in shown_tiles:
        tile_id = tile["tile_id"]
        colour = _tile_colour(tile_id)
        label = _display_tokens(tile["tokens"], tokenizer).strip()
        legend_bits.append(
            f"""
            <label class="gst-legend-chip gst-legend-t-{tile_id}"
                   for="{viz_uid}-t-{tile_id}"
                   style="background:{colour};"
                   title="{_html_escape(label)}">
                T{tile_id}: {_html_escape(label)}
            </label>
            """
        )

    if max_legend_items is not None and len(unique_tiles) > max_legend_items:
        legend_bits.append(
            f"""
            <span class="gst-legend-more">
                ... {len(unique_tiles) - max_legend_items} more tiles not shown
            </span>
            """
        )

    return f"""
    <details class="gst-legend" open>
        <summary>Interactive tile legend ({len(unique_tiles)} tiles)</summary>
        <div class="gst-legend-body">{''.join(legend_bits)}</div>
    </details>
    """


def _build_radio_inputs_html(
    *,
    viz_uid: str,
    unique_tiles: Sequence[dict[str, Any]],
    max_legend_items: int | None,
) -> str:
    shown_tiles = (
        list(unique_tiles)
        if max_legend_items is None
        else list(unique_tiles)[:max_legend_items]
    )

    inputs = [
        f"""
        <input type="radio"
               name="{viz_uid}-selected-tile"
               id="{viz_uid}-none"
               class="gst-selector"
               checked>
        """
    ]

    for tile in shown_tiles:
        tile_id = tile["tile_id"]
        inputs.append(
            f"""
            <input type="radio"
                   name="{viz_uid}-selected-tile"
                   id="{viz_uid}-t-{tile_id}"
                   class="gst-selector">
            """
        )

    return "\n".join(inputs)


def _build_selection_css(
    *,
    viz_uid: str,
    unique_tiles: Sequence[dict[str, Any]],
    max_legend_items: int | None,
) -> str:
    shown_tiles = (
        list(unique_tiles)
        if max_legend_items is None
        else list(unique_tiles)[:max_legend_items]
    )

    rules = []

    for tile in shown_tiles:
        tile_id = tile["tile_id"]
        rules.append(
            f"""
            #{viz_uid}-t-{tile_id}:checked ~ .gst-paper-container .gst-token-hit {{
                opacity: 0.18;
                filter: grayscale(0.85);
            }}

            #{viz_uid}-t-{tile_id}:checked ~ .gst-paper-container .gst-token-hit.gst-t-{tile_id} {{
                opacity: 1;
                filter: none;
                outline: 3px solid #111827;
                outline-offset: 2px;
                box-shadow: 0 0 0 4px rgba(250, 204, 21, 0.55);
                position: relative;
                z-index: 3;
            }}

            #{viz_uid}-t-{tile_id}:checked ~ .gst-paper-container .gst-legend-chip {{
                opacity: 0.35;
            }}

            #{viz_uid}-t-{tile_id}:checked ~ .gst-paper-container .gst-legend-t-{tile_id},
            #{viz_uid}-t-{tile_id}:checked ~ .gst-paper-container .gst-legend-clear {{
                opacity: 1;
            }}

            #{viz_uid}-t-{tile_id}:checked ~ .gst-paper-container .gst-legend-t-{tile_id} {{
                outline: 3px solid #111827;
                outline-offset: 2px;
                font-weight: 800;
            }}
            """
        )

    return "\n".join(rules)


def build_many_knowns_gst_tiles_html(
    result: dict[str, Any],
    *,
    tokenizer=None,
    title: str = "Greedy string tiling",
    max_tokens_per_text: int | None = None,
    show_token_indices: bool = True,
    show_start_labels: bool = True,
    show_legend: bool = True,
    show_meta: bool = True,
    max_legend_items: int | None = 200,
) -> str:
    """
    Build a paper-style GST visualisation matching the n-gram figure design.

    This changes presentation only. It does not recompute or alter GST tiles.
    """
    if "known_token_lists" not in result or "unknown_tokens" not in result:
        raise ValueError(
            "result must contain token lists. Run "
            "gst_many_knowns_vs_unknown_tiles(..., include_token_lists=True)."
        )

    known_token_lists = result["known_token_lists"]
    unknown_tokens = result["unknown_tokens"]
    tiles = result["tiles"]
    known_ids = result.get("known_ids", list(range(len(known_token_lists))))
    unique_tiles = _group_unknown_tile_associations(tiles)
    viz_uid = f"gstviz-{uuid.uuid4().hex[:8]}"

    radio_inputs_html = _build_radio_inputs_html(
        viz_uid=viz_uid,
        unique_tiles=unique_tiles,
        max_legend_items=max_legend_items,
    )
    selection_css = _build_selection_css(
        viz_uid=viz_uid,
        unique_tiles=unique_tiles,
        max_legend_items=max_legend_items,
    )

    unknown_stream = _render_token_stream_html_multi(
        unknown_tokens,
        tiles,
        side="unknown",
        tokenizer=tokenizer,
        max_tokens=max_tokens_per_text,
        show_token_indices=show_token_indices,
        show_start_labels=show_start_labels,
    )
    unknown_panel = _build_panel_html(
        panel_title="Unknown text",
        panel_subtitle=(
            f"tokens: {len(unknown_tokens)} · "
            f"tiles: {len(unique_tiles)}"
        ),
        token_stream_html=unknown_stream,
    )

    known_panels: list[str] = []

    for known_doc_idx, known_tokens in enumerate(known_token_lists):
        known_tiles = [
            tile
            for tile in tiles
            if tile["known_doc_idx"] == known_doc_idx
        ]
        known_id = (
            known_ids[known_doc_idx]
            if known_doc_idx < len(known_ids)
            else known_doc_idx
        )
        known_stream = _render_token_stream_html_multi(
            known_tokens,
            tiles,
            side="known",
            tokenizer=tokenizer,
            known_doc_idx=known_doc_idx,
            max_tokens=max_tokens_per_text,
            show_token_indices=show_token_indices,
            show_start_labels=show_start_labels,
        )
        known_panels.append(
            _build_panel_html(
                panel_title=f"Known text {known_doc_idx + 1}",
                panel_subtitle=(
                    f"ID: {_html_escape(known_id)} · "
                    f"tokens: {len(known_tokens)} · "
                    f"tile associations: {len(known_tiles)}"
                ),
                token_stream_html=known_stream,
            )
        )

    legend_html = (
        _build_tile_legend_html(
            unique_tiles,
            tokenizer=tokenizer,
            viz_uid=viz_uid,
            max_legend_items=max_legend_items,
        )
        if show_legend
        else ""
    )

    meta_html = ""
    if show_meta:
        meta_html = f"""
        <div class="gst-meta">
            <span class="gst-pill">Tiles: {len(unique_tiles)}</span>
            <span class="gst-pill">Score: {result['score']}</span>
            <span class="gst-pill">Coverage: {result['coverage_of_shorter']:.4f}</span>
            <span class="gst-pill">Dice similarity: {result['dice_similarity']:.4f}</span>
            <span class="gst-pill">Unknown tokens: {result['n_unknown_tokens']}</span>
            <span class="gst-pill">Known texts: {result['n_known_texts']}</span>
            <span class="gst-pill">Known tokens: {result['n_known_tokens']}</span>
            <span class="gst-pill">Minimum match: {result['min_match']}</span>
        </div>
        """

    return f"""
    {radio_inputs_html}

    <style>
        .gst-selector {{
            display: none;
        }}

        .gst-paper-container {{
            box-sizing: border-box;
            width: 100%;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            line-height: 1.45;
            color: #222;
            background: white;
        }}

        .gst-paper-container *,
        .gst-paper-container *::before,
        .gst-paper-container *::after {{
            box-sizing: border-box;
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
            break-inside: avoid;
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
            overflow: visible;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            line-height: 2.2;
        }}

        .gst-token {{
            display: inline-flex;
            align-items: center;
            gap: 3px;
            margin: 2px 2px;
            padding: 2px 4px;
            border-radius: 5px;
            background: #f3f4f6;
            border: 1px solid transparent;
            vertical-align: middle;
        }}

        .gst-token-hit {{
            border: 1px solid rgba(0, 0, 0, 0.35);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.25);
            transition: opacity 0.15s ease, filter 0.15s ease, outline 0.15s ease;
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

        .gst-hit-label {{
            display: inline-block;
            background: rgba(17, 24, 39, 0.85);
            color: white;
            border-radius: 4px;
            padding: 0 3px;
            font-size: 9px;
            font-weight: 700;
            margin-right: 2px;
        }}

        .gst-legend {{
            margin-bottom: 14px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: white;
            padding: 8px 10px;
            break-inside: avoid;
        }}

        .gst-legend summary {{
            cursor: pointer;
            font-weight: 700;
        }}

        .gst-legend-body {{
            margin-top: 8px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}

        .gst-legend-chip {{
            display: inline-block;
            border: 1px solid rgba(0, 0, 0, 0.25);
            border-radius: 999px;
            padding: 3px 8px;
            max-width: 360px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 12px;
            cursor: pointer;
            user-select: none;
            transition: opacity 0.15s ease, outline 0.15s ease;
        }}

        .gst-legend-clear {{
            background: #111827;
            color: white;
            font-weight: 700;
        }}

        .gst-legend-more,
        .gst-omitted {{
            color: #6b7280;
            font-style: italic;
            padding: 3px 8px;
        }}

        {selection_css}

        @media print {{
            .gst-paper-container {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            .gst-legend summary {{
                list-style: none;
            }}

            .gst-legend summary::-webkit-details-marker {{
                display: none;
            }}
        }}
    </style>

    <div class="gst-paper-container">
        <div class="gst-title">{_html_escape(title)}</div>
        {meta_html}
        {legend_html}
        <div class="gst-layout">
            {unknown_panel}
            {''.join(known_panels)}
        </div>
    </div>
    """


def show_many_knowns_gst_tiles_html(
    result: dict[str, Any],
    *,
    tokenizer=None,
    title: str = "Greedy string tiling",
    max_tokens_per_text: int | None = None,
    show_token_indices: bool = True,
    show_start_labels: bool = True,
    show_legend: bool = True,
    show_meta: bool = True,
    max_legend_items: int | None = 200,
):
    """Display the paper-style GST figure in a notebook."""
    from IPython.display import HTML, display

    out_html = build_many_knowns_gst_tiles_html(
        result,
        tokenizer=tokenizer,
        title=title,
        max_tokens_per_text=max_tokens_per_text,
        show_token_indices=show_token_indices,
        show_start_labels=show_start_labels,
        show_legend=show_legend,
        show_meta=show_meta,
        max_legend_items=max_legend_items,
    )
    display(HTML(out_html))


def save_many_knowns_gst_tiles_html(
    result: dict[str, Any],
    output_path: str,
    *,
    tokenizer=None,
    title: str = "Greedy string tiling",
    max_tokens_per_text: int | None = None,
    show_token_indices: bool = True,
    show_start_labels: bool = True,
    show_legend: bool = True,
    show_meta: bool = True,
    max_legend_items: int | None = 200,
    width_px: int = 1100,
) -> str:
    """Save the GST figure as a standalone HTML document."""
    figure_html = build_many_knowns_gst_tiles_html(
        result,
        tokenizer=tokenizer,
        title=title,
        max_tokens_per_text=max_tokens_per_text,
        show_token_indices=show_token_indices,
        show_start_labels=show_start_labels,
        show_legend=show_legend,
        show_meta=show_meta,
        max_legend_items=max_legend_items,
    )

    document = f"""<!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{_html_escape(title)}</title>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: white;
            }}
            body {{
                width: {width_px}px;
                padding: 12px;
            }}
        </style>
    </head>
    <body>{figure_html}</body>
    </html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return str(path)


async def save_many_knowns_gst_tiles_pdf(
    result: dict[str, Any],
    output_path: str,
    *,
    tokenizer=None,
    title: str = "Greedy string tiling",
    max_tokens_per_text: int | None = None,
    show_token_indices: bool = True,
    show_start_labels: bool = True,
    show_legend: bool = True,
    show_meta: bool = True,
    max_legend_items: int | None = 200,
    width_px: int = 1100,
    margin_px: int = 12,
) -> str:
    """
    Save a tightly cropped, vector PDF using Playwright's async API.

    This version is suitable for Jupyter notebooks, which already run an
    asyncio event loop. Call it with:

        await save_many_knowns_gst_tiles_pdf(...)
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ImportError(
            "PDF export requires Playwright. Install it with:\n"
            "pip install playwright\n"
            "playwright install chromium"
        ) from exc

    figure_html = build_many_knowns_gst_tiles_html(
        result,
        tokenizer=tokenizer,
        title=title,
        max_tokens_per_text=max_tokens_per_text,
        show_token_indices=show_token_indices,
        show_start_labels=show_start_labels,
        show_legend=show_legend,
        show_meta=show_meta,
        max_legend_items=max_legend_items,
    )

    document = f"""<!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{ margin: 0; }}
            html, body {{
                margin: 0;
                padding: 0;
                background: white;
            }}
            body {{
                display: inline-block;
            }}
            #gst-export-wrap {{
                width: {width_px}px;
                padding: {margin_px}px;
                background: white;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
                box-sizing: border-box;
            }}
        </style>
    </head>
    <body>
        <div id="gst-export-wrap">{figure_html}</div>
    </body>
    </html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch()
        except Exception:
            system_chromium = (
                shutil.which("chromium")
                or shutil.which("chromium-browser")
                or shutil.which("google-chrome")
                or shutil.which("Google Chrome")
            )
            if system_chromium is None:
                raise RuntimeError(
                    "No Chromium executable was found. Run "
                    "'playwright install chromium' or install Chromium."
                )
            browser = await playwright.chromium.launch(
                executable_path=system_chromium,
                args=["--no-sandbox"],
            )

        try:
            page = await browser.new_page(
                viewport={
                    "width": width_px + 2 * margin_px,
                    "height": 1200,
                }
            )

            await page.set_content(document, wait_until="load")
            await page.emulate_media(media="print")
            await page.evaluate("document.fonts.ready")

            bounds = await page.locator("#gst-export-wrap").evaluate(
                """element => ({
                    width: Math.ceil(element.getBoundingClientRect().width),
                    height: Math.ceil(element.getBoundingClientRect().height)
                })"""
            )

            await page.pdf(
                path=str(path),
                width=f"{max(1, math.ceil(bounds['width']))}px",
                height=f"{max(1, math.ceil(bounds['height']))}px",
                margin={
                    "top": "0",
                    "right": "0",
                    "bottom": "0",
                    "left": "0",
                },
                print_background=True,
                prefer_css_page_size=False,
            )
        finally:
            await browser.close()

    return str(path)

