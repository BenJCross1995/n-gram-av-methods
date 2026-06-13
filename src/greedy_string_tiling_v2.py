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

def gst_many_knowns_vs_unknown_tiles_v2(
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
    n_texts = len(known_texts)

    if known_ids is None:
        known_ids = list(range(n_texts))

    if len(known_ids) != n_texts:
        raise ValueError("known_ids must have the same length as known_texts")

    # Tokenize all inputs once.
    known_token_lists = [
        tokenize_to_tokens(text, tokenizer, lowercase=lowercase)
        for text in known_texts
    ]

    unknown_tokens = tokenize_to_tokens(
        unknown_text,
        tokenizer,
        lowercase=lowercase,
    )

    # Cache per-document token counts; used for similarity stats and for
    # building the position map without repeated len() calls.
    known_lens = [len(tokens) for tokens in known_token_lists]
    n_known_tokens = sum(known_lens)
    n_unknown_tokens = len(unknown_tokens)

    # ------------------------------------------------------------------
    # Build the concatenated known sequence and a position map.
    #
    # We use three parallel flat lists rather than a list of dicts, because
    # for large corpora the dict-per-token overhead is substantial both in
    # memory and allocation time. Separator slots are marked with -1 in
    # pos_doc_idx, which is cheaper than None for the hot path check.
    # ------------------------------------------------------------------
    total_concat_len = n_known_tokens + max(0, n_texts - 1)

    known_concat: list[Any] = []
    pos_doc_idx: list[int] = []     # -1 at separator slots
    pos_token_idx: list[int] = []   # 0 at separator slots (unused)

    # Pre-extend underlying buffers; minor allocation win on large inputs.
    known_concat_append = known_concat.append
    pos_doc_idx_append = pos_doc_idx.append
    pos_token_idx_append = pos_token_idx.append

    for doc_idx, tokens in enumerate(known_token_lists):
        if doc_idx > 0:
            # Unique separator object: cannot equal any token.
            known_concat_append(object())
            pos_doc_idx_append(-1)
            pos_token_idx_append(0)

        for tok_idx, token in enumerate(tokens):
            known_concat_append(token)
            pos_doc_idx_append(doc_idx)
            pos_token_idx_append(tok_idx)

    # Sanity check on buffer length; cheap and catches accidental drift.
    assert len(known_concat) == total_concat_len, (
        f"concat length mismatch: {len(known_concat)} vs {total_concat_len}"
    )

    # ------------------------------------------------------------------
    # Run GST once over the concatenated known sequence.
    # ------------------------------------------------------------------
    raw_tiles, score = greedy_string_tiling(
        known_tokens=known_concat,
        unknown_tokens=unknown_tokens,
        min_match=min_match,
    )

    # ------------------------------------------------------------------
    # Map each raw tile (in concat coordinates) back to its source document.
    #
    # By construction of the separator scheme, no tile can cross a boundary,
    # so start and end positions always lie in the same document. We still
    # guard against it because it costs nothing.
    # ------------------------------------------------------------------
    tiles: list[dict[str, Any]] = []

    for tile_id, tile in enumerate(raw_tiles, start=1):
        concat_start = tile.known_start
        concat_last = concat_start + tile.length - 1

        start_doc = pos_doc_idx[concat_start]
        end_doc = pos_doc_idx[concat_last]

        # Defensive: separators or cross-document spans should be impossible.
        if start_doc < 0 or end_doc < 0 or start_doc != end_doc:
            continue

        known_doc_idx = start_doc
        known_id = known_ids[known_doc_idx]
        known_start = pos_token_idx[concat_start]
        known_end = known_start + tile.length

        unknown_start = tile.unknown_start
        unknown_end = unknown_start + tile.length

        # Slice once from the per-document token list.
        known_tile_tokens = list(
            known_token_lists[known_doc_idx][known_start:known_end]
        )

        # Defensive verification: if GST is correct this never fires, but the
        # check is O(L) per tile and tile counts are small, so keep it.
        unknown_tile_tokens = unknown_tokens[unknown_start:unknown_end]

        if known_tile_tokens != list(unknown_tile_tokens):
            raise ValueError(
                f"Tile {tile_id} does not match after mapping. "
                f"known_tile_tokens={known_tile_tokens}, "
                f"unknown_tile_tokens={list(unknown_tile_tokens)}"
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

    # ------------------------------------------------------------------
    # Build the deduped, sorted token-list population.
    #
    # Dedupe before sorting: dedupe is O(n) on tuple hashes, sort is
    # O(k log k) on the deduped set. Sorting first wastes work on
    # duplicates that will be dropped.
    # ------------------------------------------------------------------
    tile_token_lists = sort_ngrams(
        dedupe_ngrams([list(tile["tokens"]) for tile in tiles])
    )

    # ------------------------------------------------------------------
    # Similarity stats over total token counts.
    # ------------------------------------------------------------------
    shorter_len = min(n_known_tokens, n_unknown_tokens)
    total_len = n_known_tokens + n_unknown_tokens

    result: dict[str, Any] = {
        "tiles": tiles,
        "tile_token_lists": tile_token_lists,
        "raw_tiles": raw_tiles,
        "score": score,
        "coverage_of_shorter": score / shorter_len if shorter_len else 0.0,
        "dice_similarity": (2 * score / total_len) if total_len else 0.0,
        "n_known_tokens": n_known_tokens,
        "n_unknown_tokens": n_unknown_tokens,
        "n_known_texts": n_texts,
        "min_match": min_match,
        "lowercase": lowercase,
    }

    if include_token_lists:
        result["known_token_lists"] = known_token_lists
        result["known_concat_tokens"] = known_concat
        result["unknown_tokens"] = unknown_tokens

    return result