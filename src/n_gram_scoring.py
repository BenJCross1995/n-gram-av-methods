from typing import Any, Dict, List, Optional, Sequence, Union

import time
import torch
import pandas as pd
import torch.nn.functional as F

from n_gram_tracing import (
    tokenize_to_tokens,
    tokens_to_ids,
    tokens_to_text,
    texts_around_each_token_ngram,
    get_trimmed_context_before_span,
    texts_around_each_independent_token_ngram,
)
from model_loading import get_max_tokens


def score_ngrams(
    ngram: Union[str, Sequence[str]],
    model: Any,
    tokenizer: Any,
    text: Optional[str] = None,
    *,
    lowercase: bool = True,
    use_bos: bool = False,
) -> Dict[str, Any]:
    """
    Score an n-gram using a causal language model.

    The n-gram is assumed to occur at the end of `text`.

    Parameters
    ----------
    ngram : str or sequence of str
        N-gram to score. This can either be raw text or a sequence
        of tokenizer tokens.

    model : Any
        Causal language model used for scoring.

    tokenizer : Any
        Tokenizer associated with the model.

    text : str, optional
        Full sequence to score, where the n-gram occurs at the end.
        If None, the n-gram itself is used as the input text.

    lowercase : bool, default=True
        Whether to lowercase/casefold text before tokenisation.

    use_bos : bool, default=False
        Whether to prepend the model's BOS token, where available.

    Returns
    -------
    dict
        Dictionary containing the n-gram, its tokens, token-level
        log probabilities, and summed log probability.
    """
    # Convert a raw string n-gram into tokenizer tokens.
    if isinstance(ngram, str):
        ngram_tokens = tokenize_to_tokens(
            ngram,
            tokenizer=tokenizer,
            lowercase=lowercase,
        )

        phrase = ngram.casefold() if lowercase else ngram

    else:
        # If tokens have already been supplied, retain them directly.
        ngram_tokens = list(ngram)
        phrase = tokens_to_text(ngram_tokens, tokenizer)

    # An empty n-gram cannot be scored.
    if len(ngram_tokens) < 1:
        raise ValueError("ngram must have at least 1 token")

    # Convert the n-gram tokens to model token IDs.
    phrase_ids_list = tokens_to_ids(
        ngram_tokens,
        tokenizer,
    )

    ngram_len = len(phrase_ids_list)

    # If no surrounding text is supplied, score the n-gram itself.
    seq_text = phrase if text is None else text

    # Apply the same lowercasing behaviour to the complete input.
    seq_for_tok = seq_text.casefold() if lowercase else seq_text
    
    # Tokenise without special tokens.
    # BOS is added manually below if required.
    input_ids = tokenizer(
        seq_for_tok,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    # Put the model into evaluation mode.
    model.eval()

    # Move the input onto the same device as the model.
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Try to obtain the BOS token from the tokenizer.
    bos_id = getattr(
        tokenizer,
        "bos_token_id",
        None,
    )

    # Fall back to the model configuration if necessary.
    if bos_id is None and hasattr(model, "config"):
        bos_id = getattr(
            model.config,
            "bos_token_id",
            None,
        )

    # Only use BOS if requested and one is actually available.
    has_bos = use_bos and (bos_id is not None)
    
    # Add BOS manually where required.
    if has_bos:
        bos = torch.tensor(
            [[int(bos_id)]],
            dtype=torch.long,
            device=device,
        )

        ids_for_model = torch.cat(
            [bos, input_ids],
            dim=1,
        )

    else:
        ids_for_model = input_ids

    # Convert the token IDs back to strings for reporting.
    tokens: List[str] = tokenizer.convert_ids_to_tokens(
        input_ids[0].tolist()
    )

    text_len = len(tokens)

    # ------------------------------------------------------------
    # Calculate token-level causal log probabilities
    # ------------------------------------------------------------
    
    if text_len == 0:
        log_probs: List[Optional[float]] = []

    elif text_len == 1:

        # If BOS is available, it can be used to predict the first
        # actual token in the sequence.
        if has_bos:
            with torch.no_grad():
                logits = model(
                    input_ids=ids_for_model
                ).logits

                lp_vocab = F.log_softmax(
                    logits[:, :-1, :],
                    dim=-1,
                )

                val = (
                    lp_vocab
                    .gather(
                        -1,
                        ids_for_model[:, 1:].unsqueeze(-1),
                    )
                    .squeeze(-1)[0, 0]
                    .item()
                )

            log_probs = [float(val)]

        else:
            # Without BOS, there is no preceding token from which
            # to calculate the probability of the first token.
            log_probs = [None]

    else:
        with torch.no_grad():

            # Run the complete sequence through the causal LM.
            logits = model(
                input_ids=ids_for_model
            ).logits

            # Convert logits to log probabilities.
            #
            # Each position predicts the token immediately following it.
            lp_vocab = F.log_softmax(
                logits[:, :-1, :],
                dim=-1,
            )

            next_ids = ids_for_model[:, 1:]

            # Extract the log probability assigned to the token that
            # actually occurred at each subsequent position.
            vals = (
                lp_vocab
                .gather(
                    -1,
                    next_ids.unsqueeze(-1),
                )
                .squeeze(-1)[0]
                .detach()
                .cpu()
                .tolist()
            )

        if has_bos:
            # BOS allows every actual input token to receive a score.
            log_probs = [float(v) for v in vals]

        else:
            # Without BOS, the first token cannot receive a score.
            log_probs = [None] + [float(v) for v in vals]

    # The n-gram occurs at the end of the input sequence, so select
    # the final ngram_len token probabilities.
    tail = (
        log_probs[-ngram_len:]
        if ngram_len <= len(log_probs)
        else log_probs
    )

    # Remove any unavailable probability values.
    ngram_log_probs = [
        v
        for v in tail
        if v is not None
    ]

    # Sum the token-level log probabilities to produce the
    # occurrence-level n-gram score.
    ngram_sum_log_probs = float(
        sum(ngram_log_probs)
    )

    return {
        "phrase": phrase,
        "tokens": ngram_tokens,
        "num_tokens": ngram_len,
        "text_len": text_len,
        "log_probs": ngram_log_probs,
        "sum_log_probs": ngram_sum_log_probs,
        # "text_tokens": tokens,
        # "text_log_probs": log_probs,
    }

def score_ngrams_to_df(
    ngrams,
    model,
    tokenizer,
    full_text: Optional[str] = None,
    *,
    lowercase: bool = True,
    use_bos: bool = False,
    num_tokens: Optional[int] = None,
    greatest_common: bool = False,
) -> pd.DataFrame:
    """
    Score n-grams and return occurrence-level results as a DataFrame.

    If `full_text` is None:
        Each n-gram is scored once without surrounding document context.

    If `full_text` is provided:
        Each token-based occurrence is found and scored separately.

    When `num_tokens` is an explicit positive integer, the requested
    amount of preceding context is automatically capped so that the
    context, n-gram, and optional BOS token fit within the model's
    maximum token limit.

    IMPORTANT:
        `num_tokens=None` retains the original behaviour and uses the
        complete prefix without applying this context-window cap.

    Parameters
    ----------
    ngrams : iterable
        N-grams to score.

    model : Any
        Causal language model used for scoring.

    tokenizer : Any
        Tokenizer associated with the model.

    full_text : str, optional
        Complete text in which occurrences of each n-gram are found.

    lowercase : bool, default=True
        Whether to lowercase/casefold text before tokenisation.

    use_bos : bool, default=False
        Whether to prepend a BOS token when scoring.

    num_tokens : int, optional
        Number of preceding context tokens to include.

        - 0:
            Score without preceding context.
        - Positive integer:
            Include up to this many preceding context tokens, capped
            where necessary by the model's maximum sequence length.
        - None:
            Preserve the original behaviour and use the full prefix.

    greatest_common : bool, default=False
        Whether to use the greatest-common n-gram occurrence method.

    Returns
    -------
    pandas.DataFrame
        One row per scored n-gram occurrence.
    """
    rows = []

    # Get the model's maximum sequence length once.
    #
    # This is only used when num_tokens is explicitly specified.
    # num_tokens=None retains the original uncapped prefix behaviour.
    model_max_tokens = get_max_tokens(
        model=model,
        tokenizer=tokenizer,
    )
    
    for phrase_num, ng in enumerate(ngrams, start=1):
        
        # Convert string n-grams into tokenizer tokens.
        if isinstance(ng, str):
            phrase_tokens = tokenize_to_tokens(
                ng,
                tokenizer=tokenizer,
                lowercase=lowercase,
            )

        else:
            phrase_tokens = list(ng)

        # Ignore empty n-grams.
        if len(phrase_tokens) == 0:
            continue

        # ============================================================
        # No full text supplied
        # ============================================================
        #
        # In this case there are no document occurrences or preceding
        # context to identify. Score the n-gram directly.
        # ============================================================

        if full_text is None:

            score_start = time.perf_counter()

            res = score_ngrams(
                ngram=phrase_tokens,
                model=model,
                tokenizer=tokenizer,
                text=None,
                lowercase=lowercase,
                use_bos=use_bos,
            )

            score_time_seconds = (
                time.perf_counter()
                - score_start
            )

            rows.append({
                "phrase_num": phrase_num,
                "phrase_occurrence": 1,
                "score_time_seconds": score_time_seconds,
                **res,
            })

            continue

        # ============================================================
        # Find occurrences of the n-gram in the complete text
        # ============================================================

        # greatest_common means that we use the greatest-common
        # n-gram method. This allows subgrams of a larger n-gram
        # where they also occur independently elsewhere.
        if greatest_common:

            prefixes, token_spans, tokenized_text = (
                texts_around_each_independent_token_ngram(
                    full_text,
                    phrase_tokens,
                    all_ngrams=ngrams,
                    tokenizer=tokenizer,
                    start=0,
                    lowercase=lowercase,
                    allow_overlaps=False,
                    return_spans=True,
                    return_tokenized_text=True,
                )
            )

        else:
            
            prefixes, token_spans, tokenized_text = (
                texts_around_each_token_ngram(
                    full_text,
                    phrase_tokens,
                    tokenizer=tokenizer,
                    start=0,
                    lowercase=lowercase,
                    allow_overlaps=False,
                    return_spans=True,
                    return_tokenized_text=True,
                )
            )

        # ============================================================
        # Score each occurrence separately
        # ============================================================
        
        for i, (prefix, tok_span) in enumerate(
            zip(prefixes, token_spans),
            start=1,
        ):

            # --------------------------------------------------------
            # No preceding context
            # --------------------------------------------------------

            if num_tokens == 0:

                # Score only the n-gram itself.
                occ_text = tokens_to_text(
                    phrase_tokens,
                    tokenizer,
                )

                # Preserve your existing context-free behaviour.
                effective_use_bos = True
                
            # --------------------------------------------------------
            # Explicit context length
            # --------------------------------------------------------

            elif num_tokens is not None:

                effective_use_bos = use_bos

                # Convert the n-gram tokens to IDs so that the length
                # used here matches the actual model input length.
                phrase_ids = tokens_to_ids(
                    phrase_tokens,
                    tokenizer,
                )

                ngram_len = len(phrase_ids)

                # Determine whether a BOS token will also occupy one
                # position in the model's context window.
                bos_id = getattr(
                    tokenizer,
                    "bos_token_id",
                    None,
                )

                if (
                    bos_id is None
                    and hasattr(model, "config")
                ):
                    bos_id = getattr(
                        model.config,
                        "bos_token_id",
                        None,
                    )

                bos_tokens = (
                    1
                    if effective_use_bos
                    and bos_id is not None
                    else 0
                )

                # If we cannot determine a maximum model length, retain
                # the requested num_tokens value rather than changing
                # the existing behaviour.
                if model_max_tokens is None:

                    effective_num_tokens = num_tokens

                else:

                    # Calculate how much space remains for preceding
                    # context once the full n-gram and optional BOS
                    # token have been accounted for.
                    #
                    # Example:
                    #
                    # model maximum = 1024
                    # n-gram length = 961
                    # BOS           = 0
                    #
                    # available context = 1024 - 961 = 63
                    available_context_tokens = (
                        model_max_tokens
                        - ngram_len
                        - bos_tokens
                    )

                    # If this is negative, the n-gram itself is larger
                    # than the model's context window. Removing context
                    # cannot solve that case.
                    if available_context_tokens < 0:
                        raise ValueError(
                            f"N-gram {phrase_num} contains "
                            f"{ngram_len} tokens and cannot fit "
                            f"within the model's "
                            f"{model_max_tokens}-token context window."
                        )

                    # Use the requested context length unless it would
                    # cause the complete model input to exceed the
                    # context window.
                    #
                    # For example, if only 63 context tokens fit:
                    #
                    # num_tokens=50  -> use 50
                    # num_tokens=60  -> use 60
                    # num_tokens=70  -> use 63
                    # num_tokens=100 -> use 63
                    effective_num_tokens = min(
                        num_tokens,
                        available_context_tokens,
                    )

                # Build the occurrence text using the capped context
                # length followed by the complete n-gram.
                occ_text = get_trimmed_context_before_span(
                    tokens=tokenized_text,
                    token_span=tok_span,
                    max_tokens=effective_num_tokens,
                    return_text=True,
                    tokenizer=tokenizer,
                )

            # --------------------------------------------------------
            # num_tokens=None
            # --------------------------------------------------------
            #
            # IMPORTANT: This is unchanged from your original code.
            #
            # The full prefix returned by the occurrence-finding
            # function is passed directly to score_ngrams.
            # --------------------------------------------------------

            else:
                occ_text = prefix
                effective_use_bos = use_bos

            # --------------------------------------------------------
            # Score this individual occurrence
            # --------------------------------------------------------

            score_start = time.perf_counter()

            res = score_ngrams(
                ngram=phrase_tokens,
                model=model,
                tokenizer=tokenizer,
                text=occ_text,
                lowercase=lowercase,
                use_bos=effective_use_bos,
            )

            score_time_seconds = (
                time.perf_counter()
                - score_start
            )

            rows.append({
                "phrase_num": phrase_num,
                "phrase_occurrence": i,
                "score_time_seconds": score_time_seconds,
                **res,
            })

    # Convert all occurrence-level scores into a DataFrame.
    df = pd.DataFrame(rows)

    if not df.empty:

        # Keep the identifying columns at the front.
        first = [
            "phrase_num",
            "phrase_occurrence",
        ]

        df = df[
            first
            + [
                c
                for c in df.columns
                if c not in first
            ]
        ]

    return df