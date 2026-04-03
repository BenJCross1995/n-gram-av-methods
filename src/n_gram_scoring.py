from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import pandas as pd
import torch.nn.functional as F

from n_gram_tracing import (
    tokenize_to_tokens,
    tokens_to_ids,
    tokens_to_text,
    texts_around_each_token_ngram,
    get_trimmed_context_before_span,
)

def score_ngrams(
    ngram: Union[str, Sequence[str]],
    model: Any,
    tokenizer: Any,
    text: Optional[Union[str, Sequence[Any]]] = None,
    *,
    lowercase: bool = True,
    use_bos: bool = False,
) -> Dict[str, Any]:
    """
    Causal LM scoring. N-gram is assumed to be at the END of the text.

    `ngram` may be:
      - a string
      - a sequence of tokenizer tokens

    `text` may be:
      - None
      - a string
      - a sequence of tokenizer tokens
    """
    if isinstance(ngram, str):
        ngram_tokens = tokenize_to_tokens(
            ngram,
            tokenizer=tokenizer,
            lowercase=lowercase,
        )
        phrase = ngram.casefold() if lowercase else ngram
    else:
        ngram_tokens = list(ngram)
        phrase = tokens_to_text(ngram_tokens, tokenizer)

    if len(ngram_tokens) < 1:
        raise ValueError("ngram must have at least 1 token")

    phrase_ids_list = tokens_to_ids(ngram_tokens, tokenizer)
    ngram_len = len(phrase_ids_list)

    # Build input ids with minimal unnecessary work
    if text is None:
        seq_tokens = ngram_tokens
        input_ids_list = phrase_ids_list

    elif isinstance(text, str):
        seq_for_tok = text.casefold() if lowercase else text
        enc = tokenizer(
            seq_for_tok,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids_list = enc.get("input_ids", [])
        if input_ids_list and isinstance(input_ids_list[0], (list, tuple)):
            input_ids_list = input_ids_list[0]
        seq_tokens = tokenizer.convert_ids_to_tokens(input_ids_list)

    else:
        seq_tokens = list(text)
        input_ids_list = tokens_to_ids(seq_tokens, tokenizer)

    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None and hasattr(model, "config"):
        bos_id = getattr(model.config, "bos_token_id", None)

    has_bos = use_bos and (bos_id is not None)
    if has_bos:
        bos = torch.tensor([[int(bos_id)]], dtype=torch.long, device=device)
        ids_for_model = torch.cat([bos, input_ids], dim=1)
    else:
        ids_for_model = input_ids

    # Hard cap for model forward pass
    max_positions = getattr(model.config, "n_positions", None)
    if max_positions is None:
        max_positions = getattr(model.config, "max_position_embeddings", None)

    if max_positions is not None and ids_for_model.shape[1] > max_positions:
        ids_for_model = ids_for_model[:, -max_positions:]

        # Rebuild visible seq_tokens to match the truncated model input
        if has_bos:
            visible_ids = ids_for_model[:, 1:]
        else:
            visible_ids = ids_for_model
        seq_tokens = tokenizer.convert_ids_to_tokens(visible_ids[0].tolist())

    tokens: List[str] = list(seq_tokens)
    text_len = len(tokens)

    if text_len == 0:
        log_probs: List[Optional[float]] = []

    elif text_len == 1:
        if has_bos:
            with torch.no_grad():
                logits = model(input_ids=ids_for_model).logits
                lp_vocab = F.log_softmax(logits[:, :-1, :], dim=-1)
                val = (
                    lp_vocab.gather(-1, ids_for_model[:, 1:].unsqueeze(-1))
                    .squeeze(-1)[0, 0]
                    .item()
                )
            log_probs = [float(val)]
        else:
            log_probs = [None]

    else:
        with torch.no_grad():
            logits = model(input_ids=ids_for_model).logits
            lp_vocab = F.log_softmax(logits[:, :-1, :], dim=-1)
            next_ids = ids_for_model[:, 1:]
            vals = (
                lp_vocab.gather(-1, next_ids.unsqueeze(-1))
                .squeeze(-1)[0]
                .detach()
                .cpu()
                .tolist()
            )

        if has_bos:
            log_probs = [float(v) for v in vals]
        else:
            log_probs = [None] + [float(v) for v in vals]

    tail = log_probs[-ngram_len:] if ngram_len <= len(log_probs) else log_probs
    ngram_log_probs = [v for v in tail if v is not None]
    ngram_sum_log_probs = float(sum(ngram_log_probs))

    return {
        "phrase": phrase,
        "tokens": ngram_tokens,
        "num_tokens": ngram_len,
        "text_len": text_len,
        "log_probs": ngram_log_probs,
        "sum_log_probs": ngram_sum_log_probs,
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
) -> pd.DataFrame:
    """
    Build a DataFrame with:
      - phrase_num
      - phrase_occurrence

    If full_text is None:
      - score each n-gram once without context

    If full_text is provided:
      - find all token-based occurrences in the full tokenised text
      - score each occurrence separately
    """
    rows = []

    for phrase_num, ng in enumerate(ngrams, start=1):
        if isinstance(ng, str):
            phrase_tokens = tokenize_to_tokens(
                ng,
                tokenizer=tokenizer,
                lowercase=lowercase,
            )
        else:
            phrase_tokens = list(ng)

        if len(phrase_tokens) == 0:
            continue

        if full_text is None:
            res = score_ngrams(
                ngram=phrase_tokens,
                model=model,
                tokenizer=tokenizer,
                text=None,
                lowercase=lowercase,
                use_bos=use_bos,
            )
            rows.append({
                "phrase_num": phrase_num,
                "phrase_occurrence": 1,
                **res,
            })
            continue

        _, token_spans, tokenized_text = texts_around_each_token_ngram(
            full_text,
            phrase_tokens,
            tokenizer=tokenizer,
            start=0,
            lowercase=lowercase,
            allow_overlaps=False,
            return_spans=True,
            return_tokenized_text=True,
            return_text=False,
        )

        for i, tok_span in enumerate(token_spans, start=1):
            if num_tokens == 0:
                occ_text = phrase_tokens
                effective_use_bos = True
            elif num_tokens is not None:
                occ_text = get_trimmed_context_before_span(
                    tokens=tokenized_text,
                    token_span=tok_span,
                    max_tokens=num_tokens,
                    return_text=False,
                    tokenizer=tokenizer,
                )
                effective_use_bos = use_bos
            else:
                _, end = tok_span
                occ_text = tokenized_text[:end]
                effective_use_bos = use_bos

            res = score_ngrams(
                ngram=phrase_tokens,
                model=model,
                tokenizer=tokenizer,
                text=occ_text,
                lowercase=lowercase,
                use_bos=effective_use_bos,
            )

            rows.append({
                "phrase_num": phrase_num,
                "phrase_occurrence": i,
                **res,
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        first = ["phrase_num", "phrase_occurrence"]
        df = df[first + [c for c in df.columns if c not in first]]

    return df