from typing import Any, Dict, List, Optional, Sequence, Union
import torch

import pandas as pd
import torch.nn.functional as F

from n_gram_tracing import tokens_to_text, texts_around_each_ngram, get_trimmed_context_before_span
    
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
    Causal LM scoring. N-gram is assumed to be at the END of the text.

    If use_bos=True and a BOS token id is available, the first token gets a log-prob
    P(x1 | BOS). Otherwise first token log-prob is None.

    Output keys:
      phrase, ngram_tokens, ngram_len, tokens, text_len, log_probs,
      ngram_log_probs, ngram_sum_log_probs
    """
    # phrase as text
    phrase = ngram if isinstance(ngram, str) else tokens_to_text(list(ngram), tokenizer)

    # ngram tokens + ngram_len
    phrase_for_tok = phrase.casefold() if lowercase else phrase
    phrase_ids = tokenizer(phrase_for_tok, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    ngram_len = int(phrase_ids.numel())
    if ngram_len < 1:
        raise ValueError("ngram must have at least 1 token")
    ngram_tokens: List[str] = tokenizer.convert_ids_to_tokens(phrase_ids.tolist())

    # choose sequence to score
    seq_text = phrase if text is None else text
    seq_for_tok = seq_text.casefold() if lowercase else seq_text
    input_ids = tokenizer(seq_for_tok, add_special_tokens=False, return_tensors="pt")["input_ids"]  # (1, T)

    # move to device
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # maybe prepend BOS
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None and hasattr(model, "config"):
        bos_id = getattr(model.config, "bos_token_id", None)

    has_bos = use_bos and (bos_id is not None)
    if has_bos:
        bos = torch.tensor([[int(bos_id)]], dtype=torch.long, device=device)
        ids_for_model = torch.cat([bos, input_ids], dim=1)  # (1, T+1)
    else:
        ids_for_model = input_ids  # (1, T)

    # tokens for display (original sequence, not including BOS)
    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    text_len = len(tokens)

    # compute per-token log probs aligned to `tokens`
    if text_len == 0:
        log_probs: List[Optional[float]] = []
    elif text_len == 1:
        if has_bos:
            # score single token given BOS
            with torch.no_grad():
                logits = model(input_ids=ids_for_model).logits  # (1,2,V)
                lp_vocab = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1,1,V)
                val = lp_vocab.gather(-1, ids_for_model[:, 1:].unsqueeze(-1)).squeeze(-1)[0, 0].item()
            log_probs = [float(val)]
        else:
            log_probs = [None]
    else:
        with torch.no_grad():
            logits = model(input_ids=ids_for_model).logits  # (1, T' , V)
            lp_vocab = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T'-1, V)
            next_ids = ids_for_model[:, 1:]  # (1, T'-1)
            vals = lp_vocab.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)[0].detach().cpu().tolist()

        if has_bos:
            # vals aligns to the original tokens (length T)
            log_probs = [float(v) for v in vals]
        else:
            # vals aligns to tokens[1:] (length T-1)
            log_probs = [None] + [float(v) for v in vals]

    # ngram at end -> last ngram_len token log probs
    tail = log_probs[-ngram_len:] if ngram_len <= len(log_probs) else log_probs
    ngram_log_probs = [v for v in tail if v is not None]
    ngram_sum_log_probs = float(sum(ngram_log_probs))

    return {
        "phrase": phrase,
        "tokens": ngram_tokens,
        "num_tokens": ngram_len,
        "log_probs": ngram_log_probs,
        "sum_log_probs": ngram_sum_log_probs,
        "text_tokens": tokens,
        "text_len": text_len,
        "text_log_probs": log_probs,
    }
    
def score_ngrams_to_df(
    ngrams,
    model,
    tokenizer,
    full_text: str | None = None,
    *,
    lowercase: bool = True,
    use_bos: bool = False,
    num_tokens: Optional[int] = None
) -> pd.DataFrame:
    """Build a test DataFrame with:
      - phrase_num (1-based index in ngrams list)
      - phrase_occurrence (1-based occurrence in text; 1 if no-context)
    and all outputs from score_ngram_end.

    If full_text is None:
      - scores each n-gram once with text=None (no context)
    If full_text is provided:
      - uses texts_around_each_ngram(full_text, phrase, ...) to get all occurrences
        (strings that end with the n-gram), then scores each.
    """
    rows = []

    for phrase_num, ng in enumerate(ngrams, start=1):
        phrase = ng if isinstance(ng, str) else tokens_to_text(list(ng), tokenizer)

        if full_text is None:
            # no-context: score once
            res = score_ngrams(
                ngram=ng,
                model=model,
                tokenizer=tokenizer,
                text=None,
                lowercase=lowercase,
                use_bos=use_bos
            )
            rows.append({"phrase_num": phrase_num, "phrase_occurrence": 1, **res})
            continue

        # with context: score every occurrence
        prefixes, token_spans, tokenized_text = texts_around_each_ngram(
            full_text,
            phrase, 
            lowercase=lowercase,
            return_token_spans=True, 
            return_tokenized_text=True,
            tokenizer=tokenizer
        )

        for i, (prefix, tok_span) in enumerate(zip(prefixes, token_spans), start=1):
            
            if num_tokens:
                occ_text = get_trimmed_context_before_span(
                    tokens = tokenized_text,
                    token_span = tok_span,
                    max_tokens = num_tokens,
                    return_text = True,
                    tokenizer = tokenizer
                )
            else:
                occ_text = prefix
                
            res = score_ngrams(
                ngram=ng,
                model=model,
                tokenizer=tokenizer,
                text=occ_text,
                lowercase=lowercase,
                use_bos=use_bos
            )
            
            rows.append({
                "phrase_num": phrase_num,
                "phrase_occurrence": i,
                **res
            })

    df = pd.DataFrame(rows)

    # Ensure first two cols are phrase_num + phrase_occurrence
    first = ["phrase_num", "phrase_occurrence"]
    df = df[first + [c for c in df.columns if c not in first]]

    return df