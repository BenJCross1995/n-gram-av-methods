#!/usr/bin/env python3
import argparse
import sys
import os

import pandas as pd

from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from model_loading import load_model
from read_and_write_docs import read_jsonl, read_rds
from utils import apply_temp_doc_id, build_metadata_df
from n_gram_tracing import (
    common_ngrams,
    filter_len_common_ngrams
)
from weighted_n_gram_tracing import (
    get_tokens,
    weighted_ngram_tracing_df,
    aggregate_weighted_df,
    create_excel_template
)

def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline to complete weighted n-gram tracing")
    # Paths
    ap.add_argument("--known_loc")
    ap.add_argument("--unknown_loc")
    ap.add_argument("--metadata_loc")
    ap.add_argument("--model_loc")
    ap.add_argument("--save_loc")
    ap.add_argument("--completed_loc", default=None)
    # Dataset hinting
    ap.add_argument("--corpus", default="Wiki")
    ap.add_argument("--data_type", default="training")
    ap.add_argument("--known_doc")
    ap.add_argument("--unknown_doc")
    # N-gram
    ap.add_argument("--min_len", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--lowercase", action="store_true")
    # Weighting
    ap.add_argument("--weight", type=str, choices=["linear", "power", "exp"], default="linear")
    ap.add_argument("--alpha", type=float, default=1.0, help="Only used when --weight=power (w(n)=n**alpha).")
    ap.add_argument("--base", type=float, default=2.0, help="Only used when --weight=exp (w(n)=base**n).")
    
    return ap.parse_args()

def main():
    
    args=parse_args()
    
    # Ensure the directory exists before beginning
    os.makedirs(args.save_loc, exist_ok=True)
    
    # -----
    # LOAD DATA & LOCAL MODEL
    # -----
    specific_problem = f"{args.known_doc} vs {args.unknown_doc}"
    save_loc = f"{args.save_loc}/{specific_problem}.xlsx"
    
    if args.completed_loc:
        completed_loc = f"{args.completed_loc}/{specific_problem}.xlsx"
        if os.path.exists(completed_loc):
            print(f"Result for {specific_problem} already exists in the completed folder. Exiting.")
            sys.exit()
    
    # Skip the problem if already exists
    if os.path.exists(save_loc):
        print(f"Path {save_loc} already exists. Exiting.")
        sys.exit()
        
    print(f"Working on problem: {specific_problem}")
    
    print("Loading model")
    tokenizer = load_model(args.model_loc, load_model=False)

    print("Loading data")
    known = read_jsonl(args.known_loc)
    known = apply_temp_doc_id(known)
    
    unknown = read_jsonl(args.unknown_loc)
    unknown = apply_temp_doc_id(unknown)
    
    print("Data loaded")
    
    # NOTE - Is this used?
    metadata = read_rds(args.metadata_loc)
    filtered_metadata = metadata[metadata['corpus'] == args.corpus]
    agg_metadata = build_metadata_df(filtered_metadata, known, unknown)
    
    # -----
    # Get the chosen text & metadata
    # -----
    
    known_text = known[known['doc_id'] == args.known_doc].reset_index().loc[0, 'text'].lower()
    unknown_text = unknown[unknown['doc_id'] == args.unknown_doc].reset_index().loc[0, 'text'].lower()
    
    problem_metadata = agg_metadata[(agg_metadata['known_doc_id'] == args.known_doc)
                                    & (agg_metadata['unknown_doc_id'] == args.unknown_doc)].reset_index()
    problem_metadata['target'] = problem_metadata['known_author'] == problem_metadata['unknown_author']
    
    # Some column rearranging
    # data_type before corpus
    corpus_idx = problem_metadata.columns.get_loc('corpus')
    if 'data_type' in problem_metadata.columns:
        problem_metadata.drop(columns=['data_type'], inplace=True)
    problem_metadata.insert(corpus_idx, 'data_type', args.data_type)
    
    # problem before known_author (always move; adjust index if problem was before)
    if 'problem' in problem_metadata.columns and 'known_author' in problem_metadata.columns:
        problem_idx = problem_metadata.columns.get_loc('problem')
        known_author_idx = problem_metadata.columns.get_loc('known_author')

        problem_col = problem_metadata.pop('problem')

        # if problem was before known_author, known_author shifted left by 1 after pop
        if problem_idx < known_author_idx:
            known_author_idx -= 1

        problem_metadata.insert(known_author_idx, 'problem', problem_col)
        
            # -----
    # Create document dataframe
    # -----
    
    # This is used to display the text
    docs_df = pd.DataFrame(
    {
        "known":   [args.corpus, args.data_type, args.known_doc, known_text],
        "unknown": [args.corpus, args.data_type, args.unknown_doc, unknown_text],
    },
    index=["corpus", "data type", "doc", "text"],
    )
    
    # -----
    # Get common n-grams
    # -----
    
    print("Getting common n-grams")
    common = common_ngrams(known_text, unknown_text, tokenizer, lowercase=args.lowercase)
    filtered_common = filter_len_common_ngrams(common, min_len=args.min_len, max_len=args.max_len)
    
    print(f"There are {len(filtered_common)} n-grams in common!")
    
    # -----
    # Get tokens for texts
    # -----
    
    print("Getting tokens for known and unknown text")
    unknown_tokens = get_tokens(unknown_text, tokenizer, lowercase=args.lowercase)
    known_tokens = get_tokens(known_text, tokenizer, lowercase=args.lowercase)
    
    # -----
    # Get Weighting
    # -----
    
    print("Applying the weighted n-gram tracing")
    weighted_df = weighted_ngram_tracing_df(
        known_tokens=known_tokens,
        unknown_tokens=unknown_tokens,
        common_n_grams=filtered_common,
        weight=args.weight,
        alpha=args.alpha,          # used only if weight="power": w(n)=n**alpha
        base=args.base,           # used only if weight="exp":   w(n)=base**n
        decimals = 5,
        validate_common = True,
    )
    
    print("Aggregating the results by token level")
    agg_weighted_df = aggregate_weighted_df(
        df=weighted_df,
        problem_metadata=problem_metadata,
        level_col="token_level",
        weight=args.weight,
        alpha=args.alpha,
        base=args.base
    )
    
    print("Saving")
    create_excel_template(
        weighted_df=weighted_df,
        agg_weighted_df=agg_weighted_df,
        docs=docs_df,
        path=save_loc
    )
    
if __name__ == "__main__":
    main()