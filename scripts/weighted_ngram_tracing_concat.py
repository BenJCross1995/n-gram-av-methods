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
    tokenize_to_tokens,
    common_ngrams,
    filter_len_common_ngrams
)
from weighted_n_gram_tracing import (
    get_tokens,
    weighted_ngram_tracing_df,
    aggregate_weighted_df,
    create_excel_template,
    create_ngram_df
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
    ap.add_argument("--problem")
    # N-gram
    ap.add_argument("--min_len", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--lowercase", dest="lowercase", action="store_true")
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
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
    
    selected_problem = args.problem.strip().strip('"').strip("'")
    
    save_loc = f"{args.save_loc}/{selected_problem}.xlsx"
    
    if args.completed_loc:
        completed_loc = f"{args.completed_loc}/{selected_problem}.xlsx"
        if os.path.exists(completed_loc):
            print(f"Result for {selected_problem} already exists in the completed folder. Exiting.")
            sys.exit()
    
    # Skip the problem if already exists
    if os.path.exists(save_loc):
        print(f"Path {save_loc} already exists. Exiting.")
        sys.exit()
        
    print(f"Working on problem: {selected_problem}")
    
    print("Loading model")
    tokenizer = load_model(args.model_loc, load_model=False)
    model_name = os.path.basename(os.path.normpath(args.model_loc))
        
    print("Loading data")
    known = read_jsonl(args.known_loc)
    known = apply_temp_doc_id(known)
    
    unknown = read_jsonl(args.unknown_loc)
    unknown = apply_temp_doc_id(unknown)
    
    print("Data loaded")

    # NOTE - Is this used?
    metadata = read_rds(args.metadata_loc)
    filtered_metadata = metadata[
        (metadata['corpus'] == args.corpus)
        & (metadata['problem'] == selected_problem)
    ]

    # agg_metadata = build_metadata_df(filtered_metadata, known, unknown)
    
    # -----
    # Get the chosen text & metadata
    # -----
    
    known_author = filtered_metadata['known_author'].iloc[0]
    unknown_author = filtered_metadata['unknown_author'].iloc[0]
    
    selected_known = known[known['author'] == known_author]
    selected_unknown = unknown[unknown['author'] == unknown_author]

    unknown_doc = selected_unknown['doc_id'].iloc[0]
    unknown_text = selected_unknown['text'].iloc[0]
    unknown_tokens = tokenize_to_tokens(unknown_text, tokenizer=tokenizer, lowercase=args.lowercase)
    
    num_known_problems = len(selected_known)
    print(f"There are {num_known_problems} known texts in the problem")
    
    # -----
    # Get the common n-grams between the unknown and known texts
    # -----
    
    known_tokens_list = []
    ngram_list = []
    problem_metadata_list = []

    print("Getting common n-grams")
    for i in range(1, num_known_problems + 1):
        print(f"Working on doc {i}")
        known_doc = selected_known['doc_id'].iloc[i-1]
        known_text = selected_known['text'].iloc[i - 1]
        known_tokens = tokenize_to_tokens(known_text, tokenizer=tokenizer, lowercase=args.lowercase)
        known_tokens_list.append(known_tokens)
        
        # Perform a try/except to try to find common n-grams
        # Leave a flag if unable to find them
        try:
            common = common_ngrams(
                text1=known_text,
                text2=unknown_text,
                tokenizer=tokenizer,
                include_subgrams=False,
                lowercase=args.lowercase
            )
            ngrams_found = True
        except:
            common = []
            ngrams_found = False
        ngrams_shared = len(common)
        ngram_list.extend(common)
    
        row = {
            "data_type": args.data_type,
            "corpus": args.corpus,
            "scoring_model": model_name,
            "problem": selected_problem,
            "known_author": known_author,
            "unknown_author": unknown_author,
            "target": known_author == unknown_author,
            "known_doc": known_doc,
            "unknown_doc": unknown_doc,
            "ngrams_found": ngrams_found,
            "num_ngrams": ngrams_shared,
        }
        problem_metadata_list.append(row)

    problem_metadata = pd.DataFrame(problem_metadata_list)
    # Only keep the distinct list
    distinct_ngram_list = [list(x) for x in dict.fromkeys(tuple(x) for x in ngram_list)]

    # Sort first by the number of tokens and second by character length
    distinct_ngram_list = sorted(
        distinct_ngram_list,
        key=lambda x: (len(x), sum(len(str(token)) for token in x))
    )
    
    # Filter by token length if desired
    filtered_ngrams = filter_len_common_ngrams(
        distinct_ngram_list,
        min_len=args.min_len,
        max_len=args.max_len
    )
    print(f"There are {len(filtered_ngrams)} n-grams in common!")
    
    # Crete an ngram table
    ngram_df = create_ngram_df(filtered_ngrams)
    
    overall_problem_metadata = (
        problem_metadata[['data_type', 'corpus', 'scoring_model', 
                          'problem', 'known_author', 'unknown_author', 'target']]
        .drop_duplicates()
    )
    overall_problem_metadata['num_problems'] = len(problem_metadata)
    overall_problem_metadata['ngrams_found'] = sum(problem_metadata['ngrams_found'])
    overall_problem_metadata['completed'] = overall_problem_metadata['num_problems'] == overall_problem_metadata['ngrams_found']
    
    # -----
    # Get Weighting
    # -----
    
    print("Applying the weighted n-gram tracing")
    weighted_df = weighted_ngram_tracing_df(
        known_tokens_list=known_tokens_list,
        unknown_tokens=unknown_tokens,
        common_n_grams=filtered_ngrams,
        weight=args.weight,
        alpha=args.alpha,          # used only if weight="power": w(n)=n**alpha
        base=args.base,           # used only if weight="exp":   w(n)=base**n
        decimals = 5,
        validate_common = True,
    )
    
    print("Aggregating the results by token level")
    agg_weighted_df = aggregate_weighted_df(
        df=weighted_df,
        problem_metadata=overall_problem_metadata,
        level_col="token_level",
        weight=args.weight,
        alpha=args.alpha,
        base=args.base
    )
    # -----
    # Create document dataframe
    # -----
    
    print("Saving")
    create_excel_template(
        weighted_df=weighted_df,
        agg_weighted_df=agg_weighted_df,
        ngrams=ngram_df,
        path=save_loc
    )
    
if __name__ == "__main__":
    main()