#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd

from from_root import from_root

sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_jsonl, read_rds
from model_loading import load_model
from utils import apply_temp_doc_id, build_metadata_df
from n_gram_tracing import common_ngrams
from n_gram_scoring import score_ngrams_to_df
from excel_functions import create_excel_template

def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline to score the raw data")
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
    ap.add_argument("--compute_type", default="himem")
    # N-gram
    ap.add_argument("--ngram_n", type=int, default=2)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--num_tokens", type=int, default=None)
        
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
    tokenizer, model = load_model(args.model_loc)
    # special_tokens = distinct_special_chars(tokenizer=tokenizer)
    model_name = os.path.basename(os.path.normpath(args.model_loc))
    
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
    problem_metadata['max_context_tokens'] = args.num_tokens
    problem_metadata['compute_type'] = args.compute_type
    
    # Some column rearranging
    # data_type before corpus
    corpus_idx = problem_metadata.columns.get_loc('corpus')
    if 'data_type' in problem_metadata.columns:
        problem_metadata.drop(columns=['data_type'], inplace=True)
    problem_metadata.insert(corpus_idx, 'data_type', args.data_type)

    # scoring_model after corpus
    corpus_idx = problem_metadata.columns.get_loc('corpus')  # re-fetch (indices may have shifted)
    if 'scoring_model' in problem_metadata.columns:
        problem_metadata.drop(columns=['scoring_model'], inplace=True)
    problem_metadata.insert(corpus_idx + 1, 'scoring_model', model_name)
    
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
    common = common_ngrams(known_text, unknown_text, args.ngram_n, tokenizer, lowercase=args.lowercase)
    
    print(f"There are {len(common)} n-grams in common!")
    
    # Get the no context scores
    print("Scoring the No Context n-grams")
    no_context_df = score_ngrams_to_df(common, model, tokenizer, full_text=None, use_bos=True)
    
    # Get the known scores
    print("Scoring the Known n-grams")
    known_scored_df = score_ngrams_to_df(common, model, tokenizer, full_text=known_text, use_bos=True, num_tokens=args.num_tokens)
    
    # Score the unknown phrases
    print("Scoring the Unknown n-grams")
    unknown_scored_df = score_ngrams_to_df(common, model, tokenizer, full_text=unknown_text, use_bos=True, num_tokens=args.num_tokens)
    
    create_excel_template(
        known = known_scored_df,
        unknown = unknown_scored_df,
        no_context = no_context_df,
        metadata = problem_metadata,
        docs = docs_df,
        path = save_loc
    )
    
if __name__ == "__main__":
    main()