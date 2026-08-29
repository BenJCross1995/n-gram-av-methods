#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
from from_root import from_root
from transformers import AutoTokenizer

sys.path.insert(
    0,
    str(from_root("src")),
)

from read_and_write_docs import (
    read_jsonl,
    read_rds,
    write_rds,
)
from utils import apply_temp_doc_id

from n_gram_tracing import (
    common_ngrams,
    filter_len_common_ngrams,
    filter_ngrams_with_outside_occurrences_in_both_texts,
)

PIPELINE_VERSION = "known_unknown_v2"


from t5_paraphrase import (
    build_local_t5_context,
    create_expanded_occurrence_spans,
    find_ngram_occurrence_spans,
    generate_t5_candidates,
    load_t5_paraphrase_model,
    ngram_tokens_to_text,
    prepare_source_token_data,
    reconstruct_candidate_context,
)


# ============================================================
# Arguments
# ============================================================

def add_bool_pair(
    parser,
    name,
    default,
    help_text,
):
    dest = name.replace("-", "_")

    parser.add_argument(
        f"--{name}",
        dest=dest,
        action="store_true",
        help=help_text,
    )

    parser.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
    )

    parser.set_defaults(
        **{dest: default}
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Generate deterministic T5 span-infilling alternatives for "
            "common n-gram occurrences in one authorship-verification problem, "
            "including all known documents and the unknown document in one dataframe."
        )
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    ap.add_argument(
        "--known_loc",
        required=True,
    )

    ap.add_argument(
        "--unknown_loc",
        required=True,
    )

    ap.add_argument(
        "--metadata_loc",
        required=True,
    )

    ap.add_argument(
        "--model_loc",
        required=True,
        help=(
            "Tokenizer/model directory used to define the existing common "
            "token n-grams, e.g. GPT-2."
        ),
    )

    ap.add_argument(
        "--t5_model_loc",
        required=True,
        help=(
            "Local T5 model directory used for span infilling."
        ),
    )

    ap.add_argument(
        "--save_loc",
        required=True,
    )

    ap.add_argument(
        "--completed_loc",
        default=None,
    )

    ap.add_argument(
        "--error_loc",
        default=None,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    ap.add_argument(
        "--corpus",
        default="Wiki",
    )

    ap.add_argument(
        "--data_type",
        default="training",
    )

    ap.add_argument(
        "--problem",
        required=True,
    )

    # --------------------------------------------------------
    # Common n-grams
    # --------------------------------------------------------
    ap.add_argument(
        "--min_len",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--max_len",
        type=int,
        default=None,
    )

    add_bool_pair(
        ap,
        name="lowercase",
        default=True,
        help_text=(
            "Lowercase/casefold during canonical common n-gram "
            "collection and occurrence matching."
        ),
    )

    ap.add_argument(
        "--greatest_common",
        action="store_true",
        help=(
            "Use the same occurrence-aware greatest-common behaviour "
            "as the existing scoring pipeline."
        ),
    )

    # --------------------------------------------------------
    # Span/context settings
    # --------------------------------------------------------
    ap.add_argument(
        "--max_left_expansion",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--max_right_expansion",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--context_tokens",
        type=int,
        default=128,
        help=(
            "Number of ORIGINAL n-gram-tokenizer tokens retained on "
            "each side of the expanded span."
        ),
    )

    add_bool_pair(
        ap,
        name="lowercase-input",
        default=True,
        help_text=(
            "Lowercase/casefold the local context passed into T5."
        ),
    )

    add_bool_pair(
        ap,
        name="lowercase-output",
        default=True,
        help_text=(
            "Lowercase/casefold generated candidates and reconstructed text."
        ),
    )

    # --------------------------------------------------------
    # T5 decoding
    # --------------------------------------------------------
    ap.add_argument(
        "--num_beams",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--num_beam_groups",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--num_return_sequences",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--diversity_penalty",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
    )

    # --------------------------------------------------------
    # Optional smoke-test limits
    # --------------------------------------------------------
    ap.add_argument(
        "--max_ngrams",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--max_occurrences_per_ngram",
        type=int,
        default=None,
    )

    return ap.parse_args()


# ============================================================
# Helpers
# ============================================================

def safe_problem_name(problem):
    return (
        problem
        .replace("/", "_")
        .replace("\\", "_")
    )


def build_common_ngram_set(
    selected_known,
    unknown_text,
    tokenizer,
    args,
):
    """
    Reproduce the common-n-gram collection stage from the existing
    scoring pipeline, but stop before any LM scoring.
    """
    ngram_list = []
    metadata_rows = []

    num_known_docs = len(
        selected_known
    )

    print("Getting common n-grams")

    for i in range(
        num_known_docs
    ):
        known_doc = (
            selected_known["doc_id"]
            .iloc[i]
        )

        known_text = (
            selected_known["text"]
            .iloc[i]
        )

        print(
            f"Working on known doc "
            f"{i + 1}/{num_known_docs}: "
            f"{known_doc}"
        )

        try:
            common = common_ngrams(
                text1=known_text,
                text2=unknown_text,
                tokenizer=tokenizer,
                include_subgrams=args.greatest_common,
                lowercase=args.lowercase,
            )

            if args.greatest_common:
                common = (
                    filter_ngrams_with_outside_occurrences_in_both_texts(
                        ngrams=common,
                        known_text=known_text,
                        unknown_text=unknown_text,
                        tokenizer=tokenizer,
                        lowercase=args.lowercase,
                    )
                )

            ngrams_found = True

        except Exception:
            print(
                f"WARNING: n-gram collection failed "
                f"for known doc {known_doc}"
            )

            print(
                traceback.format_exc()
            )

            common = []
            ngrams_found = False

        ngram_list.extend(
            common
        )

        metadata_rows.append({
            "known_doc": known_doc,
            "ngrams_found": ngrams_found,
            "num_ngrams": len(common),
        })

    # Same order-preserving problem-level deduplication pattern as
    # the supplied scoring runner.
    distinct_ngrams = [
        list(x)
        for x in dict.fromkeys(
            tuple(x)
            for x in ngram_list
        )
    ]

    distinct_ngrams = sorted(
        distinct_ngrams,
        key=lambda x: (
            len(x),
            sum(
                len(str(token))
                for token in x
            ),
        ),
    )

    print(
        f"There are {len(distinct_ngrams)} "
        f"distinct n-grams before length filtering"
    )

    filtered_ngrams = (
        filter_len_common_ngrams(
            distinct_ngrams,
            min_len=args.min_len,
            max_len=args.max_len,
        )
    )

    if args.max_ngrams is not None:
        filtered_ngrams = (
            filtered_ngrams[
                :args.max_ngrams
            ]
        )

    print(
        f"There are {len(filtered_ngrams)} "
        f"n-grams to paraphrase"
    )

    return (
        filtered_ngrams,
        pd.DataFrame(metadata_rows),
    )


def process_document_paraphrases(
    *,
    document_type,
    doc_number,
    doc_id,
    author,
    text,
    filtered_ngrams,
    ngram_tokenizer,
    ngram_model_name,
    t5_tokenizer,
    t5_model,
    t5_model_name,
    device,
    args,
    problem_metadata,
):
    """
    Generate T5 candidate replacements for every eligible occurrence of the
    retained common n-grams in one document.

    The same function is used for known and unknown documents so both sides
    of the AV problem are represented identically in the final dataframe.
    """
    print()
    print("=" * 80)
    print(
        f"Processing {document_type} document "
        f"{doc_number}: {doc_id}"
    )
    print("=" * 80)

    source_token_data = prepare_source_token_data(
        text=text,
        tokenizer=ngram_tokenizer,
        lowercase=args.lowercase,
    )

    full_tokens = source_token_data["tokens"]
    rows = []

    for ngram_index, ngram in enumerate(
        filtered_ngrams,
        start=1,
    ):
        ngram_text = ngram_tokens_to_text(
            ngram,
            ngram_tokenizer,
        )

        print(
            f"[{document_type} {doc_number}] "
            f"N-gram {ngram_index}/{len(filtered_ngrams)} "
            f"(length={len(ngram)}): {ngram_text!r}"
        )

        # Mirror score_ngrams_to_df():
        #   standard        -> all token occurrences
        #   greatest_common -> only independent occurrences in THIS document
        occurrence_spans = find_ngram_occurrence_spans(
            full_tokens=full_tokens,
            ngram_tokens=ngram,
            all_ngrams=filtered_ngrams,
            greatest_common=args.greatest_common,
            allow_overlaps=False,
        )

        if args.max_occurrences_per_ngram is not None:
            occurrence_spans = occurrence_spans[
                :args.max_occurrences_per_ngram
            ]

        num_document_occurrences = len(occurrence_spans)

        if num_document_occurrences == 0:
            continue

        expanded_spans = create_expanded_occurrence_spans(
            source_token_data=source_token_data,
            occurrence_spans=occurrence_spans,
            max_left_expansion=args.max_left_expansion,
            max_right_expansion=args.max_right_expansion,
        )

        for span in expanded_spans:
            local = build_local_t5_context(
                source_token_data=source_token_data,
                span=span,
                context_tokens=args.context_tokens,
                lowercase_input=args.lowercase_input,
            )

            generation_start = time.perf_counter()

            candidates = generate_t5_candidates(
                masked_text=local["masked_text"],
                original_span=local["generation_original_span"],
                tokenizer=t5_tokenizer,
                model=t5_model,
                device=device,
                num_beams=args.num_beams,
                num_beam_groups=args.num_beam_groups,
                num_return_sequences=args.num_return_sequences,
                diversity_penalty=args.diversity_penalty,
                max_new_tokens=args.max_new_tokens,
                lowercase_output=args.lowercase_output,
            )

            generation_time_seconds = (
                time.perf_counter()
                - generation_start
            )

            for candidate_info in candidates:
                reconstructed_text = reconstruct_candidate_context(
                    generation_left_context=local[
                        "generation_left_context"
                    ],
                    generation_right_context=local[
                        "generation_right_context"
                    ],
                    candidate=candidate_info["candidate"],
                    starts_with_space=candidate_info[
                        "starts_with_space"
                    ],
                    lowercase_output=args.lowercase_output,
                )

                rows.append({
                    # ----------------------------------------
                    # Problem metadata
                    # ----------------------------------------
                    **problem_metadata,

                    # ----------------------------------------
                    # Document metadata
                    # ----------------------------------------
                    "document_type": document_type,
                    "doc_number": doc_number,
                    "doc_id": doc_id,
                    "author": author,
                    "document_num_tokens": len(full_tokens),

                    # ----------------------------------------
                    # Models/settings
                    # ----------------------------------------
                    "ngram_model": ngram_model_name,
                    "t5_model": t5_model_name,
                    "lowercase_ngrams": args.lowercase,
                    "lowercase_input": args.lowercase_input,
                    "lowercase_output": args.lowercase_output,
                    "greatest_common": args.greatest_common,
                    "context_tokens_each_side": args.context_tokens,
                    "max_left_expansion": args.max_left_expansion,
                    "max_right_expansion": args.max_right_expansion,
                    "num_beams": args.num_beams,
                    "num_beam_groups": args.num_beam_groups,
                    "num_return_sequences": args.num_return_sequences,
                    "diversity_penalty": args.diversity_penalty,
                    "max_new_tokens": args.max_new_tokens,

                    # ----------------------------------------
                    # Original n-gram
                    # ----------------------------------------
                    "ngram_index": ngram_index,
                    "ngram_len": len(ngram),
                    "ngram_tokens": json.dumps(
                        list(ngram),
                        ensure_ascii=False,
                    ),
                    "ngram_text": ngram_text,
                    "num_document_occurrences": num_document_occurrences,

                    # ----------------------------------------
                    # Occurrence / expanded source span
                    # ----------------------------------------
                    "occurrence_index": span["occurrence_index"],
                    "ngram_token_start": span["ngram_token_start"],
                    "ngram_token_end": span["ngram_token_end"],
                    "span_token_start": span["span_token_start"],
                    "span_token_end": span["span_token_end"],
                    "left_expansion": span["left_expansion"],
                    "right_expansion": span["right_expansion"],
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "span_tokens": json.dumps(
                        span["span_tokens"],
                        ensure_ascii=False,
                    ),
                    "original_span": span["original_span"],

                    # ----------------------------------------
                    # Local T5 context
                    # ----------------------------------------
                    "original_context": local["original_context"],
                    "generation_original_context": local[
                        "generation_original_context"
                    ],
                    "masked_text": local["masked_text"],

                    # ----------------------------------------
                    # Candidate
                    # ----------------------------------------
                    "candidate_rank": candidate_info["candidate_rank"],
                    "generation_rank": candidate_info["generation_rank"],
                    "candidate": candidate_info["candidate"],
                    "generation_score": candidate_info[
                        "generation_score"
                    ],
                    "starts_with_space": candidate_info[
                        "starts_with_space"
                    ],
                    "candidate_tokens": json.dumps(
                        candidate_info["candidate_tokens"],
                        ensure_ascii=False,
                    ),
                    "reconstructed_text": reconstructed_text,
                    "generation_time_seconds": generation_time_seconds,
                    "offset_text_is_casefolded": source_token_data[
                        "offset_text_is_casefolded"
                    ],
                })

    print(
        f"Finished {document_type} document {doc_number}. "
        f"Generated {len(rows)} candidate rows."
    )

    return rows


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(args):
    print(f"Pipeline version: {PIPELINE_VERSION}")

    os.makedirs(
        args.save_loc,
        exist_ok=True,
    )

    selected_problem = (
        args.problem
        .strip()
        .strip('"')
        .strip("'")
    )

    output_name = (
        safe_problem_name(
            selected_problem
        )
        + ".rds"
    )

    save_file = os.path.join(
        args.save_loc,
        output_name,
    )

    if args.completed_loc:
        os.makedirs(
            args.completed_loc,
            exist_ok=True,
        )

        completed_file = os.path.join(
            args.completed_loc,
            output_name,
        )

        if os.path.exists(
            completed_file
        ):
            print(
                f"Result for {selected_problem} "
                f"already exists in completed_loc. "
                f"Exiting."
            )

            return

    if os.path.exists(
        save_file
    ):
        print(
            f"Path {save_file} already exists. "
            f"Exiting."
        )

        return

    print(
        f"Working on problem: "
        f"{selected_problem}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("Loading data")

    known = read_jsonl(
        args.known_loc
    )

    known = apply_temp_doc_id(
        known
    )

    unknown = read_jsonl(
        args.unknown_loc
    )

    unknown = apply_temp_doc_id(
        unknown
    )

    metadata = read_rds(
        args.metadata_loc
    )

    filtered_metadata = metadata[
        (metadata["corpus"] == args.corpus)
        & (
            metadata["problem"]
            == selected_problem
        )
    ]

    if filtered_metadata.empty:
        raise ValueError(
            "No metadata rows found for "
            f"corpus={args.corpus!r}, "
            f"problem={selected_problem!r}"
        )

    known_author = (
        filtered_metadata[
            "known_author"
        ].iloc[0]
    )

    unknown_author = (
        filtered_metadata[
            "unknown_author"
        ].iloc[0]
    )

    target = (
        known_author
        == unknown_author
    )

    selected_known = known[
        known["author"]
        == known_author
    ]

    selected_unknown = unknown[
        unknown["author"]
        == unknown_author
    ]

    if selected_known.empty:
        raise ValueError(
            f"No known documents found "
            f"for {known_author!r}"
        )

    if selected_unknown.empty:
        raise ValueError(
            f"No unknown documents found "
            f"for {unknown_author!r}"
        )

    # Preserve the existing scoring pipeline behaviour.
    unknown_doc = (
        selected_unknown[
            "doc_id"
        ].iloc[0]
    )

    unknown_text = (
        selected_unknown[
            "text"
        ].iloc[0]
    )

    num_known_docs = len(
        selected_known
    )

    print(
        f"There are {num_known_docs} "
        f"known texts in the problem"
    )

    # --------------------------------------------------------
    # Load ONLY the tokenizer that defines the AV n-grams
    # --------------------------------------------------------
    print(
        f"Loading n-gram tokenizer: "
        f"{args.model_loc}"
    )

    ngram_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model_loc,
            use_fast=True,
        )
    )

    ngram_model_name = (
        os.path.basename(
            os.path.normpath(
                args.model_loc
            )
        )
    )

    # --------------------------------------------------------
    # Get the problem-level common n-gram set
    # --------------------------------------------------------
    filtered_ngrams, per_known_metadata = (
        build_common_ngram_set(
            selected_known=selected_known,
            unknown_text=unknown_text,
            tokenizer=ngram_tokenizer,
            args=args,
        )
    )

    # --------------------------------------------------------
    # Load T5
    # --------------------------------------------------------
    print(
        f"Loading T5 model: "
        f"{args.t5_model_loc}"
    )

    (
        t5_tokenizer,
        t5_model,
        device,
    ) = load_t5_paraphrase_model(
        args.t5_model_loc
    )

    t5_model_name = (
        os.path.basename(
            os.path.normpath(
                args.t5_model_loc
            )
        )
    )

    print(
        f"T5 device: {device}"
    )

    # --------------------------------------------------------
    # Generate candidates from BOTH sides of the AV problem
    # --------------------------------------------------------
    rows = []

    problem_start = (
        time.perf_counter()
    )

    problem_metadata = {
        "data_type": args.data_type,
        "corpus": args.corpus,
        "problem": selected_problem,
        "known_author": known_author,
        "unknown_author": unknown_author,
        "target": target,
        "num_known_docs": num_known_docs,
        "unknown_doc": unknown_doc,
        "pipeline_version": PIPELINE_VERSION,
    }

    # ========================================================
    # Known documents
    # ========================================================
    for known_doc_number in range(
        1,
        num_known_docs + 1,
    ):
        known_doc = selected_known[
            "doc_id"
        ].iloc[
            known_doc_number - 1
        ]

        known_text = selected_known[
            "text"
        ].iloc[
            known_doc_number - 1
        ]

        rows.extend(
            process_document_paraphrases(
                document_type="known",
                doc_number=known_doc_number,
                doc_id=known_doc,
                author=known_author,
                text=known_text,
                filtered_ngrams=filtered_ngrams,
                ngram_tokenizer=ngram_tokenizer,
                ngram_model_name=ngram_model_name,
                t5_tokenizer=t5_tokenizer,
                t5_model=t5_model,
                t5_model_name=t5_model_name,
                device=device,
                args=args,
                problem_metadata=problem_metadata,
            )
        )

    # ========================================================
    # Unknown document
    # ========================================================
    rows.extend(
        process_document_paraphrases(
            document_type="unknown",
            doc_number=1,
            doc_id=unknown_doc,
            author=unknown_author,
            text=unknown_text,
            filtered_ngrams=filtered_ngrams,
            ngram_tokenizer=ngram_tokenizer,
            ngram_model_name=ngram_model_name,
            t5_tokenizer=t5_tokenizer,
            t5_model=t5_model,
            t5_model_name=t5_model_name,
            device=device,
            args=args,
            problem_metadata=problem_metadata,
        )
    )

    # --------------------------------------------------------
    # Final combined dataframe
    # --------------------------------------------------------
    results_df = pd.DataFrame(
        rows
    )

    problem_time_seconds = (
        time.perf_counter()
        - problem_start
    )

    if not results_df.empty:
        results_df[
            "num_distinct_common_ngrams"
        ] = len(
            filtered_ngrams
        )

        results_df[
            "num_known_ngram_searches"
        ] = len(
            per_known_metadata
        )

        results_df[
            "num_successful_known_ngram_searches"
        ] = int(
            per_known_metadata[
                "ngrams_found"
            ].sum()
        )

        results_df[
            "problem_generation_time_seconds"
        ] = problem_time_seconds

    if not results_df.empty:
        first_columns = [
            "data_type",
            "corpus",
            "problem",
            "known_author",
            "unknown_author",
            "target",
            "document_type",
            "doc_number",
            "doc_id",
            "author",
            "ngram_index",
            "ngram_len",
            "ngram_text",
            "occurrence_index",
            "left_expansion",
            "right_expansion",
            "original_span",
            "candidate_rank",
            "candidate",
            "generation_score",
            "reconstructed_text",
        ]

        first_columns = [
            col
            for col in first_columns
            if col in results_df.columns
        ]

        results_df = results_df[
            first_columns
            + [
                col
                for col in results_df.columns
                if col not in first_columns
            ]
        ]

    print(
        f"Generated {len(results_df)} "
        f"candidate rows"
    )

    if not results_df.empty:
        type_counts = (
            results_df["document_type"]
            .value_counts()
            .to_dict()
        )

        print(
            f"Known candidate rows: "
            f"{type_counts.get('known', 0)}"
        )

        print(
            f"Unknown candidate rows: "
            f"{type_counts.get('unknown', 0)}"
        )

    print(
        f"Generation time: "
        f"{problem_time_seconds:.2f} seconds"
    )

    print(
        f"Saving to {save_file}"
    )

    write_rds(
        results_df,
        save_file,
    )

    print(
        "Saved successfully"
    )


# ============================================================
# Error handling
# ============================================================

def write_error(
    args,
    exc,
    tb,
):
    if args.error_loc is None:
        return

    os.makedirs(
        args.error_loc,
        exist_ok=True,
    )

    selected_problem = (
        args.problem
        .strip()
        .strip('"')
        .strip("'")
    )

    error_file = os.path.join(
        args.error_loc,
        safe_problem_name(
            selected_problem
        )
        + ".rds",
    )

    new_error_df = pd.DataFrame([
        {
            "error_sent_datetime": (
                datetime.now()
                .astimezone()
                .isoformat(
                    timespec="seconds"
                )
            ),

            "data_type": args.data_type,
            "corpus": args.corpus,
            "problem": selected_problem,

            "ngram_model": (
                os.path.basename(
                    os.path.normpath(
                        args.model_loc
                    )
                )
            ),

            "t5_model": (
                os.path.basename(
                    os.path.normpath(
                        args.t5_model_loc
                    )
                )
            ),

            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": tb,
        }
    ])

    if os.path.exists(
        error_file
    ):
        existing_error_df = (
            read_rds(
                error_file
            )
        )

        error_df = pd.concat(
            [
                existing_error_df,
                new_error_df,
            ],
            ignore_index=True,
        )

    else:
        error_df = new_error_df

    write_rds(
        error_df,
        error_file,
    )

    print(
        f"Error info written to "
        f"{error_file}"
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    try:
        run_pipeline(
            args
        )

    except Exception as exc:
        tb = traceback.format_exc()

        print(
            f"ERROR encountered while "
            f"processing problem: "
            f"{args.problem}"
        )

        print(
            tb
        )

        write_error(
            args,
            exc,
            tb,
        )

        raise


if __name__ == "__main__":
    main()
