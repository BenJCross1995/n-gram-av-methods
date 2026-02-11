#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

FIXED_NUM_TOKENS = [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run score_ngrams.py in a loop over fixed num_tokens values"
    )

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
    ap.add_argument("--ngram_n", type=int, default=2)
    ap.add_argument("--lowercase", action="store_true")

    # Keep for "same args" compatibility; ignored here (we override per loop)
    ap.add_argument("--num_tokens", type=int, default=None)

    return ap.parse_args()

def with_suffix_dir(base_dir: str, num_tokens: int | None) -> str:
    base_dir = base_dir.rstrip("/\\")
    if num_tokens is None:
        return base_dir
    return f"{base_dir}_{num_tokens}"

def main():
    args = parse_args()

    # score_ngrams.py is in the same directory as this wrapper
    this_dir = Path(__file__).resolve().parent
    target_script = this_dir / "score_ngrams.py"

    if not target_script.exists():
        raise FileNotFoundError(f"Could not find {target_script}")

    base_cmd = [
        sys.executable,
        str(target_script),
        "--known_loc", args.known_loc,
        "--unknown_loc", args.unknown_loc,
        "--metadata_loc", args.metadata_loc,
        "--model_loc", args.model_loc,
        "--corpus", args.corpus,
        "--data_type", args.data_type,
        "--known_doc", args.known_doc,
        "--unknown_doc", args.unknown_doc,
        "--ngram_n", str(args.ngram_n),
    ]

    if args.lowercase:
        base_cmd.append("--lowercase")

    # Only pass completed_loc if provided
    has_completed = bool(args.completed_loc)

    for nt in FIXED_NUM_TOKENS:
        save_loc_nt = with_suffix_dir(args.save_loc, nt)
        completed_loc_nt = with_suffix_dir(args.completed_loc, nt) if has_completed else None

        cmd = base_cmd.copy()
        cmd += ["--save_loc", save_loc_nt]

        if has_completed:
            cmd += ["--completed_loc", completed_loc_nt]

        cmd += ["--num_tokens", str(nt)]

        print("\n" + "=" * 80)
        print(f"Running: num_tokens={nt}")
        print("Command:", " ".join(cmd))
        print("=" * 80)

        # Run and stream stdout/stderr directly to your terminal
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n[ERROR] num_tokens={nt} failed with return code {result.returncode}")
            # Stop immediately on failure (change to `continue` if you'd rather skip failures)
            sys.exit(result.returncode)

    print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
