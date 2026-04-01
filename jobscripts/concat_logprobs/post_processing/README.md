# CSF3 post-processing pipeline

This bundle replaces the original serial GPU wrappers with a dependency-driven `himem` pipeline.

## What changed

- All jobs now use:
  - `#SBATCH -p himem`
  - `#SBATCH -t 7-0`
  - `#SBATCH -n 1`
  - `#SBATCH --mem=64G`
- `combine_files` step 1 runs as a Slurm array: one task per existing `data_type / corpus / model / raw_dir`.
- `combine_files` steps 2-5 run once centrally after the whole array succeeds.
- `create_like_for_like` step 1 runs as a Slurm array: one task per existing model-level combined file.
- `create_like_for_like` combine steps run once centrally after its array succeeds.
- performance scoring is discovered dynamically from `BASE_LOC/_results/combined*_raw.rds`, then submitted as one array task per input file.

## Files

- `csf3_postprocessing_config.sh` - shared config
- `combine_step1_array.sh` - raw-subdir combine worker
- `combine_postprocess.sh` - combine steps 2-5
- `lfl_step1_array.sh` - model-level like-for-like worker
- `lfl_postprocess.sh` - like-for-like combine steps
- `launch_performance_array.sh` - discovers current combined outputs and submits performance array
- `performance_array.sh` - one performance job per discovered input file
- `submit_postprocessing_pipeline.sh` - central submitter

## Usage

Copy these scripts to a directory on CSF3, then run:

```bash
bash submit_postprocessing_pipeline.sh
```

## Notes

- Edit arrays like `MODEL_LIST`, `CORPORA`, and `DATA_TYPES` in `csf3_postprocessing_config.sh`.
- Edit `*_MAX_CONCURRENT` if you want to be more or less aggressive with parallelism.
- The performance stage is dynamic at launch time, so if you add another `combined*_raw.rds` output before that stage begins, it will automatically get its own array task.
- For unknown future combined file names, the performance wrapper generates output names from the input stem using a generic `performance_<stem>_*` convention.
