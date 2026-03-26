#!/usr/bin/env Rscript

if (file.exists("renv/activate.R")) {
  source("renv/activate.R")
}

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(tibble)
  library(writexl)
  library(idiolect)
})

# ------------------------------------------------------------------------------
# parse_args
#
# Parse command-line arguments passed in the form:
#   --arg_name value
#
# Returns:
#   A named list of argument values.
# ------------------------------------------------------------------------------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  arg_list <- list()

  i <- 1
  while (i <= length(args)) {
    key <- args[i]

    if (startsWith(key, "--")) {
      key <- sub("^--", "", key)

      # Handle flags with no explicit value
      if (i == length(args) || startsWith(args[i + 1], "--")) {
        arg_list[[key]] <- TRUE
        i <- i + 1
      } else {
        arg_list[[key]] <- args[i + 1]
        i <- i + 2
      }
    } else {
      i <- i + 1
    }
  }

  arg_list
}

# ------------------------------------------------------------------------------
# require_arg
#
# Helper to require that a specific command-line argument is present.
#
# Args:
#   args: Named list of parsed arguments.
#   name: Name of the argument to retrieve.
#
# Returns:
#   The requested argument value.
#
# Throws:
#   An error if the argument is missing.
# ------------------------------------------------------------------------------
require_arg <- function(args, name) {
  if (is.null(args[[name]])) {
    stop(sprintf("Missing required argument: --%s", name), call. = FALSE)
  }
  args[[name]]
}

# ------------------------------------------------------------------------------
# split_csv
#
# Split a comma-separated string into a character vector, trimming whitespace.
#
# Args:
#   x: A single comma-separated string.
#
# Returns:
#   A character vector.
# ------------------------------------------------------------------------------
split_csv <- function(x) {
  trimws(unlist(strsplit(x, ",")))
}

# ------------------------------------------------------------------------------
# save_df_outputs
#
# Save a data frame to .xlsx and, optionally, also save an .rds file with the
# same base filename in the same location.
#
# Args:
#   df: Data frame to save.
#   save_loc: Output .xlsx file path.
#   save_rds: Logical; if TRUE, also save an .rds version alongside the .xlsx.
#
# Returns:
#   Invisible NULL.
# ------------------------------------------------------------------------------
save_df_outputs <- function(df, save_loc, save_rds = FALSE) {
  dir.create(dirname(save_loc), recursive = TRUE, showWarnings = FALSE)

  writexl::write_xlsx(df, path = save_loc)

  if (isTRUE(save_rds)) {
    rds_loc <- sub("\\.xlsx$", ".rds", save_loc, ignore.case = TRUE)

    if (identical(rds_loc, save_loc)) {
      rds_loc <- paste0(save_loc, ".rds")
    }

    saveRDS(df, file = rds_loc)
  }

  invisible(NULL)
}

# ------------------------------------------------------------------------------
# score_grouped_performance
#
# Compute uncalibrated performance metrics within groups for one or more score
# columns using idiolect::performance().
#
# For each scoring column:
#   1. Rename that scoring column to "score"
#   2. Group by grouping_cols
#   3. Run idiolect::performance(training = ...)
#   4. Extract perf$evaluation
#   5. Bind the grouping columns and the scoring column name back onto the result
#
# Any group that fails during scoring is skipped entirely and does not appear in
# the final output.
#
# Args:
#   completed_df: Input data frame containing grouping columns, target, and score
#                 columns.
#   grouping_cols: Character vector of grouping column names.
#   scoring_cols: Character vector of scoring column names to evaluate.
#   by: Value passed to idiolect::performance(by = ...). Default is "case".
#   progress: Logical passed to idiolect::performance(progress = ...).
#
# Returns:
#   A data frame of uncalibrated performance metrics.
# ------------------------------------------------------------------------------
score_grouped_performance <- function(
  completed_df,
  grouping_cols,
  scoring_cols,
  by = "case",
  progress = FALSE
) {
  purrr::map_dfr(scoring_cols, function(score_col) {

    message("Running uncalibrated scoring for: ", score_col)

    completed_df %>%
      # Keep only the columns needed for this scoring run
      dplyr::select(dplyr::all_of(c(grouping_cols, "target", score_col))) %>%
      # Remove rows with missing target or score
      dplyr::filter(!is.na(.data[[score_col]]), !is.na(target)) %>%
      # Rename the active scoring column to "score" for idiolect::performance()
      dplyr::rename(score = !!score_col) %>%
      # Group by the requested grouping columns
      dplyr::group_by(dplyr::across(dplyr::all_of(grouping_cols))) %>%
      # Compute performance within each group
      dplyr::group_modify(function(.x, .y) {

        perf <- tryCatch(
          idiolect::performance(
            training = .x %>% dplyr::select(score, target),
            by = by,
            progress = progress
          ),
          error = function(e) {
            message(
              "Skipping uncalibrated group: ",
              paste(names(.y), as.character(.y[1, ]), sep = "=", collapse = ", "),
              ", scoring_col=", score_col,
              " | error: ", e$message
            )
            NULL
          }
        )

        # If performance failed or returned no evaluation, skip this group
        if (is.null(perf) || is.null(perf$evaluation) || nrow(perf$evaluation) == 0) {
          return(tibble::tibble())
        }

        # group_modify automatically re-attaches grouping columns, so only return
        # the new columns here
        dplyr::bind_cols(
          tibble::tibble(scoring_col = score_col),
          tibble::as_tibble(perf$evaluation)
        )
      }) %>%
      dplyr::ungroup()
  })
}

# ------------------------------------------------------------------------------
# score_grouped_calibrated_performance
#
# Compute calibrated performance metrics by splitting the data into training and
# test sets using the data_type column.
#
# Workflow:
#   1. Split completed_df into training and test where data_type is
#      "training" or "test"
#   2. Remove data_type from the matching keys
#   3. Keep only groups that exist in both training and test
#   4. For each scoring column:
#        - rename that column to "score" in both datasets
#        - loop over matched groups
#        - run idiolect::performance(training = ..., test = ...)
#        - extract perf$evaluation
#        - bind grouping info and scoring column name back on
#
# Any group that fails during scoring is skipped entirely and does not appear in
# the final output.
#
# Args:
#   completed_df: Input data frame containing data_type, grouping columns,
#                 target, and score columns.
#   grouping_cols: Character vector of grouping column names. Must include
#                  data_type if calibration is based on that field.
#   scoring_cols: Character vector of scoring column names to evaluate.
#   progress: Logical passed to idiolect::performance(progress = ...).
#
# Returns:
#   A data frame of calibrated performance metrics.
# ------------------------------------------------------------------------------
score_grouped_calibrated_performance <- function(
  completed_df,
  grouping_cols,
  scoring_cols,
  progress = FALSE
) {
  # Ensure data_type exists
  if (!"data_type" %in% names(completed_df)) {
    stop("completed_df must contain a 'data_type' column", call. = FALSE)
  }

  # Ensure both training and test are present
  if (!all(c("training", "test") %in% unique(completed_df$data_type))) {
    stop("completed_df$data_type must contain both 'training' and 'test'", call. = FALSE)
  }

  # Matching should be done on grouping columns excluding data_type
  match_cols <- setdiff(grouping_cols, "data_type")

  if (length(match_cols) == 0) {
    stop("After excluding 'data_type', no grouping columns remain for matching", call. = FALSE)
  }

  # Split the full data into training and test subsets
  training_df <- completed_df %>%
    dplyr::filter(data_type == "training")

  test_df <- completed_df %>%
    dplyr::filter(data_type == "test")

  # Determine which groups exist in BOTH training and test
  matched_groups <- training_df %>%
    dplyr::distinct(dplyr::across(dplyr::all_of(match_cols))) %>%
    dplyr::inner_join(
      test_df %>% dplyr::distinct(dplyr::across(dplyr::all_of(match_cols))),
      by = match_cols
    )

  message("Number of matched groups for calibration: ", nrow(matched_groups))

  purrr::map_dfr(scoring_cols, function(score_col) {

    message("Running calibrated scoring for: ", score_col)

    # Prepare training subset for this score column
    training_sub <- training_df %>%
      dplyr::inner_join(matched_groups, by = match_cols) %>%
      dplyr::select(dplyr::all_of(c(match_cols, "target", score_col))) %>%
      dplyr::filter(!is.na(.data[[score_col]]), !is.na(target)) %>%
      dplyr::rename(score = !!score_col)

    # Prepare test subset for this score column
    test_sub <- test_df %>%
      dplyr::inner_join(matched_groups, by = match_cols) %>%
      dplyr::select(dplyr::all_of(c(match_cols, "target", score_col))) %>%
      dplyr::filter(!is.na(.data[[score_col]]), !is.na(target)) %>%
      dplyr::rename(score = !!score_col)

    # Turn each row of matched_groups into a separate group definition
    group_list <- matched_groups %>%
      split(seq_len(nrow(.)))

    purrr::map_dfr(group_list, function(group_row) {

      # Pull the relevant rows for this matched group from training and test
      train_group <- training_sub %>%
        dplyr::semi_join(group_row, by = match_cols)

      test_group <- test_sub %>%
        dplyr::semi_join(group_row, by = match_cols)

      perf <- tryCatch(
        idiolect::performance(
          training = train_group %>% dplyr::select(score, target),
          test = test_group %>% dplyr::select(score, target),
          progress = progress
        ),
        error = function(e) {
          message(
            "Skipping calibrated group: ",
            paste(names(group_row), as.character(group_row[1, ]), sep = "=", collapse = ", "),
            ", scoring_col=", score_col,
            " | error: ", e$message
          )
          NULL
        }
      )

      # If performance failed or returned no evaluation, skip this group
      if (is.null(perf) || is.null(perf$evaluation) || nrow(perf$evaluation) == 0) {
        return(tibble::tibble())
      }

      # Bind group info + score column name + evaluation metrics
      dplyr::bind_cols(
        tibble::as_tibble(group_row),
        tibble::tibble(scoring_col = score_col),
        tibble::as_tibble(perf$evaluation)
      )
    })
  })
}

# ------------------------------------------------------------------------------
# main
#
# Main entry point for the script.
#
# Expected command-line arguments:
#   --input_loc                Path to input .rds file
#   --save_loc_uncalibrated    Path to save uncalibrated .xlsx output
#   --save_loc_calibrated      Path to save calibrated .xlsx output
#   --grouping_cols            Comma-separated grouping columns
#   --scoring_cols             Comma-separated scoring columns
#   --by                       Optional; passed to uncalibrated performance()
#   --progress                 Optional TRUE/FALSE
#   --save_rds                 Optional TRUE/FALSE; if TRUE, also save .rds
#                              files alongside the .xlsx outputs
#
# Returns:
#   Invisible NULL.
# ------------------------------------------------------------------------------
main <- function() {
  args <- parse_args()

  # Required file arguments
  input_loc <- require_arg(args, "input_loc")
  save_loc_uncalibrated <- require_arg(args, "save_loc_uncalibrated")
  save_loc_calibrated <- require_arg(args, "save_loc_calibrated")

  # Optional grouping/scoring arguments with defaults
  grouping_cols <- if (!is.null(args$grouping_cols)) {
    split_csv(args$grouping_cols)
  } else {
    c(
      "data_type",
      "corpus",
      "scoring_model",
      "min_token_size",
      "weight",
      "alpha",
      "base"
    )
  }

  scoring_cols <- if (!is.null(args$scoring_cols)) {
    split_csv(args$scoring_cols)
  } else {
    c("simpson_score", "jaccard_score")
  }

  # Optional idiolect arguments
  by <- if (!is.null(args$by)) args$by else "case"
  progress <- if (!is.null(args$progress)) as.logical(args$progress) else FALSE
  save_rds <- if (!is.null(args$save_rds)) as.logical(args$save_rds) else FALSE

  # Read input
  message("Reading input file: ", input_loc)
  completed_df <- readRDS(input_loc)

  # Print useful diagnostics for SLURM logs
  message("Input shape: ", paste(dim(completed_df), collapse = " x "))
  message("Grouping columns: ", paste(grouping_cols, collapse = ", "))
  message("Scoring columns: ", paste(scoring_cols, collapse = ", "))
  message("by = ", by)
  message("progress = ", progress)
  message("save_rds = ", save_rds)

  # Validate required columns exist
  missing_group_cols <- setdiff(grouping_cols, names(completed_df))
  missing_score_cols <- setdiff(c("target", scoring_cols), names(completed_df))

  if (length(missing_group_cols) > 0) {
    stop(
      "Missing grouping columns in completed_df: ",
      paste(missing_group_cols, collapse = ", "),
      call. = FALSE
    )
  }

  if (length(missing_score_cols) > 0) {
    stop(
      "Missing required columns in completed_df: ",
      paste(missing_score_cols, collapse = ", "),
      call. = FALSE
    )
  }

  # --------------------------------------------------------------------------
  # Stage 1: Uncalibrated performance
  # --------------------------------------------------------------------------
  message("Starting uncalibrated performance...")

  performance_df <- score_grouped_performance(
    completed_df = completed_df,
    grouping_cols = grouping_cols,
    scoring_cols = scoring_cols,
    by = by,
    progress = progress
  )

  message("Uncalibrated output shape: ", paste(dim(performance_df), collapse = " x "))
  message("Saving uncalibrated output to: ", save_loc_uncalibrated)

  save_df_outputs(
    df = performance_df,
    save_loc = save_loc_uncalibrated,
    save_rds = save_rds
  )

  # --------------------------------------------------------------------------
  # Stage 2: Calibrated performance
  # --------------------------------------------------------------------------
  message("Starting calibrated performance...")

  calibrated_performance_df <- score_grouped_calibrated_performance(
    completed_df = completed_df,
    grouping_cols = grouping_cols,
    scoring_cols = scoring_cols,
    progress = progress
  )

  message("Calibrated output shape: ", paste(dim(calibrated_performance_df), collapse = " x "))
  message("Saving calibrated output to: ", save_loc_calibrated)

  save_df_outputs(
    df = calibrated_performance_df,
    save_loc = save_loc_calibrated,
    save_rds = save_rds
  )

  message("Done.")
}

main()