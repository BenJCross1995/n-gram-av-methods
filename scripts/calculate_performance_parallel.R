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
  library(parallel)
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
# Require that a specific command-line argument is present.
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
# Split a comma-separated string into a character vector.
# ------------------------------------------------------------------------------
split_csv <- function(x) {
  trimws(unlist(strsplit(x, ",")))
}

# ------------------------------------------------------------------------------
# parse_bool
#
# Parse common boolean string representations into TRUE/FALSE.
# ------------------------------------------------------------------------------
parse_bool <- function(x, default = FALSE) {
  if (is.null(x)) {
    return(default)
  }

  if (is.logical(x)) {
    return(x)
  }

  x <- trimws(tolower(as.character(x)))

  if (x %in% c("true", "t", "1", "yes", "y")) {
    return(TRUE)
  }

  if (x %in% c("false", "f", "0", "no", "n")) {
    return(FALSE)
  }

  stop("Could not parse boolean value: ", x, call. = FALSE)
}

# ------------------------------------------------------------------------------
# get_n_workers
#
# Determine the number of worker processes to use.
#
# Priority:
#   1. Explicit --n_workers argument
#   2. SLURM_CPUS_PER_TASK
#   3. Fallback to 1
# ------------------------------------------------------------------------------
get_n_workers <- function(args = NULL) {
  if (!is.null(args) && !is.null(args$n_workers)) {
    n <- suppressWarnings(as.integer(args$n_workers))
    if (!is.na(n) && n >= 1) {
      return(n)
    }
  }

  n <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "1")))
  if (is.na(n) || n < 1) {
    n <- 1L
  }

  n
}

# ------------------------------------------------------------------------------
# save_df_outputs
#
# Save a data frame to .xlsx and optionally also to .rds.
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
# make_group_key
#
# Build a stable string key from one or more grouping columns.
# ------------------------------------------------------------------------------
make_group_key <- function(df, cols) {
  if (length(cols) == 0) {
    stop("cols must contain at least one column", call. = FALSE)
  }

  parts <- lapply(df[cols], function(x) {
    x <- as.character(x)
    x[is.na(x)] <- "<NA>"
    x
  })

  if (length(parts) == 1) {
    return(parts[[1]])
  }

  do.call(paste, c(parts, sep = "\r"))
}

# ------------------------------------------------------------------------------
# run_parallel
#
# Apply a function over X using mclapply on Unix-like systems if n_workers > 1,
# otherwise fall back to lapply.
# ------------------------------------------------------------------------------
run_parallel <- function(X, FUN, n_workers = 1L) {
  n_workers <- as.integer(n_workers)
  if (is.na(n_workers) || n_workers < 1L) {
    n_workers <- 1L
  }

  if (.Platform$OS.type == "unix" && n_workers > 1L && length(X) > 1L) {
    parallel::mclapply(
      X = X,
      FUN = FUN,
      mc.cores = min(n_workers, length(X)),
      mc.preschedule = TRUE
    )
  } else {
    lapply(X, FUN)
  }
}

# ------------------------------------------------------------------------------
# score_one_uncalibrated_group
#
# Score one uncalibrated group.
# ------------------------------------------------------------------------------
score_one_uncalibrated_group <- function(group_df, group_row, score_col, by, progress) {
  perf <- tryCatch(
    idiolect::performance(
      training = group_df %>% dplyr::select(score, target),
      by = by,
      progress = progress
    ),
    error = function(e) {
      message(
        "Skipping uncalibrated group: ",
        paste(names(group_row), as.character(group_row[1, ]), sep = "=", collapse = ", "),
        ", scoring_col=", score_col,
        " | error: ", e$message
      )
      NULL
    }
  )

  if (is.null(perf) || is.null(perf$evaluation) || nrow(perf$evaluation) == 0) {
    return(tibble::tibble())
  }

  dplyr::bind_cols(
    tibble::as_tibble(group_row),
    tibble::tibble(scoring_col = score_col),
    tibble::as_tibble(perf$evaluation)
  )
}

# ------------------------------------------------------------------------------
# score_one_calibrated_group
#
# Score one calibrated train/test group pair.
# ------------------------------------------------------------------------------
score_one_calibrated_group <- function(train_group, test_group, group_row, score_col, progress) {
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

  if (is.null(perf) || is.null(perf$evaluation) || nrow(perf$evaluation) == 0) {
    return(tibble::tibble())
  }

  dplyr::bind_cols(
    tibble::as_tibble(group_row),
    tibble::tibble(scoring_col = score_col),
    tibble::as_tibble(perf$evaluation)
  )
}

# ------------------------------------------------------------------------------
# score_grouped_performance
#
# Compute uncalibrated performance metrics within groups.
#
# Parallelisation:
#   - splits the input once by grouping columns
#   - evaluates groups in parallel
# ------------------------------------------------------------------------------
score_grouped_performance <- function(
  completed_df,
  grouping_cols,
  scoring_cols,
  by = "case",
  progress = FALSE,
  n_workers = 1L
) {
  purrr::map_dfr(scoring_cols, function(score_col) {

    message("Running uncalibrated scoring for: ", score_col)

    df_sub <- completed_df %>%
      dplyr::select(dplyr::all_of(c(grouping_cols, "target", score_col))) %>%
      dplyr::filter(!is.na(.data[[score_col]]), !is.na(target)) %>%
      dplyr::rename(score = !!score_col)

    if (nrow(df_sub) == 0) {
      message("No usable rows found for uncalibrated scoring_col=", score_col)
      return(tibble::tibble())
    }

    group_rows <- df_sub %>%
      dplyr::distinct(dplyr::across(dplyr::all_of(grouping_cols)))

    if (nrow(group_rows) == 0) {
      return(tibble::tibble())
    }

    df_sub$..group_key <- make_group_key(df_sub, grouping_cols)
    group_rows$..group_key <- make_group_key(group_rows, grouping_cols)

    split_groups <- split(df_sub, df_sub$..group_key)

    results <- run_parallel(
      X = seq_len(nrow(group_rows)),
      n_workers = n_workers,
      FUN = function(i) {
        group_row <- group_rows[i, , drop = FALSE]
        key <- group_row$..group_key[[1]]

        group_df <- split_groups[[key]]
        if (is.null(group_df) || nrow(group_df) == 0) {
          return(tibble::tibble())
        }

        group_df <- group_df %>% dplyr::select(-..group_key)
        group_row <- group_row %>% dplyr::select(-..group_key)

        score_one_uncalibrated_group(
          group_df = group_df,
          group_row = group_row,
          score_col = score_col,
          by = by,
          progress = progress
        )
      }
    )

    dplyr::bind_rows(results)
  })
}

# ------------------------------------------------------------------------------
# score_grouped_calibrated_performance
#
# Compute calibrated performance metrics by matching training/test groups.
#
# Parallelisation:
#   - split training and test once by match_cols
#   - evaluate matched groups in parallel
# ------------------------------------------------------------------------------
score_grouped_calibrated_performance <- function(
  completed_df,
  grouping_cols,
  scoring_cols,
  progress = FALSE,
  n_workers = 1L
) {
  if (!"data_type" %in% names(completed_df)) {
    stop("completed_df must contain a 'data_type' column", call. = FALSE)
  }

  if (!all(c("training", "test") %in% unique(completed_df$data_type))) {
    stop("completed_df$data_type must contain both 'training' and 'test'", call. = FALSE)
  }

  match_cols <- setdiff(grouping_cols, "data_type")

  if (length(match_cols) == 0) {
    stop("After excluding 'data_type', no grouping columns remain for matching", call. = FALSE)
  }

  training_df <- completed_df %>%
    dplyr::filter(data_type == "training")

  test_df <- completed_df %>%
    dplyr::filter(data_type == "test")

  matched_groups <- training_df %>%
    dplyr::distinct(dplyr::across(dplyr::all_of(match_cols))) %>%
    dplyr::inner_join(
      test_df %>% dplyr::distinct(dplyr::across(dplyr::all_of(match_cols))),
      by = match_cols
    )

  message("Number of matched groups for calibration: ", nrow(matched_groups))

  if (nrow(matched_groups) == 0) {
    return(tibble::tibble())
  }

  purrr::map_dfr(scoring_cols, function(score_col) {

    message("Running calibrated scoring for: ", score_col)

    training_sub <- training_df %>%
      dplyr::select(dplyr::all_of(c(match_cols, "target", score_col))) %>%
      dplyr::filter(!is.na(.data[[score_col]]), !is.na(target)) %>%
      dplyr::rename(score = !!score_col)

    test_sub <- test_df %>%
      dplyr::select(dplyr::all_of(c(match_cols, "target", score_col))) %>%
      dplyr::filter(!is.na(.data[[score_col]]), !is.na(target)) %>%
      dplyr::rename(score = !!score_col)

    if (nrow(training_sub) == 0 || nrow(test_sub) == 0) {
      message("No usable training/test rows found for calibrated scoring_col=", score_col)
      return(tibble::tibble())
    }

    training_sub$..group_key <- make_group_key(training_sub, match_cols)
    test_sub$..group_key <- make_group_key(test_sub, match_cols)
    matched_groups$..group_key <- make_group_key(matched_groups, match_cols)

    train_split <- split(training_sub, training_sub$..group_key)
    test_split <- split(test_sub, test_sub$..group_key)

    available_keys <- intersect(intersect(names(train_split), names(test_split)), matched_groups$..group_key)

    matched_groups2 <- matched_groups %>%
      dplyr::filter(..group_key %in% available_keys)

    if (nrow(matched_groups2) == 0) {
      return(tibble::tibble())
    }

    results <- run_parallel(
      X = seq_len(nrow(matched_groups2)),
      n_workers = n_workers,
      FUN = function(i) {
        group_row <- matched_groups2[i, , drop = FALSE]
        key <- group_row$..group_key[[1]]

        train_group <- train_split[[key]]
        test_group <- test_split[[key]]

        if (is.null(train_group) || is.null(test_group) ||
            nrow(train_group) == 0 || nrow(test_group) == 0) {
          return(tibble::tibble())
        }

        train_group <- train_group %>% dplyr::select(-..group_key)
        test_group <- test_group %>% dplyr::select(-..group_key)
        group_row <- group_row %>% dplyr::select(-..group_key)

        score_one_calibrated_group(
          train_group = train_group,
          test_group = test_group,
          group_row = group_row,
          score_col = score_col,
          progress = progress
        )
      }
    )

    dplyr::bind_rows(results)
  })
}

# ------------------------------------------------------------------------------
# main
#
# Expected command-line arguments:
#   --input_loc
#   --save_loc_uncalibrated
#   --save_loc_calibrated
#   --grouping_cols
#   --scoring_cols
#   --by
#   --progress
#   --save_rds
#   --calculate_uncalibrated
#   --n_workers
# ------------------------------------------------------------------------------
main <- function() {
  args <- parse_args()

  input_loc <- require_arg(args, "input_loc")
  save_loc_calibrated <- require_arg(args, "save_loc_calibrated")

  calculate_uncalibrated <- if (!is.null(args$calculate_uncalibrated)) {
    parse_bool(args$calculate_uncalibrated)
  } else {
    TRUE
  }

  save_loc_uncalibrated <- if (isTRUE(calculate_uncalibrated)) {
    require_arg(args, "save_loc_uncalibrated")
  } else {
    args$save_loc_uncalibrated
  }

  grouping_cols <- if (!is.null(args$grouping_cols)) {
    split_csv(args$grouping_cols)
  } else {
    c(
      "data_type",
      "corpus",
      "scoring_model",
      "max_context_tokens",
      "min_token_size"
    )
  }

  scoring_cols <- if (!is.null(args$scoring_cols)) {
    split_csv(args$scoring_cols)
  } else {
    c("unknown_sum_log_probs")
  }

  by <- if (!is.null(args$by)) args$by else "case"
  progress <- if (!is.null(args$progress)) parse_bool(args$progress) else FALSE
  save_rds <- if (!is.null(args$save_rds)) parse_bool(args$save_rds) else FALSE
  n_workers <- get_n_workers(args)

  message("Reading input file: ", input_loc)
  completed_df <- readRDS(input_loc)

  message("Input shape: ", paste(dim(completed_df), collapse = " x "))
  message("Grouping columns: ", paste(grouping_cols, collapse = ", "))
  message("Scoring columns: ", paste(scoring_cols, collapse = ", "))
  message("by = ", by)
  message("progress = ", progress)
  message("save_rds = ", save_rds)
  message("calculate_uncalibrated = ", calculate_uncalibrated)
  message("n_workers = ", n_workers)

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

  if (isTRUE(calculate_uncalibrated)) {
    message("Starting uncalibrated performance...")

    performance_df <- score_grouped_performance(
      completed_df = completed_df,
      grouping_cols = grouping_cols,
      scoring_cols = scoring_cols,
      by = by,
      progress = progress,
      n_workers = n_workers
    )

    message("Uncalibrated output shape: ", paste(dim(performance_df), collapse = " x "))
    message("Saving uncalibrated output to: ", save_loc_uncalibrated)

    save_df_outputs(
      df = performance_df,
      save_loc = save_loc_uncalibrated,
      save_rds = save_rds
    )
  } else {
    message("Skipping uncalibrated performance because calculate_uncalibrated = FALSE")
  }

  message("Starting calibrated performance...")

  calibrated_performance_df <- score_grouped_calibrated_performance(
    completed_df = completed_df,
    grouping_cols = grouping_cols,
    scoring_cols = scoring_cols,
    progress = progress,
    n_workers = n_workers
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