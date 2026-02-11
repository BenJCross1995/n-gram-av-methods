read_jsonl <- function(path, flatten = TRUE, ...) {
  # Reads a newline-delimited JSON (.jsonl) file into a data.frame / tibble-like data.frame
  #
  # Args:
  #   path: file path to .jsonl
  #   flatten: whether to flatten nested objects (jsonlite feature)
  #   ...: passed to jsonlite::fromJSON (e.g., simplifyVector = TRUE)
  #
  # Returns:
  #   data.frame
  
  if (!file.exists(path)) stop("File does not exist: ", path)

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }

  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  lines <- lines[nzchar(trimws(lines))]  # drop empty lines

  if (length(lines) == 0) {
    return(data.frame())
  }

  objs <- lapply(lines, jsonlite::fromJSON, flatten = flatten, ...)
  df <- jsonlite::rbind_pages(objs)

  df
}
