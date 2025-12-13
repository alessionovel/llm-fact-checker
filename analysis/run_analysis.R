options(repos = c(CRAN = "https://cloud.r-project.org"))

# Robust CSV reader that stitches lines when stray newlines break rows
count_csv_fields <- function(line) {
  parts <- strsplit(line, ",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", perl = TRUE)[[1]]
  length(parts)
}

read_csv_repaired <- function(path) {
  raw_lines <- readLines(path, warn = FALSE)
  if (!length(raw_lines)) stop("File is empty: ", path)
  expected_fields <- count_csv_fields(raw_lines[1])
  if (expected_fields < 2) stop("Header appears invalid in: ", path)
  rows <- character(0)
  buffer <- character(0)
  fixed_rows <- 0L

  flush_buffer <- function() {
    if (!length(buffer)) return()
    if (length(buffer) > 1) fixed_rows <<- fixed_rows + 1L
    rows <<- c(rows, paste(buffer, collapse = "\n"))
    buffer <<- character(0)
  }

  for (ln in raw_lines[-1]) {
    buffer <- c(buffer, ln)
    row_text <- paste(buffer, collapse = "\n")
    field_count <- count_csv_fields(row_text)
    if (field_count >= expected_fields) flush_buffer()
  }
  flush_buffer()

  repaired_text <- paste(c(raw_lines[1], rows), collapse = "\n")
  con <- textConnection(repaired_text)
  on.exit(close(con), add = TRUE)
  df <- read.csv(con, stringsAsFactors = FALSE)
  if (fixed_rows > 0) message("Repaired ", fixed_rows, " broken CSV row(s) caused by stray newlines.")
  return(df)
}

# ==========================================
# 1. DATA LOADING & PREPROCESSING (GLOBAL)
# ==========================================
{
  args <- commandArgs(trailingOnly = TRUE)
  script_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
  candidate_files <- if (length(args) > 0) normalizePath(args, mustWork = FALSE) else file.path(script_dir, c("results.rds", "results.csv", "example.csv"))
  loaded <- FALSE
  input_name <- "results" # Default fallback
  
  for (f in candidate_files) {
    if (!file.exists(f)) next
    input_name <- tools::file_path_sans_ext(basename(f))
    
    if (grepl("\\.rds$", f, ignore.case = TRUE)) {
      results <- readRDS(f)
    } else {
      results <- read_csv_repaired(f)
    }
    loaded <- TRUE
    message("Loaded data from: ", f)
    break
  }
  if (!loaded) stop("No data found. Please place results.csv or results.rds next to this script.")
}

# Standardize Column Names
names(results) <- gsub("\\.", "_", names(results))
names(results) <- gsub("-", "_", names(results)) 
names(results) <- tolower(names(results)) 

# Create Output Directory (map results-xxx input to report-xxx output)
base_name <- sub("^results[-_]", "", input_name)
if (base_name == "") base_name <- input_name
output_dir <- paste0("report-", base_name)
if (!dir.exists(output_dir)) dir.create(output_dir)

# Helper: Normalize Truth/Verdict to Strict TRUE/FALSE
normalize_bool <- function(vec) {
  v <- toupper(trimws(as.character(vec)))
  v[v %in% c("YES", "TRUE", "1")] <- "TRUE"
  v[v %in% c("NO", "FALSE", "0")] <- "FALSE"
  return(v)
}

# Normalize Ground Truth immediately (used in Accuracy)
if("ground_truth" %in% names(results)) {
  results$ground_truth <- normalize_bool(results$ground_truth)
} else {
  warning("Column 'ground_truth' not found. Accuracy metrics will fail.")
}

# Normalize Statement Type (lowercase) to ensure consistency
if("type" %in% names(results)) {
  results$type <- tolower(trimws(results$type))
}

# ==========================================
# 2. ACCURACY ANALYSIS (Binary correctness)
# ==========================================
message("--- Running Accuracy Analysis ---")

# Define columns to check
verdict_cols <- c(
  "verdict_prompt1_initial", 
  "verdict_prompt2_initial", 
  "verdict_prompt1_reconsidered", 
  "verdict_prompt2_reconsidered"
)

# Check accuracy (Binary 1/0)
check_accuracy <- function(truth, verdict) {
  v_norm <- normalize_bool(verdict)
  
  # Logic:
  # 1. If verdict is "INSUFFICIENT INFO", return NA (Abstained)
  # 2. Else, if verdict matches truth, return 1 (Correct)
  # 3. Else, return 0 (Incorrect)
  
  return(ifelse(v_norm == "INSUFFICIENT INFO", NA, 
         ifelse(truth == v_norm, 1, 0)))
}

# Generate Accuracy Columns
results$acc_prompt1_initial <- check_accuracy(results$ground_truth, results$verdict_prompt1_initial)
results$acc_prompt2_initial <- check_accuracy(results$ground_truth, results$verdict_prompt2_initial)
results$acc_prompt1_reconsidered <- check_accuracy(results$ground_truth, results$verdict_prompt1_reconsidered)
results$acc_prompt2_reconsidered <- check_accuracy(results$ground_truth, results$verdict_prompt2_reconsidered)

# -- Report Generation: Stats --
calc_acc_stats <- function(col_data, label) {
  # Count abstentions (NAs)
  abstained_count <- sum(is.na(col_data))
  
  # Calculate accuracy only on valid answers (removing NAs)
  mean_acc <- mean(col_data, na.rm = TRUE)
  correct_count <- sum(col_data == 1, na.rm = TRUE)
  
  # Total valid attempts (excluding abstentions)
  valid_total <- sum(!is.na(col_data))
  
  return(data.frame(
    Scenario = label,
    Accuracy_Pct = sprintf("%.2f%%", mean_acc * 100),
    Correct = correct_count,
    Abstained = abstained_count,     # <--- New Column
    Total_Attempted = valid_total,   # <--- Renamed for clarity
    stringsAsFactors = FALSE
  ))
}

acc_summary <- rbind(
  calc_acc_stats(results$acc_prompt1_initial,      "Prompt 1 (Initial)"),
  calc_acc_stats(results$acc_prompt1_reconsidered, "Prompt 1 (Reconsidered)"),
  calc_acc_stats(results$acc_prompt2_initial,      "Prompt 2 (Initial)"),
  calc_acc_stats(results$acc_prompt2_reconsidered, "Prompt 2 (Reconsidered)")
)

# Pooled stats
pool_total <- c(results$acc_prompt1_initial, results$acc_prompt1_reconsidered, results$acc_prompt2_initial, results$acc_prompt2_reconsidered)

acc_summary <- rbind(acc_summary,
  data.frame(Scenario = "---", Accuracy_Pct="", Correct=NA, Abstained=NA, Total_Attempted=NA),
  calc_acc_stats(pool_total, "Overall Total")
)

# -- Report Generation: McNemar --
run_mcnemar <- function(vec1, vec2, label) {
  tbl <- table(factor(vec1, levels=c(0,1)), factor(vec2, levels=c(0,1)))
  # Handle cases with zero variance to avoid crash
  if(all(dim(tbl) == c(2,2))) {
    test <- mcnemar.test(tbl)
    p_val <- test$p.value
    chi <- test$statistic
    df <- test$parameter
    # Clamp p-value to valid range [0, 1] to handle numerical precision issues
    p_val <- pmax(0, pmin(1, p_val))
  } else {
    p_val <- NA; chi <- NA; df <- NA
  }
  
  diff_pct <- (mean(vec1, na.rm=TRUE) - mean(vec2, na.rm=TRUE)) * 100
  
  return(data.frame(
    Comparison = label,
    Diff_Pct = sprintf("%+.2f%%", diff_pct),
    Chi_Sq = ifelse(is.na(chi), "NA", sprintf("%.3f", chi)),
    P_Value = ifelse(is.na(p_val), "NA", sprintf("%.4g", p_val)),
    Sig = ifelse(!is.na(p_val) & p_val < 0.05, "Yes (*)", "No"),
    stringsAsFactors = FALSE
  ))
}

mcnemar_table <- rbind(
  run_mcnemar(results$acc_prompt1_initial, results$acc_prompt2_initial, "P1 Initial vs P2 Initial"),
  run_mcnemar(results$acc_prompt1_reconsidered, results$acc_prompt2_reconsidered, "P1 Recons. vs P2 Recons."),
  run_mcnemar(results$acc_prompt1_reconsidered, results$acc_prompt1_initial, "P1 Recons. vs P1 Initial"),
  run_mcnemar(results$acc_prompt2_reconsidered, results$acc_prompt2_initial, "P2 Recons. vs P2 Initial")
)

# Write Accuracy Report
acc_file <- file.path(output_dir, "task1-accuracy.csv")
write.table(acc_summary, file = acc_file, sep = ",", row.names = FALSE, col.names = TRUE, na = "")
cat("\n\nSTATISTICAL SIGNIFICANCE (McNemar Tests)\n", file = acc_file, append = TRUE)
suppressWarnings(write.table(mcnemar_table, file = acc_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE))
message("Saved accuracy report to: ", acc_file)

# ==========================================
# 3. CONSISTENCY ANALYSIS (Triplet logic checks)
# ==========================================
message("--- Running Consistency Analysis ---")

# Check triplet structure
if(nrow(results) %% 3 != 0) {
  warning("Data length not divisible by 3. Consistency grouping might be incorrect.")
}

# Create Triplet ID
results$triplet_id <- ceiling(seq_len(nrow(results)) / 3)

# Subset for reshaping
cols_to_keep <- c("triplet_id", "type", verdict_cols)
df_subset <- results[, cols_to_keep]
df_subset$type <- trimws(tolower(df_subset$type)) 

# Reshape to Wide (1 row per triplet)
df_wide <- reshape(
  df_subset,
  idvar = "triplet_id",
  timevar = "type",
  direction = "wide",
  sep = "_"
)

# Function to check logical consistency (opposite verdicts)
check_pair_consistency <- function(main_verdict, comp_verdict) {
  v1 <- normalize_bool(main_verdict)
  v2 <- normalize_bool(comp_verdict)
  
  # If both are INSUFFICIENT INFO, consider it consistent (both abstained coherently)
  both_insufficient <- (v1 == "INSUFFICIENT INFO" & v2 == "INSUFFICIENT INFO")
  
  # Logic: (Aff=T AND Comp=F) OR (Aff=F AND Comp=T) -> 1 (Consistent)
  consistent <- ifelse(
    both_insufficient,
    1,  # Both insufficient = consistent abstention
    ifelse(
      (v1 == "TRUE" & v2 == "FALSE") | (v1 == "FALSE" & v2 == "TRUE"),
      1, 
      0
    )
  )
  consistent[is.na(consistent)] <- 0  # If only one is insufficient = inconsistent
  return(consistent)
}

# Function to check if verdicts are the same (for negation vs antonym)
check_same_consistency <- function(verdict1, verdict2) {
  v1 <- normalize_bool(verdict1)
  v2 <- normalize_bool(verdict2)
  
  # If both are INSUFFICIENT INFO, consider it consistent (both abstained coherently)
  both_insufficient <- (v1 == "INSUFFICIENT INFO" & v2 == "INSUFFICIENT INFO")
  
  # Logic: (v1 == v2) -> 1 (Consistent), else 0
  consistent <- ifelse(
    both_insufficient,
    1,  # Both insufficient = consistent abstention
    ifelse(v1 == v2, 1, 0)
  )
  consistent[is.na(consistent)] <- 0  # If only one is insufficient = inconsistent
  return(consistent)
}

# Function to check triplet consistency (affirmation, negation, antonym)
# Negation and Antonym should be opposite of Affirmation, or all should be INSUFFICIENT INFO
check_triplet_consistency <- function(aff_verdict, neg_verdict, ant_verdict) {
  v_aff <- normalize_bool(aff_verdict)
  v_neg <- normalize_bool(neg_verdict)
  v_ant <- normalize_bool(ant_verdict)
  
  # If all three are INSUFFICIENT INFO, consistent
  all_insufficient <- (v_aff == "INSUFFICIENT INFO" & v_neg == "INSUFFICIENT INFO" & v_ant == "INSUFFICIENT INFO")
  
  # Check if any are insufficient (but not all)
  any_insufficient <- (v_aff == "INSUFFICIENT INFO" | v_neg == "INSUFFICIENT INFO" | v_ant == "INSUFFICIENT INFO")
  
  # Logic: Check if negation and antonym are opposite of affirmation
  # If Aff=T, then Neg and Ant should both be F
  # If Aff=F, then Neg and Ant should both be T
  opposite_check <- ifelse(
    v_aff == "TRUE",
    (v_neg == "FALSE" & v_ant == "FALSE"),
    ifelse(
      v_aff == "FALSE",
      (v_neg == "TRUE" & v_ant == "TRUE"),
      FALSE
    )
  )
  
  consistent <- ifelse(
    all_insufficient,
    1,  # All insufficient = consistent abstention
    ifelse(
      any_insufficient,
      0,  # Some insufficient but not all = inconsistent
      ifelse(opposite_check, 1, 0)
    )
  )
  consistent[is.na(consistent)] <- 0
  return(consistent)
}

metrics_df <- data.frame(triplet_id = df_wide$triplet_id)

# Loop through verdict columns to calculate consistency metrics
# Checking consistency between: (1) Affirmation vs Negation (opposite), (2) Affirmation vs Antonym (opposite), 
# (3) Negation vs Antonym (same), (4) Triplet (Aff, Neg, Ant should all be consistent)
for (v_col in verdict_cols) {
  col_aff <- paste0(v_col, "_affirmation")
  col_neg <- paste0(v_col, "_negation")
  col_ant <- paste0(v_col, "_antonym")
  
  # Consistency: Affirmation vs Negation (should be opposite)
  metric_name_an <- paste0("consist_", v_col, "_aff_neg")
  metrics_df[[metric_name_an]] <- check_pair_consistency(df_wide[[col_aff]], df_wide[[col_neg]])
  
  # Consistency: Affirmation vs Antonym (should be opposite)
  metric_name_aa <- paste0("consist_", v_col, "_aff_ant")
  metrics_df[[metric_name_aa]] <- check_pair_consistency(df_wide[[col_aff]], df_wide[[col_ant]])
  
  # Consistency: Negation vs Antonym (should be same)
  metric_name_na <- paste0("consist_", v_col, "_neg_ant")
  metrics_df[[metric_name_na]] <- check_same_consistency(df_wide[[col_neg]], df_wide[[col_ant]])
  
  # Consistency: Triplet (Aff, Neg, Ant all consistent)
  metric_name_triplet <- paste0("consist_", v_col, "_triplet")
  metrics_df[[metric_name_triplet]] <- check_triplet_consistency(df_wide[[col_aff]], df_wide[[col_neg]], df_wide[[col_ant]])
}

# -- Report Generation: Consistency --
calc_consist_stats <- function(col_data, label) {
  mean_val <- mean(col_data, na.rm = TRUE)
  count_correct <- sum(col_data == 1, na.rm = TRUE)
  total <- length(col_data)
  
  return(data.frame(
    Scenario = label,
    Consistency_Pct = sprintf("%.2f%%", mean_val * 100),
    Consistent_Triplets = count_correct,
    Total_Triplets = total,
    stringsAsFactors = FALSE
  ))
}

# Generate summary tables grouped by consistency type
aff_neg_table <- rbind(
  calc_consist_stats(metrics_df$consist_verdict_prompt1_initial_aff_neg, "Prompt 1 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt1_reconsidered_aff_neg, "Prompt 1 (Reconsidered)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_initial_aff_neg, "Prompt 2 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_reconsidered_aff_neg, "Prompt 2 (Reconsidered)")
)

# Add pooled consistency (Initial, Reconsidered, Overall)
aff_neg_table <- rbind(
  aff_neg_table,
  data.frame(Scenario = "---", Consistency_Pct = "", Consistent_Triplets = NA, Total_Triplets = NA, stringsAsFactors = FALSE),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_aff_neg, metrics_df$consist_verdict_prompt2_initial_aff_neg),
    "Initial (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_reconsidered_aff_neg, metrics_df$consist_verdict_prompt2_reconsidered_aff_neg),
    "Reconsidered (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_aff_neg,
      metrics_df$consist_verdict_prompt2_initial_aff_neg,
      metrics_df$consist_verdict_prompt1_reconsidered_aff_neg,
      metrics_df$consist_verdict_prompt2_reconsidered_aff_neg),
    "Overall Total"
  )
)

aff_ant_table <- rbind(
  calc_consist_stats(metrics_df$consist_verdict_prompt1_initial_aff_ant, "Prompt 1 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt1_reconsidered_aff_ant, "Prompt 1 (Reconsidered)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_initial_aff_ant, "Prompt 2 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_reconsidered_aff_ant, "Prompt 2 (Reconsidered)")
)

# Add pooled consistency (Initial, Reconsidered, Overall)
aff_ant_table <- rbind(
  aff_ant_table,
  data.frame(Scenario = "---", Consistency_Pct = "", Consistent_Triplets = NA, Total_Triplets = NA, stringsAsFactors = FALSE),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_aff_ant, metrics_df$consist_verdict_prompt2_initial_aff_ant),
    "Initial (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_reconsidered_aff_ant, metrics_df$consist_verdict_prompt2_reconsidered_aff_ant),
    "Reconsidered (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_aff_ant,
      metrics_df$consist_verdict_prompt2_initial_aff_ant,
      metrics_df$consist_verdict_prompt1_reconsidered_aff_ant,
      metrics_df$consist_verdict_prompt2_reconsidered_aff_ant),
    "Overall Total"
  )
)

neg_ant_table <- rbind(
  calc_consist_stats(metrics_df$consist_verdict_prompt1_initial_neg_ant, "Prompt 1 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt1_reconsidered_neg_ant, "Prompt 1 (Reconsidered)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_initial_neg_ant, "Prompt 2 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_reconsidered_neg_ant, "Prompt 2 (Reconsidered)")
)

# Add pooled consistency (Initial, Reconsidered, Overall)
neg_ant_table <- rbind(
  neg_ant_table,
  data.frame(Scenario = "---", Consistency_Pct = "", Consistent_Triplets = NA, Total_Triplets = NA, stringsAsFactors = FALSE),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_neg_ant, metrics_df$consist_verdict_prompt2_initial_neg_ant),
    "Initial (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_reconsidered_neg_ant, metrics_df$consist_verdict_prompt2_reconsidered_neg_ant),
    "Reconsidered (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_neg_ant,
      metrics_df$consist_verdict_prompt2_initial_neg_ant,
      metrics_df$consist_verdict_prompt1_reconsidered_neg_ant,
      metrics_df$consist_verdict_prompt2_reconsidered_neg_ant),
    "Overall Total"
  )
)

triplet_table <- rbind(
  calc_consist_stats(metrics_df$consist_verdict_prompt1_initial_triplet, "Prompt 1 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt1_reconsidered_triplet, "Prompt 1 (Reconsidered)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_initial_triplet, "Prompt 2 (Initial)"),
  calc_consist_stats(metrics_df$consist_verdict_prompt2_reconsidered_triplet, "Prompt 2 (Reconsidered)")
)

# Add pooled consistency (Initial, Reconsidered, Overall)
triplet_table <- rbind(
  triplet_table,
  data.frame(Scenario = "---", Consistency_Pct = "", Consistent_Triplets = NA, Total_Triplets = NA, stringsAsFactors = FALSE),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_triplet, metrics_df$consist_verdict_prompt2_initial_triplet),
    "Initial (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_reconsidered_triplet, metrics_df$consist_verdict_prompt2_reconsidered_triplet),
    "Reconsidered (P1+P2)"
  ),
  calc_consist_stats(
    c(metrics_df$consist_verdict_prompt1_initial_triplet,
      metrics_df$consist_verdict_prompt2_initial_triplet,
      metrics_df$consist_verdict_prompt1_reconsidered_triplet,
      metrics_df$consist_verdict_prompt2_reconsidered_triplet),
    "Overall Total"
  )
)

# -- Report Generation: Statistical Significance (McNemar Tests) --
run_mcnemar_consist <- function(vec1, vec2, label) {
  tbl <- table(factor(vec1, levels=c(0,1)), factor(vec2, levels=c(0,1)))
  # Handle cases with zero variance to avoid crash
  if(all(dim(tbl) == c(2,2))) {
    test <- mcnemar.test(tbl)
    p_val <- test$p.value
    chi <- test$statistic
    df <- test$parameter
    # Clamp p-value to valid range [0, 1] to handle numerical precision issues
    p_val <- pmax(0, pmin(1, p_val))
  } else {
    p_val <- NA; chi <- NA; df <- NA
  }
  
  diff_pct <- (mean(vec1, na.rm=TRUE) - mean(vec2, na.rm=TRUE)) * 100
  
  return(data.frame(
    Comparison = label,
    Diff_Pct = sprintf("%+.2f%%", diff_pct),
    Chi_Sq = ifelse(is.na(chi), "NA", sprintf("%.3f", chi)),
    P_Value = ifelse(is.na(p_val), "NA", sprintf("%.4g", p_val)),
    Sig = ifelse(!is.na(p_val) & p_val < 0.05, "Yes (*)", "No"),
    stringsAsFactors = FALSE
  ))
}

# Generate McNemar tables for each consistency type
mcnemar_aff_neg <- rbind(
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_initial_aff_neg, metrics_df$consist_verdict_prompt2_initial_aff_neg, "P1 Initial vs P2 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_aff_neg, metrics_df$consist_verdict_prompt2_reconsidered_aff_neg, "P1 Recons. vs P2 Recons."),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_aff_neg, metrics_df$consist_verdict_prompt1_initial_aff_neg, "P1 Recons. vs P1 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt2_reconsidered_aff_neg, metrics_df$consist_verdict_prompt2_initial_aff_neg, "P2 Recons. vs P2 Initial")
)

mcnemar_aff_ant <- rbind(
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_initial_aff_ant, metrics_df$consist_verdict_prompt2_initial_aff_ant, "P1 Initial vs P2 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_aff_ant, metrics_df$consist_verdict_prompt2_reconsidered_aff_ant, "P1 Recons. vs P2 Recons."),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_aff_ant, metrics_df$consist_verdict_prompt1_initial_aff_ant, "P1 Recons. vs P1 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt2_reconsidered_aff_ant, metrics_df$consist_verdict_prompt2_initial_aff_ant, "P2 Recons. vs P2 Initial")
)

mcnemar_neg_ant <- rbind(
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_initial_neg_ant, metrics_df$consist_verdict_prompt2_initial_neg_ant, "P1 Initial vs P2 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_neg_ant, metrics_df$consist_verdict_prompt2_reconsidered_neg_ant, "P1 Recons. vs P2 Recons."),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_neg_ant, metrics_df$consist_verdict_prompt1_initial_neg_ant, "P1 Recons. vs P1 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt2_reconsidered_neg_ant, metrics_df$consist_verdict_prompt2_initial_neg_ant, "P2 Recons. vs P2 Initial")
)

mcnemar_triplet <- rbind(
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_initial_triplet, metrics_df$consist_verdict_prompt2_initial_triplet, "P1 Initial vs P2 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_triplet, metrics_df$consist_verdict_prompt2_reconsidered_triplet, "P1 Recons. vs P2 Recons."),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt1_reconsidered_triplet, metrics_df$consist_verdict_prompt1_initial_triplet, "P1 Recons. vs P1 Initial"),
  run_mcnemar_consist(metrics_df$consist_verdict_prompt2_reconsidered_triplet, metrics_df$consist_verdict_prompt2_initial_triplet, "P2 Recons. vs P2 Initial")
)

# Write Consistency Report
consist_file <- file.path(output_dir, "task2-consistency.csv")

cat("--- CONSISTENCY: AFFIRMATION VS NEGATION ---\n", file = consist_file, append = FALSE)
suppressWarnings(write.table(aff_neg_table, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))
cat("\nSTATISTICAL SIGNIFICANCE (McNemar Tests)\n", file = consist_file, append = TRUE)
suppressWarnings(write.table(mcnemar_aff_neg, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))
cat("\n", file = consist_file, append = TRUE)

cat("--- CONSISTENCY: AFFIRMATION VS ANTONYM ---\n", file = consist_file, append = TRUE)
suppressWarnings(write.table(aff_ant_table, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))
cat("\nSTATISTICAL SIGNIFICANCE (McNemar Tests)\n", file = consist_file, append = TRUE)
suppressWarnings(write.table(mcnemar_aff_ant, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))
cat("\n", file = consist_file, append = TRUE)

cat("--- CONSISTENCY: NEGATION VS ANTONYM ---\n", file = consist_file, append = TRUE)
suppressWarnings(write.table(neg_ant_table, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))
cat("\nSTATISTICAL SIGNIFICANCE (McNemar Tests)\n", file = consist_file, append = TRUE)
suppressWarnings(write.table(mcnemar_neg_ant, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))
cat("\n", file = consist_file, append = TRUE)

cat("--- CONSISTENCY: TRIPLET (Affirmation VS Negation & Antonym) ---\n", file = consist_file, append = TRUE)
suppressWarnings(write.table(triplet_table, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))
cat("\nSTATISTICAL SIGNIFICANCE (McNemar Tests)\n", file = consist_file, append = TRUE)
suppressWarnings(write.table(mcnemar_triplet, file = consist_file, sep = ",", row.names = FALSE, col.names = TRUE, append = TRUE, na = ""))

message("Saved consistency report to: ", consist_file)

# ==========================================
# 4. CONFIDENCE & CORRELATION ANALYSIS
# ==========================================
message("--- Running Confidence & Correlation Analysis ---")

# Define Confidence Columns (mapped from input standardizing: confidence-prompt1-initial -> confidence_prompt1_initial)
conf_cols <- c(
  "confidence_prompt1_initial",
  "confidence_prompt2_initial",
  "confidence_prompt1_reconsidered",
  "confidence_prompt2_reconsidered"
)

# Ensure confidence columns are numeric
# (Empty strings from 'INSUFFICIENT INFO' become NA)
for(col in conf_cols) {
  if(col %in% names(results)) {
    results[[col]] <- suppressWarnings(as.numeric(as.character(results[[col]])))
  } else {
    warning(paste("Column not found:", col))
    results[[col]] <- NA
  }
}

# --- 4a. Overall Average Confidence ---
calc_avg_conf <- function(col_name, label) {
  mean_val <- mean(results[[col_name]], na.rm = TRUE)
  return(data.frame(Scenario = label, Avg_Confidence = sprintf("%.2f", mean_val)))
}

overall_conf <- rbind(
  calc_avg_conf("confidence_prompt1_initial", "Prompt 1 (Initial)"),
  calc_avg_conf("confidence_prompt1_reconsidered", "Prompt 1 (Reconsidered)"),
  calc_avg_conf("confidence_prompt2_initial", "Prompt 2 (Initial)"),
  calc_avg_conf("confidence_prompt2_reconsidered", "Prompt 2 (Reconsidered)")
)

# Add total average confidence across all scenarios
all_conf_total <- c(
  results$confidence_prompt1_initial,
  results$confidence_prompt1_reconsidered,
  results$confidence_prompt2_initial,
  results$confidence_prompt2_reconsidered
)
total_all_conf <- mean(all_conf_total, na.rm = TRUE)
total_all_conf_sd <- sd(all_conf_total, na.rm = TRUE)

overall_conf <- rbind(
  overall_conf,
  data.frame(Scenario = "---", Avg_Confidence = ""),
  data.frame(Scenario = "Total", Avg_Confidence = sprintf("%.2f", total_all_conf))
)

# Add Standard Deviation column to overall_conf
overall_conf$StdDev_Confidence <- ""
overall_conf$StdDev_Confidence[nrow(overall_conf)] <- sprintf("%.2f", total_all_conf_sd)

# --- 4b. Average Confidence by Correctness ---
calc_avg_conf_correctness <- function(conf_col, acc_col, label) {
  correct_conf <- mean(results[[conf_col]][results[[acc_col]] == 1], na.rm = TRUE)
  incorrect_conf <- mean(results[[conf_col]][results[[acc_col]] == 0], na.rm = TRUE)
  return(data.frame(Scenario = label, Avg_Correct_Confidence = sprintf("%.2f", correct_conf), Avg_Incorrect_Confidence = sprintf("%.2f", incorrect_conf)))
}

# Calculate average confidence by correctness
avg_conf_correctness <- rbind(
  calc_avg_conf_correctness("confidence_prompt1_initial", "acc_prompt1_initial", "Prompt 1 (Initial)"),
  calc_avg_conf_correctness("confidence_prompt1_reconsidered", "acc_prompt1_reconsidered", "Prompt 1 (Reconsidered)"),
  calc_avg_conf_correctness("confidence_prompt2_initial", "acc_prompt2_initial", "Prompt 2 (Initial)"),
  calc_avg_conf_correctness("confidence_prompt2_reconsidered", "acc_prompt2_reconsidered", "Prompt 2 (Reconsidered)")
)

# Calculate total statistics across all scenarios
all_correct_conf_vec <- c(
  results$confidence_prompt1_initial[results$acc_prompt1_initial == 1],
  results$confidence_prompt1_reconsidered[results$acc_prompt1_reconsidered == 1],
  results$confidence_prompt2_initial[results$acc_prompt2_initial == 1],
  results$confidence_prompt2_reconsidered[results$acc_prompt2_reconsidered == 1]
)
all_correct_conf <- mean(all_correct_conf_vec, na.rm = TRUE)
all_correct_conf_sd <- sd(all_correct_conf_vec, na.rm = TRUE)

all_incorrect_conf_vec <- c(
  results$confidence_prompt1_initial[results$acc_prompt1_initial == 0],
  results$confidence_prompt1_reconsidered[results$acc_prompt1_reconsidered == 0],
  results$confidence_prompt2_initial[results$acc_prompt2_initial == 0],
  results$confidence_prompt2_reconsidered[results$acc_prompt2_reconsidered == 0]
)
all_incorrect_conf <- mean(all_incorrect_conf_vec, na.rm = TRUE)
all_incorrect_conf_sd <- sd(all_incorrect_conf_vec, na.rm = TRUE)

avg_conf_correctness <- rbind(
  avg_conf_correctness,
  data.frame(Scenario = "---", Avg_Correct_Confidence = "", Avg_Incorrect_Confidence = ""),
  data.frame(Scenario = "Total", Avg_Correct_Confidence = sprintf("%.2f", all_correct_conf), Avg_Incorrect_Confidence = sprintf("%.2f", all_incorrect_conf))
)

# Add Standard Deviation columns
avg_conf_correctness$StdDev_Correct_Confidence <- ""
avg_conf_correctness$StdDev_Incorrect_Confidence <- ""
avg_conf_correctness$StdDev_Correct_Confidence[nrow(avg_conf_correctness)] <- sprintf("%.2f", all_correct_conf_sd)
avg_conf_correctness$StdDev_Incorrect_Confidence[nrow(avg_conf_correctness)] <- sprintf("%.2f", all_incorrect_conf_sd)

# --- 4c. Confidence by Statement Type (Affirmation, Negation, Antonym) ---
conf_by_type <- data.frame()

for(col in conf_cols) {
  # Group by 'type' (normalized to lowercase in Step 1)
  agg <- aggregate(
    list(Avg_Conf = results[[col]]), 
    by = list(Type = results$type), 
    FUN = mean, 
    na.rm=TRUE
  )
  
  agg$Scenario <- col
  conf_by_type <- rbind(conf_by_type, agg)
}

# Clean Scenario names
conf_by_type$Scenario <- gsub("confidence_", "", conf_by_type$Scenario)
conf_by_type$Avg_Conf <- sprintf("%.2f", conf_by_type$Avg_Conf)

# Reshape to wide format for easier reading (Rows=Scenario, Cols=Types)
conf_by_type_wide <- reshape(conf_by_type, idvar="Scenario", timevar="Type", direction="wide", sep="_")

# Add total confidence across all prompts by statement type
types_unique <- unique(results$type)
if (length(types_unique) > 0) {
  total_row <- data.frame(Scenario = "Total")
  for (t in types_unique) {
    stacked_vals <- c(
      results$confidence_prompt1_initial[results$type == t],
      results$confidence_prompt1_reconsidered[results$type == t],
      results$confidence_prompt2_initial[results$type == t],
      results$confidence_prompt2_reconsidered[results$type == t]
    )
    mean_val <- mean(stacked_vals, na.rm = TRUE)
    col_name <- paste0("Avg_Conf_", t)
    total_row[[col_name]] <- sprintf("%.2f", mean_val)
  }
  # Ensure missing type columns are present with empty strings to align rbind
  for (cn in setdiff(names(conf_by_type_wide), names(total_row))) {
    total_row[[cn]] <- ""
  }
  # Align column order
  total_row <- total_row[names(conf_by_type_wide)]
  conf_by_type_wide <- rbind(conf_by_type_wide, total_row)
}

# --- 4d. Correlation (Confidence vs Accuracy) ---
# Correlation requires the accuracy columns computed in Step 2.
# Using Point-Biserial Correlation (which is equivalent to Pearson when one var is binary).
# We exclude 'NA' pairs (INSUFFICIENT INFO).

acc_map <- c(
  "confidence_prompt1_initial" = "acc_prompt1_initial",
  "confidence_prompt2_initial" = "acc_prompt2_initial",
  "confidence_prompt1_reconsidered" = "acc_prompt1_reconsidered",
  "confidence_prompt2_reconsidered" = "acc_prompt2_reconsidered"
)

corr_results <- data.frame()

for(c_col in names(acc_map)) {
  a_col <- acc_map[[c_col]]
  
  if(a_col %in% names(results) & c_col %in% names(results)) {
    # Calculate Correlation
    # use="complete.obs" removes rows where confidence is NA (Insufficient Info)
    cor_val <- cor(results[[c_col]], results[[a_col]], use = "complete.obs", method = "pearson")
    
    # Count valid pairs used for correlation
    valid_pairs <- sum(!is.na(results[[c_col]]) & !is.na(results[[a_col]]))
    
    corr_results <- rbind(corr_results, data.frame(
      Scenario = gsub("confidence_", "", c_col),
      Correlation_Coef = sprintf("%.4f", cor_val),
      Valid_Data_Points = valid_pairs
    ))
  }
}

# Calculate total correlation across all scenarios
all_conf <- c(
  results$confidence_prompt1_initial,
  results$confidence_prompt1_reconsidered,
  results$confidence_prompt2_initial,
  results$confidence_prompt2_reconsidered
)

all_acc <- c(
  results$acc_prompt1_initial,
  results$acc_prompt1_reconsidered,
  results$acc_prompt2_initial,
  results$acc_prompt2_reconsidered
)

total_cor_val <- cor(all_conf, all_acc, use = "complete.obs", method = "pearson")
total_valid_pairs <- sum(!is.na(all_conf) & !is.na(all_acc))

corr_results <- rbind(
  corr_results,
  data.frame(Scenario = "---", Correlation_Coef = "", Valid_Data_Points = ""),
  data.frame(
    Scenario = "Total",
    Correlation_Coef = sprintf("%.4f", total_cor_val),
    Valid_Data_Points = total_valid_pairs
  )
)

# Write Confidence Report
conf_file <- file.path(output_dir, "task3-confidence.csv")

cat("--- OVERALL AVERAGE CONFIDENCE ---\n", file = conf_file)
suppressWarnings(write.table(overall_conf, file = conf_file, sep = ",", row.names = FALSE, append = TRUE))

cat("\n\n--- AVERAGE CONFIDENCE BY CORRECTNESS ---\n", file = conf_file, append = TRUE)
suppressWarnings(write.table(avg_conf_correctness, file = conf_file, sep = ",", row.names = FALSE, append = TRUE))

cat("\n\n--- CONFIDENCE BY STATEMENT TYPE ---\n", file = conf_file, append = TRUE)
suppressWarnings(write.table(conf_by_type_wide, file = conf_file, sep = ",", row.names = FALSE, append = TRUE))

cat("\n\n--- CORRELATION: CONFIDENCE VS ACCURACY ---\n", file = conf_file, append = TRUE)
suppressWarnings(write.table(corr_results, file = conf_file, sep = ",", row.names = FALSE, append = TRUE))

message("Saved confidence report to: ", conf_file)

# ==========================================
# 5. FLIP RATE ANALYSIS (Initial vs Reconsidered)
# ==========================================
message("--- Running Flip Rate Analysis ---")

# Flip Rate: Percentage of cases where initial and reconsidered verdicts differ
# Computed at triplet level (comparing same statement across prompts)

# Prepare data for flip rate analysis
results$triplet_id <- ceiling(seq_len(nrow(results)) / 3)

# Function to check if two verdicts are different (flipped)
check_flip <- function(initial_verdict, reconsidered_verdict) {
  v_init <- normalize_bool(initial_verdict)
  v_recon <- normalize_bool(reconsidered_verdict)
  
  # Return 1 if they differ (flipped), 0 if same, NA if either is INSUFFICIENT INFO
  return(ifelse(v_init == "INSUFFICIENT INFO" | v_recon == "INSUFFICIENT INFO", NA,
         ifelse(v_init != v_recon, 1, 0)))
}

# Calculate flip rates for each prompt
results$flip_prompt1 <- check_flip(results$verdict_prompt1_initial, results$verdict_prompt1_reconsidered)
results$flip_prompt2 <- check_flip(results$verdict_prompt2_initial, results$verdict_prompt2_reconsidered)

# Aggregate flip rate by statement type
calc_flip_stats <- function(col_data, label) {
  flip_count <- sum(col_data == 1, na.rm = TRUE)
  abstain_count <- sum(is.na(col_data))
  valid_total <- sum(!is.na(col_data))
  flip_rate <- if(valid_total > 0) mean(col_data, na.rm = TRUE) else NA
  
  return(data.frame(
    Scenario = label,
    Flip_Rate_Pct = sprintf("%.2f%%", flip_rate * 100),
    Flipped_Cases = flip_count,
    Stable_Cases = valid_total - flip_count,
    Abstained = abstain_count,
    Total_Cases = valid_total,
    stringsAsFactors = FALSE
  ))
}

flip_summary <- rbind(
  calc_flip_stats(results$flip_prompt1, "Prompt 1"),
  calc_flip_stats(results$flip_prompt2, "Prompt 2")
)

# Flip rate by statement type (within each prompt)
flip_by_type <- data.frame()

for(prompt in c("prompt1", "prompt2")) {
  flip_col <- paste0("flip_", prompt)
  
  agg <- aggregate(
    list(Flip_Rate = results[[flip_col]]),
    by = list(Type = results$type),
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  
  agg$Scenario <- prompt
  agg$Flip_Rate <- sprintf("%.2f%%", agg$Flip_Rate * 100)
  flip_by_type <- rbind(flip_by_type, agg)
}

# Reshape to wide format (Rows=Prompt, Cols=Type)
flip_by_type_wide <- reshape(flip_by_type, idvar = "Scenario", timevar = "Type", direction = "wide", sep = "_")

# --- Flip rate by correctness (Initial answers that were correct vs incorrect) ---
flip_by_correctness <- data.frame()

# Prompt 1
correct_flip_p1 <- mean(results$flip_prompt1[results$acc_prompt1_initial == 1], na.rm = TRUE)
incorrect_flip_p1 <- mean(results$flip_prompt1[results$acc_prompt1_initial == 0], na.rm = TRUE)
correct_count_p1 <- sum(results$flip_prompt1[results$acc_prompt1_initial == 1] == 1, na.rm = TRUE)
incorrect_count_p1 <- sum(results$flip_prompt1[results$acc_prompt1_initial == 0] == 1, na.rm = TRUE)
correct_valid_p1 <- sum(!is.na(results$flip_prompt1[results$acc_prompt1_initial == 1]))
incorrect_valid_p1 <- sum(!is.na(results$flip_prompt1[results$acc_prompt1_initial == 0]))

flip_by_correctness <- rbind(flip_by_correctness, data.frame(
  Scenario = "Prompt 1 - Initially Correct",
  Flip_Rate_Pct = sprintf("%.2f%%", correct_flip_p1 * 100),
  Flipped_Cases = correct_count_p1,
  Stable_Cases = correct_valid_p1 - correct_count_p1,
  Total_Cases = correct_valid_p1,
  stringsAsFactors = FALSE
))

flip_by_correctness <- rbind(flip_by_correctness, data.frame(
  Scenario = "Prompt 1 - Initially Incorrect",
  Flip_Rate_Pct = sprintf("%.2f%%", incorrect_flip_p1 * 100),
  Flipped_Cases = incorrect_count_p1,
  Stable_Cases = incorrect_valid_p1 - incorrect_count_p1,
  Total_Cases = incorrect_valid_p1,
  stringsAsFactors = FALSE
))

# Prompt 2
correct_flip_p2 <- mean(results$flip_prompt2[results$acc_prompt2_initial == 1], na.rm = TRUE)
incorrect_flip_p2 <- mean(results$flip_prompt2[results$acc_prompt2_initial == 0], na.rm = TRUE)
correct_count_p2 <- sum(results$flip_prompt2[results$acc_prompt2_initial == 1] == 1, na.rm = TRUE)
incorrect_count_p2 <- sum(results$flip_prompt2[results$acc_prompt2_initial == 0] == 1, na.rm = TRUE)
correct_valid_p2 <- sum(!is.na(results$flip_prompt2[results$acc_prompt2_initial == 1]))
incorrect_valid_p2 <- sum(!is.na(results$flip_prompt2[results$acc_prompt2_initial == 0]))

flip_by_correctness <- rbind(flip_by_correctness, data.frame(
  Scenario = "Prompt 2 - Initially Correct",
  Flip_Rate_Pct = sprintf("%.2f%%", correct_flip_p2 * 100),
  Flipped_Cases = correct_count_p2,
  Stable_Cases = correct_valid_p2 - correct_count_p2,
  Total_Cases = correct_valid_p2,
  stringsAsFactors = FALSE
))

flip_by_correctness <- rbind(flip_by_correctness, data.frame(
  Scenario = "Prompt 2 - Initially Incorrect",
  Flip_Rate_Pct = sprintf("%.2f%%", incorrect_flip_p2 * 100),
  Flipped_Cases = incorrect_count_p2,
  Stable_Cases = incorrect_valid_p2 - incorrect_count_p2,
  Total_Cases = incorrect_valid_p2,
  stringsAsFactors = FALSE
))

# Write Flip Rate Report
fliprate_file <- file.path(output_dir, "task4-fliprate.csv")

cat("--- OVERALL FLIP RATE ---\n", file = fliprate_file)
suppressWarnings(write.table(flip_summary, file = fliprate_file, sep = ",", row.names = FALSE, append = TRUE))

cat("\n\n--- FLIP RATE BY STATEMENT TYPE ---\n", file = fliprate_file, append = TRUE)
suppressWarnings(write.table(flip_by_type_wide, file = fliprate_file, sep = ",", row.names = FALSE, append = TRUE))

cat("\n\n--- FLIP RATE BY INITIAL CORRECTNESS ---\n", file = fliprate_file, append = TRUE)
suppressWarnings(write.table(flip_by_correctness, file = fliprate_file, sep = ",", row.names = FALSE, append = TRUE))

message("Saved flip rate report to: ", fliprate_file)
message("All tasks complete.")