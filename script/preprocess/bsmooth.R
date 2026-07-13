arguments = commandArgs(trailingOnly = TRUE)

if (length(arguments) < 5) {
    stop(
        paste(
            "Usage: Rscript bsmooth.R <input_dir> <input_suffix> <thread_number>",
            "<output_sample_info_file> <output_methylation_file> [smooth_ns] [smooth_h]"
        )
    )
}

input_dir = arguments[1]
input_suffix = arguments[2]
thread_number = as.integer(arguments[3])
output_sample_info_file = arguments[4]
output_methylation_file = arguments[5]

if (length(arguments) > 5) {
    smooth_ns = as.integer(arguments[6])
    smooth_h = as.integer(arguments[7])
} else {
    smooth_ns = 70
    smooth_h = 1000
}

suppressPackageStartupMessages({
    library(arrow)
    library(BiocParallel)
    library(bsseq)
    library(data.table)
    library(glue)
})

# 识别输入/输出文件格式，仅支持文本表格和 feather。
# Detect input/output file formats. Only text tables and feather are supported.
get_table_format = function(file_path) {
    lower_path = tolower(file_path)
    if (endsWith(lower_path, ".feather")) {
        return("feather")
    }
    if (endsWith(lower_path, ".tsv") || endsWith(lower_path, ".tsv.gz")) {
        return("tsv")
    }
    if (endsWith(lower_path, ".txt") || endsWith(lower_path, ".txt.gz")) {
        return("txt")
    }
    stop(glue(
        "Unsupported file format: {file_path}. ",
        "Only txt, tsv, and feather are supported."
    ))
}

check_positive_integer = function(value, argument_name) {
    if (is.na(value) || value <= 0) {
        stop(glue("{argument_name} must be a positive integer."))
    }
}

check_output_file = function(file_path) {
    output_dir = dirname(file_path)
    if (!dir.exists(output_dir)) {
        stop(glue("Output directory does not exist: {output_dir}"))
    }
    if (file.exists(file_path)) {
        stop(glue("Output file already exists, please choose another path: {file_path}"))
    }
}

read_table = function(file_path) {
    table_format = get_table_format(file_path)
    message(glue("Reading methylation file: {file_path}"))
    if (table_format == "feather") {
        methylation_dataframe = as.data.frame(arrow::read_feather(file_path))
    } else {
        methylation_dataframe = data.table::fread(
            file_path,
            header = TRUE,
            sep = "\t",
            quote = "",
            nThread = thread_number,
            data.table = FALSE
        )
    }
    return(methylation_dataframe)
}

write_table = function(output_dataframe, file_path) {
    table_format = get_table_format(file_path)
    message(glue("Writing output file: {file_path}"))
    if (table_format == "feather") {
        arrow::write_feather(output_dataframe, file_path, version = 2, compression = "zstd")
    } else {
        data.table::fwrite(
            output_dataframe,
            file = file_path,
            sep = "\t",
            quote = FALSE,
            row.names = FALSE,
            na = "NA"
        )
    }
}

validate_methylation_dataframe = function(methylation_dataframe, methylation_file) {
    required_columns = c("chr", "start", "end", "mc", "cov")
    missing_columns = setdiff(required_columns, colnames(methylation_dataframe))
    if (length(missing_columns) > 0) {
        stop(glue(
            "Input file {methylation_file} is missing required columns: ",
            "{paste(missing_columns, collapse = ', ')}"
        ))
    }
    if (nrow(methylation_dataframe) == 0) {
        stop(glue("Input file is empty after reading: {methylation_file}"))
    }
}

read_methylation_bsseq = function(methylation_file, dataset_index) {
    methylation_dataframe = read_table(methylation_file)
    validate_methylation_dataframe(methylation_dataframe, methylation_file)

    # 构建单样本 BSseq 对象，后续按样本顺序 combine。
    # Build one single-sample BSseq object, then combine samples in order.
    methylation_bsseq = BSseq(
        chr = methylation_dataframe$chr,
        pos = methylation_dataframe$start,
        M = matrix(methylation_dataframe$mc, ncol = 1),
        Cov = matrix(methylation_dataframe$cov, ncol = 1),
        sampleNames = as.character(dataset_index)
    )
    return(methylation_bsseq)
}

make_parallel_param = function(workers) {
    manager_port = 42000 + (Sys.getpid() %% 10000)
    message(glue("BiocParallel manager_port: {manager_port}"))
    return(MulticoreParam(
        workers = workers,
        progressbar = TRUE,
        manager.port = manager_port
    ))
}

message(glue("input_dir: {input_dir}"))
message(glue("input_suffix: {input_suffix}"))
message(glue("thread_number: {thread_number}"))
message(glue("output_sample_info_file: {output_sample_info_file}"))
message(glue("output_methylation_file: {output_methylation_file}"))
message(glue("smooth_ns: {smooth_ns}"))
message(glue("smooth_h: {smooth_h}"))

if (!dir.exists(input_dir)) {
    stop(glue("Input directory does not exist: {input_dir}"))
}
check_positive_integer(thread_number, "thread_number")
check_positive_integer(smooth_ns, "smooth_ns")
check_positive_integer(smooth_h, "smooth_h")
invisible(get_table_format(input_suffix))
invisible(get_table_format(output_sample_info_file))
invisible(get_table_format(output_methylation_file))
check_output_file(output_sample_info_file)
check_output_file(output_methylation_file)

all_input_files = list.files(input_dir, full.names = TRUE)
input_file_vector = sort(all_input_files[endsWith(basename(all_input_files), input_suffix)])
if (length(input_file_vector) == 0) {
    stop(glue("No input files found in {input_dir} with suffix {input_suffix}"))
}
if (any(file.size(input_file_vector) == 0)) {
    stop("At least one input file is empty.")
}

sample_info_dataframe = data.frame(
    dataset_index = seq_along(input_file_vector),
    methylation_file = basename(input_file_vector)
)
write_table(sample_info_dataframe, output_sample_info_file)

# 逐个读取样本并合并，避免一次性读取所有输入表格。
# Read and combine samples one by one to avoid loading all source tables at once.
combined_methylation_bsseq = read_methylation_bsseq(input_file_vector[1], 1)
if (length(input_file_vector) >= 2) {
    for (methylation_dataset_index in 2:length(input_file_vector)) {
        new_methylation_bsseq = read_methylation_bsseq(
            input_file_vector[methylation_dataset_index],
            methylation_dataset_index
        )
        combined_methylation_bsseq = combine(combined_methylation_bsseq, new_methylation_bsseq)
    }
}

message("Running BSmooth.")
smooth_combined_methylation_bsseq = BSmooth(
    combined_methylation_bsseq,
    ns = smooth_ns,
    h = smooth_h,
    BPPARAM = make_parallel_param(thread_number)
)

# 输出坐标、平滑甲基化、原始甲基化和测序深度。
# Export coordinates, smoothed methylation, raw methylation, and coverage.
output_position = data.frame(granges(smooth_combined_methylation_bsseq))[, c(1, 2, 3)]
output_position$end = output_position$end + 2
output_dataframe = data.frame(
    output_position,
    getMeth(smooth_combined_methylation_bsseq, type = "smooth"),
    getMeth(smooth_combined_methylation_bsseq, type = "raw"),
    getCoverage(smooth_combined_methylation_bsseq, type = "Cov")
)
output_dataframe$start = as.integer(output_dataframe$start)
output_dataframe$end = as.integer(output_dataframe$end)
colnames(output_dataframe) = c(
    "chr", "start", "end",
    paste0("smooth_", sample_info_dataframe$dataset_index),
    paste0("raw_", sample_info_dataframe$dataset_index),
    paste0("coverage_", sample_info_dataframe$dataset_index)
)

write_table(output_dataframe, output_methylation_file)
message("Finish.")
