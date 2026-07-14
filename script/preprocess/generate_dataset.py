import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from MethylAI.src.utils.log import setup_logger
from MethylAI.src.preprocess.model_dataset import MethylationDataset

def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed_value


def main_argparse() -> None:
    parser = argparse.ArgumentParser(description="Generate modeling dataset")
    # 必需参数：与 README 教程保持一致。
    # Required arguments: keep the same workflow as the README tutorial.
    parser.add_argument("--smooth_methylation_file", required=True, help="Smoothed methylation matrix file.")
    parser.add_argument("--data_info_file", required=True, help="Sample information file.")
    parser.add_argument("--genome_fasta_file", required=True, help="Reference genome FASTA file.")
    parser.add_argument("--chrom_size_file", required=True, help="Chromosome size file.")
    parser.add_argument("--output_folder", required=True, help="Output folder for generated dataset files.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output filenames.")

    # 可选参数：默认值用于教程和复现。
    # Optional arguments: defaults are used by the tutorial and reproducibility runs.
    parser.add_argument("--model_input_dna_length", type=int, default=18432, help="Model input DNA sequence length.")
    parser.add_argument("--threshold_min_coverage", type=int, default=5, help="Minimum coverage threshold.")
    parser.add_argument(
        "--threshold_max_missing_cpg_ratio",
        type=float,
        default=0.5,
        help="Maximum low-coverage CpG ratio allowed for a sample.",
    )
    parser.add_argument(
        "--threshold_max_n_base_ratio",
        type=float,
        default=0.02,
        help="Maximum N-base ratio allowed for an input DNA sequence.",
    )
    parser.add_argument(
        "--threshold_max_missing_sample_ratio",
        type=float,
        default=0.5,
        help="Maximum low-coverage sample ratio allowed for a CpG site.",
    )
    parser.add_argument(
        "--complete_dataset_format",
        choices=["tsv", "tsv.gz", "feather"],
        default="tsv",
        help="Output format for the complete dataset. Default: tsv.",
    )
    parser.add_argument("--output_sampled_training_set", nargs="+", type=float, default=[], help="")
    parser.add_argument("--output_slice_training_set", action="store_true", help="")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    parser.add_argument(
        "--calculate_regional_methylation",
        nargs="+",
        type=int,
        default=[1000, 500, 200],
        help="Window sizes for regional methylation. Use 0 to disable.",
    )
    parser.add_argument(
        "--regional_methylation_dtype",
        choices=["float32", "float64"],
        default="float32",
        help=(
            "Numeric dtype for prefix-sum regional methylation. Default: float32. "
            "float64 is mainly for reproducibility analysis and increases runtime/memory use."
        ),
    )
    parser.add_argument(
        "--mvp_cpg_per_chr",
        type=_positive_int,
        default=None,
        help="MVP test mode: keep only the first N CpGs per chromosome before running the full workflow.",
    )
    parser.add_argument(
        "--training_chr",
        nargs="+",
        default=[f"chr{i}" for i in range(1, 10)] + [f"chr{i}" for i in range(12, 23)],
        help="Chromosomes assigned to the training set.",
    )
    parser.add_argument("--validation_chr", nargs="+", default=["chr10"], help="Chromosomes assigned to validation.")
    parser.add_argument("--test_chr", nargs="+", default=["chr11"], help="Chromosomes assigned to testing.")
    parser.add_argument(
        "--log_file",
        default=None,
        help="Optional log file. Default: <output_folder>/<output_prefix>_generate_dataset.log.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")

    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    log_file = Path(args.log_file) if args.log_file else output_folder / f"{args.output_prefix}_generate_dataset.log"
    logger = setup_logger("generate_dataset", log_file=log_file, level=logging.WARNING if args.quiet else logging.INFO)

    quiet = bool(args.quiet)

    logger.info("Start generating MethylAI model datasets.")
    logger.info("smooth_methylation_file: %s", args.smooth_methylation_file)
    logger.info("data_info_file: %s", args.data_info_file)
    logger.info("output_folder: %s", args.output_folder)
    logger.info("output_prefix: %s", args.output_prefix)
    if args.mvp_cpg_per_chr is not None:
        logger.info("MVP mode enabled: keep first %s CpGs per chromosome.", args.mvp_cpg_per_chr)

    methylation_dataset = MethylationDataset(
        methylation_file=args.smooth_methylation_file,
        data_info_file=args.data_info_file,
        chromosome_size_file=args.chrom_size_file,
        genome_fasta_file=args.genome_fasta_file,
        model_input_dna_length=args.model_input_dna_length,
        minimal_coverage=args.threshold_min_coverage,
        is_quiet=quiet,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix,
        logger=logger,
        overwrite=args.overwrite,
    )

    if args.mvp_cpg_per_chr is not None:
        methylation_dataset.keep_first_cpg_per_chr(args.mvp_cpg_per_chr)

    methylation_dataset.methylation_dataframe_drop_sample(
        max_low_coverage_ratio=args.threshold_max_missing_cpg_ratio
    )
    if args.calculate_regional_methylation[0] != 0:
        methylation_dataset.calculate_regional_methylation(
            args.calculate_regional_methylation,
            regional_methylation_dtype=args.regional_methylation_dtype,
        )
    methylation_dataset.methylation_dataframe_fill_na()
    methylation_dataset.calculate_input_dna_coordinate()
    methylation_dataset.count_input_dna_n_base_number()
    methylation_dataset.count_missing_sample()
    methylation_dataset.reset_methylation_df_col_order()
    methylation_dataset.output_complete_dataset(output_format=args.complete_dataset_format)
    methylation_dataset.trim_methylation_df(
        max_n_base_ratio=args.threshold_max_n_base_ratio,
        max_missing_sample_ratio=args.threshold_max_missing_sample_ratio,
    )
    methylation_dataset.output_train_validation_test_set(
        train_chr_list=args.training_chr,
        validation_chr_list=args.validation_chr,
        test_chr_list=args.test_chr,
        output_sampled_train_set_fraction_list=args.output_sampled_training_set,
        is_output_slice_train_set=args.output_slice_training_set,
    )
    logger.info("Finished generating MethylAI model datasets.")


if __name__ == "__main__":
    main_argparse()
