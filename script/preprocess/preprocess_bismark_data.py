import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from MethylAI.src.preprocess.preprocess import BismarkMethylationFile

def main_argparse() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DNA methylation files from Bismark.")

    parser.add_argument("--input_folder", required=True, help="Folder containing Bismark cov files.")
    parser.add_argument("--input_file_suffix", required=True, help="Suffix of Bismark cov files.")
    parser.add_argument("--output_folder", required=True, help="Folder for preprocessed methylation data.")

    parser.add_argument(
        "--reference_cpg_coordinate_file",
        help="Reference CpG coordinate BED file. If provided, output coordinates are aligned to it.",
    )
    parser.add_argument(
        "--output_file_format",
        choices=["tsv", "feather"],
        default="tsv",
        help="Output format for preprocessed methylation data. Default: tsv.",
    )
    parser.add_argument(
        "--metadata_folder_name",
        default="preprocess_metadata",
        help="Subfolder under output_folder for log, summary, and manifest files.",
    )
    parser.add_argument(
        "--output_summary_file",
        default="preprocess_summary.tsv",
        help="Summary filename written under metadata_folder_name.",
    )
    parser.add_argument(
        "--output_manifest_file",
        default="preprocess_manifest.tsv",
        help="Manifest filename written under metadata_folder_name.",
    )
    parser.add_argument(
        "--output_log_file",
        default="preprocess.log",
        help="Runtime log filename written under metadata_folder_name.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        help="Optional maximum number of input files to process, mainly for small tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output, summary, manifest, and log files.",
    )

    args = parser.parse_args()
    methylation_bed_file = BismarkMethylationFile(
        input_folder=Path(args.input_folder),
        input_file_suffix=args.input_file_suffix,
        reference_cpg_coordinate_file=args.reference_cpg_coordinate_file,
        output_folder=Path(args.output_folder),
        output_file_format=args.output_file_format,
        metadata_folder_name=args.metadata_folder_name,
        output_summary_file=args.output_summary_file,
        output_manifest_file=args.output_manifest_file,
        output_log_file=args.output_log_file,
        overwrite=args.overwrite,
    )
    methylation_bed_file.preprocess_all_bed_file(max_files=args.max_files)


if __name__ == "__main__":
    main_argparse()