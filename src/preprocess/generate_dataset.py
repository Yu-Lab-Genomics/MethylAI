import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.model_dataset import MethylationDataset

def main_argparse():
    parser = argparse.ArgumentParser(description='Generate modeling dataset')
    # required parameter
    parser.add_argument('--smooth_methylation_file', required=True, help='')
    parser.add_argument('--data_info_file', required=True, help='')
    parser.add_argument('--genome_fasta_file', required=True, help='')
    parser.add_argument('--chrom_size_file', required=True, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--model_input_dna_length', type=int, default=18432, help='')
    parser.add_argument('--threshold_min_coverage', type=int, default=5, help='')
    parser.add_argument('--threshold_max_missing_cpg_ratio', type=float, default=0.5, help='')
    parser.add_argument('--threshold_max_n_base_ratio', type=float, default=0.02, help='')
    parser.add_argument('--threshold_max_missing_sample_ratio', type=float, default=0.5, help='')
    parser.add_argument('--output_format', choices=['pickle', 'feather'], default='pickle', help='')
    parser.add_argument('--output_sampled_training_set', action='store_true', help='')
    parser.add_argument('--output_slice_training_set', action='store_true', help='')
    parser.add_argument('--quite', action='store_true', help='')
    parser.add_argument('--calculate_regional_methylation', nargs='+', type=int,
                        default=[1000, 500, 200], help='')
    parser.add_argument('--training_chr', nargs='+',
                        default=[f'chr{i}' for i in range(1, 10)] + [f'chr{i}' for i in range(12, 23)], help='')
    parser.add_argument('--validation_chr', nargs='+',
                        default=['chr10'], help='')
    parser.add_argument('--test_chr', nargs='+',
                        default=['chr11'], help='')
    # 解析参数
    args = parser.parse_args()
    methylation_dataset = MethylationDataset(
        methylation_file=args.smooth_methylation_file,
        data_info_file=args.data_info_file,
        chromosome_size_file=args.chrom_size_file,
        genome_fasta_file=args.genome_fasta_file,
        model_input_dna_length=args.model_input_dna_length,
        minimal_coverage=args.threshold_min_coverage,
        is_quiet=args.quite,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix
    )
    methylation_dataset.methylation_dataframe_drop_sample(max_low_coverage_ratio=args.threshold_max_missing_cpg_ratio)
    if args.calculate_regional_methylation:
        methylation_dataset.calculate_regional_methylation(args.calculate_regional_methylation)
    methylation_dataset.methylation_dataframe_fill_na()
    methylation_dataset.calculate_input_dna_coordinate()
    methylation_dataset.count_input_dna_n_base_number()
    methylation_dataset.count_missing_sample()
    methylation_dataset.reset_methylation_df_col_order()
    methylation_dataset.output_methylation_df(args.output_complete_dataset_file)
    methylation_dataset.trim_methylation_df(
        max_n_base_ratio=args.threshold_max_n_base_ratio, max_missing_sample_ratio=args.threshold_max_missing_sample_ratio
    )
    methylation_dataset.output_train_validation_test_set(
        train_chr_list=args.training_chr,
        validation_chr_list=args.validation_chr,
        test_chr_list=args.test_chr,
        output_format=args.output_format,
        is_output_sampled_train_set=args.output_sampled_training_set,
        is_output_slice_train_set=args.output_slice_training_set
    )


if __name__ == '__main__':
    main_argparse()
