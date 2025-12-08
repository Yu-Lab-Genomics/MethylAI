import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.dataset_tools import LowMethylationRegion

def main_argparse():
    parser = argparse.ArgumentParser(description='Generate modeling dataset')
    # required parameter
    parser.add_argument('--complete_dataset_file', required=True, help='')
    parser.add_argument('--dataset_index', required=True, type=int, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--region_interval', type=int, default=1000, help='')
    parser.add_argument('--threshold_low_methylation', type=float, default=0.25, help='')
    parser.add_argument('--threshold_min_cpg_number', type=int, default=5, help='')
    parser.add_argument('--threshold_min_region_length', type=int, default=50, help='')
    # 解析参数
    args = parser.parse_args()
    # 构建LowMethylationRegion
    low_methylation_region = LowMethylationRegion(
        complete_dataset_file=args.complete_dataset_file,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix,
    )
    low_methylation_region.select_dataset_index(dataset_index=args.dataset_index)
    low_methylation_region.mark_low_me_region(
        threshold_low_me_value=args.threshold_low_methylation,
        threshold_min_cpg_num=args.threshold_min_cpg_number,
        threshold_min_region_len=args.threshold_min_region_length
    )
    low_methylation_region.retain_low_me_region_and_mark_represent_cpg(interval=args.region_interval)
    low_methylation_region.output_processing_dataset_pldf(
        f'smooth_{args.col_index_number}_low_methylation_region.txt'
    )
    low_methylation_region.retain_representative_cpg()
    low_methylation_region.output_processing_dataset_pldf(
        f'smooth_{args.col_index_number}_low_methylation_region_representative_cpg.txt'
    )

if __name__ == '__main__':
    main_argparse()
