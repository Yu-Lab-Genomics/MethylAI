import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.variant import VariantAnalysis

def main_argparse():
    parser = argparse.ArgumentParser(description='Analyze variant effect.')
    # required parameter
    parser.add_argument('--variant_effect_file', required=True, help='')
    parser.add_argument('--variant_active_motif_detail_file', required=True, help='')
    parser.add_argument('--dataset_index', required=True, type=int, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--methylation_type', default='low', choices=['low', 'high'], help='')
    parser.add_argument('--threshold_min_variant_effect', type=float, default=0, help='')
    # 解析参数
    args = parser.parse_args()
    # variant effect analysis
    variant_analysis = VariantAnalysis(
        variant_effect_file=args.variant_effect_file,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix
    )
    if args.methylation_type == 'low':
        is_low_methylation = True
    else:
        is_low_methylation = False
    variant_analysis.generate_variant_analysis_result(
        is_low_methylation=is_low_methylation,
        threshold_min_variant_effect=args.threshold_min_variant_effect,
        dataset_index=args.dataset_index,
        variant_active_motif_detail_file=args.variant_active_motif_detail_file
    )
    variant_analysis.output_result()


if __name__ == "__main__":
    main_argparse()













