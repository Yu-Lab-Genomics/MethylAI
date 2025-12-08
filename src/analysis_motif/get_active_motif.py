import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.attribution_motif import MotifAnalysis

def main_argparse():
    parser = argparse.ArgumentParser(description='Get active motif results.')
    # required parameter
    parser.add_argument('--motif_statistic_file', required=True, help='')
    parser.add_argument('--captum_cpg_file', required=True, help='')
    parser.add_argument('--evaluation_file', required=True, help='')
    parser.add_argument('--dataset_index', required=True, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--methylation_type', default='low', choices=['low', 'high'], help='')
    parser.add_argument('--threshold_max_motif_cpg_distance', type=int, default=1000, help='')
    parser.add_argument('--threshold_max_prediction_error', type=float, default=0.2, help='')
    parser.add_argument('--threshold_motif_attribution_mean', type=float, default=-0.02, help='')
    parser.add_argument('--bedtools_path', help='')
    # 解析参数
    args = parser.parse_args()
    # motif analysis
    motif_analysis = MotifAnalysis(
        motif_statistic_file=args.motif_statistic_file,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix
    )
    if args.methylation_type == 'low':
        is_low_methylation = True
        region_col = 'low_me_region_id'
    else:
        is_low_methylation = False
        region_col = 'high_me_region_id'
    motif_analysis.join_captum_cpg_pldf_region_col(
        captum_cpg_file=args.captum_cpg_file,
        region_col=region_col)
    motif_analysis.join_evaluation_pldf(
        evaluation_file=args.evaluation_file,
        dataset_index=args.dataset_index
    )
    motif_analysis.filter_motif_statistic_pldf(
        threshold_max_prediction_error=args.threshold_max_prediction_error,
        threshold_max_motif_cpg_distance=args.threshold_max_motif_cpg_distance
    )
    motif_analysis.set_motif_statistic_filtered_pldf_col_order()
    motif_analysis.generate_active_motif_pldf(
        is_low_methylation=is_low_methylation,
        threshold_motif_attribution_mean=args.threshold_motif_attribution_mean
    )
    motif_analysis.generate_active_motif_summary_pldf()
    motif_analysis.generate_motif_bed_pldf()
    motif_analysis.output_result(bedtools_path=args.bedtools_path)


if __name__ == "__main__":
    main_argparse()

