import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.attribution_motif import EvaluationResult, MotifAnalysis

def main_argparse():
    parser = argparse.ArgumentParser(description='Get active motif results.')
    # required parameter
    parser.add_argument('--motif_statistic_file', required=True, help='')
    parser.add_argument('--motif_statistic_file', required=True, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    parser.add_argument('--evaluation_file', required=True, help='')
    # optional parameter
    parser.add_argument('--print_per_step', type=int, default=500, help='')
    # 解析参数
    args = parser.parse_args()
    # motif analysis
    motif_analysis = MotifAnalysis(
        motif_statistic_file=args.motif_statistic_file,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix
    )
    motif_analysis.join_captum_cpg_df_col(captum_cpg_file=args.captum_cpg_file, join_col=)
    motif_analysis.join_evaluation_df()

















