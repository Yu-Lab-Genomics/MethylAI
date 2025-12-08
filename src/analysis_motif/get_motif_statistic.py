import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.attribution_motif import JasparBed, CaptumResult, MotifStatistic

def main_argparse():
    parser = argparse.ArgumentParser(description='Get motif attribution statistic.')
    # required parameter
    parser.add_argument('--sequence_attribution_folder', required=True, help='')
    parser.add_argument('--jaspar_bed_file', required=True, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--print_per_step', type=int, default=500, help='')
    parser.add_argument('--captum_cpg_file', help='')
    parser.add_argument('--dna_attribution_file', help='')
    # 解析参数
    args = parser.parse_args()
    # JASPAR TF track
    jaspar_bed = JasparBed(
        jaspar_bed_file=args.jaspar_bed_file
    )
    # sequence attribution result
    if args.captum_cpg_file:
        captum_cpg_file = args.captum_cpg_file
    else:
        captum_cpg_file = f'{args.sequence_attribution_folder}/captum_cpg_dataframe.txt'
    if args.dna_attribution_file:
        dna_attribution_file = args.dna_attribution_file
    else:
        dna_attribution_file = f'{args.sequence_attribution_folder}/numpy/dna_attribution.npy'
    captum_result = CaptumResult(
        captum_cpg_file=captum_cpg_file,
        dna_attribution_file=dna_attribution_file
    )
    # motif statistic
    motif_statistic = MotifStatistic(
        captum_result=captum_result,
        jaspar_bed=jaspar_bed,
        print_per_step=args.print_per_step,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix
    )
    motif_statistic.do_motif_statistic()
    motif_statistic.output_dataframe()

if __name__ == "__main__":
    main_argparse()




























