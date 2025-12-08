import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.model.methylai import MethylAI
from MethylAI.src.dataset.captum_dataset import CaptumDataset
from MethylAI.src.utils.attribution import CaptumTools
from MethylAI.src.utils.utils import load_config

def main_argparse():
    parser = argparse.ArgumentParser(description='Get sequence attribution.')
    # required parameter
    parser.add_argument('--representative_cpg_file', required=True, help='')
    parser.add_argument('--config_file', required=True, help='')
    parser.add_argument('--config_dict_name', required=True, help='')
    parser.add_argument('--model_ckpt', required=True, help='')
    parser.add_argument('--gpu_id', required=True, type=int, help='')
    parser.add_argument('--analyze_name', required=True, help='')
    parser.add_argument('--analyze_output_index', required=True, type=int, help='')
    parser.add_argument('--n_permutation', required=True, type=int, help='')
    parser.add_argument('--output_folder', required=True, help='')
    # optional parameter
    parser.add_argument('--print_per_step', type=int, default=500, help='')
    parser.add_argument('--output_bedgraph', action='store_true', help='')
    # 解析参数
    args = parser.parse_args()
    # model & captum_tools
    methylai_parameter_dict = load_config(args.config_file, args.config_dict_name)
    methylai_model = MethylAI(methylai_parameter_dict)
    captum_tools = CaptumTools(
        model=methylai_model,
        model_state_file=args.model_ckpt,
        gpu_number=args.gpu_id,
        model_input_dna_length=methylai_parameter_dict['input_dna_length'],
        print_per_step=args.print_per_step,
        output_folder=args.output_folder
    )
    # prepare dataset
    captum_dataset = CaptumDataset(
        captum_cpg_file=args.representative_cpg_file,
        genome_fasta_file=methylai_parameter_dict['genome_fasta_file'],
        model_input_dna_length=methylai_parameter_dict['input_dna_length'],
        n_permutation=args.n_permutation
    )
    # get attribution
    captum_tools.iter_captum_dataset(
        captum_dataset=captum_dataset,
        captum_target_name=args.analyze_name,
        captum_target_index=args.analyze_output_index,
        is_output_bedgraph=args.output_bedgraph
    )


if __name__ == "__main__":
    main_argparse()












