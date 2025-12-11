import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.model.methylai import MethylAI
from MethylAI.src.dataset.variant_dataset import VariantDataset
from MethylAI.src.utils.inference_variant import VariantInferenceTools
from MethylAI.src.utils.utils import load_config

def main_argparse():
    parser = argparse.ArgumentParser(description='Get variant effect.')
    # required parameter
    parser.add_argument('--variant_dataset_file', required=True, help='')
    parser.add_argument('--dataset_info_file', required=True, help='')
    parser.add_argument('--config_file', required=True, help='')
    parser.add_argument('--config_dict_name', required=True, help='')
    parser.add_argument('--model_ckpt', required=True, help='')
    parser.add_argument('--gpu_id', required=True, type=int, help='')
    parser.add_argument('--batch_size', required=True, type=int, help='')
    parser.add_argument('--num_workers', required=True, type=int, help='')
    parser.add_argument('--dataset_index', required=True, type=int, nargs='+', help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--print_per_step', type=int, default=500, help='')
    parser.add_argument('--reverse_complement_augmentation', action='store_true', help='')
    # 解析参数
    args = parser.parse_args()
    # model & variant_inference_tools
    methylai_parameter_dict = load_config(args.config_file, args.config_dict_name)
    methylai_model = MethylAI(methylai_parameter_dict)
    variant_inference_tools = VariantInferenceTools(
        model=methylai_model,
        model_state_file=args.model_ckpt,
        gpu_number=args.gpu_id,
        is_reverse_complement_augmentation=args.reverse_complement_augmentation,
        print_per_step=args.print_per_step,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix,
    )
    variant_dataset = VariantDataset(
        variant_dataset_file=args.variant_dataset_file,
        genome_fasta_file=methylai_parameter_dict['genome_fasta_file'],
        model_input_dna_length=methylai_parameter_dict['input_dna_length'],
        is_reverse_complement_augmentation=args.reverse_complement_augmentation,
    )
    variant_inference_tools.generate_prediction_list(
        variant_dataset=variant_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    variant_inference_tools.generate_prediction_df_header(dataset_info_file=args.dataset_info_file)
    variant_inference_tools.generate_dataset_prediction_dataframe()
    if args.dataset_index[0] != 0:
        output_postfix = tuple([f'{dataset_index}' for dataset_index in args.dataset_index])
        variant_inference_tools.output_variant_prediction_dataframe(output_postfix_tuple=output_postfix)
    else:
        variant_inference_tools.output_variant_prediction_dataframe()


if __name__ == "__main__":
    main_argparse()

















