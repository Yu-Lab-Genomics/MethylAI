import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.model.methylai import MethylAI
from MethylAI.src.utils.inference import InferenceTools
from MethylAI.src.utils.utils import load_config
from MethylAI.src.dataset.validation_dataset import MethylAIValidationDataset

def main_argparse():
    parser = argparse.ArgumentParser(description='Evaluate prediction accuracy of representative CpG.')
    # required parameter
    parser.add_argument('--representative_cpg_file', required=True, help='')
    parser.add_argument('--dataset_info_file', required=True, help='')
    parser.add_argument('--config_file', required=True, help='')
    parser.add_argument('--config_dict_name', required=True, help='')
    parser.add_argument('--model_ckpt', required=True, help='')
    parser.add_argument('--gpu_id', required=True, type=int, help='')
    parser.add_argument('--batch_size', required=True, type=int, help='')
    parser.add_argument('--num_workers', required=True, type=int, help='')
    parser.add_argument('--col_index_number', required=True, type=int, help='')
    parser.add_argument('--output_folder', required=True, help='')
    parser.add_argument('--output_prefix', required=True, help='')
    # optional parameter
    parser.add_argument('--genome_fasta', help='')
    parser.add_argument('--reverse_complement_augmentation', action='store_true', help='')
    # 解析参数
    args = parser.parse_args()
    # prepare model & inference tool
    methylai_parameter_dict = load_config(args.config_file, args.config_dict_name)
    methylai_model = MethylAI(methylai_parameter_dict)
    inference_tools = InferenceTools(
        model=methylai_model,
        model_state_file=args.model_ckpt,
        gpu_number=args.gpu_id,
        is_reverse_complement_augmentation=args.reverse_complement_augmentation,
        print_per_step=200,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix
    )
    # prepare dataset
    if args.genome_fasta:
        genome_fasta_file = args.genome_fasta
    else:
        genome_fasta_file = methylai_parameter_dict['genome_fasta_file']
    inference_dataset = MethylAIValidationDataset(
        dataset_file=args.representative_cpg_file,
        genome_fasta_file=genome_fasta_file,
        model_input_dna_length=methylai_parameter_dict['input_dna_length'],
        is_reverse_complement_augmentation=args.reverse_complement_augmentation
    )
    inference_tools.generate_prediction_and_embedding_list(
        inference_dataset=inference_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    inference_tools.generate_prediction_df_header(data_info_file=args.dataset_info_file)
    inference_tools.generate_dataset_prediction_df()
    inference_tools.select_output_dataset_prediction_df(col_index_number=args.col_index_number)


if __name__ == "__main__":
    main_argparse()









