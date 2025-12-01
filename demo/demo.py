import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))
from MethylAI.src.model.methylai import MethlyAI
from MethylAI.src.utils.utils import load_config
from MethylAI.src.dataset.validation_dataset import MethylAIValidationDataset
from MethylAI.src.utils.inference import InferenceTools

def demo(cpg_coordinate_file: str, genome_fasta_file: str,
         config_file: str, config_dict_name, model_state_file: str, gpu_id: int,
         batch_size: int, num_workers: int,
         is_reverse_complement_augmentation: bool, is_output_bedgraph_format:bool,
         output_folder: str, output_prefix: str):
    # prepare model
    methylai_parameter_dict = load_config(config_file, config_dict_name)
    methylai_model = MethlyAI(methylai_parameter_dict)
    # prepare dataset
    demo_dataset = MethylAIValidationDataset(
        dataset_file=cpg_coordinate_file,
        genome_fasta_file=genome_fasta_file,
        model_input_dna_length=methylai_parameter_dict['input_dna_length'],
        is_reverse_complement_augmentation=is_reverse_complement_augmentation
    )
    # inference tools
    inference_tools = InferenceTools(
        model=methylai_model,
        model_state_file=model_state_file,
        gpu_number=gpu_id,
        is_reverse_complement_augmentation=is_reverse_complement_augmentation,
        print_per_step=200,
        output_folder=output_folder,
        output_prefix=output_prefix
    )
    inference_tools.generate_prediction_and_embedding_list(
        inference_dataset=demo_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    inference_tools.generate_prediction_dataframe()
    inference_tools.output_prediction_dataframe()
    if is_output_bedgraph_format:
        inference_tools.output_prediction_bedgraph_format()

def test_demo():
    demo(cpg_coordinate_file='demo_data/cpg_coordinate.txt',
         genome_fasta_file='../data/genome/hg38.fa',
         config_file='../configs/methylai_finetune_encode.py',
         config_dict_name='methylai_config_dict',
         model_state_file='../checkpoint/MethylAI_finetune_encode.pth',
         gpu_id=0,
         batch_size=200,
         num_workers=8,
         is_reverse_complement_augmentation=True,
         is_output_bedgraph_format=False,
         output_folder='demo_result',
         output_prefix='demo')

def main_argparse():
    parser = argparse.ArgumentParser(description='demo script')
    # required parameter
    parser.add_argument('--cpg_coordinate', required=True, help='BED-formatted file (zero-base) containing CpG site coordinates.')
    parser.add_argument('--genome_fasta', required=True, help='Reference genome FASTA file for sequence extraction.')
    parser.add_argument('--config_file', required=True, help='Python configuration file defining model architecture and hyperparameters.')
    parser.add_argument('--config_dict_name', required=True, help='Name of the Python dictionary variable containing configuration parameters.')
    parser.add_argument('--model_ckpt', required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--gpu_id', required=True, type=int, help='GPU device for computation.')
    parser.add_argument('--batch_size', required=True, type=int, help='Batch size for DataLoader during inference.')
    parser.add_argument('--num_workers', required=True, type=int, help='Number of parallel workers for DataLoader.')
    parser.add_argument('--output_folder', required=True, help='Directory for saving prediction results.')
    # optional parameter
    parser.add_argument('--output_prefix', help='Custom prefix for output files.')
    parser.add_argument('--reverse_complement_augmentation', action='store_true', help='Enable reverse complement data augmentation.')
    parser.add_argument('--output_bedgraph', action='store_true', help='Generate methylation tracks in bedGraph format for genome browser visualization.')
    # 解析参数
    args = parser.parse_args()
    demo(cpg_coordinate_file=args.cpg_coordinate,
         genome_fasta_file=args.genome_fasta,
         config_file=args.config_file,
         config_dict_name=args.config_dict_name,
         model_state_file=args.model_ckpt,
         gpu_id=args.gpu_id,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         is_reverse_complement_augmentation=args.reverse_complement_augmentation,
         is_output_bedgraph_format=args.output_bedgraph,
         output_folder=args.output_folder,
         output_prefix=args.output_prefix)

if __name__ == "__main__":
    # test_demo()
    main_argparse()
