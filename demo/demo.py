import argparse
from MethylAI.src.model.methylai import MethlyAI
from MethylAI.src.utils.utils import load_config
from MethylAI.src.dataset.validation_dataset import MethylAIValidationDataset
from MethylAI.src.utils.inference import InferenceTools

def demo(config_file: str, config_dict_name, model_state_file: str, gpu_id: int, cpg_coordinate_file: str, genome_fasta_file: str,
         is_reverse_complement_augmentation: bool, batch_size: int, num_workers: int,
         is_output_bedgraph_format:bool, output_folder: str, output_prefix: str):
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

def main_demo():
    demo(config_file='../configs/methylai_finetune_encode.py',
         config_dict_name='',
         model_state_file='../checkpoint/MethylAI_finetune_encode.pth',
         gpu_id=0,
         cpg_coordinate_file='demo_data/cpg_coordinate.txt',
         genome_fasta_file='../data/genome/hg38.fa',
         is_reverse_complement_augmentation=True,
         batch_size=200,
         num_workers=8,
         is_output_bedgraph_format=False,
         output_folder='demo_result',
         output_prefix='demo')

def main_argparse():
    parser = argparse.ArgumentParser(description='demo script')
    # 添加位置参数
    parser.add_argument('input_file', help='输入文件路径')

    # 添加可选参数
    parser.add_argument('-o', '--output', help='输出文件路径', default='output.txt')
    parser.add_argument('-v', '--verbose', help='详细输出', action='store_true')
    parser.add_argument('--count', type=int, help='重复次数', default=1)
    parser.add_argument('--mode', choices=['fast', 'normal', 'slow'],
                        default='normal', help='运行模式')

    # 解析参数
    args = parser.parse_args()

if __name__ == "__main__":
    main_demo()
