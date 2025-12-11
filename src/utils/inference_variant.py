import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import datetime
from MethylAI.src.dataset.variant_dataset import VariantDataset
from MethylAI.src.utils.utils import check_output_folder, debug_methods

@debug_methods
class VariantInferenceTools:
    def __init__(self, model: nn.Module, model_state_file: str, gpu_number: int,
                 is_reverse_complement_augmentation: bool, print_per_step: int,
                 output_folder: str, output_prefix: str):
        # model & load state
        self.model = model
        self.device = torch.device(gpu_number)
        self.model.to(self.device)
        self._load_model_state(model_state_file)
        # 是否使用负链数据增强
        self.is_reverse_complement_augmentation = is_reverse_complement_augmentation
        # 多少步print一次
        self.print_per_step = print_per_step
        # 输出目录
        self.output_folder = output_folder
        self.output_prefix = f'{output_folder}/{output_prefix}'
        check_output_folder(self.output_folder)
        # dataset
        self.dataset_df = pd.DataFrame()
        self.dataset_col_list = []
        # 计算结果
        self.ref_prediction_list = []
        self.alt_prediction_list = []
        self.cg_change_list = []
        self.prediction_df_header = []
        self.prediction_df = pd.DataFrame()
        self.dataset_prediction_df = pd.DataFrame()

    def _load_model_state(self, model_stat_file):
        print(f'load model checkpoint: {model_stat_file}')
        all_state = torch.load(model_stat_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(all_state['self.model'])

    def generate_prediction_df_header(self, dataset_info_file: str,
                                      col_prefix_tuple=('smooth', 'raw', 'window_1000', 'window_500', 'window_200')):
        # 读取data_info_df，产生表头
        data_info_df = pd.read_table(dataset_info_file, sep='\t', header=0)
        keep_index = data_info_df[data_info_df['is_keep'] == 'yes'].index
        col_index_list = data_info_df.loc[keep_index, 'dataset_index'].tolist()
        for col_prefix in col_prefix_tuple:
            self.prediction_df_header.extend([f'prediction_{col_prefix}_{index}' for index in col_index_list])

    def generate_prediction_list(
        self, variant_dataset: VariantDataset, batch_size=100, num_workers=8
    ):
        # 创建dataloader
        variant_dataloader = DataLoader(
            dataset=variant_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # 记录step
        total_step = len(variant_dataloader)
        # 获取evaluation_dataset里的dataframe，存放到self.dataset_dataframe
        self.dataset_df = variant_dataset.get_dataset_df()
        self.dataset_col_list = self.dataset_df.columns.tolist()
        # 把模型设置为evaluation模式
        self.model.eval()
        # 记录开始时间
        evaluation_start_time = datetime.datetime.now()
        # 运行模型，获取输出
        if self.is_reverse_complement_augmentation:
            with torch.no_grad():
                for batch_index, (
                        forward_ref_dna_one_hot, reverse_ref_dna_one_hot,
                        forward_alt_dna_one_hot, reverse_alt_dna_one_hot,
                        cg_tensor
                ) in enumerate(variant_dataloader):
                    # 把DNA tensor移到GPU
                    forward_ref_dna_one_hot_tensor = forward_ref_dna_one_hot.to(self.device)
                    reverse_ref_dna_one_hot_tensor = reverse_ref_dna_one_hot.to(self.device)
                    forward_alt_dna_one_hot_tensor = forward_alt_dna_one_hot.to(self.device)
                    reverse_alt_dna_one_hot_tensor = reverse_alt_dna_one_hot.to(self.device)
                    # 计算cpg_embedding, module output
                    forward_ref_model_output = self.model(forward_ref_dna_one_hot_tensor)
                    reverse_ref_model_output = self.model(reverse_ref_dna_one_hot_tensor)
                    forward_alt_model_output = self.model(forward_alt_dna_one_hot_tensor)
                    reverse_alt_model_output = self.model(reverse_alt_dna_one_hot_tensor)
                    # 计算model_output的平均
                    average_ref_model_output = (forward_ref_model_output.detach() + reverse_ref_model_output.detach()) / 2
                    average_alt_model_output = (forward_alt_model_output.detach() + reverse_alt_model_output.detach()) / 2
                    # 把上述模型输出结果存放到对应的list
                    self.ref_prediction_list.append(average_ref_model_output.detach().cpu().numpy())
                    self.alt_prediction_list.append(average_alt_model_output.detach().cpu().numpy())
                    self.cg_change_list.append(cg_tensor.numpy())
                    if batch_index % self.print_per_step == 0:
                        print(f'batch index: {batch_index:5d}|{total_step:5d} (reverse complement mode)')
                        print(f'{forward_ref_dna_one_hot.shape=}')
                        print(f'{forward_alt_dna_one_hot.shape=}')
                        using_time = datetime.datetime.now() - evaluation_start_time
                        print(f'using time is: {using_time}\n')
        else:
            with torch.no_grad():
                for batch_index, (
                        ref_dna_one_hot, alt_dna_one_hot, cg_tensor
                ) in enumerate(variant_dataloader):
                    ref_dna_one_hot_tensor = ref_dna_one_hot.to(self.device)
                    alt_dna_one_hot_tensor = alt_dna_one_hot.to(self.device)
                    ref_cpg_embedding, ref_model_output = self.model(ref_dna_one_hot_tensor)
                    alt_cpg_embedding, alt_model_output = self.model(alt_dna_one_hot_tensor)
                    # 把上述模型输出结果存放到对应的list
                    self.ref_prediction_list.append(ref_model_output.detach().cpu().numpy())
                    self.alt_prediction_list.append(alt_model_output.detach().cpu().numpy())
                    self.cg_change_list.append(cg_tensor.numpy())
                    if batch_index % self.print_per_step == 0:
                        print(f'batch index: {batch_index:5d}|{total_step:5d}')
                        print(f'{ref_dna_one_hot.shape=}')
                        print(f'{alt_dna_one_hot.shape=}')
                        using_time = datetime.datetime.now() - evaluation_start_time
                        print(f'using time is: {using_time}\n')
        # 运行结束，把模型移除
        self.model.to('cpu')
        torch.cuda.empty_cache()

    def generate_dataset_prediction_dataframe(self):
        # ref_prediction_df
        ref_prediction_numpy = np.concatenate(self.ref_prediction_list, axis=0)
        ref_prediction_df = pd.DataFrame(ref_prediction_numpy)
        if not self.prediction_df_header:
            self.prediction_df_header = [
                f'{index}' for index in range(ref_prediction_numpy.shape[1])
            ]
        ref_prediction_df.columns = [f'ref_{col}' for col in self.prediction_df_header]
        # alt_prediction_df
        alt_prediction_numpy = np.concatenate(self.alt_prediction_list, axis=0)
        alt_prediction_df = pd.DataFrame(alt_prediction_numpy)
        alt_prediction_df.columns = [f'alt_{col}' for col in self.prediction_df_header]
        # cg_change_df
        cg_change_numpy = np.concatenate(self.cg_change_list, axis=0)
        cg_change_df = pd.DataFrame(cg_change_numpy)
        cg_change_df.columns = ['cg_change']
        # self.prediction_dataframe
        self.dataset_prediction_df = pd.concat(
            [self.dataset_df, ref_prediction_df, alt_prediction_df, cg_change_df], axis=1
        )

    def output_variant_prediction_dataframe(
            self, output_prefix_tuple: tuple = None, output_postfix_tuple: tuple = None, output_file: str = None):
        # 选择需要输出的列
        output_prediction_col_list = self.dataset_prediction_df.columns.tolist()
        if output_prefix_tuple:
            output_prediction_col_list = [col for col in output_prediction_col_list if col.startswith(output_prefix_tuple)]
        if output_postfix_tuple:
            output_prediction_col_list = [col for col in output_prediction_col_list if col.endswith(output_postfix_tuple)]
        output_prediction_col_list = self.dataset_col_list + output_prediction_col_list + ['cg_change']
        output_prediction_dataframe = self.dataset_prediction_df[output_prediction_col_list]
        # 设置输出文件名
        if output_file:
            output_file= f'{self.output_prefix}_{output_file}'
        else:
            output_file = f'{self.output_prefix}_variant_prediction_dataframe.txt'
        # 输出文件
        print(f'output: {output_file}')
        output_prediction_dataframe.to_csv(output_file, sep='\t', index=False)
