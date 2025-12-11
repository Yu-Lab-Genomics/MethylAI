import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import datetime
from MethylAI.src.dataset.validation_dataset import MethylAIValidationDataset
from MethylAI.src.utils.utils import check_output_folder, debug_methods

@debug_methods
class InferenceTools:
    def __init__(self, model: nn.Module, model_state_file: str, gpu_number: int,
                 is_reverse_complement_augmentation: bool, print_per_step: int,
                 output_folder: str, output_prefix: str = None):
        # model & load state
        self.model = model
        self.device = torch.device(gpu_number)
        self.model.to(self.device)
        self._load_model_state(model_state_file)
        # reverse_complement_augmentation
        self.is_reverse_complement_augmentation = is_reverse_complement_augmentation
        # print per step
        self.print_per_step = print_per_step
        # output folder
        self.output_folder = output_folder
        check_output_folder(self.output_folder)
        self.output_bedgraph_folder = f'{self.output_folder}/bedgraph'
        if output_prefix:
            self.output_prefix = f'{output_folder}/{output_prefix}_'
        else:
            self.output_prefix = f'{output_folder}/'
        # output dataframe: dataset_df & prediction_df & cpg_embedding_df
        self.prediction_df_header = []
        self.dataset_df = pd.DataFrame()
        self.prediction_df = pd.DataFrame()
        self.dataset_prediction_df = pd.DataFrame()
        self.cpg_embedding_df = pd.DataFrame()
        # prediction_list & cpg_embedding_list
        self.prediction_list = []
        self.cpg_embedding_list = []

    def _load_model_state(self, model_stat_file):
        print(f'load model checkpoint: {model_stat_file}')
        all_state = torch.load(model_stat_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(all_state['self.model'])

    def generate_prediction_df_header(self, data_info_file,
                                      col_prefix_tuple = ('smooth', 'raw', 'window_1000', 'window_500', 'window_200')):
        # 读取data_info_df，产生表头
        data_info_df = pd.read_table(data_info_file, sep='\t', header=0)
        keep_index = data_info_df[data_info_df['is_keep'] == 'yes'].index
        col_index_list = data_info_df.loc[keep_index, 'dataset_index'].tolist()
        for col_prefix in col_prefix_tuple:
            self.prediction_df_header.extend([f'prediction_{col_prefix}_{index}' for index in col_index_list])

    def generate_prediction_and_embedding_list(
            self, inference_dataset: MethylAIValidationDataset, batch_size=200, num_workers=8,
    ):
        # 创建dataloader
        evaluation_dataloader = DataLoader(
            dataset=inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # 获取evaluation_dataset里的df，存放到self.dataset_df
        self.dataset_df = inference_dataset.get_dataset_df()
        # 记录总步数
        total_step = len(evaluation_dataloader)
        # 把模型设置为evaluation模式
        self.model.eval()
        # 记录开始时间
        evaluation_start_time = datetime.datetime.now()
        # 开始运行
        # self.is_reverse_complement_augmentation == True
        if self.is_reverse_complement_augmentation:
            with torch.no_grad():
                for batch_index, (forward_dna_one_hot, reverse_dna_one_hot, methylation_level, loss_weight
                                  ) in enumerate(evaluation_dataloader):
                    forward_dna_sequence_one_hot_tensor = forward_dna_one_hot.to(self.device)
                    reverse_dna_sequence_one_hot_tensor = reverse_dna_one_hot.to(self.device)
                    # 计算output
                    forward_cpg_embedding_tensor, forward_output_tensor = self.model(
                        forward_dna_sequence_one_hot_tensor, is_return_cpg_embedding=True
                    )
                    reverse_cpg_embedding_tensor, reverse_output_tensor = self.model(
                        reverse_dna_sequence_one_hot_tensor, is_return_cpg_embedding=True
                    )
                    average_output_tensor = (forward_output_tensor.detach() + reverse_output_tensor.detach()) / 2
                    concat_cpg_embedding_tensor = torch.concatenate(
                        [forward_cpg_embedding_tensor.detach(), reverse_cpg_embedding_tensor.detach()], dim=-1)
                    # 把tensor转换为numpy，并保存到对应的list中
                    self.prediction_list.append(average_output_tensor.cpu().numpy())
                    self.cpg_embedding_list.append(concat_cpg_embedding_tensor.cpu().numpy())
                    # print当前进度
                    if batch_index % self.print_per_step == 0:
                        print(f'batch index: {batch_index:5d}|{total_step:5d} (reverse complement mode)')
                        using_time = datetime.datetime.now() - evaluation_start_time
                        print(f'using time is: {using_time}\n')
        # self.is_reverse_complement_augmentation == False
        else:
            with torch.no_grad():
                for batch_index, (dna_sequence_one_hot_encoding, methylation_level, loss_weight) in enumerate(
                        evaluation_dataloader):
                    dna_sequence_one_hot_encoding_tensor = dna_sequence_one_hot_encoding.to(self.device)
                    cpg_embedding_tensor, output_tensor = self.model(
                        dna_sequence_one_hot_encoding_tensor, is_return_cpg_embedding=True
                    )
                    # 把tensor转换为numpy，并保存到对应的list中
                    self.prediction_list.append(output_tensor.detach().cpu().numpy())
                    self.cpg_embedding_list.append(cpg_embedding_tensor.detach().cpu().numpy())
                    # print当前进度
                    if batch_index % self.print_per_step == 0:
                        print(f'batch index: {batch_index:5d}|{total_step:5d}')
                        using_time = datetime.datetime.now() - evaluation_start_time
                        print(f'using time is: {using_time}\n')
        # 运行结束，把模型移除
        self.model.to('cpu')
        torch.cuda.empty_cache()

    def generate_prediction_df(self):
        # 产生self.prediction_df
        prediction_numpy = np.concatenate(self.prediction_list, axis=0)
        prediction_df = pd.DataFrame(prediction_numpy)
        if not self.prediction_df_header:
            self.prediction_df_header = [
                f'prediction_{index}' for index in range(prediction_numpy.shape[1])
            ]
        prediction_df.columns = self.prediction_df_header
        # 为prediction_df拼接self.dataset_df的前3列（基因组坐标信息）
        self.prediction_df = pd.concat(
            [self.dataset_df.iloc[:, 0:3], prediction_df], axis=1
        )

    def generate_dataset_prediction_df(self):
        if self.prediction_df.empty:
            self.generate_prediction_df()
        self.dataset_df.reset_index(inplace=True, drop=True)
        self.dataset_prediction_df = pd.concat(
            [self.dataset_df, self.prediction_df.iloc[:, 3:]], axis=1
        )

    def generate_cpg_embedding_df(self):
        # 产生self.cpg_embedding_df
        cpg_embedding_numpy = np.concatenate(self.cpg_embedding_list, axis=0)
        cpg_embedding_df = pd.DataFrame(cpg_embedding_numpy)
        cpg_embedding_col_name_list = [
            f'embedding_{index}' for index in range(cpg_embedding_numpy.shape[1])
        ]
        cpg_embedding_df.columns = cpg_embedding_col_name_list
        # 为cpg_embedding_col_name_list拼接self.dataset_df的前3列（基因组坐标信息）
        self.cpg_embedding_df = pd.concat(
            [self.dataset_df.iloc[:, 0:3], cpg_embedding_df], axis=1
        )

    def output_cpg_embedding_df(self):
        # 输出文件
        file_name = f'{self.output_prefix}cpg_embedding_dataframe.txt'
        print('output:', file_name)
        self.cpg_embedding_df.to_csv(file_name, sep='\t', index=False)

    def output_prediction_df(self):
        # 输出文件
        file_name = f'{self.output_prefix}prediction_dataframe.txt'
        print('output:', file_name)
        self.prediction_df.to_csv(file_name, sep='\t', index=False)

    def output_dataset_prediction_df(self):
        # 输出文件
        file_name = f'{self.output_prefix}dataset_prediction_dataframe.txt'
        print('output:', file_name)
        self.dataset_prediction_df.to_csv(file_name, sep='\t', index=False)

    def select_output_dataset_prediction_df(self, dataset_index: int):
        coordinate_col_list = self.dataset_prediction_df.columns[0: 3].tolist()
        col_postfix = f'_{dataset_index}'
        select_col_list = coordinate_col_list + [col for col in self.dataset_prediction_df.columns.tolist()
                                                 if col.endswith(col_postfix)]
        output_dataset_prediction_df = self.dataset_prediction_df.loc[:, select_col_list]
        # 输出文件
        file_name = f'{self.output_prefix}dataset_{dataset_index}_evaluation_dataframe.txt'
        print('output:', file_name)
        output_dataset_prediction_df.to_csv(file_name, sep='\t', index=False)

    def output_prediction_bedgraph_format(self):
        # check bedgraph folder
        check_output_folder(self.output_bedgraph_folder)
        # prediction_df: 排序并删除完全重复的行
        self.prediction_df.sort_values(by=['chr', 'start'], inplace=True)
        self.prediction_df.drop_duplicates(inplace=True)
        # 输出bedgraph格式，前3列是坐标，因此从3开始遍历
        for col_index in range(3, len(self.prediction_df.columns)):
            output_bed_df = self.prediction_df.iloc[:, [0, 1, 2, col_index]]
            file_name = f'{self.output_bedgraph_folder}/{self.prediction_df.columns[col_index]}.bedgraph'
            print('output:', file_name)
            output_bed_df.to_csv(file_name, sep='\t', index=False, header=False)

