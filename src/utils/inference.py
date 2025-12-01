import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import gc
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
        # 保存参数
        self.output_folder = output_folder
        check_output_folder(self.output_folder)
        self.output_bedgraph_folder = f'{self.output_folder}/bedgraph'
        if output_prefix:
            self.output_prefix = f'{output_folder}/{output_prefix}_'
        else:
            self.output_prefix = f'{output_folder}/'
        #################################################################
        # 这部分用于计算cpg embedding等
        # dataset_dataframe
        self.dataset_dataframe = pd.DataFrame()
        # prediction_dataframe & cpg_embedding_dataframe
        self.prediction_dataframe = pd.DataFrame()
        self.cpg_embedding_dataframe = pd.DataFrame()
        # prediction_list & cpg_embedding_list
        self.prediction_list = []
        self.cpg_embedding_list = []

    def _load_model_state(self, model_stat_file):
        print(f'load model stat: {model_stat_file}')
        all_state = torch.load(model_stat_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(all_state['self.model'])

    def generate_prediction_dataframe_header(self, sample_index_file,
                                             col_prefix_tuple = ('smooth', 'raw', 'window_1000', 'window_500', 'window_200')):
        # 读取sample_info_dataframe，产生表头
        sample_info_dataframe = pd.read_table(sample_index_file, sep='\t', header=0)
        keep_index = sample_info_dataframe[sample_info_dataframe['keep'] == 'Yes'].index
        col_index_list = sample_info_dataframe.loc[keep_index, 'index'].tolist()
        prediction_dataframe_header_list = []
        for col_prefix in col_prefix_tuple:
            prediction_dataframe_header_list.extend([f'prediction_{col_prefix}_{index}' for index in col_index_list])
        return prediction_dataframe_header_list

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
        # 获取evaluation_dataset里的dataframe，存放到self.dataset_dataframe
        self.dataset_dataframe = inference_dataset.get_dataset_df()
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
        gc.collect()

    def generate_prediction_dataframe(self):
        print('generate_dataset_prediction_dataframe')
        # 产生self.prediction_dataframe
        prediction_numpy = np.concatenate(self.prediction_list, axis=0)
        prediction_dataframe = pd.DataFrame(prediction_numpy)
        prediction_dataframe_col_name_list = [
            f'prediction_{index}' for index in range(prediction_numpy.shape[1])
        ]
        prediction_dataframe.columns = prediction_dataframe_col_name_list
        # 为prediction_dataframe拼接self.dataset_dataframe的前4列（基因组坐标信息）
        self.prediction_dataframe = pd.concat(
            [self.dataset_dataframe.iloc[:, 0:3], prediction_dataframe], axis=1
        )

    def generate_cpg_embedding_dataframe(self):
        print('generate_cpg_embedding_dataframe')
        # 产生self.cpg_embedding_dataframe
        cpg_embedding_numpy = np.concatenate(self.cpg_embedding_list, axis=0)
        cpg_embedding_dataframe = pd.DataFrame(cpg_embedding_numpy)
        cpg_embedding_col_name_list = [
            f'embedding_{index}'  for index in range(cpg_embedding_numpy.shape[1])
        ]
        cpg_embedding_dataframe.columns = cpg_embedding_col_name_list
        # 为cpg_embedding_col_name_list拼接self.dataset_dataframe的前4列（基因组坐标信息）
        self.cpg_embedding_dataframe = pd.concat(
            [self.dataset_dataframe.iloc[:, 0:3], cpg_embedding_dataframe], axis=1
        )

    def output_cpg_embedding_dataframe(self):
        # 输出文件
        file_name = f'{self.output_prefix}cpg_embedding_dataframe.txt'
        print('output:', file_name)
        self.cpg_embedding_dataframe.to_csv(file_name, sep='\t', index=False)

    def output_prediction_dataframe(self):
        # 输出文件
        file_name = f'{self.output_prefix}prediction_dataframe.txt'
        print('output:', file_name)
        self.prediction_dataframe.to_csv(file_name, sep='\t', index=False)

    def output_prediction_bedgraph_format(self):
        # check bedgraph folder
        check_output_folder(self.output_bedgraph_folder)
        # prediction_dataframe: 排序并删除完全重复的行
        self.prediction_dataframe.sort_values(by=['chr', 'start'], inplace=True)
        self.prediction_dataframe.drop_duplicates(inplace=True)
        # 输出bedgraph格式，前4列是坐标，因此从4开始遍历
        for col_index in range(3, len(self.prediction_dataframe.columns)):
            output_bed_dataframe = self.prediction_dataframe.iloc[:, [0, 1, 2, col_index]]
            file_name = f'{self.output_bedgraph_folder}/{self.prediction_dataframe.columns[col_index]}.bedgraph'
            print('output:', file_name)
            output_bed_dataframe.to_csv(file_name, sep='\t', index=False, header=False)

