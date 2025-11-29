import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import gc
import subprocess
import datetime

from ..dataset.validation_dataset import MethylAIValidationDataset

class EvaluationTools:
    def __init__(self, model: nn.Module, model_stat_file: str, gpu_number: int, reverse_complement_augmentation: bool,
                 sample_info_file, chrom_size_file, print_per_step: int,
                 bedgraph_to_bigwig_script, output_prefix, output_folder,):
        # 读入模型结构
        self.model_stat_file = model_stat_file
        self.model = model
        # 把模型移到GPU和载入参数
        self.device = torch.device(gpu_number)
        self.model.to(self.device)
        self._load_model_stat(self.model_stat_file)
        # 负链增强
        self.reverse_complement_augmentation = reverse_complement_augmentation
        # print per step
        self.print_per_step = print_per_step
        # sample_info_file
        self.sample_info_file = sample_info_file
        # 脚本
        self.bedgraph_to_bigwig_script = bedgraph_to_bigwig_script
        self.chrom_size_file = chrom_size_file
        # 保存参数
        self.output_folder = output_folder
        self.output_bedgraph_folder = os.path.join(self.output_folder, 'bedgraph')
        self._check_output_folder()
        self.output_prefix = os.path.join(output_folder, output_prefix)
        #################################################################
        # 这部分用于计算cpg embedding等
        # dataset_dataframe
        self.dataset_dataframe = pd.DataFrame()
        # dataset_dataframe+prediction_dataframe
        self.dataset_prediction_dataframe = pd.DataFrame()
        # true dataframe
        self.true_dataframe = pd.DataFrame()
        # prediction_dataframe & cpg_embedding_dataframe
        self.prediction_dataframe = pd.DataFrame()
        self.cpg_embedding_dataframe = pd.DataFrame()
        # prediction_list & cpg_embedding_list
        self.prediction_list = []
        self.cpg_embedding_list = []

    def _check_output_folder(self):
        if not os.path.exists(self.output_folder):
            print('mkdir', self.output_folder)
            os.makedirs(self.output_folder)
        if not os.path.exists(self.output_bedgraph_folder):
            print('mkdir', self.output_bedgraph_folder)
            os.makedirs(self.output_bedgraph_folder)

    def _load_model_stat(self, model_stat_file):
        print('load model stat:', model_stat_file)
        all_state = torch.load(model_stat_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(all_state['self.model'])

    def get_true_value_col_index(self, true_value_col_prefix_list: list):
        print('get_true_value_col_index')
        print(true_value_col_prefix_list)
        # 获取true value的列
        validation_true_value_col_index = [
            index for index, col_name in enumerate(self.dataset_dataframe.columns) if
            col_name.startswith(tuple(true_value_col_prefix_list))
        ]
        return validation_true_value_col_index

    def generate_prediction_dataframe_header(self, col_prefix_tuple = ('smooth', 'raw', 'window_1000', 'window_500', 'window_200')):
        print('generate_prediction_dataframe_header')
        # 读取sample_info_dataframe，产生表头
        sample_info_dataframe = pd.read_table(self.sample_info_file, sep='\t', header=0)
        keep_index = sample_info_dataframe[sample_info_dataframe['keep'] == 'Yes'].index
        col_index_list = sample_info_dataframe.loc[keep_index, 'index'].tolist()
        prediction_dataframe_header_list = []
        for col_prefix in col_prefix_tuple:
            prediction_dataframe_header_list.extend([f'prediction_{col_prefix}_{index}' for index in col_index_list])
        return prediction_dataframe_header_list

    def generate_prediction_and_embedding_list(
            self, evaluation_dataset: MethylAIValidationDataset, batch_size=250, num_workers=8,
    ):
        print('generate_prediction_and_embedding_list')
        # 创建dataloader
        evaluation_dataloader = DataLoader(
            dataset=evaluation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # 获取evaluation_dataset里的dataframe，存放到self.dataset_dataframe
        self.dataset_dataframe = evaluation_dataset.get_dataset_dataframe()
        # 把self.dataset_dataframe坐标调整到中间的CG
        seq_length = self.dataset_dataframe.iloc[0, 2] - self.dataset_dataframe.iloc[0, 1]
        seq_length = (seq_length - 1) // 2
        self.dataset_dataframe.iloc[:, 1] = self.dataset_dataframe.iloc[:, 1] + seq_length
        self.dataset_dataframe.iloc[:, 2] = self.dataset_dataframe.iloc[:, 2] - seq_length
        # 记录总步数
        total_step = len(evaluation_dataloader)
        # 把模型设置为evaluation模式
        self.model.eval()
        # 记录开始时间
        evaluation_start_time = datetime.datetime.now()
        # 开始运行
        # self.reverse_complement_augmentation == True
        if self.reverse_complement_augmentation:
            with torch.no_grad():
                for batch_index, (forward_dna_one_hot, reverse_dna_one_hot, methylation_level, loss_weight
                                  ) in enumerate(evaluation_dataloader):
                    forward_dna_sequence_one_hot_tensor = forward_dna_one_hot.to(self.device)
                    reverse_dna_sequence_one_hot_tensor = reverse_dna_one_hot.to(self.device)
                    # 计算output
                    forward_cpg_embedding_tensor, forward_output_tensor = self.model(forward_dna_sequence_one_hot_tensor)
                    reverse_cpg_embedding_tensor, reverse_output_tensor = self.model(reverse_dna_sequence_one_hot_tensor)
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
        # self.reverse_complement_augmentation == False
        else:
            with torch.no_grad():
                for batch_index, (dna_sequence_one_hot_encoding, methylation_level, loss_weight) in enumerate(
                        evaluation_dataloader):
                    dna_sequence_one_hot_encoding_tensor = dna_sequence_one_hot_encoding.to(self.device)
                    cpg_embedding_tensor, output_tensor = self.model(dna_sequence_one_hot_encoding_tensor)
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

    def generate_dataset_prediction_dataframe(
            self, true_col_prefix=('smooth', 'raw', 'window'), output_prediction_index_tuple: tuple = None
    ):
        print('generate_dataset_prediction_dataframe')
        # 获取true_dataframe：包含坐标和数据，删除不必要的内容，并用于后续产生bedgraph
        true_col_index = self.get_true_value_col_index(true_col_prefix)
        self.true_dataframe = self.dataset_dataframe.iloc[:, [0, 1, 2, 3] + true_col_index].copy()
        # 产生self.prediction_dataframe
        prediction_numpy = np.concatenate(self.prediction_list, axis=0)
        prediction_dataframe = pd.DataFrame(prediction_numpy)
        prediction_dataframe_col_name_list = self.generate_prediction_dataframe_header()
        prediction_dataframe.columns = prediction_dataframe_col_name_list
        # 如果选择某些index输出
        if output_prediction_index_tuple:
            output_prediction_col_list = [
                col for col in prediction_dataframe_col_name_list if col.endswith(output_prediction_index_tuple)
            ]
            prediction_dataframe = prediction_dataframe[output_prediction_col_list]
        # 为prediction_dataframe拼接self.dataset_dataframe的前4列（基因组坐标信息）
        self.prediction_dataframe = pd.concat(
            [self.true_dataframe.iloc[:, 0:4], prediction_dataframe], axis=1
        )
        # 产生self.dataset_prediction_dataframe
        self.dataset_prediction_dataframe = pd.concat(
            [self.dataset_dataframe, prediction_dataframe], axis=1
        )

    def generate_cpg_embedding_dataframe(self):
        print('generate_cpg_embedding_dataframe')
        # 产生self.cpg_embedding_dataframe
        cpg_embedding_numpy = np.concatenate(self.cpg_embedding_list, axis=0)
        cpg_embedding_dataframe = pd.DataFrame(cpg_embedding_numpy)
        cpg_embedding_col_name_list = [
            'embedding_' + str(num) for num in list(range(1, cpg_embedding_numpy.shape[1] + 1))
        ]
        cpg_embedding_dataframe.columns = cpg_embedding_col_name_list
        # 为cpg_embedding_col_name_list拼接self.dataset_dataframe的前4列（基因组坐标信息）
        self.cpg_embedding_dataframe = pd.concat(
            [self.true_dataframe.iloc[:, 0:4], cpg_embedding_dataframe], axis=1
        )

    def output_dataset_prediction_dataframe(self):
        # 输出文件
        file_name = f'{self.output_prefix}_dataset_prediction_dataframe.txt'
        print('output:', file_name)
        self.dataset_prediction_dataframe.to_csv(file_name, sep='\t', index=False)

    def sort_and_drop_duplicates_dataframes(self):
        # prediction_dataframe: 排序并删除完全重复的行
        self.prediction_dataframe.sort_values(by=['chr', 'start'], inplace=True)
        self.prediction_dataframe.drop_duplicates(inplace=True)
        # true_dataframe: 排序并删除完全重复的行
        self.true_dataframe.sort_values(by=['chr', 'start'], inplace=True)
        self.true_dataframe.drop_duplicates(inplace=True)
        # cpg_embedding: 排序并删除完全重复的行
        self.cpg_embedding_dataframe.sort_values(by=['chr', 'start'], inplace=True)
        self.cpg_embedding_dataframe.drop_duplicates(inplace=True)

    def output_cpg_embedding_dataframe(self):
        # 输出文件
        file_name = f'{self.output_prefix}_cpg_embedding_dataframe.txt'
        print('output:', file_name)
        self.cpg_embedding_dataframe.to_csv(file_name, sep='\t', index=False)

    def output_true_and_prediction_dataframe(self):
        print('output_true_and_prediction_dataframe')
        # 该方法输出1个true_dataframe和1个prediction_dataframe，全部数据都在这2个文件里面
        # 先处理self.dataset_dataframe
        # 输出文件
        file_name = f'{self.output_prefix}_true_dataframe.txt'
        print('output:', file_name)
        self.dataset_dataframe.to_csv(file_name, sep='\t', index=False)
        # 再处理self.prediction_dataframe
        # 输出文件
        file_name = f'{self.output_prefix}_prediction_dataframe.txt'
        print('output:', file_name)
        self.prediction_dataframe.to_csv(file_name, sep='\t', index=False)

    def output_true_and_prediction_dataframe_2(self, output_prefix=None):
        # 该方法把self.dataset_dataframe和self.prediction_dataframe对应的列拼接到一起产生1个文件并输出，因此有多少列就输出多少个文件
        print('output_true_and_prediction_dataframe_2')
        # 设置输出文件名
        if output_prefix:
            output_prefix = self.output_prefix + output_prefix
        else:
            output_prefix = self.output_prefix
        # 断言2个数据框性状相同
        print(self.dataset_dataframe.shape)
        print(self.prediction_dataframe.shape)
        assert self.dataset_dataframe.shape == self.prediction_dataframe.shape
        # 遍历self.dataset_dataframe的列，0,1,2,3储存坐标信息，因此从第4列开始遍历，每列都产生1个新文件
        for col_index in range(4, self.dataset_dataframe.shape[1], 1):
            output_dataframe = pd.concat(
                [self.dataset_dataframe.iloc[:, [0, 1, 2, 3, col_index]], self.prediction_dataframe.iloc[:, col_index]], axis=1
            )
            col_name = self.dataset_dataframe.columns[col_index]
            file_name = output_prefix + col_name + '.txt'
            print('output:', file_name)
            output_dataframe.to_csv(file_name, sep='\t', index=False)

    def output_prediction_bedgraph_format(self):
        # 输出bedgraph格式，前4列是坐标，因此从4开始遍历
        for col_index in range(4, len(self.prediction_dataframe.columns)):
            output_bed_dataframe = self.prediction_dataframe.iloc[:, [0, 1, 2, col_index]]
            file_name = f'{self.output_bedgraph_folder}/{self.prediction_dataframe.columns[col_index]}.bedgraph'
            print('output:', file_name)
            output_bed_dataframe.to_csv(file_name, sep='\t', index=False, header=False)

    def output_true_bedgraph_format(self):
        # 输出bedgraph格式，前4列是坐标，因此从4开始遍历
        for col_index in range(4, len(self.true_dataframe.columns)):
            output_bed_dataframe = self.true_dataframe.iloc[:, [0, 1, 2, col_index]]
            file_name = f'{self.output_bedgraph_folder}/{self.true_dataframe.columns[col_index]}.bedgraph'
            print('output:', file_name)
            output_bed_dataframe.to_csv(file_name, sep='\t', index=False, header=False)

    def output_difference_bedgraph_format(self):
        # 获取前3列
        validation_difference_dataframe = self.true_dataframe.iloc[:, 0:3].copy()
        # 计算差值并取绝对值
        col_name_list = ['difference_' + col_name for col_name in self.true_dataframe.columns[4:]]
        validation_difference_dataframe[col_name_list] = np.absolute(
            self.true_dataframe.iloc[:, 4:].to_numpy() - self.prediction_dataframe.iloc[:, 4:].to_numpy()
        )
        # 需要输出的列
        difference_col_index = [
            index for index, col_name in enumerate(validation_difference_dataframe.columns)
            if col_name.startswith('difference_')
        ]
        # 输出文件
        for col_index in difference_col_index:
            output_bed_dataframe = validation_difference_dataframe.iloc[:, [0, 1, 2, col_index]]
            file_name = f'{self.output_bedgraph_folder}/{validation_difference_dataframe.columns[col_index]}.bedgraph'
            print('output:', file_name)
            output_bed_dataframe.to_csv(file_name, sep='\t', index=False, header=False)

    def convert_bedgraph_to_bigwig(self):
        # 获取当前路径
        current_path = os.getcwd()
        bedgraph_folder = f'{current_path}/{self.output_bedgraph_folder}'
        bigwig_folder = bedgraph_folder.replace('/bedgraph', '/bigwig')
        if not os.path.exists(bigwig_folder):
            print(f'mkdir {bigwig_folder}')
            os.mkdir(bigwig_folder)
        # bedgraph to bigwig
        bedgraph_to_bigwig_command = f'sh {self.bedgraph_to_bigwig_script} {bedgraph_folder} {bigwig_folder} {self.chrom_size_file}'
        print(bedgraph_to_bigwig_command)
        subprocess.run(bedgraph_to_bigwig_command, shell=True)

    def process_cell_type_result(self, cell_type_file: str, max_predict_error_tuple: tuple, output_file: str):
        cell_type_dataframe = pd.read_table(cell_type_file, header=0, sep='\t')
        # 初始化numpy储存正确率，行数与cell_type_dataframe相同，列数与max_predict_error_tuple长度相同
        # all为所有染色体，validation为chr10
        all_correct_rate_numpy = np.zeros((len(cell_type_dataframe), len(max_predict_error_tuple)))
        validation_correct_rate_numpy = np.zeros((len(cell_type_dataframe), len(max_predict_error_tuple)))
        for row_index, row in cell_type_dataframe.iterrows():
            cell_type = row['cell_type']
            cell_type_dataset_prediction_dataframe = self.dataset_prediction_dataframe[
                self.dataset_prediction_dataframe['cell_type'] == cell_type
                ]
            col_postfix = row['target']
            true_col = f'smooth_{col_postfix}'
            prediction_col = f'prediction_smooth_{col_postfix}'
            abs_diff_series = (
                cell_type_dataset_prediction_dataframe[true_col] - cell_type_dataset_prediction_dataframe[prediction_col]
            ).abs()
            validation_index = cell_type_dataset_prediction_dataframe[cell_type_dataset_prediction_dataframe['chr'] == 'chr10'].index
            validation_abs_diff_series = abs_diff_series[validation_index]
            for i in range(len(max_predict_error_tuple)):
                max_predict_error = max_predict_error_tuple[i]
                # 所有染色体
                all_correct_series = abs_diff_series <= max_predict_error
                all_correct_rate = all_correct_series.sum() / len(all_correct_series)
                all_correct_rate_numpy[row_index, i] = all_correct_rate
                # chr10
                validation_correct_series = validation_abs_diff_series <= max_predict_error
                validation_correct_correct_rate = validation_correct_series.sum() / len(validation_correct_series)
                validation_correct_rate_numpy[row_index, i] = validation_correct_correct_rate
        # for循环结束，汇总结果
        all_correct_dataframe = pd.DataFrame(all_correct_rate_numpy)
        all_correct_dataframe.columns = [f'all_correct_rate_{max_predict_error}' for max_predict_error in
                                         max_predict_error_tuple]
        validation_correct_dataframe = pd.DataFrame(validation_correct_rate_numpy)
        validation_correct_dataframe.columns = [f'validation_correct_rate_{max_predict_error}' for max_predict_error in
                                                max_predict_error_tuple]
        cell_type_dataframe = pd.concat([cell_type_dataframe, all_correct_dataframe, validation_correct_dataframe], axis=1)
        output_file = f'{self.output_folder}/{output_file}'
        cell_type_dataframe.to_csv(output_file, sep='\t', index=False)

