import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import datetime
from captum.attr import DeepLiftShap as CaptumAlgorithm
from MethylAI.src.dataset.captum_dataset import CaptumDataset

class AttributionTools:
    def __init__(self, model: nn.Module, model_stat_file: str, gpu_number: int,
                 genome_fasta_file: str, model_input_dna_length: int,
                 captum_target_file: str, n_permutation: int,
                 print_per_step: int,
                 output_folder: str, output_prefix: str,
                 ):
        # 读入模型结构、载入参数、使用GPU
        self.model = model
        self.device = torch.device(gpu_number)
        self.model.to(self.device)
        self.model_stat_file = model_stat_file
        self._load_model_stat(self.model_stat_file)
        # 加载captum算法
        self.model.eval()
        self.captum_algorithm = CaptumAlgorithm
        self.captum_model = self.captum_algorithm(self.model)
        # captum target & captum cg & running_number
        self.captum_target_file = captum_target_file
        self.captum_target_df = pd.DataFrame()
        self.captum_cg_df = pd.DataFrame()
        self.captum_target_i = 99999
        self.captum_target_name = ''
        self.captum_running_number = 0
        self.captum_reset_number = 10_0000
        # 设置
        self.genome_fasta_file = genome_fasta_file
        self.n_permutation = n_permutation
        # 保存结果文件夹
        self.output_folder = f'{output_folder}'
        self.output_prefix = output_prefix
        self.output_captum_folder = ''
        self.bedgraph_folder = ''
        self.numpy_folder = ''
        # 保存attribution结果
        self.dna_one_hot_list = []
        self.attribution_list = []
        self.dna_attribution_list = []
        self.delta_list = []
        # dna长度
        self.model_input_dna_length = model_input_dna_length
        self._input_data()
        # 多少步print一次
        self.print_per_step = print_per_step

    def _check_output_folder(self):
        def mkdir(folder):
            if not os.path.exists(folder):
                print(f'mkdir {folder}')
                os.makedirs(folder)
        mkdir(self.output_captum_folder)
        mkdir(self.bedgraph_folder)
        mkdir(self.numpy_folder)

    def _load_model_stat(self, model_stat_file):
        print('load model stat:', model_stat_file)
        all_state = torch.load(model_stat_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(all_state['self.model'])

    def _input_data(self):
        print('_input_data')
        # 读取self.captum_target_df
        self.captum_target_df = pd.read_table(self.captum_target_file, header=0)

    def _input_captum_cg_file(self, captum_cg_file: str):
        print('_input_captum_cg_file')
        # 读取captum_cg_file
        self.captum_cg_df = pd.read_table(captum_cg_file, header=0)
        self.captum_cg_df = self.captum_cg_df[self.captum_cg_df['represent_cg'] == 1].copy()
        self.captum_cg_df.sort_values(by=['chr', 'start'], inplace=True, ignore_index=True)
        # 计算输入dna的坐标
        dataset_length = self.captum_cg_df.loc[0, 'end'] - self.captum_cg_df.loc[0, 'start']
        dataset_extent_length = dataset_length // 2
        model_extent_length = (self.model_input_dna_length // 2) - dataset_extent_length
        self.captum_cg_df.loc[:, 'input_dna_start'] = self.captum_cg_df.loc[:, 'start'] - model_extent_length
        self.captum_cg_df.loc[:, 'input_dna_end'] = self.captum_cg_df.loc[:, 'end'] + model_extent_length

    def output_captum_cg_df(self, output_file: str):
        output_file = f'{self.output_captum_folder}/{output_file}'
        self.captum_cg_df.to_csv(output_file, sep='\t')

    def iter_captum_target_df(self, captum_cg_file_folder: str, captum_target_from, captum_target_to,
                              is_output_bedgraph: bool):
        self.captum_target_df = self.captum_target_df.iloc[captum_target_from: captum_target_to, :]
        for _, row in self.captum_target_df.iterrows():
            row_index = row['row_index']
            organ_tissue = row['Organ_Tissue'].lower().replace(' ', '_')
            captum_target_i = row['model_output_index']
            # captum_cg_file & captum_cg_dataset
            captum_cg_file = f'{captum_cg_file_folder}/{row_index}_{organ_tissue}_low_me_region_represent_cg.txt'
            self._input_captum_cg_file(captum_cg_file)
            captum_dataset = CaptumDataset(
                captum_dataset_dataframe=self.captum_cg_df,
                genome_fasta_file_name=self.genome_fasta_file,
                n_permutation=self.n_permutation,
            )
            print(f'captum_target_name: {organ_tissue}, captum_target_i: {captum_target_i}')
            self.iter_captum_dataset(
                captum_dataset=captum_dataset,
                captum_target_name=organ_tissue,
                captum_target_i=captum_target_i,
                is_output_bedgraph=is_output_bedgraph,
            )

    def iter_captum_dataset(
            self, captum_dataset: CaptumDataset, captum_target_name: str, captum_target_i: int, is_output_bedgraph: bool,
    ):
        print(f'target: {captum_target_i}')
        # 外部index从1开始，python index从0开始，因此减去1
        true_target_i = captum_target_i - 1
        # 设置文件夹
        self.output_captum_folder = f'{self.output_folder}/{captum_target_name}_target{captum_target_i}'
        self.bedgraph_folder = f'{self.output_captum_folder}/bedgraph'
        self.numpy_folder = f'{self.output_captum_folder}/numpy'
        self._check_output_folder()
        # 输出captum_cg_df
        self.output_captum_cg_df('captum_cg_df.txt')
        # captum_dataloader
        captum_dataloader = DataLoader(
            dataset=captum_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        # 记录开始时间
        start_time = datetime.datetime.now()
        # for循环计算所有的序列
        for row_i, (dna_one_hot_tensor, base_line_tensor) in enumerate(captum_dataloader):
            dna_one_hot_gpu_tensor = dna_one_hot_tensor[0].to(self.device)
            base_line_gpu_tensor = base_line_tensor[0].to(self.device)
            # captum
            attribution, delta = self.captum_model.attribute(
                inputs=dna_one_hot_gpu_tensor,
                baselines=base_line_gpu_tensor,
                target=true_target_i,
                return_convergence_delta=True,
            )
            # 计算dna_attribution
            dna_attribution_tensor = torch.squeeze(attribution * dna_one_hot_gpu_tensor)
            dna_attribution_tensor = torch.sum(dna_attribution_tensor, dim=0)
            dna_attribution_numpy = dna_attribution_tensor.detach().cpu().numpy()
            # 把attribution和dna_one_hot转化为numpy，并把序列顺序从ATGC转换为ACGT
            attribution_numpy = torch.squeeze(attribution.detach().cpu()).numpy()[[0, 3, 2, 1], :]
            dna_one_hot_numpy = torch.squeeze(dna_one_hot_tensor).numpy()[[0, 3, 2, 1], :]
            delta_numpy = delta.detach().cpu().numpy()
            # 以bedgraph形式输出
            if is_output_bedgraph:
                bedgraph_output_prefix = f'{self.bedgraph_folder}/target{self.captum_target_i}'
                chr_number = self.captum_cg_df.loc[row_i, 'chr']
                dna_start_position = self.captum_cg_df.loc[row_i, 'input_dna_start']
                dna_end_position = self.captum_cg_df.loc[row_i, 'input_dna_end']
                self.attribution_to_bedgraph(bedgraph_output_prefix, chr_number, dna_start_position, dna_end_position,
                                             dna_attribution_numpy)
            # 把结果保存到list
            self.dna_attribution_list.append(dna_attribution_numpy)
            self.attribution_list.append(attribution_numpy)
            self.dna_one_hot_list.append(dna_one_hot_numpy)
            self.delta_list.append(delta_numpy)
            # 每self.print_per_step打印一次结果
            if row_i % self.print_per_step == 0:
                using_time = datetime.datetime.now() - start_time
                print(f'captum_target_name: {captum_target_name}, captum_target_i: {captum_target_i}')
                print(f'using_time: {using_time}')
                print(f'row index: {row_i:5d}|{len(captum_dataloader):5d}')
                print(f'delta: {delta_numpy}\n')
            # 记录captum次数，达到一定次数后重置captum_model
            self.record_captum_running()
        # captum完成，总结并输出
        self.output_and_clean_attribution_numpy()

    def output_and_clean_attribution_numpy(self):
        # captum完成，保存、输出attribution结果，并清空对应list
        output_attribution_file = f'{self.numpy_folder}/attribution.npy'
        output_dna_one_hot_file = f'{self.numpy_folder}/dna_one_hot.npy'
        output_dna_attribution_file = f'{self.numpy_folder}/dna_attribution.npy'
        output_delta_file = f'{self.numpy_folder}/delta.npy'
        self.save_numpy(self.attribution_list, output_attribution_file)
        self.save_numpy(self.dna_one_hot_list, output_dna_one_hot_file)
        self.save_numpy(self.dna_attribution_list, output_dna_attribution_file)
        self.save_numpy(self.delta_list, output_delta_file)
        self.dna_one_hot_list = []
        self.attribution_list = []
        self.dna_attribution_list = []
        self.delta_list = []

    def save_numpy(self, numpy_list: list, output_file: str):
        output_numpy = np.stack(numpy_list)
        print(f'output: {output_file}')
        np.save(file=output_file, arr=output_numpy)

    def attribution_to_bedgraph(self, output_prefix, chr_number, dna_start_position, dna_end_position, dna_attribution_numpy):
        attribution_bedgraph = pd.DataFrame({
            'chr': chr_number,
            'start': range(dna_start_position, dna_end_position),
            'end': range(dna_start_position+1, dna_end_position+1),
            'attribution': dna_attribution_numpy
        })
        # 输出文件的文件名
        cpg_start_position = dna_start_position + (self.model_input_dna_length // 2) - 1
        output_bedgraph_file_name = f'{output_prefix}_{chr_number}_{dna_start_position}_{cpg_start_position}.bedgraph'
        attribution_bedgraph.to_csv(output_bedgraph_file_name, sep='\t', header=False, index=False)

    def record_captum_running(self):
        # 记录captum次数，达到一定次数后重置captum_model
        self.captum_running_number = self.captum_running_number + 1
        if self.captum_running_number % self.captum_reset_number == 0:
            self.captum_model = self.captum_algorithm(self.model)