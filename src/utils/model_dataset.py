import os
import pandas as pd
import polars as pl
import numpy as np
import datetime
from typing import Literal
import sys
from pathlib import Path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))
from MethylAI.src.utils.genome_fasta import GenomeFasta
from MethylAI.src.utils.utils import check_output_folder, debug_methods

@debug_methods
class MethylationDataset:
    def __init__(self, methylation_file: str, data_info_file: str, genome_fasta_file: str, chromosome_size_file: str,
                 minimal_coverage: int, model_input_dna_length: int,
                 output_folder: str, output_prefix: str, is_quiet: bool = False,):
        # methylation data
        self.methylation_file = methylation_file
        self.methylation_df = pd.DataFrame()
        # 保存每条染色体长度
        self.chromosome_size_file = chromosome_size_file
        self.chromosome_size_df = pd.DataFrame()
        # sample info 文件名和dataframe
        self.data_info_file = data_info_file
        self.data_info_df = pd.DataFrame()
        # input data
        self._input_data()
        # 基因组文件
        self.genome_fasta = GenomeFasta(genome_fasta_file)
        # setting
        self.minimal_coverage = minimal_coverage
        self.is_verbose = not is_quiet
        # 输出文件夹
        self.output_folder = output_folder
        check_output_folder(self.output_folder)
        self.output_prefix = f'{self.output_folder}/{output_prefix}'
        # 保存输入模型的序列长度
        self.model_input_dna_length = model_input_dna_length
        self.cpg_length = 2

    def _input_data(self):
        # methylation dataset
        self.methylation_df = pl.read_csv(
            self.methylation_file, separator='\t', null_values=['NA'], infer_schema_length=100000
        ).to_pandas()
        self.methylation_df['start'] = self.methylation_df['start'].astype(int)
        self.methylation_df['end'] = self.methylation_df['end'].astype(int)
        self.cpg_length = self.methylation_df.loc[0, 'end'] - self.methylation_df.loc[0, 'start']
        # data info
        self.data_info_df = pd.read_table(self.data_info_file, sep='\t', header=0)
        # chrom size
        self.chromosome_size_df = pd.read_table(self.chromosome_size_file, sep='\t', header=None)
        self.chromosome_size_df.columns = ['chr', 'chr_length']

    def methylation_dataframe_drop_sample(self, max_low_coverage_ratio=0.5, file_name=None):
        # 计算self.methylation_df，以'coverage_'开头列 < self.minimal_coverage的个数
        coverage_df = self.methylation_df.filter(regex='^coverage_')
        low_coverage_series = coverage_df.apply(lambda x: (x < self.minimal_coverage).sum())
        low_coverage_series.reset_index(drop=True, inplace=True)
        self.data_info_df[f'coverage_lower_{self.minimal_coverage}'] = low_coverage_series
        # 计算low_coverage最大阈值，超过max_low_coverage_threshold的样品标记为不保留(keep=No)
        max_low_coverage_threshold = int(self.methylation_df.shape[0] * max_low_coverage_ratio)
        self.data_info_df['is_pass_qc'] = self.data_info_df[f'coverage_lower_{self.minimal_coverage}'].apply(
            lambda x: 'yes' if x < max_low_coverage_threshold else 'no'
        )
        # 保留methylation_dataframe前4列 + 保留keep=Yes的样本
        self.data_info_df['dataset_index'] = self.data_info_df['dataset_index'].astype(str)
        keep_col_postfix_list = self.data_info_df['dataset_index'][
            self.data_info_df['is_pass_qc'] == 'yes'].to_list()
        keep_col_postfix_list = [f'_{index}' for index in keep_col_postfix_list]
        keep_col_list = self.methylation_df.columns[
            self.methylation_df.columns.str.endswith(tuple(keep_col_postfix_list))
        ].to_list()
        keep_col_list = self.methylation_df.columns[0:3].to_list() + keep_col_list
        self.methylation_df = self.methylation_df[keep_col_list]
        # 处理data_info_df，产生new_index，keep=Yes产生从1开始的下标，keep=No则赋值-1
        self.data_info_df['model_output_index'] = (self.data_info_df['is_pass_qc'] == 'yes').cumsum() - 1
        no_keep_index = self.data_info_df[self.data_info_df['is_pass_qc'] == 'no'].index
        self.data_info_df.loc[no_keep_index, 'model_output_index'] = -1
        # 设置输出文件名
        if file_name:
            file_name = f'{self.output_prefix}_{file_name}'
        else:
            file_name = f'{self.output_prefix}_data_info.txt'
        # 输出统计结果
        print('output:', file_name)
        self.data_info_df.to_csv(file_name, sep='\t', index=False)

    def calculate_regional_methylation(self, window_size_list = None):
        if window_size_list is None:
            window_size_list = [1000, 500, 200]
        else:
            window_size_list = [item for item in window_size_list if isinstance(item, int) and not isinstance(item, bool)]
            window_size_list = sorted(window_size_list, reverse=True)
        # 延伸长度
        extend_length_list = [(size - self.cpg_length) // 2 for size in window_size_list]
        # smooth_methylation_dataframe中存放raw_methylation的列index
        raw_methylation_col_index_list = [
            index for index, col_name in enumerate(self.methylation_df.columns) if col_name.startswith('raw_')
        ]
        raw_methylation_col_name_list = [
            col_name for col_name in self.methylation_df.columns if col_name.startswith('raw_')
        ]
        # smooth_methylation_dataframe中存放coverage的列index
        coverage_col_index = [
            index for index, col_name in enumerate(self.methylation_df.columns) if col_name.startswith('coverage_')
        ]
        # 初始化嵌套list存储window methylation
        window_methylation_list = [[] for _ in range(len(extend_length_list))]
        # 记录开始时间
        window_methylation_start_time = datetime.datetime.now()
        # 第1层
        for chr_number in pd.unique(self.methylation_df['chr']):
            chr_methylation_df = self.methylation_df[self.methylation_df['chr'] == chr_number]
            # 第2层
            for row_index in range(len(chr_methylation_df)):
                # cpg位点的坐标
                cpg_start = chr_methylation_df.iloc[row_index, 1]
                cpg_end = chr_methylation_df.iloc[row_index, 2]
                window_start = cpg_start - extend_length_list[0]
                window_end = cpg_end + extend_length_list[0]
                # window坐标对应的row_index（此处是最大的window）
                max_window_row_index = chr_methylation_df[
                    (chr_methylation_df['start'] > window_start) &
                    (chr_methylation_df['end'] < window_end)
                ].index
                # 对chr_methylation_df进行切片
                max_window_chr_methylation_df = chr_methylation_df.loc[max_window_row_index, :]
                raw_methylation_numpy = max_window_chr_methylation_df.iloc[
                                        :, raw_methylation_col_index_list].to_numpy(dtype=np.float32)
                coverage_numpy = max_window_chr_methylation_df.iloc[
                                 :, coverage_col_index].to_numpy(dtype=np.float32)
                # 对coverage太低的位点赋nan值（使该位点不参与求平均）
                raw_methylation_numpy[coverage_numpy < self.minimal_coverage] = np.nan
                raw_methylation_mean_numpy = np.nanmean(raw_methylation_numpy, axis=0)
                # 用于记录当前的list index
                list_index = 0
                # 把算好的window methylation存放到对应的list中
                window_methylation_list[list_index].append(raw_methylation_mean_numpy)
                # 用于debug
                if row_index % 100000 == 0 and self.is_verbose:
                    print(f'chr: {chr_number}')
                    print(f'row_index: {row_index:7d}|{len(chr_methylation_df):7d}')
                    print(f'max_window_row_index:\n{max_window_row_index}')
                    print(f'raw_methylation_mean_numpy:\n{raw_methylation_mean_numpy}')
                    using_time = datetime.datetime.now() - window_methylation_start_time
                    print(f'using_time: {using_time}\n')
                # 第3层，遍历除了第1个以外的extend_lengths
                for extend_length in extend_length_list[1:]:
                    # window坐标
                    window_start = cpg_start - extend_length
                    window_end = cpg_end + extend_length
                    # window坐标范围的row_index
                    window_row_index = max_window_chr_methylation_df[
                        (max_window_chr_methylation_df['start'] > window_start) &
                        (max_window_chr_methylation_df['end'] < window_end)
                    ].index
                    # 对max_window_chr_methylation_df进行切片
                    window_chr_methylation_df = max_window_chr_methylation_df.loc[window_row_index, :]
                    raw_methylation_numpy = window_chr_methylation_df.iloc[
                                            :, raw_methylation_col_index_list].to_numpy(dtype=np.float32)
                    coverage_numpy =  window_chr_methylation_df.iloc[
                                      :, coverage_col_index].to_numpy(dtype=np.float32)
                    # 对coverage太低的位点赋nan值（使该位点不参与求平均）
                    raw_methylation_numpy[coverage_numpy < self.minimal_coverage] = np.nan
                    raw_methylation_mean_numpy = np.nanmean(raw_methylation_numpy, axis=0)
                    # index + 1
                    list_index = list_index + 1
                    # 把算好的window methylation存放到对应的list中
                    window_methylation_list[list_index].append(raw_methylation_mean_numpy)
                    # 此处结束3层嵌套的for循环
        # 把所有window methylation存放到self.methylation_df
        for i in range(len(window_size_list)):
            window_length = window_size_list[i]
            window_methylation_numpy = np.stack(window_methylation_list[i])
            window_methylation_col_name = [f'window_{window_length}{col_name[3:]}' for col_name in
                                           raw_methylation_col_name_list]
            window_methylation_dataframe = pd.DataFrame(window_methylation_numpy)
            window_methylation_dataframe.columns = window_methylation_col_name
            self.methylation_df = pd.concat(
                [self.methylation_df, window_methylation_dataframe], axis=1
            )

    def methylation_dataframe_fill_na(self):
        for col_name in self.methylation_df.columns:
            if str(col_name).startswith('raw_'):
                self.methylation_df[col_name] = self.methylation_df[col_name].fillna(-1.0)
            if str(col_name).startswith('window_'):
                self.methylation_df[col_name] = self.methylation_df[col_name].fillna(-1.0)

    def calculate_input_dna_coordinate(self):
        # 保存输入模型的序列长度
        extend_length = (self.model_input_dna_length - self.cpg_length) // 2
        # 根据序列长度，延伸输入序列
        self.methylation_df['input_dna_start'] = self.methylation_df['start'] - extend_length
        self.methylation_df['input_dna_end'] = self.methylation_df['end'] + extend_length
        # 根据chromosome_size，删除长度越界的序列
        self.methylation_df = pd.merge(self.methylation_df, self.chromosome_size_df, on='chr', how='left')
        self.methylation_df = self.methylation_df[
            (self.methylation_df['input_dna_start'] > 0) &
            (self.methylation_df['input_dna_end'] < self.methylation_df['chr_length'])
        ]

    def count_input_dna_n_base_number(self):
        n_number_list = []
        for row_index in self.methylation_df.index.to_list():
            chr_number = self.methylation_df.loc[row_index, 'chr']
            start_position = self.methylation_df.loc[row_index, 'input_dna_start']
            end_position = self.methylation_df.loc[row_index, 'input_dna_end']
            n_number = self.genome_fasta.get_n_number(chr_number, start_position, end_position)
            n_number_list.append(n_number)
        self.methylation_df['N_number'] = n_number_list

    def count_missing_sample(self):
        coverage_col_index = [index for index, col_name in enumerate(self.methylation_df.columns)
                              if col_name.startswith('coverage_')]
        missing_values_number = (self.methylation_df.iloc[:, coverage_col_index] < self.minimal_coverage).sum(axis=1)
        self.methylation_df['missing_sample'] = missing_values_number

    def reset_methylation_df_col_order(self):
        # 把最后5列('chr_length','input_dna_start', 'input_dna_end', 'N_number', 'missing_number')插入第4列之后，之间的列移到后面
        insert_col_number = 3
        last_col_number = -5
        begin_col = self.methylation_df.columns[:insert_col_number].to_list()
        end_col = self.methylation_df.columns[last_col_number:].to_list()
        remain_col = self.methylation_df.columns[insert_col_number: last_col_number].to_list()
        self.methylation_df = self.methylation_df[begin_col + end_col + remain_col]

    def trim_methylation_df(self, max_n_base_ratio: float, max_missing_sample_ratio: float):
        # 删除行：包含过多N碱基的序列
        n_base_max_number = self.model_input_dna_length * max_n_base_ratio
        self.methylation_df = self.methylation_df[
            self.methylation_df['N_number'] <= n_base_max_number
        ]
        # 删除行：包含过多缺省值
        missing_value_max_number = sum(1 for col_name in self.methylation_df.columns if
                                       col_name.startswith('smooth_')) * max_missing_sample_ratio
        self.methylation_df = self.methylation_df[
            self.methylation_df['missing_sample'] <= missing_value_max_number
        ]

    def output_train_validation_test_set(
            self, train_chr_list: list, validation_chr_list: list, test_chr_list: list,
            output_sampled_train_set_fraction_list: list, is_output_slice_train_set: bool,
            output_format: Literal['pickle', 'feather'] = 'pickle'
    ):
        # 训练集
        train_set_df = self.methylation_df[self.methylation_df['chr'].isin(train_chr_list)]
        output_file = f'{self.output_prefix}_train_set'
        self.output_dataset_df(train_set_df, output_file, output_format)
        # 不同长度的训练集
        if output_sampled_train_set_fraction_list:
            train_output_prefix = f'{self.output_prefix}_train_set'
            self.output_sampled_train_set(train_set_df, train_output_prefix, output_sampled_train_set_fraction_list,
                                          output_format)
        if is_output_slice_train_set:
            self.output_slice_train_set(f'{self.output_prefix}_train_set', train_set_df)
        # 验证集
        validation_set_df = self.methylation_df[self.methylation_df['chr'].isin(validation_chr_list)]
        output_file = f'{self.output_prefix}_validation_set'
        self.output_dataset_df(validation_set_df, output_file, output_format)
        # 测试集
        test_set_df = self.methylation_df[self.methylation_df['chr'].isin(test_chr_list)]
        output_file = f'{self.output_prefix}_test_set'
        self.output_dataset_df(test_set_df, output_file, output_format)

    def output_sampled_train_set(
            self, train_set_df, output_prefix, fraction_list: list,
            output_format: Literal['pickle', 'feather'] = 'pickle',
            random_state=42,
    ):
        # 打乱数据集
        shuffled_train_set_df = train_set_df.sample(frac=1, ignore_index=True, random_state=random_state)
        # 产生不同长度的训练集
        train_set_length = len(shuffled_train_set_df)
        for frac in fraction_list:
            train_length = int(train_set_length * frac)
            frac_train_set_df = shuffled_train_set_df[0: train_length]
            output_file = f'{output_prefix}_fraction_{frac}'
            self.output_dataset_df(frac_train_set_df, output_file, output_format)

    def output_slice_train_set(self, output_folder: str, train_set_df: pd.DataFrame):
        # 检查输出目录
        check_output_folder(output_folder)
        # 重置index
        train_set_df.reset_index(drop=True, inplace=True)
        train_dataset_len = len(train_set_df)
        # 输出第0行，然后去除表头
        row_index = 0
        train_set_df.loc[row_index, :].to_pickle(f'{output_folder}/{row_index}.pkl')
        train_set_df.columns = range(0, len(train_set_df.columns))
        # 记录开始时间
        start_time = datetime.datetime.now()
        for row_index in range(1, train_dataset_len):
            train_set_df.loc[row_index, :].to_pickle(f'{output_folder}/{row_index}.pkl')
            if row_index % 100_0000 == 0 and self.is_verbose:
                print(f'{row_index:8d}|{train_dataset_len:8d}')
                using_time = datetime.datetime.now() - start_time
                print(f'using_time: {using_time}\n')

    def output_dataset_df(self, dataset_df, output_file: str, output_format: Literal['pickle', 'feather'] = 'pickle'):
        if output_format == 'feather':
            output_file = f'{output_file}.feather'
            print('output: ' + output_file)
            dataset_df.to_feather(output_file)
        else:
            output_file = f'{output_file}.pkl'
            print('output: ' + output_file)
            dataset_df.to_pickle(output_file)

    def output_methylation_df(self, output_file: str):
        output_file = f'{self.output_prefix}_{output_file}'
        print('output: ' + output_file)
        self.methylation_df.to_csv(output_file, sep='\t', index=False)

def main_methylation_dataset():
    os.chdir('/home/chenfaming/tmp_pool2/project/240507_DNA_methylation_model_result/251125_github_test/data')
    methylation_dataset = MethylationDataset(
        methylation_file='encode_preprocess/smooth_methylation_dataset.txt.gz',
        data_info_file='encode_preprocess/smooth_methylation_info.txt',
        chromosome_size_file='/home/chenfaming/genome/ucsc_hg38/hg38.chrom.sizes',
        genome_fasta_file='/home/chenfaming/genome/ucsc_hg38/hg38.fa',
        model_input_dna_length = 9*2**11,
        minimal_coverage=5,
        is_quiet=False,
        output_folder='encode_dataset',
        output_prefix='encode'
    )
    methylation_dataset.methylation_dataframe_drop_sample(max_low_coverage_ratio=0.5)
    methylation_dataset.calculate_regional_methylation()
    methylation_dataset.methylation_dataframe_fill_na()
    methylation_dataset.calculate_input_dna_coordinate()
    methylation_dataset.count_input_dna_n_base_number()
    methylation_dataset.count_missing_sample()
    methylation_dataset.reset_methylation_df_col_order()
    methylation_dataset.output_methylation_df('complete_dataset.txt')
    methylation_dataset.trim_methylation_df(
        max_n_base_ratio=0.02, max_missing_sample_ratio=0.5
    )
    methylation_dataset.output_train_validation_test_set(
        train_chr_list=[f'chr{i}' for i in range(1, 10)] + [f'chr{i}' for i in range(12, 23)],
        validation_chr_list=['chr10'],
        test_chr_list=['chr11'],
        is_output_sampled_train_set=False,
        is_output_slice_train_set=False
    )

if __name__ == '__main__':
    main_methylation_dataset()