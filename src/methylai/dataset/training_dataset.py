import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from ..utils.genome_fasta import GenomeFasta
from ..utils.dna_sequence import dna_to_one_hot_tensor, get_reverse_complement


class MethylAITrainDataset(Dataset):
    def __init__(self, dataset_file: str, bed_dataset_to_repetition_dict: dict, genome_fasta_file: str,
                 model_input_dna_length: int,
                 minimal_coverage=5, loss_weight_factor=5.0, max_loss_weight_factor=1.0,
                 is_keep_smooth_methylation=True, is_keep_raw_methylation=True, is_keep_window_methylation=True,
                 is_reverse_complement_augmentation=True):
        # dataset
        self.dataset_file_name = dataset_file
        self.dataset_dataframe = pd.DataFrame()
        # resampling
        self.bed_dataset_to_repetition_dict = bed_dataset_to_repetition_dict
        self.resample_index_series = pd.Series()
        # input dna
        self.genome_fasta_file = GenomeFasta(genome_fasta_file)
        self.model_input_dna_length = model_input_dna_length
        # length of dataset
        self.dataset_dataframe_length = 0
        self.resample_length = 0
        self.dataset_length = 0
        self._input_dataset()
        self._input_bed_dataset_to_repetition()
        #
        self.minimal_coverage = minimal_coverage
        self.loss_weight_factor = loss_weight_factor
        self.max_loss_weight_factor = max_loss_weight_factor
        self.is_reverse_complement_augmentation = is_reverse_complement_augmentation
        #
        self.is_keep_raw_methylation = is_keep_raw_methylation
        self.is_keep_smooth_methylation = is_keep_smooth_methylation
        self.is_keep_window_methylation = is_keep_window_methylation
        assert (self.is_keep_raw_methylation or self.is_keep_smooth_methylation or self.is_keep_window_methylation)
        #
        self.smooth_methylation_col_index = []
        self.raw_methylation_col_index = []
        self.coverage_col_index = []
        self.window_methylation_col_index = []
        self._infer_col_index()

    def __len__(self):
        if self.is_reverse_complement_augmentation:
            dataset_length = self.dataset_length * 2
        else:
            dataset_length = self.dataset_length
        return dataset_length

    def __getitem__(self, idx):
        # idx < self.dataset_length, return forward DNA
        if idx < self.dataset_length:
            # idx >= self.dataset_dataframe_length, resample_index
            if idx >= self.dataset_dataframe_length:
                resample_index = idx - self.dataset_dataframe_length
                idx = self.resample_index_series[resample_index]
            dna_one_hot_tensor = self._get_dna_one_hot_tensor(idx, reverse_compliment=False)
            target_tensor, loss_weight_tensor = self._get_predict_target_and_loss_weight(idx)
        # idx > self.dataset_length, return reverse DNA
        else:
            idx = idx - self.dataset_length
            # idx >= self.dataset_dataframe_length, resample_index
            if idx >= self.dataset_dataframe_length:
                resample_index = idx - self.dataset_dataframe_length
                idx = self.resample_index_series[resample_index]
            dna_one_hot_tensor = self._get_dna_one_hot_tensor(idx, reverse_compliment=True)
            target_tensor, loss_weight_tensor = self._get_predict_target_and_loss_weight(idx)
        return dna_one_hot_tensor, target_tensor, loss_weight_tensor

    def _input_dataset(self):
        if self.dataset_file_name.endswith('pkl'):
            self.dataset_dataframe = pd.read_pickle(self.dataset_file_name)
        elif self.dataset_file_name.endswith('txt'):
            self.dataset_dataframe = pd.read_table(self.dataset_file_name, sep='\t', header=0)
        # 记录dataset_dataframe_length
        self.dataset_dataframe_length = len(self.dataset_dataframe)
        # 重置index
        self.dataset_dataframe.reset_index(drop=True, inplace=True)
        # 新增功能，重新把dataset_dataframe的坐标改到中间，然后再根据需要输入的DNA长度延伸至正确的坐标
        dataset_length = self.dataset_dataframe.iloc[0, 2] - self.dataset_dataframe.iloc[0, 1]
        dataset_extent_length = dataset_length // 2
        model_extent_length = self.model_input_dna_length // 2
        self.dataset_dataframe.iloc[:, 1] = self.dataset_dataframe.iloc[:, 1] + dataset_extent_length - model_extent_length
        self.dataset_dataframe.iloc[:, 2] = self.dataset_dataframe.iloc[:, 2] - dataset_extent_length + model_extent_length

    def _input_bed_dataset_to_repetition(self):
        # 遍历所有bed_dataset_file
        for bed_dataset_file, repetition in self.bed_dataset_to_repetition_dict.items():
            bed_dataset_dataframe = pd.read_table(bed_dataset_file, header=None, sep='\t')
            # 根据self.resample_index_series是否为空，使用不同的concat方法
            if self.resample_index_series.empty:
                self.resample_index_series = pd.concat(
                    [bed_dataset_dataframe[3]] * repetition, ignore_index=True
                )
            else:
                self.resample_index_series = pd.concat(
                    [self.resample_index_series] + [bed_dataset_dataframe[3]] * repetition, ignore_index=True
                )
        # 计算dataset最终长度
        self.resample_length = len(self.resample_index_series)
        self.dataset_length = self.dataset_dataframe_length + self.resample_length
        print(f'resample_length: {self.resample_length}')
        print(f'dataset_length: {self.dataset_length}')

    def get_dataset_dataframe(self):
        return self.dataset_dataframe.copy()

    def _infer_col_index(self):
        #meth_dataframe中存放smooth_methylation的列index
        self.smooth_methylation_col_index = [
            index for index, col_name in enumerate(self.dataset_dataframe.columns) if col_name.startswith('smooth_')
        ]
        # meth_dataframe中存放raw_methylation的列index
        self.raw_methylation_col_index = [
            index for index, col_name in enumerate(self.dataset_dataframe.columns) if col_name.startswith('raw_')
        ]
        # meth_dataframe中存放coverage的列index
        self.coverage_col_index = [
            index for index, col_name in enumerate(self.dataset_dataframe.columns) if col_name.startswith('coverage_')
        ]
        # meth_dataframe中存放window_methylation的列index
        self.window_methylation_col_index = [
            index for index, col_name in enumerate(self.dataset_dataframe.columns) if col_name.startswith('window_')
        ]

    def _get_dna_one_hot_tensor(self, idx, reverse_compliment=False):
        chr_number = self.dataset_dataframe.iloc[idx, 0]
        dna_start_position = self.dataset_dataframe.iloc[idx, 1]
        dna_end_position = self.dataset_dataframe.iloc[idx, 2]
        dna_sequence = self.genome_fasta_file.get_sequence_tuple(
            chr_number, dna_start_position, dna_end_position, upper_sequence=True
        )[1]
        if reverse_compliment:
            dna_sequence = get_reverse_complement(dna_sequence)
        dna_one_hot_tensor = dna_to_one_hot_tensor(dna_sequence)
        return dna_one_hot_tensor

    def _get_predict_target_and_loss_weight(self, idx):
        smooth_methylation_numpy = self.dataset_dataframe.iloc[idx, self.smooth_methylation_col_index].to_numpy(
            dtype=np.float32, copy=True)
        raw_methylation_numpy = self.dataset_dataframe.iloc[idx, self.raw_methylation_col_index].to_numpy(
            dtype=np.float32, copy=True)
        window_methylation_numpy = self.dataset_dataframe.iloc[idx, self.window_methylation_col_index].to_numpy(
            dtype=np.float32, copy=True)
        coverage_numpy = self.dataset_dataframe.iloc[idx, self.coverage_col_index].to_numpy(
            dtype=np.float32, copy=True)
        # smooth, raw, target tensor
        smooth_methylation_tensor = torch.tensor(smooth_methylation_numpy, dtype=torch.float)
        raw_methylation_tensor = torch.tensor(raw_methylation_numpy, dtype=torch.float)
        window_methylation_tensor = torch.tensor(window_methylation_numpy, dtype=torch.float)
        # coverage <= self.minimal_read_count被设为0，将不产生loss
        coverage_numpy[coverage_numpy <= self.minimal_coverage] = 0
        # loss weight 1 (for smooth)
        loss_weight_numpy_1 = coverage_numpy  / self.loss_weight_factor
        loss_weight_numpy_1 = np.minimum(loss_weight_numpy_1, self.max_loss_weight_factor)
        loss_weight_tensor_1 = torch.tensor(loss_weight_numpy_1, dtype=torch.float)
        # loss weight 2 (for raw)
        loss_weight_numpy_2 = coverage_numpy  / self.loss_weight_factor
        loss_weight_numpy_2 = np.minimum(loss_weight_numpy_2, self.max_loss_weight_factor)
        loss_weight_tensor_2 = torch.tensor(loss_weight_numpy_2, dtype=torch.float)
        # loss weight 3 (for window)
        loss_weight_numpy_3 = np.ones(window_methylation_numpy.shape, dtype=np.float32)
        loss_weight_numpy_3[window_methylation_numpy == -1.0] = 0
        loss_weight_tensor_3 = torch.tensor(loss_weight_numpy_3, dtype=torch.float)
        # cat loss weight
        if self.is_keep_smooth_methylation and self.is_keep_raw_methylation and self.is_keep_window_methylation:
            target_tensor = torch.cat([smooth_methylation_tensor, raw_methylation_tensor, window_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_1, loss_weight_tensor_2, loss_weight_tensor_3], dim=-1)
        elif self.is_keep_smooth_methylation and self.is_keep_raw_methylation:
            target_tensor = torch.cat([smooth_methylation_tensor, raw_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_1, loss_weight_tensor_2], dim=-1)
        elif self.is_keep_smooth_methylation and self.is_keep_window_methylation:
            target_tensor = torch.cat([smooth_methylation_tensor, window_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_1, loss_weight_tensor_3], dim=-1)
        elif self.is_keep_raw_methylation and self.is_keep_window_methylation:
            target_tensor = torch.cat([raw_methylation_tensor, window_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_2, loss_weight_tensor_3], dim=-1)
        elif self.is_keep_smooth_methylation:
            target_tensor = smooth_methylation_tensor
            loss_weight_tensor = loss_weight_numpy_1
        elif self.is_keep_raw_methylation:
            target_tensor = raw_methylation_tensor
            loss_weight_tensor = loss_weight_numpy_2
        elif self.is_keep_window_methylation:
            target_tensor = window_methylation_tensor
            loss_weight_tensor = loss_weight_numpy_3
        return target_tensor, loss_weight_tensor

