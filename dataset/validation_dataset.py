import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from ..utils.genome_fasta import GenomeFasta
from ..utils.dna_sequence import dna_to_one_hot_tensor, get_reverse_complement


class MethylAIValidationDataset(Dataset):
    def __init__(self, dataset_file, genome_fasta_file, model_input_dna_length: int,
                 minimal_coverage=5, loss_weight_factor=10.0, max_loss_weight_factor=1.0,
                 keep_smooth_methylation=True, keep_raw_methylation=True, keep_window_methylation=True,
                 reverse_complement_augmentation=False):
        # 读取dataframe
        self.dataset_file = dataset_file
        self.dataset_df = pd.DataFrame()
        self.model_input_dna_length = model_input_dna_length
        self._input_dataset()
        self.dataset_df_length = len(self.dataset_df)
        self.genome_fasta = GenomeFasta(genome_fasta_file)
        self.minimal_coverage = minimal_coverage
        self.loss_weight_factor = loss_weight_factor
        self.max_loss_weight_factor = max_loss_weight_factor
        self.keep_raw_methylation = keep_raw_methylation
        self.keep_smooth_methylation = keep_smooth_methylation
        self.keep_window_methylation = keep_window_methylation
        assert (self.keep_raw_methylation or self.keep_smooth_methylation or self.keep_window_methylation)
        self.reverse_complement_augmentation = reverse_complement_augmentation
        # 需要infer的数值
        self.smooth_methylation_col_index = [0]
        self.raw_methylation_col_index = [0]
        self.coverage_col_index = [0]
        self.window_methylation_col_index = [0]
        self._infer_col_index()

    def __len__(self):
        return self.dataset_df_length

    def __getitem__(self, idx):
        if self.reverse_complement_augmentation:
            forward_dna_one_hot_tensor = self.get_dna_one_hot_tensor(idx, is_reverse_compliment=False)
            reverse_dna_one_hot_tensor = self.get_dna_one_hot_tensor(idx, is_reverse_compliment=True)
            target_tensor, loss_weight_tensor = self.get_predict_target_and_loss_weight(idx)
            return forward_dna_one_hot_tensor, reverse_dna_one_hot_tensor, target_tensor, loss_weight_tensor
        else:
            forward_dna_one_hot_tensor = self.get_dna_one_hot_tensor(idx, is_reverse_compliment=False)
            target_tensor, loss_weight_tensor = self.get_predict_target_and_loss_weight(idx)
            return forward_dna_one_hot_tensor, target_tensor, loss_weight_tensor

    def _input_dataset(self):
        if self.dataset_file.endswith('pkl'):
            self.dataset_df = pd.read_pickle(self.dataset_file)
        elif self.dataset_file.endswith('txt'):
            self.dataset_df = pd.read_table(self.dataset_file, sep='\t', header=0)
        # 重置index
        self.dataset_df.reset_index(drop=True, inplace=True)
        # 新增功能，重新把dataset_df的坐标改到中间，然后再根据需要输入的DNA长度延伸至正确的坐标
        cg_length = self.dataset_df.iloc[0, 2] - self.dataset_df.iloc[0, 1]
        cg_extent_length = cg_length // 2
        model_extent_length = self.model_input_dna_length // 2
        self.dataset_df.loc[:, 'input_dna_start'] = self.dataset_df.loc[:, 'start'] + cg_extent_length - model_extent_length
        self.dataset_df.loc[:, 'input_dna_end'] = self.dataset_df.loc[:, 'end'] - cg_extent_length + model_extent_length

    def get_dataset_df(self):
        return self.dataset_df.copy()

    def _infer_col_index(self):
        self.coverage_col_index = [
            index for index, col_name in enumerate(self.dataset_df.columns) if col_name.startswith('coverage_')
        ]
        self.raw_methylation_col_index = [
            index for index, col_name in enumerate(self.dataset_df.columns) if col_name.startswith('raw_')
        ]
        self.smooth_methylation_col_index = [
            index for index, col_name in enumerate(self.dataset_df.columns) if col_name.startswith('smooth_')
        ]
        self.window_methylation_col_index = [
            index for index, col_name in enumerate(self.dataset_df.columns) if col_name.startswith('window_')
        ]
        coverage_col_len = len(self.coverage_col_index)
        assert len(self.raw_methylation_col_index) == coverage_col_len
        assert len(self.smooth_methylation_col_index) == coverage_col_len
        assert len(self.window_methylation_col_index) // coverage_col_len == 0

    def get_dna_one_hot_tensor(self, idx, is_reverse_compliment: bool):
        chr_number = self.dataset_df.loc[idx, 'chr']
        dna_start_position = self.dataset_df.loc[idx, 'input_dna_start']
        dna_end_position = self.dataset_df.loc[idx, 'input_dna_end']
        # upper_sequence=True使获取的DNA序列全部为大写字母
        dna_sequence = self.genome_fasta.get_sequence_tuple(
            chr_number, dna_start_position, dna_end_position, upper_sequence=True
        )[1]
        if is_reverse_compliment:
            dna_sequence = get_reverse_complement(dna_sequence)
        dna_one_hot_tensor = dna_to_one_hot_tensor(dna_sequence)
        return dna_one_hot_tensor

    def get_predict_target_and_loss_weight(self, idx):
        # 预测target和损失权重
        smooth_methylation_numpy = self.dataset_df.iloc[idx, self.smooth_methylation_col_index].to_numpy(
            dtype=np.float32, copy=True)
        raw_methylation_numpy = self.dataset_df.iloc[idx, self.raw_methylation_col_index].to_numpy(
            dtype=np.float32, copy=True)
        window_methylation_numpy = self.dataset_df.iloc[idx, self.window_methylation_col_index].to_numpy(
            dtype=np.float32, copy=True)
        coverage_numpy = self.dataset_df.iloc[idx, self.coverage_col_index].to_numpy(
            dtype=np.float32, copy=True)
        # smooth, raw, target tensor
        smooth_methylation_tensor = torch.tensor(smooth_methylation_numpy, dtype=torch.float)
        raw_methylation_tensor = torch.tensor(raw_methylation_numpy, dtype=torch.float)
        window_methylation_tensor = torch.tensor(window_methylation_numpy, dtype=torch.float)
        # coverage <= self.minimal_read_count被设为0，将不产生loss
        coverage_numpy[coverage_numpy < self.minimal_coverage] = 0
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
        loss_weight_numpy_3[window_methylation_numpy == -1] = 0
        loss_weight_tensor_3 = torch.tensor(loss_weight_numpy_3, dtype=torch.float)
        # cat loss weight
        if self.keep_smooth_methylation and self.keep_raw_methylation and self.keep_window_methylation:
            target_tensor = torch.cat([smooth_methylation_tensor, raw_methylation_tensor, window_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_1, loss_weight_tensor_2, loss_weight_tensor_3], dim=-1)
        elif self.keep_smooth_methylation and self.keep_raw_methylation:
            target_tensor = torch.cat([smooth_methylation_tensor, raw_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_1, loss_weight_tensor_2], dim=-1)
        elif self.keep_smooth_methylation and self.keep_window_methylation:
            target_tensor = torch.cat([smooth_methylation_tensor, window_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_1, loss_weight_tensor_3], dim=-1)
        elif self.keep_raw_methylation and self.keep_window_methylation:
            target_tensor = torch.cat([raw_methylation_tensor, window_methylation_tensor], dim=-1)
            loss_weight_tensor = torch.cat([loss_weight_tensor_2, loss_weight_tensor_3], dim=-1)
        elif self.keep_smooth_methylation:
            target_tensor = smooth_methylation_tensor
            loss_weight_tensor = loss_weight_numpy_1
        elif self.keep_raw_methylation:
            target_tensor = raw_methylation_tensor
            loss_weight_tensor = loss_weight_numpy_2
        elif self.keep_window_methylation:
            target_tensor = window_methylation_tensor
            loss_weight_tensor = loss_weight_numpy_3
        return target_tensor, loss_weight_tensor


