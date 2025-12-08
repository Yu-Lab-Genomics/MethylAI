import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Literal
from ..utils.genome_fasta import GenomeFasta
from ..utils.dna_sequence import dna_to_one_hot_tensor, get_reverse_complement

class CaptumDataset(Dataset):
    def __init__(self, captum_cpg_file: str, genome_fasta_file: str, model_input_dna_length: int, n_permutation: int,
                 is_reverse_complement_augmentation=False, permutation_random_seed=42,
                 permutation_method: Literal[1, 2] = 1):
        # dataframe
        self.captum_cpg_file = captum_cpg_file
        self.captum_cpg_df = pd.DataFrame()
        # 初始化genome_fasta
        self.genome_fasta = GenomeFasta(genome_fasta_file)
        self.model_input_dna_length = model_input_dna_length
        # permutation次数
        self.n_permutation = n_permutation
        self.permutation_random_seed = permutation_random_seed
        self._input_data()
        # permutation method
        if permutation_method == 1:
            self._permutation_method = self._get_permutate_tensor
        else:
            self._permutation_method = self._get_permutate_tensor_2
        # 反向互补
        self.is_reverse_complement_augmentation = is_reverse_complement_augmentation

    def __len__(self):
        return len(self.captum_cpg_df)

    def __getitem__(self, idx):
        if self.is_reverse_complement_augmentation:
            forward_dna_one_hot_tensor = self._get_dna_one_hot_tensor(idx, reverse_compliment=False)
            reverse_dna_one_hot_tensor = self._get_dna_one_hot_tensor(idx, reverse_compliment=True)
            forward_base_line_tensor = self._permutation_method(forward_dna_one_hot_tensor)
            reverse_base_line_tensor = self._permutation_method(reverse_dna_one_hot_tensor)
            return forward_dna_one_hot_tensor, reverse_dna_one_hot_tensor, forward_base_line_tensor, reverse_base_line_tensor
        else:
            dna_one_hot_tensor = self._get_dna_one_hot_tensor(idx, reverse_compliment=False)
            dna_one_hot_tensor = torch.unsqueeze(dna_one_hot_tensor, dim=0)
            base_line_tensor = self._permutation_method(dna_one_hot_tensor)
            return dna_one_hot_tensor, base_line_tensor

    def _input_data(self):
        self.captum_cpg_df = pd.read_table(self.captum_cpg_file, header=0)
        self.captum_cpg_df.reset_index(drop=True, inplace=True)
        cg_length = self.captum_cpg_df.iloc[0, 2] - self.captum_cpg_df.iloc[0, 1]
        cg_extent_length = cg_length // 2
        model_extent_length = self.model_input_dna_length // 2
        self.captum_cpg_df.loc[:, 'input_dna_start'] = self.captum_cpg_df.loc[:, 'start'] + cg_extent_length - model_extent_length
        self.captum_cpg_df.loc[:, 'input_dna_end'] = self.captum_cpg_df.loc[:, 'end'] - cg_extent_length + model_extent_length

    def get_captum_cpg_df(self):
        return self.captum_cpg_df.copy()

    def _get_dna_one_hot_tensor(self, idx, reverse_compliment: bool):
        chr_number = self.captum_cpg_df.loc[idx, 'chr']
        dna_start_position = self.captum_cpg_df.loc[idx, 'input_dna_start']
        dna_end_position = self.captum_cpg_df.loc[idx, 'input_dna_end']
        dna_sequence = self.genome_fasta.get_sequence_tuple(
            chr_number, dna_start_position, dna_end_position, upper_sequence=True
        )[1]
        if reverse_compliment:
            dna_sequence = get_reverse_complement(dna_sequence)
        dna_one_hot_tensor = dna_to_one_hot_tensor(dna_sequence)
        return dna_one_hot_tensor

    def _get_permutate_tensor(self, dna_tensor):
        permuted_tensor_list = []
        torch.manual_seed(self.permutation_random_seed)
        for _ in range(self.n_permutation):
            permuted_indices = torch.randperm(dna_tensor.shape[2])
            permuted_tensor = dna_tensor[:, :, permuted_indices]
            permuted_tensor_list.append(permuted_tensor)
        permuted_stack_tensor = torch.concatenate(permuted_tensor_list, dim=0)
        return permuted_stack_tensor

    # 改进：打乱序列时不会影响中间的CpG位点
    def _get_permutate_tensor_2(self, dna_tensor):
        permuted_tensor_list = []
        torch.manual_seed(self.permutation_random_seed)
        # 取CpG的下标
        cg_start_index = dna_tensor.shape[2] // 2 - 1
        cg_end_index = cg_start_index + 2
        for _ in range(self.n_permutation):
            # 去除DNA中间的CpG
            dna_remove_cg_tensor = torch.concat(
                [dna_tensor[:, :, :cg_start_index], dna_tensor[:, :, cg_end_index:]], dim=-1
            )
            cg_tensor = dna_tensor[:, :, cg_start_index: cg_end_index]
            # 打乱下标
            permuted_indices = torch.randperm(dna_remove_cg_tensor.shape[2])
            permuted_dna_remove_cg_tensor = dna_remove_cg_tensor[:, :, permuted_indices]
            # 重新把CpG拼回去
            permuted_dna_tensor = torch.concat(
                [permuted_dna_remove_cg_tensor[:, :, :cg_start_index], cg_tensor, permuted_dna_remove_cg_tensor[:, :, cg_start_index:]],
                dim=-1
            )
            permuted_tensor_list.append(permuted_dna_tensor)
        permuted_stack_tensor = torch.concatenate(permuted_tensor_list, dim=0)
        return permuted_stack_tensor









