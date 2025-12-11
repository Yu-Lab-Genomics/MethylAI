import torch
from torch.utils.data import Dataset
import pandas as pd
from ..utils.genome_fasta import GenomeFasta
from ..utils.dna_sequence import dna_to_one_hot_tensor, get_reverse_complement

class VariantDataset(Dataset):
    def __init__(
            self, variant_dataset_file: str, genome_fasta_file: str, model_input_dna_length: int,
            is_reverse_complement_augmentation: bool):
        # input data
        self.variant_dataset_file = variant_dataset_file
        self.genome_fasta = GenomeFasta(genome_fasta_file)
        self.variant_dataset_df = pd.DataFrame()
        # 储存模型输入长度
        self.model_input_dna_length = model_input_dna_length
        # 从CG往左、右延伸的长度
        self.extent_dna_length = (self.model_input_dna_length - 2) // 2
        self.sequence_cg_start = self.model_input_dna_length // 2 - 1
        self._input_data()
        # 是否数据增强
        self.is_reverse_complement_augmentation = is_reverse_complement_augmentation

    def _input_data(self):
        if self.variant_dataset_file.endswith('pkl'):
            self.variant_dataset_df = pd.read_pickle(self.variant_dataset_file)
        elif self.variant_dataset_file.endswith('txt'):
            self.variant_dataset_df = pd.read_table(self.variant_dataset_file, sep='\t', header=0)
        # 删掉variant_cg_distance超过模型输入长度的行
        self.variant_dataset_df = self.variant_dataset_df[
            self.variant_dataset_df['variant_cg_distance'] < self.extent_dna_length
        ]
        # reset index
        self.variant_dataset_df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.variant_dataset_df)

    def __getitem__(self, idx):
        ref_dna, alt_dna, cg_tensor = self._get_ref_alt_dna_sequence(idx)
        # 如果使用数据增强，则返回正、负链的序列
        if self.is_reverse_complement_augmentation:
            forward_ref_dna = ref_dna
            reverse_ref_dna = get_reverse_complement(forward_ref_dna)
            forward_alt_dna = alt_dna
            reverse_alt_dna = get_reverse_complement(forward_alt_dna)
            # 获取对应的tensor
            forward_ref_dna_one_hot_tensor = dna_to_one_hot_tensor(forward_ref_dna)
            reverse_ref_dna_one_hot_tensor = dna_to_one_hot_tensor(reverse_ref_dna)
            forward_alt_dna_one_hot_tensor = dna_to_one_hot_tensor(forward_alt_dna)
            reverse_alt_dna_one_hot_tensor = dna_to_one_hot_tensor(reverse_alt_dna)
            return (forward_ref_dna_one_hot_tensor, reverse_ref_dna_one_hot_tensor,
                    forward_alt_dna_one_hot_tensor, reverse_alt_dna_one_hot_tensor, cg_tensor)
        else:
            ref_dna_one_hot_tensor = dna_to_one_hot_tensor(ref_dna)
            alt_dna_one_hot_tensor = dna_to_one_hot_tensor(alt_dna)
            return ref_dna_one_hot_tensor, alt_dna_one_hot_tensor, cg_tensor

    def get_dataset_df(self):
        return self.variant_dataset_df

    def _get_ref_alt_dna_sequence(self, idx):
        # 获取CG位点坐标
        chr = self.variant_dataset_df.loc[idx, 'chr']
        cg_start = self.variant_dataset_df.loc[idx, 'cg_start']
        cg_end = self.variant_dataset_df.loc[idx, 'cg_end']
        # 获取variant坐标
        variant_start = self.variant_dataset_df.loc[idx, 'variant_start']
        # 获取variant序列
        variant_alt = self.variant_dataset_df.loc[idx, 'ALT_split']
        # 获取variant长度
        variant_ref_len = self.variant_dataset_df.loc[idx, 'variant_ref_len']
        variant_alt_len = self.variant_dataset_df.loc[idx, 'variant_alt_len']
        variant_different_len = variant_ref_len - variant_alt_len
        # 计算输入模型的ref DNA的起始、终止坐标
        ref_dna_start = cg_start - self.extent_dna_length
        ref_dna_end = cg_end + self.extent_dna_length
        # 根据variant_start位置、variant_different_len调整alt DNA的起始、终止坐标
        if variant_start < cg_start:
            alt_dna_start = ref_dna_start - variant_different_len
            alt_dna_end = ref_dna_end
        else:
            alt_dna_start = ref_dna_start
            alt_dna_end = ref_dna_end + variant_different_len
        # 获取ref,alt DNA序列
        ref_dna_sequence = self.genome_fasta.get_sequence_tuple(chr, ref_dna_start, ref_dna_end, upper_sequence=True)[1]
        alt_dna_sequence = self.genome_fasta.get_sequence_tuple(chr, alt_dna_start, alt_dna_end, upper_sequence=True)[1]
        # 计算variant的相对坐标（相对于DNA的起始、终止坐标）
        relative_variant_ref_start = variant_start - alt_dna_start
        relative_variant_ref_end = relative_variant_ref_start + variant_ref_len
        # 获取alt DNA序列
        alt_dna_sequence = alt_dna_sequence[:relative_variant_ref_start] + \
                           variant_alt + alt_dna_sequence[relative_variant_ref_end:]
        # 产生一个tensor，表明中间位置的CG是否发生变异，0代表未变异，1代表变异
        sequence_cg_start = self.sequence_cg_start
        sequence_cg_end = sequence_cg_start + 2
        if alt_dna_sequence[sequence_cg_start: sequence_cg_end] == 'CG':
            cg_tensor = torch.tensor([0], dtype=torch.int)
        else:
            cg_tensor = torch.tensor([1], dtype=torch.int)
        return ref_dna_sequence, alt_dna_sequence, cg_tensor
