import torch

def dna_to_one_hot_tensor(dna_sequence: str):
    # Convert a DNA sequence to one hot encoding.
    # Args:
    #     dna_sequence (str): The input DNA sequence.
    # Returns:
    #     list: A list of one-hot-encoded DNA sequences.
    # Define the mapping of bases to one hot encoding
    base_to_one_hot = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'C': [0, 0, 0, 1]
    }
    # Initialize the one hot encoding list
    one_hot_sequences = []
    # Upper DNA sequence
    dna_sequence = dna_sequence.upper()
    # Convert each base in the DNA sequence to one hot encoding
    for base in dna_sequence:
        one_hot = base_to_one_hot.get(base, [0, 0, 0, 0])
        one_hot_sequences.append(one_hot)
    # 改为tensor格式
    one_hot_sequences_tensor = torch.tensor(one_hot_sequences, dtype=torch.float)
    # 转置
    one_hot_sequences_tensor = one_hot_sequences_tensor.T
    return one_hot_sequences_tensor

def get_reverse_complement(dna_sequence: str):
    dna_sequence = dna_sequence.upper()
    complement_dict = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
    }
    # 生成反向互补序列。获取互补序列时，如果获取序列不在ATCG内，则返回'N'
    reverse_complement_dna = ''.join(complement_dict.get(base, 'N') for base in reversed(dna_sequence))
    return reverse_complement_dna

