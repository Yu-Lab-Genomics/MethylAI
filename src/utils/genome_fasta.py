from pathlib import Path
import numpy as np

class GenomeFasta:
    """
    读取参考基因组 FASTA，并提供序列截取和 N 碱基计数功能。
    Read a reference genome FASTA and provide sequence extraction plus N-base
    counting utilities.

    参数 / Parameters
    ----------
    fasta_file:
        参考基因组 FASTA 文件路径。FASTA header 第一段会作为染色体 key，
        例如 `chr1`。
        Path to the reference genome FASTA file. The first FASTA header field is
        used as the chromosome key, for example `chr1`.

    说明 / Notes
    ----------
    - `get_sequence_tuple()` 保留旧代码的切片行为，用于兼容已有调用。
      `get_sequence_tuple()` preserves the old slicing behavior for
      compatibility.
    - `get_n_number_array()` 使用每条染色体的 N 前缀和批量计数，避免对每个
      CpG 重复截取字符串。
      `get_n_number_array()` uses per-chromosome N-prefix sums for batch
      counting, avoiding repeated string slicing for every CpG.
    """

    def __init__(self, fasta_file: str | Path):
        self.fasta_file = str(fasta_file)
        self.fa_file_list: list[str] = []
        self.chr_to_dna_dict: dict[str, str] = {}
        self._chr_to_n_prefix_sum_dict: dict[str, np.ndarray] = {}
        self._input_file()
        self._generate_chr_to_dna_dict()

    def _input_file(self) -> None:
        fasta_path = Path(self.fasta_file)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file does not exist: {fasta_path}")
        if fasta_path.stat().st_size == 0:
            raise ValueError(f"FASTA file is empty: {fasta_path}")
        with fasta_path.open(encoding="utf-8") as fa_file:
            self.fa_file_list = fa_file.read().split(">")[1:]
        if not self.fa_file_list:
            raise ValueError(f"No FASTA records found in file: {fasta_path}")

    def _generate_chr_to_dna_dict(self) -> None:
        for file_line in self.fa_file_list:
            line_parts = file_line.split("\n")
            chr_key = line_parts[0].split()[0]
            fa_value = "".join(line_parts[1:])
            if chr_key in self.chr_to_dna_dict:
                raise ValueError(f"Duplicated chromosome name in FASTA: {chr_key}")
            self.chr_to_dna_dict[chr_key] = fa_value

    def _get_chr_sequence(self, chr_number: str) -> str:
        if chr_number not in self.chr_to_dna_dict:
            available_chr = ", ".join(list(self.chr_to_dna_dict)[:10])
            raise KeyError(f"Chromosome {chr_number} is not found in FASTA. Available examples: {available_chr}")
        return self.chr_to_dna_dict[chr_number]

    def _get_n_prefix_sum(self, chr_number: str) -> np.ndarray:
        if chr_number not in self._chr_to_n_prefix_sum_dict:
            chr_sequence = self._get_chr_sequence(chr_number)
            # 前缀和长度为 chr_length + 1，方便用 prefix[end] - prefix[start] 计数。
            # Prefix length is chr_length + 1 so intervals can be counted as
            # prefix[end] - prefix[start].
            is_n_base = np.frombuffer(chr_sequence.upper().encode("ascii"), dtype="S1") == b"N"
            n_prefix_sum = np.empty(len(is_n_base) + 1, dtype=np.int32)
            n_prefix_sum[0] = 0
            n_prefix_sum[1:] = np.cumsum(is_n_base, dtype=np.int32)
            self._chr_to_n_prefix_sum_dict[chr_number] = n_prefix_sum
        return self._chr_to_n_prefix_sum_dict[chr_number]

    def get_sequence_tuple(
        self,
        chr: str,
        start: int,
        end: int,
        upper_sequence: bool = True,
    ) -> tuple[str, str]:
        # FASTA header 使用 `chr_start_end` 格式，沿用旧代码输出。
        # FASTA header uses the old `chr_start_end` format.
        fasta_name = "_".join([chr, str(start), str(end)])
        if start < 0:
            fasta_name = fasta_name + "(start_position_smaller_than_0)"

        chr_fasta = self._get_chr_sequence(chr)
        chr_fasta_length = len(chr_fasta)
        if end > chr_fasta_length:
            fasta_name = fasta_name + "(end_position_bigger_than_" + str(chr_fasta_length) + ")"

        sequence = chr_fasta[start:end]
        if upper_sequence:
            sequence = sequence.upper()
        return fasta_name, sequence

    def get_n_number(self, chr_number: str, start: int, end: int) -> int:
        n_prefix_sum = self._get_n_prefix_sum(chr_number)
        start_int = int(start)
        end_int = int(end)
        return int(n_prefix_sum[end_int] - n_prefix_sum[start_int])

    def get_n_number_array(
        self,
        chr_number: str,
        start_array: np.ndarray,
        end_array: np.ndarray,
    ) -> np.ndarray:
        """
        批量统计同一条染色体上多个区间内的 N 碱基数量。
        Count N bases for multiple intervals on the same chromosome in batch.
        """

        n_prefix_sum = self._get_n_prefix_sum(chr_number)
        start_int_array = start_array.astype(np.int64, copy=False)
        end_int_array = end_array.astype(np.int64, copy=False)
        return n_prefix_sum[end_int_array] - n_prefix_sum[start_int_array]
