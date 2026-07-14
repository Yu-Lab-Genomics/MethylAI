import datetime
import logging
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import polars as pl

project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from MethylAI.src.utils.genome_fasta import GenomeFasta
from MethylAI.src.utils.utils import check_output_path_empty
from MethylAI.src.utils.log import setup_logger

RegionalDtype = Literal["float32", "float64"]
DatasetFormat = Literal["pickle", "feather"]
CompleteDatasetFormat = Literal["tsv", "tsv.gz", "feather"]


class MethylationDataset:
    """
    生成 MethylAI 训练、验证和测试数据集。
    Generate train, validation, and test datasets for MethylAI.

    功能 / Features
    --------------
    - 读取 BSmooth 后的甲基化矩阵和样本信息。
      Read the post-BSmooth methylation matrix and sample information.
    - 根据 coverage 对样本和 CpG 位点进行质量控制。
      Perform sample-level and CpG-level QC using coverage values.
    - 使用前缀和快速计算区域甲基化水平。
      Calculate regional methylation efficiently with prefix sums.
    - 使用参考基因组 FASTA 提取输入 DNA 坐标，并快速统计 N 碱基数。
      Use the reference genome FASTA to derive input DNA coordinates and count
      N bases efficiently.
    - 输出 complete dataset 以及按染色体划分的 train/validation/test pickle。
      Write the complete dataset and chromosome-split train/validation/test
      pickle files.

    参数 / Parameters
    ----------------
    methylation_file:
        平滑甲基化矩阵文件，包含 `chr`, `start`, `end`, `smooth_*`,
        `raw_*`, `coverage_*` 列。
        Smoothed methylation matrix with `chr`, `start`, `end`, `smooth_*`,
        `raw_*`, and `coverage_*` columns.
    data_info_file:
        样本信息文件，至少包含 `dataset_index`。
        Sample information file containing at least `dataset_index`.
    genome_fasta_file:
        与甲基化坐标匹配的参考基因组 FASTA 文件。
        Reference genome FASTA matching the methylation coordinate system.
    chromosome_size_file:
        染色体长度文件，两列分别为染色体名称和长度。
        Chromosome size file with chromosome name and length.
    minimal_coverage:
        判断 CpG 是否为有效观测的最小 coverage。
        Minimum coverage used to define a valid CpG observation.
    model_input_dna_length:
        模型输入 DNA 序列长度。
        DNA sequence length used as model input.
    output_folder:
        输出目录。
        Output directory.
    output_prefix:
        输出文件名前缀。
        Output filename prefix.
    is_quiet:
        是否减少进度日志。
        Whether to suppress progress logs.
    logger:
        可选 logger；未提供时自动创建模块 logger。
        Optional logger; a module logger is created if omitted.
    overwrite:
        是否允许覆盖已有输出文件。
        Whether existing output files may be overwritten.
    """

    def __init__(
        self,
        methylation_file: str,
        data_info_file: str,
        genome_fasta_file: str,
        chromosome_size_file: str,
        minimal_coverage: int,
        model_input_dna_length: int,
        output_folder: str,
        output_prefix: str,
        is_quiet: bool = False,
        logger: logging.Logger | None = None,
        overwrite: bool = False,
    ):
        self.methylation_file = methylation_file
        self.data_info_file = data_info_file
        self.chromosome_size_file = chromosome_size_file
        self.genome_fasta_file = genome_fasta_file
        self.minimal_coverage = minimal_coverage
        self.model_input_dna_length = model_input_dna_length
        self.is_verbose = not is_quiet
        self.output_folder = Path(output_folder)
        self.output_prefix_name = output_prefix
        self.output_prefix = self.output_folder / output_prefix
        self.overwrite = overwrite
        self.logger = logger or setup_logger("MethylationDataset")

        self.methylation_df: pd.DataFrame = pd.DataFrame()
        self.chromosome_size_df: pd.DataFrame = pd.DataFrame()
        self.data_info_df: pd.DataFrame = pd.DataFrame()
        self.cpg_length = 2

        self._input_data()
        self.genome_fasta = GenomeFasta(genome_fasta_file)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _check_input_file(self, file_path: str | Path, description: str) -> Path:
        checked_path = Path(file_path)
        if not checked_path.exists():
            raise FileNotFoundError(f"{description} does not exist: {checked_path}")
        if checked_path.stat().st_size == 0:
            raise ValueError(f"{description} is empty: {checked_path}")
        return checked_path

    def _check_can_write(self, output_file: str | Path) -> Path:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.overwrite:
            check_output_path_empty(output_path=output_path)
        return output_path

    def _build_output_path(self, suffix: str) -> Path:
        return Path(f"{self.output_prefix}_{suffix}")

    def _input_data(self) -> None:
        methylation_path = self._check_input_file(self.methylation_file, "Methylation matrix file")
        data_info_path = self._check_input_file(self.data_info_file, "Data info file")
        chromosome_size_path = self._check_input_file(self.chromosome_size_file, "Chromosome size file")

        self.logger.info("Read methylation matrix: %s", methylation_path)
        self.methylation_df = pl.read_csv(
            methylation_path,
            separator="\t",
            null_values=["NA"],
            infer_schema_length=100_000,
        ).to_pandas()
        required_columns = {"chr", "start", "end"}
        missing_columns = required_columns - set(self.methylation_df.columns)
        if missing_columns:
            raise ValueError(f"Methylation matrix is missing required columns: {sorted(missing_columns)}")
        self.methylation_df["start"] = self.methylation_df["start"].astype(np.int64)
        self.methylation_df["end"] = self.methylation_df["end"].astype(np.int64)
        if self.methylation_df.empty:
            raise ValueError(f"Methylation matrix has no rows: {methylation_path}")
        self.cpg_length = int(self.methylation_df.loc[self.methylation_df.index[0], "end"] - self.methylation_df.loc[self.methylation_df.index[0], "start"])

        self.logger.info("Read data info: %s", data_info_path)
        self.data_info_df = pd.read_table(data_info_path, sep="\t", header=0)
        if "dataset_index" not in self.data_info_df.columns:
            raise ValueError("Data info file must contain a 'dataset_index' column.")

        self.logger.info("Read chromosome sizes: %s", chromosome_size_path)
        self.chromosome_size_df = pd.read_table(chromosome_size_path, sep="\t", header=None)
        if self.chromosome_size_df.shape[1] < 2:
            raise ValueError("Chromosome size file must contain at least two columns.")
        self.chromosome_size_df = self.chromosome_size_df.iloc[:, :2].copy()
        self.chromosome_size_df.columns = ["chr", "chr_length"]
        self.chromosome_size_df["chr_length"] = self.chromosome_size_df["chr_length"].astype(np.int64)

        self.logger.info("Methylation matrix shape: %s rows x %s columns", *self.methylation_df.shape)

    def keep_first_cpg_per_chr(self, cpg_per_chr: int) -> None:
        """
        MVP 测试模式：每条染色体只保留前 N 个 CpG。
        MVP test mode: keep only the first N CpGs from each chromosome.
        """

        before_rows = len(self.methylation_df)
        self.methylation_df = (
            self.methylation_df.groupby("chr", sort=False, group_keys=False)
            .head(cpg_per_chr)
            .reset_index(drop=True)
        )
        self.logger.info(
            "MVP CpG subset applied: %s -> %s rows, first %s CpGs per chromosome.",
            before_rows,
            len(self.methylation_df),
            cpg_per_chr,
        )

    def methylation_dataframe_drop_sample(
        self,
        max_low_coverage_ratio: float = 0.5,
        output_data_info_file_name: str = "data_info.txt",
    ) -> None:
        """
        根据 coverage 缺失比例筛选样本，并同步删除未通过 QC 样本的矩阵列。
        Filter samples by low-coverage ratio and remove columns belonging to
        samples that fail QC from the methylation matrix.

        主要意图是保证后续模型输出只包含通过样本级 QC 的组织/细胞，同时在
        `dataset_info.tsv` 中保留原始 `dataset_index` 到 `model_output_index`
        的映射关系。
        The main intent is to keep only sample-level-QC-passed tissues/cells in
        model outputs while preserving the mapping from original `dataset_index`
        to `model_output_index` in `dataset_info.tsv`.
        """

        coverage_col_name_list = [col_name for col_name in self.methylation_df.columns if str(col_name).startswith("coverage_")]
        if not coverage_col_name_list:
            raise ValueError("No coverage columns found. Expected columns starting with 'coverage_'.")
        if len(coverage_col_name_list) != len(self.data_info_df):
            raise ValueError(
                "The number of coverage columns does not match the number of data info rows: "
                f"{len(coverage_col_name_list)} vs {len(self.data_info_df)}."
            )

        coverage_df = self.methylation_df[coverage_col_name_list]
        low_coverage_series = (coverage_df < self.minimal_coverage).sum(axis=0).reset_index(drop=True)
        coverage_summary_col = f"coverage_lower_{self.minimal_coverage}"
        self.data_info_df[coverage_summary_col] = low_coverage_series

        max_low_coverage_threshold = int(self.methylation_df.shape[0] * max_low_coverage_ratio)
        self.data_info_df["is_pass_qc"] = np.where(
            self.data_info_df[coverage_summary_col] < max_low_coverage_threshold,
            "yes",
            "no",
        )

        self.data_info_df["dataset_index"] = self.data_info_df["dataset_index"].astype(str)
        keep_col_postfix_list = self.data_info_df.loc[
            self.data_info_df["is_pass_qc"] == "yes",
            "dataset_index",
        ].to_list()
        keep_col_postfix_tuple = tuple(f"_{index}" for index in keep_col_postfix_list)
        keep_col_list = self.methylation_df.columns[
            self.methylation_df.columns.astype(str).str.endswith(keep_col_postfix_tuple)
        ].to_list()
        keep_col_list = self.methylation_df.columns[0:3].to_list() + keep_col_list
        self.methylation_df = cast(pd.DataFrame, self.methylation_df.loc[:, keep_col_list].copy())

        self.data_info_df["model_output_index"] = (self.data_info_df["is_pass_qc"] == "yes").cumsum() - 1
        no_keep_index = self.data_info_df[self.data_info_df["is_pass_qc"] == "no"].index
        self.data_info_df.loc[no_keep_index, "model_output_index"] = -1

        output_file = self._check_can_write(self._build_output_path(output_data_info_file_name))
        self.logger.info("Write dataset info: %s", output_file)
        self.data_info_df.to_csv(output_file, sep="\t", index=False)

    @staticmethod
    def _normalize_window_sizes(window_size_list: Sequence[int] | None) -> list[int]:
        if window_size_list is None:
            return [1000, 500, 200]
        return sorted(
            [item for item in window_size_list if isinstance(item, int) and not isinstance(item, bool)],
            reverse=True,
        )

    @staticmethod
    def _validate_sorted_chr_dataframe(chr_name: str, start_array: np.ndarray, end_array: np.ndarray) -> None:
        if np.any(np.diff(start_array) < 0):
            raise ValueError(f"CpG start positions are not sorted within chromosome {chr_name}.")
        if np.any(np.diff(end_array) < 0):
            raise ValueError(f"CpG end positions are not sorted within chromosome {chr_name}.")

    def calculate_regional_methylation(
        self,
        window_size_list: Sequence[int] | None = None,
        regional_methylation_dtype: RegionalDtype = "float32",
    ) -> None:
        """
        使用每条染色体的坐标二分查找和前缀和计算区域甲基化。
        Calculate regional methylation with per-chromosome coordinate binary
        search and prefix sums.

        默认 `float32` 速度更快且内存占用更低；与旧版逐窗口 `nanmean`
        的差异通常在 10^-7 量级。`float64` 仅建议用于 reproducibility
        分析，会明显增加运行时间和内存占用。
        The default `float32` is faster and uses less memory; differences from
        the old per-window `nanmean` implementation are usually around 1e-7.
        `float64` is intended only for reproducibility analysis and increases
        runtime and memory use.
        """

        window_size_list = self._normalize_window_sizes(window_size_list)
        if not window_size_list:
            self.logger.info("No valid regional methylation window sizes were provided; skip calculation.")
            return

        compute_dtype = np.float32 if regional_methylation_dtype == "float32" else np.float64
        if regional_methylation_dtype == "float64":
            self.logger.warning(
                "regional_methylation_dtype=float64 is intended for reproducibility analysis only "
                "and may noticeably increase runtime and memory use."
            )

        raw_col_name_list = [col_name for col_name in self.methylation_df.columns if str(col_name).startswith("raw_")]
        coverage_col_name_list = [
            col_name for col_name in self.methylation_df.columns if str(col_name).startswith("coverage_")
        ]
        if not raw_col_name_list:
            raise ValueError("No raw methylation columns found. Expected columns starting with 'raw_'.")
        if len(raw_col_name_list) != len(coverage_col_name_list):
            raise ValueError(
                "The number of raw methylation columns must match the number of coverage columns. "
                f"Found {len(raw_col_name_list)} raw columns and {len(coverage_col_name_list)} coverage columns."
            )

        extend_length_list = [(size - self.cpg_length) // 2 for size in window_size_list]
        row_number = len(self.methylation_df)
        sample_number = len(raw_col_name_list)
        window_methylation_by_size = {
            window_size: np.full((row_number, sample_number), np.nan, dtype=compute_dtype)
            for window_size in window_size_list
        }

        chr_array = self.methylation_df["chr"].to_numpy()
        start_all = self.methylation_df["start"].to_numpy(dtype=np.int64)
        end_all = self.methylation_df["end"].to_numpy(dtype=np.int64)
        start_time = datetime.datetime.now()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            for chr_name in pd.unique(self.methylation_df["chr"]):
                row_position = np.flatnonzero(chr_array == chr_name)
                start_array = start_all[row_position]
                end_array = end_all[row_position]
                self._validate_sorted_chr_dataframe(str(chr_name), start_array, end_array)

                chr_index = self.methylation_df.index[row_position]
                raw_array = self.methylation_df.loc[chr_index, raw_col_name_list].to_numpy(dtype=compute_dtype)
                coverage_array = self.methylation_df.loc[chr_index, coverage_col_name_list].to_numpy(
                    dtype=compute_dtype
                )
                valid_raw_array = raw_array.copy()
                valid_raw_array[coverage_array < self.minimal_coverage] = np.nan
                valid_mask = ~np.isnan(valid_raw_array)

                # 将 NaN 替换为 0 后做前缀和，同时单独累计有效观测数量。
                # Replace NaN with 0 for prefix sums and track valid counts
                # separately.
                value_for_sum = np.where(valid_mask, valid_raw_array, compute_dtype(0))
                sum_prefix = np.vstack(
                    [
                        np.zeros((1, sample_number), dtype=compute_dtype),
                        np.cumsum(value_for_sum, axis=0, dtype=compute_dtype),
                    ]
                )
                count_prefix = np.vstack(
                    [
                        np.zeros((1, sample_number), dtype=np.int64),
                        np.cumsum(valid_mask.astype(np.int64, copy=False), axis=0),
                    ]
                )

                for window_size, extend_length in zip(window_size_list, extend_length_list, strict=True):
                    window_start = start_array - extend_length
                    window_end = end_array + extend_length
                    left_index = np.searchsorted(start_array, window_start, side="right")
                    right_index = np.searchsorted(end_array, window_end, side="left")
                    if np.any(right_index < left_index):
                        raise ValueError(f"Invalid window bounds found within chromosome {chr_name}.")

                    window_sum = sum_prefix[right_index] - sum_prefix[left_index]
                    window_count = count_prefix[right_index] - count_prefix[left_index]
                    mean_array = np.divide(
                        window_sum,
                        window_count,
                        out=np.full_like(window_sum, np.nan, dtype=compute_dtype),
                        where=window_count > 0,
                    )
                    window_methylation_by_size[window_size][row_position, :] = mean_array

                if self.is_verbose:
                    using_time = datetime.datetime.now() - start_time
                    self.logger.info(
                        "Regional methylation finished for %s (%s CpGs). Elapsed: %s",
                        chr_name,
                        len(row_position),
                        using_time,
                    )

        window_dataframes = []
        for window_size in window_size_list:
            window_methylation_col_name = [
                f"window_{window_size}{col_name[3:]}" for col_name in raw_col_name_list
            ]
            window_dataframes.append(
                pd.DataFrame(
                    window_methylation_by_size[window_size],
                    columns=pd.Index(window_methylation_col_name),
                    index=self.methylation_df.index,
                )
            )

        self.methylation_df = cast(pd.DataFrame, pd.concat([self.methylation_df, *window_dataframes], axis=1))
        self.logger.info("Regional methylation columns added: %s", sum(len(df.columns) for df in window_dataframes))

    def methylation_dataframe_fill_na(self) -> None:
        for col_name in self.methylation_df.columns:
            if str(col_name).startswith("raw_") or str(col_name).startswith("window_"):
                self.methylation_df[col_name] = self.methylation_df[col_name].fillna(-1.0)

    def calculate_input_dna_coordinate(self) -> None:
        extend_length = (self.model_input_dna_length - self.cpg_length) // 2
        self.methylation_df["input_dna_start"] = self.methylation_df["start"] - extend_length
        self.methylation_df["input_dna_end"] = self.methylation_df["end"] + extend_length

        before_rows = len(self.methylation_df)
        self.methylation_df = cast(
            pd.DataFrame,
            pd.merge(self.methylation_df, self.chromosome_size_df, on="chr", how="left"),
        )
        if bool(self.methylation_df["chr_length"].isna().any()):
            missing_chr = sorted(self.methylation_df.loc[self.methylation_df["chr_length"].isna(), "chr"].unique())
            raise ValueError(f"Chromosome size is missing for: {missing_chr}")
        self.methylation_df = cast(
            pd.DataFrame,
            self.methylation_df[
                (self.methylation_df["input_dna_start"] > 0)
                & (self.methylation_df["input_dna_end"] < self.methylation_df["chr_length"])
            ].copy(),
        )
        self.methylation_df.reset_index(drop=True, inplace=True)
        self.logger.info("Input DNA coordinate filter: %s -> %s rows.", before_rows, len(self.methylation_df))

    def count_input_dna_n_base_number(self) -> None:
        """
        使用每条染色体 FASTA 的 N 前缀和批量统计输入序列中的 N 数量。
        Count N bases in input sequences using per-chromosome FASTA N-prefix sums.
        """

        n_number_array = np.empty(len(self.methylation_df), dtype=np.int32)
        chr_array = self.methylation_df["chr"].to_numpy()
        start_all = self.methylation_df["input_dna_start"].to_numpy(dtype=np.int64)
        end_all = self.methylation_df["input_dna_end"].to_numpy(dtype=np.int64)
        start_time = datetime.datetime.now()

        for chr_name in pd.unique(self.methylation_df["chr"]):
            row_position = np.flatnonzero(chr_array == chr_name)
            n_number_array[row_position] = self.genome_fasta.get_n_number_array(
                str(chr_name),
                start_all[row_position],
                end_all[row_position],
            )
            if self.is_verbose:
                self.logger.info(
                    "N-base count finished for %s (%s intervals).",
                    chr_name,
                    len(row_position),
                )

        self.methylation_df["N_number"] = n_number_array
        self.logger.info("N-base counting elapsed: %s", datetime.datetime.now() - start_time)

    def count_missing_sample(self) -> None:
        coverage_col_name_list = [col_name for col_name in self.methylation_df.columns if str(col_name).startswith("coverage_")]
        if not coverage_col_name_list:
            raise ValueError("No coverage columns found. Expected columns starting with 'coverage_'.")
        missing_values_number = (self.methylation_df[coverage_col_name_list] < self.minimal_coverage).sum(axis=1)
        self.methylation_df["missing_sample"] = missing_values_number

    def reset_methylation_df_col_order(self) -> None:
        # 将坐标和 QC 辅助列放在前部，甲基化矩阵列保留在后部。
        # Move coordinate and QC helper columns near the front while keeping
        # methylation matrix columns after them.
        insert_col_number = 3
        last_col_number = -5
        begin_col = self.methylation_df.columns[:insert_col_number].to_list()
        end_col = self.methylation_df.columns[last_col_number:].to_list()
        remain_col = self.methylation_df.columns[insert_col_number:last_col_number].to_list()
        self.methylation_df = cast(pd.DataFrame, self.methylation_df.loc[:, begin_col + end_col + remain_col])

    def trim_methylation_df(self, max_n_base_ratio: float, max_missing_sample_ratio: float) -> None:
        before_rows = len(self.methylation_df)
        n_base_max_number = self.model_input_dna_length * max_n_base_ratio
        self.methylation_df = cast(
            pd.DataFrame,
            self.methylation_df[self.methylation_df["N_number"] <= n_base_max_number].copy(),
        )

        smooth_sample_number = sum(1 for col_name in self.methylation_df.columns if str(col_name).startswith("smooth_"))
        missing_value_max_number = smooth_sample_number * max_missing_sample_ratio
        self.methylation_df = cast(
            pd.DataFrame,
            self.methylation_df[self.methylation_df["missing_sample"] <= missing_value_max_number].copy(),
        )
        self.methylation_df.reset_index(drop=True, inplace=True)
        self.logger.info("CpG QC filter: %s -> %s rows.", before_rows, len(self.methylation_df))

    def output_train_validation_test_set(
        self,
        train_chr_list: list[str],
        validation_chr_list: list[str],
        test_chr_list: list[str],
        output_sampled_train_set_fraction_list: list[float],
        is_output_slice_train_set: bool,
        output_format: DatasetFormat = "pickle",
    ) -> None:
        train_set_df = cast(pd.DataFrame, self.methylation_df[self.methylation_df["chr"].isin(train_chr_list)])
        self.output_dataset_df(train_set_df, f"{self.output_prefix}_train_set", output_format)

        validation_set_df = cast(
            pd.DataFrame,
            self.methylation_df[self.methylation_df["chr"].isin(validation_chr_list)],
        )
        self.output_dataset_df(validation_set_df, f"{self.output_prefix}_validation_set", output_format)

        test_set_df = cast(pd.DataFrame, self.methylation_df[self.methylation_df["chr"].isin(test_chr_list)])
        self.output_dataset_df(test_set_df, f"{self.output_prefix}_test_set", output_format)

        if output_sampled_train_set_fraction_list:
            self.output_sampled_train_set(
                train_set_df,
                f"{self.output_prefix}_train_set",
                output_sampled_train_set_fraction_list,
                output_format,
            )

        if is_output_slice_train_set:
            self.output_slice_train_set(f"{self.output_prefix}_train_set", train_set_df)

    def output_sampled_train_set(
        self,
        train_set_df: pd.DataFrame,
        output_prefix: str,
        fraction_list: list[float],
        output_format: DatasetFormat = "pickle",
        random_state: int = 42,
    ) -> None:
        shuffled_train_set_df = train_set_df.sample(frac=1, ignore_index=True, random_state=random_state)
        train_set_length = len(shuffled_train_set_df)
        for frac in fraction_list:
            if not 0 < frac <= 1:
                raise ValueError(f"Training set sampling fraction must be in (0, 1], got {frac}.")
            train_length = int(train_set_length * frac)
            frac_train_set_df = shuffled_train_set_df.iloc[0:train_length]
            output_file = f"{output_prefix}_fraction_{frac}"
            self.output_dataset_df(frac_train_set_df, output_file, output_format)

    def output_slice_train_set(self, output_folder: str, train_set_df: pd.DataFrame) -> None:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        train_set_df = train_set_df.reset_index(drop=True)
        train_dataset_len = len(train_set_df)
        if train_dataset_len == 0:
            raise ValueError("Training set is empty; cannot output sliced training set.")

        first_output_file = self._check_can_write(output_path / "0.pkl")
        train_set_df.loc[0, :].to_pickle(first_output_file)
        train_set_df.columns = range(0, len(train_set_df.columns))

        start_time = datetime.datetime.now()
        for row_index in range(1, train_dataset_len):
            output_file = self._check_can_write(output_path / f"{row_index}.pkl")
            train_set_df.loc[row_index, :].to_pickle(output_file)
            if row_index % 1_000_000 == 0 and self.is_verbose:
                self.logger.info(
                    "Slice train set output: %8d|%8d, elapsed: %s",
                    row_index,
                    train_dataset_len,
                    datetime.datetime.now() - start_time,
                )

    def output_dataset_df(
        self,
        dataset_df: pd.DataFrame,
        output_file: str,
        output_format: DatasetFormat = "pickle",
    ) -> None:
        if output_format == "feather":
            output_path = self._check_can_write(f"{output_file}.feather")
            self.logger.info("Write dataset: %s", output_path)
            dataset_df.reset_index(drop=True).to_feather(output_path, compression="zstd", version=2)
            return

        output_path = self._check_can_write(f"{output_file}.pkl")
        self.logger.info("Write dataset: %s", output_path)
        dataset_df.to_pickle(output_path)

    def output_complete_dataset(self, output_format: CompleteDatasetFormat = "tsv") -> None:
        if output_format == "feather":
            output_path = self._check_can_write(self._build_output_path("complete_dataset.feather"))
            self.logger.info("Write complete dataset: %s", output_path)
            self.methylation_df.reset_index(drop=True).to_feather(output_path, compression="zstd", version=2)
            return

        output_suffix = "complete_dataset.tsv.gz" if output_format == "tsv.gz" else "complete_dataset.tsv"
        output_path = self._check_can_write(self._build_output_path(output_suffix))
        self.logger.info("Write complete dataset: %s", output_path)
        self.methylation_df.to_csv(output_path, sep="\t", index=False)

    def output_methylation_df(self, output_file: str) -> None:
        output_path = self._check_can_write(Path(f"{self.output_prefix}_{output_file}"))
        self.logger.info("Write methylation dataframe: %s", output_path)
        suffixes = [suffix.lower() for suffix in output_path.suffixes]
        if ".feather" in suffixes:
            self.methylation_df.reset_index(drop=True).to_feather(output_path, compression="zstd", version=2)
        else:
            self.methylation_df.to_csv(output_path, sep="\t", index=False)
