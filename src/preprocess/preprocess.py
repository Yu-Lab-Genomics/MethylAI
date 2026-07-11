import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import polars as pl

from MethylAI.src.utils.utils import check_output_folder, check_output_path_empty, output_dataframe
from MethylAI.src.utils.log import setup_logger

class MethylationFile(ABC):
    """
    甲基化文件预处理基类。
    Base class for methylation file preprocessing.

    功能：
    1. 批量读取输入目录中的甲基化文件。
    2. 将正负链或输入坐标统一到参考 CpG 坐标。
    3. 输出预处理后的 mc/cov 表、运行日志、summary 和 manifest。

    Functions:
    1. Batch-read methylation files from an input directory.
    2. Harmonize strand-specific or input coordinates to reference CpG sites.
    3. Write preprocessed mc/cov tables, runtime logs, summary, and manifest.

    Args:
        input_folder:
            输入甲基化文件所在文件夹。
            Folder containing input methylation files.

        input_file_suffix:
            输入文件后缀，例如 `.bed.gz`。
            Input filename suffix, for example `.bed.gz`.

        output_folder:
            预处理数据输出文件夹。数据文件直接保存到该路径下。
            Output folder for preprocessed data files.

        output_file_format:
            预处理数据输出格式，可选 `tsv` 或 `feather`。
            Output format for preprocessed data, either `tsv` or `feather`.

        reference_cpg_coordinate_file:
            可选参考 CpG 坐标 BED 文件。如果提供，输出坐标与该文件对齐。
            Optional reference CpG BED file. If provided, output coordinates
            are aligned to this reference.

        chr_list:
            需要保留的染色体列表，默认保留 `chr1` 到 `chr22`。
            Chromosome list to retain. Defaults to `chr1` through `chr22`.

        metadata_folder_name:
            保存 log、summary 和 manifest 的子文件夹名称。
            Subfolder name for log, summary, and manifest files.

        output_summary_file:
            summary 文件名。
            Summary filename.

        output_manifest_file:
            manifest 文件名。
            Manifest filename.

        output_log_file:
            logger 运行日志文件名。
            Runtime logger filename.

        overwrite:
            是否允许覆盖已有输出文件。
            Whether existing output files may be overwritten.
    """

    def __init__(
        self,
        input_folder: str | Path,
        input_file_suffix: str,
        output_folder: str | Path,
        output_file_format: str = "tsv",
        reference_cpg_coordinate_file: str | Path | None = None,
        chr_list: list[str] | None = None,
        metadata_folder_name: str = "preprocess_metadata",
        output_summary_file: str = "preprocess_summary.tsv",
        output_manifest_file: str = "preprocess_manifest.tsv",
        output_log_file: str = "preprocess.log",
        overwrite: bool = False,
    ) -> None:
        self.input_folder = Path(input_folder)
        self.input_file_suffix = input_file_suffix
        if not self.input_file_suffix.startswith("."):
            self.input_file_suffix = f".{self.input_file_suffix}"

        self.output_file_format = self._normalize_output_file_format(output_file_format)
        output_data_suffix = self._get_preprocessed_output_suffix()
        if not overwrite:
            check_output_path_empty(
                output_prefix=f"{Path(output_folder)}/",
                output_suffix=output_data_suffix,
            )
        self.output_folder = check_output_folder(output_folder)
        self.metadata_folder = check_output_folder(self.output_folder / metadata_folder_name)
        self.output_summary_file = self.metadata_folder / output_summary_file
        self.output_manifest_file = self.metadata_folder / output_manifest_file
        self.output_log_file = self.metadata_folder / output_log_file
        self.overwrite = overwrite

        self.methylation_file = Path()
        self.methylation_pldf = pl.DataFrame()
        self.reference_cpg_coordinate_file = (
            Path(reference_cpg_coordinate_file)
            if reference_cpg_coordinate_file is not None
            else None
        )
        self.reference_cpg_coordinate_pldf = pl.DataFrame()
        self.chr_list = chr_list if chr_list is not None else [f"chr{i}" for i in range(1, 23)]

        self.current_summary: dict[str, Any] = {}
        self.summary_record_list: list[dict[str, Any]] = []
        self.manifest_record_list: list[dict[str, Any]] = []

        self._validate_input_settings()
        self._prepare_metadata_outputs()
        self.logger = setup_logger(
            f"methylai.preprocess.{id(self)}",
            log_file=self.output_log_file,
        )
        if self.reference_cpg_coordinate_file is not None:
            self._input_reference_cpg_coordinate()

    def _normalize_output_file_format(self, output_file_format: str) -> str:
        output_format = output_file_format.lower().strip(".")
        if output_format not in {"tsv", "feather"}:
            raise ValueError("--output_file_format must be either 'tsv' or 'feather'.")
        return output_format

    def _get_preprocessed_output_suffix(self) -> str:
        if self.output_file_format == "feather":
            return ".preprocessed.feather"
        return ".preprocessed.tsv"

    def _validate_input_settings(self) -> None:
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {self.input_folder}")
        if not self.input_folder.is_dir():
            raise NotADirectoryError(f"Input folder is not a directory: {self.input_folder}")
        if self.reference_cpg_coordinate_file is not None and not self.reference_cpg_coordinate_file.exists():
            raise FileNotFoundError(
                f"Reference CpG coordinate file does not exist: {self.reference_cpg_coordinate_file}"
            )

    def _prepare_metadata_outputs(self) -> None:
        metadata_files = [
            self.output_summary_file,
            self.output_manifest_file,
            self.output_log_file,
        ]
        existing_files = [path for path in metadata_files if path.exists()]
        if existing_files and not self.overwrite:
            existing_text = ", ".join(str(path) for path in existing_files)
            raise FileExistsError(
                f"Metadata output file(s) already exist: {existing_text}. "
                "Use --overwrite to replace them."
            )
        for path in existing_files:
            path.unlink()

    def _input_reference_cpg_coordinate(self) -> None:
        """
        读取参考基因组 CpG 坐标并保留常染色体。
        Read reference-genome CpG coordinates and retain autosomes.
        """

        if self.reference_cpg_coordinate_file is None:
            raise ValueError("reference_cpg_coordinate_file is required before reading reference CpG coordinates.")

        col_name_list = ["chr", "start", "end"]
        self.logger.info("Input reference CpG coordinate: %s", self.reference_cpg_coordinate_file)
        self.reference_cpg_coordinate_pldf = pl.read_csv(
            self.reference_cpg_coordinate_file,
            separator="\t",
            has_header=False,
            new_columns=col_name_list,
        )
        self.reference_cpg_coordinate_pldf = self.reference_cpg_coordinate_pldf.filter(
            pl.col("chr").is_in(self.chr_list)
        )
        self.logger.info(
            "Reference CpG coordinates retained: %s rows",
            self.reference_cpg_coordinate_pldf.height,
        )

    def preprocess_all_bed_file(self, max_files: int | None = None) -> None:
        """
        批量预处理输入目录中的甲基化文件。
        Batch-preprocess methylation files in the input folder.

        Args:
            max_files:
                可选的最大处理文件数，主要用于小规模测试。
                Optional maximum number of files to process, mainly for MVP tests.
        """

        methylation_file_list = sorted(
            Path(path)
            for path in glob.glob(str(self.input_folder / f"*{self.input_file_suffix}"))
        )
        if max_files is not None:
            methylation_file_list = methylation_file_list[:max_files]
        if not methylation_file_list:
            raise FileNotFoundError(
                f"No input files with suffix {self.input_file_suffix} found in {self.input_folder}"
            )

        self.logger.info("Found %s input file(s).", len(methylation_file_list))
        for self.methylation_file in methylation_file_list:
            self._reset_current_summary()
            self.logger.info("Input methylation file: %s", self.methylation_file)
            self._input_methylation_pldf()
            if self.reference_cpg_coordinate_file is not None:
                self._merge_forward_reverse_reference_cpg_coordinate()
            else:
                self._merge_forward_reverse()
            self._output_methylation_pldf_summary_and_manifest()

    @abstractmethod
    def _input_methylation_pldf(self) -> None:
        pass

    def _reset_current_summary(self) -> None:
        self.current_summary = {
            "input_file": str(self.methylation_file),
            "input_basename": self.methylation_file.name,
            "output_file": "",
            "output_format": self.output_file_format,
            "input_rows": None,
            "retained_rows": None,
            "output_rows": None,
            "before_merge_mc": None,
            "before_merge_cov": None,
            "after_merge_mc": None,
            "after_merge_cov": None,
            "note": "",
            "status": "started",
        }

    def _add_summary_note(self, note: str) -> None:
        existing_note = self.current_summary.get("note") or ""
        self.current_summary["note"] = note if not existing_note else f"{existing_note}; {note}"

    def _merge_forward_reverse(self) -> None:
        """
        在没有参考 CpG 坐标时合并正负链。
        Merge forward and reverse strands when no reference CpG coordinate is provided.
        """

        if "strand" not in self.methylation_pldf.columns:
            raise ValueError(
                "Input data do not contain a strand column. "
                "Please provide --reference_cpg_coordinate_file."
            )

        merge_on_col_name_list = ["chr", "start"]
        final_col_list = ["chr", "start", "end", "merge_mc", "merge_cov"]
        rename_final_col_list = ["chr", "start", "end", "mc", "cov"]

        self.methylation_pldf = self.methylation_pldf.sort(by=["chr", "start"])
        self._record_mc_cov_info_to_summary("before")

        forward_methylation_pldf = self.methylation_pldf.filter(pl.col("strand") == "+")
        reverse_methylation_pldf = self.methylation_pldf.filter(pl.col("strand") == "-")
        reverse_methylation_pldf = reverse_methylation_pldf.with_columns(
            (pl.col("start") - 1).alias("start")
        )

        merge_methylation_pldf = forward_methylation_pldf.join(
            reverse_methylation_pldf,
            on=merge_on_col_name_list,
            how="full",
            coalesce=True,
        )
        merge_methylation_pldf = merge_methylation_pldf.fill_null(pl.lit(0))
        merge_methylation_pldf = merge_methylation_pldf.with_columns(
            (pl.col("mc") + pl.col("mc_right")).alias("merge_mc"),
            (pl.col("cov") + pl.col("cov_right")).alias("merge_cov"),
            (pl.col("start") + 2).alias("end"),
        )
        self.methylation_pldf = merge_methylation_pldf[final_col_list]
        self.methylation_pldf.columns = rename_final_col_list
        self._record_mc_cov_info_to_summary("after")

    def _merge_forward_reverse_reference_cpg_coordinate(self) -> None:
        """
        按参考 CpG 坐标合并正负链甲基化计数。
        Merge forward and reverse methylation counts against reference CpG coordinates.
        """

        select_col_list = ["chr", "start", "mc", "cov"]
        join_on_col_list = ["chr", "start"]
        final_col_list = ["chr", "start", "end", "merge_mc", "merge_cov"]
        rename_final_col_list = ["chr", "start", "end", "mc", "cov"]

        self.methylation_pldf = self.methylation_pldf.sort(by=["chr", "start"])
        self._record_mc_cov_info_to_summary("before")

        forward_methylation_pldf = self.reference_cpg_coordinate_pldf.join(
            self.methylation_pldf,
            on=join_on_col_list,
            how="left",
            coalesce=True,
        )
        forward_methylation_pldf = forward_methylation_pldf[select_col_list]
        reverse_methylation_pldf = self.methylation_pldf.with_columns(
            (pl.col("start") - 1).alias("start")
        )
        reverse_methylation_pldf = self.reference_cpg_coordinate_pldf.join(
            reverse_methylation_pldf,
            on=join_on_col_list,
            how="left",
            coalesce=True,
        )
        reverse_methylation_pldf = reverse_methylation_pldf[select_col_list]

        merge_methylation_pldf = forward_methylation_pldf.join(
            reverse_methylation_pldf,
            on=join_on_col_list,
            how="full",
            coalesce=True,
        )
        merge_methylation_pldf = merge_methylation_pldf.fill_null(pl.lit(0))
        merge_methylation_pldf = merge_methylation_pldf.with_columns(
            (pl.col("mc") + pl.col("mc_right")).alias("merge_mc"),
            (pl.col("cov") + pl.col("cov_right")).alias("merge_cov"),
            (pl.col("start") + 2).alias("end"),
        )
        self.methylation_pldf = merge_methylation_pldf[final_col_list]
        self.methylation_pldf.columns = rename_final_col_list
        self._record_mc_cov_info_to_summary("after")

    def _record_mc_cov_info_to_summary(self, state: str) -> None:
        """
        记录合并前后的甲基化读数与覆盖度总和。
        Record methylated-read and coverage sums before and after merging.
        """

        mc_sum = self.methylation_pldf["mc"].sum()
        cov_sum = self.methylation_pldf["cov"].sum()
        self.current_summary[f"{state}_merge_mc"] = mc_sum
        self.current_summary[f"{state}_merge_cov"] = cov_sum
        self.logger.info("%s merge: mc=%s, cov=%s", state.capitalize(), mc_sum, cov_sum)

    def _build_output_file(self) -> Path:
        output_suffix = self._get_preprocessed_output_suffix()
        base_name = self.methylation_file.name.removesuffix(self.input_file_suffix)
        return self.output_folder / f"{base_name}{output_suffix}"

    def _output_methylation_pldf_summary_and_manifest(self) -> None:
        output_file = self._build_output_file()
        if not self.overwrite:
            output_suffix = self._get_preprocessed_output_suffix()
            output_prefix = str(output_file).removesuffix(output_suffix)
            check_output_path_empty(output_prefix=output_prefix, output_suffix=output_suffix)

        self.logger.info("Output preprocessed methylation data: %s", output_file)
        output_dataframe(self.methylation_pldf, output_file)

        self.current_summary["output_file"] = str(output_file)
        self.current_summary["output_rows"] = self.methylation_pldf.height
        self.current_summary["status"] = "finished"
        self.summary_record_list.append(dict(self.current_summary))

        self.manifest_record_list.append(
            {
                "sample_id": output_file.name.removesuffix(output_file.suffix),
                "input_file": str(self.methylation_file),
                "output_file": str(output_file),
                "output_format": self.output_file_format,
                "output_rows": self.methylation_pldf.height,
                "output_columns": ",".join(self.methylation_pldf.columns),
                "summary_file": str(self.output_summary_file),
                "log_file": str(self.output_log_file),
                "status": "finished",
            }
        )
        output_dataframe(pl.DataFrame(self.summary_record_list), self.output_summary_file)
        output_dataframe(pl.DataFrame(self.manifest_record_list), self.output_manifest_file)
        self.logger.info("Updated summary: %s", self.output_summary_file)
        self.logger.info("Updated manifest: %s", self.output_manifest_file)


class EncodeMethylationFile(MethylationFile):
    """
    ENCODE WGBS 甲基化文件预处理类。
    Preprocessor for ENCODE WGBS methylation files.

    功能：
    读取 ENCODE `.bed`/`.bed.gz` 甲基化文件，提取 CG 位点的覆盖度和甲基化读数，
    并输出统一坐标的 `chr/start/end/mc/cov` 数据表。

    Function:
    Read ENCODE `.bed`/`.bed.gz` methylation files, extract coverage and
    methylated-read counts for CG sites, and write coordinate-harmonized
    `chr/start/end/mc/cov` tables.

    参数说明继承自 `MethylationFile`。
    Parameters are inherited from `MethylationFile`.
    """

    def _input_methylation_pldf(self) -> None:
        col_11_name_list = [
            "chr",
            "start",
            "end",
            "V4",
            "V5",
            "strand",
            "V7",
            "V8",
            "V9",
            "cov",
            "mc_percent",
        ]
        col_name_list = [
            "chr",
            "start",
            "end",
            "V4",
            "V5",
            "strand",
            "V7",
            "V8",
            "V9",
            "cov",
            "mc_percent",
            "ref_genotype",
            "sample_genotype",
            "quality_score_genotype",
        ]

        self.methylation_pldf = pl.read_csv(
            self.methylation_file,
            separator="\t",
            has_header=False,
        )
        self.current_summary["input_rows"] = self.methylation_pldf.height
        self.logger.info("Input rows: %s", self.methylation_pldf.height)

        if self.methylation_pldf.width == 11:
            self._add_summary_note("input_has_11_columns")
            self.methylation_pldf.columns = col_11_name_list
        elif self.methylation_pldf.width == 14:
            self.methylation_pldf.columns = col_name_list
            self.methylation_pldf = self.methylation_pldf.filter(
                pl.col("sample_genotype").is_in(["CG"])
            )
        else:
            raise ValueError(
                f"Unexpected ENCODE column count for {self.methylation_file}: "
                f"{self.methylation_pldf.width}. Expected 11 or 14 columns."
            )

        self.methylation_pldf = self.methylation_pldf.filter(pl.col("chr").is_in(self.chr_list))
        self.current_summary["retained_rows"] = self.methylation_pldf.height
        self.logger.info("Rows retained after CG/autosome filtering: %s", self.methylation_pldf.height)

        self.methylation_pldf = self.methylation_pldf.with_columns(
            (pl.col("cov") * pl.col("mc_percent") / 100.0).round(0).alias("mc")
        )
        self.methylation_pldf = self.methylation_pldf[["chr", "start", "strand", "mc", "cov"]]
        self.methylation_pldf = self.methylation_pldf.cast({"mc": int, "start": int, "cov": int})


class BismarkMethylationFile(MethylationFile):
    """
    Bismark cov 文件预处理类。
    Preprocessor for Bismark cov-format methylation files.

    功能：
    读取 Bismark `*.cov` 类文件，计算覆盖度 `cov = mc + unmc`，并输出统一坐标的
    `chr/start/end/mc/cov` 数据表。

    Function:
    Read Bismark `*.cov`-style files, calculate `cov = mc + unmc`, and write
    coordinate-harmonized `chr/start/end/mc/cov` tables.

    参数说明继承自 `MethylationFile`。
    Parameters are inherited from `MethylationFile`.
    """

    def _input_methylation_pldf(self) -> None:
        col_name_list = ["chr", "start", "end", "mc_percent", "mc", "unmc"]
        self.methylation_pldf = pl.read_csv(
            self.methylation_file,
            separator="\t",
            has_header=False,
            infer_schema_length=100000,
        )
        self.current_summary["input_rows"] = self.methylation_pldf.height
        self.logger.info("Input rows: %s", self.methylation_pldf.height)

        if self.methylation_pldf.width != len(col_name_list):
            raise ValueError(
                f"Unexpected Bismark column count for {self.methylation_file}: "
                f"{self.methylation_pldf.width}. Expected {len(col_name_list)} columns."
            )
        self.methylation_pldf.columns = col_name_list

        self.methylation_pldf = self.methylation_pldf.filter(pl.col("chr").is_in(self.chr_list))
        self.current_summary["retained_rows"] = self.methylation_pldf.height
        self.logger.info("Rows retained after autosome filtering: %s", self.methylation_pldf.height)

        self.methylation_pldf = self.methylation_pldf.with_columns(
            (pl.col("mc") + pl.col("unmc")).alias("cov")
        )
        self.methylation_pldf = self.methylation_pldf[["chr", "start", "mc", "cov"]]
        self.methylation_pldf = self.methylation_pldf.cast({"mc": int, "start": int, "cov": int})



