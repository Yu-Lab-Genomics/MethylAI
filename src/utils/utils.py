import os
import glob
from pathlib import Path
import functools
import importlib
import sys
import polars as pl

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def debug_methods(cls):
    """类装饰器，自动为所有方法添加调试输出"""
    for name, method in cls.__dict__.items():
        if callable(method) and not name.startswith('_'):
            setattr(cls, name, log_function_call(method))
    return cls

def load_config(config_file, dict_name):
    """通过文件路径导入配置字典"""
    try:
        # 获取模块名
        module_name = os.path.splitext(os.path.basename(config_file))[0]
        # 从文件路径加载模块
        spec = importlib.util.spec_from_file_location(module_name, config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # 获取字典
        config_dict = getattr(module, dict_name)
        return config_dict
    except FileNotFoundError:
        print(f"错误: 找不到文件 {config_file}")
        sys.exit(1)
    except AttributeError as e:
        print(f"错误: 文件中未找到字典 {dict_name}")
        # 列出文件中所有不以_开头的属性
        available_dicts = [name for name in dir(module) if not name.startswith('_') and isinstance(getattr(module, name), dict)]
        print(f"可用的字典: {available_dicts}")
        sys.exit(1)

def check_output_folder(folder: str | Path) -> Path:
    """
    检查并创建输出文件夹。
    Check and create an output folder.

    Args:
        folder:
            需要创建或复用的输出文件夹路径。
            Path to the output folder that should be created or reused.

    Returns:
        标准化后的 Path 对象。
        Normalized Path object.
    """

    output_folder = Path(folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def check_output_path_empty(
    output_path: str | Path | None = None,
    output_prefix: str | Path | None = None,
    output_suffix: str | None = None,
) -> Path:
    """
    检查输出目录是否为空，或输出文件是否已经存在。
    Check whether an output directory is empty or output files already exist.

    支持两种输入方式：
    1. `output_prefix` + `output_suffix`，用于查找已存在的 `prefix*suffix` 匹配文件。
    2. `output_path`，用于直接传入完整文件名或输出目录。

    Two input modes are supported:
    1. `output_prefix` + `output_suffix`, used to find existing files matching
       `prefix*suffix`.
    2. `output_path`, used to pass a complete filename or an output directory.

    Args:
        output_path:
            完整输出文件名或输出目录路径。
            Complete output filename or output directory path.

        output_prefix:
            输出文件前缀或路径前缀，需要与 `output_suffix` 同时提供。函数会检查
            是否存在 `output_prefix*output_suffix` 形式的文件。
            Output file prefix or path prefix. Must be provided with
            `output_suffix`. The function checks whether files matching
            `output_prefix*output_suffix` already exist.

        output_suffix:
            输出文件后缀，需要与 `output_prefix` 同时提供。
            Output file suffix. Must be provided with `output_prefix`.

    Returns:
        标准化后的待检查路径。
        Normalized checked path.

    Raises:
        ValueError:
            当输入模式不完整或混用两种输入模式时抛出。
            Raised when input modes are incomplete or mixed.

        FileExistsError:
            当完整输出文件已存在、完整输出目录中已有文件，或存在匹配前缀和后缀的
            输出文件时抛出。
            Raised when a complete output file already exists, a complete output
            directory already contains files, or files matching the prefix and
            suffix already exist.
    """

    uses_output_path = output_path is not None
    uses_prefix_suffix = output_prefix is not None or output_suffix is not None
    if uses_output_path and uses_prefix_suffix:
        raise ValueError("Use either output_path or output_prefix + output_suffix, not both.")
    if not uses_output_path and not uses_prefix_suffix:
        raise ValueError("Provide either output_path or output_prefix + output_suffix.")
    if uses_prefix_suffix and (output_prefix is None or output_suffix is None):
        raise ValueError("output_prefix and output_suffix must be provided together.")

    if output_prefix is not None:
        assert output_suffix is not None
        suffix = output_suffix if output_suffix.startswith(".") else f".{output_suffix}"
        matched_files = sorted(
            Path(path)
            for path in glob.glob(f"{output_prefix}*{suffix}")
            if Path(path).is_file()
        )
        if matched_files:
            example_files = ", ".join(str(path) for path in matched_files[:5])
            raise FileExistsError(
                f"Output file(s) matching prefix and suffix already exist: {example_files}. "
                "Please change the output path or filename prefix."
            )
        return Path(output_prefix)

    assert output_path is not None
    checked_path = Path(output_path)

    if checked_path.exists() and checked_path.is_file():
        raise FileExistsError(
            f"Output file already exists: {checked_path}. "
            "Please change the output path or filename."
        )

    if checked_path.exists() and checked_path.is_dir():
        existing_files = [path for path in checked_path.rglob("*") if path.is_file()]
        if existing_files:
            example_files = ", ".join(str(path) for path in existing_files[:5])
            raise FileExistsError(
                f"Output directory is not empty: {checked_path}. "
                f"Existing file(s): {example_files}. "
                "Please change the output path."
            )

    return checked_path


def output_dataframe(dataframe: pl.DataFrame, output_file: str | Path) -> Path:
    """
    根据文件后缀输出 Polars DataFrame。
    Write a Polars DataFrame according to the output filename suffix.

    支持 `.tsv`、`.txt`、`.tab` 和 `.feather`。如果文件以 `.gz` 结尾，
    会继续向前识别真实数据格式，例如 `.tsv.gz`。
    Supported suffixes are `.tsv`, `.txt`, `.tab`, and `.feather`. If the
    filename ends with `.gz`, the data suffix before `.gz` is used, for example
    `.tsv.gz`.

    Args:
        dataframe:
            需要输出的 Polars DataFrame。
            Polars DataFrame to write.

        output_file:
            输出文件路径，函数会自动创建父目录。
            Output file path. Parent directories are created automatically.

    Returns:
        标准化后的输出文件 Path 对象。
        Normalized output Path object.

    Raises:
        ValueError:
            当文件后缀不是支持的格式时抛出。
            Raised when the file suffix is not supported.
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffixes = [suffix.lower() for suffix in output_path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    data_suffix = suffixes[-1] if suffixes else ""

    if data_suffix in {".tsv", ".txt", ".tab"}:
        dataframe.write_csv(output_path, separator="\t")
    elif data_suffix == ".feather":
        dataframe.write_ipc(output_path)
    else:
        supported_suffixes = ".tsv, .txt, .tab, .tsv.gz, .txt.gz, .tab.gz, .feather"
        raise ValueError(
            f"Unsupported output suffix for {output_path}. "
            f"Supported suffixes: {supported_suffixes}"
        )

    return output_path

