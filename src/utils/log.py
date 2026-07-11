import logging
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    创建或复用一个 logger 实例。
    Create or reuse a logger instance.
    
    Args:
        name:
            logger 的名称，用于区分不同模块的日志记录器。
            The name of the logger, used to identify different logging modules.
            
        log_file:
            可选的日志文件路径。如果提供，则同时保存日志到文件。
            Optional path to the log file. If provided, logs will also be saved to the file.
            
        level:
            日志级别，例如 INFO、DEBUG 等。
            Logging level, such as INFO or DEBUG.
    
    Returns:
        配置完成的 logger 对象。
        A configured logger object.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 如果 logger 已经配置过 handler，直接返回避免重复添加
    # Return directly if handlers already exist to avoid adding duplicated handlers
    if logger.handlers:
        return logger

    # 定义统一的日志格式，包括时间、级别和消息内容
    # Define a unified log format including timestamp, level, and message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # 创建终端输出 handler
    # Create a stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 如果指定日志文件，则创建文件输出 handler
    # Create a file handler if a log file path is provided
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_block(logger: logging.Logger, title: str) -> None:
    """
    输出块状标题日志，用于区分不同运行阶段。

    Print a block-style log header to separate different execution stages.
    """
    line = "=" * 80
    logger.info(line)
    logger.info(title)
    logger.info(line)
