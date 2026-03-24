"""日志系统模块"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "paper_rag",
    level: int = logging.INFO,
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    配置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志文件目录
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出

    Returns:
        配置好的 Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件 Handler
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 按日期生成日志文件
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 创建默认 logger
default_logger = setup_logger()


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 记录器名称，默认使用 'paper_rag'

    Returns:
        Logger 对象
    """
    if name is None:
        return default_logger
    return setup_logger(name)


# 便捷的日志函数
def debug(msg, *args, **kwargs):
    default_logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    default_logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    default_logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    default_logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    default_logger.critical(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    default_logger.exception(msg, *args, **kwargs)
