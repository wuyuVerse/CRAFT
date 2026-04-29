"""
日志配置模块
"""

from pathlib import Path

from loguru import logger


def setup_logging(output_dir: Path, log_filename: str = "run.log"):
    """
    配置日志输出。
    
    Args:
        output_dir: 输出目录
        log_filename: 日志文件名
    """
    log_file = output_dir / log_filename

    # 移除默认处理器
    logger.remove()

    # 添加控制台输出
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    # 添加文件输出
    logger.add(
        log_file,
        rotation="100 MB",
        retention="10 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )

    logger.info(f"日志文件: {log_file}")
    
    return log_file
