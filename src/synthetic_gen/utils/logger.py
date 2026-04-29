"""
统一日志模块 - 所有日志同时输出到控制台和文件
"""
import logging
import os
from datetime import datetime
from pathlib import Path

# 默认日志目录（可以被覆盖）
_default_log_dir = os.environ.get("CRAFT_LOG_DIR", "output/logs")
_log_dir = None
_log_file = None
_logger = None

def setup_logger(output_dir: str = None, log_filename: str = None):
    """
    设置日志配置
    
    Args:
        output_dir: 输出目录，如果为None则使用默认目录
        log_filename: 日志文件名，如果为None则自动生成
    """
    global _log_dir, _log_file, _logger
    
    # 确定日志目录
    if output_dir:
        _log_dir = os.path.join(output_dir, "logs")
    else:
        _log_dir = _default_log_dir
    
    os.makedirs(_log_dir, exist_ok=True)
    
    # 确定日志文件名
    if log_filename is None:
        log_filename = f"synthetic_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    _log_file = os.path.join(_log_dir, log_filename)
    
    # 创建logger
    _logger = logging.getLogger("synthetic_gen")
    _logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    _logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(_log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    
    _logger.info(f"日志文件: {_log_file}")
    
    return _log_file

def get_logger():
    """获取logger实例"""
    global _logger
    if _logger is None:
        # 如果没有初始化，使用默认配置
        setup_logger()
    return _logger

def log(message: str, level: str = "info"):
    """统一日志函数"""
    logger = get_logger()
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)

def get_log_file():
    """获取当前日志文件路径"""
    global _log_file
    if _log_file is None:
        setup_logger()
    return _log_file
