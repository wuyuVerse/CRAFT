"""
Synthetic Data Generation Module

集成 tau-bench-gen 的合成数据生成功能，用于扩展训练数据的多样性和规模。
"""

# 导出主要接口（供用户使用）
from .runners.synthetic_runner import SyntheticRunner
from .runners.data_merger import DataMerger
from .runners.quality_filter import QualityFilter

# 导出核心类（供高级用户使用）
from .core.playground import PlayGround

__all__ = [
    # 主要接口
    'SyntheticRunner',
    'DataMerger',
    'QualityFilter',
    # 核心类
    'PlayGround',
]

__version__ = '1.0.0'
