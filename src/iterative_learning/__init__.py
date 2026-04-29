"""
Iterative Learning Framework for tau2-bench

迭代学习框架：通过失败分析和重试机制提升 Agent 性能，并生成 SFT 训练数据。

模块结构:
- agents/: Agent 实现（带分析注入的 Agent）
- analysis/: 轨迹分析模块（失败分析、成功分析、对比分析）
- data/: 数据处理模块（轨迹格式化、SFT 数据生成）
- runners/: 任务执行器
- utils/: 工具函数
"""

__version__ = "0.1.0"
