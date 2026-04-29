"""
数据模型定义
"""

from dataclasses import dataclass, field
from typing import List, Optional

from tau2.data_model.simulation import SimulationRun


@dataclass
class AttemptRecord:
    """单次尝试的记录"""
    attempt: int
    reward: float
    termination: str
    analysis_used: str
    simulation: SimulationRun


@dataclass
class TaskRecord:
    """任务执行记录"""
    task_id: str
    success: bool
    attempts: List[AttemptRecord]
    final_reward: float
    finished_at: str


@dataclass
class TaskResult:
    """任务执行结果统计"""
    task_id: str
    success: bool
    num_attempts: int
    final_reward: float
    error: Optional[str] = None
