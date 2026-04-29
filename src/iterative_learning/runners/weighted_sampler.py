"""
加权任务采样器

根据任务难度和历史失败率进行加权采样。
"""

import random
from typing import List, Tuple, Optional, Dict

from loguru import logger

from tau2.data_model.tasks import Task

from ..analysis.task_classifier import TaskDifficultyClassifier, TaskDifficulty


class WeightedTaskSampler:
    """根据任务难度和历史失败率进行加权采样"""
    
    def __init__(
        self,
        domain: str,
        failure_history: Optional[Dict[str, dict]] = None,
    ):
        """
        Args:
            domain: 领域名称
            failure_history: 历史失败记录 {task_id: {"fail_rate": float, ...}}
        """
        self.domain = domain
        self.classifier = TaskDifficultyClassifier(failure_history)
        self.failure_history = failure_history or {}
    
    def sample(
        self,
        tasks: List[Task],
        num_samples: int,
        prioritize_weak: bool = True,
    ) -> List[Tuple[Task, TaskDifficulty]]:
        """
        加权采样任务
        
        Args:
            tasks: 任务列表
            num_samples: 采样数量
            prioritize_weak: 是否优先采样高难度任务
            
        Returns:
            [(task, difficulty), ...] - 任务和其难度信息
        """
        if not tasks:
            return []
        
        # 分类所有任务
        classified = [
            (task, self.classifier.classify(task, self.domain))
            for task in tasks
        ]
        
        if not prioritize_weak:
            # 均匀采样
            sampled = random.sample(classified, min(num_samples, len(classified)))
            return sampled
        
        # 计算采样权重
        weights = [diff.priority_weight for _, diff in classified]
        total_weight = sum(weights)
        
        if total_weight == 0:
            # 所有权重为0，均匀采样
            sampled_indices = random.sample(range(len(tasks)), min(num_samples, len(tasks)))
        else:
            # 归一化权重
            normalized = [w / total_weight for w in weights]
            
            # 加权采样（允许重复）
            sampled_indices = random.choices(
                range(len(tasks)),
                weights=normalized,
                k=num_samples
            )
        
        return [classified[i] for i in sampled_indices]
    
    def get_priority_queue(
        self,
        tasks: List[Task],
    ) -> List[Tuple[Task, TaskDifficulty]]:
        """
        获取按优先级排序的任务队列
        
        Args:
            tasks: 任务列表
            
        Returns:
            [(task, difficulty), ...] 按权重降序排列
        """
        classified = [
            (task, self.classifier.classify(task, self.domain))
            for task in tasks
        ]
        
        # 按权重降序排序
        classified.sort(key=lambda x: x[1].priority_weight, reverse=True)
        
        return classified
    
    def get_weak_tasks(
        self,
        tasks: List[Task],
        threshold: float = 2.0,
    ) -> List[Tuple[Task, TaskDifficulty]]:
        """
        获取高难度（弱点）任务
        
        Args:
            tasks: 任务列表
            threshold: 权重阈值
            
        Returns:
            权重 >= threshold 的任务列表
        """
        classified = [
            (task, self.classifier.classify(task, self.domain))
            for task in tasks
        ]
        
        weak_tasks = [
            (task, diff) for task, diff in classified
            if diff.priority_weight >= threshold
        ]
        
        logger.info(f"识别到 {len(weak_tasks)}/{len(tasks)} 个高难度任务 (threshold={threshold})")
        
        return weak_tasks
    
    def update_failure_history(
        self,
        task_id: str,
        success: bool,
        fail_tools: Optional[List[str]] = None,
    ):
        """
        更新失败历史
        
        Args:
            task_id: 任务ID
            success: 是否成功
            fail_tools: 失败涉及的工具
        """
        if task_id not in self.failure_history:
            self.failure_history[task_id] = {
                "attempts": 0,
                "failures": 0,
                "fail_rate": 0.0,
                "fail_tools": [],
            }
        
        history = self.failure_history[task_id]
        history["attempts"] += 1
        
        if not success:
            history["failures"] += 1
            if fail_tools:
                history["fail_tools"].extend(fail_tools)
        
        history["fail_rate"] = history["failures"] / history["attempts"]
        
        # 更新分类器的失败历史
        self.classifier.failure_history = self.failure_history
    
    def get_statistics(self) -> dict:
        """获取采样统计信息"""
        if not self.failure_history:
            return {"total_tasks": 0, "failed_tasks": 0, "avg_fail_rate": 0}
        
        total = len(self.failure_history)
        failed = sum(1 for h in self.failure_history.values() if h["failures"] > 0)
        avg_rate = sum(h["fail_rate"] for h in self.failure_history.values()) / total
        
        # 统计失败工具
        tool_failures = {}
        for h in self.failure_history.values():
            for tool in h.get("fail_tools", []):
                tool_failures[tool] = tool_failures.get(tool, 0) + 1
        
        return {
            "total_tasks": total,
            "failed_tasks": failed,
            "avg_fail_rate": avg_rate,
            "tool_failures": tool_failures,
        }
