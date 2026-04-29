"""
从tau2-bench提取任务模式

从tau2-bench的50个真实评测任务中提取可复用的模式，用于生成新的训练任务。
"""
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Tau2TaskPattern:
    """从真实tau2任务中提取的模式"""
    task_id: str                          # 原始任务ID
    domain: str                           # airline/retail/telecom
    task_type: str                        # cancellation/booking/inquiry等
    complexity: str                       # simple/medium/complex

    # 从user_scenario提取
    persona: Optional[str] = None         # 用户画像描述
    reason_for_call: str = ""             # 来电原因
    known_info: str = ""                  # 已知信息
    unknown_info: str = ""                # 未知信息
    task_instructions: str = ""           # 任务指令

    # 从evaluation_criteria提取
    expected_actions: List[Dict] = field(default_factory=list)  # 期望的工具调用序列
    tool_sequence: List[str] = field(default_factory=list)      # 工具名称序列
    nl_assertions: List[str] = field(default_factory=list)      # 自然语言断言
    reward_basis: List[str] = field(default_factory=list)       # 评估基础

    # 原始任务信息
    description: Dict = field(default_factory=dict)             # 任务描述

    # 统计信息
    frequency: float = 0.0                # 该模式在50个任务中的频率


class Tau2TaskExtractor:
    """从tau2-bench提取任务模式"""

    def __init__(self, tau2_data_dir: str = None):
        self.tau2_data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR", "tau2-bench/data/tau2")
        self._patterns_cache: Dict[str, List[Tau2TaskPattern]] = {}
        self._tasks_cache: Dict[str, List[Dict]] = {}

    def extract_patterns(self, domain: str) -> List[Tau2TaskPattern]:
        """
        提取所有任务模式（使用所有50个任务）

        Args:
            domain: 领域名称 (airline/retail/telecom)

        Returns:
            每个任务一个pattern的列表
        """
        if domain in self._patterns_cache:
            return self._patterns_cache[domain]

        # 加载tasks.json
        tasks = self._load_tasks(domain)

        patterns = []
        for task in tasks:
            pattern = self._extract_pattern_from_task(task, domain)
            patterns.append(pattern)

        # 计算频率
        total = len(patterns)
        for pattern in patterns:
            pattern.frequency = 1.0 / total if total > 0 else 0.0

        self._patterns_cache[domain] = patterns
        return patterns

    def _load_tasks(self, domain: str) -> List[Dict]:
        """
        加载所有任务（不过滤split）

        Args:
            domain: 领域名称

        Returns:
            任务列表
        """
        if domain in self._tasks_cache:
            return self._tasks_cache[domain]

        task_file = os.path.join(self.tau2_data_dir, "domains", domain, "tasks.json")

        if not os.path.exists(task_file):
            raise FileNotFoundError(f"Tasks file not found: {task_file}")

        with open(task_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

        self._tasks_cache[domain] = tasks
        return tasks

    def _extract_pattern_from_task(self, task: Dict, domain: str) -> Tau2TaskPattern:
        """
        从单个任务提取模式

        Args:
            task: tau2-bench任务对象
            domain: 领域名称

        Returns:
            提取的任务模式
        """
        user_scenario = task.get("user_scenario", {})
        instructions = user_scenario.get("instructions", {})
        eval_criteria = task.get("evaluation_criteria", {})
        actions = eval_criteria.get("actions", [])

        # 提取unknown_info
        unknown_info = instructions.get("unknown_info", "")
        if isinstance(unknown_info, dict):
            # 如果是字典格式，提取items
            unknown_info = ", ".join(unknown_info.get("items", []))

        return Tau2TaskPattern(
            task_id=task["id"],
            domain=domain,
            task_type=self._classify_task_type(instructions.get("reason_for_call", "")),
            complexity=self._classify_complexity(actions),

            persona=user_scenario.get("persona"),
            reason_for_call=instructions.get("reason_for_call", ""),
            known_info=instructions.get("known_info", ""),
            unknown_info=unknown_info,
            task_instructions=instructions.get("task_instructions", ""),

            expected_actions=actions,
            tool_sequence=[a["name"] for a in actions],
            nl_assertions=eval_criteria.get("nl_assertions", []),
            reward_basis=eval_criteria.get("reward_basis", []),

            description=task.get("description", {})
        )

    def _classify_task_type(self, reason: str) -> str:
        """
        根据reason_for_call分类任务类型

        Args:
            reason: 来电原因

        Returns:
            任务类型
        """
        reason_lower = reason.lower()

        if "cancel" in reason_lower:
            return "cancellation"
        elif "book" in reason_lower or "reservation" in reason_lower:
            return "booking"
        elif "modify" in reason_lower or "change" in reason_lower or "update" in reason_lower:
            return "modification"
        elif "check" in reason_lower or "status" in reason_lower or "track" in reason_lower or "query" in reason_lower:
            return "inquiry"
        elif "compensat" in reason_lower or "refund" in reason_lower or "delay" in reason_lower:
            return "compensation"
        elif "exchange" in reason_lower or "return" in reason_lower:
            return "exchange"
        elif "suspend" in reason_lower or "resume" in reason_lower:
            return "service_control"
        elif "payment" in reason_lower or "bill" in reason_lower:
            return "billing"
        else:
            return "other"

    def _classify_complexity(self, actions: List[Dict]) -> str:
        """
        根据actions数量分类复杂度

        Args:
            actions: 期望的工具调用列表

        Returns:
            复杂度级别 (simple/medium/complex)
        """
        num = len(actions)
        if num <= 2:
            return "simple"
        elif num <= 5:
            return "medium"
        else:
            return "complex"

    def get_task_type_distribution(self, domain: str) -> Dict[str, int]:
        """
        统计任务类型分布

        Args:
            domain: 领域名称

        Returns:
            任务类型到数量的映射
        """
        patterns = self.extract_patterns(domain)
        return dict(Counter(p.task_type for p in patterns))

    def get_complexity_distribution(self, domain: str) -> Dict[str, int]:
        """
        统计复杂度分布

        Args:
            domain: 领域名称

        Returns:
            复杂度到数量的映射
        """
        patterns = self.extract_patterns(domain)
        return dict(Counter(p.complexity for p in patterns))

    def get_tool_sequence_distribution(self, domain: str) -> Dict[tuple, int]:
        """
        统计工具序列分布

        Args:
            domain: 领域名称

        Returns:
            工具序列到数量的映射
        """
        patterns = self.extract_patterns(domain)
        return dict(Counter(tuple(p.tool_sequence) for p in patterns))

    def get_statistics(self, domain: str) -> Dict[str, Any]:
        """
        获取领域的完整统计信息

        Args:
            domain: 领域名称

        Returns:
            统计信息字典
        """
        patterns = self.extract_patterns(domain)

        return {
            "total_tasks": len(patterns),
            "task_type_distribution": self.get_task_type_distribution(domain),
            "complexity_distribution": self.get_complexity_distribution(domain),
            "tool_sequence_distribution": {
                str(k): v for k, v in self.get_tool_sequence_distribution(domain).items()
            },
            "avg_actions_per_task": sum(len(p.tool_sequence) for p in patterns) / len(patterns) if patterns else 0,
            "total_unique_tools": len(set(tool for p in patterns for tool in p.tool_sequence))
        }

    def print_summary(self, domain: str) -> None:
        """
        打印领域摘要信息

        Args:
            domain: 领域名称
        """
        stats = self.get_statistics(domain)

        print(f"\n{'='*60}")
        print(f"Tau2-Bench任务分析 - {domain.upper()}")
        print(f"{'='*60}")
        print(f"\n总任务数: {stats['total_tasks']}")

        print(f"\n任务类型分布:")
        for task_type, count in sorted(stats['task_type_distribution'].items(), key=lambda x: -x[1]):
            percentage = count / stats['total_tasks'] * 100
            print(f"  {task_type:20s}: {count:3d} ({percentage:5.1f}%)")

        print(f"\n复杂度分布:")
        for complexity, count in sorted(stats['complexity_distribution'].items(), key=lambda x: -x[1]):
            percentage = count / stats['total_tasks'] * 100
            print(f"  {complexity:20s}: {count:3d} ({percentage:5.1f}%)")

        print(f"\n平均工具调用数: {stats['avg_actions_per_task']:.2f}")
        print(f"唯一工具数量: {stats['total_unique_tools']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # 测试代码
    extractor = Tau2TaskExtractor()

    for domain in ["airline", "retail", "telecom"]:
        try:
            extractor.print_summary(domain)

            # 显示前3个任务模式
            patterns = extractor.extract_patterns(domain)
            print(f"示例任务模式 (前3个):")
            for i, pattern in enumerate(patterns[:3], 1):
                print(f"\n  [{i}] Task ID: {pattern.task_id}")
                print(f"      类型: {pattern.task_type} | 复杂度: {pattern.complexity}")
                print(f"      原因: {pattern.reason_for_call[:60]}...")
                print(f"      工具序列: {' -> '.join(pattern.tool_sequence)}")
        except Exception as e:
            print(f"Error processing {domain}: {e}")
