"""
真实任务加载器 - 从tau-bench任务库加载真实任务作为Synthetic生成的seed

解决V0的问题：
- V0完全自由生成，导致action模式不符合真实场景
- V1使用真实任务作为约束，确保生成的对话符合实际需求
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from pathlib import Path


def log(msg):
    """简单日志函数"""
    print(f"[RealTaskLoader] {msg}")


class RealTaskLoader:
    """从tau-bench加载真实任务"""

    def __init__(self, tau_bench_data_dir: str = None):
        """
        Args:
            tau_bench_data_dir: tau-bench数据目录路径
        """
        default_data_dir = Path(os.environ.get("TAU2_DATA_DIR", "tau2-bench/data/tau2")) / "domains"
        self.data_dir = Path(tau_bench_data_dir) if tau_bench_data_dir else default_data_dir
        self.tasks_cache = {}
        self._load_all_tasks()

    def _load_all_tasks(self):
        """加载所有领域的任务"""
        domains = ["airline", "retail", "telecom"]

        for domain in domains:
            task_file = self.data_dir / domain / "tasks.json"
            if task_file.exists():
                try:
                    with open(task_file, 'r', encoding='utf-8') as f:
                        tasks = json.load(f)
                        self.tasks_cache[domain] = tasks
                        log(f"[RealTaskLoader] 加载 {domain} 任务: {len(tasks)}个")
                except Exception as e:
                    log(f"[错误] 加载 {domain} 任务失败: {e}")
                    self.tasks_cache[domain] = []
            else:
                log(f"[警告] {domain} 任务文件不存在: {task_file}")
                self.tasks_cache[domain] = []

    def get_random_task(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        随机获取一个任务

        Args:
            domain: 领域 (airline/retail/telecom)

        Returns:
            任务字典，如果没有任务则返回None
        """
        tasks = self.tasks_cache.get(domain, [])
        if not tasks:
            log(f"[警告] {domain} 没有可用任务")
            return None

        task = random.choice(tasks)
        log(f"[RealTaskLoader] 选择任务 {domain}/{task.get('id', 'unknown')}")
        return task

    def convert_task_to_seed(self, task: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        将tau-bench任务转换为synthetic生成的seed

        Args:
            task: tau-bench任务
            domain: 领域

        Returns:
            seed字典，包含user_info, task_description, expected_actions等
        """
        seed = {
            "domain": domain,
            "task_id": task.get("id", "unknown"),
        }

        # 提取用户场景信息
        user_scenario = task.get("user_scenario", {})
        instructions = user_scenario.get("instructions", {})

        # 用户信息
        known_info = instructions.get("known_info", "")
        reason_for_call = instructions.get("reason_for_call", "")
        task_instructions = instructions.get("task_instructions", "")

        # 组合成user_info
        seed["user_info"] = {
            "known_info": known_info,
            "reason_for_call": reason_for_call,
            "additional_instructions": task_instructions
        }

        # 任务描述
        description = task.get("description", {})
        seed["task_description"] = {
            "purpose": description.get("purpose", ""),
            "notes": description.get("notes", "")
        }

        # 期望的actions（如果有）
        evaluation_criteria = task.get("evaluation_criteria", {})
        expected_actions = evaluation_criteria.get("actions", [])

        if expected_actions:
            # 转换action格式
            seed["expected_actions"] = [
                {
                    "name": action.get("name", ""),
                    "arguments": action.get("arguments", {}),
                    "sequence_order": action.get("sequence_order", 999)
                }
                for action in expected_actions
            ]
        else:
            seed["expected_actions"] = []

        # NL断言（自然语言约束）
        nl_assertions = evaluation_criteria.get("nl_assertions", [])
        if nl_assertions:
            seed["constraints"] = nl_assertions

        log(f"[RealTaskLoader] 转换任务为seed: {seed['task_id']}, expected_actions={len(seed.get('expected_actions', []))}")

        return seed

    def get_random_seed(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        随机获取一个seed（完整流程）

        Args:
            domain: 领域

        Returns:
            seed字典
        """
        task = self.get_random_task(domain)
        if not task:
            return None

        seed = self.convert_task_to_seed(task, domain)
        return seed

    def get_seed_batch(self, domain: str, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        批量获取seeds

        Args:
            domain: 领域
            batch_size: 批次大小

        Returns:
            seed列表
        """
        seeds = []
        for _ in range(batch_size):
            seed = self.get_random_seed(domain)
            if seed:
                seeds.append(seed)

        log(f"[RealTaskLoader] 生成 {len(seeds)} 个seeds for {domain}")
        return seeds

    def format_seed_as_prompt(self, seed: Dict[str, Any]) -> str:
        """
        将seed格式化为prompt（用于生成对话时的约束）

        Args:
            seed: seed字典

        Returns:
            格式化的prompt字符串
        """
        prompt_parts = []

        prompt_parts.append("## Task Seed (MUST FOLLOW)")

        # 用户信息
        user_info = seed.get("user_info", {})
        if user_info.get("known_info"):
            prompt_parts.append(f"\n**User Known Info:**\n{user_info['known_info']}")

        if user_info.get("reason_for_call"):
            prompt_parts.append(f"\n**Reason for Call:**\n{user_info['reason_for_call']}")

        if user_info.get("additional_instructions"):
            prompt_parts.append(f"\n**Additional Instructions:**\n{user_info['additional_instructions']}")

        # 期望的actions
        expected_actions = seed.get("expected_actions", [])
        if expected_actions:
            prompt_parts.append("\n**Expected Action Sequence (YOU MUST FOLLOW):**")
            for action in expected_actions:
                name = action.get("name", "unknown")
                args = action.get("arguments", {})
                prompt_parts.append(f"  - {name}({', '.join(f'{k}={v}' for k, v in args.items())})")

        # 约束条件
        constraints = seed.get("constraints", [])
        if constraints:
            prompt_parts.append("\n**Constraints:**")
            for constraint in constraints:
                prompt_parts.append(f"  - {constraint}")

        # 任务描述
        task_desc = seed.get("task_description", {})
        if task_desc.get("purpose"):
            prompt_parts.append(f"\n**Task Purpose:**\n{task_desc['purpose']}")

        prompt = "\n".join(prompt_parts)
        return prompt
