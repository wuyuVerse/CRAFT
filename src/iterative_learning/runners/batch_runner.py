"""
批量任务执行器

负责并发执行多个任务。
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from loguru import logger

from tau2.data_model.tasks import Task
from tau2.registry import registry

from ..data.models import TaskResult
from .task_runner import TaskRunner


class BatchRunner:
    """批量任务执行器"""

    def __init__(
        self,
        domain: str,
        agent_llm: str,
        user_llm: str,
        analysis_llm: str,
        max_attempts: int = 5,
        max_steps: int = 100,
        llm_args: Optional[dict] = None,
        output_dir: Optional[str] = None,
        max_concurrency: int = 10,
        analysis_concurrency: int = 3,
    ):
        """
        Args:
            domain: 领域名称
            agent_llm: Agent 使用的 LLM
            user_llm: User Simulator 使用的 LLM
            analysis_llm: 分析使用的 LLM
            max_attempts: 每个任务的最大尝试次数
            max_steps: 每次尝试的最大步数
            llm_args: LLM 额外参数
            output_dir: 输出目录
            max_concurrency: 任务并发数
            analysis_concurrency: 分析任务并发数
        """
        self.domain = domain
        self.agent_llm = agent_llm
        self.user_llm = user_llm
        self.analysis_llm = analysis_llm
        self.max_attempts = max_attempts
        self.max_steps = max_steps
        self.llm_args = llm_args or {}
        self.output_dir = output_dir
        self.max_concurrency = max_concurrency
        self.analysis_concurrency = analysis_concurrency

    def load_tasks(self, task_ids: Optional[List[str]] = None, num_trials: int = 1) -> List[Task]:
        """
        加载任务。
        
        Args:
            task_ids: 指定的任务 ID 列表（可选）
            num_trials: 每个任务重复次数
            
        Returns:
            任务列表
        """
        loader = registry.get_tasks_loader(self.domain)
        tasks = loader()
        
        if task_ids:
            tasks = [task for idx, task in enumerate(tasks) if str(idx) in task_ids]
            logger.info(f"筛选任务 IDs: {task_ids}")
        
        tasks = tasks * num_trials
        logger.info(f"加载了 {len(tasks)} 个任务 (重复 {num_trials} 次)")
        
        return tasks

    def load_completed_tasks(self) -> set:
        """
        从 checkpoint 文件加载已完成的任务。
        
        Returns:
            已完成任务的 (task_id, trial_idx) 集合
        """
        if not self.output_dir:
            return set()
        
        checkpoint_file = Path(self.output_dir) / "checkpoint.json"
        if not checkpoint_file.exists():
            return set()
        
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            completed = set(tuple(item) for item in data.get("completed", []))
            logger.info(f"从 checkpoint 加载了 {len(completed)} 个已完成任务")
            return completed
        except Exception as e:
            logger.warning(f"加载 checkpoint 失败: {e}")
            return set()

    def save_checkpoint(self, completed: set):
        """
        保存 checkpoint。
        
        Args:
            completed: 已完成任务的 (task_id, trial_idx) 集合
        """
        if not self.output_dir:
            return
        
        checkpoint_file = Path(self.output_dir) / "checkpoint.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump({"completed": list(completed)}, f)
        except Exception as e:
            logger.warning(f"保存 checkpoint 失败: {e}")

    def run(self, tasks: List[Task], resume: bool = False) -> List[TaskResult]:
        """
        批量执行任务。
        
        Args:
            tasks: 任务列表
            resume: 是否断点续传
            
        Returns:
            任务结果列表
        """
        # 加载已完成的任务
        completed = self.load_completed_tasks() if resume else set()
        
        # 为每个任务分配 trial_idx
        tasks_with_idx = []
        task_count = {}
        for task in tasks:
            trial_idx = task_count.get(task.id, 0)
            task_count[task.id] = trial_idx + 1
            tasks_with_idx.append((task, trial_idx))
        
        # 过滤掉已完成的任务
        if resume and completed:
            pending = [(t, idx) for t, idx in tasks_with_idx if (t.id, idx) not in completed]
            logger.info(f"断点续传: 跳过 {len(tasks_with_idx) - len(pending)} 个已完成任务")
            tasks_with_idx = pending
        
        logger.info(f"开始执行 {len(tasks_with_idx)} 个任务，最大并发数: {self.max_concurrency}")

        def run_single(task_info: tuple) -> TaskResult:
            task, trial_idx = task_info
            runner = TaskRunner(
                domain=self.domain,
                agent_llm=self.agent_llm,
                user_llm=self.user_llm,
                analysis_llm=self.analysis_llm,
                max_attempts=self.max_attempts,
                max_steps=self.max_steps,
                llm_args=self.llm_args,
                output_dir=self.output_dir,
                analysis_concurrency=self.analysis_concurrency,
            )
            try:
                result = runner.run(task)
                # 标记完成并保存 checkpoint
                completed.add((task.id, trial_idx))
                self.save_checkpoint(completed)
                return result
            except Exception as e:
                logger.error(f"任务 {task.id} 执行失败: {e}")
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    num_attempts=0,
                    final_reward=0.0,
                    error=str(e),
                )

        results = []
        
        # 使用动态worker池模式：谁完成了就执行下一个任务
        from concurrent.futures import as_completed
        
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # 初始提交max_concurrency个任务
            pending_tasks = list(tasks_with_idx)
            active_futures = {}
            
            # 提交初始批次
            initial_batch = pending_tasks[:self.max_concurrency]
            for task_info in initial_batch:
                future = executor.submit(run_single, task_info)
                active_futures[future] = task_info
            
            pending_tasks = pending_tasks[self.max_concurrency:]
            completed_count = 0
            
            # 动态处理完成的任务并提交新任务
            while active_futures:
                # 等待任何一个任务完成
                for future in as_completed(active_futures):
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    logger.info(f"进度: {completed_count}/{len(tasks_with_idx)} - 完成任务: {result.task_id}")
                    
                    # 移除已完成的future
                    del active_futures[future]
                    
                    # 如果还有待处理的任务，立即提交下一个
                    if pending_tasks:
                        next_task = pending_tasks.pop(0)
                        new_future = executor.submit(run_single, next_task)
                        active_futures[new_future] = next_task
                    
                    # 只处理一个完成的任务，然后重新检查
                    break

        logger.info("所有任务执行完成！")
        return results

    def compute_statistics(self, results: List[TaskResult]) -> dict:
        """
        计算统计信息。
        
        Args:
            results: 任务结果列表
            
        Returns:
            统计信息字典
        """
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        total_attempts = sum(r.num_attempts for r in results)
        avg_attempts = total_attempts / total_tasks if total_tasks > 0 else 0.0

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": success_rate,
            "total_attempts": total_attempts,
            "avg_attempts_per_task": avg_attempts,
            "max_attempts_setting": self.max_attempts,
            "task_results": [asdict(r) for r in results],
        }

    def save_statistics(self, statistics: dict, output_path: Path):
        """
        保存统计信息。
        
        Args:
            statistics: 统计信息字典
            output_path: 输出路径
        """
        stats_file = output_path / "statistics.json"
        with open(stats_file, "w") as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"统计信息已保存到: {stats_file}")
        logger.info("=" * 60)
        logger.info(f"总任务数: {statistics['total_tasks']}")
        logger.info(f"成功任务: {statistics['successful_tasks']} ({statistics['success_rate']:.2%})")
        logger.info(f"失败任务: {statistics['failed_tasks']}")
        logger.info(f"总尝试次数: {statistics['total_attempts']}")
        logger.info(f"平均尝试次数: {statistics['avg_attempts_per_task']:.2f}")
        logger.info("=" * 60)
