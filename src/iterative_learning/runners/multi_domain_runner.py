"""
多领域任务执行器

支持同时处理多个领域的任务，包含加权采样和任务难度分类。
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from tau2.data_model.tasks import Task
from tau2.registry import registry

from ..analysis.task_classifier import TaskDifficultyClassifier, TaskDifficulty
from ..data.models import TaskResult
from .task_runner import TaskRunner
from .enhanced_task_runner import EnhancedTaskRunner
from .weighted_sampler import WeightedTaskSampler
from ..injection import ErrorInjectionConfig


def load_tasks_from_file(file_path: str) -> List[Task]:
    """从文件加载任务（支持 JSON 和 JSONL 格式）"""
    tasks = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task_data = json.loads(line)
                    tasks.append(Task.model_validate(task_data))
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            task_list = json.load(f)
            for task_data in task_list:
                tasks.append(Task.model_validate(task_data))
    return tasks


class MultiDomainRunner:
    """多领域任务执行器"""

    def __init__(
        self,
        domains: List[str],
        agent_llm: str,
        user_llm: str,
        analysis_llm: str,
        max_attempts: int = 5,
        max_steps: int = 100,
        llm_args: Optional[dict] = None,
        output_dir: Optional[str] = None,
        max_concurrency: int = 10,
        analysis_concurrency: int = 3,
        enable_cot: bool = True,
        quality_threshold: float = 0.6,
        prioritize_weak: bool = True,
        enable_contrast_feedback: bool = True,
        enable_error_injection: bool = False,
        error_injection_base_rate: float = 0.12,
        error_injection_max_errors: int = 8,
        error_db_path: Optional[str] = None,
        # V4 新增参数
        injection_mode: str = "rule",  # "rule" (V3) 或 "agent" (V4)
        use_llm_for_recovery: bool = True,
        error_type_weights: Optional[dict] = None,
        api_base: Optional[str] = None,
        correct_trajectory_weight: int = 1,  # 正确轨迹重复次数
        save_analysis_data: bool = False,  # 是否保存分析数据
        include_failed_in_sft: bool = False,  # 是否将失败轨迹加入SFT数据
        # 自定义任务路径
        tasks_path: Optional[str] = None,  # 支持 synthetic 数据，格式: "dir_path" 或 "{domain}_path"
    ):
        self.domains = domains
        self.agent_llm = agent_llm
        self.user_llm = user_llm
        self.analysis_llm = analysis_llm
        self.max_attempts = max_attempts
        self.max_steps = max_steps
        self.llm_args = llm_args or {}
        self.output_dir = output_dir
        self.max_concurrency = max_concurrency
        self.analysis_concurrency = analysis_concurrency
        self.enable_cot = enable_cot
        self.quality_threshold = quality_threshold
        self.prioritize_weak = prioritize_weak
        self.enable_contrast_feedback = enable_contrast_feedback
        self.enable_error_injection = enable_error_injection
        self.error_injection_base_rate = error_injection_base_rate
        self.error_injection_max_errors = error_injection_max_errors
        self.error_db_path = error_db_path
        # V4 新增
        self.injection_mode = injection_mode
        self.use_llm_for_recovery = use_llm_for_recovery
        self.error_type_weights = error_type_weights
        self.api_base = api_base
        self.correct_trajectory_weight = correct_trajectory_weight
        self.save_analysis_data = save_analysis_data
        self.include_failed_in_sft = include_failed_in_sft
        self.tasks_path = tasks_path

        self.classifier = TaskDifficultyClassifier()
        self.samplers: Dict[str, WeightedTaskSampler] = {}

    def load_all_tasks(
        self,
        task_ids: Optional[List[str]] = None,
        num_trials: int = 1,
        num_trials_per_domain: Optional[Dict[str, int]] = None,
    ) -> List[tuple]:
        """
        加载所有领域的任务，并进行难度分类

        Args:
            task_ids: 任务ID列表
            num_trials: 默认重复次数
            num_trials_per_domain: 按领域设置重复次数（覆盖默认值）

        Returns:
            [(domain, task, task_idx, trial_idx, difficulty), ...] 列表
            任务按领域交错排列，便于并行执行时同时处理多个领域
        """
        # 按领域分别存储任务
        domain_tasks: Dict[str, List[tuple]] = {d: [] for d in self.domains}
        num_trials_per_domain = num_trials_per_domain or {}

        for domain in self.domains:
            if domain not in self.samplers:
                self.samplers[domain] = WeightedTaskSampler(domain)

            # 加载任务：优先使用自定义路径，否则使用registry
            tasks = self._load_domain_tasks(domain)

            # 保存原始索引
            if task_ids:
                indexed_tasks = [(idx, task) for idx, task in enumerate(tasks) if str(idx) in task_ids]
            else:
                indexed_tasks = list(enumerate(tasks))

            if self.prioritize_weak:
                # 需要修改get_priority_queue以支持indexed_tasks
                classified = []
                for idx, task in indexed_tasks:
                    difficulty = self.classifier.classify(task, domain)
                    classified.append((idx, task, difficulty))
            else:
                classified = [(idx, task, self.classifier.classify(task, domain)) for idx, task in indexed_tasks]

            # 获取该领域的重复次数（优先使用领域特定配置）
            domain_trials = num_trials_per_domain.get(domain, num_trials)

            for trial_idx in range(domain_trials):
                for idx, task, difficulty in classified:
                    domain_tasks[domain].append((domain, task, idx, trial_idx, difficulty))

            logger.info(f"加载 {domain} 领域: {len(indexed_tasks)} 个任务 × {domain_trials} 次 = {len(indexed_tasks) * domain_trials} 条数据")

        # 交错排列各领域的任务，使并行执行时能同时处理多个领域
        all_tasks = []
        domain_iterators = {d: iter(tasks) for d, tasks in domain_tasks.items()}
        active_domains = set(self.domains)

        while active_domains:
            for domain in list(active_domains):
                try:
                    task = next(domain_iterators[domain])
                    all_tasks.append(task)
                except StopIteration:
                    active_domains.discard(domain)

        self._log_difficulty_distribution(all_tasks)

        logger.info(f"总计加载了 {len(all_tasks)} 个任务（已按领域交错排列）")
        return all_tasks

    def _load_domain_tasks(self, domain: str) -> List[Task]:
        """
        加载领域任务

        优先使用自定义路径，否则使用registry加载默认任务。
        自定义路径支持两种格式:
        1. 目录路径：自动查找 {dir}/{domain}/synthetic_tasks_{domain}.json
        2. 文件路径：直接加载指定文件

        Args:
            domain: 领域名称

        Returns:
            Task列表
        """
        if self.tasks_path:
            tasks_path = Path(self.tasks_path)

            if tasks_path.is_dir():
                # 目录模式：查找 {dir}/{domain}/synthetic_tasks_{domain}.json
                domain_file = tasks_path / domain / f"synthetic_tasks_{domain}.json"
                if not domain_file.exists():
                    # 也尝试 jsonl 格式
                    domain_file = tasks_path / domain / "synthetic_data.jsonl"

                if domain_file.exists():
                    logger.info(f"从自定义路径加载 {domain} 任务: {domain_file}")
                    return load_tasks_from_file(str(domain_file))
                else:
                    logger.warning(f"自定义路径下未找到 {domain} 任务文件，使用默认任务")
            else:
                # 文件模式：直接加载
                if tasks_path.exists():
                    logger.info(f"从自定义文件加载任务: {tasks_path}")
                    return load_tasks_from_file(str(tasks_path))
                else:
                    logger.warning(f"自定义任务文件不存在: {tasks_path}，使用默认任务")

        # 默认：使用registry加载
        loader = registry.get_tasks_loader(domain)
        return loader()

    def _log_difficulty_distribution(self, all_tasks: List[tuple]):
        """记录任务难度分布"""
        stats = {
            "total": len(all_tasks),
            "multi_step": sum(1 for _, _, _, _, d in all_tasks if d.is_multi_step),
            "parameter_sensitive": sum(1 for _, _, _, _, d in all_tasks if d.is_parameter_sensitive),
            "context_complex": sum(1 for _, _, _, _, d in all_tasks if d.is_context_complex),
            "high_priority": sum(1 for _, _, _, _, d in all_tasks if d.priority_weight >= 3.0),
        }
        logger.info(f"任务难度分布: {stats}")

    def load_checkpoint(self) -> set:
        """加载 checkpoint"""
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
        """保存 checkpoint"""
        if not self.output_dir:
            return
        
        checkpoint_file = Path(self.output_dir) / "checkpoint.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump({"completed": list(completed)}, f)
        except Exception as e:
            logger.warning(f"保存 checkpoint 失败: {e}")

    def run(
        self,
        task_ids: Optional[List[str]] = None,
        num_trials: int = 1,
        num_trials_per_domain: Optional[Dict[str, int]] = None,
        resume: bool = False,
    ) -> Dict[str, List[TaskResult]]:
        """
        执行所有领域的任务
        
        Args:
            task_ids: 任务ID列表
            num_trials: 默认重复次数
            num_trials_per_domain: 按领域设置重复次数（覆盖默认值）
            resume: 是否断点续传
        
        Returns:
            {domain: [TaskResult, ...], ...}
        """
        all_tasks = self.load_all_tasks(task_ids, num_trials, num_trials_per_domain)
        completed = self.load_checkpoint() if resume else set()
        
        if resume and completed:
            pending = [
                (d, t, idx, trial, diff) for d, t, idx, trial, diff in all_tasks
                if (d, t.id, trial) not in completed
            ]
            logger.info(f"断点续传: 跳过 {len(all_tasks) - len(pending)} 个已完成任务")
            all_tasks = pending
        
        logger.info(f"开始执行 {len(all_tasks)} 个任务，最大并发数: {self.max_concurrency}")

        for domain in self.domains:
            if self.output_dir:
                domain_dir = Path(self.output_dir) / domain
                domain_dir.mkdir(parents=True, exist_ok=True)

        # 连续失败检测（用于检测远程服务宕机）
        import threading
        consecutive_failures = {"count": 0, "lock": threading.Lock()}
        MAX_CONSECUTIVE_FAILURES = 50  # 连续50个任务失败则终止

        class ServiceDownError(Exception):
            """远程服务宕机异常"""
            pass

        def run_single(task_info: tuple) -> tuple:
            domain, task, task_idx, trial_idx, difficulty = task_info
            
            domain_output = str(Path(self.output_dir) / domain) if self.output_dir else None
            
            # 根据是否启用错误注入选择不同的 Runner
            if self.enable_error_injection:
                # 创建错误注入配置对象
                error_injection_config = ErrorInjectionConfig(
                    enabled=True,
                    base_rate=self.error_injection_base_rate,
                    max_errors_per_task=self.error_injection_max_errors,
                    error_db_path=self.error_db_path,
                    correct_trajectory_weight=self.correct_trajectory_weight,
                    save_analysis_data=self.save_analysis_data,
                )
                
                runner = EnhancedTaskRunner(
                    error_injection_config=error_injection_config,
                    injection_mode=self.injection_mode,  # V4: "agent" 或 "rule"
                    use_llm_for_recovery=self.use_llm_for_recovery,  # V4
                    error_type_weights=self.error_type_weights,  # V4
                    api_base=self.api_base,  # V4: API地址
                    include_failed_in_sft=self.include_failed_in_sft,  # 是否将失败轨迹加入SFT
                    domain=domain,
                    agent_llm=self.agent_llm,
                    user_llm=self.user_llm,
                    analysis_llm=self.analysis_llm,
                    max_attempts=self.max_attempts,
                    max_steps=self.max_steps,
                    llm_args=self.llm_args,
                    output_dir=domain_output,
                    analysis_concurrency=self.analysis_concurrency,
                    enable_cot=self.enable_cot,
                    quality_threshold=self.quality_threshold,
                    enable_contrast_feedback=self.enable_contrast_feedback,
                )
            else:
                runner = TaskRunner(
                    domain=domain,
                    agent_llm=self.agent_llm,
                    user_llm=self.user_llm,
                    analysis_llm=self.analysis_llm,
                    max_attempts=self.max_attempts,
                    max_steps=self.max_steps,
                    llm_args=self.llm_args,
                    output_dir=domain_output,
                    analysis_concurrency=self.analysis_concurrency,
                    enable_cot=self.enable_cot,
                    quality_threshold=self.quality_threshold,
                    enable_contrast_feedback=self.enable_contrast_feedback,
                )
            
            try:
                result = runner.run(task, difficulty, task_idx)
                completed.add((domain, task.id, trial_idx))
                self.save_checkpoint(completed)

                # 成功执行，重置连续失败计数
                with consecutive_failures["lock"]:
                    consecutive_failures["count"] = 0

                if domain in self.samplers:
                    self.samplers[domain].update_failure_history(
                        task_id=task.id,
                        success=result.success,
                        fail_tools=difficulty.weak_tools if not result.success else None,
                    )

                return (domain, trial_idx, difficulty, result)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"任务 {domain}/{task.id} trial {trial_idx} 执行失败: {error_msg}")

                # 检测是否是连接错误（远程服务宕机）
                is_connection_error = any(keyword in error_msg.lower() for keyword in [
                    "connection", "timeout", "refused", "unavailable",
                    "503", "502", "500", "network", "socket", "eof"
                ])

                if is_connection_error:
                    with consecutive_failures["lock"]:
                        consecutive_failures["count"] += 1
                        current_count = consecutive_failures["count"]

                    logger.warning(f"连续连接失败次数: {current_count}/{MAX_CONSECUTIVE_FAILURES}")

                    if current_count >= MAX_CONSECUTIVE_FAILURES:
                        raise ServiceDownError(
                            f"检测到远程服务宕机！连续 {current_count} 个任务因连接错误失败。"
                            f"最后错误: {error_msg}"
                        )

                return (domain, trial_idx, difficulty, TaskResult(
                    task_id=task.id,
                    success=False,
                    num_attempts=0,
                    final_reward=0.0,
                    error=str(e),
                ))

        results: Dict[str, List[TaskResult]] = {d: [] for d in self.domains}
        
        # 使用动态worker池模式：谁完成了就执行下一个任务
        from concurrent.futures import as_completed
        
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # 初始提交max_concurrency个任务
            pending_tasks = list(all_tasks)
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
                    try:
                        domain, trial_idx, difficulty, result = future.result()
                    except ServiceDownError as e:
                        # 远程服务宕机，终止所有任务
                        logger.critical(f"远程服务宕机，终止执行: {e}")
                        # 取消所有pending的future
                        for f in active_futures:
                            f.cancel()
                        raise SystemExit(f"远程服务宕机: {e}")

                    results[domain].append(result)
                    completed_count += 1

                    status = "✓" if result.success else "✗"
                    logger.info(
                        f"进度: {completed_count}/{len(all_tasks)} {status} "
                        f"{domain}/{result.task_id} (trial {trial_idx}, weight={difficulty.priority_weight:.1f})"
                    )

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

    def save_statistics(self, results: Dict[str, List[TaskResult]], output_path: Path):
        """保存统计信息并合并数据"""
        all_stats = {}
        total_tasks = 0
        total_success = 0
        
        for domain, domain_results in results.items():
            stats = self._compute_domain_stats(domain, domain_results)
            all_stats[domain] = stats
            total_tasks += stats["total_tasks"]
            total_success += stats["successful_tasks"]
            
            domain_dir = output_path / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            with open(domain_dir / "statistics.json", "w") as f:
                json.dump(stats, f, indent=2)
        
        summary = {
            "total_tasks": total_tasks,
            "total_success": total_success,
            "overall_success_rate": total_success / total_tasks if total_tasks > 0 else 0,
            "domains": all_stats,
            "config": {
                "enable_cot": self.enable_cot,
                "quality_threshold": self.quality_threshold,
                "prioritize_weak": self.prioritize_weak,
                "enable_contrast_feedback": self.enable_contrast_feedback,
                "enable_error_injection": self.enable_error_injection,
                "injection_mode": self.injection_mode if self.enable_error_injection else None,
                "error_injection_base_rate": self.error_injection_base_rate if self.enable_error_injection else None,
                "error_injection_max_errors": self.error_injection_max_errors if self.enable_error_injection else None,
                "use_llm_for_recovery": self.use_llm_for_recovery if self.enable_error_injection and self.injection_mode == "agent" else None,
            },
        }
        
        with open(output_path / "statistics.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # 合并所有领域的数据
        self._merge_all_data(output_path)
        
        logger.info("=" * 60)
        logger.info(f"统计信息已保存到: {output_path}")
        logger.info(f"总任务数: {total_tasks}")
        if total_tasks > 0:
            logger.info(f"总成功数: {total_success} ({total_success/total_tasks:.2%})")
        for domain, stats in all_stats.items():
            logger.info(f"  {domain}: {stats['successful_tasks']}/{stats['total_tasks']} ({stats['success_rate']:.2%})")
        logger.info("=" * 60)

    def _merge_all_data(self, output_path: Path):
        """合并所有领域的 SFT 数据"""
        # 合并 SFT 数据
        sft_all_path = output_path / "sft_data_all.jsonl"
        sft_count = 0
        
        with open(sft_all_path, "w") as out_f:
            for domain in self.domains:
                domain_sft = output_path / domain / "sft_data.jsonl"
                if domain_sft.exists():
                    with open(domain_sft, "r") as in_f:
                        for line in in_f:
                            out_f.write(line)
                            sft_count += 1
        
        logger.info(f"SFT 数据已合并: {sft_all_path} ({sft_count} 条)")

    def _compute_domain_stats(self, domain: str, results: List[TaskResult]) -> dict:
        """计算单个领域的统计"""
        total = len(results)
        success = sum(1 for r in results if r.success)
        
        sampler_stats = {}
        if domain in self.samplers:
            sampler_stats = self.samplers[domain].get_statistics()
        
        return {
            "total_tasks": total,
            "successful_tasks": success,
            "failed_tasks": total - success,
            "success_rate": success / total if total > 0 else 0,
            "total_attempts": sum(r.num_attempts for r in results),
            "avg_attempts": sum(r.num_attempts for r in results) / total if total > 0 else 0,
            "sampler_stats": sampler_stats,
            "task_results": [asdict(r) for r in results],
        }
