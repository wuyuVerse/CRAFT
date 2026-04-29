"""
增强的任务执行器

集成错误注入、轨迹清理、分析数据分离等功能。
"""

import random
from pathlib import Path
from typing import List, Optional

from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task

from ..analysis.task_classifier import TaskDifficulty
from ..data.enhanced_formatter import DataQualityScore
from ..data.trajectory_extractor import CleanTrajectoryExtractor
from ..injection import ErrorInjectionConfig, ErrorInjectionOrchestrator
from ..injection.injectors import RuleBasedErrorInjector, AgentBasedErrorInjector
from .task_runner import TaskRunner


class EnhancedTaskRunner(TaskRunner):
    """增强的任务执行器 - V3/V4 版本"""

    def __init__(
        self,
        error_injection_config: ErrorInjectionConfig,
        injection_mode: str = "rule",  # "rule" (V3) 或 "agent" (V4)
        use_llm_for_recovery: bool = True,
        error_type_weights: dict = None,
        api_base: str = None,  # V4: API地址
        include_failed_in_sft: bool = False,  # 是否将失败轨迹加入SFT数据
        **kwargs
    ):
        """
        Args:
            error_injection_config: 错误注入配置
            injection_mode: 注入模式 - "rule" 使用V3规则注入, "agent" 使用V4 Agent注入
            use_llm_for_recovery: V4模式下是否使用LLM生成恢复
            error_type_weights: V4模式下的错误类型权重
            api_base: V4模式下的API地址
            include_failed_in_sft: 是否将失败轨迹也保存到SFT数据
            **kwargs: 传递给父类 TaskRunner 的参数
        """
        super().__init__(**kwargs)

        self.error_injection_config = error_injection_config
        self.injection_mode = injection_mode
        self.api_base = api_base
        self.include_failed_in_sft = include_failed_in_sft
        self.trajectory_extractor = CleanTrajectoryExtractor(self.domain)
        
        # 根据模式初始化注入器
        if injection_mode == "agent":
            # V4: 基于Agent的错误注入
            self.error_injector = AgentBasedErrorInjector(
                llm=kwargs.get('agent_llm', 'openai/deepseek-v3'),
                error_db_path=error_injection_config.error_db_path,
                domain=self.domain,
                api_base=api_base,
                use_llm_for_recovery=use_llm_for_recovery,
                error_type_weights=error_type_weights,
            )
            logger.info(f"Initialized V4 AgentBasedErrorInjector for {self.domain}")
        else:
            # V3: 基于规则的轨迹错误注入
            self.error_injector = RuleBasedErrorInjector(
                error_injection_config.error_db_path,
                self.domain
            )
            logger.info(f"Initialized V3 RuleBasedErrorInjector for {self.domain}")
        
        # 保留旧的引用以兼容
        self.trajectory_error_injector = self.error_injector
        
        # 分析数据单独存储
        self.analysis_dir = Path(self.output_dir) / "analysis"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)
        
        # 失败轨迹单独存储（用于RL等）
        self.failed_trajectories_dir = Path(self.output_dir) / "failed_trajectories"
        self.failed_trajectories_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(
            f"Initialized EnhancedTaskRunner ({injection_mode.upper()}) for {self.domain}, "
            f"error_injection_enabled={error_injection_config.enabled}, "
            f"include_failed_in_sft={include_failed_in_sft}"
        )
    
    def _build_orchestrator(
        self,
        env,
        agent,
        user,
        task: Task,
        max_steps: int
    ) -> ErrorInjectionOrchestrator:
        """
        构建支持错误注入的编排器
        
        Returns:
            ErrorInjectionOrchestrator 实例
        """
        return ErrorInjectionOrchestrator(
            domain=self.domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=max_steps,
            error_injection_config=self.error_injection_config,
        )
    
    def run(self, task: Task, difficulty: Optional[TaskDifficulty] = None, task_idx: Optional[int] = None):
        """
        执行任务并生成数据
        
        V3 流程：
        1. 正常执行任务直到成功（不注入错误）
        2. 如果成功且启用错误注入，基于成功轨迹生成带错误恢复的新轨迹
        3. 提取干净轨迹保存为 SFT 数据
        """
        # 分类任务难度
        if difficulty is None:
            difficulty = self.task_classifier.classify(task, self.domain)
        
        attempts = []
        success = False
        analysis = ""
        
        # 检查缓存的成功轨迹（用于对比分析）
        cached_success = self._success_cache.get(task.id)
        
        # 第一阶段：正常执行任务直到成功（不注入错误）
        for attempt_num in range(1, self.max_attempts + 1):
            simulation = None
            
            for retry in range(3):  # 最多重试3次
                try:
                    # 构建环境和 Agent
                    env = self._build_environment()
                    tools = [tool.openai_schema for tool in env.get_tools()]
                    agent = self._build_agent(env, analysis, difficulty)
                    system_prompt = agent.original_system_prompt
                    user = self._build_user(env, task)
                    
                    # 使用普通编排器（不注入错误）
                    from tau2.orchestrator.orchestrator import Orchestrator
                    orchestrator = Orchestrator(
                        domain=self.domain,
                        agent=agent,
                        user=user,
                        environment=env,
                        task=task,
                        max_steps=self._adjust_max_steps(difficulty)
                    )
                    
                    logger.info(
                        f"=========== Running task {task.id} attempt {attempt_num} "
                        f"(retry {retry}, weight={difficulty.priority_weight:.1f}) ==========="
                    )
                    
                    # 执行
                    simulation = orchestrator.run()
                    break
                    
                except Exception as e:
                    logger.warning(f"Task {task.id} attempt {attempt_num} retry {retry} 失败: {e}")
                    if retry == 2:
                        logger.error(f"Task {task.id} attempt {attempt_num} 所有重试都失败")
                        raise
                    continue
            
            if simulation is None:
                continue
            
            # 评估
            reward, reward_info = self._evaluate(task, simulation)
            termination = getattr(
                simulation.termination_reason,
                "value",
                str(simulation.termination_reason)
            )
            
            logger.info(
                f"Task {task.id} attempt {attempt_num} 完成 - "
                f"Reward: {reward}, Termination: {termination}"
            )
            
            # 记录尝试
            from ..data.models import AttemptRecord
            attempt_record = AttemptRecord(
                attempt=attempt_num,
                reward=reward,
                termination=termination,
                analysis_used=analysis,
                simulation=simulation,
            )
            
            if reward != 0:
                # 成功！
                success = True
                attempts.append(attempt_record)
                self._success_cache[task.id] = simulation
                logger.info(f"✓ Task {task.id} 成功! 尝试次数: {attempt_num}")
                break
            else:
                # 失败，生成分析
                logger.warning(f"✗ Task {task.id} attempt {attempt_num} 失败，生成分析...")
                
                # 保存失败轨迹（用于RL等）
                self._save_failed_trajectory(
                    simulation,
                    system_prompt,
                    tools,
                    task_idx if task_idx is not None else task.id,
                    attempt_num
                )
                
                if self.enable_contrast_feedback and cached_success:
                    logger.info(f"使用对比分析（与缓存的成功轨迹对比）")
                    analysis = self.failure_analyzer.build_contrast_analysis(
                        attempt_record,
                        cached_success,
                        tools,
                        system_prompt,
                    )
                else:
                    analysis = self.failure_analyzer.build_analysis(
                        attempt_record,
                        tools,
                        system_prompt,
                    )
                
                attempts.append(attempt_record)
        
        # 处理结果
        if success and attempts:
            success_simulation = attempts[-1].simulation
            
            # 1. 保存原始成功轨迹（根据 correct_trajectory_weight 重复保存）
            weight = getattr(self.error_injection_config, 'correct_trajectory_weight', 1)
            for i in range(weight):
                self._process_success_trajectory(
                    attempts[-1], 
                    system_prompt, 
                    tools,
                    difficulty,
                    is_error_recovery=False
                )
                if i == 0:
                    logger.info(f"✓ 保存正确轨迹: {task.id} (weight={weight})")
            
            # 2. 根据 base_rate 决定是否生成错误注入版本
            if self.error_injection_config.enabled:
                import random
                if random.random() < self.error_injection_config.base_rate:
                    logger.info(f"生成错误注入版本: {task.id}")
                    error_recovery_data = self._generate_error_recovery_trajectory(
                        task, success_simulation, system_prompt, tools, difficulty
                    )
                    if error_recovery_data:
                        # 直接保存错误恢复轨迹（已经是dict格式）
                        self._save_error_recovery_trajectory(
                            error_recovery_data,
                            task.id,
                            difficulty
                        )
                        logger.info(f"✓ 保存错误恢复轨迹: {task.id}")
                else:
                    logger.info(f"跳过错误注入 (概率 {self.error_injection_config.base_rate}): {task.id}")
        
        # 保存分析数据（根据配置决定是否保存到SFT）
        save_to_sft = getattr(self.error_injection_config, 'save_analysis_data', False)
        # 使用task_idx（如果提供）或task.id作为文件名
        file_task_id = str(task_idx) if task_idx is not None else task.id
        self._save_analysis_data(attempts, success, system_prompt, tools, save_to_sft, file_task_id)
        
        # 返回结果
        from ..data.models import TaskResult
        return TaskResult(
            task_id=task.id,
            success=success,
            num_attempts=len(attempts),
            final_reward=attempts[-1].reward if attempts else 0.0,
        )
    
    def _process_success_trajectory(
        self,
        success_attempt,
        system_prompt: str,
        tools: List[dict],
        difficulty: TaskDifficulty,
        is_error_recovery: bool = False,
    ):
        """
        处理成功轨迹
        
        Args:
            success_attempt: 成功的尝试记录
            system_prompt: 系统提示
            tools: 工具列表
            difficulty: 任务难度
            is_error_recovery: 是否是错误恢复轨迹
        """
        simulation = success_attempt.simulation
        
        # 1. 提取干净轨迹（移除分析内容）
        clean_data = self.trajectory_extractor.extract_clean_trajectory(
            simulation,
            system_prompt,
            tools,
        )
        
        if clean_data is None:
            logger.warning(f"轨迹提取失败: {simulation.task_id}")
            return
        
        # 2. 质量评分
        score = self._compute_quality_score(simulation, difficulty)
        
        if score.overall < self.quality_threshold:
            logger.warning(
                f"数据质量不足: {simulation.task_id}, "
                f"score={score.overall:.2f}"
            )
            return
        
        # 3. 保存 SFT 数据
        sft_file = Path(self.output_dir) / "sft_data.jsonl"
        with open(sft_file, "a", encoding='utf-8') as f:
            import json
            f.write(json.dumps(clean_data, ensure_ascii=False) + "\n")
        
        error_count = self._count_error_recoveries(clean_data['messages'])
        trajectory_type = "错误恢复" if is_error_recovery else "原始成功"
        
        logger.info(
            f"✓ 保存{trajectory_type}轨迹: {simulation.task_id}, "
            f"quality={score.overall:.2f}, "
            f"steps={len(simulation.messages)}, "
            f"errors_recovered={error_count}"
        )
    
    def _save_error_recovery_trajectory(
        self,
        trajectory_data: dict,
        task_id: str,
        difficulty: TaskDifficulty,
    ):
        """
        保存错误恢复轨迹（直接从dict保存）
        
        Args:
            trajectory_data: 轨迹数据 {"messages": [...], "tools": "...", "injected_errors": N}
            task_id: 任务ID
            difficulty: 任务难度
        """
        import json
        
        # 保存 SFT 数据
        sft_file = Path(self.output_dir) / "sft_data.jsonl"
        
        # 移除injected_errors字段（不需要保存到SFT数据中）
        save_data = {
            "messages": trajectory_data["messages"],
            "tools": trajectory_data["tools"]
        }
        
        with open(sft_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(save_data, ensure_ascii=False) + "\n")
        
        error_count = trajectory_data.get('injected_errors', 0)
        
        logger.info(
            f"✓ 保存错误恢复轨迹: {task_id}, "
            f"steps={len(trajectory_data['messages'])}, "
            f"injected_errors={error_count}"
        )
    
    def _save_failed_trajectory(
        self,
        simulation: SimulationRun,
        system_prompt: str,
        tools: List[dict],
        task_id: str,
        attempt_num: int,
    ):
        """
        保存失败轨迹（用于RL等）

        Args:
            simulation: 失败的模拟轨迹
            system_prompt: 系统提示
            tools: 工具列表
            task_id: 任务ID
            attempt_num: 尝试次数
        """
        import json

        try:
            # 提取干净轨迹
            clean_data = self.trajectory_extractor.extract_clean_trajectory(
                simulation,
                system_prompt,
                tools,
            )

            if clean_data is None:
                logger.warning(f"无法提取失败轨迹: {task_id}")
                return

            # 添加元数据
            clean_data['metadata'] = {
                'task_id': str(task_id),
                'attempt': attempt_num,
                'reward': 0,
                'termination': getattr(
                    simulation.termination_reason,
                    "value",
                    str(simulation.termination_reason)
                ),
            }

            # 保存到失败轨迹目录
            failed_file = self.failed_trajectories_dir / f"task_{task_id}_attempt{attempt_num}.json"
            with open(failed_file, "w", encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)

            # 同时追加到汇总文件（用于批量处理）
            failed_jsonl = Path(self.output_dir) / "failed_trajectories.jsonl"
            with open(failed_jsonl, "a", encoding='utf-8') as f:
                f.write(json.dumps(clean_data, ensure_ascii=False) + "\n")

            # 如果配置了 include_failed_in_sft，同时保存到SFT数据
            if self.include_failed_in_sft:
                sft_data = {
                    "messages": clean_data["messages"],
                    "tools": clean_data["tools"],
                }
                sft_file = Path(self.output_dir) / "sft_data.jsonl"
                with open(sft_file, "a", encoding='utf-8') as f:
                    f.write(json.dumps(sft_data, ensure_ascii=False) + "\n")
                logger.info(f"✓ 保存失败轨迹到SFT: {task_id} attempt {attempt_num}")
            else:
                logger.info(f"✓ 保存失败轨迹: {task_id} attempt {attempt_num}")

        except Exception as e:
            logger.error(f"保存失败轨迹失败: {task_id}, {e}")
    
    def _save_analysis_data(
        self,
        attempts: List,
        success: bool,
        system_prompt: str,
        tools: List[dict],
        save_to_sft: bool = False,  # 是否保存到SFT数据
        task_id: str = None,  # 任务ID（数字）
    ):
        """
        保存分析数据（并发执行）
        
        Args:
            attempts: 尝试记录列表
            success: 是否成功
            system_prompt: 系统提示词
            tools: 工具列表
            save_to_sft: 是否保存到SFT数据文件（默认False，只保存到analysis目录）
            task_id: 任务ID（数字，如"0", "1"等）
        """
        if not attempts:
            return
        
        import json
        from concurrent.futures import ThreadPoolExecutor
        
        # 使用传入的task_id（数字），如果没有则从simulation获取
        if task_id is None:
            task_id = attempts[0].simulation.task_id
        
        # 创建分析任务列表
        analysis_jobs = []
        
        # 为每个尝试生成分析
        for att in attempts:
            analysis_type = "success" if att.reward > 0 else "error"
            
            def generate_and_save_analysis(att=att, analysis_type=analysis_type):
                # 生成分析（如果save_to_sft=True，会自动保存到sft_data.jsonl）
                if analysis_type == "success":
                    analysis_content = self.trajectory_analyzer.analyze_success(
                        att.simulation,
                        system_prompt,
                        tools,
                        self.output_dir if save_to_sft else None,
                    )
                else:
                    analysis_content = self.trajectory_analyzer.analyze_error(
                        att.simulation,
                        system_prompt,
                        tools,
                        self.output_dir if save_to_sft else None,
                    )
                
                # 保存到分析目录（用于调试和查看）- 使用简短的task_id
                analysis_file = (
                    self.analysis_dir / 
                    f"task_{task_id}_attempt{att.attempt}_{analysis_type}.json"
                )
                
                with open(analysis_file, "w", encoding='utf-8') as f:
                    json.dump({
                        'task_id': task_id,
                        'attempt': att.attempt,
                        'reward': att.reward,
                        'analysis': analysis_content,
                    }, f, indent=2, ensure_ascii=False)
            
            analysis_jobs.append(generate_and_save_analysis)
        
        # 如果有成功和失败，生成对比分析
        if success and len(attempts) > 1:
            success_att = attempts[-1]
            for fail_att in attempts[:-1]:
                def generate_and_save_contrast(fail_att=fail_att, success_att=success_att):
                    contrast_analysis = self.trajectory_analyzer.analyze_contrast(
                        fail_att.simulation,
                        system_prompt,
                        success_att.simulation,
                        tools,
                        self.output_dir if save_to_sft else None,
                    )
                    
                    # 使用简短的task_id
                    contrast_file = (
                        self.analysis_dir /
                        f"task_{task_id}_attempt{fail_att.attempt}_contrast.json"
                    )
                    
                    with open(contrast_file, "w", encoding='utf-8') as f:
                        json.dump({
                            'task_id': task_id,
                            'failed_attempt': fail_att.attempt,
                            'success_attempt': success_att.attempt,
                            'contrast_analysis': contrast_analysis,
                        }, f, indent=2, ensure_ascii=False)
                
                analysis_jobs.append(generate_and_save_contrast)
        
        # 并发执行所有分析任务
        if analysis_jobs:
            logger.info(f"开始并发生成分析 - Task: {task_id}, 分析任务数: {len(analysis_jobs)}")
            
            with ThreadPoolExecutor(max_workers=self.analysis_concurrency) as executor:
                list(executor.map(lambda fn: fn(), analysis_jobs))
            
            logger.info(f"所有分析生成完成 - Task: {task_id}")
    
    def _count_error_recoveries(self, messages: List[dict]) -> int:
        """统计错误恢复次数"""
        count = 0
        for msg in messages:
            if msg['role'] == 'tool':
                content = msg.get('content', '')
                try:
                    if content.startswith('{'):
                        import json
                        tool_data = json.loads(content)
                        result = tool_data.get('result', '')
                        if isinstance(result, str) and result.startswith('Error:'):
                            count += 1
                    elif 'Error:' in content:
                        count += 1
                except:
                    pass
        return count
    
    def _compute_quality_score(
        self,
        simulation: SimulationRun,
        difficulty: TaskDifficulty,
    ) -> DataQualityScore:
        """计算数据质量评分"""
        from tau2.data_model.message import AssistantMessage, ToolMessage
        
        messages = simulation.messages
        
        # 1. 完整性：是否包含预期动作
        called_tools = []
        for msg in messages:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                called_tools.extend([tc.name for tc in msg.tool_calls])
        
        if difficulty.expected_actions:
            matched = sum(1 for t in difficulty.expected_actions if t in called_tools)
            completeness = matched / len(difficulty.expected_actions)
        else:
            completeness = 1.0 if called_tools else 0.5
        
        # 2. 参数准确性：基于 reward_info
        parameter_accuracy = 1.0
        if hasattr(simulation, 'reward_info') and simulation.reward_info:
            reward_info = simulation.reward_info
            if hasattr(reward_info, 'action_reward_info'):
                action_info = reward_info.action_reward_info
                if action_info and hasattr(action_info, 'action_checks'):
                    checks = action_info.action_checks or []
                    if checks:
                        correct = sum(1 for c in checks if getattr(c, 'score', 0) > 0)
                        parameter_accuracy = correct / len(checks)
        
        # 3. 错误恢复质量
        error_count = 0
        recovery_count = 0
        
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                try:
                    import json
                    result = json.loads(msg.content).get('result', '')
                    if isinstance(result, str) and result.startswith('Error:'):
                        error_count += 1
                        # 检查后续是否有恢复
                        if i + 1 < len(messages):
                            next_msg = messages[i + 1]
                            if isinstance(next_msg, AssistantMessage):
                                recovery_count += 1
                except:
                    pass
        
        error_recovery_quality = (
            recovery_count / error_count if error_count > 0 else 1.0
        )
        
        # 综合评分
        overall = (
            completeness * 0.35 +
            parameter_accuracy * 0.35 +
            error_recovery_quality * 0.30
        )
        
        return DataQualityScore(
            completeness=completeness,
            parameter_accuracy=parameter_accuracy,
            reasoning_quality=error_recovery_quality,  # 复用字段
            overall=overall,
        )

    def _generate_error_recovery_trajectory(
        self,
        task: Task,
        success_simulation: SimulationRun,
        system_prompt: str,
        tools: List[dict],
        difficulty: TaskDifficulty,
    ) -> Optional[dict]:
        """
        基于成功轨迹生成带错误恢复的新轨迹
        
        支持两种模式：
        - V3 (rule): 基于规则的错误注入
        - V4 (agent): 基于Agent的错误生成和恢复
        
        Args:
            task: 任务
            success_simulation: 成功的模拟轨迹
            system_prompt: 系统提示
            tools: 工具列表
            difficulty: 任务难度
            
        Returns:
            带错误恢复的轨迹数据（dict格式），如果生成失败则返回 None
        """
        try:
            # 1. 先提取干净的成功轨迹
            clean_data = self.trajectory_extractor.extract_clean_trajectory(
                success_simulation,
                system_prompt,
                tools,
            )
            
            if clean_data is None:
                logger.warning(f"无法提取干净轨迹: {task.id}")
                return None
            
            # 2. 计算要注入的错误数量
            max_errors = self.error_injection_config.max_errors_per_task
            num_errors = random.randint(1, max_errors)
            
            # 3. 使用错误注入器（V3或V4）
            logger.info(f"在成功轨迹上注入错误 ({self.injection_mode}): {task.id}")
            
            injected_trajectory = self.error_injector.inject_errors(
                clean_data,
                num_errors=num_errors,
                error_rate=1.0  # 对选中的点100%注入
            )
            
            if injected_trajectory is None:
                logger.warning(f"无法注入错误: {task.id}")
                return None
            
            error_count = injected_trajectory.get('injected_errors', 0)
            error_types = injected_trajectory.get('error_types', [])
            
            logger.info(
                f"✓ 错误注入成功: {task.id}, "
                f"注入错误数: {error_count}, "
                f"错误类型: {error_types}"
            )
            
            return injected_trajectory
                
        except Exception as e:
            logger.error(f"生成错误注入轨迹失败: {task.id}, {e}")
            import traceback
            traceback.print_exc()
            return None
