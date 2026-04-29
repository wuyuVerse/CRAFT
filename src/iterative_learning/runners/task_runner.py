"""
单任务执行器

负责执行单个任务的迭代学习流程。
支持 CoT 推理、数据质量控制、对比分析注入。
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

from ..agents.cot_agent import AnalysisCoTAgent
from ..analysis import FailureAnalyzer, TrajectoryAnalyzer
from ..analysis.task_classifier import TaskDifficultyClassifier, TaskDifficulty
from ..data import AttemptRecord
from ..data.models import TaskResult
from ..data.enhanced_formatter import EnhancedDataFormatter


class TaskRunner:
    """单任务执行器"""

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
        analysis_concurrency: int = 3,
        enable_cot: bool = True,
        quality_threshold: float = 0.6,
        enable_contrast_feedback: bool = True,
    ):
        """
        Args:
            domain: 领域名称
            agent_llm: Agent 使用的 LLM
            user_llm: User Simulator 使用的 LLM
            analysis_llm: 分析使用的 LLM
            max_attempts: 最大尝试次数
            max_steps: 每次尝试的最大步数
            llm_args: LLM 额外参数
            output_dir: 输出目录
            analysis_concurrency: 分析任务并发数
            enable_cot: 是否启用思维链推理
            quality_threshold: 数据质量阈值
            enable_contrast_feedback: 是否启用对比反馈（成功后用于后续同任务的 retry）
        """
        self.domain = domain
        self.agent_llm = agent_llm
        self.user_llm = user_llm
        self.analysis_llm = analysis_llm
        self.max_attempts = max_attempts
        self.max_steps = max_steps
        self.llm_args = llm_args or {}
        self.output_dir = output_dir
        self.analysis_concurrency = analysis_concurrency
        self.enable_cot = enable_cot
        self.quality_threshold = quality_threshold
        self.enable_contrast_feedback = enable_contrast_feedback

        self.failure_analyzer = FailureAnalyzer(analysis_llm)
        self.trajectory_analyzer = TrajectoryAnalyzer(analysis_llm)
        self.task_classifier = TaskDifficultyClassifier()
        
        # 缓存成功轨迹，用于对比分析
        self._success_cache: dict[str, SimulationRun] = {}

    def run(self, task: Task, difficulty: Optional[TaskDifficulty] = None, task_idx: Optional[int] = None) -> TaskResult:
        """
        执行单个任务的迭代学习流程。
        
        Args:
            task: 要执行的任务
            difficulty: 预计算的任务难度（可选）
            
        Returns:
            任务执行结果
        """
        # 分类任务难度
        if difficulty is None:
            difficulty = self.task_classifier.classify(task, self.domain)
        
        # 根据难度调整参数
        effective_max_steps = self._adjust_max_steps(difficulty)
        
        # 创建增强格式化器
        formatter = EnhancedDataFormatter(
            domain=self.domain,
            expected_actions=difficulty.expected_actions,
            quality_threshold=self.quality_threshold,
        )
        
        attempts: List[AttemptRecord] = []
        success = False
        analysis = ""
        max_retries = 3
        system_prompt = None
        tools = None
        
        # 检查是否有该任务的成功轨迹缓存（用于对比分析）
        cached_success = self._success_cache.get(task.id)

        for attempt_num in range(1, self.max_attempts + 1):
            simulation = None
            
            for retry in range(max_retries):
                try:
                    env = self._build_environment()
                    tools = [tool.openai_schema for tool in env.get_tools()]
                    
                    agent = self._build_agent(env, analysis, difficulty)
                    system_prompt = agent.original_system_prompt
                    
                    user = self._build_user(env, task)

                    orchestrator = Orchestrator(
                        domain=self.domain,
                        agent=agent,
                        user=user,
                        environment=env,
                        task=task,
                        max_steps=effective_max_steps,
                    )

                    logger.info(
                        f"=========== Running task {task.id} attempt {attempt_num} "
                        f"(retry {retry}, weight={difficulty.priority_weight:.1f}) ==========="
                    )
                    simulation = orchestrator.run()
                    break
                    
                except Exception as e:
                    logger.warning(f"Task {task.id} attempt {attempt_num} retry {retry} 失败: {e}")
                    if retry == max_retries - 1:
                        logger.error(f"Task {task.id} attempt {attempt_num} 所有重试都失败")
                        raise
                    continue
            
            if simulation is None:
                continue

            reward, reward_info = self._evaluate(task, simulation)
            termination = getattr(
                simulation.termination_reason,
                "value",
                str(simulation.termination_reason)
            )

            logger.info(f"Task {task.id} attempt {attempt_num} 完成 - Reward: {reward}, Termination: {termination}")

            attempt_record = AttemptRecord(
                attempt=attempt_num,
                reward=reward,
                termination=termination,
                analysis_used=analysis,
                simulation=simulation,
            )

            if reward != 0:
                success = True
                attempts.append(attempt_record)
                logger.info(f"✓ Task {task.id} 成功! 尝试次数: {attempt_num}")
                
                # 缓存成功轨迹
                self._success_cache[task.id] = simulation
                break
            else:
                logger.warning(f"✗ Task {task.id} attempt {attempt_num} 失败，生成分析...")
                
                # 生成分析（如果有成功轨迹缓存，使用对比分析）
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

        if attempts and system_prompt and tools:
            self._generate_analysis_and_sft(
                attempts, success, system_prompt, tools, formatter, difficulty
            )

        return TaskResult(
            task_id=task.id,
            success=success,
            num_attempts=len(attempts),
            final_reward=attempts[-1].reward if attempts else 0.0,
        )

    def _adjust_max_steps(self, difficulty: TaskDifficulty) -> int:
        """根据任务难度调整最大步数"""
        base_steps = self.max_steps
        if difficulty.is_multi_step:
            base_steps = max(base_steps, 150)
        if difficulty.is_context_complex:
            base_steps = int(base_steps * 1.2)
        return base_steps

    def _build_agent(
        self,
        env: Environment,
        analysis: str,
        difficulty: TaskDifficulty,
    ) -> AnalysisCoTAgent:
        """构建 Agent"""
        use_cot = self.enable_cot and (
            difficulty.is_parameter_sensitive or
            difficulty.is_multi_step or
            len(difficulty.weak_tools) > 0
        )
        
        return AnalysisCoTAgent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm=self.agent_llm,
            llm_args=self.llm_args,
            analysis=analysis,
            enable_cot=use_cot,
            weak_tools=difficulty.weak_tools if use_cot else None,
        )

    def _build_environment(self) -> Environment:
        """构建环境"""
        return registry.get_env_constructor(self.domain)(solo_mode=False)

    def _build_user(self, env: Environment, task: Task) -> UserSimulator:
        """构建用户模拟器"""
        try:
            user_tools = env.get_user_tools()
        except Exception:
            user_tools = None

        return UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=self.user_llm,
        )

    def _evaluate(self, task: Task, simulation):
        """评估模拟结果"""
        reward_info = evaluate_simulation(
            domain=self.domain,
            task=task,
            simulation=simulation,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )
        simulation.reward_info = reward_info
        reward = getattr(reward_info, "reward", 0.0) or 0.0
        return reward, reward_info

    def _generate_analysis_and_sft(
        self,
        attempts: List[AttemptRecord],
        success: bool,
        system_prompt: str,
        tools: list,
        formatter: EnhancedDataFormatter,
        difficulty: TaskDifficulty,
    ):
        """生成分析和 SFT 数据"""
        if not attempts:
            return

        jobs = []
        task_id = attempts[0].simulation.task_id

        if success:
            success_attempt = attempts[-1]
            
            # 使用增强格式化器（带质量检查）
            if self.output_dir:
                logger.info(f"正在格式化成功轨迹 - Task: {task_id}")
                
                formatted, score, saved = formatter.format_with_quality_check(
                    messages=success_attempt.simulation.messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    task_evaluation=success_attempt.simulation.reward_info,
                    output_path=self.output_dir,
                )
                
                logger.info(
                    f"数据质量评分 - Task: {task_id}, "
                    f"完整性: {score.completeness:.2f}, "
                    f"参数准确性: {score.parameter_accuracy:.2f}, "
                    f"综合: {score.overall:.2f}, "
                    f"已保存: {saved}"
                )

            # 添加正样本分析任务
            jobs.append(
                lambda: self.trajectory_analyzer.analyze_success(
                    success_attempt.simulation,
                    system_prompt,
                    tools,
                    self.output_dir,
                )
            )

            # 为失败的尝试添加分析任务（错误分析 + 对比分析）
            for att in attempts[:-1]:
                jobs.append(
                    lambda att=att: self.trajectory_analyzer.analyze_error(
                        att.simulation,
                        system_prompt,
                        tools,
                        self.output_dir,
                    )
                )
                jobs.append(
                    lambda att=att: self.trajectory_analyzer.analyze_contrast(
                        att.simulation,
                        system_prompt,
                        success_attempt.simulation,
                        tools,
                        self.output_dir,
                    )
                )
        else:
            # 没有成功轨迹，仅生成错误分析
            for att in attempts:
                jobs.append(
                    lambda att=att: self.trajectory_analyzer.analyze_error(
                        att.simulation,
                        system_prompt,
                        tools,
                        self.output_dir,
                    )
                )

        # 并行执行分析任务
        if jobs:
            logger.info(f"开始并行生成分析 - Task: {task_id}, 分析任务数: {len(jobs)}")
            
            with ThreadPoolExecutor(max_workers=self.analysis_concurrency) as executor:
                list(executor.map(lambda fn: fn(), jobs))
            
            logger.info(f"所有分析生成完成 - Task: {task_id}")
