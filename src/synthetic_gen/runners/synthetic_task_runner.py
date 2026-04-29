"""
SyntheticTaskRunner

主编排器：协调多个LLM Agent生成tau2格式任务。
"""

import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.iterative_learning.utils.llm_client import LLMClient, create_llm_client
from src.synthetic_gen.core.tau2.extractors.task_extractor import Tau2TaskExtractor, Tau2TaskPattern
from src.synthetic_gen.core.tau2.extractors.parameter_extractor import Tau2ParameterAnalyzer
from src.synthetic_gen.core.tau2.extractors.parameter_enricher import ParameterEnricher, EnrichedParams
from src.synthetic_gen.core.tau2.generators.task_designer import TaskDesigner, TaskDesign
from src.synthetic_gen.core.tau2.generators.scenario_writer import ScenarioWriter, UserScenario
from src.synthetic_gen.core.tau2.generators.criteria_writer import CriteriaWriter, EvaluationCriteria
from src.synthetic_gen.core.tau2.validators.task_validator import TaskValidator, ValidationResult


def _expand_env(value):
    """Expand ${VAR} placeholders in YAML values."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
            lambda m: os.environ.get(m.group(1), m.group(2) or ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class GenerationConfig:
    """生成配置"""
    # LLM配置
    model: str = "deepseek-v3"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"

    # Agent温度配置
    task_designer_temperature: float = 0.7
    scenario_writer_temperature: float = 0.8
    criteria_writer_temperature: float = 0.6

    # 生成配置
    num_tasks: int = 100
    max_concurrent: int = 10
    seed: Optional[int] = None

    # 复杂度权重
    complexity_weights: Dict[str, float] = None

    # 质量控制
    min_reason_length: int = 30
    min_task_instructions_length: int = 50
    require_all_actions: bool = True

    def __post_init__(self):
        if self.complexity_weights is None:
            self.complexity_weights = {
                "simple": 0.3,
                "medium": 0.5,
                "complex": 0.2,
            }


class SyntheticTaskRunner:
    """
    合成任务生成器

    编排流程：
    1. 提取tau2模式和参数
    2. 采样模式和参数
    3. Agent1 (TaskDesigner): 设计任务
    4. Agent2 (ScenarioWriter): 编写场景
    5. Agent3 (CriteriaWriter): 编写标准
    6. 验证和保存
    """

    def __init__(self, config: GenerationConfig):
        """
        初始化Runner

        Args:
            config: 生成配置
        """
        self.config = config

        # 设置随机种子
        if config.seed is not None:
            random.seed(config.seed)

        # 创建LLM客户端
        self.llm_client = create_llm_client(
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            timeout=120,  # 增加超时时间到120秒
        )

        # 创建Agents
        self.task_designer = TaskDesigner(
            self.llm_client,
            temperature=config.task_designer_temperature,
        )
        self.scenario_writer = ScenarioWriter(
            self.llm_client,
            temperature=config.scenario_writer_temperature,
        )
        self.criteria_writer = CriteriaWriter(
            self.llm_client,
            temperature=config.criteria_writer_temperature,
        )

        # 创建验证器
        self.validator = TaskValidator(
            min_reason_length=config.min_reason_length,
            min_task_instructions_length=config.min_task_instructions_length,
            require_all_actions=config.require_all_actions,
        )

        # 创建提取器
        self.extractor = Tau2TaskExtractor()
        self.parameter_analyzer = Tau2ParameterAnalyzer()

        # 创建参数丰富器（从db.json采样真实参数）
        self.parameter_enricher = ParameterEnricher()

    def run(
        self,
        domain: str,
        output_dir: str,
        num_tasks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        运行生成流程

        Args:
            domain: 领域名称
            output_dir: 输出目录
            num_tasks: 生成数量（覆盖配置）

        Returns:
            生成报告
        """
        num_tasks = num_tasks or self.config.num_tasks
        start_time = time.time()

        # 创建输出目录结构
        os.makedirs(output_dir, exist_ok=True)
        domain_dir = os.path.join(output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # 配置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"synthetic_gen_{domain}_{timestamp}.log")
        logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")
        logger.info(f"日志文件: {log_file}")

        logger.info(f"开始生成 {num_tasks} 个 {domain} 任务")

        # 1. 提取模式和参数
        logger.info("Step 1: 提取tau2模式和参数空间...")
        patterns = self.extractor.extract_patterns(domain)
        params_space = self.parameter_analyzer.analyze_parameter_space(domain)
        logger.info(f"  提取了 {len(patterns)} 个模式")

        # 2. 采样任务
        logger.info("Step 2: 采样任务配置...")
        task_configs = self._sample_task_configs(patterns, params_space, num_tasks, domain)
        logger.info(f"  采样了 {len(task_configs)} 个任务配置")

        # 3. 生成任务
        logger.info("Step 3: 使用LLM Agents生成任务...")
        tasks = self._generate_tasks_parallel(task_configs, domain, domain_dir)
        logger.info(f"  生成了 {len(tasks)} 个任务")

        # 4. 验证任务
        logger.info("Step 4: 验证任务质量...")
        validation_report = self.validator.validate_batch(tasks)
        logger.info(f"  验证通过率: {validation_report['pass_rate']:.1%}")
        logger.info(f"  平均质量分: {validation_report['average_score']:.2f}")

        # 5. 保存
        logger.info("Step 5: 保存任务...")
        self._save_tasks(tasks, domain, domain_dir, validation_report)

        elapsed_time = time.time() - start_time
        logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")

        return {
            "domain": domain,
            "num_generated": len(tasks),
            "num_valid": validation_report["valid_tasks"],
            "pass_rate": validation_report["pass_rate"],
            "average_score": validation_report["average_score"],
            "output_dir": domain_dir,
            "log_file": log_file,
            "elapsed_time": elapsed_time,
        }

    def _sample_task_configs(
        self,
        patterns: List[Tau2TaskPattern],
        params_space: Dict[str, Any],
        num_tasks: int,
        domain: str,
    ) -> List[Dict[str, Any]]:
        """采样任务配置"""
        configs = []

        # 按复杂度分组
        patterns_by_complexity = {
            "simple": [p for p in patterns if p.complexity == "simple"],
            "medium": [p for p in patterns if p.complexity == "medium"],
            "complex": [p for p in patterns if p.complexity == "complex"],
        }

        # 根据权重计算每种复杂度的数量
        weights = self.config.complexity_weights
        for i in range(num_tasks):
            # 选择复杂度
            complexity = random.choices(
                list(weights.keys()),
                weights=list(weights.values()),
                k=1
            )[0]

            # 从该复杂度的模式中随机选择
            available_patterns = patterns_by_complexity.get(complexity, [])
            if not available_patterns:
                available_patterns = patterns

            pattern = random.choice(available_patterns)

            # 采样参数
            params = self._sample_params(params_space, domain)

            configs.append({
                "task_id": f"synthetic_{domain}_{i}",
                "pattern": pattern,
                "params": params,
            })

        return configs

    def _sample_params(self, params_space: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        从数据库采样真实参数

        使用ParameterEnricher从tau2-bench的db.json中采样，
        确保参数与数据库一致，任务可以正确执行。
        """
        # 使用ParameterEnricher获取真实参数
        enriched = self.parameter_enricher.enrich_params(domain)
        params = enriched.to_dict()

        # 转换字段名以匹配现有接口
        if domain == "airline":
            # 确保有日期字段
            if "flight_date" in params:
                params["date"] = params.pop("flight_date")

        elif domain == "retail":
            # 添加user_name字段
            if "first_name" in params and "last_name" in params:
                params["user_name"] = f"{params['first_name']} {params['last_name']}"

        elif domain == "telecom":
            pass  # telecom参数已经是正确的格式

        return params

    def _generate_tasks_parallel(
        self,
        task_configs: List[Dict[str, Any]],
        domain: str,
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        """并行生成任务，带增量保存"""
        tasks = []
        failed_count = 0
        total = len(task_configs)

        # 增量保存文件
        incremental_file = os.path.join(output_dir, "synthetic_data.jsonl")

        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            futures = {
                executor.submit(self._generate_single_task, config, domain): config
                for config in task_configs
            }

            for i, future in enumerate(as_completed(futures)):
                try:
                    task = future.result()
                    if task:
                        tasks.append(task)
                        # 增量保存到jsonl文件
                        with open(incremental_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(task, ensure_ascii=False) + '\n')
                    else:
                        failed_count += 1

                    # 进度日志 - 每10个输出一次
                    progress = i + 1
                    if progress % 10 == 0 or progress == total:
                        success_rate = len(tasks) / progress * 100 if progress > 0 else 0
                        logger.info(f"  进度: {progress}/{total} ({progress/total*100:.1f}%) | 成功: {len(tasks)} | 失败: {failed_count} | 成功率: {success_rate:.1f}%")

                except Exception as e:
                    config = futures[future]
                    failed_count += 1
                    logger.error(f"生成任务 {config['task_id']} 失败: {e}")

        logger.info(f"  增量数据已保存到: {incremental_file}")
        return tasks

    def _generate_single_task(
        self,
        config: Dict[str, Any],
        domain: str,
    ) -> Optional[Dict[str, Any]]:
        """生成单个任务"""
        try:
            task_id = config["task_id"]
            pattern = config["pattern"]
            params = config["params"]

            # Agent1: 设计任务
            design = self.task_designer.design_task(pattern, params, task_id)
            if design is None:
                return None

            # Agent2: 编写场景
            scenario = self.scenario_writer.write_scenario(design)
            if scenario is None:
                return None

            # Agent3: 编写标准
            criteria = self.criteria_writer.write_criteria(design, scenario, pattern.expected_actions)
            if criteria is None:
                return None

            # 组装tau2格式任务
            task = self._assemble_task(task_id, design, scenario, criteria, pattern)

            return task

        except Exception as e:
            logger.error(f"生成任务失败: {e}")
            return None

    def _assemble_task(
        self,
        task_id: str,
        design: TaskDesign,
        scenario: UserScenario,
        criteria: EvaluationCriteria,
        pattern: Tau2TaskPattern,
    ) -> Dict[str, Any]:
        """组装tau2格式任务（完全符合tau2-bench格式）"""
        # 处理actions - 移除compare_args，添加info字段
        actions = []
        for action in criteria.actions:
            clean_action = {
                "action_id": action.get("action_id", ""),
                "name": action.get("name", ""),
                "arguments": action.get("arguments", {}),
                "info": None,  # tau2-bench格式需要这个字段
            }
            actions.append(clean_action)

        return {
            "id": task_id,
            "description": {
                "purpose": f"Generated from tau2 task {pattern.task_id}",
                "relevant_policies": pattern.description.get("relevant_policies"),
                "notes": f"Type: {pattern.task_type} | Complexity: {pattern.complexity} | Tool sequence: {' -> '.join(pattern.tool_sequence)}"
            },
            "user_scenario": {
                "persona": scenario.persona,
                "instructions": scenario.instructions,
            },
            "initial_state": None,  # tau2-bench格式需要这个字段
            "evaluation_criteria": {
                "actions": actions,
                "communicate_info": [],  # tau2-bench格式需要这个字段
                "nl_assertions": criteria.nl_assertions,
            },
            "annotations": None,  # tau2-bench格式需要这个字段
        }

    def _save_tasks(
        self,
        tasks: List[Dict[str, Any]],
        domain: str,
        output_dir: str,
        validation_report: Dict[str, Any],
    ):
        """保存任务和报告"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存任务 (JSON格式)
        tasks_file = os.path.join(output_dir, f"synthetic_tasks_{domain}.json")
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        logger.info(f"  任务保存到: {tasks_file}")

        # 保存验证报告
        report_file = os.path.join(output_dir, f"aggregated_stats_{timestamp}.json")
        # 移除详细信息以减小文件大小
        summary_report = {k: v for k, v in validation_report.items() if k != "details"}
        summary_report["timestamp"] = timestamp
        summary_report["domain"] = domain
        summary_report["total_tasks"] = len(tasks)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        logger.info(f"  报告保存到: {report_file}")


def load_config_from_yaml(config_path: str, domain: str = "airline") -> GenerationConfig:
    """
    从YAML加载配置

    Args:
        config_path: 配置文件路径
        domain: 领域名称，用于获取该领域的任务数
    """
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = _expand_env(yaml.safe_load(f))

    # 提取配置
    llm_config = config_dict.get("llm", {})
    generation_config = config_dict.get("generation", {})
    agents_config = config_dict.get("agents", {})
    quality_config = config_dict.get("quality", {})

    # 处理api_pool
    api_pool = llm_config.get("api_pool", [])
    if api_pool:
        first_api = api_pool[0]
        api_base = first_api.get("base_url", "")
        api_key = first_api.get("api_key", "EMPTY")
        model = first_api.get("model", "deepseek-v3")
    else:
        api_base = llm_config.get("api_base", "")
        api_key = llm_config.get("api_key", "EMPTY")
        model = llm_config.get("model", "deepseek-v3")

    # 获取该领域的任务数
    num_tasks_per_domain = generation_config.get("num_tasks_per_domain", {})
    num_tasks = num_tasks_per_domain.get(domain, 100)

    return GenerationConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        task_designer_temperature=agents_config.get("task_designer", {}).get("temperature", 0.7),
        scenario_writer_temperature=agents_config.get("scenario_writer", {}).get("temperature", 0.8),
        criteria_writer_temperature=agents_config.get("criteria_writer", {}).get("temperature", 0.6),
        num_tasks=num_tasks,
        max_concurrent=generation_config.get("max_concurrent", 10),
        complexity_weights=generation_config.get("complexity_weights", None),
        min_reason_length=quality_config.get("min_reason_length", 30),
        min_task_instructions_length=quality_config.get("min_task_instructions_length", 50),
        require_all_actions=quality_config.get("require_all_actions", True),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Task Runner")
    parser.add_argument("--domain", type=str, default="airline", help="Domain")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks (overrides config)")
    parser.add_argument("--output_dir", type=str, default="output/synthetic_tasks", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # 加载配置（传入domain以获取正确的任务数）
    if args.config and os.path.exists(args.config):
        config = load_config_from_yaml(args.config, domain=args.domain)
    else:
        config = GenerationConfig()

    # 命令行参数覆盖配置文件
    if args.seed is not None:
        config.seed = args.seed
    if args.num_tasks is not None:
        config.num_tasks = args.num_tasks

    # 如果命令行没有指定num_tasks，使用配置文件中的值
    num_tasks = args.num_tasks if args.num_tasks is not None else config.num_tasks

    # 运行
    runner = SyntheticTaskRunner(config)
    report = runner.run(
        domain=args.domain,
        output_dir=args.output_dir,
        num_tasks=num_tasks,
    )

    print("\n" + "="*60)
    print("生成完成!")
    print("="*60)
    print(f"领域: {report['domain']}")
    print(f"生成数量: {report['num_generated']}")
    print(f"有效数量: {report['num_valid']}")
    print(f"通过率: {report['pass_rate']:.1%}")
    print(f"平均分: {report['average_score']:.2f}")
    print(f"输出目录: {report['output_dir']}")
