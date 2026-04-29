"""
迭代学习命令行入口
"""

import argparse
import os
import re
from pathlib import Path

import yaml
from loguru import logger

from .runners import MultiDomainRunner
from .utils import setup_logging


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        return _expand_env(yaml.safe_load(f))


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


def main():
    parser = argparse.ArgumentParser(description="Iterative Learning for tau2-bench")
    parser.add_argument(
        '--config', type=str, required=True,
        help='配置文件路径 (YAML 格式)'
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    
    # 解析配置
    domains = config.get("domains", ["airline", "retail", "telecom"])
    
    llm_config = config.get("llm", {})
    agent_llm = llm_config.get("agent", "openai/deepseek-v3")
    user_llm = llm_config.get("user", "openai/deepseek-v3")
    analysis_llm = llm_config.get("analysis", "openai/deepseek-v3")
    llm_args = llm_config.get("args", {})
    
    # 设置 api_base
    api_base = llm_config.get("api_base")
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    
    task_config = config.get("task", {})
    max_attempts = task_config.get("max_attempts", 5)
    max_steps = task_config.get("max_steps", 100)
    num_trials = task_config.get("num_trials", 1)
    num_trials_per_domain = task_config.get("num_trials_per_domain", {})  # 按领域设置
    task_ids = task_config.get("task_ids")
    tasks_path = task_config.get("tasks_path")  # 自定义任务文件路径（支持synthetic数据）
    
    concurrency_config = config.get("concurrency", {})
    max_concurrency = concurrency_config.get("max_concurrency", 10)
    analysis_concurrency = concurrency_config.get("analysis_concurrency", 3)
    
    enhancement_config = config.get("enhancement", {})
    enable_cot = enhancement_config.get("enable_cot", True)
    quality_threshold = enhancement_config.get("quality_threshold", 0.6)
    prioritize_weak = enhancement_config.get("prioritize_weak", True)
    enable_contrast_feedback = enhancement_config.get("enable_contrast_feedback", True)
    
    # V3/V4 错误注入配置
    error_injection_config = config.get("error_injection", {})
    enable_error_injection = error_injection_config.get("enabled", False)
    error_injection_base_rate = error_injection_config.get("base_rate", 0.12)
    error_injection_max_errors = error_injection_config.get("max_errors_per_task", 8)
    error_db_path = error_injection_config.get("error_db_path")
    injection_mode = error_injection_config.get("mode", "rule")  # V4: "agent" 或 "rule"
    correct_trajectory_weight = error_injection_config.get("correct_trajectory_weight", 1)  # 正确轨迹重复次数
    save_analysis_data = error_injection_config.get("save_analysis_data", False)  # 是否保存分析数据
    include_failed_in_sft = error_injection_config.get("include_failed_in_sft", False)  # 是否将失败轨迹加入SFT数据

    # V4 Agent模式配置
    agent_config = error_injection_config.get("agent", {})
    use_llm_for_recovery = agent_config.get("use_llm_for_recovery", True)
    error_type_weights = agent_config.get("error_type_weights")
    
    output_config = config.get("output", {})
    output_dir = Path(output_config.get("dir", "./output/iterative_learning"))
    resume = output_config.get("resume", False)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志
    setup_logging(output_dir)

    # 打印配置
    logger.info("=" * 60)
    logger.info("Iterative Learning 配置")
    logger.info("=" * 60)
    logger.info(f"Domains: {domains}")
    logger.info(f"Agent LLM: {agent_llm}")
    logger.info(f"User LLM: {user_llm}")
    logger.info(f"Analysis LLM: {analysis_llm}")
    logger.info(f"Max Attempts: {max_attempts}")
    logger.info(f"Max Steps: {max_steps}")
    logger.info(f"Num Trials: {num_trials} (default)")
    if num_trials_per_domain:
        logger.info(f"Num Trials Per Domain:")
        for domain, trials in num_trials_per_domain.items():
            logger.info(f"  - {domain}: {trials}")
    logger.info(f"Max Concurrency: {max_concurrency}")
    logger.info(f"Enable CoT: {enable_cot}")
    logger.info(f"Quality Threshold: {quality_threshold}")
    logger.info(f"Prioritize Weak: {prioritize_weak}")
    logger.info(f"Enable Contrast Feedback: {enable_contrast_feedback}")
    logger.info(f"Enable Error Injection: {enable_error_injection}")
    if enable_error_injection:
        logger.info(f"  - Mode: {injection_mode}")
        logger.info(f"  - Base Rate: {error_injection_base_rate}")
        logger.info(f"  - Max Errors Per Task: {error_injection_max_errors}")
        logger.info(f"  - Correct Trajectory Weight: {correct_trajectory_weight}")
        logger.info(f"  - Save Analysis Data: {save_analysis_data}")
        logger.info(f"  - Include Failed in SFT: {include_failed_in_sft}")
        logger.info(f"  - Error DB Path: {error_db_path}")
        if injection_mode == "agent":
            logger.info(f"  - Use LLM for Recovery: {use_llm_for_recovery}")
            logger.info(f"  - Error Type Weights: {error_type_weights}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Resume: {resume}")
    logger.info("=" * 60)

    # 打印任务路径
    if tasks_path:
        logger.info(f"Tasks Path: {tasks_path}")

    # 创建执行器
    runner = MultiDomainRunner(
        domains=domains,
        agent_llm=agent_llm,
        user_llm=user_llm,
        analysis_llm=analysis_llm,
        max_attempts=max_attempts,
        max_steps=max_steps,
        llm_args=llm_args,
        output_dir=str(output_dir),
        max_concurrency=max_concurrency,
        analysis_concurrency=analysis_concurrency,
        enable_cot=enable_cot,
        quality_threshold=quality_threshold,
        prioritize_weak=prioritize_weak,
        enable_contrast_feedback=enable_contrast_feedback,
        enable_error_injection=enable_error_injection,
        error_injection_base_rate=error_injection_base_rate,
        error_injection_max_errors=error_injection_max_errors,
        error_db_path=error_db_path,
        # V4 新增参数
        injection_mode=injection_mode,
        use_llm_for_recovery=use_llm_for_recovery,
        error_type_weights=error_type_weights,
        api_base=api_base,
        correct_trajectory_weight=correct_trajectory_weight,
        save_analysis_data=save_analysis_data,
        include_failed_in_sft=include_failed_in_sft,
        # 自定义任务路径
        tasks_path=tasks_path,
    )

    # 执行任务
    results = runner.run(
        task_ids=task_ids,
        num_trials=num_trials,
        num_trials_per_domain=num_trials_per_domain,
        resume=resume,
    )

    # 保存统计
    runner.save_statistics(results, output_dir)
    
    # 保存使用的配置副本
    config_save_path = output_dir / "config_used.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
