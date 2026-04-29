"""
失败分析模块

生成用于下一次尝试的改进建议，支持对比分析。
"""

import json
from typing import Optional, List

from litellm import completion
from loguru import logger

from tau2.data_model.message import Message
from tau2.data_model.simulation import SimulationRun

from ..data.models import AttemptRecord
from ..data.formatter import build_history
from .trajectory_analyzer import TrajectoryAnalyzer


class FailureAnalyzer:
    """失败分析器，生成改进建议用于下一次尝试"""

    def __init__(self, model: str, temperature: float = 0.3):
        """
        Args:
            model: 用于分析的 LLM 模型名称
            temperature: 生成温度
        """
        self.model = model
        self.temperature = temperature
        self.trajectory_analyzer = TrajectoryAnalyzer(model, temperature=0.4)

    def build_analysis(
        self,
        attempt: AttemptRecord,
        tools: list,
        system_prompt: str,
    ) -> str:
        """
        基于失败轨迹生成改进分析。
        
        Args:
            attempt: 尝试记录
            tools: 工具列表
            system_prompt: 系统提示词
            
        Returns:
            改进分析字符串，用于注入到下一次尝试的 prompt 中
        """
        # 先生成轨迹摘要
        history_summary = self.trajectory_analyzer.summarize(attempt, tools, system_prompt)
        simulation = attempt.simulation

        prompt = f"""# Task Failure Analysis

## Task Information
- Task ID: {simulation.task_id}
- Attempt: {attempt.attempt}
- Final Reward: {attempt.reward:.2f}
- Termination: {attempt.termination}
- Steps: {len(simulation.messages)}
- Instruction Must Follow: {system_prompt}
- Tools Can Use: {tools}

## Trajectory Summary
{history_summary}

Return ONLY JSON in the format:
{{
    "failure_reasons": "Detailed explanation of why it failed, including which rules or policies were violated",
    "key_mistakes": [
        "Key mistake 1: Specifically describe what was done wrong at which step",
        "Key mistake 2: ...",
        "Key mistake 3: ..."
    ],
    "improvement_suggestions": [
        "Suggestion 1: How to improve next time",
        "Suggestion 2: What to pay attention to",
        "Suggestion 3: The correct step sequence",
        "Suggestion 4: The correct way to call tools",
        "Suggestion 5: Improvements to dialogue strategy"
    ],
    "correct_approach": "Detailed description of the complete method and steps to correctly complete this task"
}}

Focus on:
1. Which domain policies or rules were violated
2. Whether tool calls were correct (parameters, order, timing)
3. Whether the dialogue strategy was appropriate
4. Whether key steps were missed
5. How to improve in the next attempt

Provide specific and actionable answers."""

        try:
            resp = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict failure analyst. Return actionable next-step improvements. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            content = resp.choices[0].message.content
            analysis_json = json.loads(content)
            
        except Exception as e:
            logger.error(f"Failed to build failure analysis: {e}")
            return f"[failure analysis error: {e}]"

        return self._format_analysis(simulation.task_id, attempt.attempt, attempt.reward, analysis_json)

    def build_contrast_analysis(
        self,
        failed_attempt: AttemptRecord,
        success_simulation: SimulationRun,
        tools: list,
        system_prompt: str,
    ) -> str:
        """
        基于失败轨迹和成功轨迹的对比生成改进分析。
        
        这个分析会注入到下一次 retry 的 prompt 中，帮助模型理解正确做法。
        
        Args:
            failed_attempt: 失败的尝试记录
            success_simulation: 成功的模拟运行
            tools: 工具列表
            system_prompt: 系统提示词
            
        Returns:
            对比分析字符串，用于注入到下一次尝试的 prompt 中
        """
        failed_sim = failed_attempt.simulation
        
        # 构建两个轨迹的历史
        failed_history = build_history(failed_sim.messages)
        success_history = build_history(success_simulation.messages)

        prompt = f"""# Contrast Analysis: Failed vs Successful Trajectory

## Task Information
- Task ID: {failed_sim.task_id}
- Failed Attempt: {failed_attempt.attempt}
- Failed Reward: {failed_attempt.reward:.2f}
- Instruction Must Follow: {system_prompt}
- Tools Can Use: {tools}

## Failed Trajectory
{failed_history}

## Successful Trajectory (Reference)
{success_history}

Compare the failed trajectory with the successful one and return ONLY JSON:
{{
    "divergence_point": "Describe where the two trajectories started to differ",
    "failure_reasons": "Why did the failed trajectory fail compared to the successful one",
    "key_differences": [
        "Difference 1: What the failed trajectory did vs what the successful one did",
        "Difference 2: ...",
        "Difference 3: ..."
    ],
    "correct_steps": [
        "Step 1: What should be done first (from successful trajectory)",
        "Step 2: ...",
        "Step 3: ..."
    ],
    "critical_lessons": "The most important lesson to learn from this comparison"
}}

Focus on:
1. Where exactly did the failed trajectory go wrong
2. What tool calls were different (name, parameters, order)
3. What the correct sequence of actions should be
4. Specific and actionable guidance for the next attempt"""

        try:
            resp = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a contrast analyst. Compare failed and successful trajectories to provide actionable improvements. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            content = resp.choices[0].message.content
            analysis_json = json.loads(content)
            
        except Exception as e:
            logger.error(f"Failed to build contrast analysis: {e}")
            return f"[contrast analysis error: {e}]"

        return self._format_contrast_analysis(
            failed_sim.task_id, 
            failed_attempt.attempt, 
            failed_attempt.reward, 
            analysis_json
        )

    def _format_analysis(
        self,
        task_id: str,
        attempt: int,
        reward: float,
        analysis_json: dict,
    ) -> str:
        """格式化分析结果"""
        failure_reasons = analysis_json.get("failure_reasons", "")
        key_mistakes = analysis_json.get("key_mistakes", [])
        suggestions = analysis_json.get("improvement_suggestions", [])
        correct_approach = analysis_json.get("correct_approach", "")

        lines = [
            "## Failure Analysis",
            f"- Task ID: {task_id}",
            f"- Attempt: {attempt}",
            f"- Reward: {reward:.2f}",
            "",
            "### Failure Reasons",
            failure_reasons,
            "",
            "### Key Mistakes",
            *[f"{idx+1}. {item}" for idx, item in enumerate(key_mistakes)],
            "",
            "### Improvement Suggestions",
            *[f"{idx+1}. {item}" for idx, item in enumerate(suggestions)],
            "",
            "### Correct Approach Reference",
            correct_approach,
        ]

        return "\n".join(line for line in lines if line)

    def _format_contrast_analysis(
        self,
        task_id: str,
        attempt: int,
        reward: float,
        analysis_json: dict,
    ) -> str:
        """格式化对比分析结果"""
        divergence_point = analysis_json.get("divergence_point", "")
        failure_reasons = analysis_json.get("failure_reasons", "")
        key_differences = analysis_json.get("key_differences", [])
        correct_steps = analysis_json.get("correct_steps", [])
        critical_lessons = analysis_json.get("critical_lessons", "")

        lines = [
            "## Contrast Analysis (Failed vs Successful)",
            f"- Task ID: {task_id}",
            f"- Failed Attempt: {attempt}",
            f"- Reward: {reward:.2f}",
            "",
            "### Divergence Point",
            divergence_point,
            "",
            "### Why It Failed",
            failure_reasons,
            "",
            "### Key Differences from Successful Trajectory",
            *[f"{idx+1}. {item}" for idx, item in enumerate(key_differences)],
            "",
            "### Correct Steps to Follow",
            *[f"{idx+1}. {item}" for idx, item in enumerate(correct_steps)],
            "",
            "### Critical Lesson",
            critical_lessons,
        ]

        return "\n".join(line for line in lines if line)
