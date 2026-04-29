"""
轨迹分析模块

提供轨迹摘要、成功分析、错误分析、对比分析等功能。
"""

import json
from pathlib import Path
from typing import Optional

from litellm import completion
from loguru import logger

from tau2.data_model.simulation import SimulationRun

from ..data.formatter import build_history
from ..data.models import AttemptRecord


class TrajectoryAnalyzer:
    """轨迹分析器"""

    def __init__(self, model: str, temperature: float = 0.4):
        """
        Args:
            model: 用于分析的 LLM 模型名称
            temperature: 生成温度
        """
        self.model = model
        self.temperature = temperature

    def summarize(
        self,
        attempt: AttemptRecord,
        tools: list,
        system_prompt: str,
    ) -> str:
        """
        生成轨迹摘要。
        
        Args:
            attempt: 尝试记录
            tools: 工具列表
            system_prompt: 系统提示词
            
        Returns:
            轨迹摘要字符串
        """
        simulation = attempt.simulation
        history = build_history(simulation.messages)

        prompt = f"""## Trajectory Analysis Request

**Task ID:** {simulation.task_id}
**Total Steps:** {len(simulation.messages)}

Here are the instruction agent should follow:
{system_prompt}

And tools agent can use:
{tools}

**Detailed Step-by-Step Flow:**
{history}

Please provide a DETAILED English summary of this trajectory that includes:
1. **User Request**: What did the user want to accomplish?
2. **Agent's Approach**: What strategy did the agent use? What steps were taken?
3. **Key Actions**: List the important actions/tool calls made by the agent

Be thorough and specific. Include concrete details about tool calls, parameters, and decisions made at each step.
The summary should be detailed enough but more concise than the raw transcript."""

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior trajectory analyst. Generate information-rich English summaries"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate trajectory summary: {e}")
            summary = f"[summary failed: {e}]"

        return (
            f"## Trajectory Summary\n"
            f"**Task ID:** {simulation.task_id}\n"
            f"**Attempt:** {attempt.attempt}\n"
            f"**Total Steps:** {len(simulation.messages)}\n"
            f"**Final Reward:** {attempt.reward:.2f}\n"
            f"**Termination:** {attempt.termination}\n\n"
            f"**Detailed Analysis:**\n{summary}"
        )

    def analyze_success(
        self,
        simulation: SimulationRun,
        system_prompt: str,
        tools: list = None,
        output_path: str = None,
    ) -> str:
        """
        分析成功轨迹，并展示完整的正确步骤。
        
        Args:
            simulation: 模拟运行结果
            system_prompt: 系统提示词
            tools: 工具列表
            output_path: 输出路径
            
        Returns:
            成功分析字符串
        """
        logger.info(f"生成正样本分析 - Task: {simulation.task_id}")
        history = build_history(simulation.messages)

        prompt = f"""# Success Analysis with Best Practice Examples

**Task ID:** {simulation.task_id}
**Reward:** 1.0 (Success)
**Termination:** {simulation.termination_reason}
**Instruction Must Follow:** {system_prompt}
**Tools Can Use:** {tools}

## Successful Trajectory
{history}

Please provide a detailed analysis with concrete examples:

1. **Success Summary**: Why did this trajectory succeed?
2. **Key Steps**: Show the correct tool calls that were made, using the exact format:

Step 1: [description of what and why]
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>

Step 2: [description]
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>

3. **Best Practices**: What lessons can be learned from this approach?

Use the exact <tool_call> format with proper JSON. Include the most important tool calls that led to success."""

        try:
            resp = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a success analyst. Explain why this trajectory worked well with concrete tool call examples. Use the <tool_call> format to show correct usage."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            
            analysis = resp.choices[0].message.content.strip()
            
            if output_path:
                self._save_analysis(system_prompt, prompt, analysis, tools, output_path)
            
            logger.info(f"正样本分析已生成 - Task: {simulation.task_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate success analysis: {e}")
            return f"[success analysis failed: {e}]"

    def analyze_error(
        self,
        simulation: SimulationRun,
        system_prompt: str,
        tools: list = None,
        output_path: str = None,
    ) -> str:
        """
        分析错误轨迹，并给出正确的做法示例。
        
        Args:
            simulation: 模拟运行结果
            system_prompt: 系统提示词
            tools: 工具列表
            output_path: 输出路径
            
        Returns:
            错误分析字符串
        """
        logger.info(f"生成错误分析 - Task: {simulation.task_id}")
        history = build_history(simulation.messages)

        prompt = f"""# Error Analysis with Correct Approach

**Task ID:** {simulation.task_id}
**Reward:** 0
**Termination:** {simulation.termination_reason}
**Instruction Must Follow:** {system_prompt}
**Tools Can Use:** {tools}

## Failed Trajectory
{history}

Please provide:

1. **Error Analysis**: What went wrong and why? List the key mistakes.

2. **Correct Approach**: Show the correct tool calls that should have been made, using the exact format:

Step 1: [description of what should be done]
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>

Step 2: [description]
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>

3. **Key Lessons**: What should be avoided in the future?

Use the exact <tool_call> format with proper JSON. Show concrete examples of correct tool usage."""

        try:
            resp = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an error analyst. Identify mistakes and provide concrete correct examples using <tool_call> format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            
            analysis = resp.choices[0].message.content.strip()
            
            if output_path:
                self._save_analysis(system_prompt, prompt, analysis, tools, output_path)
            
            logger.info(f"错误分析已生成 - Task: {simulation.task_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate error analysis: {e}")
            return f"[error analysis failed: {e}]"

    def analyze_contrast(
        self,
        failed_simulation: SimulationRun,
        system_prompt: str,
        success_simulation: Optional[SimulationRun] = None,
        tools: list = None,
        output_path: str = None,
    ) -> str:
        """
        对比分析失败和成功轨迹，并排展示错误和正确的做法。
        
        Args:
            failed_simulation: 失败的模拟运行
            system_prompt: 系统提示词
            success_simulation: 成功的模拟运行（可选）
            tools: 工具列表
            output_path: 输出路径
            
        Returns:
            对比分析字符串
        """
        logger.info(f"生成对比分析 - Task: {failed_simulation.task_id}")
        
        fail_history = build_history(failed_simulation.messages)
        success_block = ""
        
        if success_simulation:
            success_history = build_history(success_simulation.messages)
            success_block = f"\n\n## Successful Trajectory (Reference)\n{success_history}"

        prompt = f"""# Contrast Analysis: Failed vs Successful

**Task ID:** {failed_simulation.task_id}
**Failed Reward:** 0
**Failed Termination:** {failed_simulation.termination_reason}
**Instruction Must Follow:** {system_prompt}
**Tools Can Use:** {tools}

## Failed Trajectory
{fail_history}
{success_block}

Please provide a detailed comparison:

1. **Key Differences**: What were the main differences between failed and successful approaches?

2. **Side-by-Side Comparison**: Show the incorrect and correct tool calls:

❌ **Wrong Approach:**
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>
→ Why this failed: [explanation]

✅ **Correct Approach:**
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>
→ Why this works: [explanation]

3. **Critical Lessons**: What is the most important takeaway from this comparison?

Use the exact <tool_call> format with proper JSON. Show concrete examples of both wrong and correct approaches."""

        try:
            resp = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a contrast analyst. Compare failed and successful approaches with concrete tool call examples using <tool_call> format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            
            analysis = resp.choices[0].message.content.strip()
            
            if output_path:
                self._save_analysis(system_prompt, prompt, analysis, tools, output_path)
            
            logger.info(f"对比分析已生成 - Task: {failed_simulation.task_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate contrast analysis: {e}")
            return f"[contrast analysis failed: {e}]"

    def _save_analysis(
        self,
        system_prompt: str,
        prompt: str,
        analysis: str,
        tools: list,
        output_path: str,
    ):
        """保存分析结果到 SFT 数据文件"""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': analysis},
        ]
        
        # 保存到 sft_data.jsonl（和任务执行轨迹混合）
        sft_file = Path(output_path) / "sft_data.jsonl"
        with open(sft_file, "a", encoding='utf-8') as f:
            f.write(json.dumps({'messages': messages, 'tools': json.dumps(tools)}, ensure_ascii=False) + "\n")
