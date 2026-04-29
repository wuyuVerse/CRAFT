"""
TaskDesigner Agent

Agent1: 设计任务的核心目标和reason_for_call
基于提取的tau2模式和参数，使用LLM生成任务设计。
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger

# 复用iterative_learning的LLM客户端
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))

from src.iterative_learning.utils.llm_client import LLMClient


@dataclass
class TaskDesign:
    """任务设计结果"""
    task_id: str                           # 生成的任务ID
    domain: str                            # 领域
    task_type: str                         # 任务类型
    complexity: str                        # 复杂度

    reason_for_call: str                   # 用户来电原因
    task_goal: str                         # 任务目标
    context: str                           # 背景信息

    # 来自原始模式
    tool_sequence: List[str] = field(default_factory=list)  # 期望的工具序列
    original_pattern_id: str = ""          # 原始模式ID
    params: Dict[str, Any] = field(default_factory=dict)    # 采样的参数


class TaskDesigner:
    """
    任务设计Agent

    职责：
    1. 基于tau2模式设计任务核心目标
    2. 生成自然的reason_for_call
    3. 提供任务背景和context
    """

    DESIGN_PROMPT_TEMPLATE = """You are a task design expert. Design a customer service task based on the following information.

## Original Task Pattern
- Domain: {domain}
- Task Type: {task_type}
- Complexity: {complexity}
- Tool Sequence: {tool_sequence}
- Original Reason for Call: {original_reason}

## Sampled Parameters
{params_str}

## Requirements
Generate a new task design containing:

1. **reason_for_call**: The reason for the user's call (50-100 words, natural description, using specific parameters)
2. **task_goal**: The final goal of the task (brief, 20-30 words)
3. **context**: Background information such as user mood, urgency, etc. (optional, 30-50 words)

## Constraints
- reason_for_call MUST use the provided parameters (routes, dates, names, etc.)
- Description should be natural, like real user language
- Do NOT copy original content, create variations
- Output in ENGLISH

## Output Format
Please output strictly in JSON format:
```json
{{
    "reason_for_call": "...",
    "task_goal": "...",
    "context": "..."
}}
```"""

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        初始化TaskDesigner

        Args:
            llm_client: LLM客户端
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def design_task(
        self,
        pattern: "Tau2TaskPattern",
        params: Dict[str, Any],
        task_id: str,
    ) -> TaskDesign:
        """
        设计单个任务

        Args:
            pattern: 从tau2-bench提取的任务模式
            params: 采样的参数（航线、日期、用户信息等）
            task_id: 生成的任务ID

        Returns:
            TaskDesign对象
        """
        # 构建参数字符串
        params_str = self._format_params(params, pattern.domain)

        # 构建prompt
        prompt = self.DESIGN_PROMPT_TEMPLATE.format(
            domain=pattern.domain,
            task_type=pattern.task_type,
            complexity=pattern.complexity,
            tool_sequence=" → ".join(pattern.tool_sequence),
            original_reason=pattern.reason_for_call[:200],
            params_str=params_str,
        )

        # 调用LLM（带重试）
        design_data = None
        max_retries = 3
        for attempt in range(max_retries):
            response = self.llm_client.generate_from_prompt(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            design_data = self._parse_response(response)
            if design_data:
                break
            if attempt < max_retries - 1:
                logger.warning(f"LLM failed for task {task_id}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(1)

        # 如果所有重试都失败，返回None让调用者跳过此任务
        if not design_data:
            logger.error(f"LLM failed for task {task_id} after {max_retries} retries, skipping")
            return None

        return TaskDesign(
            task_id=task_id,
            domain=pattern.domain,
            task_type=pattern.task_type,
            complexity=pattern.complexity,
            reason_for_call=design_data.get("reason_for_call", pattern.reason_for_call),
            task_goal=design_data.get("task_goal", "完成客户请求"),
            context=design_data.get("context", ""),
            tool_sequence=pattern.tool_sequence,
            original_pattern_id=pattern.task_id,
            params=params,
        )

    def _format_params(self, params: Dict[str, Any], domain: str) -> str:
        """格式化参数为易读字符串"""
        lines = []

        if domain == "airline":
            if "origin" in params:
                lines.append(f"- 出发地: {params['origin']}")
            if "destination" in params:
                lines.append(f"- 目的地: {params['destination']}")
            if "date" in params:
                lines.append(f"- 日期: {params['date']}")
            if "first_name" in params and "last_name" in params:
                lines.append(f"- 乘客姓名: {params['first_name']} {params['last_name']}")
            if "confirmation" in params:
                lines.append(f"- 预订号: {params['confirmation']}")
            if "cabin" in params:
                lines.append(f"- 舱位: {params['cabin']}")

        elif domain == "retail":
            if "order_id" in params:
                lines.append(f"- 订单号: {params['order_id']}")
            if "user_name" in params:
                lines.append(f"- 用户名: {params['user_name']}")
            if "product_name" in params:
                lines.append(f"- 商品: {params['product_name']}")
            if "address" in params:
                lines.append(f"- 地址: {params['address']}")

        elif domain == "telecom":
            if "user_id" in params:
                lines.append(f"- 用户ID: {params['user_id']}")
            if "phone_number" in params:
                lines.append(f"- 电话号码: {params['phone_number']}")
            if "plan_name" in params:
                lines.append(f"- 套餐: {params['plan_name']}")

        # 通用参数
        for key, value in params.items():
            if key not in ["origin", "destination", "date", "first_name", "last_name",
                          "confirmation", "cabin", "order_id", "user_name", "product_name",
                          "address", "user_id", "phone_number", "plan_name"]:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) if lines else "无特定参数"

    def _parse_response(self, response: Optional[str]) -> Optional[Dict[str, Any]]:
        """解析LLM响应"""
        if not response:
            return None

        try:
            # 尝试提取JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)

            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None

    def _fallback_design(self, pattern: "Tau2TaskPattern", params: Dict[str, Any]) -> Dict[str, Any]:
        """当LLM失败时的fallback设计"""

        # 基于模式类型生成基础设计
        domain = pattern.domain
        task_type = pattern.task_type

        if domain == "airline":
            if task_type == "cancellation":
                reason = f"我需要取消预订号为{params.get('confirmation', 'XXXXX')}的航班"
                goal = "取消航班预订"
            elif task_type == "booking":
                reason = f"我想预订从{params.get('origin', 'A城市')}到{params.get('destination', 'B城市')}的航班"
                goal = "预订航班"
            elif task_type == "modification":
                reason = f"我需要修改预订号{params.get('confirmation', 'XXXXX')}的航班信息"
                goal = "修改航班预订"
            else:
                reason = f"我想查询预订号{params.get('confirmation', 'XXXXX')}的航班状态"
                goal = "查询航班信息"
        else:
            # 通用fallback
            reason = pattern.reason_for_call
            goal = f"完成{task_type}相关请求"

        return {
            "reason_for_call": reason,
            "task_goal": goal,
            "context": "用户态度友好，希望尽快解决问题",
        }


if __name__ == "__main__":
    # 测试代码
    from src.synthetic_gen.core.tau2.extractors.task_extractor import Tau2TaskExtractor
    from src.synthetic_gen.core.tau2.extractors.parameter_extractor import Tau2ParameterAnalyzer

    # 创建LLM客户端
    llm_client = LLMClient(
        model="deepseek-v3",
        api_base=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key="EMPTY",
    )

    # 创建设计器
    designer = TaskDesigner(llm_client)

    # 提取模式和参数
    extractor = Tau2TaskExtractor()
    analyzer = Tau2ParameterAnalyzer()

    patterns = extractor.extract_patterns("airline")
    params = analyzer.analyze_parameter_space("airline")

    # 测试生成
    if patterns:
        pattern = patterns[0]
        sample_params = {
            "origin": "JFK",
            "destination": "LAX",
            "date": "2024-03-15",
            "first_name": "John",
            "last_name": "Doe",
            "confirmation": "ABC123",
        }

        design = designer.design_task(pattern, sample_params, "test_001")
        print(f"Task ID: {design.task_id}")
        print(f"Reason: {design.reason_for_call}")
        print(f"Goal: {design.task_goal}")
        print(f"Context: {design.context}")
