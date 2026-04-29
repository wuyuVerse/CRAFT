"""
ScenarioWriter Agent

Agent2: 编写详细的user_scenario
基于TaskDesign生成完整的用户场景描述。
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))

from src.iterative_learning.utils.llm_client import LLMClient
from .task_designer import TaskDesign


@dataclass
class UserScenario:
    """用户场景定义"""
    persona: str                          # 用户画像
    instructions: Dict[str, Any]          # 指令详情

    # 指令详情展开
    domain: str = ""
    reason_for_call: str = ""
    known_info: str = ""
    unknown_info: str = ""
    task_instructions: str = ""


class ScenarioWriter:
    """
    场景编写Agent

    职责：
    1. 生成详细的persona描述
    2. 编写known_info（用户已知信息）
    3. 编写unknown_info（用户不知道的信息）
    4. 编写task_instructions（用户行为指导）
    """

    # Persona模板库
    PERSONA_TEMPLATES = [
        "Professional and direct communicator who values efficiency",
        "Friendly and patient customer who is understanding of delays",
        "Busy professional who needs quick resolution",
        "Polite but firm customer who knows their rights",
        "First-time user who may need additional guidance",
        "Experienced traveler who knows the process well",
        "Anxious customer who needs reassurance",
        "Detail-oriented person who wants to verify everything",
    ]

    SCENARIO_PROMPT_TEMPLATE = """你是一个场景编写专家。基于以下任务设计，编写详细的用户场景。

## 任务设计
- 领域: {domain}
- 任务类型: {task_type}
- 来电原因: {reason_for_call}
- 任务目标: {task_goal}
- 背景: {context}
- 工具序列: {tool_sequence}

## 参数
{params_str}

## 要求
请生成完整的用户场景，包含：

1. **persona**: 用户的性格和沟通风格（30-50字英文）
   - 例如: "Professional and direct communicator who values efficiency"
   - 描述用户的态度、耐心程度、沟通方式

2. **known_info**: 用户已知的信息（具体列出）
   - 使用具体参数值
   - 用英文描述
   - 例如: "Confirmation number: ABC123, Passenger name: John Doe"

3. **unknown_info**: 用户不知道但可能被问到的信息（可选）
   - 列出用户可能不清楚的细节
   - 可以为空字符串

4. **task_instructions**: 用户应该如何与agent交互的详细指导（100-150字英文）
   - 描述用户应该做什么
   - 需要问什么问题
   - 如何响应agent的请求
   - 最终目标是什么

## 输出格式
请严格按JSON格式输出:
```json
{{
    "persona": "...",
    "known_info": "...",
    "unknown_info": "...",
    "task_instructions": "..."
}}
```"""

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.8,
        max_tokens: int = 800,
    ):
        """
        初始化ScenarioWriter

        Args:
            llm_client: LLM客户端
            temperature: 生成温度（稍高以增加多样性）
            max_tokens: 最大token数
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def write_scenario(self, task_design: TaskDesign) -> UserScenario:
        """
        编写用户场景

        Args:
            task_design: TaskDesigner生成的任务设计

        Returns:
            UserScenario对象
        """
        # 构建参数字符串
        params_str = self._format_params(task_design.params, task_design.domain)

        # 构建prompt
        prompt = self.SCENARIO_PROMPT_TEMPLATE.format(
            domain=task_design.domain,
            task_type=task_design.task_type,
            reason_for_call=task_design.reason_for_call,
            task_goal=task_design.task_goal,
            context=task_design.context,
            tool_sequence=" → ".join(task_design.tool_sequence),
            params_str=params_str,
        )

        # 调用LLM（带重试）
        scenario_data = None
        max_retries = 3
        for attempt in range(max_retries):
            response = self.llm_client.generate_from_prompt(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            scenario_data = self._parse_response(response)
            if scenario_data:
                break
            if attempt < max_retries - 1:
                logger.warning(f"LLM failed for scenario {task_design.task_id}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(1)

        # 如果所有重试都失败，返回None
        if not scenario_data:
            logger.error(f"LLM failed for scenario {task_design.task_id} after {max_retries} retries, skipping")
            return None

        # 构建instructions字典
        instructions = {
            "domain": task_design.domain,
            "reason_for_call": task_design.reason_for_call,
            "known_info": scenario_data.get("known_info", ""),
            "unknown_info": scenario_data.get("unknown_info", ""),
            "task_instructions": scenario_data.get("task_instructions", ""),
        }

        return UserScenario(
            persona=scenario_data.get("persona", self.PERSONA_TEMPLATES[0]),
            instructions=instructions,
            domain=task_design.domain,
            reason_for_call=task_design.reason_for_call,
            known_info=scenario_data.get("known_info", ""),
            unknown_info=scenario_data.get("unknown_info", ""),
            task_instructions=scenario_data.get("task_instructions", ""),
        )

    def _format_params(self, params: Dict[str, Any], domain: str) -> str:
        """格式化参数"""
        lines = []
        for key, value in params.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "无特定参数"

    def _parse_response(self, response: Optional[str]) -> Optional[Dict[str, Any]]:
        """解析LLM响应"""
        if not response:
            return None

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)

            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None

    def _fallback_scenario(self, task_design: TaskDesign) -> Dict[str, Any]:
        """当LLM失败时的fallback场景"""
        import random

        domain = task_design.domain
        params = task_design.params

        # 选择随机persona
        persona = random.choice(self.PERSONA_TEMPLATES)

        # 构建known_info
        if domain == "airline":
            known_parts = []
            if params.get("confirmation"):
                known_parts.append(f"Confirmation number: {params['confirmation']}")
            if params.get("first_name") and params.get("last_name"):
                known_parts.append(f"Passenger name: {params['first_name']} {params['last_name']}")
            if params.get("origin") and params.get("destination"):
                known_parts.append(f"Route: {params['origin']} to {params['destination']}")
            if params.get("date"):
                known_parts.append(f"Date: {params['date']}")
            known_info = ", ".join(known_parts) if known_parts else "Basic booking information"

        elif domain == "retail":
            known_parts = []
            if params.get("order_id"):
                known_parts.append(f"Order ID: {params['order_id']}")
            if params.get("user_name"):
                known_parts.append(f"Username: {params['user_name']}")
            known_info = ", ".join(known_parts) if known_parts else "Order details"

        else:
            known_info = ", ".join([f"{k}: {v}" for k, v in params.items()][:3])

        # 构建task_instructions
        task_type = task_design.task_type
        if task_type == "cancellation":
            task_instructions = f"Request to cancel the booking. Provide your {', '.join(list(params.keys())[:2])} when asked. Confirm the cancellation when the agent processes it. Ask about any cancellation fees or refund policy if applicable."
        elif task_type == "booking":
            task_instructions = f"Request to make a new booking. Provide your preferred dates and requirements. Confirm the booking details when presented. Ask about pricing and policies."
        elif task_type == "modification":
            task_instructions = f"Request to modify the existing booking. Explain what changes you need. Provide verification information when asked. Confirm the modifications."
        else:
            task_instructions = f"Explain your request clearly. Provide the necessary information when asked. Follow up on any questions from the agent. Confirm the resolution."

        return {
            "persona": persona,
            "known_info": known_info,
            "unknown_info": "",
            "task_instructions": task_instructions,
        }


if __name__ == "__main__":
    # 测试代码
    from src.synthetic_gen.core.tau2.extractors.task_extractor import Tau2TaskExtractor
    from .task_designer import TaskDesigner

    # 创建LLM客户端
    llm_client = LLMClient(
        model="deepseek-v3",
        api_base=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key="EMPTY",
    )

    # 创建设计器和场景编写器
    designer = TaskDesigner(llm_client)
    writer = ScenarioWriter(llm_client)

    # 提取模式
    extractor = Tau2TaskExtractor()
    patterns = extractor.extract_patterns("airline")

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

        # 设计任务
        design = designer.design_task(pattern, sample_params, "test_001")

        # 编写场景
        scenario = writer.write_scenario(design)
        print(f"Persona: {scenario.persona}")
        print(f"Known Info: {scenario.known_info}")
        print(f"Task Instructions: {scenario.task_instructions}")
