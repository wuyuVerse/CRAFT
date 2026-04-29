"""
CriteriaWriter Agent

Agent3: 编写evaluation_criteria
基于TaskDesign和UserScenario生成评估标准。
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
from .scenario_writer import UserScenario


@dataclass
class EvaluationCriteria:
    """评估标准定义"""
    actions: List[Dict[str, Any]]        # 期望的工具调用序列
    nl_assertions: List[str]             # 自然语言断言
    reward_basis: List[str]              # 评估基础


class CriteriaWriter:
    """
    标准编写Agent

    职责：
    1. 基于工具序列生成具体的actions
    2. 编写nl_assertions（自然语言断言）
    3. 确定reward_basis（评估基础）
    """

    CRITERIA_PROMPT_TEMPLATE = """你是一个评估标准专家。基于以下任务设计和用户场景，编写评估标准。

## 任务信息
- 领域: {domain}
- 任务类型: {task_type}
- 来电原因: {reason_for_call}
- 工具序列: {tool_sequence}

## 用户场景
- Persona: {persona}
- Known Info: {known_info}
- Task Instructions: {task_instructions}

## 参数
{params_str}

## 原始Actions模板
{original_actions}

## 要求
请基于以上信息生成评估标准：

1. **actions**: 期望的工具调用序列（JSON数组）
   - 每个action包含: action_id, name, arguments, compare_args
   - arguments使用提供的参数值
   - compare_args指定需要验证的参数

2. **nl_assertions**: 自然语言断言（英文字符串数组）
   - 描述agent应该做的事情
   - 例如: "Agent should verify customer identity before proceeding"
   - 3-5条断言

3. **reward_basis**: 评估基础
   - 选择: ["DB"], ["ACTION"], ["DB", "ACTION"], ["COMMUNICATE"] 之一
   - DB: 检查数据库状态变化
   - ACTION: 检查工具调用
   - COMMUNICATE: 检查沟通结果

## 输出格式
请严格按JSON格式输出:
```json
{{
    "actions": [
        {{
            "action_id": "action_1",
            "name": "tool_name",
            "arguments": {{"param": "value"}},
            "compare_args": ["param"]
        }}
    ],
    "nl_assertions": [
        "Agent should ...",
        "Agent must ..."
    ],
    "reward_basis": ["DB", "ACTION"]
}}
```"""

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.6,
        max_tokens: int = 1000,
    ):
        """
        初始化CriteriaWriter

        Args:
            llm_client: LLM客户端
            temperature: 生成温度（较低以保证准确性）
            max_tokens: 最大token数
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def write_criteria(
        self,
        task_design: TaskDesign,
        scenario: UserScenario,
        original_actions: List[Dict[str, Any]],
    ) -> EvaluationCriteria:
        """
        编写评估标准

        Args:
            task_design: TaskDesigner生成的任务设计
            scenario: ScenarioWriter生成的用户场景
            original_actions: 原始模式中的actions模板

        Returns:
            EvaluationCriteria对象
        """
        # 构建参数字符串
        params_str = self._format_params(task_design.params)

        # 格式化原始actions
        original_actions_str = json.dumps(original_actions, indent=2, ensure_ascii=False) if original_actions else "[]"

        # 构建prompt
        prompt = self.CRITERIA_PROMPT_TEMPLATE.format(
            domain=task_design.domain,
            task_type=task_design.task_type,
            reason_for_call=task_design.reason_for_call,
            tool_sequence=" → ".join(task_design.tool_sequence),
            persona=scenario.persona,
            known_info=scenario.known_info,
            task_instructions=scenario.task_instructions,
            params_str=params_str,
            original_actions=original_actions_str,
        )

        # 调用LLM（带重试）
        criteria_data = None
        max_retries = 3
        for attempt in range(max_retries):
            response = self.llm_client.generate_from_prompt(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            criteria_data = self._parse_response(response)

            # 检查是否有占位符
            if criteria_data and self._has_placeholders(criteria_data.get("actions", [])):
                logger.warning(f"LLM generated placeholders for {task_design.task_id}, retrying ({attempt + 1}/{max_retries})...")
                criteria_data = None

            if criteria_data:
                break
            if attempt < max_retries - 1:
                logger.warning(f"LLM failed for criteria {task_design.task_id}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(1)

        # 如果所有重试都失败，返回None
        if not criteria_data:
            logger.error(f"LLM failed for criteria {task_design.task_id} after {max_retries} retries, skipping")
            return None

        return EvaluationCriteria(
            actions=criteria_data.get("actions", []),
            nl_assertions=criteria_data.get("nl_assertions", []),
            reward_basis=criteria_data.get("reward_basis", ["ACTION"]),
        )

    def _format_params(self, params: Dict[str, Any]) -> str:
        """格式化参数"""
        lines = []
        for key, value in params.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "无特定参数"

    def _has_placeholders(self, actions: List[Dict[str, Any]]) -> bool:
        """检查actions中是否包含占位符"""
        for action in actions:
            args = action.get("arguments", {})
            for key, val in args.items():
                if isinstance(val, str):
                    # 检查常见占位符模式
                    if '<' in val and '>' in val:
                        return True
                    if '{{' in val and '}}' in val:
                        return True
                    if val.upper() == val and '_' in val and len(val) > 10:
                        # 检查类似 PAYMENT_METHOD_ID 的占位符
                        return True
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and '<' in item and '>' in item:
                            return True
        return False

    def _parse_response(self, response: Optional[str]) -> Optional[Dict[str, Any]]:
        """解析LLM响应"""
        if not response:
            return None

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                # 验证必要字段
                if "actions" not in data:
                    data["actions"] = []
                if "nl_assertions" not in data:
                    data["nl_assertions"] = []
                if "reward_basis" not in data:
                    data["reward_basis"] = ["ACTION"]

                return data

            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None

    def _fallback_criteria(
        self,
        task_design: TaskDesign,
        original_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """当LLM失败时的fallback标准"""

        # 基于原始actions和新参数构建actions
        new_actions = []
        params = task_design.params

        for i, orig_action in enumerate(original_actions):
            new_action = {
                "action_id": f"action_{i+1}",
                "name": orig_action.get("name", "unknown"),
                "arguments": self._substitute_arguments(
                    orig_action.get("arguments", {}),
                    params,
                    task_design.domain
                ),
                "compare_args": orig_action.get("compare_args", []),
            }
            new_actions.append(new_action)

        # 生成基本的nl_assertions
        nl_assertions = self._generate_basic_assertions(task_design)

        # 确定reward_basis
        if new_actions:
            reward_basis = ["DB", "ACTION"]
        else:
            reward_basis = ["COMMUNICATE"]

        return {
            "actions": new_actions,
            "nl_assertions": nl_assertions,
            "reward_basis": reward_basis,
        }

    def _substitute_arguments(
        self,
        orig_args: Dict[str, Any],
        params: Dict[str, Any],
        domain: str,
    ) -> Dict[str, Any]:
        """替换参数值"""
        new_args = dict(orig_args)

        # 参数映射
        param_mapping = {
            "airline": {
                "reservation_id": params.get("confirmation"),
                "confirmation_number": params.get("confirmation"),
                "passenger_first_name": params.get("first_name"),
                "passenger_last_name": params.get("last_name"),
                "origin": params.get("origin"),
                "destination": params.get("destination"),
                "date": params.get("date"),
            },
            "retail": {
                "order_id": params.get("order_id"),
                "user_id": params.get("user_id"),
                "user_name": params.get("user_name"),
            },
            "telecom": {
                "user_id": params.get("user_id"),
                "phone_number": params.get("phone_number"),
            },
        }

        mapping = param_mapping.get(domain, {})

        for key, value in new_args.items():
            if key in mapping and mapping[key] is not None:
                new_args[key] = mapping[key]

        return new_args

    def _generate_basic_assertions(self, task_design: TaskDesign) -> List[str]:
        """生成基本的nl_assertions"""
        assertions = []

        task_type = task_design.task_type

        # 通用断言
        assertions.append("Agent should verify customer identity before proceeding with any changes")

        # 任务类型特定断言
        if task_type == "cancellation":
            assertions.append("Agent should confirm the cancellation request with the customer")
            assertions.append("Agent should inform the customer about any applicable fees or refund policy")
        elif task_type == "booking":
            assertions.append("Agent should verify the requested booking details")
            assertions.append("Agent should confirm the booking with the customer before finalizing")
        elif task_type == "modification":
            assertions.append("Agent should confirm the modification details with the customer")
            assertions.append("Agent should verify any additional charges before proceeding")
        elif task_type == "inquiry":
            assertions.append("Agent should provide accurate information based on the customer's query")
            assertions.append("Agent should ask if the customer needs any additional assistance")
        else:
            assertions.append("Agent should address the customer's request appropriately")
            assertions.append("Agent should confirm the resolution with the customer")

        return assertions


if __name__ == "__main__":
    # 测试代码
    from src.synthetic_gen.core.tau2.extractors.task_extractor import Tau2TaskExtractor
    from .task_designer import TaskDesigner
    from .scenario_writer import ScenarioWriter

    # 创建LLM客户端
    llm_client = LLMClient(
        model="deepseek-v3",
        api_base=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key="EMPTY",
    )

    # 创建所有agents
    designer = TaskDesigner(llm_client)
    scenario_writer = ScenarioWriter(llm_client)
    criteria_writer = CriteriaWriter(llm_client)

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
        scenario = scenario_writer.write_scenario(design)

        # 编写评估标准
        criteria = criteria_writer.write_criteria(design, scenario, pattern.expected_actions)

        print(f"Actions: {json.dumps(criteria.actions, indent=2)}")
        print(f"NL Assertions: {criteria.nl_assertions}")
        print(f"Reward Basis: {criteria.reward_basis}")
