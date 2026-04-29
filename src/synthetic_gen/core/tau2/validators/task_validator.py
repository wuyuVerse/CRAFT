"""
TaskValidator

验证生成的任务格式和质量。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 质量分数 0-1


class TaskValidator:
    """
    任务验证器

    职责：
    1. 验证任务格式是否符合tau2规范
    2. 检查必填字段
    3. 验证内容质量
    4. 计算质量分数
    """

    # 必填字段
    REQUIRED_FIELDS = {
        "id": str,
        "description": dict,
        "user_scenario": dict,
        "evaluation_criteria": dict,
    }

    REQUIRED_USER_SCENARIO_FIELDS = {
        "persona": str,
        "instructions": dict,
    }

    REQUIRED_INSTRUCTIONS_FIELDS = {
        "domain": str,
        "reason_for_call": str,
        "known_info": str,
        "task_instructions": str,
    }

    # tau2-bench格式的evaluation_criteria必填字段
    REQUIRED_EVALUATION_FIELDS = {
        "actions": list,
        "nl_assertions": list,
        # "communicate_info": list,  # 可选
    }

    def __init__(
        self,
        min_reason_length: int = 30,
        min_task_instructions_length: int = 50,
        require_all_actions: bool = True,
        validate_format: bool = True,
    ):
        """
        初始化验证器

        Args:
            min_reason_length: reason_for_call最小长度
            min_task_instructions_length: task_instructions最小长度
            require_all_actions: 是否要求actions非空
            validate_format: 是否严格验证格式
        """
        self.min_reason_length = min_reason_length
        self.min_task_instructions_length = min_task_instructions_length
        self.require_all_actions = require_all_actions
        self.validate_format = validate_format

    def validate(self, task: Dict[str, Any]) -> ValidationResult:
        """
        验证单个任务

        Args:
            task: 任务字典

        Returns:
            ValidationResult对象
        """
        errors = []
        warnings = []
        score = 1.0

        # 1. 检查顶层必填字段
        for field, field_type in self.REQUIRED_FIELDS.items():
            if field not in task:
                errors.append(f"Missing required field: {field}")
                score -= 0.25
            elif not isinstance(task[field], field_type):
                errors.append(f"Field {field} should be {field_type.__name__}")
                score -= 0.1

        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                score=max(0, score)
            )

        # 2. 检查user_scenario
        user_scenario = task.get("user_scenario", {})
        for field, field_type in self.REQUIRED_USER_SCENARIO_FIELDS.items():
            if field not in user_scenario:
                errors.append(f"Missing user_scenario.{field}")
                score -= 0.1
            elif not isinstance(user_scenario[field], field_type):
                errors.append(f"user_scenario.{field} should be {field_type.__name__}")
                score -= 0.05

        # 3. 检查instructions
        instructions = user_scenario.get("instructions", {})
        for field, field_type in self.REQUIRED_INSTRUCTIONS_FIELDS.items():
            if field not in instructions:
                errors.append(f"Missing instructions.{field}")
                score -= 0.1

        # 4. 检查evaluation_criteria
        eval_criteria = task.get("evaluation_criteria", {})
        for field, field_type in self.REQUIRED_EVALUATION_FIELDS.items():
            if field not in eval_criteria:
                errors.append(f"Missing evaluation_criteria.{field}")
                score -= 0.1

        # 5. 内容质量检查
        reason = instructions.get("reason_for_call", "")
        if len(reason) < self.min_reason_length:
            warnings.append(f"reason_for_call too short ({len(reason)} < {self.min_reason_length})")
            score -= 0.05

        task_instructions = instructions.get("task_instructions", "")
        if len(task_instructions) < self.min_task_instructions_length:
            warnings.append(f"task_instructions too short ({len(task_instructions)} < {self.min_task_instructions_length})")
            score -= 0.05

        # 6. actions验证
        actions = eval_criteria.get("actions", [])
        if self.require_all_actions and not actions:
            warnings.append("actions is empty")
            score -= 0.1

        # 7. 验证每个action的格式
        if self.validate_format and actions:
            for i, action in enumerate(actions):
                if not isinstance(action, dict):
                    errors.append(f"actions[{i}] should be dict")
                    score -= 0.05
                    continue

                if "name" not in action:
                    errors.append(f"actions[{i}] missing 'name'")
                    score -= 0.05
                if "arguments" not in action:
                    warnings.append(f"actions[{i}] missing 'arguments'")
                    score -= 0.02

        # 8. nl_assertions验证
        nl_assertions = eval_criteria.get("nl_assertions", [])
        if not nl_assertions:
            warnings.append("nl_assertions is empty")
            score -= 0.05

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0, score)
        )

    def validate_batch(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量验证任务

        Args:
            tasks: 任务列表

        Returns:
            验证报告
        """
        results = []
        valid_count = 0
        total_score = 0

        for task in tasks:
            result = self.validate(task)
            results.append({
                "task_id": task.get("id", "unknown"),
                "is_valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "score": result.score,
            })
            if result.is_valid:
                valid_count += 1
            total_score += result.score

        return {
            "total_tasks": len(tasks),
            "valid_tasks": valid_count,
            "invalid_tasks": len(tasks) - valid_count,
            "average_score": total_score / len(tasks) if tasks else 0,
            "pass_rate": valid_count / len(tasks) if tasks else 0,
            "details": results,
        }


if __name__ == "__main__":
    # 测试代码
    validator = TaskValidator()

    # 测试有效任务
    valid_task = {
        "id": "test_001",
        "description": {
            "purpose": "Test task",
            "notes": "Generated for testing"
        },
        "user_scenario": {
            "persona": "Professional and direct communicator",
            "instructions": {
                "domain": "airline",
                "reason_for_call": "I need to cancel my flight reservation due to a schedule change.",
                "known_info": "Confirmation: ABC123, Name: John Doe",
                "unknown_info": "",
                "task_instructions": "Request to cancel the booking. Provide confirmation number when asked. Confirm the cancellation."
            }
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "action_1",
                    "name": "cancel_reservation",
                    "arguments": {"reservation_id": "ABC123"},
                    "compare_args": ["reservation_id"]
                }
            ],
            "nl_assertions": [
                "Agent should verify customer identity",
                "Agent should confirm cancellation"
            ],
            "reward_basis": ["DB", "ACTION"]
        }
    }

    result = validator.validate(valid_task)
    print(f"Valid task result:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Score: {result.score}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")

    # 测试无效任务
    invalid_task = {
        "id": "test_002",
        "description": {},
    }

    result = validator.validate(invalid_task)
    print(f"\nInvalid task result:")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Score: {result.score}")
    print(f"  Errors: {result.errors}")
