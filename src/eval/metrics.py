"""
扩展评估指标

除了基本的任务成功率（Task Success Rate）外，增加4个细粒度指标：
1. 工具调用准确率 (Tool Call Accuracy)
2. 参数准确率 (Parameter Accuracy)
3. 对话效率 (Conversation Efficiency)
4. 错误恢复率 (Error Recovery Rate)
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExtendedMetrics:
    """扩展评估指标"""
    # 基础指标
    task_success_rate: float = 0.0      # 任务成功率
    avg_reward: float = 0.0             # 平均奖励

    # 新增指标
    tool_call_accuracy: float = 0.0     # 工具调用准确率: 正确调用的工具数 / 期望的工具数
    parameter_accuracy: float = 0.0     # 参数准确率: 参数正确的调用数 / 总调用数
    conversation_efficiency: float = 0.0 # 对话效率: 期望步数 / 实际步数 (越高越好)
    error_recovery_rate: float = 0.0    # 错误恢复率: 工具返回错误后成功恢复的比例

    # 统计数据
    total_tasks: int = 0
    successful_tasks: int = 0
    total_tool_calls: int = 0
    correct_tool_calls: int = 0
    total_expected_actions: int = 0
    matched_actions: int = 0
    total_errors: int = 0
    recovered_errors: int = 0
    avg_turns: float = 0.0
    avg_expected_turns: float = 0.0


def analyze_simulation(sim: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析单个simulation，提取详细指标

    Args:
        sim: simulation结果
        task: 对应的task定义

    Returns:
        包含各项指标的字典
    """
    messages = sim.get("messages", [])
    reward_info = sim.get("reward_info", {})

    # 基础信息
    reward = reward_info.get("reward", 0) or 0
    success = reward > 0

    # 1. 从 messages 中提取实际的工具调用
    tool_calls = []
    tool_results = []

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append({
                    "name": tc.get("name"),
                    "arguments": tc.get("arguments", {}),
                    "index": len(tool_calls)
                })
        elif msg.get("role") == "tool":
            content = msg.get("content", "")
            is_error = _is_error_response(content)
            tool_results.append({
                "content": content[:200],
                "is_error": is_error,
                "index": len(tool_results)
            })

    total_tool_calls = len(tool_calls)

    # 2. 获取 action_checks（tau2的评估结果，用于参考）
    action_checks = reward_info.get("action_checks", []) or []

    # 3. 获取期望的actions
    # 只统计 agent 的 actions（requestor 为 "agent" 或 None/缺失）
    # 排除 requestor 为 "user" 的 actions（这些是用户侧操作，如 telecom 的 toggle_airplane_mode）
    all_expected_actions = task.get("evaluation_criteria", {}).get("actions", []) or []
    expected_actions = [
        a for a in all_expected_actions
        if a.get("requestor") != "user"
    ]
    total_expected_actions = len(expected_actions)

    # 4. 实现匹配逻辑
    tool_matches = 0  # 工具名称匹配数
    param_partial_scores = []  # 参数部分匹配分数

    for expected in expected_actions:
        expected_name = expected.get("name", "")
        expected_args = expected.get("arguments", {})

        # 在实际调用中查找匹配的工具
        best_tool_match = False
        best_param_score = 0.0

        for actual in tool_calls:
            actual_name = actual.get("name", "")
            actual_args = actual.get("arguments", {})

            # 工具名称匹配
            if actual_name == expected_name:
                best_tool_match = True

                # 计算参数匹配分数
                param_score = _compute_param_match_score(expected_args, actual_args)
                best_param_score = max(best_param_score, param_score)

        if best_tool_match:
            tool_matches += 1
        param_partial_scores.append(best_param_score)

    # 计算工具调用准确率
    if total_expected_actions > 0:
        tool_call_accuracy = tool_matches / total_expected_actions
    else:
        # 没有期望的actions时，标记为N/A（用-1表示）
        tool_call_accuracy = -1.0

    # 计算参数准确率（使用部分匹配分数的平均值）
    if param_partial_scores and total_expected_actions > 0:
        parameter_accuracy = sum(param_partial_scores) / len(param_partial_scores)
    else:
        # 没有期望的actions时，标记为N/A
        parameter_accuracy = -1.0

    # 更新匹配数用于统计
    matched_actions = tool_matches

    # 4. 对话效率
    # 期望轮数来源：
    # 1. 如果task中有expected_turns字段，直接使用
    # 2. 否则基于期望的工具调用数估算（每个工具调用需要约2轮：用户请求+助手响应）
    expected_turns_from_task = task.get("expected_turns", 0)
    if expected_turns_from_task > 0:
        expected_turns = expected_turns_from_task
    else:
        # 基于expected_actions估算：每个action约2轮对话
        expected_turns = max(total_expected_actions * 2, 2)

    actual_turns = len([m for m in messages if m.get("role") in ["assistant", "user"]])

    if actual_turns > 0:
        conversation_efficiency = min(expected_turns / actual_turns, 1.0)
    else:
        conversation_efficiency = 0.0

    # 5. 错误恢复率
    total_errors = sum(1 for tr in tool_results if tr.get("is_error"))
    recovered_errors = 0

    # 检查错误后是否恢复
    for i, tr in enumerate(tool_results):
        if tr.get("is_error"):
            # 检查后续是否有成功的相关调用
            if i + 1 < len(tool_results):
                # 简化：如果最终任务成功，认为错误已恢复
                if success:
                    recovered_errors += 1

    if total_errors > 0:
        error_recovery_rate = recovered_errors / total_errors
    else:
        error_recovery_rate = 1.0  # 无错误视为完美恢复

    return {
        "task_id": sim.get("task_id"),
        "success": success,
        "reward": reward,
        "tool_call_accuracy": tool_call_accuracy,
        "parameter_accuracy": parameter_accuracy,
        "conversation_efficiency": conversation_efficiency,
        "error_recovery_rate": error_recovery_rate,
        "total_tool_calls": total_tool_calls,
        "total_expected_actions": total_expected_actions,
        "matched_actions": matched_actions,
        "actual_turns": actual_turns,
        "total_errors": total_errors,
        "recovered_errors": recovered_errors,
    }


def _is_error_response(content: str) -> bool:
    """判断工具返回是否为错误"""
    if not content:
        return False
    content_lower = content.lower()
    error_indicators = [
        "error",
        "not found",
        "invalid",
        "failed",
        "does not exist",
        "no such",
        "cannot",
        "unable to",
        "exception",
    ]
    return any(indicator in content_lower for indicator in error_indicators)


def _compute_param_match_score(expected_args: Dict, actual_args: Dict) -> float:
    """
    计算参数匹配分数

    Args:
        expected_args: 期望的参数
        actual_args: 实际的参数

    Returns:
        0.0-1.0 的匹配分数
    """
    if not expected_args:
        return 1.0

    # 将actual_args转为dict（可能是JSON字符串）
    if isinstance(actual_args, str):
        try:
            import json
            actual_args = json.loads(actual_args)
        except:
            actual_args = {}

    if not isinstance(actual_args, dict):
        return 0.0

    total_params = len(expected_args)
    matched_params = 0

    for key, expected_value in expected_args.items():
        if key in actual_args:
            actual_value = actual_args[key]

            # 比较值（支持多种类型）
            if _values_match(expected_value, actual_value):
                matched_params += 1

    return matched_params / total_params if total_params > 0 else 1.0


def _values_match(expected, actual) -> bool:
    """
    比较两个值是否匹配（支持模糊匹配）
    """
    # 完全相等
    if expected == actual:
        return True

    # 字符串比较（忽略大小写和空格）
    if isinstance(expected, str) and isinstance(actual, str):
        exp_normalized = expected.strip().lower()
        act_normalized = actual.strip().lower()
        if exp_normalized == act_normalized:
            return True
        # 检查是否包含（用于ID匹配等）
        if exp_normalized in act_normalized or act_normalized in exp_normalized:
            return True

    # 数字比较
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 0.001

    # 列表比较（顺序无关）
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) == len(actual):
            # 转为字符串比较
            exp_set = set(str(x).lower() for x in expected)
            act_set = set(str(x).lower() for x in actual)
            return exp_set == act_set

    # 尝试字符串化比较
    return str(expected).strip().lower() == str(actual).strip().lower()


def compute_extended_metrics(
    result_file: str,
    tasks: Optional[List[Dict]] = None
) -> ExtendedMetrics:
    """
    从评测结果文件计算扩展指标

    Args:
        result_file: tau2评测结果JSON文件路径
        tasks: 可选的任务列表（如果result_file中没有tasks）

    Returns:
        ExtendedMetrics对象
    """
    with open(result_file, 'r') as f:
        data = json.load(f)

    simulations = data.get("simulations", [])

    # 如果没有传入tasks，尝试从data中获取
    if tasks is None:
        tasks = data.get("tasks", [])

    # 构建task_id到task的映射
    task_map = {}
    for i, task in enumerate(tasks):
        task_id = task.get("id", str(i))
        task_map[task_id] = task
        task_map[str(i)] = task  # 也用索引作为key

    # 分析每个simulation
    all_metrics = []
    for sim in simulations:
        task_id = sim.get("task_id")
        task = task_map.get(str(task_id), {})
        metrics = analyze_simulation(sim, task)
        all_metrics.append(metrics)

    # 聚合指标
    if not all_metrics:
        return ExtendedMetrics()

    total_tasks = len(all_metrics)
    successful_tasks = sum(1 for m in all_metrics if m["success"])

    # 平均各项指标（忽略-1值，表示N/A）
    valid_tool_call = [m["tool_call_accuracy"] for m in all_metrics if m["tool_call_accuracy"] >= 0]
    valid_param = [m["parameter_accuracy"] for m in all_metrics if m["parameter_accuracy"] >= 0]

    avg_tool_call_accuracy = sum(valid_tool_call) / len(valid_tool_call) if valid_tool_call else -1.0
    avg_parameter_accuracy = sum(valid_param) / len(valid_param) if valid_param else -1.0
    avg_conversation_efficiency = sum(m["conversation_efficiency"] for m in all_metrics) / total_tasks
    avg_error_recovery_rate = sum(m["error_recovery_rate"] for m in all_metrics) / total_tasks
    avg_reward = sum(m["reward"] for m in all_metrics) / total_tasks

    # 统计总数
    total_tool_calls = sum(m["total_tool_calls"] for m in all_metrics)
    total_expected_actions = sum(m["total_expected_actions"] for m in all_metrics)
    matched_actions = sum(m["matched_actions"] for m in all_metrics)
    total_errors = sum(m["total_errors"] for m in all_metrics)
    recovered_errors = sum(m["recovered_errors"] for m in all_metrics)
    avg_turns = sum(m["actual_turns"] for m in all_metrics) / total_tasks

    # 处理N/A情况（-1转为0或特殊显示）
    tool_call_pct = avg_tool_call_accuracy * 100 if avg_tool_call_accuracy >= 0 else -1.0
    param_pct = avg_parameter_accuracy * 100 if avg_parameter_accuracy >= 0 else -1.0

    return ExtendedMetrics(
        task_success_rate=successful_tasks / total_tasks * 100,
        avg_reward=avg_reward,
        tool_call_accuracy=tool_call_pct,
        parameter_accuracy=param_pct,
        conversation_efficiency=avg_conversation_efficiency * 100,
        error_recovery_rate=avg_error_recovery_rate * 100,
        total_tasks=total_tasks,
        successful_tasks=successful_tasks,
        total_tool_calls=total_tool_calls,
        correct_tool_calls=matched_actions,
        total_expected_actions=total_expected_actions,
        matched_actions=matched_actions,
        total_errors=total_errors,
        recovered_errors=recovered_errors,
        avg_turns=avg_turns,
        avg_expected_turns=total_expected_actions / total_tasks * 2 if total_tasks > 0 else 0,
    )


def format_metrics_report(metrics: ExtendedMetrics) -> str:
    """格式化指标报告"""
    lines = [
        "=" * 60,
        "扩展评估指标报告",
        "=" * 60,
        "",
        "基础指标:",
        f"  任务成功率 (Task Success Rate):     {metrics.task_success_rate:.2f}%",
        f"  平均奖励 (Average Reward):          {metrics.avg_reward:.4f}",
        "",
        "细粒度指标:",
        f"  工具调用准确率 (Tool Call Accuracy): {metrics.tool_call_accuracy:.2f}%",
        f"  参数准确率 (Parameter Accuracy):     {metrics.parameter_accuracy:.2f}%",
        f"  对话效率 (Conversation Efficiency):  {metrics.conversation_efficiency:.2f}%",
        f"  错误恢复率 (Error Recovery Rate):    {metrics.error_recovery_rate:.2f}%",
        "",
        "统计数据:",
        f"  总任务数: {metrics.total_tasks}",
        f"  成功任务数: {metrics.successful_tasks}",
        f"  总工具调用数: {metrics.total_tool_calls}",
        f"  期望Action数: {metrics.total_expected_actions}",
        f"  匹配Action数: {metrics.matched_actions}",
        f"  总错误数: {metrics.total_errors}",
        f"  恢复错误数: {metrics.recovered_errors}",
        f"  平均对话轮数: {metrics.avg_turns:.1f}",
        "=" * 60,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
        metrics = compute_extended_metrics(result_file)
        print(format_metrics_report(metrics))
