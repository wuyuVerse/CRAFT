"""
轨迹错误注入器

直接在成功轨迹上插入错误和恢复步骤，而不是重新执行任务。
这样可以保证：
1. 最终一定成功（因为我们知道正确答案）
2. 格式一致（前后轨迹保持不变）
3. 真实错误（从error_database中选择）

核心思路：
1. 选择一个工具调用点
2. 生成一个错误的工具调用（参数错误或调用错误工具）
3. 插入系统返回的错误信息
4. 让模型生成恢复的思考和正确的调用
5. 拼接回原始轨迹
"""

import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from loguru import logger


@dataclass
class ErrorInjectionPoint:
    """错误注入点"""
    position: int  # 在messages中的位置
    tool_name: str  # 工具名称
    tool_args: dict  # 工具参数
    tool_result: dict  # 工具返回结果
    

@dataclass 
class InjectedError:
    """注入的错误"""
    error_type: str  # 错误类型：parameter_error, wrong_tool, missing_field
    error_message: str  # 错误消息
    wrong_call: dict  # 错误的工具调用
    recovery_thought: str  # 恢复思考
    correct_call: dict  # 正确的工具调用


class RuleBasedErrorInjector:
    """基于规则的错误注入器 - V3版本"""
    
    def __init__(self, error_db_path: str, domain: str):
        """
        Args:
            error_db_path: 错误数据库路径
            domain: 领域名称
        """
        self.domain = domain
        self.error_db = self._load_error_db(error_db_path)
        
        # 可恢复的错误类型（避免"资源不存在"这类不可恢复的错误）
        self.recoverable_error_patterns = [
            # 参数格式错误
            r"missing.*required.*argument",
            r"unexpected keyword argument",
            r"validation error",
            r"invalid.*format",
            r"should be",
            # 业务逻辑错误（可以通过修改参数恢复）
            r"payment.*does not add up",
            r"number.*does not match",
            r"should match",
            r"balance.*not enough",
            r"insufficient",
        ]
        
        # 不可恢复的错误类型（避免使用）
        self.unrecoverable_error_patterns = [
            r"not found",
            r"does not exist", 
            r"Tool.*not found",
            r"cannot be",
        ]
        
        logger.info(f"Initialized TrajectoryErrorInjector for {domain}")
    
    def _load_error_db(self, path: str) -> dict:
        """加载错误数据库"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load error database: {e}")
            return {}
    
    def find_injection_points(self, messages: List[dict]) -> List[ErrorInjectionPoint]:
        """
        找到所有可以注入错误的点
        
        Args:
            messages: 消息列表
            
        Returns:
            可注入错误的位置列表
        """
        points = []
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'assistant' and msg.get('content'):
                # 检查是否包含工具调用
                content = msg['content']
                if '<tool_call>' in content:
                    # 提取工具调用
                    tool_call = self._extract_tool_call(content)
                    if tool_call:
                        # 检查下一条消息是否是工具响应
                        if i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                            tool_response = messages[i + 1].get('content', '')
                            tool_result = self._parse_tool_response(tool_response)
                            
                            # 只选择成功的工具调用（我们要在成功调用前插入错误）
                            if tool_result and not self._is_error_response(tool_result):
                                points.append(ErrorInjectionPoint(
                                    position=i,
                                    tool_name=tool_call['name'],
                                    tool_args=tool_call['arguments'],
                                    tool_result=tool_result
                                ))
        
        return points
    
    def _extract_tool_call(self, content: str) -> Optional[dict]:
        """从assistant消息中提取工具调用"""
        try:
            pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception as e:
            logger.debug(f"Failed to extract tool call: {e}")
        return None
    
    def _parse_tool_response(self, content: str) -> Optional[dict]:
        """解析工具响应"""
        try:
            if content.startswith('{'):
                return json.loads(content)
        except:
            pass
        return None
    
    def _is_error_response(self, result: dict) -> bool:
        """检查是否是错误响应"""
        if isinstance(result.get('result'), str):
            return result['result'].startswith('Error:')
        return False
    
    def generate_error_for_tool(self, tool_name: str, tool_args: dict) -> Optional[InjectedError]:
        """
        为指定工具生成一个可恢复的错误
        
        Args:
            tool_name: 工具名称
            tool_args: 正确的工具参数
            
        Returns:
            注入的错误，如果无法生成则返回None
        """
        # 获取该工具的错误列表
        domain_errors = self.error_db.get(self.domain, {})
        tool_errors = domain_errors.get(tool_name, [])
        
        # 过滤出可恢复的错误
        recoverable_errors = []
        for err in tool_errors:
            error_msg = err['error']
            if self._is_recoverable_error(error_msg):
                recoverable_errors.append(err)
        
        if not recoverable_errors:
            # 如果没有可恢复的错误，生成一个参数错误
            return self._generate_parameter_error(tool_name, tool_args)
        
        # 加权选择一个错误
        weights = [err['count'] for err in recoverable_errors]
        selected_error = random.choices(recoverable_errors, weights=weights)[0]
        
        # 生成错误调用和恢复
        return self._create_injected_error(
            tool_name, 
            tool_args, 
            selected_error['error']
        )
    
    def _is_recoverable_error(self, error_msg: str) -> bool:
        """检查错误是否可恢复"""
        error_lower = error_msg.lower()
        
        # 检查是否匹配不可恢复模式
        for pattern in self.unrecoverable_error_patterns:
            if re.search(pattern, error_lower):
                return False
        
        # 检查是否匹配可恢复模式
        for pattern in self.recoverable_error_patterns:
            if re.search(pattern, error_lower):
                return True
        
        # 默认不使用
        return False
    
    def _generate_parameter_error(self, tool_name: str, tool_args: dict) -> InjectedError:
        """生成一个参数错误"""
        # 选择一个参数进行修改
        if not tool_args:
            return None
            
        param_name = random.choice(list(tool_args.keys()))
        original_value = tool_args[param_name]
        
        # 生成错误的参数值
        wrong_value = self._corrupt_value(original_value)
        wrong_args = copy.deepcopy(tool_args)
        wrong_args[param_name] = wrong_value
        
        # 生成错误消息（更真实的格式）
        error_templates = [
            f"Invalid {param_name}: '{wrong_value}' is not a valid format",
            f"Parameter '{param_name}' validation failed: expected valid identifier, got '{wrong_value}'",
            f"Error processing {param_name}: '{wrong_value}' does not match expected pattern",
        ]
        error_message = random.choice(error_templates)
        
        # 生成恢复思考（更自然）
        recovery_templates = [
            f"I made an error with the {param_name} parameter. The correct value should be '{original_value}'. Let me try again with the correct information.",
            f"I see there was an issue with the {param_name} I provided. Let me correct it to '{original_value}' and retry the request.",
            f"The {param_name} '{wrong_value}' was incorrect. I need to use '{original_value}' instead. Let me make the correction.",
        ]
        recovery_thought = random.choice(recovery_templates)
        
        return InjectedError(
            error_type="parameter_error",
            error_message=error_message,
            wrong_call={"name": tool_name, "arguments": wrong_args},
            recovery_thought=recovery_thought,
            correct_call={"name": tool_name, "arguments": tool_args}
        )
    
    def _corrupt_value(self, value) -> any:
        """破坏一个值"""
        if isinstance(value, str):
            # 字符串：添加typo、截断、或改变格式
            if len(value) <= 1:
                return value + "_typo"
            
            corruptions = [
                lambda v: v[:-1],  # 截断最后一个字符
                lambda v: v[:-2] if len(v) > 2 else v[:-1],  # 截断更多
                lambda v: v.replace("_", "-") if "_" in v else v + "_typo",  # 改变分隔符
                lambda v: v[:len(v)//2],  # 截断一半
                lambda v: "invalid_" + v[:5],  # 添加前缀并截断
            ]
            
            # 随机选择一个破坏方式
            corrupted = random.choice(corruptions)(value)
            
            # 确保破坏后的值和原值不同
            if corrupted == value:
                corrupted = value[:-1] if len(value) > 1 else value + "_x"
            
            return corrupted
        elif isinstance(value, int):
            # 整数：改变值（确保不同）
            delta = random.choice([-100, -50, -10, 10, 50, 100])
            return value + delta
        elif isinstance(value, float):
            # 浮点数：改变值
            return value * random.choice([0.5, 0.8, 1.2, 1.5])
        elif isinstance(value, list):
            # 列表：移除一个元素
            if len(value) > 1:
                return value[:-1]
            return []
        elif isinstance(value, dict):
            # 字典：移除一个键
            if value:
                key = random.choice(list(value.keys()))
                new_dict = copy.deepcopy(value)
                del new_dict[key]
                return new_dict
            return {}
        return value
    
    def _create_injected_error(
        self, 
        tool_name: str, 
        tool_args: dict, 
        error_template: str
    ) -> InjectedError:
        """创建注入的错误"""
        # 填充错误模板
        error_message = self._fill_error_template(error_template, tool_args)
        
        # 根据错误类型生成错误调用
        wrong_args = self._generate_wrong_args(tool_args, error_template)
        
        # 生成恢复思考
        recovery_thought = self._generate_recovery_thought(error_message, tool_name, tool_args)
        
        return InjectedError(
            error_type="real_error",
            error_message=error_message,
            wrong_call={"name": tool_name, "arguments": wrong_args},
            recovery_thought=recovery_thought,
            correct_call={"name": tool_name, "arguments": tool_args}
        )
    
    def _fill_error_template(self, template: str, args: dict) -> str:
        """填充错误模板"""
        result = template
        for key, value in args.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result
    
    def _generate_wrong_args(self, correct_args: dict, error_template: str) -> dict:
        """根据错误模板生成错误的参数"""
        wrong_args = copy.deepcopy(correct_args)
        
        # 根据错误类型修改参数
        if "payment" in error_template.lower() and "amount" in error_template.lower():
            # 支付金额错误
            for key in wrong_args:
                if "amount" in key.lower() or "payment" in key.lower():
                    if isinstance(wrong_args[key], (int, float)):
                        wrong_args[key] = int(wrong_args[key] * 0.8)  # 减少20%
                    elif isinstance(wrong_args[key], list):
                        # 修改支付列表中的金额
                        for item in wrong_args[key]:
                            if isinstance(item, dict) and 'amount' in item:
                                item['amount'] = int(item['amount'] * 0.8)
        elif "missing" in error_template.lower():
            # 缺少参数
            if wrong_args:
                key_to_remove = random.choice(list(wrong_args.keys()))
                del wrong_args[key_to_remove]
        elif "number" in error_template.lower() and "match" in error_template.lower():
            # 数量不匹配
            for key in wrong_args:
                if isinstance(wrong_args[key], list):
                    if len(wrong_args[key]) > 1:
                        wrong_args[key] = wrong_args[key][:-1]
        else:
            # 默认：修改一个参数
            if wrong_args:
                key = random.choice(list(wrong_args.keys()))
                wrong_args[key] = self._corrupt_value(wrong_args[key])
        
        return wrong_args
    
    def _generate_recovery_thought(
        self, 
        error_message: str, 
        tool_name: str, 
        correct_args: dict
    ) -> str:
        """生成恢复思考"""
        # 根据错误类型生成不同的恢复思考
        error_lower = error_message.lower()
        
        if "payment" in error_lower and "amount" in error_lower:
            return (
                f"I see the payment amount doesn't match the total price. "
                f"Let me recalculate and provide the correct payment amount."
            )
        elif "missing" in error_lower:
            return (
                f"I'm missing a required parameter. "
                f"Let me include all the necessary information and try again."
            )
        elif "number" in error_lower and "match" in error_lower:
            return (
                f"The number of items doesn't match. "
                f"Let me verify the correct count and retry."
            )
        elif "validation" in error_lower or "invalid" in error_lower:
            return (
                f"There's a validation error with my input. "
                f"Let me check the format and correct it."
            )
        elif "balance" in error_lower or "insufficient" in error_lower:
            return (
                f"There seems to be an issue with the balance. "
                f"Let me verify the payment method and amount."
            )
        else:
            return (
                f"I encountered an error: {error_message}. "
                f"Let me analyze the issue and try again with the correct parameters."
            )
    
    def inject_error_into_trajectory(
        self,
        messages: List[dict],
        injection_point: ErrorInjectionPoint,
        injected_error: InjectedError
    ) -> List[dict]:
        """
        在轨迹中注入错误
        
        Args:
            messages: 原始消息列表
            injection_point: 注入点
            injected_error: 要注入的错误
            
        Returns:
            注入错误后的消息列表
        """
        result = []
        pos = injection_point.position
        
        # 复制注入点之前的消息（深拷贝）
        for i in range(pos):
            result.append(copy.deepcopy(messages[i]))
        
        # 1. 插入错误的工具调用（assistant消息）
        wrong_call_content = (
            f"\n<tool_call>\n"
            f'{json.dumps(injected_error.wrong_call, ensure_ascii=False)}\n'
            f"</tool_call>"
        )
        result.append({
            "role": "assistant",
            "content": wrong_call_content
        })
        
        # 2. 插入错误响应（tool消息）
        error_response = {
            "name": injected_error.wrong_call["name"],
            "result": f"Error: {injected_error.error_message}"
        }
        result.append({
            "role": "tool",
            "content": json.dumps(error_response, ensure_ascii=False)
        })
        
        # 3. 插入恢复思考和正确调用（assistant消息）
        correct_call_content = (
            f"{injected_error.recovery_thought}\n\n"
            f"<tool_call>\n"
            f'{json.dumps(injected_error.correct_call, ensure_ascii=False)}\n'
            f"</tool_call>"
        )
        result.append({
            "role": "assistant", 
            "content": correct_call_content
        })
        
        # 4. 复制原始的工具响应（跳过原始的assistant调用，保留tool响应）
        # 注意：pos是assistant消息的位置，pos+1是对应的tool响应
        if pos + 1 < len(messages):
            result.append(copy.deepcopy(messages[pos + 1]))
        
        # 5. 复制剩余的消息（深拷贝）
        for i in range(pos + 2, len(messages)):
            result.append(copy.deepcopy(messages[i]))
        
        return result
    
    def inject_errors(
        self,
        trajectory: dict,
        num_errors: int = 1,
        error_rate: float = 0.5
    ) -> Optional[dict]:
        """
        在轨迹中注入多个错误
        
        Args:
            trajectory: 原始轨迹 {"messages": [...], "tools": "..."}
            num_errors: 要注入的错误数量
            error_rate: 每个注入点的注入概率
            
        Returns:
            注入错误后的轨迹，如果无法注入则返回None
        """
        import copy
        messages = copy.deepcopy(trajectory['messages'])  # 深拷贝避免修改原始数据
        tools = trajectory.get('tools', '[]')
        
        # 找到所有可注入点
        injection_points = self.find_injection_points(messages)
        
        if not injection_points:
            logger.warning("No injection points found in trajectory")
            return None
        
        # 随机选择要注入的点
        num_to_inject = min(num_errors, len(injection_points))
        selected_points = random.sample(injection_points, num_to_inject)
        
        # 按位置倒序排列（从后往前注入，避免位置偏移）
        selected_points.sort(key=lambda p: p.position, reverse=True)
        
        injected_count = 0
        for point in selected_points:
            # 根据概率决定是否注入
            if random.random() > error_rate:
                continue
            
            # 生成错误
            error = self.generate_error_for_tool(point.tool_name, point.tool_args)
            if error is None:
                continue
            
            # 注入错误
            messages = self.inject_error_into_trajectory(messages, point, error)
            injected_count += 1
            
            logger.debug(
                f"Injected error at position {point.position}: "
                f"{point.tool_name} -> {error.error_type}"
            )
        
        if injected_count == 0:
            logger.warning("No errors were injected")
            return None
        
        logger.info(f"Injected {injected_count} errors into trajectory")
        
        return {
            "messages": messages,
            "tools": tools,
            "injected_errors": injected_count
        }
