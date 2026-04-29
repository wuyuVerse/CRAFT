"""
基于Agent的错误注入器 (V4)

整合错误生成Agent和恢复生成Agent，在成功轨迹上注入错误和恢复。
"""

import json
import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from loguru import logger

from ...agents.error_generation_agent import ErrorGenerationAgent, ErrorType, GeneratedError
from ...agents.recovery_generation_agent import RecoveryGenerationAgent, RecoveryResponse
from ...utils import deep_copy
from ..validator import TrajectoryValidator


@dataclass
class InjectionResult:
    """注入结果"""
    success: bool
    messages: List[dict]
    injected_count: int
    error_types: List[str]


class AgentBasedErrorInjector:
    """基于Agent的错误注入器 - V4版本"""
    
    # 错误类型权重
    DEFAULT_ERROR_WEIGHTS = {
        ErrorType.PARAMETER_ERROR: 0.30,
        ErrorType.BUSINESS_LOGIC_ERROR: 0.30,
        ErrorType.STATE_ERROR: 0.20,
        ErrorType.TOOL_HALLUCINATION: 0.20,
    }
    
    def __init__(
        self,
        llm: str,
        error_db_path: str,
        domain: str,
        api_base: str = None,
        use_llm_for_recovery: bool = True,
        error_type_weights: Dict[str, float] = None,
    ):
        """
        Args:
            llm: LLM模型名称
            error_db_path: 错误数据库路径
            domain: 领域名称
            api_base: API基础URL
            use_llm_for_recovery: 是否使用LLM生成恢复
            error_type_weights: 错误类型权重
        """
        self.llm = llm
        self.domain = domain
        self.api_base = api_base
        
        # 加载错误库
        self.error_db = self._load_error_db(error_db_path)
        
        # 初始化组件
        self.error_agent = ErrorGenerationAgent(
            error_db=self.error_db,
            domain=domain,
            llm=llm,
            api_base=api_base,
            use_llm=True,  # V4默认使用LLM生成错误
        )
        
        self.recovery_agent = RecoveryGenerationAgent(
            llm=llm,
            api_base=api_base,
            use_llm=use_llm_for_recovery,
        )
        
        self.validator = TrajectoryValidator()
        
        # 错误类型权重
        if error_type_weights:
            self.error_weights = {
                ErrorType(k): v for k, v in error_type_weights.items()
            }
        else:
            self.error_weights = self.DEFAULT_ERROR_WEIGHTS
        
        logger.info(
            f"Initialized AgentBasedErrorInjector for {domain}, "
            f"use_llm_for_recovery={use_llm_for_recovery}"
        )
    
    def _load_error_db(self, path: str) -> dict:
        """加载错误数据库"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load error database: {e}")
            return {}
    
    def inject_errors(
        self,
        trajectory: dict,
        num_errors: int = 1,
        error_rate: float = 1.0,
    ) -> Optional[dict]:
        """
        在轨迹中注入错误
        
        Args:
            trajectory: 原始轨迹 {"messages": [...], "tools": "..."}
            num_errors: 要注入的错误数量
            error_rate: 每个注入点的注入概率
            
        Returns:
            注入错误后的轨迹，如果失败则返回None
        """
        # 验证输入
        if not trajectory or 'messages' not in trajectory:
            logger.error("Invalid trajectory: missing 'messages' field")
            return None
        
        messages = deep_copy(trajectory['messages'])
        tools = trajectory.get('tools', '[]')
        
        # 1. 找到所有可注入点
        injection_points = self._find_injection_points(messages)
        
        if not injection_points:
            logger.warning("No injection points found in trajectory")
            return None
        
        # 2. 选择要注入的点
        num_to_inject = min(num_errors, len(injection_points))
        if num_to_inject <= 0:
            logger.warning("num_errors must be positive")
            return None
            
        selected_points = random.sample(injection_points, num_to_inject)
        
        # 按位置倒序排列（从后往前注入，避免位置偏移）
        selected_points.sort(key=lambda p: p['position'], reverse=True)
        
        injected_count = 0
        error_types = []
        failed_attempts = 0
        max_failed_attempts = num_to_inject * 2  # 防止无限循环
        
        for point in selected_points:
            # 根据概率决定是否注入
            if random.random() > error_rate:
                continue
            
            # 选择错误类型
            error_type = self._select_error_type(point['tool_name'])
            
            try:
                # 生成错误
                generated_error = self.error_agent.generate_error(
                    correct_call={"name": point['tool_name'], "arguments": point['tool_args']},
                    tool_result=point['tool_result'],
                    context=messages[:point['position']],
                    error_type=error_type,
                )
                
                if generated_error is None:
                    logger.warning(f"Failed to generate error for {point['tool_name']}")
                    failed_attempts += 1
                    if failed_attempts >= max_failed_attempts:
                        logger.error("Too many failed attempts, stopping injection")
                        break
                    continue
                
                # 生成恢复
                recovery = self.recovery_agent.generate_recovery(
                    wrong_call=generated_error.wrong_call,
                    error_message=generated_error.error_message,
                    correct_call=generated_error.original_call,
                    context=messages[:point['position']],
                    error_type=error_type.value,
                )
                
                if recovery is None:
                    logger.warning(f"Failed to generate recovery for {point['tool_name']}")
                    failed_attempts += 1
                    if failed_attempts >= max_failed_attempts:
                        logger.error("Too many failed attempts, stopping injection")
                        break
                    continue
                
                # 注入错误和恢复
                messages = self._inject_at_point(
                    messages,
                    point,
                    generated_error,
                    recovery,
                )
                
                injected_count += 1
                error_types.append(error_type.value)
                
                logger.debug(
                    f"Injected {error_type.value} at position {point['position']}: "
                    f"{point['tool_name']}"
                )
                
            except Exception as e:
                logger.error(f"Error during injection at {point['tool_name']}: {e}")
                failed_attempts += 1
                if failed_attempts >= max_failed_attempts:
                    break
                continue
        
        if injected_count == 0:
            logger.warning("No errors were injected")
            return None
        
        # 验证最终轨迹
        valid, error = self.validator.validate_trajectory(messages)
        if not valid:
            logger.error(f"Generated trajectory is invalid: {error}")
            return None
        
        logger.info(
            f"Successfully injected {injected_count} errors: {error_types}"
        )
        
        return {
            "messages": messages,
            "tools": tools,
            "injected_errors": injected_count,
            "error_types": error_types,
        }
    
    def _find_injection_points(self, messages: List[dict]) -> List[dict]:
        """找到所有可注入错误的点"""
        points = []
        
        for i, msg in enumerate(messages):
            if msg.get('role') == 'assistant' and msg.get('content'):
                content = msg['content']
                
                # 检查是否包含工具调用
                if '<tool_call>' in content:
                    tool_call = self.validator.extract_tool_call(content)
                    
                    if tool_call and isinstance(tool_call.get('arguments'), dict):
                        # 检查下一条消息是否是工具响应
                        if i + 1 < len(messages) and messages[i + 1].get('role') == 'tool':
                            tool_response = messages[i + 1].get('content', '')
                            tool_result = self._parse_tool_response(tool_response)
                            
                            # 只选择成功的工具调用
                            if tool_result and not self._is_error_response(tool_result):
                                points.append({
                                    'position': i,
                                    'tool_name': tool_call['name'],
                                    'tool_args': tool_call['arguments'],
                                    'tool_result': tool_result,
                                })
        
        return points
    
    def _parse_tool_response(self, content: str) -> Optional[dict]:
        """解析工具响应"""
        if not content or not isinstance(content, str):
            return None
        try:
            if content.startswith('{'):
                result = json.loads(content)
                # 验证必要字段
                if isinstance(result, dict) and 'name' in result:
                    return result
        except json.JSONDecodeError:
            pass
        return None
    
    def _is_error_response(self, result: dict) -> bool:
        """检查是否是错误响应"""
        if isinstance(result.get('result'), str):
            return result['result'].startswith('Error:')
        return False
    
    def _select_error_type(self, tool_name: str) -> ErrorType:
        """根据权重选择错误类型"""
        types = list(self.error_weights.keys())
        weights = list(self.error_weights.values())
        return random.choices(types, weights=weights)[0]
    
    def _inject_at_point(
        self,
        messages: List[dict],
        point: dict,
        error: GeneratedError,
        recovery: RecoveryResponse,
    ) -> List[dict]:
        """在指定点注入错误和恢复"""
        result = []
        pos = point['position']
        
        # 复制注入点之前的消息
        for i in range(pos):
            result.append(deep_copy(messages[i]))
        
        # 1. 插入错误的工具调用（assistant消息）
        wrong_call_content = self.validator.format_tool_call(
            error.wrong_call['name'],
            error.wrong_call['arguments']
        )
        result.append({
            "role": "assistant",
            "content": wrong_call_content
        })
        
        # 2. 插入错误响应（tool消息）
        error_response = self.validator.format_tool_response(
            error.wrong_call['name'],
            error.error_message
        )
        result.append({
            "role": "tool",
            "content": error_response
        })
        
        # 3. 插入恢复思考和正确调用（assistant消息）
        result.append({
            "role": "assistant",
            "content": recovery.full_content
        })
        
        # 4. 复制原始的工具响应
        if pos + 1 < len(messages):
            result.append(deep_copy(messages[pos + 1]))
        
        # 5. 复制剩余的消息
        for i in range(pos + 2, len(messages)):
            result.append(deep_copy(messages[i]))
        
        return result
