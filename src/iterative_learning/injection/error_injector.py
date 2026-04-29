"""
真实错误注入器

基于错误库注入真实的工具调用错误。
"""

import json
import random
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from tau2.data_model.message import ToolCall, ToolMessage

from ..data.error_database import ErrorDatabase


@dataclass
class ErrorInjectionConfig:
    """错误注入配置"""
    enabled: bool = True
    base_rate: float = 0.5  # 基础错误率 50%（成功任务中生成错误恢复版本的概率）
    max_errors_per_task: int = 8  # 每个任务最多 8 次错误
    error_db_path: str = "eval_results/error_database.json"
    correct_trajectory_weight: int = 1  # 正确轨迹重复次数（用于增加正确样本比例）
    save_analysis_data: bool = False  # 是否保存分析数据到sft_data_analysis.jsonl


class RealErrorInjector:
    """基于真实错误库的注入器"""
    
    def __init__(self, config: ErrorInjectionConfig, domain: str):
        """
        Args:
            config: 错误注入配置
            domain: 领域名称（airline, retail, telecom）
        """
        self.config = config
        self.domain = domain
        self.error_db = ErrorDatabase(config.error_db_path)
        
        logger.info(
            f"Initialized RealErrorInjector for {domain}, "
            f"error_rate={config.base_rate}, max_errors={config.max_errors_per_task}"
        )
    
    def should_inject(self, tool_call: ToolCall, current_error_count: int) -> bool:
        """
        判断是否应该注入错误
        
        Args:
            tool_call: 工具调用
            current_error_count: 当前已注入的错误次数
            
        Returns:
            是否应该注入错误
        """
        if not self.config.enabled:
            return False
        
        if current_error_count >= self.config.max_errors_per_task:
            return False
        
        # 简单策略：固定概率
        return random.random() < self.config.base_rate
    
    def generate_error(self, tool_call: ToolCall) -> ToolMessage:
        """
        生成真实的错误响应
        
        Args:
            tool_call: 工具调用
            
        Returns:
            错误响应消息
        """
        tool_name = tool_call.name
        args = tool_call.arguments
        
        # 从错误库中获取该工具的真实错误
        errors = self.error_db.get_errors(self.domain, tool_name)
        
        if not errors:
            # 如果错误库中没有，返回通用错误
            error_msg = "Tool execution failed"
            logger.warning(
                f"No errors found in database for {self.domain}/{tool_name}, "
                f"using generic error"
            )
        else:
            # 根据出现次数加权选择错误
            weights = [err['count'] for err in errors]
            selected_error = random.choices(errors, weights=weights)[0]
            
            # 填充错误模板
            error_template = selected_error['error']
            error_msg = self._fill_error_template(error_template, args)
            
            logger.debug(
                f"Injected error for {tool_name}: {error_msg} "
                f"(template: {error_template})"
            )
        
        return ToolMessage(
            id=tool_call.id,
            content=f"Error: {error_msg}",
            requestor=tool_call.requestor,
            role="tool",
            error=True,
        )
    
    def _fill_error_template(self, template: str, args: dict) -> str:
        """
        填充错误模板中的占位符
        
        Args:
            template: 错误模板（如 "User {user_id} not found"）
            args: 工具调用参数
            
        Returns:
            填充后的错误消息
        """
        result = template
        
        # 替换占位符
        for key, value in args.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        # 如果还有未填充的占位符，尝试智能填充
        remaining_placeholders = re.findall(r'\{(\w+)\}', result)
        for placeholder in remaining_placeholders:
            # 根据占位符类型生成示例值
            example_value = self._get_example_value(placeholder, args)
            result = result.replace(f'{{{placeholder}}}', example_value)
        
        return result
    
    def _get_example_value(self, placeholder: str, args: dict) -> str:
        """
        为占位符生成示例值
        
        Args:
            placeholder: 占位符名称
            args: 工具调用参数（可能包含相关信息）
            
        Returns:
            示例值
        """
        # 尝试从参数中推断
        if placeholder in args:
            return str(args[placeholder])
        
        # 使用默认示例值
        examples = {
            'user_id': 'john_doe_123',
            'order_id': '#W1234567',
            'reservation_id': 'ABC123',
            'flight_number': 'HAT001',
            'date': '2024-05-15',
            'amount': '100',
            'phone_number': '+1234567890',
            'customer_id': 'C1001',
            'product_id': '1234567890',
        }
        
        return examples.get(placeholder, 'unknown')
