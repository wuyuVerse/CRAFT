"""
模板工具函数

提供模板填充、选择等通用功能。
"""

import random
from typing import Dict, List, Any


def fill_template(template: str, values: Dict[str, Any]) -> str:
    """
    填充模板中的占位符
    
    Args:
        template: 模板字符串，包含 {key} 格式的占位符
        values: 填充值字典
        
    Returns:
        填充后的字符串
    """
    result = template
    
    # 填充提供的值
    for key, value in values.items():
        placeholder = f"{{{key}}}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    
    # 填充常见的默认占位符
    defaults = {
        '{total}': str(random.randint(200, 1000)),
        '{paid}': str(random.randint(100, 500)),
        '{flight_number}': 'HAT' + str(random.randint(100, 999)),
        '{date}': '2024-05-15',
        '{payment_id}': 'card_' + str(random.randint(1000, 9999)),
    }
    
    for placeholder, default in defaults.items():
        if placeholder in result:
            result = result.replace(placeholder, default)
    
    return result


def select_template(templates: List[str], weights: List[float] = None) -> str:
    """
    从模板列表中随机选择一个
    
    Args:
        templates: 模板列表
        weights: 权重列表（可选）
        
    Returns:
        选中的模板
    """
    if not templates:
        return ""
    
    if weights:
        return random.choices(templates, weights=weights)[0]
    else:
        return random.choice(templates)


def format_template_safe(template: str, **kwargs) -> str:
    """
    安全地格式化模板（处理KeyError）
    
    Args:
        template: 模板字符串
        **kwargs: 格式化参数
        
    Returns:
        格式化后的字符串，如果失败返回默认消息
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        from loguru import logger
        logger.warning(f"Template formatting failed: {e}, using default message")
        return "I encountered an error. Let me correct this and try again with the proper parameters."
