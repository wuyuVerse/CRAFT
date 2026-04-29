"""
数据处理工具函数

提供深拷贝、验证等通用功能。
"""

import copy
import json
import re
from typing import Any, Optional, Dict


def deep_copy(obj: Any) -> Any:
    """
    深拷贝对象
    
    Args:
        obj: 要拷贝的对象
        
    Returns:
        深拷贝后的对象
    """
    return copy.deepcopy(obj)


def safe_json_loads(text: str) -> Optional[Dict]:
    """
    安全地解析JSON
    
    Args:
        text: JSON字符串
        
    Returns:
        解析后的字典，失败返回None
    """
    if not text or not isinstance(text, str):
        return None
    
    try:
        if text.startswith('{'):
            return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return None


def extract_tool_call(content: str) -> Optional[Dict]:
    """
    从assistant消息中提取工具调用
    
    Args:
        content: assistant消息内容
        
    Returns:
        工具调用字典 {"name": "...", "arguments": {...}}，失败返回None
    """
    if not content or not isinstance(content, str):
        return None
    
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        try:
            call = json.loads(match.group(1))
            # 验证必要字段
            if 'name' in call and 'arguments' in call:
                return call
        except json.JSONDecodeError:
            pass
    
    return None


def remove_think_tags(text: str) -> str:
    """
    移除LLM响应中的思考标签（如DeepSeek的<think>标签）
    
    这些标签不应该出现在最终的SFT数据中
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return text
    
    # 移除 <think>...</think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 移除 <thinking>...</thinking> 标签及其内容
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 移除 <thought>...</thought> 标签及其内容
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    
    # 清理多余的空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def validate_dict_structure(obj: Any, required_keys: list) -> bool:
    """
    验证字典结构
    
    Args:
        obj: 要验证的对象
        required_keys: 必需的键列表
        
    Returns:
        是否有效
    """
    if not isinstance(obj, dict):
        return False
    
    for key in required_keys:
        if key not in obj:
            return False
    
    return True


def corrupt_value(value: Any) -> Any:
    """
    破坏一个值（用于生成错误参数）
    
    注意：对于ID格式的字符串（如L1001, P1001），保持格式正确但修改数值，
    避免模型学习到错误的格式模式。
    
    Args:
        value: 原始值
        
    Returns:
        破坏后的值
    """
    import random
    
    if isinstance(value, str):
        if len(value) <= 1:
            return value + "_invalid"
        
        # 检查是否是ID格式 (L1001, P1001, B1001, C1001, D1001等)
        id_match = re.match(r'^([LPBCD])(\d{4,})$', value)
        if id_match:
            prefix = id_match.group(1)
            num = int(id_match.group(2))
            # 生成一个不存在的ID，保持格式正确
            wrong_num = num + random.choice([1000, 2000, 5000, 9000])
            return f"{prefix}{wrong_num}"
        
        # 检查是否是订单ID格式 (#W1234567)
        order_match = re.match(r'^(#W)(\d+)$', value)
        if order_match:
            prefix = order_match.group(1)
            num = int(order_match.group(2))
            wrong_num = num + random.choice([1000000, 2000000])
            return f"{prefix}{wrong_num}"
        
        # 检查是否是预订ID格式 (6位大写字母数字)
        if re.match(r'^[A-Z0-9]{6}$', value):
            # 修改最后两个字符
            chars = list(value)
            chars[-1] = random.choice('XYZQW')
            chars[-2] = random.choice('0123456789')
            return ''.join(chars)
        
        # 检查是否是用户ID格式 (name_name_1234)
        user_match = re.match(r'^([a-z]+_[a-z]+_)(\d+)$', value)
        if user_match:
            prefix = user_match.group(1)
            num = int(user_match.group(2))
            wrong_num = num + random.choice([1000, 5000])
            return f"{prefix}{wrong_num}"
        
        # 检查是否是电话号码格式
        if re.match(r'^[\d\-\s\+]+$', value) and len(value) >= 7:
            # 修改最后几位数字
            chars = list(value)
            for i in range(-1, -4, -1):
                if chars[i].isdigit():
                    chars[i] = str((int(chars[i]) + 1) % 10)
            return ''.join(chars)
        
        # 其他字符串：使用安全的腐蚀策略
        corruptions = [
            lambda v: v + "_invalid",  # 添加后缀
            lambda v: "invalid_" + v,  # 添加前缀
            lambda v: v.replace("_", "-") if "_" in v else v + "_x",  # 改变分隔符
            lambda v: v.upper() if v.islower() else v.lower(),  # 改变大小写
        ]
        
        corrupted = random.choice(corruptions)(value)
        if corrupted == value:
            corrupted = value + "_invalid"
        return corrupted
        
    elif isinstance(value, int):
        delta = random.choice([-100, -50, -10, 10, 50, 100])
        return value + delta
    
    elif isinstance(value, float):
        return value * random.choice([0.5, 0.8, 1.2, 1.5])
    
    elif isinstance(value, list):
        if len(value) > 1:
            return value[:-1]
        return []
    
    elif isinstance(value, dict):
        if value:
            key = random.choice(list(value.keys()))
            new_dict = copy.deepcopy(value)
            del new_dict[key]
            return new_dict
        return {}
    
    return value
