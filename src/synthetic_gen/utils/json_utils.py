"""
JSON处理工具函数
"""
import json
import re
from typing import Optional


def clean_json_response(content: Optional[str]) -> str:
    """
    清理LLM返回的JSON内容
    
    处理以下情况：
    1. Markdown代码块格式（```json ... ```）
    2. 额外的空白字符
    3. None值
    
    Args:
        content: LLM返回的原始内容
    
    Returns:
        清理后的JSON字符串
    
    Raises:
        ValueError: 如果内容为None或空
    """
    if content is None:
        raise ValueError("Response content is None")
    
    # 去除首尾空白
    content = content.strip()
    
    if not content:
        raise ValueError("Response content is empty")
    
    # 移除markdown代码块标记
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    
    if content.endswith("```"):
        content = content[:-3]
    
    # 再次去除空白
    content = content.strip()
    
    return content


def safe_json_loads(content: Optional[str], default: dict = None) -> dict:
    """
    安全地解析JSON，带有错误处理
    
    Args:
        content: 要解析的JSON字符串
        default: 解析失败时返回的默认值
    
    Returns:
        解析后的字典，或default值
    
    Raises:
        ValueError: 如果解析失败且没有提供default值
    """
    try:
        cleaned = clean_json_response(content)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        if default is not None:
            return default
        raise ValueError(f"Failed to parse JSON: {e}")


def extract_json_from_text(text: str) -> Optional[str]:
    """
    从文本中提取JSON内容
    
    尝试找到文本中的JSON对象或数组
    
    Args:
        text: 包含JSON的文本
    
    Returns:
        提取出的JSON字符串，如果找不到则返回None
    """
    text = text.strip()
    
    # 尝试找到JSON对象 {...}
    json_obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_obj_match:
        return json_obj_match.group(0)
    
    # 尝试找到JSON数组 [...]
    json_arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_arr_match:
        return json_arr_match.group(0)
    
    return None

