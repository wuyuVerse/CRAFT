"""
干净轨迹提取器

从包含分析的轨迹中提取干净的训练数据。
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from loguru import logger

from tau2.data_model.message import AssistantMessage, Message, ToolMessage
from tau2.data_model.simulation import SimulationRun

from .enhanced_formatter import EnhancedDataFormatter


class CleanTrajectoryExtractor:
    """干净轨迹提取器 - 移除分析内容，保留错误恢复"""
    
    def __init__(self, domain: str):
        """
        Args:
            domain: 领域名称
        """
        self.domain = domain
        self.formatter = EnhancedDataFormatter(domain)
    
    def extract_clean_trajectory(
        self,
        simulation: SimulationRun,
        system_prompt: str,
        tools: List[dict],
    ) -> Optional[dict]:
        """
        提取干净的训练轨迹
        
        规则：
        1. 移除所有分析内容（error_analysis, success_analysis, contrast_analysis）
        2. 保留完整的对话流程（包括错误恢复）
        3. 确保格式符合 tau2-bench 标准
        4. 验证工具调用格式
        
        Args:
            simulation: 模拟运行结果
            system_prompt: 系统提示词
            tools: 工具定义列表
            
        Returns:
            干净的 SFT 数据，如果验证失败则返回 None
        """
        messages = simulation.messages
        
        # 1. 移除分析内容
        clean_messages = self._remove_analysis_content(messages)
        
        # 2. 获取干净的系统提示词
        clean_system_prompt = self._get_clean_system_prompt(system_prompt)
        
        # 3. 格式化为 SFT 格式
        formatted = self.formatter._format_messages(
            clean_messages, 
            clean_system_prompt,
            tools
        )
        
        # 4. 验证格式
        if not self._validate_format(formatted, tools):
            logger.warning(f"轨迹格式验证失败: {simulation.task_id}")
            return None
        
        # 5. 验证错误恢复序列
        if not self._validate_error_recovery(formatted):
            logger.warning(f"错误恢复序列验证失败: {simulation.task_id}")
            return None
        
        return {
            'messages': formatted,
            'tools': json.dumps(tools)
        }
    
    def _remove_analysis_content(self, messages: List[Message]) -> List[Message]:
        """
        移除分析内容
        
        Args:
            messages: 原始消息列表
            
        Returns:
            清理后的消息列表
        """
        clean = []
        
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                # 检查是否包含分析标记
                if msg.content and self._contains_analysis_markers(msg.content):
                    # 提取分析前的内容
                    clean_content = self._extract_pre_analysis_content(msg.content)
                    if clean_content:
                        # 创建新的消息对象，保留 tool_calls
                        clean_msg = AssistantMessage(
                            content=clean_content,
                            tool_calls=msg.tool_calls if hasattr(msg, 'tool_calls') else None
                        )
                        clean.append(clean_msg)
                else:
                    clean.append(msg)
            else:
                clean.append(msg)
        
        return clean
    
    def _contains_analysis_markers(self, content: str) -> bool:
        """检查是否包含分析标记"""
        markers = [
            '## Error Analysis',
            '## Success Analysis', 
            '## Contrast Analysis',
            '### Why It Failed',
            '### Key Differences',
            '### Correct Steps',
        ]
        return any(marker in content for marker in markers)
    
    def _extract_pre_analysis_content(self, content: str) -> str:
        """提取分析前的内容"""
        markers = [
            '## Error Analysis',
            '## Success Analysis',
            '## Contrast Analysis',
        ]
        
        min_pos = len(content)
        for marker in markers:
            pos = content.find(marker)
            if pos != -1 and pos < min_pos:
                min_pos = pos
        
        if min_pos < len(content):
            return content[:min_pos].strip()
        return content
    
    def _get_clean_system_prompt(self, system_prompt: str) -> str:
        """获取干净的系统提示词（移除分析内容）"""
        if '## Error Analysis' in system_prompt:
            system_prompt = system_prompt.split('## Error Analysis')[0]
        if '## Contrast Analysis' in system_prompt:
            system_prompt = system_prompt.split('## Contrast Analysis')[0]
        
        return system_prompt.strip()
    
    def _validate_format(self, messages: List[dict], tools: List[dict]) -> bool:
        """验证消息格式"""
        if not messages:
            return False
        
        # 检查基本结构
        if messages[0]['role'] != 'system':
            logger.warning("First message is not system")
            return False
        
        # 检查工具调用格式
        for msg in messages:
            if msg['role'] == 'assistant' and msg.get('content') and '<tool_call>' in msg['content']:
                if not self._validate_tool_call_format(msg['content']):
                    logger.warning(f"Invalid tool call format: {msg['content'][:100]}")
                    return False
            
            if msg['role'] == 'tool':
                if not self._validate_tool_response_format(msg.get('content', '')):
                    logger.warning(f"Invalid tool response format: {msg.get('content', '')[:100]}")
                    return False
        
        return True
    
    def _validate_tool_call_format(self, content: str) -> bool:
        """验证工具调用格式"""
        try:
            pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for match in matches:
                call = json.loads(match)
                if 'name' not in call or 'arguments' not in call:
                    return False
            
            return len(matches) > 0
        except Exception as e:
            logger.debug(f"Tool call validation error: {e}")
            return False
    
    def _validate_tool_response_format(self, content: str) -> bool:
        """验证工具响应格式"""
        try:
            # 工具响应可能是 JSON 字符串或直接的字符串
            if not content:
                return False
            
            # 尝试解析为 JSON
            if content.startswith('{'):
                # 使用更宽容的JSON解析
                try:
                    # 尝试找到第一个完整的JSON对象
                    decoder = json.JSONDecoder()
                    response, idx = decoder.raw_decode(content)
                    
                    # 验证必要字段
                    if 'name' in response and 'result' in response:
                        return True
                    
                    # 如果没有name字段，可能是简化格式
                    if 'result' in response or 'error' in response:
                        return True
                        
                except json.JSONDecodeError as e:
                    # JSON解析失败，可能是截断的
                    logger.debug(f"Tool response JSON decode error: {e}, content: {content[:200]}")
                    # 检查是否至少包含基本结构
                    if '"name"' in content and ('"result"' in content or '"error"' in content):
                        # 看起来像是被截断的有效JSON，接受它
                        return True
                    return False
            
            # 或者是直接的字符串响应（如"Error: ..."）
            return True
        except Exception as e:
            logger.debug(f"Tool response validation error: {e}")
            return False
    
    def _validate_error_recovery(self, messages: List[dict]) -> bool:
        """验证错误恢复序列的合理性"""
        error_count = 0
        consecutive_errors = 0
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'tool':
                content = msg.get('content', '')
                
                # 检查是否是错误
                is_error = False
                try:
                    if content.startswith('{'):
                        tool_data = json.loads(content)
                        result = tool_data.get('result', '')
                        if isinstance(result, str) and result.startswith('Error:'):
                            is_error = True
                    elif 'Error:' in content:
                        is_error = True
                except:
                    pass
                
                if is_error:
                    error_count += 1
                    consecutive_errors += 1
                    
                    # 检查错误次数限制
                    if error_count > 8:
                        logger.warning(f"错误次数超过限制: {error_count}")
                        return False
                    
                    # 检查是否有恢复尝试
                    if i + 1 < len(messages):
                        next_msg = messages[i + 1]
                        if next_msg['role'] != 'assistant':
                            logger.warning(f"错误后没有恢复尝试")
                            return False
                else:
                    consecutive_errors = 0
                
                # 检查连续错误
                if consecutive_errors > 3:
                    logger.warning(f"连续错误次数过多: {consecutive_errors}")
                    return False
        
        return True
