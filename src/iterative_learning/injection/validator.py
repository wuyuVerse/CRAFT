"""
轨迹格式验证器

确保生成的轨迹符合tau2-bench格式要求。
"""

import json
import re
from typing import List, Tuple, Optional
from loguru import logger


class TrajectoryValidator:
    """轨迹格式验证器 - 确保生成的轨迹符合tau2-bench格式"""
    
    def validate_message(self, msg: dict) -> Tuple[bool, str]:
        """验证单条消息格式"""
        if 'role' not in msg:
            return False, "Missing 'role' field"
        
        role = msg['role']
        
        if role == 'system':
            return self._validate_system_message(msg)
        elif role == 'user':
            return self._validate_user_message(msg)
        elif role == 'assistant':
            return self._validate_assistant_message(msg)
        elif role == 'tool':
            return self._validate_tool_message(msg)
        else:
            return False, f"Invalid role: {role}"
    
    def _validate_system_message(self, msg: dict) -> Tuple[bool, str]:
        """验证system消息"""
        content = msg.get('content')
        if not content or not isinstance(content, str):
            return False, "System message must have string content"
        return True, "OK"
    
    def _validate_user_message(self, msg: dict) -> Tuple[bool, str]:
        """验证user消息"""
        content = msg.get('content')
        if not content or not isinstance(content, str):
            return False, "User message must have string content"
        return True, "OK"
    
    def _validate_assistant_message(self, msg: dict) -> Tuple[bool, str]:
        """验证assistant消息"""
        content = msg.get('content')
        
        if content is None:
            return False, "Assistant message must have content"
        
        if not isinstance(content, str):
            return False, "Assistant content must be string"
        
        # 如果包含tool_call，验证格式
        if '<tool_call>' in content:
            valid, error = self._validate_tool_call_format(content)
            if not valid:
                return False, f"Invalid tool_call format: {error}"
        
        return True, "OK"
    
    def _validate_tool_call_format(self, content: str) -> Tuple[bool, str]:
        """验证工具调用格式"""
        # 格式: \n<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            return False, "No valid tool_call found"
        
        for match in matches:
            try:
                call = json.loads(match)
                
                if 'name' not in call:
                    return False, "tool_call missing 'name'"
                if 'arguments' not in call:
                    return False, "tool_call missing 'arguments'"
                if not isinstance(call['name'], str):
                    return False, "'name' must be string"
                if not isinstance(call['arguments'], dict):
                    return False, "'arguments' must be dict"
                    
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON in tool_call: {e}"
        
        return True, "OK"
    
    def _validate_tool_message(self, msg: dict) -> Tuple[bool, str]:
        """验证tool消息"""
        content = msg.get('content')
        
        if not content:
            return False, "Tool message must have content"
        
        if not isinstance(content, str):
            return False, "Tool content must be string"
        
        try:
            # 格式: {"name": "...", "result": ...}
            data = json.loads(content)
            
            if 'name' not in data:
                return False, "Tool response missing 'name'"
            if 'result' not in data:
                return False, "Tool response missing 'result'"
                
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in tool response: {e}"
        
        return True, "OK"
    
    def validate_trajectory(self, messages: List[dict]) -> Tuple[bool, str]:
        """验证完整轨迹"""
        if not messages:
            return False, "Empty trajectory"
        
        # 1. 检查第一条是system
        if messages[0].get('role') != 'system':
            return False, "First message must be system"
        
        # 2. 验证每条消息
        prev_role = None
        for i, msg in enumerate(messages):
            valid, error = self.validate_message(msg)
            if not valid:
                return False, f"Message {i}: {error}"
            
            role = msg.get('role')
            
            # 3. 检查消息顺序：tool必须跟在assistant后面
            if role == 'tool' and prev_role != 'assistant':
                return False, f"Message {i}: tool must follow assistant"
            
            prev_role = role
        
        return True, "Valid"
    
    def validate_sft_data(self, data: dict) -> Tuple[bool, str]:
        """验证完整的SFT数据"""
        # 检查必要字段
        if 'messages' not in data:
            return False, "Missing 'messages' field"
        if 'tools' not in data:
            return False, "Missing 'tools' field"
        
        # 验证messages
        valid, error = self.validate_trajectory(data['messages'])
        if not valid:
            return False, f"Invalid messages: {error}"
        
        # 验证tools是有效的JSON字符串
        try:
            tools = json.loads(data['tools'])
            if not isinstance(tools, list):
                return False, "tools must be a JSON array"
        except json.JSONDecodeError as e:
            return False, f"Invalid tools JSON: {e}"
        
        return True, "Valid"
    
    def extract_tool_call(self, content: str) -> Optional[dict]:
        """从assistant消息中提取工具调用"""
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
    
    def format_tool_call(self, name: str, arguments: dict) -> str:
        """格式化工具调用为标准格式"""
        call = {"name": name, "arguments": arguments}
        return f"\n<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"
    
    def format_tool_response(self, name: str, result: any) -> str:
        """格式化工具响应为标准格式"""
        response = {"name": name, "result": result}
        return json.dumps(response, ensure_ascii=False)
