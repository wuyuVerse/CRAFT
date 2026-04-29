"""
轨迹数据格式化模块

将 SimulationRun 的消息转换为可读格式或 SFT 训练数据格式。
"""

import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolMessage,
    UserMessage,
)


def build_history(messages: List[Message]) -> str:
    """
    将 SimulationRun.messages 转成可读的对话与工具调用流水。
    
    Args:
        messages: 消息列表
        
    Returns:
        格式化的对话历史字符串
    """
    history = ""
    tool_call_dict = {}
    
    for message in messages:
        if isinstance(message, UserMessage):
            history += f"User: {message.content}\n"
        elif isinstance(message, AssistantMessage) and not message.tool_calls:
            history += f"Assistant: {message.content}\n"
        elif isinstance(message, AssistantMessage) and message.tool_calls:
            history += f"Assistant: {message.content}\n"
            history += f"And called Tools: {[{'name': tc.name, 'arguments': tc.arguments} for tc in message.tool_calls]}\n"
            for tool_call in message.tool_calls:
                tool_call_dict[tool_call.id] = tool_call.name
        elif isinstance(message, ToolMessage):
            tool_name = tool_call_dict.get(message.id, "unknown_tool")
            history += f"Tool: {{'name': {tool_name}, 'result': {message.content}}}\n"
    
    return history.strip()


def format_sft_data(
    messages: List[Message],
    system_prompt: str,
    tools: List[dict],
    output_path: Optional[str] = None,
) -> List[dict]:
    """
    将轨迹消息格式化为 SFT 训练数据格式。
    
    Args:
        messages: 原始消息列表
        system_prompt: 系统提示词
        tools: 工具定义列表
        output_path: 输出目录路径（可选，如果提供则写入文件）
        
    Returns:
        格式化后的消息列表
    """
    new_messages = []
    tool_call_dict = {}
    tool_response = []
    skip_until_clean_user = False
    skipped_count = 0

    for message in messages:
        if isinstance(message, UserMessage):
            if message.tool_calls:
                # 遇到带 tool_calls 的 UserMessage，开始跳过模式
                skip_until_clean_user = True
                skipped_count += 1
                logger.debug("跳过 UserMessage with tool_calls")
                continue
            else:
                if skip_until_clean_user:
                    skip_until_clean_user = False
                    logger.debug(f"结束跳过模式，共跳过 {skipped_count} 条消息")
                
                # 正常处理 UserMessage
                if tool_response:
                    # 合并所有工具响应，去掉所有<tool_response>标签
                    combined = ''.join(tool_response)
                    # 移除所有<tool_response>标签
                    combined = combined.replace('\n<tool_response>\n', '').replace('\n</tool_response>', '')
                    new_messages.append({
                        'role': 'tool',
                        'content': combined.strip()
                    })
                    tool_response = []
                new_messages.append({'role': 'user', 'content': message.content})
                
        elif isinstance(message, AssistantMessage) and not message.tool_calls:
            if tool_response:
                # 合并所有工具响应，去掉所有<tool_response>标签
                combined = ''.join(tool_response)
                combined = combined.replace('\n<tool_response>\n', '').replace('\n</tool_response>', '')
                new_messages.append({
                    'role': 'tool',
                    'content': combined.strip()
                })
                tool_response = []
            new_messages.append({'role': 'assistant', 'content': message.content})
            
        elif isinstance(message, AssistantMessage) and message.tool_calls:
            if tool_response:
                # 合并所有工具响应，去掉所有<tool_response>标签
                combined = ''.join(tool_response)
                combined = combined.replace('\n<tool_response>\n', '').replace('\n</tool_response>', '')
                new_messages.append({
                    'role': 'tool',
                    'content': combined.strip()
                })
                tool_response = []
            
            this_tool_call = []
            for tool_call in message.tool_calls:
                this_tool_call.append(
                    f"\n<tool_call>\n{json.dumps({'name': tool_call.name, 'arguments': tool_call.arguments})}\n</tool_call>"
                )
                tool_call_dict[tool_call.id] = tool_call.name
            
            new_messages.append({'role': 'assistant', 'content': ''.join(this_tool_call)})
            
        elif isinstance(message, ToolMessage):
            if skip_until_clean_user:
                skipped_count += 1
                logger.debug("跳过 ToolMessage in user tool interaction")
                continue
            
            try:
                result = json.loads(message.content)
            except:
                result = message.content
            
            tool_response.append(
                f"\n<tool_response>\n{json.dumps({'name': tool_call_dict.get(message.id, 'unknown'), 'result': result})}\n</tool_response>"
            )

    # 写入文件
    if output_path and new_messages:
        # 清理首尾
        if new_messages and new_messages[0]['role'] == 'assistant':
            new_messages.pop(0)
        if new_messages and new_messages[-1]['role'] == 'user':
            new_messages.pop(-1)
        
        # 添加 system prompt
        new_messages.insert(0, {'role': 'system', 'content': system_prompt})
        
        sft_file = Path(output_path) / "sft_data.jsonl"
        with open(sft_file, "a") as f:
            f.write(json.dumps({'messages': new_messages, 'tools': json.dumps(tools)}) + "\n")
        
        logger.info(f"成功写入 SFT 数据，消息数: {len(new_messages)}, 跳过: {skipped_count}")
    
    return new_messages
