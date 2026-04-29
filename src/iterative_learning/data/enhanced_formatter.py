"""
增强数据格式化器

包含质量验证功能。
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolMessage,
    UserMessage,
)


@dataclass
class DataQualityScore:
    """数据质量评分"""
    completeness: float       # 步骤完整性 (0-1)
    parameter_accuracy: float # 参数准确性 (0-1)
    reasoning_quality: float  # 推理质量 (0-1)
    overall: float            # 综合评分
    
    def to_dict(self) -> dict:
        return asdict(self)


class EnhancedDataFormatter:
    """增强的数据格式化器"""
    
    # 推理关键词（用于评估推理质量）
    REASONING_KEYWORDS = [
        "let me", "i need to", "first", "before", "verify",
        "check", "confirm", "make sure", "looking at", "based on",
    ]
    
    def __init__(
        self,
        domain: str,
        expected_actions: Optional[List[str]] = None,
        quality_threshold: float = 0.6,
    ):
        """
        Args:
            domain: 领域名称
            expected_actions: 期望的动作列表
            quality_threshold: 质量阈值
        """
        self.domain = domain
        self.expected_actions = expected_actions or []
        self.quality_threshold = quality_threshold
    
    def format_with_quality_check(
        self,
        messages: List[Message],
        system_prompt: str,
        tools: List[dict],
        task_evaluation: Optional[dict] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[List[dict], DataQualityScore, bool]:
        """
        格式化数据并进行质量检查
        
        Args:
            messages: 原始消息列表
            system_prompt: 系统提示词
            tools: 工具定义
            task_evaluation: 任务评估结果
            output_path: 输出路径
            
        Returns:
            (formatted_messages, quality_score, should_save)
        """
        # 基础格式化
        formatted = self._format_messages(messages, system_prompt, tools)
        
        # 质量评分
        score = self._compute_quality_score(messages, task_evaluation)
        
        # 判断是否应该保存
        should_save = score.overall >= self.quality_threshold
        
        # 保存数据
        if should_save and output_path and formatted:
            self._save_sft_data(formatted, tools, output_path)
            logger.info(f"✓ 高质量数据已保存 (score: {score.overall:.2f})")
        elif not should_save and output_path:
            logger.warning(f"✗ 数据质量不足，跳过 (score: {score.overall:.2f})")
        
        return formatted, score, should_save
    
    def _format_messages(
        self,
        messages: List[Message],
        system_prompt: str,
        tools: List[dict],
    ) -> List[dict]:
        """格式化消息为 SFT 格式"""
        new_messages = []
        tool_call_dict = {}
        tool_response = []
        skip_until_clean_user = False

        for message in messages:
            if isinstance(message, UserMessage):
                if message.tool_calls:
                    skip_until_clean_user = True
                    continue
                else:
                    if skip_until_clean_user:
                        skip_until_clean_user = False
                    
                    if tool_response:
                        # 合并所有工具响应，去掉所有<tool_response>标签
                        combined = ''.join(tool_response)
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
                # 清理assistant消息中的<think>标签
                content = message.content if message.content else ""
                clean_content = self._remove_think_tags(content)
                new_messages.append({'role': 'assistant', 'content': clean_content})
                
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
                    continue
                
                try:
                    result = json.loads(message.content)
                except:
                    result = message.content
                
                tool_response.append(
                    f"\n<tool_response>\n{json.dumps({'name': tool_call_dict.get(message.id, 'unknown'), 'result': result})}\n</tool_response>"
                )

        # 清理首尾
        if new_messages and new_messages[0]['role'] == 'assistant':
            new_messages.pop(0)
        if new_messages and new_messages[-1]['role'] == 'user':
            new_messages.pop(-1)
        
        # 添加 system prompt
        if new_messages:
            new_messages.insert(0, {'role': 'system', 'content': system_prompt})
        
        return new_messages
    
    def _remove_think_tags(self, text: str) -> str:
        """
        移除LLM响应中的思考标签（如DeepSeek的<think>标签）
        
        这些标签不应该出现在最终的SFT数据中
        """
        import re
        
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
    
    def _compute_quality_score(
        self,
        messages: List[Message],
        task_evaluation: Optional[dict],
    ) -> DataQualityScore:
        """计算数据质量评分"""
        
        # 提取实际调用的工具
        called_tools = []
        for msg in messages:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                called_tools.extend([tc.name for tc in msg.tool_calls])
        
        # 1. 完整性评分
        if self.expected_actions:
            matched = sum(1 for t in self.expected_actions if t in called_tools)
            completeness = matched / len(self.expected_actions)
        else:
            completeness = 1.0 if called_tools else 0.5
        
        # 2. 参数准确性评分
        parameter_accuracy = 1.0
        if task_evaluation:
            reward_info = task_evaluation
            # 从 reward_info 提取 action checks
            if hasattr(reward_info, 'action_reward_info'):
                action_info = reward_info.action_reward_info
                if action_info and hasattr(action_info, 'action_checks'):
                    checks = action_info.action_checks or []
                    if checks:
                        correct = sum(1 for c in checks if getattr(c, 'score', 0) > 0)
                        parameter_accuracy = correct / len(checks)
        
        # 3. 推理质量评分
        reasoning_count = 0
        for msg in messages:
            if isinstance(msg, AssistantMessage) and msg.content:
                content_lower = msg.content.lower()
                if any(kw in content_lower for kw in self.REASONING_KEYWORDS):
                    reasoning_count += 1
        reasoning_quality = min(reasoning_count / 3, 1.0)
        
        # 综合评分
        overall = (
            completeness * 0.4 +
            parameter_accuracy * 0.4 +
            reasoning_quality * 0.2
        )
        
        return DataQualityScore(
            completeness=completeness,
            parameter_accuracy=parameter_accuracy,
            reasoning_quality=reasoning_quality,
            overall=overall,
        )
    
    def _save_sft_data(
        self,
        messages: List[dict],
        tools: List[dict],
        output_path: str,
    ):
        """保存 SFT 数据"""
        sft_file = Path(output_path) / "sft_data.jsonl"
        with open(sft_file, "a") as f:
            f.write(json.dumps({
                'messages': messages,
                'tools': json.dumps(tools)
            }) + "\n")
