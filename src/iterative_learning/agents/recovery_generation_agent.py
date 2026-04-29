"""
恢复生成Agent

使用LLM或模板生成错误恢复的思考和正确调用。
"""

import json
import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from loguru import logger

from ..utils import LLMClient, deep_copy, extract_tool_call, remove_think_tags
from ..templates import (
    RECOVERY_TEMPLATES,
    RECOVERY_GENERATION_PROMPT,
    ERROR_TYPE_GUIDANCE,
    select_template,
    format_template_safe,
)
from ..injection.validator import TrajectoryValidator


@dataclass
class RecoveryResponse:
    """恢复响应"""
    thinking: str           # 恢复思考
    correct_call: dict      # 正确的工具调用 {"name": "...", "arguments": {...}}
    full_content: str       # 完整的assistant消息内容


class RecoveryGenerationAgent:
    """恢复生成Agent - 使用LLM或模板生成恢复过程"""
    
    def __init__(
        self, 
        llm: str, 
        api_base: str = None,
        use_llm: bool = True,
        timeout: int = 30,
    ):
        """
        Args:
            llm: LLM模型名称
            api_base: API基础URL
            use_llm: 是否使用LLM生成（False则使用模板）
            timeout: API调用超时时间
        """
        self.use_llm = use_llm
        self.validator = TrajectoryValidator()
        
        if use_llm:
            self.llm_client = LLMClient(
                model=llm,
                api_base=api_base,
                timeout=timeout,
            )
        
        logger.info(f"Initialized RecoveryGenerationAgent, use_llm={use_llm}")
    
    def generate_recovery(
        self,
        wrong_call: dict,
        error_message: str,
        correct_call: dict,
        context: List[dict],
        error_type: str = "parameter_error",
    ) -> Optional[RecoveryResponse]:
        """
        生成恢复思考和正确调用
        
        Args:
            wrong_call: 错误的工具调用 {"name": "...", "arguments": {...}}
            error_message: 错误消息
            correct_call: 正确的工具调用
            context: 对话上下文
            error_type: 错误类型
            
        Returns:
            RecoveryResponse 或 None
        """
        if self.use_llm:
            response = self._generate_with_llm(
                wrong_call, error_message, correct_call, context, error_type
            )
            if response:
                return response
            # LLM失败，回退到规则生成
            logger.warning("LLM generation failed, falling back to rule-based")
        
        return self._generate_with_rules(
            wrong_call, error_message, correct_call, error_type
        )
    
    def _generate_with_llm(
        self,
        wrong_call: dict,
        error_message: str,
        correct_call: dict,
        context: List[dict],
        error_type: str,
    ) -> Optional[RecoveryResponse]:
        """使用LLM生成恢复"""
        try:
            prompt = self._build_recovery_prompt(
                wrong_call, error_message, correct_call, context, error_type
            )
            
            # 调用LLM
            response_text = self.llm_client.generate_from_prompt(prompt)
            if not response_text:
                return None
            
            # 解析响应
            return self._parse_llm_response(response_text, correct_call)
            
        except Exception as e:
            logger.error(f"LLM recovery generation failed: {e}")
            return None
    
    def _build_recovery_prompt(
        self,
        wrong_call: dict,
        error_message: str,
        correct_call: dict,
        context: List[dict],
        error_type: str,
    ) -> str:
        """构建恢复生成Prompt"""
        # 格式化上下文（只取最近几条）
        context_str = ""
        recent_context = context[-6:] if len(context) > 6 else context
        for msg in recent_context:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]  # 截断长内容
            context_str += f"[{role}]: {content}\n"
        
        # 获取错误类型指导
        guidance = ERROR_TYPE_GUIDANCE.get(error_type, "Analyze the error and explain how you'll fix it.")
        
        # 填充prompt模板
        prompt = RECOVERY_GENERATION_PROMPT.format(
            context=context_str,
            wrong_call=json.dumps(wrong_call, ensure_ascii=False, indent=2),
            error_message=error_message,
            correct_tool=correct_call['name'],
            correct_args=json.dumps(correct_call['arguments'], ensure_ascii=False),
            guidance=guidance,
        )
        
        return prompt
    
    def _parse_llm_response(
        self, 
        response_text: str, 
        correct_call: dict
    ) -> Optional[RecoveryResponse]:
        """解析LLM响应"""
        try:
            if not response_text or not response_text.strip():
                logger.warning("Empty LLM response")
                return None
            
            # 移除可能的 <think>...</think> 标签
            cleaned_response = remove_think_tags(response_text)
            
            # 提取工具调用
            extracted_call = extract_tool_call(cleaned_response)
            
            if extracted_call:
                # 验证提取的调用 - 工具名必须正确
                if extracted_call['name'] != correct_call['name']:
                    logger.warning(
                        f"LLM generated wrong tool name: {extracted_call['name']} "
                        f"vs expected {correct_call['name']}, using correct call"
                    )
                    extracted_call = deep_copy(correct_call)
                
                # 验证参数结构
                if not isinstance(extracted_call.get('arguments'), dict):
                    logger.warning("Invalid arguments structure, using correct call")
                    extracted_call = deep_copy(correct_call)
            else:
                # 无法提取，使用正确的调用
                logger.warning("Could not extract tool_call from LLM response, using correct call")
                extracted_call = deep_copy(correct_call)
            
            # 提取思考部分（tool_call之前的内容）
            thinking = cleaned_response
            if '<tool_call>' in cleaned_response:
                thinking = cleaned_response.split('<tool_call>')[0].strip()
            
            # 清理思考内容
            thinking = thinking.replace('```', '').strip()
            
            # 确保思考内容不为空
            if not thinking:
                thinking = "Let me correct this and try again."
            
            # 构建完整内容
            full_content = f"{thinking}\n{self.validator.format_tool_call(extracted_call['name'], extracted_call['arguments'])}"
            
            return RecoveryResponse(
                thinking=thinking,
                correct_call=extracted_call,
                full_content=full_content,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_with_rules(
        self,
        wrong_call: dict,
        error_message: str,
        correct_call: dict,
        error_type: str,
    ) -> RecoveryResponse:
        """使用规则模板生成恢复"""
        # 深拷贝避免修改原始数据
        wrong_args = deep_copy(wrong_call.get('arguments', {}))
        correct_args = deep_copy(correct_call.get('arguments', {}))
        
        # 找出不同的参数
        diff_param = None
        wrong_value = None
        correct_value = None
        
        for key in correct_args:
            if key in wrong_args and wrong_args[key] != correct_args[key]:
                diff_param = key
                wrong_value = str(wrong_args[key])[:50]
                correct_value = str(correct_args[key])[:50]
                break
        
        if diff_param is None and wrong_args:
            diff_param = list(wrong_args.keys())[0]
            wrong_value = str(wrong_args.get(diff_param, ''))[:50]
            correct_value = str(correct_args.get(diff_param, ''))[:50]
        
        # 选择模板
        templates = RECOVERY_TEMPLATES.get(error_type, RECOVERY_TEMPLATES["parameter_error"])
        template = select_template(templates)
        
        # 填充模板
        thinking = format_template_safe(
            template,
            param=diff_param or "parameter",
            wrong_value=wrong_value or "incorrect",
            correct_value=correct_value or "correct",
            error_type=error_message[:100] if error_message else "error",
            wrong_tool=wrong_call.get('name', 'unknown'),
            correct_tool=correct_call.get('name', 'unknown'),
        )
        
        # 构建完整内容
        full_content = f"{thinking}\n{self.validator.format_tool_call(correct_call['name'], correct_call['arguments'])}"
        
        return RecoveryResponse(
            thinking=thinking,
            correct_call=correct_call,
            full_content=full_content,
        )
