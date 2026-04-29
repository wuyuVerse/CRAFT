"""
自我审视模块：用于多轮对话中的质量检查
包括：Assistant回复检查、ToolCall幻觉检测、User回复自我审视检查
"""
import json
from typing import Dict, List, Any, Optional, Tuple
from openai import AsyncOpenAI
from ..utils.logger import log


class AssistantResponseChecker:
    """Assistant回复检查器：确保回复简洁、重点突出、无情绪化表达"""
    
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
    
    async def check_and_correct(self, response: str, context: str = "") -> Dict[str, Any]:
        """
        检查assistant的文本回复是否简洁、重点突出
        
        Args:
            response: assistant的回复内容
            context: 上下文信息（可选）
        
        Returns:
            {
                "is_valid": bool,
                "reason": str,
                "corrected_response": str
            }
        """
        check_prompt = f"""
You are an Assistant Response Quality Checker. Evaluate if the assistant's response is concise, clear, and professional.

**Validation Rules:**
1. **Conciseness**: Should be brief (1-3 sentences max), get to the point quickly
2. **No Emotional Language**: Avoid unnecessary emotional expressions like "Great!", "Wonderful!", "I'm happy to help!"
3. **Focus on Key Points**: Directly address the core information, no fluff
4. **Professional Tone**: Clear, direct, informative

**Current Response:**
{response}

**Context (if any):**
{context}

**Check if the response violates any rules above.**

Output ONLY valid JSON:
{{
    "is_valid": true/false,
    "reason": "Brief explanation if invalid, empty if valid",
    "corrected_response": "Corrected response if invalid (concise, professional, focused), empty if valid"
}}
"""
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a quality checker for assistant responses. Focus on conciseness and professionalism."},
                    {"role": "user", "content": check_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = completion.choices[0].message.content
            if not content:
                return {"is_valid": True, "reason": "", "corrected_response": ""}
            
            result = json.loads(content)
            return {
                "is_valid": result.get("is_valid", True),
                "reason": result.get("reason", ""),
                "corrected_response": result.get("corrected_response", "")
            }
            
        except Exception as e:
            log(f"[Assistant检查器] 错误: {e}")
            return {"is_valid": True, "reason": "", "corrected_response": ""}


class UserResponseReflectionChecker:
    """User回复自我审视检查器：检查是否符合人类习惯、是否简洁"""
    
    def __init__(self, client, model_name, max_iterations: int = 1):
        self.client = client
        self.model_name = model_name
        self.max_iterations = max_iterations
    
    def _check_basic_rules(self, response: str, user_info: Dict) -> Tuple[bool, List[str]]:
        """
        简单检查：检查简洁性和长度
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 检查长度（简洁性）- 超过5句话或超过300字符认为过长
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) > 5:
            errors.append(f"回复过长({len(sentences)}句)，用户回复应控制在1-5句话内")
        
        if len(response) > 300:
            errors.append(f"回复字符过多({len(response)}字符)，应更简洁")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    async def _reflect_and_improve(
        self,
        response: str,
        user_info: Dict,
        validation_errors: List[str],
        iteration: int
    ) -> Optional[str]:
        """
        自我审视：要求符合人类习惯、简洁明了
        
        Returns:
            改进后的回复，如果失败返回None
        """
        reflection_prompt = f"""
Check if this user response is concise, natural and sounds like a real person.

**User Response:**
{response}

**Validation Errors (if any):**
{chr(10).join(validation_errors) if validation_errors else "No errors"}

**Requirements:**
1. **Conciseness**: Keep it brief (1-3 sentences max, under 200 characters)
2. **Natural tone**: Sounds like a real person talking, not a chatbot
3. **Focus on key points**: Get to the point quickly, no unnecessary details
4. **Simple language**: Avoid overly complex or verbose expressions

**DO NOT change:**
- Technical terms or specific requests (keep user's intent)
- Essential information needed for the task

Output JSON:
{{
    "is_concise": true/false,
    "is_natural": true/false,
    "needs_correction": true/false,
    "improved_response": "simplified and concise response" or null,
    "reason": "brief explanation if correction needed"
}}

Set needs_correction=true if the response is too verbose, unnatural, or contains unnecessary details.
"""
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": reflection_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = completion.choices[0].message.content
            if not content:
                return None
            
            result = json.loads(content)
            
            log(f"[User审视] 简洁: {result.get('is_concise', True)}, 自然: {result.get('is_natural', True)}, 需修正: {result.get('needs_correction', False)}")
            if result.get('reason'):
                log(f"[User审视] 原因: {result.get('reason')}")
            
            if result.get('needs_correction', False) and result.get('improved_response'):
                return result['improved_response']
            
            return None
            
        except Exception as e:
            log(f"[User审视] 错误: {e}")
            return None
    
    async def iterative_check(
        self,
        response: str,
        user_info: Dict
    ) -> Tuple[bool, str, int]:
        """
        迭代式检查，直到用户回复符合要求或达到最大迭代次数
        
        Returns:
            (is_valid, final_response, iterations_used)
        """
        current_response = response
        iterations_used = 0
        
        for iteration in range(self.max_iterations):
            log(f"\n{'='*60}")
            log(f"User回复审视迭代 {iteration + 1}/{self.max_iterations}")
            log(f"{'='*60}")
            
            # 1. 检查基本规则
            is_valid, errors = self._check_basic_rules(current_response, user_info)
            
            if errors:
                log(f"[验证] 发现 {len(errors)} 个问题:")
                for error in errors:
                    log(f"  - {error}")
            else:
                log(f"[验证] 通过 - 符合人类回复习惯")
            
            # 2. 进行自我审视
            improved = await self._reflect_and_improve(
                response=current_response,
                user_info=user_info,
                validation_errors=errors,
                iteration=iteration
            )
            
            if improved:
                log(f"[修正] 应用改进的回复")
                current_response = improved
                iterations_used = iteration + 1  # 记录使用的迭代次数
            elif is_valid:
                # 没有改进建议且验证通过
                log(f"\n[成功] User回复审视完成，回复符合要求")
                return True, current_response, iterations_used
            else:
                # 没有改进建议但验证未通过，继续下一轮
                log(f"[警告] 未提供改进建议，继续下一轮")
        
        # 最终检查
        is_valid, errors = self._check_basic_rules(current_response, user_info)
        
        if not is_valid:
            log(f"\n[失败] 达到最大迭代次数，仍存在问题:")
            for error in errors:
                log(f"  - {error}")
            # 虽然有问题，但还是返回当前版本（可以考虑返回False）
            return False, current_response, iterations_used
        
        log(f"\n[完成] User回复审视完成，返回最终版本")
        return True, current_response, iterations_used


class ToolCallHallucinationChecker:
    """ToolCall幻觉检查器：检测幻觉工具和幻觉参数"""
    
    def __init__(self, client, model_name, max_iterations: int = 1):
        self.client = client
        self.model_name = model_name
        self.max_iterations = max_iterations
    
    def validate_tool_calls_against_schema(
        self, 
        tool_calls: List[Any], 
        available_tools: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        验证工具调用是否符合工具定义
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 构建工具名称到定义的映射
        tool_map = {}
        for tool in available_tools:
            if 'function' in tool:
                func = tool['function']
                tool_map[func['name']] = func
        
        # 检查每个工具调用
        for idx, tool_call in enumerate(tool_calls):
            func_name = tool_call.function.name
            func_args_str = tool_call.function.arguments
            
            # 1. 检查工具是否存在（幻觉工具检测）
            if func_name not in tool_map:
                errors.append(
                    f"Tool call {idx}: 幻觉工具 '{func_name}' - "
                    f"该工具不在可用工具列表中。可用工具: {list(tool_map.keys())}"
                )
                continue
            
            # 2. 解析参数
            try:
                func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
            except json.JSONDecodeError as e:
                errors.append(f"Tool call {idx}: 参数格式错误 - {str(e)}")
                continue
            
            # 3. 检查参数是否在工具定义中（幻觉参数检测）
            tool_def = tool_map[func_name]
            defined_params = tool_def.get('parameters', {}).get('properties', {})
            
            for param_name in func_args.keys():
                if param_name not in defined_params:
                    errors.append(
                        f"Tool call {idx} ({func_name}): 幻觉参数 '{param_name}' - "
                        f"该参数不在工具定义中。可用参数: {list(defined_params.keys())}"
                    )
            
            # 4. 检查必需参数是否都提供了
            required_params = tool_def.get('parameters', {}).get('required', [])
            for param_name in required_params:
                if param_name not in func_args:
                    errors.append(
                        f"Tool call {idx} ({func_name}): 缺少必需参数 '{param_name}'"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    async def reflect_and_correct(
        self,
        question: str,
        tool_calls: List[Any],
        available_tools: List[Dict],
        validation_errors: List[str],
        iteration: int
    ) -> Optional[List[Any]]:
        """
        对工具调用进行自我审视和修正
        
        Returns:
            修正后的tool_calls，如果失败返回None
        """
        # 将tool_calls转换为可序列化的格式
        tool_calls_dicts = []
        for tc in tool_calls:
            tool_calls_dicts.append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            })
        
        reflection_prompt = f"""
You are a self-reflection agent for tool calls. Fix any hallucinations in the tool calls.

**Current Question:**
{question}

**Available Tools:**
{json.dumps(available_tools, ensure_ascii=False, indent=2)}

**Current Tool Calls:**
{json.dumps(tool_calls_dicts, ensure_ascii=False, indent=2)}

**Validation Errors:**
{chr(10).join(validation_errors) if validation_errors else "No errors detected."}

**CRITICAL Requirements:**
- DO NOT hallucinate tools that don't exist in Available Tools
- DO NOT use parameters that are not defined in the tool's schema
- Ensure all required parameters are provided
- Keep the same tool call format

Output ONLY valid JSON:
{{
    "needs_correction": true/false,
    "corrected_tool_calls": [...] or null
}}

If needs_correction is true, provide corrected_tool_calls in the exact same format as input.
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a critical tool call validator and corrector."},
                    {"role": "user", "content": reflection_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                return None
            
            result = json.loads(content)
            
            if result.get('needs_correction', False) and result.get('corrected_tool_calls'):
                log(f"[ToolCall审视 - 第{iteration + 1}轮] 应用修正")
                # 返回修正后的tool calls（字典格式）
                return result['corrected_tool_calls']
            
            return None
            
        except Exception as e:
            log(f"[ToolCall审视] 错误: {e}")
            return None
    
    async def iterative_check(
        self,
        question: str,
        tool_calls: List[Any],
        available_tools: List[Dict]
    ) -> Tuple[bool, List[Any]]:
        """
        迭代式检查，直到工具调用正确或达到最大迭代次数
        
        Returns:
            (is_valid, final_tool_calls)
        """
        current_tool_calls = tool_calls
        
        for iteration in range(self.max_iterations):
            # 验证当前工具调用
            is_valid, errors = self.validate_tool_calls_against_schema(
                current_tool_calls, 
                available_tools
            )
            
            if errors:
                log(f"[ToolCall验证 - 第{iteration + 1}轮] 发现 {len(errors)} 个错误:")
                for error in errors:
                    log(f"  - {error}")
            else:
                log(f"[ToolCall验证] 通过 - 无幻觉")
                return True, current_tool_calls
            
            # 如果还有重试机会，尝试修正
            if iteration < self.max_iterations - 1:
                corrected = await self.reflect_and_correct(
                    question=question,
                    tool_calls=current_tool_calls,
                    available_tools=available_tools,
                    validation_errors=errors,
                    iteration=iteration
                )
                
                if corrected:
                    # 将字典转换回tool call对象
                    from openai.types.chat.chat_completion_message_tool_call import (
                        ChatCompletionMessageToolCall,
                        Function
                    )
                    
                    new_tool_calls = []
                    for tc_dict in corrected:
                        func = Function(
                            name=tc_dict['function']['name'],
                            arguments=tc_dict['function']['arguments']
                        )
                        tc_obj = ChatCompletionMessageToolCall(
                            id=tc_dict.get('id', f"call_{iteration}_{len(new_tool_calls)}"),
                            type='function',
                            function=func
                        )
                        new_tool_calls.append(tc_obj)
                    
                    current_tool_calls = new_tool_calls
                else:
                    log(f"[ToolCall审视] 未能生成修正，继续下一轮")
        
        # 最终检查
        is_valid, errors = self.validate_tool_calls_against_schema(
            current_tool_calls, 
            available_tools
        )
        
        if not is_valid:
            log(f"[ToolCall验证失败] 达到最大迭代次数，仍存在错误")
            return False, current_tool_calls
        
        return True, current_tool_calls
