"""
错误生成Agent

使用规则和模板生成各类错误的工具调用。
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from ..templates import (
    ERROR_MESSAGE_TEMPLATES,
    BUSINESS_LOGIC_ERRORS,
    TOOL_HALLUCINATIONS,
    STATE_ERROR_TEMPLATES,
    STATE_ERROR_KEYWORDS,
    fill_template,
    select_template,
)
from ..utils import deep_copy, corrupt_value


class ErrorType(Enum):
    """错误类型"""
    PARAMETER_ERROR = "parameter_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    STATE_ERROR = "state_error"
    TOOL_HALLUCINATION = "tool_hallucination"


@dataclass
class GeneratedError:
    """生成的错误"""
    error_type: ErrorType
    wrong_call: dict          # {"name": "...", "arguments": {...}}
    error_message: str        # "Error: ..."
    original_call: dict       # 原始正确调用


class ErrorGenerationAgent:
    """错误生成Agent - 使用LLM和错误库样例生成各类错误"""
    
    def __init__(
        self, 
        error_db: dict, 
        domain: str,
        llm: str = None,
        api_base: str = None,
        use_llm: bool = True,
    ):
        """
        Args:
            error_db: 错误数据库
            domain: 领域名称
            llm: LLM模型名称（如果use_llm=True）
            api_base: API基础URL
            use_llm: 是否使用LLM生成错误（False则使用规则）
        """
        self.error_db = error_db
        self.domain = domain
        self.llm = llm
        self.api_base = api_base
        self.use_llm = use_llm
        
        logger.info(f"Initialized ErrorGenerationAgent for {domain}, use_llm={use_llm}")
    
    def generate_error(
        self,
        correct_call: dict,
        tool_result: dict,
        context: List[dict],
        error_type: Optional[ErrorType] = None,
    ) -> Optional[GeneratedError]:
        """
        生成错误的工具调用
        
        Args:
            correct_call: 正确的工具调用 {"name": "...", "arguments": {...}}
            tool_result: 正确的返回结果
            context: 对话上下文
            error_type: 指定错误类型，None则随机选择
            
        Returns:
            GeneratedError 或 None（如果无法生成）
        """
        tool_name = correct_call['name']
        tool_args = correct_call['arguments']
        
        # 如果没有指定错误类型，根据工具和上下文选择
        if error_type is None:
            error_type = self._select_error_type(tool_name, tool_args)
        
        # 使用LLM生成错误
        if self.use_llm and self.llm:
            return self._generate_error_with_llm(
                correct_call, tool_result, context, error_type
            )
        
        # 使用规则生成错误（回退）
        if error_type == ErrorType.PARAMETER_ERROR:
            return self._generate_parameter_error(correct_call, tool_result)
        elif error_type == ErrorType.BUSINESS_LOGIC_ERROR:
            return self._generate_business_logic_error(correct_call, tool_result)
        elif error_type == ErrorType.STATE_ERROR:
            return self._generate_state_error(correct_call, tool_result)
        elif error_type == ErrorType.TOOL_HALLUCINATION:
            return self._generate_tool_hallucination(correct_call, tool_result)
        
        return None
    
    def _select_error_type(self, tool_name: str, tool_args: dict) -> ErrorType:
        """根据工具和参数选择合适的错误类型"""
        # 获取可用的错误类型
        available_types = [ErrorType.PARAMETER_ERROR]  # 参数错误总是可用
        
        # 检查是否有业务逻辑错误模板
        domain_errors = BUSINESS_LOGIC_ERRORS.get(self.domain, {})
        if tool_name in domain_errors and domain_errors[tool_name]:
            available_types.append(ErrorType.BUSINESS_LOGIC_ERROR)
        
        # 检查是否有工具幻觉映射
        domain_hallucinations = TOOL_HALLUCINATIONS.get(self.domain, {})
        if tool_name in domain_hallucinations:
            available_types.append(ErrorType.TOOL_HALLUCINATION)
        
        # 状态错误：检查错误库中是否有相关错误
        domain_db_errors = self.error_db.get(self.domain, {})
        
        # 检查新格式
        if 'state_error' in domain_db_errors:
            state_errors = domain_db_errors['state_error'].get(tool_name, [])
            if state_errors:
                available_types.append(ErrorType.STATE_ERROR)
        else:
            # 旧格式
            tool_db_errors = domain_db_errors.get(tool_name, [])
            has_state_errors = any(
                any(kw in err.get('error', '').lower() for kw in STATE_ERROR_KEYWORDS)
                for err in tool_db_errors
            )
            if has_state_errors:
                available_types.append(ErrorType.STATE_ERROR)
        
        # 使用权重随机选择
        weights = []
        default_weights = {
            ErrorType.PARAMETER_ERROR: 0.35,
            ErrorType.BUSINESS_LOGIC_ERROR: 0.30,
            ErrorType.STATE_ERROR: 0.15,
            ErrorType.TOOL_HALLUCINATION: 0.20,
        }
        for t in available_types:
            weights.append(default_weights.get(t, 0.25))
        
        return random.choices(available_types, weights=weights)[0]
    
    def _generate_error_with_llm(
        self,
        correct_call: dict,
        tool_result: dict,
        context: List[dict],
        error_type: ErrorType,
    ) -> Optional[GeneratedError]:
        """使用LLM生成错误"""
        import json
        from litellm import completion
        from ..templates.error_generation_prompt import (
            ERROR_GENERATION_PROMPT,
            ERROR_TYPE_GUIDANCE_MAP,
            format_conversation_history,
            format_error_examples,
        )
        from ..utils import safe_json_loads, validate_dict_structure
        
        tool_name = correct_call['name']
        tool_args = correct_call['arguments']
        
        # 从错误库中获取该错误类型的样例
        examples = self._get_error_examples(tool_name, error_type, limit=5)
        
        # 构建prompt
        prompt = ERROR_GENERATION_PROMPT.format(
            domain=self.domain,
            error_type=error_type.value,
            conversation_history=format_conversation_history(context, max_turns=3),
            tool_name=tool_name,
            tool_arguments=json.dumps(tool_args, indent=2, ensure_ascii=False),
            tool_result=json.dumps(tool_result, indent=2, ensure_ascii=False)[:500],  # 限制长度
            error_examples=format_error_examples(examples),
            error_type_guidance=ERROR_TYPE_GUIDANCE_MAP.get(error_type.value, ""),
        )
        
        try:
            response = completion(
                model=self.llm,
                messages=[
                    {"role": "system", "content": "You are an error generation expert. Generate realistic errors in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                api_base=self.api_base,
            )
            
            content = response.choices[0].message.content.strip()
            
            # 提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # 解析JSON
            result = safe_json_loads(content)
            if not result:
                raise ValueError("Failed to parse JSON response")
            
            # 验证结构
            if not validate_dict_structure(result, ['wrong_call', 'error_message']):
                raise ValueError("Invalid response structure")
            
            wrong_call = result['wrong_call']
            if not validate_dict_structure(wrong_call, ['name', 'arguments']):
                raise ValueError("Invalid wrong_call structure")
            
            error_message = result['error_message']
            if not isinstance(error_message, str) or not error_message.startswith('Error:'):
                # 自动添加"Error:"前缀
                if isinstance(error_message, str):
                    error_message = f"Error: {error_message}"
                else:
                    raise ValueError("Invalid error_message format")
            
            # 验证参数格式（对于parameter_error）
            if error_type == ErrorType.PARAMETER_ERROR:
                if not self._validate_parameter_format(wrong_call['arguments'], tool_args):
                    logger.warning(f"LLM generated invalid parameter format, falling back to rule-based")
                    return self._generate_parameter_error(correct_call, tool_result)
            
            logger.debug(f"LLM generated {error_type.value}: {error_message}")
            
            return GeneratedError(
                error_type=error_type,
                wrong_call=wrong_call,
                error_message=error_message,
                original_call=correct_call,
            )
            
        except Exception as e:
            logger.warning(f"LLM error generation failed: {e}, falling back to rule-based")
            # 回退到规则生成
            if error_type == ErrorType.PARAMETER_ERROR:
                return self._generate_parameter_error(correct_call, tool_result)
            elif error_type == ErrorType.BUSINESS_LOGIC_ERROR:
                return self._generate_business_logic_error(correct_call, tool_result)
            elif error_type == ErrorType.STATE_ERROR:
                return self._generate_state_error(correct_call, tool_result)
            elif error_type == ErrorType.TOOL_HALLUCINATION:
                return self._generate_tool_hallucination(correct_call, tool_result)
            return None
    
    def _validate_parameter_format(self, wrong_args: dict, correct_args: dict) -> bool:
        """验证参数格式是否正确（避免学习错误格式）"""
        import re
        
        for key, wrong_value in wrong_args.items():
            if key not in correct_args:
                continue
            
            correct_value = correct_args[key]
            
            # 检查类型是否一致
            if type(wrong_value) != type(correct_value):
                return False
            
            # 对于字符串，检查格式是否保持
            if isinstance(correct_value, str) and isinstance(wrong_value, str):
                # ID格式检查 (L1001, P1001等)
                if re.match(r'^[LPBCD]\d{4,}$', correct_value):
                    if not re.match(r'^[LPBCD]\d{4,}$', wrong_value):
                        return False
                
                # 订单ID格式检查 (#W1234567)
                if re.match(r'^#W\d+$', correct_value):
                    if not re.match(r'^#W\d+$', wrong_value):
                        return False
                
                # 预订ID格式检查 (6位大写字母数字)
                if re.match(r'^[A-Z0-9]{6}$', correct_value):
                    if not re.match(r'^[A-Z0-9]{6}$', wrong_value):
                        return False
                
                # 用户ID格式检查 (name_name_1234)
                if re.match(r'^[a-z]+_[a-z]+_\d+$', correct_value):
                    if not re.match(r'^[a-z]+_[a-z]+_\d+$', wrong_value):
                        return False
        
        return True
    
    def _get_error_examples(
        self, 
        tool_name: str, 
        error_type: ErrorType, 
        limit: int = 5
    ) -> List[str]:
        """从错误库中获取错误样例"""
        domain_data = self.error_db.get(self.domain, {})
        
        # 检查新格式
        if error_type.value in domain_data:
            error_type_data = domain_data[error_type.value]
            tool_errors = error_type_data.get(tool_name, [])
        else:
            # 旧格式：需要过滤
            tool_errors = domain_data.get(tool_name, [])
            if error_type == ErrorType.PARAMETER_ERROR:
                keywords = ["not found", "invalid", "missing", "validation"]
                tool_errors = [e for e in tool_errors if any(kw in e.get('error', '').lower() for kw in keywords)]
            elif error_type == ErrorType.STATE_ERROR:
                tool_errors = [e for e in tool_errors if any(kw in e.get('error', '').lower() for kw in STATE_ERROR_KEYWORDS)]
        
        if not tool_errors:
            return []
        
        # 按count排序，取前N个
        sorted_errors = sorted(tool_errors, key=lambda x: x.get('count', 1), reverse=True)
        examples = []
        for err in sorted_errors[:limit]:
            error_msg = err.get('error', '')
            count = err.get('count', 1)
            examples.append(f"{error_msg} (occurred {count} times)")
        
        return examples
    
    def _generate_parameter_error(
        self, 
        correct_call: dict, 
        tool_result: dict
    ) -> Optional[GeneratedError]:
        """生成参数错误"""
        tool_name = correct_call['name']
        tool_args = correct_call.get('arguments', {})
        
        if not tool_args or not isinstance(tool_args, dict):
            logger.warning(f"No valid arguments to corrupt for {tool_name}")
            return None
        
        # 深拷贝参数避免修改原始数据
        tool_args = deep_copy(tool_args)
        
        # 从新的错误库结构中获取错误消息
        error_message = self._get_error_from_db_v2(tool_name, ErrorType.PARAMETER_ERROR)
        
        if error_message:
            # 使用错误库中的真实错误
            error_message = fill_template(error_message, tool_args)
            
            # 修改参数以匹配错误
            wrong_args = self._modify_args_for_error(tool_args, error_message)
        else:
            # 回退：选择一个参数进行修改
            param_name = random.choice(list(tool_args.keys()))
            original_value = tool_args[param_name]
            
            # 生成错误的参数值
            wrong_value = corrupt_value(original_value)
            
            # 确保值确实被修改了
            if wrong_value == original_value:
                wrong_value = str(original_value) + "_invalid" if isinstance(original_value, str) else "invalid_value"
            
            wrong_args = deep_copy(tool_args)
            wrong_args[param_name] = wrong_value
            
            # 使用模板生成错误消息
            template = select_template(ERROR_MESSAGE_TEMPLATES["parameter_error"])
            error_message = fill_template(template, {
                "param": param_name,
                "wrong_value": str(wrong_value)[:50],
            })
        
        return GeneratedError(
            error_type=ErrorType.PARAMETER_ERROR,
            wrong_call={"name": tool_name, "arguments": wrong_args},
            error_message=f"Error: {error_message}",
            original_call=correct_call,
        )
    
    def _generate_business_logic_error(
        self, 
        correct_call: dict, 
        tool_result: dict
    ) -> Optional[GeneratedError]:
        """生成业务逻辑错误"""
        tool_name = correct_call['name']
        tool_args = deep_copy(correct_call.get('arguments', {}))
        
        # 获取该工具的业务逻辑错误模板
        domain_errors = BUSINESS_LOGIC_ERRORS.get(self.domain, {})
        tool_errors = domain_errors.get(tool_name, [])
        
        if not tool_errors:
            logger.debug(f"No business logic error templates for {tool_name}, falling back to parameter error")
            return self._generate_parameter_error(correct_call, tool_result)
        
        # 选择一个错误模板
        error_template = select_template(tool_errors)
        
        # 根据错误类型修改参数
        wrong_args = self._modify_args_for_business_error(tool_args, error_template)
        
        # 填充错误模板
        error_message = fill_template(error_template, {**tool_args, **wrong_args})
        
        return GeneratedError(
            error_type=ErrorType.BUSINESS_LOGIC_ERROR,
            wrong_call={"name": tool_name, "arguments": wrong_args},
            error_message=f"Error: {error_message}",
            original_call=correct_call,
        )
    
    def _generate_state_error(
        self, 
        correct_call: dict, 
        tool_result: dict
    ) -> Optional[GeneratedError]:
        """生成状态错误"""
        tool_name = correct_call['name']
        tool_args = deep_copy(correct_call.get('arguments', {}))
        
        # 从新的错误库结构中获取状态错误
        error_message = self._get_error_from_db_v2(tool_name, ErrorType.STATE_ERROR)
        
        if error_message:
            error_message = fill_template(error_message, tool_args)
            
            return GeneratedError(
                error_type=ErrorType.STATE_ERROR,
                wrong_call=correct_call,  # 状态错误通常参数正确，只是状态不对
                error_message=f"Error: {error_message}",
                original_call=correct_call,
            )
        
        # 使用通用状态错误模板
        templates = STATE_ERROR_TEMPLATES.get(self.domain, ["Operation not allowed in current state"])
        error_message = select_template(templates)
        
        return GeneratedError(
            error_type=ErrorType.STATE_ERROR,
            wrong_call=correct_call,
            error_message=f"Error: {error_message}",
            original_call=correct_call,
        )
    
    def _generate_tool_hallucination(
        self, 
        correct_call: dict, 
        tool_result: dict
    ) -> Optional[GeneratedError]:
        """生成工具幻觉错误"""
        tool_name = correct_call['name']
        tool_args = correct_call['arguments']
        
        # 获取该工具的幻觉映射
        domain_hallucinations = TOOL_HALLUCINATIONS.get(self.domain, {})
        hallucination_tools = domain_hallucinations.get(tool_name, [])
        
        if not hallucination_tools:
            # 生成一个通用的错误工具名
            wrong_tool_name = f"check_{tool_name.replace('get_', '')}"
        else:
            wrong_tool_name = select_template(hallucination_tools)
        
        error_message = f"Tool '{wrong_tool_name}' not found."
        
        return GeneratedError(
            error_type=ErrorType.TOOL_HALLUCINATION,
            wrong_call={"name": wrong_tool_name, "arguments": tool_args},
            error_message=f"Error: {error_message}",
            original_call=correct_call,
        )
    
    def _modify_args_for_business_error(self, args: dict, error_template: str) -> dict:
        """根据业务错误模板修改参数"""
        wrong_args = deep_copy(args)
        error_lower = error_template.lower()
        
        modified = False
        
        if "payment" in error_lower and "amount" in error_lower:
            # 支付金额错误：减少支付金额
            for key in list(wrong_args.keys()):
                if "payment" in key.lower() and isinstance(wrong_args[key], list):
                    for item in wrong_args[key]:
                        if isinstance(item, dict) and 'amount' in item:
                            item['amount'] = int(item['amount'] * 0.8)
                            modified = True
                elif "amount" in key.lower() and isinstance(wrong_args[key], (int, float)):
                    wrong_args[key] = int(wrong_args[key] * 0.8)
                    modified = True
        
        elif "number" in error_lower and "match" in error_lower:
            # 数量不匹配：减少列表长度
            for key in list(wrong_args.keys()):
                if isinstance(wrong_args[key], list) and len(wrong_args[key]) > 1:
                    wrong_args[key] = wrong_args[key][:-1]
                    modified = True
                    break
        
        elif "balance" in error_lower or "insufficient" in error_lower:
            # 余额不足：增加金额
            for key in list(wrong_args.keys()):
                if "amount" in key.lower() and isinstance(wrong_args[key], (int, float)):
                    wrong_args[key] = int(wrong_args[key] * 1.5)
                    modified = True
        
        # 如果没有修改成功，默认修改一个参数
        if not modified and wrong_args:
            key = random.choice(list(wrong_args.keys()))
            wrong_args[key] = corrupt_value(wrong_args[key])
        
        return wrong_args
    
    def _get_error_from_db(self, tool_name: str, error_type: str) -> Optional[str]:
        """从错误库中获取错误消息（旧版本，保留兼容性）"""
        domain_errors = self.error_db.get(self.domain, {})
        tool_errors = domain_errors.get(tool_name, [])
        
        if not tool_errors:
            return None
        
        # 根据错误类型过滤
        if error_type == "parameter":
            keywords = ["not found", "invalid", "missing", "validation"]
            filtered = [e for e in tool_errors if any(kw in e['error'].lower() for kw in keywords)]
        else:
            filtered = tool_errors
        
        if filtered:
            weights = [e['count'] for e in filtered]
            selected = random.choices(filtered, weights=weights)[0]
            return selected['error']
        
        return None
    
    def _get_error_from_db_v2(self, tool_name: str, error_type: ErrorType) -> Optional[str]:
        """从重组后的错误库中获取错误消息"""
        domain_data = self.error_db.get(self.domain, {})
        
        # 检查是否是新格式（按error_type组织）
        if error_type.value in domain_data:
            # 新格式：domain -> error_type -> tool -> errors
            error_type_data = domain_data.get(error_type.value, {})
            tool_errors = error_type_data.get(tool_name, [])
        else:
            # 旧格式：domain -> tool -> errors（回退）
            tool_errors = domain_data.get(tool_name, [])
            if not tool_errors:
                return None
            
            # 根据错误类型过滤
            if error_type == ErrorType.PARAMETER_ERROR:
                keywords = ["not found", "invalid", "missing", "validation"]
                tool_errors = [e for e in tool_errors if any(kw in e.get('error', '').lower() for kw in keywords)]
            elif error_type == ErrorType.STATE_ERROR:
                tool_errors = [e for e in tool_errors if any(kw in e.get('error', '').lower() for kw in STATE_ERROR_KEYWORDS)]
        
        if not tool_errors:
            return None
        
        # 加权选择
        weights = [e.get('count', 1) for e in tool_errors]
        selected = random.choices(tool_errors, weights=weights)[0]
        return selected.get('error', '')
    
    def _modify_args_for_error(self, args: dict, error_message: str) -> dict:
        """根据错误消息修改参数"""
        wrong_args = deep_copy(args)
        error_lower = error_message.lower()
        
        # 如果错误消息包含"not found"，修改ID类参数
        if "not found" in error_lower:
            for key in list(wrong_args.keys()):
                if "id" in key.lower() or "number" in key.lower() or "code" in key.lower():
                    wrong_args[key] = corrupt_value(wrong_args[key])
                    return wrong_args
        
        # 如果错误消息包含"invalid"，修改第一个参数
        if "invalid" in error_lower:
            if wrong_args:
                key = list(wrong_args.keys())[0]
                wrong_args[key] = corrupt_value(wrong_args[key])
                return wrong_args
        
        # 默认：修改第一个参数
        if wrong_args:
            key = list(wrong_args.keys())[0]
            wrong_args[key] = corrupt_value(wrong_args[key])
        
        return wrong_args
