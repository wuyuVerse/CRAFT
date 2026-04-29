import openai
from enum import Enum
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from dataclasses import dataclass
from typing import Any, Optional
import json
from ..utils.logger import log
from ..utils.json_utils import safe_json_loads
"""
在蒸馏cot中要使用的agents
"""

@dataclass
class ResponseBase:
    content: str

@dataclass
class UserResponse(ResponseBase):
    signal: bool
    tool_calls: list = None

@dataclass
class ToolResponse(ResponseBase):
    name: str
    tool_call_id: Any

@dataclass
class AgentResponse(ResponseBase):
    reasoning_content: str
    tool_calls: list

class UserSimulator:
    def __init__(self, prompt, client, model, tools=None):
        self.prompt = prompt
        self.model = model
        self.client = client
        self.tools = tools  # 支持工具调用（如telecom领域）

    def _IsStop(self, response: str):
        # 检查多种停止信号格式
        stop_signals = ["[STOP]", "###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]
        return any(signal in response for signal in stop_signals)
    
    async def generate(self, history:list) -> UserResponse | AgentResponse:
        """
        生成用户回复，如果配置了tools则可能返回tool_calls
        """
        messages = history
        # 如果有tools，传递给API（类似ToolAgent）
        if self.tools:
            response = await self.client.chat.completions.create(
                messages = messages,
                model = self.model,
                tools=self.tools,
                extra_body = {'temperature':1.0}
            )
            content = response.choices[0].message.content or ""
            tool_calls = response.choices[0].message.tool_calls
            
            return UserResponse(content, self._IsStop(content), tool_calls)
        else:
            # 没有tools，正常生成文本
            response = await self.client.chat.completions.create(
                messages = messages,
                model = self.model,
                extra_body = {'temperature':1.0}
            )
            content = response.choices[0].message.content or ""
            return UserResponse(content, self._IsStop(content), None)

class ToolAgent:
    def __init__(self, prompt, client, model, tools, validator=None):
        self.prompt = prompt
        self.model = model
        self.client = client
        self.tools = tools
        self.validator = validator  # MultiAgentValidator实例

    def _convert_to_toolcall_objects(self, toolcall_list: list) -> list:
        """
        将validator返回的字典列表转换为OpenAI ChatCompletionMessageToolCall对象
        
        Args:
            toolcall_list: validator返回的toolcall列表，格式为:
                [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
        
        Returns:
            list of ChatCompletionMessageToolCall objects
        """
        if not toolcall_list:
            return []
        
        result = []
        for tc in toolcall_list:
            # 创建 Function 对象
            func = Function(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"]
            )
            # 创建 ChatCompletionMessageToolCall 对象
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tc["id"],
                type=tc["type"],
                function=func
            )
            result.append(tool_call_obj)
        
        return result

    async def generate(self, history:list) -> AgentResponse: 
        messages = history
        response = await self.client.chat.completions.create(
            messages = messages,
            model = self.model,
            tools=self.tools,
            extra_body = {'temperature':1.0, "top_p":0.95, "top_k":40}
        )
        content = response.choices[0].message.content or ""
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None) or ""
        tool_calls = response.choices[0].message.tool_calls or []
        
        # 如果有tool_calls且配置了validator，进行验证
        if tool_calls and self.validator:
            # 将tool_calls转换为JSON字符串供validator验证
            tool_calls_dicts = []
            for tc in tool_calls:
                tc_dict = {
                    "id": tc.id,
                    "type": tc.type,
                }
                # 处理function属性，使用getattr避免类型检查错误
                func = getattr(tc, 'function', None)
                if func:
                    tc_dict["function"] = {  # type: ignore
                        "name": func.name,
                        "arguments": func.arguments
                    }
                tool_calls_dicts.append(tc_dict)
            
            tool_calls_json = json.dumps(tool_calls_dicts, ensure_ascii=False)
            
            # 构建instruction（对话格式）
            instruction = json.dumps([
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": messages[-1]["content"] if messages else ""}
            ], ensure_ascii=False)
            
            # 验证
            validation_result = await self.validator.validate(
                instruction=instruction,
                answer=tool_calls_json,
                auto_correct=True
            )
            
            # 如果验证不通过且有修正结果，使用修正后的tool_calls
            if not validation_result.final_decision and validation_result.corrected_answer:
                # 将修正后的字典列表转换为ChatCompletionMessageToolCall对象
                tool_calls = self._convert_to_toolcall_objects(validation_result.corrected_answer)
        
        return AgentResponse(
            content = content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls
        )

class ToolSimulator:
    def __init__(self, prompt, client, model, tools=None, user_info=None, task_data=None, max_retry=3):
        self.prompt = prompt
        self.model = model
        self.client = client
        self.tools = tools
        self.user_info = user_info
        self.task_data = task_data
        self.max_retry = max_retry

    async def generate(self, tool_call, toolcall_id, history) -> ToolResponse | None: 
        """
        根据工具调用生成模拟响应
        tool_call: OpenAI toolcall 对象
        如果JSON解析失败，重试直到max_retry次，仍失败则返回None
        """
        system = self.prompt
        conversations = []
        for h in history:
            if 'tool_calls' in h:
                # 将 OpenAI toolcall 对象转换为可序列化的字典
                serialized_tool_calls = []
                for tc in h['tool_calls']:
                    serialized_tool_calls.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                # 创建新的历史记录条目，替换 tool_calls
                new_h = {k: v for k, v in h.items() if k != 'tool_calls'}
                new_h['tool_calls'] = serialized_tool_calls
                conversations.append(new_h)
            else:
                conversations.append(h)
        conversations.pop()
        # 将当前 tool_call 也转换为字典格式
        tool_call_dict = {
            "name": tool_call['name'],
            "arguments": tool_call['arguments']
        }
                    
        user_input = {
            "available_tools": self.tools, 
            "conversations": conversations, 
            "new_tool_call": tool_call_dict,
            "user_info": self.user_info,
            "task_data": self.task_data
        }
        messages = [
            {'role': 'system', 'content': system}, 
            {'role': 'user', 'content': json.dumps(user_input, ensure_ascii=False)}
        ]
        
        # 重试机制：确保返回结果能被JSON解析
        response_content = None
        for attempt in range(self.max_retry):
            try:
                response = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    extra_body={'temperature': 1.0, "top_p": 0.95, "top_k": 40},
                    response_format={"type": "json_object"}
                )
                response_content = response.choices[0].message.content
                
                if not response_content:
                    log(f"[ToolSimulator] 第{attempt + 1}次尝试: 响应内容为空，重试...")
                    continue
                
                # 尝试解析JSON
                parsed_content = safe_json_loads(response_content)
                
                # 解析成功，返回结果
                return ToolResponse(
                    name=tool_call['name'],
                    content=json.dumps(parsed_content),
                    tool_call_id=toolcall_id
                )
                
            except json.JSONDecodeError as e:
                log(f"[ToolSimulator] 第{attempt + 1}次尝试: JSON解析失败 - {str(e)}")
                if attempt < self.max_retry - 1:
                    if response_content:
                        log(f"[ToolSimulator] 响应内容: {response_content[:200]}...")
                    log(f"[ToolSimulator] 重试...")
                    continue
                else:
                    log(f"[ToolSimulator] 达到最大重试次数({self.max_retry})，工具模拟失败")
                    return None
            except Exception as e:
                log(f"[ToolSimulator] 第{attempt + 1}次尝试: 未知错误 - {str(e)}")
                if attempt < self.max_retry - 1:
                    continue
                else:
                    log(f"[ToolSimulator] 达到最大重试次数({self.max_retry})，工具模拟失败")
                    return None
        
        return None

class UserResponseChecker:
    """
    用户回复检查器：验证user simulator的回复是否符合人类习惯
    检查项：
    1. 无特殊字符
    2. 无奇怪偏好
    3. 自然语言表达
    4. 不泄露技术细节
    5. 与用户信息一致
    """
    def __init__(self, prompt: str, api_key: str, base_url: str, model: str):
        self.prompt = prompt
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    async def check(self, user_info: dict, response: str) -> dict:
        """
        检查用户回复是否符合人类习惯，如果不符合则自动修正
        
        Args:
            user_info: 用户信息字典（包含user_profile等）
            response: 用户回复内容
        
        Returns:
            dict: {"is_valid": bool, "reason": str, "corrected_response": str}
        """
        user_input = {
            "user_info": user_info,
            "response": response
        }
        
        messages: list = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}
        ]
        
        result_text = ""
        try:
            completion = await self.client.chat.completions.create(
                messages=messages,  # type: ignore
                model=self.model,
                response_format={"type": "json_object"},
                extra_body={"temperature": 0.3}  # 使用较低温度以获得更稳定的验证结果
            )
            
            result_text = completion.choices[0].message.content or ""
            # 尝试解析JSON响应
            result = safe_json_loads(result_text, default={})
            
            return {
                "is_valid": result.get("is_valid", True),
                "reason": result.get("reason", ""),
                "corrected_response": result.get("corrected_response", "")
            }
        except (json.JSONDecodeError, ValueError):
            # 如果解析失败，默认认为有效
            log(f"[警告] 用户回复检查器返回了无效的JSON: {result_text}")
            return {"is_valid": True, "reason": "", "corrected_response": ""}
        except Exception as e:
            log(f"[错误] 用户回复检查失败: {str(e)}")
            return {"is_valid": True, "reason": "", "corrected_response": ""}