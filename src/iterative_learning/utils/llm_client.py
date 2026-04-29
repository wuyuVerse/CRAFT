"""
统一的LLM客户端

封装所有LLM调用逻辑，提供统一的接口。
"""

from typing import Optional, List, Dict
from loguru import logger


class LLMClient:
    """统一的LLM客户端"""
    
    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout: int = 30,
    ):
        """
        Args:
            model: 模型名称（支持 "openai/xxx" 格式）
            api_base: API基础URL
            api_key: API密钥
            timeout: 超时时间（秒）
        """
        self.model = self._normalize_model_name(model)
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
    
    def _normalize_model_name(self, model: str) -> str:
        """
        标准化模型名称
        
        将 "openai/xxx" 格式转换为 "xxx"
        """
        if model.startswith("openai/"):
            return model[7:]  # 移除 "openai/" 前缀
        return model
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Optional[str]:
        """
        调用LLM进行对话补全
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            LLM响应内容，失败返回None
        """
        try:
            import openai
            
            client = openai.OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # 安全获取响应内容
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    return message.content
            
            logger.warning("LLM returned empty response")
            return None
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def generate_from_prompt(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Optional[str]:
        """
        从单个prompt生成响应
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            LLM响应内容，失败返回None
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, temperature, max_tokens)


def create_llm_client(
    model: str,
    api_base: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    创建LLM客户端的工厂函数
    
    Args:
        model: 模型名称
        api_base: API基础URL
        **kwargs: 其他参数
        
    Returns:
        LLMClient实例
    """
    return LLMClient(model=model, api_base=api_base, **kwargs)
