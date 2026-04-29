"""
支持错误注入的编排器

继承 tau2-bench 的 Orchestrator，在工具执行前可能注入错误。
"""

from loguru import logger

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.orchestrator.orchestrator import Orchestrator

from .error_injector import RealErrorInjector, ErrorInjectionConfig


class ErrorInjectionOrchestrator(Orchestrator):
    """支持错误注入的编排器"""
    
    def __init__(
        self,
        error_injection_config: ErrorInjectionConfig,
        **kwargs
    ):
        """
        Args:
            error_injection_config: 错误注入配置
            **kwargs: 传递给父类 Orchestrator 的参数
        """
        super().__init__(**kwargs)
        
        # 创建错误注入器
        self.error_injector = RealErrorInjector(
            error_injection_config,
            self.domain
        )
        
        # 错误计数
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive = 3
        
        logger.info(
            f"Initialized ErrorInjectionOrchestrator for {self.domain}, "
            f"error_injection_enabled={error_injection_config.enabled}"
        )
    
    def get_response(self, message: ToolCall) -> ToolMessage:
        """
        重写 get_response 方法，在工具执行前可能注入错误
        
        Args:
            message: 工具调用消息
            
        Returns:
            工具响应消息（可能是注入的错误）
        """
        # 检查是否应该注入错误
        if self.error_injector.should_inject(message, self.error_count):
            # 检查连续错误限制
            if self.consecutive_errors >= self.max_consecutive:
                # 连续错误太多，这次不注入
                logger.debug(
                    f"Skipping error injection due to consecutive error limit "
                    f"({self.consecutive_errors} >= {self.max_consecutive})"
                )
                self.consecutive_errors = 0
                return super().get_response(message)
            
            # 注入错误
            self.error_count += 1
            self.consecutive_errors += 1
            error_response = self.error_injector.generate_error(message)
            
            logger.info(
                f"[ERROR INJECTION #{self.error_count}] "
                f"Tool: {message.name}, "
                f"Error: {error_response.content[:100]}"
            )
            
            return error_response
        else:
            # 正常执行
            self.consecutive_errors = 0
            return super().get_response(message)
