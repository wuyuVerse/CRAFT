"""
错误注入模块

提供基于真实错误库的错误注入功能。

V3: RuleBasedErrorInjector - 基于规则的轨迹错误注入
V4: AgentBasedErrorInjector - 基于Agent的错误生成和恢复
"""

from .error_injector import RealErrorInjector, ErrorInjectionConfig
from .orchestrator import ErrorInjectionOrchestrator
from .validator import TrajectoryValidator

# 延迟导入injectors以避免循环导入
def _get_injectors():
    from .injectors import AgentBasedErrorInjector, RuleBasedErrorInjector
    return AgentBasedErrorInjector, RuleBasedErrorInjector

# 向后兼容的别名
def __getattr__(name):
    if name in ('AgentBasedErrorInjector', 'RuleBasedErrorInjector', 'TrajectoryErrorInjector'):
        AgentBasedErrorInjector, RuleBasedErrorInjector = _get_injectors()
        if name == 'AgentBasedErrorInjector':
            return AgentBasedErrorInjector
        elif name == 'RuleBasedErrorInjector':
            return RuleBasedErrorInjector
        elif name == 'TrajectoryErrorInjector':
            return RuleBasedErrorInjector  # 向后兼容别名
    
    # 从agents模块导入（向后兼容）
    if name in ('ErrorGenerationAgent', 'ErrorType', 'GeneratedError', 
                'RecoveryGenerationAgent', 'RecoveryResponse'):
        from ..agents import (
            ErrorGenerationAgent,
            ErrorType,
            GeneratedError,
            RecoveryGenerationAgent,
            RecoveryResponse,
        )
        return locals()[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # 基础设施
    'RealErrorInjector',
    'ErrorInjectionConfig',
    'ErrorInjectionOrchestrator',
    'TrajectoryValidator',
    # V3 注入器
    'RuleBasedErrorInjector',
    'TrajectoryErrorInjector',  # 向后兼容别名
    # V4 注入器
    'AgentBasedErrorInjector',
    # V4 Agents（向后兼容导出）
    'ErrorGenerationAgent',
    'ErrorType',
    'GeneratedError',
    'RecoveryGenerationAgent',
    'RecoveryResponse',
]

