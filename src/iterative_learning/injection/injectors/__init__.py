"""
错误注入器模块

包含V3规则注入器和V4 Agent注入器。
"""

from .agent_based_injector import AgentBasedErrorInjector
from .rule_based_injector import RuleBasedErrorInjector

__all__ = [
    "AgentBasedErrorInjector",
    "RuleBasedErrorInjector",
]
