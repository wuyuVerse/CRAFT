"""
模板管理模块

集中管理所有错误消息和恢复思考的模板。
"""

from .error_templates import (
    ERROR_MESSAGE_TEMPLATES,
    BUSINESS_LOGIC_ERRORS,
    TOOL_HALLUCINATIONS,
    STATE_ERROR_TEMPLATES,
    STATE_ERROR_KEYWORDS,
)

from .recovery_templates import (
    RECOVERY_TEMPLATES,
    RECOVERY_GENERATION_PROMPT,
    ERROR_TYPE_GUIDANCE,
)

from .template_utils import (
    fill_template,
    select_template,
    format_template_safe,
)

__all__ = [
    "ERROR_MESSAGE_TEMPLATES",
    "BUSINESS_LOGIC_ERRORS",
    "TOOL_HALLUCINATIONS",
    "STATE_ERROR_TEMPLATES",
    "STATE_ERROR_KEYWORDS",
    "RECOVERY_TEMPLATES",
    "RECOVERY_GENERATION_PROMPT",
    "ERROR_TYPE_GUIDANCE",
    "fill_template",
    "select_template",
    "format_template_safe",
]
