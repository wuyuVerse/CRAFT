"""
恢复思考模板

集中管理所有错误恢复的思考模板。
"""

# 恢复思考模板（用于规则生成）
RECOVERY_TEMPLATES = {
    "parameter_error": [
        "I see there was an issue with the {param} parameter. The value '{wrong_value}' was incorrect. Let me correct it to '{correct_value}' and try again.",
        "The {param} I provided was invalid. Looking at the error, I need to use '{correct_value}' instead of '{wrong_value}'.",
        "I made an error with the {param}. Let me fix this by using the correct value '{correct_value}'.",
    ],
    "business_logic_error": [
        "I see the {error_type}. Let me recalculate and provide the correct values.",
        "The system indicates {error_type}. I need to adjust my parameters to meet the business requirements.",
        "There's a validation error: {error_type}. Let me correct the values and retry.",
    ],
    "state_error": [
        "The operation failed because {error_type}. I need to verify the current state before proceeding.",
        "I see that {error_type}. Let me check the status and use the appropriate action.",
        "The system indicates {error_type}. I should verify the state first.",
    ],
    "tool_hallucination": [
        "I apologize, I used the wrong tool name. The correct tool is '{correct_tool}', not '{wrong_tool}'. Let me try again.",
        "I see that '{wrong_tool}' doesn't exist. I should use '{correct_tool}' instead.",
        "The tool '{wrong_tool}' is not available. Let me use the correct tool '{correct_tool}'.",
    ],
}

# LLM生成恢复的Prompt模板
RECOVERY_GENERATION_PROMPT = """# Error Recovery Task

You are an AI assistant that just encountered a tool call error. Analyze the error and generate a recovery response.

## Recent Conversation Context

{context}

## Your Failed Tool Call

```json
{wrong_call}
```

## Error Message from System

{error_message}

## Correct Tool Call (Reference)

Tool: {correct_tool}
Arguments: {correct_args}

## Error Type Guidance

{guidance}

## Output Requirements

Generate a natural recovery response that includes:
1. A brief error analysis (1-2 sentences explaining what went wrong)
2. The correct tool call

Format:
```
[Your error analysis and recovery thinking - be concise and natural]

<tool_call>
{{"name": "correct_tool_name", "arguments": {{correct_arguments}}}}
</tool_call>
```

Important:
1. Keep the analysis brief and natural, like a real AI assistant
2. Do NOT say "according to reference" or "the correct answer is"
3. Show that YOU discovered and fixed the error yourself
4. The tool call MUST be wrapped in <tool_call> tags
5. Write in the same language as the conversation context (if context is in Chinese, respond in Chinese)
"""

# 错误类型指导
ERROR_TYPE_GUIDANCE = {
    "parameter_error": "The error is due to an incorrect parameter value. Identify which parameter was wrong and explain how you'll fix it.",
    "business_logic_error": "The error is due to a business rule violation. Explain what rule was violated and how you'll correct the values.",
    "state_error": "The error is due to an invalid state. Acknowledge the state issue and explain your next step.",
    "tool_hallucination": "You used a non-existent tool name. Identify the correct tool name and use it.",
}
