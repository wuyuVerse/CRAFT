"""
错误生成Prompt模板

用于LLM生成各类错误的prompt。
"""

ERROR_GENERATION_PROMPT = """You are an expert at generating realistic errors for tool-calling agents in the {domain} domain.

# Task
Generate a realistic **{error_type}** for the tool call below, based on real error examples from production data.

# Context
The agent is having a conversation with a user. Here's the recent conversation history:

{conversation_history}

# Current Tool Call (Correct)
**Tool Name:** {tool_name}
**Arguments:** {tool_arguments}
**Expected Result:** {tool_result}

# Error Examples from Production Data
{error_examples}

# Error Type Guidelines
{error_type_guidance}

# Output Requirements
Generate a JSON object with the following structure:
```json
{{
  "wrong_call": {{
    "name": "tool_name",
    "arguments": {{...}}
  }},
  "error_message": "Error: ..."
}}
```

**Important Rules:**
1. The error must be realistic and follow the patterns in the examples
2. For parameter_error: Keep the format correct (e.g., L1001 not L10), only change values
3. For state_error: Keep arguments unchanged, error is due to state
4. For tool_hallucination: Use a non-existent but plausible tool name
5. The error_message must start with "Error: "
6. The error should be recoverable with proper error handling

Generate the error now (JSON only, no explanation):"""


ERROR_TYPE_GUIDANCE_MAP = {
    "parameter_error": """**Parameter Error:**
- Modify one or more argument values to be invalid
- Keep the parameter format correct (e.g., IDs should look like IDs)
- Common causes: wrong ID, typo in name, invalid value
- Example: user_id "john_doe_123" → "john_doe_999" (not found)""",
    
    "business_logic_error": """**Business Logic Error:**
- Arguments may be individually valid but violate business rules
- Common causes: insufficient balance, quantity mismatch, policy violation
- Example: payment amount doesn't match total, not enough items""",
    
    "state_error": """**State Error:**
- Arguments are correct but operation not allowed in current state
- Keep the arguments unchanged
- Common causes: order already cancelled, booking already confirmed
- Example: trying to cancel an already cancelled order""",
    
    "tool_hallucination": """**Tool Hallucination:**
- Use a non-existent tool name that sounds plausible
- Keep the arguments unchanged
- Common patterns: check_* instead of get_*, verify_* instead of validate_*
- Example: "check_flight_status" instead of "get_flight_status" """
}


def format_conversation_history(context: list, max_turns: int = 3) -> str:
    """格式化对话历史"""
    if not context:
        return "(No previous conversation)"
    
    # 只取最近的几轮对话
    recent = context[-max_turns*2:] if len(context) > max_turns*2 else context
    
    formatted = []
    for msg in recent:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # 截断过长的内容
        if len(content) > 200:
            content = content[:200] + "..."
        
        formatted.append(f"**{role.upper()}:** {content}")
    
    return "\n".join(formatted)


def format_error_examples(examples: list) -> str:
    """格式化错误样例"""
    if not examples:
        return "(No examples available - generate based on error type guidelines)"
    
    formatted = []
    for i, example in enumerate(examples, 1):
        formatted.append(f"{i}. `{example}`")
    
    return "\n".join(formatted)
