"""
思维链注入 Agent

在关键决策点注入推理过程，帮助模型更好地处理复杂任务。
"""

from typing import List, Optional, Dict

from tau2.agent.llm_agent import LLMAgent
from tau2.environment.tool import Tool


class ChainOfThoughtAgent(LLMAgent):
    """在关键决策点注入思维链推理的 Agent"""
    
    # 针对弱点工具的推理提示模板
    COT_PROMPTS: Dict[str, str] = {
        # Airline 领域
        "search_direct_flight": """
When searching for flights, verify:
1. Departure city (user's origin)
2. Arrival city (user's destination)  
3. Travel date (exact date from conversation)
4. Preferences (time, airline, direct flight)""",

        "book_reservation": """
Before booking, confirm:
1. Flight details match user's request
2. Passenger information is correct
3. Payment method is user's preferred choice
4. All required fields are filled""",

        "update_reservation_flights": """
Before updating reservation:
1. Verify reservation ID is correct
2. Confirm current flight details
3. Identify new flight requirements
4. Check payment method for fare difference""",

        "update_reservation_baggages": """
Before updating baggage:
1. Get current reservation details
2. Calculate baggage fees correctly
3. Verify payment method
4. Confirm total cost with user""",

        # Retail 领域
        "exchange_delivered_order_items": """
Before exchanging items:
1. Verify order_id contains the item to exchange
2. Identify correct item_ids from that order
3. Find valid new_item_ids (product IDs user wants)
4. Confirm payment method for price difference""",

        "return_delivered_order_items": """
Before returning items:
1. Verify order is delivered (not pending)
2. Identify correct item_ids to return
3. Confirm return reason with user
4. Check refund method""",

        "modify_pending_order_items": """
Before modifying order:
1. Verify order is still 'pending' status
2. Identify which items to modify
3. Confirm new quantities or items
4. Verify payment method from user's saved methods""",

        "get_order_details": """
Before querying order:
1. Identify the correct order_id from context
2. If user has multiple orders, clarify which one
3. Note the order status for next steps""",

        # Telecom 领域
        "refuel_data": """
Before refueling data:
1. Check current data balance
2. Verify refuel amount requested
3. Confirm payment method
4. Calculate total cost""",

        "upgrade_plan": """
Before upgrading plan:
1. Get current plan details
2. Compare with requested plan
3. Calculate price difference
4. Confirm user agreement""",
    }
    
    # 通用推理指令
    REASONING_INSTRUCTION = """
## Reasoning Protocol

Before executing any tool call, especially for complex operations:

1. **Verify Information**: Ensure you have all required parameters from the conversation
2. **Check IDs**: Double-check order_id, item_id, flight_id, reservation_id, etc.
3. **Confirm Intent**: Make sure your planned action matches user's request
4. **Validate Parameters**: Cross-reference parameter values with conversation context

For multi-step tasks:
- Plan the complete sequence before starting
- Execute steps in logical order
- Verify each step's result before proceeding

Common mistakes to avoid:
- Using wrong IDs (order vs item vs product)
- Selecting incorrect payment methods
- Missing required steps in a workflow
- Confusing multiple orders or items
"""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        analysis: Optional[str] = None,
        enable_cot: bool = True,
        cot_tools: Optional[List[str]] = None,
    ):
        """
        Args:
            tools: 可用工具列表
            domain_policy: 领域策略
            llm: LLM 模型名称
            llm_args: LLM 参数
            analysis: 上一轮失败分析（用于迭代学习）
            enable_cot: 是否启用思维链
            cot_tools: 需要 CoT 的工具列表（None 表示使用默认列表）
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self._analysis = analysis or ""
        self.enable_cot = enable_cot
        self.cot_tools = cot_tools or list(self.COT_PROMPTS.keys())
        self.original_system_prompt = self._build_base_prompt()
    
    def _build_base_prompt(self) -> str:
        """构建基础系统提示"""
        from tau2.agent.llm_agent import SYSTEM_PROMPT, AGENT_INSTRUCTION
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
    
    @property
    def system_prompt(self) -> str:
        """构建增强的系统提示"""
        base = self.original_system_prompt
        
        parts = [base]
        
        # 添加推理指令
        if self.enable_cot:
            parts.append(self.REASONING_INSTRUCTION)
            
            # 添加工具特定的推理提示
            tool_hints = self._build_tool_hints()
            if tool_hints:
                parts.append("\n## Tool-Specific Guidelines\n" + tool_hints)
        
        # 添加失败分析（如果有）
        if self._analysis:
            parts.append(
                "\n## Previous Analysis\n"
                + self._analysis
                + "\n\nPlease follow the above improvement suggestions for this attempt."
            )
        
        return "\n".join(parts)
    
    def _build_tool_hints(self) -> str:
        """构建工具特定的提示"""
        # 获取当前可用的工具名称
        available_tools = {tool.name for tool in self.tools}
        
        hints = []
        for tool_name in self.cot_tools:
            if tool_name in available_tools and tool_name in self.COT_PROMPTS:
                hints.append(f"**{tool_name}**:{self.COT_PROMPTS[tool_name]}")
        
        return "\n\n".join(hints)
    
    def get_cot_prompt_for_tool(self, tool_name: str) -> str:
        """获取特定工具的思维链提示"""
        return self.COT_PROMPTS.get(tool_name, "")


class AnalysisCoTAgent(ChainOfThoughtAgent):
    """
    结合分析注入和思维链的 Agent
    
    用于迭代学习场景：失败后注入分析 + CoT 推理
    """
    
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        analysis: Optional[str] = None,
        enable_cot: bool = True,
        weak_tools: Optional[List[str]] = None,
    ):
        """
        Args:
            weak_tools: 当前任务涉及的弱点工具，用于定向 CoT
        """
        # 如果指定了弱点工具，只对这些工具启用 CoT
        cot_tools = weak_tools if weak_tools else None
        
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            analysis=analysis,
            enable_cot=enable_cot,
            cot_tools=cot_tools,
        )
