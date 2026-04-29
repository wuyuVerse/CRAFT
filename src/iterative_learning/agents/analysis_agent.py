"""
带分析注入功能的 LLM Agent

在 system prompt 中附加上一轮的失败分析，帮助 Agent 在重试时改进。
"""

from tau2.agent.llm_agent import LLMAgent


class AnalysisLLMAgent(LLMAgent):
    """
    在 system prompt 里附加上一轮的分析。
    
    用于迭代学习场景：当任务失败时，将失败分析注入到下一次尝试的 prompt 中。
    """

    def __init__(self, *args, analysis: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_analysis = analysis or ""
        self.original_system_prompt = super().system_prompt

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        if not self._extra_analysis:
            return base
        return (
            base
            + "\n\n# Previous Analysis\n"
            + self._extra_analysis
            + "\n# Please follow the above improvement suggestions for this attempt."
        )
