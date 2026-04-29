"""
Token成本计算器 - 用于估算数据生成的成本
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    """Token使用量"""
    input_tokens: int = 0
    output_tokens: int = 0
    
    def total(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def __add__(self, other):
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens
        )


@dataclass
class CostBreakdown:
    """成本分解"""
    component: str
    input_tokens: int
    output_tokens: int
    cost: float
    calls: int = 1


class CostCalculator:
    """成本计算器"""
    
    # Claude 3.5 Sonnet定价 (per 1M tokens)
    CLAUDE_PRICING = {
        "input": 3.0,   # $3 per 1M input tokens
        "output": 15.0  # $15 per 1M output tokens
    }
    
    # OpenAI GPT-4o定价 (per 1M tokens)
    GPT4O_PRICING = {
        "input": 2.5,   # $2.5 per 1M input tokens
        "output": 10.0  # $10 per 1M output tokens
    }
    
    # 本地模型（Qwen）- 免费
    QWEN_PRICING = {
        "input": 0.0,
        "output": 0.0
    }
    
    def __init__(self, model: str = "claude"):
        """
        初始化成本计算器
        
        Args:
            model: 使用的模型 (claude/gpt4o/qwen)
        """
        self.model = model
        if model == "claude":
            self.pricing = self.CLAUDE_PRICING
        elif model == "gpt4o":
            self.pricing = self.GPT4O_PRICING
        elif model == "qwen":
            self.pricing = self.QWEN_PRICING
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # 统计信息
        self.total_usage = TokenUsage()
        self.breakdown: List[CostBreakdown] = []
        self.task_count = 0
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        计算成本
        
        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数
        
        Returns:
            成本（美元）
        """
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        return input_cost + output_cost
    
    def add_usage(
        self, 
        component: str, 
        input_tokens: int, 
        output_tokens: int,
        calls: int = 1
    ):
        """
        添加使用量
        
        Args:
            component: 组件名称
            input_tokens: 输入token数
            output_tokens: 输出token数
            calls: 调用次数
        """
        usage = TokenUsage(input_tokens, output_tokens)
        self.total_usage = self.total_usage + usage
        
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        self.breakdown.append(CostBreakdown(
            component=component,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            calls=calls
        ))
    
    def estimate_single_task_cost(self) -> Dict:
        """
        估算生成单个任务的成本
        
        基于典型的token使用量估算
        """
        # 典型token使用量估算
        estimates = {
            "task_generation": {
                "input": 2000,   # API列表 + prompt
                "output": 800,   # user_info + task_data
                "calls": 1
            },
            "evaluation_criteria": {
                "input": 3000,   # task + user_info + APIs
                "output": 1000,  # detailed criteria
                "calls": 1
            },
            "user_simulator_prompt": {
                "input": 2500,
                "output": 600,
                "calls": 1
            },
            "conversation_simulation": {
                "input": 1500,   # Per turn
                "output": 400,   # Per turn
                "calls": 10      # 平均10轮对话
            },
            "tool_simulation": {
                "input": 2000,   # Per tool call
                "output": 300,   # Tool response
                "calls": 3       # 平均3次tool call
            },
            "quality_check": {
                "input": 5000,   # Full conversation
                "output": 500,   # Quality assessment
                "calls": 1
            },
            "reward_calculation": {
                "input": 4000,   # Conversation + criteria
                "output": 800,   # Detailed reward analysis
                "calls": 2       # ACTION + COMMUNICATE checks
            }
        }
        
        total_cost = 0.0
        details = []
        
        for component, usage in estimates.items():
            total_input = usage["input"] * usage["calls"]
            total_output = usage["output"] * usage["calls"]
            cost = self.calculate_cost(total_input, total_output)
            total_cost += cost
            
            details.append({
                "component": component,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "calls": usage["calls"],
                "cost": cost
            })
        
        return {
            "model": self.model,
            "total_cost": total_cost,
            "total_input_tokens": sum(d["input_tokens"] for d in details),
            "total_output_tokens": sum(d["output_tokens"] for d in details),
            "breakdown": details
        }
    
    def get_summary(self) -> Dict:
        """获取成本摘要"""
        total_cost = sum(b.cost for b in self.breakdown)
        
        return {
            "model": self.model,
            "total_tasks": self.task_count,
            "total_input_tokens": self.total_usage.input_tokens,
            "total_output_tokens": self.total_usage.output_tokens,
            "total_tokens": self.total_usage.total(),
            "total_cost": total_cost,
            "cost_per_task": total_cost / self.task_count if self.task_count > 0 else 0,
            "breakdown": [
                {
                    "component": b.component,
                    "input_tokens": b.input_tokens,
                    "output_tokens": b.output_tokens,
                    "calls": b.calls,
                    "cost": b.cost,
                    "percentage": (b.cost / total_cost * 100) if total_cost > 0 else 0
                }
                for b in self.breakdown
            ]
        }
    
    def print_estimate(self, num_tasks: int = 1):
        """
        打印成本估算
        
        Args:
            num_tasks: 任务数量
        """
        estimate = self.estimate_single_task_cost()
        
        print("=" * 70)
        print(f"成本估算 - {self.model.upper()} 模型")
        print("=" * 70)
        print(f"\n单个任务估算:")
        print(f"  输入Tokens:  {estimate['total_input_tokens']:,}")
        print(f"  输出Tokens:  {estimate['total_output_tokens']:,}")
        print(f"  总Tokens:    {estimate['total_input_tokens'] + estimate['total_output_tokens']:,}")
        print(f"  单任务成本:  ${estimate['total_cost']:.4f}")
        
        print(f"\n{num_tasks}个任务估算:")
        print(f"  总Tokens:    {(estimate['total_input_tokens'] + estimate['total_output_tokens']) * num_tasks:,}")
        print(f"  总成本:      ${estimate['total_cost'] * num_tasks:.2f}")
        
        print(f"\n成本分解:")
        for item in estimate['breakdown']:
            pct = (item['cost'] / estimate['total_cost'] * 100)
            print(f"  {item['component']:30s} ${item['cost']:.4f} ({pct:5.1f}%) "
                  f"[{item['calls']}次调用]")
        
        print("\n" + "=" * 70)
        print(f"模型定价:")
        print(f"  输入:  ${self.pricing['input']}/1M tokens")
        print(f"  输出:  ${self.pricing['output']}/1M tokens")
        print("=" * 70)


def compare_models(num_tasks: int = 100):
    """
    比较不同模型的成本
    
    Args:
        num_tasks: 任务数量
    """
    print("\n" + "=" * 70)
    print(f"多模型成本对比 - {num_tasks}个任务")
    print("=" * 70)
    
    models = ["claude", "gpt4o", "qwen"]
    results = []
    
    for model in models:
        calc = CostCalculator(model)
        estimate = calc.estimate_single_task_cost()
        total_cost = estimate['total_cost'] * num_tasks
        results.append({
            "model": model,
            "cost_per_task": estimate['total_cost'],
            "total_cost": total_cost
        })
    
    print(f"\n{'模型':<15} {'单任务成本':>15} {f'{num_tasks}任务总成本':>20}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<15} ${r['cost_per_task']:>14.4f} ${r['total_cost']:>19.2f}")
    
    print("\n建议:")
    cheapest = min(results, key=lambda x: x['total_cost'])
    print(f"  💰 最便宜: {cheapest['model'].upper()} (${cheapest['total_cost']:.2f})")
    
    if cheapest['model'] == 'qwen':
        print(f"  ⚡ Qwen是本地模型，完全免费但需要GPU资源")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据生成成本计算器')
    parser.add_argument('--model', type=str, default='claude',
                       choices=['claude', 'gpt4o', 'qwen'],
                       help='模型类型')
    parser.add_argument('--num-tasks', type=int, default=100,
                       help='任务数量')
    parser.add_argument('--compare', action='store_true',
                       help='比较不同模型的成本')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.num_tasks)
    else:
        calculator = CostCalculator(args.model)
        calculator.print_estimate(args.num_tasks)

