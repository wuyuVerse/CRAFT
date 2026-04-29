"""
任务难度分类器

根据任务特征和历史失败模式分类任务难度，用于定向数据增强。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from tau2.data_model.tasks import Task


@dataclass
class TaskDifficulty:
    """任务难度特征"""
    task_id: str
    domain: str
    is_multi_step: bool           # 需要多个工具调用
    is_parameter_sensitive: bool  # 参数选择敏感
    is_context_complex: bool      # 上下文复杂（多订单/商品）
    priority_weight: float        # 采样权重 (1.0-5.0)
    weak_tools: List[str] = field(default_factory=list)  # 涉及的弱点工具
    expected_actions: List[str] = field(default_factory=list)


class TaskDifficultyClassifier:
    """根据任务特征和历史失败模式分类难度"""
    
    # 基于评测报告的弱点工具（按失败次数排序）
    WEAK_TOOLS: Dict[str, List[str]] = {
        "airline": [
            "search_direct_flight",       # 14次缺失 - 最严重
            "update_reservation_flights", # 6次缺失
            "cancel_reservation",         # 6次缺失
            "book_reservation",           # 5次缺失
            "send_certificate",           # 金额计算错误
        ],
        "retail": [
            "exchange_delivered_order_items",  # 7次参数错误 - 最严重
            "get_order_details",               # 8次查询错误
            "modify_pending_order_items",      # 5次缺失+4次参数错误
            "return_delivered_order_items",    # 3次参数错误
            "get_product_details",             # 3次缺失
            "modify_user_address",             # 3次缺失
        ],
        "telecom": [
            "enable_roaming",    # 2次
            "refuel_data",       # 1次
        ],
        "mock": [],
    }
    
    # 参数敏感工具（需要精确选择ID、金额等）
    PARAMETER_SENSITIVE_TOOLS: Dict[str, List[str]] = {
        "airline": [
            "search_direct_flight",
            "book_reservation", 
            "update_reservation_flights",
            "update_reservation_baggages",
            "send_certificate",
        ],
        "retail": [
            "exchange_delivered_order_items",
            "return_delivered_order_items",
            "modify_pending_order_items",
            "modify_pending_order_address",
        ],
        "telecom": [
            "refuel_data",
            "upgrade_plan",
        ],
        "mock": [],
    }
    
    # 上下文复杂度关键词
    CONTEXT_COMPLEXITY_KEYWORDS = [
        "multiple", "several", "both", "all", "each",
        "two", "three", "different", "various",
    ]
    
    def __init__(self, failure_history: Optional[Dict[str, dict]] = None):
        """
        Args:
            failure_history: 历史失败记录 {task_id: {"fail_rate": float, "fail_tools": [str]}}
        """
        self.failure_history = failure_history or {}
    
    def classify(self, task: Task, domain: str) -> TaskDifficulty:
        """
        分类任务难度
        
        Args:
            task: 任务对象
            domain: 领域名称
            
        Returns:
            TaskDifficulty 对象
        """
        # 提取期望的动作
        expected_actions = []
        if task.evaluation_criteria and task.evaluation_criteria.actions:
            expected_actions = [a.name for a in task.evaluation_criteria.actions]
        
        # 检查是否多步骤任务
        is_multi_step = len(expected_actions) >= 3
        
        # 检查是否涉及弱点工具
        weak_tools = self.WEAK_TOOLS.get(domain, [])
        involved_weak_tools = [t for t in expected_actions if t in weak_tools]
        
        # 检查是否涉及参数敏感工具
        param_sensitive_tools = self.PARAMETER_SENSITIVE_TOOLS.get(domain, [])
        has_param_sensitive = any(t in param_sensitive_tools for t in expected_actions)
        is_parameter_sensitive = has_param_sensitive or len(involved_weak_tools) > 0
        
        # 检查上下文复杂度
        scenario = str(task.user_scenario).lower() if task.user_scenario else ""
        is_context_complex = any(kw in scenario for kw in self.CONTEXT_COMPLEXITY_KEYWORDS)
        
        # 计算优先权重
        weight = self._compute_weight(
            is_multi_step=is_multi_step,
            is_parameter_sensitive=is_parameter_sensitive,
            is_context_complex=is_context_complex,
            involved_weak_tools=involved_weak_tools,
            task_id=task.id,
        )
        
        return TaskDifficulty(
            task_id=task.id,
            domain=domain,
            is_multi_step=is_multi_step,
            is_parameter_sensitive=is_parameter_sensitive,
            is_context_complex=is_context_complex,
            priority_weight=weight,
            weak_tools=involved_weak_tools,
            expected_actions=expected_actions,
        )
    
    def _compute_weight(
        self,
        is_multi_step: bool,
        is_parameter_sensitive: bool,
        is_context_complex: bool,
        involved_weak_tools: List[str],
        task_id: str,
    ) -> float:
        """计算任务优先权重"""
        weight = 1.0
        
        # 多步骤任务 +1.0
        if is_multi_step:
            weight += 1.0
        
        # 参数敏感任务 +1.5
        if is_parameter_sensitive:
            weight += 1.5
        
        # 上下文复杂任务 +1.0
        if is_context_complex:
            weight += 1.0
        
        # 每个弱点工具 +0.5
        weight += 0.5 * len(involved_weak_tools)
        
        # 历史失败记录加权
        if task_id in self.failure_history:
            fail_rate = self.failure_history[task_id].get("fail_rate", 0)
            weight *= (1 + fail_rate)
        
        return min(weight, 5.0)
    
    def classify_batch(self, tasks: List[Task], domain: str) -> List[TaskDifficulty]:
        """批量分类任务"""
        return [self.classify(task, domain) for task in tasks]
    
    def get_priority_sorted(self, tasks: List[Task], domain: str) -> List[tuple]:
        """
        获取按优先级排序的任务列表
        
        Returns:
            [(task, difficulty), ...] 按权重降序排列
        """
        classified = [(task, self.classify(task, domain)) for task in tasks]
        classified.sort(key=lambda x: x[1].priority_weight, reverse=True)
        return classified
