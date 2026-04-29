"""
生成tau2-bench格式的任务

基于提取的任务模式和参数分布，生成新的tau2格式任务定义。
"""
from typing import List, Dict, Any, Optional
import random
import json
import re
from datetime import datetime, timedelta

# 支持直接导入和相对导入
try:
    from .tau2_task_extractor import Tau2TaskPattern, Tau2TaskExtractor
    from .tau2_parameter_analyzer import Tau2ParameterAnalyzer
except ImportError:
    from tau2_task_extractor import Tau2TaskPattern, Tau2TaskExtractor
    from tau2_parameter_analyzer import Tau2ParameterAnalyzer


class Tau2TaskGenerator:
    """生成tau2格式的任务"""

    def __init__(self,
                 tau2_extractor: Tau2TaskExtractor,
                 param_analyzer: Tau2ParameterAnalyzer,
                 seed: Optional[int] = None):
        """
        初始化生成器

        Args:
            tau2_extractor: 任务模式提取器
            param_analyzer: 参数分析器
            seed: 随机种子
        """
        self.extractor = tau2_extractor
        self.param_analyzer = param_analyzer

        if seed is not None:
            random.seed(seed)

    def generate_tasks(self,
                      domain: str,
                      num_tasks: int = 4000,
                      complexity_weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        生成tau2格式的任务

        Args:
            domain: 领域名称
            num_tasks: 目标任务数量
            complexity_weights: 复杂度权重 (可选)

        Returns:
            tau2格式的任务列表
        """
        print(f"[Tau2TaskGenerator] 开始生成 {num_tasks} 个 {domain} 任务...")

        # 1. 提取模式
        patterns = self.extractor.extract_patterns(domain)
        print(f"  从tau2-bench提取了 {len(patterns)} 个任务模式")

        # 2. 分析参数空间
        param_space = self.param_analyzer.analyze_parameter_space(domain)
        print(f"  分析了参数分布")

        # 3. 计算每个模式需要生成的数量
        if complexity_weights:
            # 按复杂度分配
            task_distribution = self._distribute_by_complexity(patterns, num_tasks, complexity_weights)
        else:
            # 均匀分配
            variations_per_pattern = num_tasks // len(patterns)
            task_distribution = {p.task_id: variations_per_pattern for p in patterns}

            # 补齐余数
            remainder = num_tasks - sum(task_distribution.values())
            for i, pattern in enumerate(patterns[:remainder]):
                task_distribution[pattern.task_id] += 1

        # 4. 为每个模式生成变体
        all_tasks = []
        for pattern in patterns:
            num_variations = task_distribution[pattern.task_id]
            if num_variations == 0:
                continue

            for i in range(num_variations):
                task = self._generate_task_from_pattern(
                    pattern,
                    variation_id=i,
                    param_space=param_space
                )
                all_tasks.append(task)

        print(f"  成功生成 {len(all_tasks)} 个任务")
        return all_tasks

    def _distribute_by_complexity(self,
                                  patterns: List[Tau2TaskPattern],
                                  num_tasks: int,
                                  complexity_weights: Dict[str, float]) -> Dict[str, int]:
        """
        按复杂度权重分配任务数量

        Args:
            patterns: 任务模式列表
            num_tasks: 总任务数
            complexity_weights: 复杂度权重

        Returns:
            任务ID到数量的映射
        """
        # 按复杂度分组
        by_complexity = {"simple": [], "medium": [], "complex": []}
        for p in patterns:
            by_complexity[p.complexity].append(p)

        # 计算每个复杂度应该生成的数量
        complexity_totals = {}
        for complexity, weight in complexity_weights.items():
            complexity_totals[complexity] = int(num_tasks * weight)

        # 分配给具体模式
        distribution = {}
        for complexity, total in complexity_totals.items():
            patterns_in_group = by_complexity[complexity]
            if not patterns_in_group:
                continue

            per_pattern = total // len(patterns_in_group)
            remainder = total - per_pattern * len(patterns_in_group)

            for i, pattern in enumerate(patterns_in_group):
                distribution[pattern.task_id] = per_pattern
                if i < remainder:
                    distribution[pattern.task_id] += 1

        return distribution

    def _generate_task_from_pattern(self,
                                    pattern: Tau2TaskPattern,
                                    variation_id: int,
                                    param_space: Dict) -> Dict:
        """
        从模式生成单个tau2格式的任务

        Args:
            pattern: 任务模式
            variation_id: 变体ID
            param_space: 参数空间

        Returns:
            完整的tau2 Task对象
        """
        # 采样参数
        params = self._sample_parameters(pattern.domain, param_space)

        # 生成任务ID
        task_id = f"synthetic_{pattern.domain}_{pattern.task_id}_v{variation_id}"

        # 构造tau2格式的Task
        task = {
            "id": task_id,

            # Description部分
            "description": {
                "purpose": pattern.description.get("purpose", f"Synthetic task based on tau2 task #{pattern.task_id}"),
                "relevant_policies": pattern.description.get("relevant_policies", []),
                "notes": f"Generated from tau2 task {pattern.task_id} | Type: {pattern.task_type} | Complexity: {pattern.complexity} | Variation: {variation_id}"
            },

            # User Scenario部分
            "user_scenario": {
                "persona": self._vary_persona(pattern.persona),
                "instructions": {
                    "domain": pattern.domain,
                    "reason_for_call": self._instantiate_template(pattern.reason_for_call, params),
                    "known_info": self._instantiate_template(pattern.known_info, params),
                    "unknown_info": pattern.unknown_info,  # 通常不需要实例化
                    "task_instructions": self._instantiate_template(pattern.task_instructions, params)
                }
            },

            # Evaluation Criteria部分
            "evaluation_criteria": {
                "actions": self._instantiate_actions(pattern.expected_actions, params),
                "nl_assertions": pattern.nl_assertions,
                "reward_basis": pattern.reward_basis
            }
        }

        return task

    def _sample_parameters(self, domain: str, param_space: Dict) -> Dict[str, Any]:
        """
        采样具体参数

        Args:
            domain: 领域名称
            param_space: 参数空间

        Returns:
            采样的参数字典
        """
        if domain == "airline":
            return self._sample_airline_params(param_space)
        elif domain == "retail":
            return self._sample_retail_params(param_space)
        elif domain == "telecom":
            return self._sample_telecom_params(param_space)
        else:
            return {}

    def _sample_airline_params(self, param_space: Dict) -> Dict[str, Any]:
        """采样airline参数（基于db.json的真实分布）"""
        # 航线
        routes = list(param_space["routes"].keys())
        route_weights = list(param_space["routes"].values())
        route = random.choices(routes, weights=route_weights)[0] if routes else ("JFK", "LAX")

        # 舱位
        cabins = list(param_space["cabins"].keys())
        cabin_weights = list(param_space["cabins"].values())
        cabin = random.choices(cabins, weights=cabin_weights)[0] if cabins else "economy"

        # 姓名
        last_names = list(param_space["last_names"].keys())
        first_names = list(param_space["first_names"].keys())
        last_name = random.choice(last_names) if last_names else "Doe"
        first_name = random.choice(first_names) if first_names else "John"

        # 乘客数量
        if param_space["passenger_counts"]:
            counts = list(param_space["passenger_counts"].keys())
            count_weights = list(param_space["passenger_counts"].values())
            passenger_count = random.choices(counts, weights=count_weights)[0]
        else:
            passenger_count = random.choices([1, 2, 3, 4], weights=[0.6, 0.2, 0.1, 0.1])[0]

        # 日期
        days_ahead = random.randint(7, 90)
        date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        # Confirmation number (模拟格式)
        confirmation = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))

        return {
            "origin": route[0],
            "destination": route[1],
            "cabin": cabin,
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}",
            "date": date,
            "date_outbound": date,
            "date_return": (datetime.now() + timedelta(days=days_ahead + random.randint(3, 14))).strftime("%Y-%m-%d"),
            "confirmation": confirmation,
            "reservation_id": confirmation,
            "passenger_count": passenger_count
        }

    def _sample_retail_params(self, param_space: Dict) -> Dict[str, Any]:
        """采样retail参数"""
        # 城市
        cities = list(param_space["cities"].keys())
        city = random.choice(cities) if cities else "New York"

        # 邮编
        zip_codes = list(param_space["zip_codes"].keys())
        zip_code = random.choice(zip_codes) if zip_codes else "10001"

        # 姓名
        names = list(param_space["names"].keys())
        name = random.choice(names) if names else "John Doe"

        # 拆分姓名
        name_parts = name.split()
        first_name = name_parts[0] if len(name_parts) > 0 else "John"
        last_name = name_parts[-1] if len(name_parts) > 1 else "Doe"

        # 订单ID (模拟格式)
        order_id = f"#W{''.join(random.choices('0123456789', k=7))}"

        return {
            "city": city,
            "zip_code": zip_code,
            "order_id": order_id,
            "first_name": first_name,
            "last_name": last_name,
            "full_name": name,
            "email": f"{first_name.lower()}.{last_name.lower()}@example.com"
        }

    def _sample_telecom_params(self, param_space: Dict) -> Dict[str, Any]:
        """采样telecom参数"""
        # 套餐
        plan_names = list(param_space["plan_names"].keys())
        plan_name = random.choice(plan_names) if plan_names else "Basic Plan"

        # 电话号码
        phone_numbers = param_space.get("phone_numbers", [])
        if phone_numbers:
            phone_number = random.choice(phone_numbers)
        else:
            phone_number = f"+1-{''.join(random.choices('0123456789', k=3))}-{''.join(random.choices('0123456789', k=3))}-{''.join(random.choices('0123456789', k=4))}"

        # 客户ID
        customer_id = f"cust_{''.join(random.choices('0123456789', k=8))}"

        # 客户名
        customer_names = list(param_space["customer_names"].keys())
        customer_name = random.choice(customer_names) if customer_names else "John Doe"

        return {
            "plan_name": plan_name,
            "phone_number": phone_number,
            "customer_id": customer_id,
            "customer_name": customer_name
        }

    def _vary_persona(self, original_persona: Optional[str]) -> str:
        """
        变化persona（生成同义表达）

        Args:
            original_persona: 原始persona

        Returns:
            变化后的persona
        """
        if not original_persona:
            return "Polite and patient customer"

        # 简单变体映射
        persona_variants = {
            "professional": ["Busy professional", "Corporate traveler", "Business person who values efficiency"],
            "direct": ["Direct communicator", "Person who prefers quick responses", "Straight-to-the-point individual"],
            "patient": ["Patient person", "Thoughtful individual", "Careful customer"],
            "cautious": ["Cautious person", "Methodical individual", "Detail-oriented customer"],
            "frustrated": ["Frustrated customer", "Upset individual", "Dissatisfied person"],
            "polite": ["Polite customer", "Courteous person", "Respectful individual"]
        }

        # 尝试匹配并替换
        persona_lower = original_persona.lower()
        for key, variants in persona_variants.items():
            if key in persona_lower:
                # 随机选择是否变化
                if random.random() < 0.5:
                    return random.choice(variants)

        return original_persona

    def _instantiate_template(self, template: str, params: Dict[str, Any]) -> str:
        """
        实例化模板字符串

        将模板中的占位符替换为具体参数值。支持多种占位符格式。

        Args:
            template: 模板字符串
            params: 参数字典

        Returns:
            实例化后的字符串
        """
        if not template:
            return ""

        result = template

        # 替换 {key} 格式的占位符
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        # 替换常见的占位符模式（即使没有花括号）
        # 例如: "confirmation ABC123" -> "confirmation XYZ789"
        # 这里使用正则表达式匹配常见模式

        # 替换confirmation numbers (6位字母数字组合)
        if "confirmation" in result.lower():
            result = re.sub(r'\b[A-Z0-9]{6}\b', params.get("confirmation", "ABC123"), result)

        # 替换order IDs (#W开头的7位数字)
        if "order" in result.lower() and "#" in result:
            result = re.sub(r'#W\d{7}', params.get("order_id", "#W1234567"), result)

        return result

    def _instantiate_actions(self, expected_actions: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """
        实例化expected_actions

        将actions中的参数占位符替换为具体值

        Args:
            expected_actions: 期望的工具调用列表
            params: 参数字典

        Returns:
            实例化后的actions列表
        """
        instantiated = []

        for action in expected_actions:
            new_action = {
                "action_id": action.get("action_id", ""),
                "name": action["name"],
                "arguments": {}
            }

            # 实例化arguments
            for arg_name, arg_value in action.get("arguments", {}).items():
                # 如果是字符串且包含占位符，替换
                if isinstance(arg_value, str):
                    new_value = self._instantiate_template(arg_value, params)
                    new_action["arguments"][arg_name] = new_value
                else:
                    new_action["arguments"][arg_name] = arg_value

            # 保留compare_args
            if "compare_args" in action:
                new_action["compare_args"] = action["compare_args"]

            instantiated.append(new_action)

        return instantiated

    def save_tasks(self, tasks: List[Dict], output_file: str) -> None:
        """
        保存任务到JSON文件

        Args:
            tasks: 任务列表
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        print(f"[Tau2TaskGenerator] 保存了 {len(tasks)} 个任务到 {output_file}")

    def get_statistics(self, tasks: List[Dict]) -> Dict[str, Any]:
        """
        计算生成任务的统计信息

        Args:
            tasks: 任务列表

        Returns:
            统计信息
        """
        from collections import Counter

        # 提取复杂度
        complexities = []
        task_types = []
        num_actions = []

        for task in tasks:
            notes = task.get("description", {}).get("notes", "")

            # 提取复杂度
            if "Complexity:" in notes:
                complexity = notes.split("Complexity:")[1].split("|")[0].strip()
                complexities.append(complexity)

            # 提取任务类型
            if "Type:" in notes:
                task_type = notes.split("Type:")[1].split("|")[0].strip()
                task_types.append(task_type)

            # 统计actions数量
            actions = task.get("evaluation_criteria", {}).get("actions", [])
            num_actions.append(len(actions))

        return {
            "total_tasks": len(tasks),
            "complexity_distribution": dict(Counter(complexities)),
            "task_type_distribution": dict(Counter(task_types)),
            "avg_actions_per_task": sum(num_actions) / len(num_actions) if num_actions else 0,
            "action_count_distribution": dict(Counter(num_actions))
        }


if __name__ == "__main__":
    # 测试代码
    from tau2_task_extractor import Tau2TaskExtractor
    from tau2_parameter_analyzer import Tau2ParameterAnalyzer

    extractor = Tau2TaskExtractor()
    analyzer = Tau2ParameterAnalyzer()
    generator = Tau2TaskGenerator(extractor, analyzer, seed=42)

    # 生成少量任务测试
    domain = "airline"
    tasks = generator.generate_tasks(domain, num_tasks=10)

    print(f"\n生成的任务示例 (前2个):")
    for i, task in enumerate(tasks[:2], 1):
        print(f"\n任务 {i}:")
        print(f"  ID: {task['id']}")
        print(f"  Domain: {task['user_scenario']['instructions']['domain']}")
        print(f"  Reason: {task['user_scenario']['instructions']['reason_for_call'][:60]}...")
        print(f"  Actions: {[a['name'] for a in task['evaluation_criteria']['actions']]}")

    # 统计信息
    stats = generator.get_statistics(tasks)
    print(f"\n统计信息:")
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  复杂度分布: {stats['complexity_distribution']}")
    print(f"  平均actions数: {stats['avg_actions_per_task']:.2f}")
