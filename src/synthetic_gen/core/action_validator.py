"""
增强版Action序列验证器 - V1.1

基于V4 vs V0的深度对比分析，新增更多质量约束：
1. V0问题：78%的tool_call包含冗余文本（V4只有23%）
2. V0问题：不必要的确认和解释过多（166个 vs V4的31个）
3. V0问题：对话平均长度是V4的2倍
4. V0问题：assistant响应过于啰嗦
"""

from typing import List, Dict, Any, Tuple
import re
import json


def log(msg):
    """简单日志函数"""
    print(f"[ActionValidator] {msg}")


class EnhancedActionValidator:
    """增强版Action序列验证器"""

    def __init__(self, domain: str):
        """
        Args:
            domain: 领域 (airline/retail/telecom)
        """
        self.domain = domain
        self.validation_rules = self._init_rules()

    def _init_rules(self) -> Dict[str, Any]:
        """初始化验证规则"""
        base_rules = {
            # ============ 原有规则 ============
            # Action序列规则
            "ban_unnecessary_list_airports": True if self.domain == "airline" else False,
            "max_search_operations": 5,  # 最多5次搜索
            "require_get_before_update": True,
            "max_conversation_length": 35,  # 最多35条消息
            "max_total_tool_calls": 12,    # 最多12次工具调用

            # ============ V1.1规则（已禁用过严规则）============
            # 以下规则已被prompt约束取代，不再在validation阶段严格检查
            "max_text_with_tool_call_ratio": 1.0,  # 禁用此规则（设为100%）
            "max_assistant_response_length": 99999,  # 禁用此规则
            "ban_redundant_confirmations": False,  # 禁用此规则
            "max_assistant_pure_text_ratio": 1.0,  # 禁用此规则
            "require_direct_tool_call": False,  # 禁用此规则
            "max_text_before_tool_call": 99999,  # 禁用此规则（prompt已约束）

            # 领域特定规则（保留）
            "domain_specific_bans": self._get_domain_specific_bans(),
        }
        return base_rules

    def _get_domain_specific_bans(self) -> List[str]:
        """获取领域特定的禁止模式"""
        if self.domain == "airline":
            return [
                "list_all_airports",  # 完全禁止
            ]
        elif self.domain == "retail":
            return [
                "list_all_products",  # 如果有这个工具，禁止
            ]
        elif self.domain == "telecom":
            return [
                "list_all_plans",     # 如果有这个工具，禁止
            ]
        return []

    def validate(self, actions: List[str], conversation_messages: List[Dict] = None) -> Tuple[bool, List[str]]:
        """
        验证action序列和对话质量

        Args:
            actions: action名称列表
            conversation_messages: 完整对话消息

        Returns:
            (is_valid, reasons): 是否有效，原因列表
        """
        reasons = []

        if not actions and not conversation_messages:
            return True, []

        # ============ 原有规则检查 ============

        # 规则1: 领域特定禁止
        banned_tools = self.validation_rules.get("domain_specific_bans", [])
        for banned in banned_tools:
            if banned in actions:
                reasons.append(f"使用了禁止的工具: {banned}")

        # 规则2: search操作不应过多
        search_patterns = ["search", "list"]
        search_count = sum(1 for action in actions if any(p in action.lower() for p in search_patterns))
        max_search = self.validation_rules.get("max_search_operations", 3)
        if search_count > max_search:
            reasons.append(f"搜索操作过多 ({search_count} > {max_search})")

        # 规则3: update前必须先get
        if self.validation_rules.get("require_get_before_update", False):
            update_actions = ["update", "modify", "change", "edit"]
            get_actions = ["get", "fetch", "retrieve", "find", "search"]

            for i, action in enumerate(actions):
                if any(update in action.lower() for update in update_actions):
                    has_get_before = any(
                        any(get_action in actions[j].lower() for get_action in get_actions)
                        for j in range(i)
                    )
                    if not has_get_before:
                        reasons.append(f"{action}前缺少查询操作")

        # 规则4: 对话长度限制
        if conversation_messages:
            max_len = self.validation_rules.get("max_conversation_length", 25)
            if len(conversation_messages) > max_len:
                reasons.append(f"对话过长 ({len(conversation_messages)} > {max_len})")

        # 规则5: tool_calls总数限制
        max_calls = self.validation_rules.get("max_total_tool_calls", 8)
        if len(actions) > max_calls:
            reasons.append(f"工具调用过多 ({len(actions)} > {max_calls})")

        # ============ V1.1新增规则检查 ============

        if conversation_messages:
            # 规则6: 检查text_with_tool_call比例
            text_with_tool = 0
            pure_tool = 0
            pure_text = 0
            verbose_responses = 0

            for msg in conversation_messages:
                if msg.get('role') != 'assistant':
                    continue

                content = str(msg.get('content', ''))

                # 检查响应长度
                max_response_len = self.validation_rules.get("max_assistant_response_length", 400)
                if len(content) > max_response_len:
                    verbose_responses += 1

                if '<tool_call>' in content:
                    text_before = content.split('<tool_call>')[0].strip()
                    if text_before:
                        text_with_tool += 1
                        # 检查文本是否过长（使用配置的阈值）
                        max_text_len = self.validation_rules.get("max_text_before_tool_call", 200)
                        if len(text_before) > max_text_len:
                            reasons.append(f"工具调用前文本过长 ({len(text_before)}>{max_text_len}字符)")
                    else:
                        pure_tool += 1
                else:
                    pure_text += 1

            # 计算比例
            total_tool_calls = text_with_tool + pure_tool
            if total_tool_calls > 0:
                text_tool_ratio = text_with_tool / total_tool_calls
                max_ratio = self.validation_rules.get("max_text_with_tool_call_ratio", 0.4)

                if text_tool_ratio > max_ratio:
                    reasons.append(f"工具调用包含文本的比例过高 ({text_tool_ratio*100:.1f}% > {max_ratio*100:.1f}%)")

            # 检查纯文本assistant响应比例
            total_assistant = text_with_tool + pure_tool + pure_text
            if total_assistant > 0:
                pure_text_ratio = pure_text / total_assistant
                max_text_ratio = self.validation_rules.get("max_assistant_pure_text_ratio", 0.6)

                if pure_text_ratio > max_text_ratio:
                    reasons.append(f"Assistant纯文本响应过多 ({pure_text_ratio*100:.1f}% > {max_text_ratio*100:.1f}%)")

            # 检查啰嗦响应
            if verbose_responses > len(conversation_messages) * 0.2:  # 超过20%的消息过长
                reasons.append(f"过多啰嗦响应 ({verbose_responses}个)")

            # 规则7: 检查冗余确认
            if self.validation_rules.get("ban_redundant_confirmations", False):
                confirmation_keywords = [
                    "let me confirm", "just to confirm", "to confirm",
                    "let me verify", "let me check if i understand",
                    "before i proceed", "before we continue"
                ]

                confirmation_count = 0
                for msg in conversation_messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '').lower()
                        if any(keyword in content for keyword in confirmation_keywords):
                            confirmation_count += 1

                # 超过2次确认认为冗余
                if confirmation_count > 2:
                    reasons.append(f"过多确认性响应 ({confirmation_count}次)")

        is_valid = len(reasons) == 0
        return is_valid, reasons

    def extract_actions_from_messages(self, messages: List[Dict]) -> List[str]:
        """从消息中提取actions"""
        actions = []

        for msg in messages:
            if msg.get('role') != 'assistant':
                continue

            content = str(msg.get('content', ''))

            if '<tool_call>' in content:
                pattern = r'<tool_call>(.*?)</tool_call>'
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        tc = json.loads(match)
                        action_name = tc.get('name', '')
                        if action_name:
                            actions.append(action_name)
                    except:
                        pass

        return actions

    def should_filter_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """判断是否应该过滤样本"""
        messages = sample.get('messages', [])
        actions = self.extract_actions_from_messages(messages)

        is_valid, reasons = self.validate(actions, messages)

        return not is_valid, reasons
