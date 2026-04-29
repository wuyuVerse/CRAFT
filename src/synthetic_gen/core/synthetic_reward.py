"""
Synthetic Reward Calculator for Generated Tasks
为生成的任务计算reward，确保数据质量
"""
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from ..utils.logger import log


class SyntheticRewardCalculator:
    """为生成的任务计算reward"""
    
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model
    
    def _match_action(self, tool_call: Dict, expected_action: Dict) -> bool:
        """
        检查tool_call是否匹配expected_action
        
        Args:
            tool_call: 实际的工具调用
            expected_action: 期望的动作
        
        Returns:
            是否匹配
        """
        # 检查工具名称
        tool_name = tool_call.get('function', {}).get('name', '')
        expected_name = expected_action.get('name', '')
        
        if tool_name != expected_name:
            return False
        
        # 检查参数
        compare_args = expected_action.get('compare_args')
        if compare_args is None:
            # 如果没有指定compare_args，只检查工具名称
            return True
        
        try:
            # 解析工具调用的参数
            tool_args_str = tool_call.get('function', {}).get('arguments', '{}')
            if isinstance(tool_args_str, str):
                tool_args = json.loads(tool_args_str)
            else:
                tool_args = tool_args_str
            
            expected_args = expected_action.get('arguments', {})
            
            # 只比较指定的参数
            for arg_name in compare_args:
                if arg_name not in tool_args:
                    return False
                # 如果expected_action中指定了参数值，则必须匹配
                if arg_name in expected_args:
                    if tool_args[arg_name] != expected_args[arg_name]:
                        return False
            
            return True
            
        except Exception as e:
            log(f"[Reward] Error matching action: {e}")
            return False
    
    def calculate_action_reward(
        self, 
        conversation_history: List[Dict],
        expected_actions: List[Dict]
    ) -> float:
        """
        检查是否调用了期望的工具，并且按照正确的顺序
        
        逻辑：
        1. 提取对话中所有assistant的tool_call，按时间顺序
        2. 按sequence_order排序expected_actions
        3. 检查expected_actions是否作为子序列出现在实际调用中（顺序匹配）
        
        Args:
            conversation_history: 对话历史
            expected_actions: 期望的动作列表（每个包含tool_name和sequence_order）
        
        Returns:
            0-1之间的分数
        """
        if not expected_actions:
            return 1.0
        
        # 按sequence_order排序期望动作
        sorted_expected = sorted(
            expected_actions, 
            key=lambda x: x.get('sequence_order', 999)
        )
        
        # 提取对话中所有的tool_call名称（按时间顺序）
        actual_tool_names = []
        for msg in conversation_history:
            if msg.get('from') != 'assistant':
                continue
            
            content = msg.get('value', '') or msg.get('content', '')
            
            # 从content中提取<tool_call>标签
            if '<tool_call>' in content:
                import re
                tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                matches = re.findall(tool_call_pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        tool_data = json.loads(match)
                        tool_name = tool_data.get('name', '')
                        if tool_name:
                            actual_tool_names.append(tool_name)
                    except:
                        pass
        
        log(f"[Reward] 实际调用序列: {actual_tool_names}")
        log(f"[Reward] 期望调用序列: {[a.get('tool_name', '?') for a in sorted_expected]}")
        
        # 检查expected_actions是否作为子序列出现在actual_tool_names中
        # 使用贪心匹配：从左到右依次匹配每个期望的动作
        expected_idx = 0
        matched_count = 0
        
        for actual_name in actual_tool_names:
            if expected_idx >= len(sorted_expected):
                break
            
            expected_name = sorted_expected[expected_idx].get('tool_name', '')
            if actual_name == expected_name:
                matched_count += 1
                log(f"[Reward] 匹配动作 {matched_count}: {expected_name} (顺序正确)")
                expected_idx += 1
        
        reward = matched_count / len(sorted_expected)
        log(f"[Reward] ACTION: {matched_count}/{len(sorted_expected)} = {reward:.2f}")
        return reward
    
    async def _check_info_communicated(
        self, 
        conversation_text: str, 
        required_info: str
    ) -> bool:
        """
        使用LLM检查对话中是否传达了必要信息
        
        Args:
            conversation_text: 对话文本
            required_info: 必须传达的信息
        
        Returns:
            是否传达了该信息
        """
        check_prompt = f"""You are an information verification assistant. Check if the required information was communicated in the conversation.

**Required Information:**
{required_info}

**Conversation:**
{conversation_text[:2000]}  # Limit length

**Task:**
Did the assistant successfully communicate the required information to the user?
Consider synonyms and paraphrasing as valid communication.

Output ONLY valid JSON:
{{
    "communicated": true/false,
    "reason": "Brief explanation"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict information verification assistant."},
                    {"role": "user", "content": check_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                return False
            
            result = json.loads(content)
            return result.get("communicated", False)
            
        except Exception as e:
            log(f"[Reward] Error checking info: {e}")
            return False
    
    async def calculate_communicate_reward(
        self,
        conversation_history: List[Dict],
        communicate_info: List[str]
    ) -> float:
        """
        检查是否传达了必要的信息
        
        Args:
            conversation_history: 对话历史
            communicate_info: 必须传达的信息列表
        
        Returns:
            0-1之间的分数
        """
        if not communicate_info:
            return 1.0
        
        # 提取assistant的所有回复
        assistant_messages = []
        for msg in conversation_history:
            if msg.get('from') == 'assistant':
                content = msg.get('content') or msg.get('value', '')
                if content:
                    # 移除think和tool标签
                    content = content.replace('<think>', '').replace('</think>', '')
                    content = content.replace('<tool>', '').replace('</tool>', '')
                    assistant_messages.append(content.strip())
        
        full_conversation = "\n".join(assistant_messages)
        
        # 检查每个必要信息是否被传达
        matched_info = 0
        for info in communicate_info:
            if await self._check_info_communicated(full_conversation, info):
                matched_info += 1
                log(f"[Reward] 传达信息: {info}")
        
        reward = matched_info / len(communicate_info)
        log(f"[Reward] COMMUNICATE: {matched_info}/{len(communicate_info)} = {reward:.2f}")
        return reward
    
    async def calculate_success_conditions_reward(
        self,
        conversation_history: List[Dict],
        success_conditions: List[str]
    ) -> float:
        """
        检查是否满足成功条件
        
        Args:
            conversation_history: 对话历史
            success_conditions: 成功条件列表
        
        Returns:
            0-1之间的分数
        """
        if not success_conditions:
            return 1.0
        
        # 构建完整对话文本
        conversation_text = []
        for msg in conversation_history:
            role = msg.get('from', 'unknown')
            content = msg.get('content') or msg.get('value', '')
            if content:
                conversation_text.append(f"[{role}] {content}")
        
        full_text = "\n".join(conversation_text[-20:])  # 只看最后20轮
        
        check_prompt = f"""You are a task completion verifier. Check if the conversation satisfies the success conditions.

**Success Conditions:**
{json.dumps(success_conditions, indent=2)}

**Conversation (last 20 turns):**
{full_text[:3000]}

**Task:**
Check if EACH success condition was met in the conversation.
Be strict but reasonable - consider the intent and outcome.

Output ONLY valid JSON:
{{
    "satisfied_conditions": [
        {{
            "condition_index": int,
            "satisfied": true/false,
            "reason": "Brief explanation"
        }},
        ...
    ],
    "overall_success": true/false
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict task completion verifier."},
                    {"role": "user", "content": check_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                return 0.0
            
            result = json.loads(content)
            satisfied = sum(
                1 for cond in result.get('satisfied_conditions', []) 
                if cond.get('satisfied', False)
            )
            
            reward = satisfied / len(success_conditions)
            log(f"[Reward] SUCCESS_CONDITIONS: {satisfied}/{len(success_conditions)} = {reward:.2f}")
            return reward
            
        except Exception as e:
            log(f"[Reward] Error checking success conditions: {e}")
            return 0.0
    
    async def calculate_reward(
        self,
        conversation_history: List[Dict],
        evaluation_criteria: Dict
    ) -> Dict[str, Any]:
        """
        计算总reward
        
        Args:
            conversation_history: 对话历史
            evaluation_criteria: 评估标准
        
        Returns:
            {
                'reward': float,  # 总reward (0-1)
                'breakdown': dict,  # 各项reward
                'success': bool,  # 是否成功
                'details': dict  # 详细信息
            }
        """
        reward_basis = evaluation_criteria.get('reward_basis', ['ACTION', 'COMMUNICATE'])
        
        rewards = []  # 收集各项reward，最后取平均
        breakdown = {}
        details = {}
        
        # 首先计算ACTION reward（总是计算，用于统计）
        action_reward = None
        if 'ACTION' in reward_basis:
            action_reward = self.calculate_action_reward(
                conversation_history,
                evaluation_criteria.get('expected_actions', [])
            )
            breakdown['action'] = action_reward
            details['expected_actions'] = len(evaluation_criteria.get('expected_actions', []))
        
        # 计算COMMUNICATE reward
        comm_reward = None
        if 'COMMUNICATE' in reward_basis:
            comm_reward = await self.calculate_communicate_reward(
                conversation_history,
                evaluation_criteria.get('communicate_info', [])
            )
            breakdown['communicate'] = comm_reward
            details['required_communications'] = len(evaluation_criteria.get('communicate_info', []))
        
        # 计算SUCCESS_CONDITIONS reward（可选）
        success_reward = None
        success_conditions = evaluation_criteria.get('success_conditions', [])
        if success_conditions and isinstance(success_conditions, list) and all(isinstance(c, dict) for c in success_conditions):
            success_reward = await self.calculate_success_conditions_reward(
                conversation_history,
                success_conditions
            )
            breakdown['success_conditions'] = success_reward
            details['success_conditions_count'] = len(success_conditions)
        
        # 决定最终使用哪些reward进行平均
        # 规则：如果有COMMUNICATE或SUCCESS_CONDITIONS，则忽略ACTION
        has_other_rewards = comm_reward is not None or success_reward is not None
        
        if has_other_rewards:
            # 有其他reward，忽略ACTION
            if comm_reward is not None:
                rewards.append(comm_reward)
            if success_reward is not None:
                rewards.append(success_reward)
            log(f"[Reward] 使用其他reward维度，忽略ACTION reward")
        else:
            # 没有其他reward，使用ACTION
            if action_reward is not None:
                rewards.append(action_reward)
            log(f"[Reward] 仅使用ACTION reward")
        
        # 计算最终reward（所有维度的平均值）
        final_reward = sum(rewards) / len(rewards) if rewards else 1.0
        
        log(f"[Reward] FINAL: {final_reward:.2f} (平均值: {[f'{r:.2f}' for r in rewards]}, breakdown: {breakdown})")
        
        return {
            'reward': final_reward,
            'breakdown': breakdown,
            'success': final_reward >= 0.7,  # 阈值可调整
            'details': details
        }


class TaskQualityClassifier:
    """任务质量分类器"""
    
    @staticmethod
    def classify(reward: float) -> str:
        """根据reward分类任务质量"""
        if reward >= 0.95:
            return "PLATINUM"  # 完美
        elif reward >= 0.9:
            return "GOLD"  # 优秀
        elif reward >= 0.75:
            return "SILVER"  # 良好
        elif reward >= 0.6:
            return "BRONZE"  # 基础
        else:
            return "FAILED"  # 失败
    
    @staticmethod
    def should_save(reward: float, min_quality: str = "BRONZE") -> bool:
        """判断是否应该保存数据"""
        quality = TaskQualityClassifier.classify(reward)
        quality_order = ["FAILED", "BRONZE", "SILVER", "GOLD", "PLATINUM"]
        
        return quality_order.index(quality) >= quality_order.index(min_quality)

