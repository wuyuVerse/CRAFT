"""
错误纠正剪枝模块：检测用户明确指出assistant错误并被纠正的对话片段，通过多agent投票决定是否剪枝
"""
import json
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from dataclasses import dataclass
from ..utils.logger import log


@dataclass
class ErrorCorrectionSegment:
    """错误纠正片段"""
    start_index: int  # 错误开始的索引
    end_index: int  # 纠正结束的索引（用户指出错误的位置）
    correction_index: int  # assistant纠正行为的索引
    user_complaint: str  # 用户的错误描述
    error_context: List[Dict]  # 错误上下文（从start到end的对话）
    corrected_response: str  # 纠正后的assistant回复


class ErrorDetectionAgent:
    """错误检测Agent：识别用户明确指出assistant错误的位置"""
    
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
    
    async def detect_error_segments(
        self,
        conversations: List[Dict]
    ) -> List[ErrorCorrectionSegment]:
        """
        检测对话中所有用户明确指出错误的片段
        
        Returns:
            错误纠正片段列表
        """
        detection_prompt = f"""
You are an Error Detection Agent. Your task is to identify ALL segments in the conversation where:
1. The user explicitly points out that the assistant made an error in a previous step
2. The assistant later corrects this error

**Detection Criteria:**
- User explicitly mentions errors like "that's wrong", "you made a mistake", "incorrect", "that's not right"
- User corrects assistant's tool call or response
- User indicates the previous assistant action was inappropriate

**Conversations:**
{json.dumps(conversations, ensure_ascii=False, indent=2)}

For EACH error-correction segment found, identify:
1. **start_index**: Index where the erroneous assistant response begins
2. **end_index**: Index where the user points out the error
3. **user_complaint**: What the user said about the error
4. **has_correction**: Whether there's a later assistant response that fixes the error

Output ONLY valid JSON:
{{
    "error_segments": [
        {{
            "start_index": int,
            "end_index": int,
            "user_complaint": "string",
            "has_correction": true/false
        }},
        ...
    ],
    "total_errors_found": int
}}

If NO error segments found, return {{"error_segments": [], "total_errors_found": 0}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting error-correction patterns in conversations."},
                    {"role": "user", "content": detection_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                return []
            
            result = json.loads(content)
            error_segments_data = result.get("error_segments", [])
            
            log(f"\n[错误检测] 发现 {len(error_segments_data)} 个潜在的错误纠正片段")
            
            # 构建ErrorCorrectionSegment对象
            segments = []
            for seg_data in error_segments_data:
                if not seg_data.get("has_correction", False):
                    log(f"  - 索引 {seg_data.get('start_index')}-{seg_data.get('end_index')}: 无后续纠正，跳过")
                    continue
                
                start_idx = seg_data.get("start_index", 0)
                end_idx = seg_data.get("end_index", 0)
                user_complaint = seg_data.get("user_complaint", "")
                
                # 查找纠正位置（end_index之后的第一个assistant回复）
                correction_idx = -1
                for i in range(end_idx + 1, len(conversations)):
                    if conversations[i].get("from") == "assistant":
                        correction_idx = i
                        break
                
                if correction_idx == -1:
                    log(f"  - 索引 {start_idx}-{end_idx}: 未找到纠正响应，跳过")
                    continue
                
                # 提取错误上下文
                error_context = conversations[start_idx:end_idx + 1]
                corrected_response = conversations[correction_idx].get("value", "")
                
                segment = ErrorCorrectionSegment(
                    start_index=start_idx,
                    end_index=end_idx,
                    correction_index=correction_idx,
                    user_complaint=user_complaint,
                    error_context=error_context,
                    corrected_response=corrected_response
                )
                segments.append(segment)
                log(f"  ✓ 索引 {start_idx}-{end_idx} → 纠正于 {correction_idx}")
            
            return segments
            
        except Exception as e:
            log(f"[错误检测] 失败: {e}")
            return []


class PruningVotingAgent:
    """剪枝投票Agent：评估是否应该剪枝某个错误片段（支持多轮投票和讨论）"""
    
    def __init__(self, client: AsyncOpenAI, model_name: str, agent_id: int):
        self.client = client
        self.model_name = model_name
        self.agent_id = agent_id
    
    async def vote(
        self,
        segment: ErrorCorrectionSegment,
        full_conversations: List[Dict],
        round_num: int,
        previous_votes: Optional[List[Dict]] = None,
        discussion_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        对一个错误片段投票：是否应该剪枝（支持多轮投票）
        
        Args:
            segment: 错误片段
            full_conversations: 完整对话历史
            round_num: 当前投票轮次（从1开始）
            previous_votes: 上一轮所有agent的投票结果
            discussion_history: 之前所有轮次的讨论记录
        
        Returns:
            {
                "should_prune": bool,
                "reasoning": str,
                "confidence": float (0-1),
                "discussion_points": str  # 提供给其他agent的讨论要点
            }
        """
        # 构建剪枝前后的对话对比
        before_context = full_conversations[max(0, segment.start_index - 2):segment.start_index]
        after_context = full_conversations[segment.correction_index:min(len(full_conversations), segment.correction_index + 3)]
        
        # 构建上一轮投票信息
        previous_round_info = ""
        if previous_votes and round_num > 1:
            previous_round_info = f"""
**Previous Round {round_num - 1} Votes:**
"""
            for vote in previous_votes:
                previous_round_info += f"""
- Agent{vote['agent_id']}: {'PRUNE' if vote['should_prune'] else 'KEEP'} (confidence: {vote['confidence']:.2f})
  Reasoning: {vote['reasoning'][:150]}...
  Discussion: {vote.get('discussion_points', 'N/A')[:100]}...
"""
        
        # 构建讨论历史
        discussion_context = ""
        if discussion_history:
            discussion_context = f"""
**Discussion History:**
{''.join(discussion_history)}
"""
        
        voting_prompt = f"""
You are Voting Agent #{self.agent_id} in Round {round_num}. 

{'This is the FIRST voting round. Make your initial assessment.' if round_num == 1 else f'This is voting round {round_num}. Consider previous votes and discussions.'}

Evaluate whether this error-correction segment should be PRUNED (removed) to improve training data quality.

**Error Segment Details:**
- Start Index: {segment.start_index}
- End Index: {segment.end_index} (user complaint)
- Correction Index: {segment.correction_index}
- User Complaint: "{segment.user_complaint}"

**Context Before Error:**
{json.dumps(before_context, ensure_ascii=False, indent=2)}

**Error Context (to be potentially pruned):**
{json.dumps(segment.error_context, ensure_ascii=False, indent=2)}

**Corrected Response:**
{json.dumps(full_conversations[segment.correction_index], ensure_ascii=False, indent=2)}

**Context After Correction:**
{json.dumps(after_context, ensure_ascii=False, indent=2)}

{previous_round_info}

{discussion_context}

**Evaluation Criteria:**

**SHOULD PRUNE if:**
1. The error was a simple mistake that doesn't add training value
2. Removing the error makes the conversation flow more naturally
3. The corrected response can seamlessly connect to the previous context
4. The error-correction cycle doesn't teach important error-recovery patterns

**SHOULD NOT PRUNE if:**
1. The error-correction shows important learning patterns
2. The user's feedback provides valuable training signal
3. Pruning would break conversation coherence
4. The error represents a common failure mode worth training on

**Instructions for Round {round_num}:**
{self._get_round_instructions(round_num, previous_votes)}

**Output Requirements:**
1. **should_prune**: Your decision (true/false)
2. **reasoning**: Detailed explanation of your decision
3. **confidence**: How confident you are (0.0-1.0)
4. **discussion_points**: Key points to share with other agents for next round discussion
5. **coherence_after_pruning**: Score 1-5
6. **training_value_lost**: Score 1-5

Output ONLY valid JSON:
{{
    "should_prune": true/false,
    "reasoning": "Detailed explanation",
    "confidence": 0.0-1.0,
    "discussion_points": "Key points for discussion",
    "coherence_after_pruning": 1-5,
    "training_value_lost": 1-5
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are Voting Agent #{self.agent_id}, an expert in evaluating training data quality. You participate in multi-round voting with discussion."},
                    {"role": "user", "content": voting_prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                return {"should_prune": False, "reasoning": "投票失败", "confidence": 0.0, "discussion_points": ""}
            
            result = json.loads(content)
            return {
                "agent_id": self.agent_id,
                "should_prune": result.get("should_prune", False),
                "reasoning": result.get("reasoning", ""),
                "confidence": float(result.get("confidence", 0.5)),
                "discussion_points": result.get("discussion_points", ""),
                "coherence_after_pruning": int(result.get("coherence_after_pruning", 3)),
                "training_value_lost": int(result.get("training_value_lost", 3))
            }
            
        except Exception as e:
            log(f"[投票Agent{self.agent_id}] 错误: {e}")
            return {"agent_id": self.agent_id, "should_prune": False, "reasoning": f"投票出错: {e}", "confidence": 0.0, "discussion_points": ""}
    
    def _get_round_instructions(self, round_num: int, previous_votes: Optional[List[Dict]]) -> str:
        """获取当前轮次的特定指令"""
        if round_num == 1:
            return """Make your initial assessment based on the error segment and evaluation criteria.
Provide clear discussion points to help other agents understand your perspective."""
        else:
            return f"""Review other agents' votes and discussion points from Round {round_num - 1}.
Consider if their arguments change your perspective.
You may change your vote if convinced by others' reasoning.
Provide updated discussion points addressing others' concerns."""


class ResponseContinuityFixer:
    """响应连贯性修复器：调整纠正后的回复使其与剪枝后的上下文连贯"""
    
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
    
    async def fix_continuity(
        self,
        before_context: List[Dict],
        corrected_response: str,
        after_context: List[Dict]
    ) -> str:
        """
        修复纠正后回复的连贯性，使其自然衔接剪枝后的上下文
        
        Returns:
            修复后的回复内容
        """
        fix_prompt = f"""
You are a Response Continuity Fixer. The conversation had an error segment that was pruned (removed).
Your task is to adjust the corrected response so it flows naturally from the previous context, AS IF THE ERROR NEVER HAPPENED.

**Context Before (what the user last said):**
{json.dumps(before_context[-2:] if len(before_context) >= 2 else before_context, ensure_ascii=False, indent=2)}

**Original Corrected Response (may reference the pruned error):**
{corrected_response}

**Context After (next messages):**
{json.dumps(after_context[:2] if len(after_context) >= 2 else after_context, ensure_ascii=False, indent=2)}

**Requirements:**
1. Remove any references to the previous error or correction
2. Make the response flow naturally from the "Context Before"
3. Keep the CORE FUNCTIONALITY and tool calls intact
4. Adjust wording to sound like a direct, correct response
5. Ensure seamless transition to "Context After"

**Example:**
- BAD: "Sorry for the earlier mistake. Let me do this correctly..."
- GOOD: "I'll help you with that. Let me..."

Output ONLY valid JSON:
{{
    "fixed_response": "The adjusted response that flows naturally",
    "changes_made": "Brief description of what was changed"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at maintaining conversation flow and continuity."},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                return corrected_response
            
            result = json.loads(content)
            fixed = result.get("fixed_response", corrected_response)
            changes = result.get("changes_made", "")
            
            log(f"  [连贯性修复] {changes}")
            return fixed
            
        except Exception as e:
            log(f"[连贯性修复] 错误: {e}，返回原始响应")
            return corrected_response


class ErrorCorrectionPruner:
    """错误纠正剪枝器：主控制器，协调多agent完成剪枝任务"""
    
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        num_voting_agents: int = 3,
        max_voting_rounds: int = 3
    ):
        self.error_detector = ErrorDetectionAgent(client, model_name)
        self.voting_agents = [
            PruningVotingAgent(client, model_name, i + 1)
            for i in range(num_voting_agents)
        ]
        self.continuity_fixer = ResponseContinuityFixer(client, model_name)
        self.num_voting_agents = num_voting_agents
        self.max_voting_rounds = max_voting_rounds
    
    async def _conduct_multi_round_voting(
        self,
        segment: ErrorCorrectionSegment,
        conversations: List[Dict]
    ) -> Tuple[bool, List[Dict]]:
        """
        进行多轮投票和讨论
        
        Returns:
            (should_prune, all_rounds_votes)
        """
        all_rounds_votes = []
        discussion_history = []
        previous_votes = None
        
        for round_num in range(1, self.max_voting_rounds + 1):
            log(f"\n  [第{round_num}轮投票] 开始...")
            
            # 所有agent并发投票
            current_round_votes = await asyncio.gather(*[
                agent.vote(
                    segment=segment,
                    full_conversations=conversations,
                    round_num=round_num,
                    previous_votes=previous_votes,
                    discussion_history=discussion_history
                )
                for agent in self.voting_agents
            ])
            
            # 统计投票结果
            prune_votes = sum(1 for v in current_round_votes if v["should_prune"])
            keep_votes = len(current_round_votes) - prune_votes
            
            log(f"  [第{round_num}轮结果] 剪枝: {prune_votes}票, 保留: {keep_votes}票")
            for vote in current_round_votes:
                decision = "剪枝" if vote["should_prune"] else "保留"
                log(f"    Agent{vote['agent_id']}: {decision} (置信度: {vote['confidence']:.2f})")
                log(f"      理由: {vote['reasoning'][:80]}...")
                if vote.get('discussion_points'):
                    log(f"      讨论点: {vote['discussion_points'][:80]}...")
            
            # 记录本轮投票
            all_rounds_votes.append({
                "round": round_num,
                "votes": current_round_votes,
                "prune_count": prune_votes,
                "keep_count": keep_votes
            })
            
            # 检查是否达成一致
            is_unanimous = (prune_votes == len(self.voting_agents) or keep_votes == len(self.voting_agents))
            
            if is_unanimous:
                decision = prune_votes == len(self.voting_agents)
                log(f"  [第{round_num}轮] ✓ 全票一致 - {'剪枝' if decision else '保留'}")
                return decision, all_rounds_votes
            
            # 如果不是最后一轮，准备讨论材料
            if round_num < self.max_voting_rounds:
                log(f"  [第{round_num}轮] 未达成一致，进入讨论阶段...")
                
                # 生成讨论摘要
                discussion_summary = f"\n--- Round {round_num} Discussion ---\n"
                discussion_summary += f"Vote Split: {prune_votes} PRUNE vs {keep_votes} KEEP\n"
                discussion_summary += "Key Discussion Points:\n"
                for vote in current_round_votes:
                    discussion_summary += f"- Agent{vote['agent_id']}: {vote.get('discussion_points', 'N/A')}\n"
                
                discussion_history.append(discussion_summary)
                previous_votes = current_round_votes
                
                log(f"  [讨论摘要] 已生成，agents将在下一轮考虑这些观点")
            else:
                # 最后一轮，使用多数决
                log(f"  [第{round_num}轮] 最终轮次，使用多数决")
                decision = prune_votes > keep_votes
                log(f"  [最终决定] {'剪枝' if decision else '保留'} ({prune_votes}/{len(self.voting_agents)}票)")
                return decision, all_rounds_votes
        
        # 默认不剪枝
        return False, all_rounds_votes
    
    async def prune_conversations(
        self,
        conversations: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        对整个对话历史进行剪枝处理
        
        Returns:
            (pruned_conversations, pruning_stats)
        """
        log("\n" + "="*60)
        log("启动错误纠正剪枝流程")
        log("="*60)
        
        # 步骤1：检测所有错误纠正片段
        log("\n[步骤1] 检测错误纠正片段...")
        error_segments = await self.error_detector.detect_error_segments(conversations)
        
        if not error_segments:
            log("[结果] 未发现需要处理的错误纠正片段")
            return conversations, {"total_segments": 0, "pruned_segments": 0, "pruning_decisions": []}
        
        log(f"[结果] 发现 {len(error_segments)} 个错误纠正片段")
        
        # 步骤2：对每个片段进行多轮多agent投票
        pruning_decisions = []
        
        for idx, segment in enumerate(error_segments, 1):
            log(f"\n[步骤2.{idx}] 评估片段 {segment.start_index}-{segment.end_index}")
            log(f"  用户反馈: \"{segment.user_complaint[:100]}...\"")
            
            # 进行多轮投票和讨论
            should_prune, all_rounds_votes = await self._conduct_multi_round_voting(
                segment=segment,
                conversations=conversations
            )
            
            # 生成最终统计
            final_round = all_rounds_votes[-1]
            
            if should_prune:
                log(f"\n  [最终决定] ✂️ 剪枝 - 经过{len(all_rounds_votes)}轮投票")
            else:
                log(f"\n  [最终决定] ✋ 保留 - 经过{len(all_rounds_votes)}轮投票")
            
            pruning_decisions.append({
                "segment": {
                    "start_index": segment.start_index,
                    "end_index": segment.end_index,
                    "correction_index": segment.correction_index,
                    "user_complaint": segment.user_complaint
                },
                "all_rounds_votes": all_rounds_votes,
                "decision": "prune" if should_prune else "keep",
                "total_rounds": len(all_rounds_votes),
                "final_prune_votes": final_round["prune_count"],
                "final_keep_votes": final_round["keep_count"]
            })
        
        # 步骤3：执行剪枝操作（从后往前，避免索引混乱）
        log(f"\n[步骤3] 执行剪枝操作...")
        
        pruned_conversations = conversations.copy()
        pruned_count = 0
        
        # 按start_index倒序排序，从后往前处理
        segments_to_prune = [
            (segment, decision)
            for segment, decision in zip(error_segments, pruning_decisions)
            if decision["decision"] == "prune"
        ]
        segments_to_prune.sort(key=lambda x: x[0].start_index, reverse=True)
        
        for segment, decision in segments_to_prune:
            log(f"\n  剪枝片段 {segment.start_index}-{segment.end_index}")
            
            # 提取上下文
            before_context = pruned_conversations[:segment.start_index]
            after_context = pruned_conversations[segment.correction_index + 1:]
            
            # 修复纠正响应的连贯性
            log(f"  修复纠正响应的连贯性...")
            fixed_response = await self.continuity_fixer.fix_continuity(
                before_context=before_context,
                corrected_response=segment.corrected_response,
                after_context=after_context
            )
            
            # 更新纠正响应
            corrected_conversation = pruned_conversations[segment.correction_index].copy()
            corrected_conversation["value"] = fixed_response
            
            # 重建对话历史：before + 修复后的纠正 + after
            pruned_conversations = before_context + [corrected_conversation] + after_context
            
            pruned_count += 1
            log(f"  ✓ 剪枝完成，对话长度: {len(conversations)} → {len(pruned_conversations)}")
        
        # 生成统计报告
        stats = {
            "total_segments": len(error_segments),
            "pruned_segments": pruned_count,
            "kept_segments": len(error_segments) - pruned_count,
            "pruning_decisions": pruning_decisions,
            "original_length": len(conversations),
            "pruned_length": len(pruned_conversations),
            "messages_removed": len(conversations) - len(pruned_conversations)
        }
        
        log(f"\n" + "="*60)
        log(f"剪枝流程完成")
        log(f"="*60)
        log(f"发现片段: {stats['total_segments']}")
        log(f"剪枝片段: {stats['pruned_segments']}")
        log(f"保留片段: {stats['kept_segments']}")
        log(f"消息数量: {stats['original_length']} → {stats['pruned_length']} (-{stats['messages_removed']})")
        log(f"="*60 + "\n")
        
        return pruned_conversations, stats


# 异步支持
import asyncio
