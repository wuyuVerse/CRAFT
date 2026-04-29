import openai
import json
import re
from . import agents
from typing import List, Dict, Any
from enum import Enum
from .agents import UserResponse, ToolResponse, AgentResponse
import ast
from ..prompts import prompt
import asyncio
from .multi_turn_task_gen import MultiTurnTaskGen
from .self_reflection import AssistantResponseChecker, ToolCallHallucinationChecker, UserResponseReflectionChecker
from .error_correction_pruner import ErrorCorrectionPruner
from .synthetic_reward import SyntheticRewardCalculator, TaskQualityClassifier
from .action_validator import EnhancedActionValidator
from .real_task_loader import RealTaskLoader
from ..utils.logger import log
from ..utils.json_utils import safe_json_loads


class Speaker(Enum):
    TOOL = "TOOL"
    ASSISTANT_CHAT = "ASSISTANT-CHAT"
    ASSISTANT_TOOLCALL = "ASSISTANT-TOOLCALL"
    USER = "USER"
    USER_TOOLCALL = "USER-TOOLCALL"
    STOP = "STOP"

class PlayGround:
    """
    多轮任务执行模拟器
    根据生成的任务数据（user_info + task_data）模拟任务执行过程
    """
    def __init__(self, client_pool, primary_client, primary_model, max_turn: int = 30, validator=None, quality_threshold: int = 3, self_reflection_iterations: int = 1, enable_pruning: bool = False, num_voting_agents: int = 3, max_voting_rounds: int = 3, domain: str = None, enable_action_validation: bool = True, use_real_task_seed: bool = True):
        """
        初始化PlayGround

        Args:
            client_pool: 客户端池实例，用于为各个组件分配客户端
            primary_client: 主客户端，用于主要的API调用
            primary_model: 主模型名称
            domain: 指定固定的领域（airline/retail/telecom），如果为None则随机选择
            enable_action_validation: 是否启用action序列验证（V1新增）
            use_real_task_seed: 是否使用真实任务作为seed（V1新增）
        """
        # 保存client_pool引用，用于轮询获取client
        self.client_pool = client_pool
        
        # 使用轮询方式为不同组件分配客户端，实现真正的负载均衡
        task_gen_client, task_gen_model = client_pool.get_client()
        user_client, user_model = client_pool.get_client()
        tool_agent_client, tool_agent_model = client_pool.get_client()
        tool_sim_client, tool_sim_model = client_pool.get_client()
        
        # MultiTurnTaskGen现在不需要api_doc_path，直接从三个固定类别加载API
        # 如果指定了domain，则只采样该domain的任务
        self.task_generator = MultiTurnTaskGen(
            client=task_gen_client,
            model=task_gen_model,
            domain=domain
        )
        self.user_agent = agents.UserSimulator(
            "", 
            client=user_client,
            model=user_model,
            tools=None  # 初始化时不分配工具，等任务生成时根据domain决定
        )
        self.tool_agent = agents.ToolAgent(
            prompt.ToolAgentPrompt,
            client=tool_agent_client,
            model=tool_agent_model,
            tools=None,
            validator=validator
        )
        self.tool_simulator = agents.ToolSimulator(
            prompt.ToolSimulatorSystem,
            client=tool_sim_client,
            model=tool_sim_model,
            tools=None
        )
        # 添加User回复自我审视检查器
        user_checker_client, user_checker_model = client_pool.get_client()
        self.user_response_checker = UserResponseReflectionChecker(
            client=user_checker_client,
            model_name=user_checker_model,
            max_iterations=self_reflection_iterations
        )
        # 添加Assistant回复检查器
        asst_checker_client, asst_checker_model = client_pool.get_client()
        self.assistant_response_checker = AssistantResponseChecker(
            client=asst_checker_client,
            model_name=asst_checker_model
        )
        # 添加ToolCall幻觉检查器
        tool_checker_client, tool_checker_model = client_pool.get_client()
        self.toolcall_hallucination_checker = ToolCallHallucinationChecker(
            client=tool_checker_client,
            model_name=tool_checker_model,
            max_iterations=self_reflection_iterations
        )
        # 添加错误纠正剪枝器
        self.enable_pruning = enable_pruning
        if enable_pruning:
            pruner_client, pruner_model = client_pool.get_client()
            self.error_pruner = ErrorCorrectionPruner(
                client=pruner_client,
                model_name=pruner_model,
                num_voting_agents=num_voting_agents,
                max_voting_rounds=max_voting_rounds
            )
        self.max_turn = max_turn
        self.tools = []
        self.history = []
        self.user_info = None
        self.task_data = None
        # 质量检查相关
        self.quality_threshold = quality_threshold
        self.self_reflection_iterations = self_reflection_iterations
        self.client = primary_client  # 使用主客户端
        self.model = primary_model
        
        # 统计信息
        self.stats = {
            "toolagent_hallucination": {
                "total_hallucination_count": 0,  # 总幻觉次数
                "hallucinated_tool_count": 0,    # 幻觉工具次数
                "hallucinated_param_count": 0,   # 幻觉参数次数
                "resolved_count": 0,              # 成功解决次数
                "failed_count": 0                 # 未能解决次数
            },
            "user_correction": {
                "total_corrections": 0,           # 用户回复总修正次数
                "iterations_used": []             # 每次修正使用的迭代次数列表
            },
            "pruning": {
                "total_segments_detected": 0,     # 检测到的错误片段数
                "segments_pruned": 0,             # 实际剪枝的片段数
                "segments_kept": 0                # 保留的片段数
            },
            "action_validation": {
                "total_validated": 0,             # 总验证次数
                "passed": 0,                      # 验证通过次数
                "failed": 0,                      # 验证失败次数
                "failure_reasons": []             # 失败原因列表
            }
        }

        # V1新功能：Action序列验证和真实任务加载
        self.enable_action_validation = enable_action_validation
        self.use_real_task_seed = use_real_task_seed

        if self.enable_action_validation:
            # domain在_initialize_task时才确定，这里先设为None
            self.action_validator = None
            log("[V1] Action序列验证已启用")

        if self.use_real_task_seed:
            self.real_task_loader = RealTaskLoader()
            log("[V1] 真实任务seed模式已启用")
        else:
            self.real_task_loader = None
        
    def _convert_apis_to_tools(self, api_dict: Dict) -> Dict:
        """将API列表转换为OpenAI tools格式"""
        try:
            tool = {
                "type": "function",
                "function": {
                    "name": api_dict.get("name", "unknown_function"),
                    "description": api_dict.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        
            if "parameters" in api_dict and isinstance(api_dict["parameters"], dict):
                if "properties" in api_dict["parameters"]:
                    tool["function"]["parameters"]["properties"] = api_dict["parameters"]["properties"]
                if "required" in api_dict["parameters"]:
                    tool["function"]["parameters"]["required"] = api_dict["parameters"]["required"]
            
            return tool
        except Exception as e:
            log(f"Error converting API to tool format: {e}")
            return {"type": "function", "function": {"name": "error", "description": "", "parameters": {}}}

    async def _initialize_task(self):
        """
        初始化任务：生成任务数据并配置各个agent
        返回：(sampled_apis, user_info, task_data, user_simulator_prompt, domain_system_prompt, evaluation_criteria)
        """
        log("正在生成任务...")

        # V1: 如果启用真实任务seed，获取seed并注入到任务生成器
        real_task_seed = None
        if self.use_real_task_seed and self.real_task_loader:
            # 获取domain（如果已指定）
            target_domain = self.task_generator.domain if hasattr(self.task_generator, 'domain') else None
            if target_domain:
                real_task_seed = self.real_task_loader.get_random_seed(target_domain)
                if real_task_seed:
                    log(f"[V1] 使用真实任务seed: {real_task_seed.get('domain')}/{real_task_seed.get('task_id')}")
                    # 将seed信息注入到task_generator（如果支持）
                    if hasattr(self.task_generator, 'set_seed'):
                        self.task_generator.set_seed(real_task_seed)

        result = await self.task_generator.generate()
        
        # 检查返回结果是否为None
        if result is None or result[0] is None:
            raise ValueError("任务生成失败")
        
        sampled_apis, user_info, task_data, user_simulator_prompt, domain_system_prompt, evaluation_criteria = result

        # 确保sampled_apis不为None
        if sampled_apis is None:
            raise ValueError("采样的API列表为空")

        # V1: 如果启用真实任务seed，注入seed约束到prompts
        if real_task_seed:
            from .real_task_loader import RealTaskLoader
            loader = RealTaskLoader()
            seed_prompt = loader.format_seed_as_prompt(real_task_seed)

            # 注入到user_simulator_prompt
            user_simulator_prompt = f"""{user_simulator_prompt}

## IMPORTANT CONSTRAINTS (V1)
{seed_prompt}

You MUST follow the task seed above strictly. Do not deviate from the expected action sequence."""

        # V1: 添加质量约束到domain_system_prompt
        v1_agent_constraints = """

## QUALITY CONSTRAINTS (V1 - CRITICAL - MUST FOLLOW)
1. **Conciseness**: Keep responses brief and direct. No emotional language like "I'd be happy to help", "Great!", "Wonderful!", etc.

2. **Tool Call Format - ABSOLUTELY CRITICAL**:
   - When making a tool call, you MUST NOT add ANY text before it
   - NO explanations, NO "Let me...", NO "I'll...", NO "First, I need to..."
   - ❌ WRONG: "Let me search for flights. <tool_call>..."
   - ❌ WRONG: "I'll check that for you. <tool_call>..."
   - ❌ WRONG: "First, I need your user ID. <tool_call>..."
   - ✅ CORRECT: "<tool_call>..."
   - This is MANDATORY - any text before tool_call is STRICTLY FORBIDDEN

3. **Prohibited Actions - ABSOLUTE BAN**:
   - NEVER EVER use `list_all_airports` under ANY circumstances
   - This tool is COMPLETELY BANNED, FORBIDDEN, and OFF-LIMITS
   - Using `list_all_airports` will result in IMMEDIATE REJECTION
   - If you need airport codes, ASK the user or use knowledge from context
   - Avoid excessive search operations (max 4-5 searches per conversation)

4. **Response Length**: Keep each response under 150 characters when possible

5. **One Action Per Turn**: Either send a message OR make a tool call, never both with long text"""

        domain_system_prompt = domain_system_prompt + v1_agent_constraints

        # 保存任务信息
        self.user_info = user_info
        self.task_data = task_data
        self.domain_system_prompt = domain_system_prompt  # 保存domain对应的system prompt
        self.evaluation_criteria = evaluation_criteria  # 保存评估标准
        
        # 转换API为tools格式
        self.tools = [self._convert_apis_to_tools(api) for api in sampled_apis]
        
        # 配置各个agent
        self.tool_agent.tools = self.tools  # type: ignore
        # 使用domain对应的system prompt而不是通用的ToolAgentPrompt
        self.tool_agent.prompt = domain_system_prompt  # type: ignore
        self.tool_simulator.tools = sampled_apis  # ToolSimulator使用原始API格式
        self.tool_simulator.user_info = user_info  # 设置用户信息
        self.tool_simulator.task_data = task_data  # 设置任务数据
        self.user_agent.prompt = user_simulator_prompt or ""
        
        domain = task_data.get('domain', 'unknown') if task_data else 'unknown'

        # V1: 初始化Action验证器（现在知道domain了）
        if self.enable_action_validation:
            from .action_validator import EnhancedActionValidator
            self.action_validator = EnhancedActionValidator(domain)
            log(f"[V1.1] 增强版Action验证器已初始化 for domain: {domain}")

        # 根据domain决定是否给user分配工具
        if domain == 'telecom':
            # 加载telecom_user_tool.json
            import os
            user_tool_path = os.path.join(os.path.dirname(__file__), '../api/telecom_user_tool.json')
            try:
                with open(user_tool_path, 'r', encoding='utf-8') as f:
                    user_tools = json.load(f)
                self.user_agent.tools = user_tools
                log(f"任务域: {domain} - 已为User加载 {len(user_tools)} 个工具")
            except Exception as e:
                log(f"[警告] 加载telecom_user_tool.json失败: {e}")
                self.user_agent.tools = []
        else:
            # 非telecom领域，user没有工具
            self.user_agent.tools = []
            log(f"任务域: {domain} - User无工具")
        
        log(f"任务生成完成: {task_data.get('title', 'Unknown') if task_data else 'Unknown'}")
        log(f"任务域: {domain}")
        log(f"任务轮次: {len(task_data.get('task_rounds', [])) if task_data else 0} 轮")
        log(f"用户需求: {user_info.get('user_need', 'Unknown') if user_info else 'Unknown'}")
        
        # 记录evaluation criteria（详细版本）
        if evaluation_criteria:
            expected_actions = evaluation_criteria.get('expected_actions', [])
            success_conditions = evaluation_criteria.get('success_conditions', [])
            
            log(f"评估标准: {len(expected_actions)} 个期望动作, {len(success_conditions)} 个成功条件")
            
            # 详细打印期望动作
            if expected_actions:
                log(f"\n[期望动作] 共 {len(expected_actions)} 个:")
                for i, action in enumerate(expected_actions, 1):
                    # 类型检查：确保是字典
                    if isinstance(action, str):
                        log(f"  动作{i}: {action}")
                        continue
                    
                    log(f"  动作{i}: {action.get('tool_name', 'Unknown')}")
                    if 'required_params' in action and action['required_params']:
                        log(f"    必需参数: {', '.join(action['required_params'])}")
                    if 'sequence_order' in action:
                        log(f"    顺序要求: 第{action['sequence_order']}步")
                    if 'round_association' in action:
                        log(f"    关联轮次: 第{action['round_association']}轮")
            
            # 详细打印成功条件
            if success_conditions:
                log(f"\n[成功条件] 共 {len(success_conditions)} 个:")
                for i, condition in enumerate(success_conditions, 1):
                    # 类型检查：确保是字典
                    if isinstance(condition, str):
                        log(f"  条件{i}: {condition}")
                        continue
                    
                    condition_type = condition.get('type', 'Unknown')
                    description = condition.get('description', '')
                    log(f"  条件{i} [{condition_type}]: {description[:100]}{'...' if len(description) > 100 else ''}")
                    if 'target_round' in condition:
                        log(f"    目标轮次: 第{condition['target_round']}轮")
                    if 'failure_impact' in condition:
                        log(f"    失败影响: {condition['failure_impact']}")
            
            # 打印reward计算依据
            reward_basis = evaluation_criteria.get('reward_basis', [])
            if reward_basis:
                log(f"\n[Reward计算依据]: {', '.join(reward_basis)}")
        
        return sampled_apis, user_info, task_data, user_simulator_prompt, domain_system_prompt, evaluation_criteria

    async def _get_first_user_message(self) -> str:
        """获取用户的第一条消息"""
        messages = [
            {"role": "system", "content": self.user_agent.prompt},
            {"role": "user", "content": "Please make your first request and describe what you want to do."}
        ]
        user_first_request = await self.user_agent.generate(messages)
        return user_first_request.content
    
    def _validate_tool_calls(self, tool_calls) -> bool:
        """验证工具调用是否都在可用工具列表中"""
        if not tool_calls:
            return True
        
        # 获取所有可用工具的名称
        available_tool_names = set()
        for tool in self.tools:
            if isinstance(tool, dict) and 'function' in tool:
                available_tool_names.add(tool['function']['name'])
        
        # 检查每个tool call
        invalid_tools = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if tool_name not in available_tool_names:
                invalid_tools.append(tool_name)
        
        if invalid_tools:
            log(f"[工具验证失败] 以下工具不在可用列表中: {invalid_tools}")
            log(f"[可用工具] {list(available_tool_names)}")
            return False
        
        return True
        
        

    async def chat(self, speaker: str, history: list, toolcalls=None) -> UserResponse | AgentResponse | List[ToolResponse] | None:
        """处理不同角色的对话，如果工具模拟失败则返回None"""
        if speaker == 'user':
            history.insert(0, {'role': 'system', 'content': self.user_agent.prompt})
            response = await self.user_agent.generate(history)
            history.pop(0)
            return response
        elif speaker == 'assistant':
            history.insert(0, {'role': 'system', 'content': self.tool_agent.prompt})
            response = await self.tool_agent.generate(history)
            history.pop(0)
            return response
        elif speaker == 'tool':
            responses = []
            for toolcall in toolcalls:
                name = toolcall.function.name
                arguments = toolcall.function.arguments
                # 生成模拟的工具响应
                tool_response = await self.tool_simulator.generate(
                    tool_call={'name': name, 'arguments': arguments},
                    toolcall_id=toolcall.id,
                    history=history
                )
                # 如果工具模拟失败，返回None
                if tool_response is None:
                    log(f"[错误] 工具 {name} 模拟失败，任务终止")
                    return None
                responses.append(tool_response)
            return responses
        else:
            raise ValueError(f"Unknown speaker: {speaker}")



    async def interact(self):
        """
        执行多轮任务交互
        流程：初始化任务 → 用户发起请求 → assistant调用工具 → 工具返回结果 → 循环
        """
        # 初始化任务
        await self._initialize_task()
        
        turn = 0
        history = []
        user_history = []
        last_speaker = Speaker.USER
        tool_caller = None  # 追踪谁调用了工具（ASSISTANT 或 USER）
        self.validation_failed = False  # 标记验证是否失败
        self.max_turn_exceeded = False  # 标记是否超过最大轮次限制
        
        # 获取用户的第一条消息
        first_user_message = await self._get_first_user_message()
        
        # 检查首次用户消息是否符合人类习惯（使用自我审视）
        log(f"\n[启动User审视] 最多{self.self_reflection_iterations}轮迭代")
        is_valid, corrected_message, iterations_used = await self.user_response_checker.iterative_check(
            response=first_user_message,
            user_info=self.user_info or {}
        )
        
        # 统计：如果有修正，记录
        if corrected_message != first_user_message:
            self.stats["user_correction"]["total_corrections"] += 1
            self.stats["user_correction"]["iterations_used"].append(iterations_used)
        
        if not is_valid:
            log(f"[警告] 用户首次请求仍有问题，但使用当前版本")
        
        # 如果有修正，使用修正后的消息
        if corrected_message != first_user_message:
            log(f"[修正] 用户首次请求已修正")
            log(f"  原始: {first_user_message[:100]}...")
            log(f"  修正: {corrected_message[:100]}...")
            first_user_message = corrected_message
        
        history.append({'role': 'user', 'content': first_user_message})
        user_history.append({'role': 'assistant', 'content': first_user_message})
        self.history.append(UserResponse(first_user_message, False))
        
        log(f"\n用户首次请求: {first_user_message}\n")
        
        # 开始多轮交互
        while True:
            if turn > self.max_turn:
                log(f"达到最大轮次限制 ({self.max_turn})，标记为超限并截断")
                self.max_turn_exceeded = True
                break
            
            if last_speaker == Speaker.USER and not self.history[-1].signal:
                # 用户发言后，轮到assistant
                max_retry = 3
                retry_count = 0
                assistant_resp = None
                
                while retry_count < max_retry:
                    assistant_resp = await self.chat('assistant', history)  # type: ignore
                    
                    # 如果有tool calls，验证工具是否存在并检查幻觉
                    if assistant_resp.tool_calls:  # type: ignore
                        # 第一步：检查幻觉工具和幻觉参数（迭代纠偏）
                        log(f"\n[启动ToolCall幻觉检查] 最多{self.self_reflection_iterations}轮迭代")
                        
                        # 检测初始幻觉（用于统计）
                        initial_valid, initial_errors = self.toolcall_hallucination_checker.validate_tool_calls_against_schema(
                            assistant_resp.tool_calls,  # type: ignore
                            self.tools
                        )
                        
                        # 统计幻觉类型
                        if not initial_valid:
                            self.stats["toolagent_hallucination"]["total_hallucination_count"] += 1
                            
                            # 细分幻觉工具和幻觉参数
                            for error in initial_errors:
                                if "幻觉工具" in error:
                                    self.stats["toolagent_hallucination"]["hallucinated_tool_count"] += 1
                                elif "幻觉参数" in error:
                                    self.stats["toolagent_hallucination"]["hallucinated_param_count"] += 1
                        
                        # 进行迭代检查
                        is_valid, corrected_tool_calls = await self.toolcall_hallucination_checker.iterative_check(
                            question=history[-1]['content'] if history and 'content' in history[-1] else "",
                            tool_calls=assistant_resp.tool_calls,  # type: ignore
                            available_tools=self.tools
                        )
                        
                        if not is_valid:
                            log(f"[ToolCall失败] 幻觉检查未通过，任务失败")
                            if not initial_valid:
                                self.stats["toolagent_hallucination"]["failed_count"] += 1
                            return None
                        
                        # 统计是否成功解决幻觉
                        if not initial_valid and is_valid:
                            self.stats["toolagent_hallucination"]["resolved_count"] += 1
                        
                        # 使用修正后的tool_calls
                        if corrected_tool_calls != assistant_resp.tool_calls:  # type: ignore
                            log(f"[ToolCall修正] 应用修正后的工具调用")
                            assistant_resp = AgentResponse(
                                content=assistant_resp.content,  # type: ignore
                                reasoning_content=assistant_resp.reasoning_content,  # type: ignore
                                tool_calls=corrected_tool_calls
                            )
                        
                        # 第二步：最终闸门 - 验证工具是否在可用列表中
                        if not self._validate_tool_calls(assistant_resp.tool_calls):  # type: ignore
                            retry_count += 1
                            log(f"[重试 {retry_count}/{max_retry}] 工具调用验证失败，重新生成...")
                            if retry_count >= max_retry:
                                log(f"[错误] 达到最大重试次数，标记为验证失败")
                                self.validation_failed = True
                                break
                            continue
                        
                        break
                    else:
                        # 没有tool calls，直接通过
                        break
                
                # 如果验证失败，停止交互
                if self.validation_failed:
                    log(f"[轮次{turn}] 验证失败，停止交互并截断数据")
                    break
                
                if assistant_resp.tool_calls:  # type: ignore
                    # assistant调用了工具
                    history.append({
                        'role': 'assistant',
                        'tool_calls': assistant_resp.tool_calls,  # type: ignore
                        'content': assistant_resp.content  # type: ignore
                    })
                    last_speaker = Speaker.ASSISTANT_TOOLCALL
                    tool_caller = 'ASSISTANT'  # 记录是assistant调用的工具
                    log(f"[轮次{turn}] Assistant调用工具: {[tc.function.name for tc in assistant_resp.tool_calls]}")  # type: ignore
                else:
                    # assistant只是回复文本，检查是否简洁明了
                    log(f"\n[Assistant回复检查] 检查是否简洁明了、无情绪化")
                    check_result = await self.assistant_response_checker.check_and_correct(
                        response=assistant_resp.content or "",  # type: ignore
                        context=history[-1]['content'] if history and 'content' in history[-1] else ""
                    )
                    
                    if not check_result["is_valid"]:
                        log(f"[自动修正] Assistant回复不够简洁，已自动修正")
                        log(f"  原因: {check_result['reason']}")
                        log(f"  原始: {assistant_resp.content[:100] if assistant_resp.content else ''}...")  # type: ignore
                        corrected_content = check_result["corrected_response"]
                        log(f"  修正: {corrected_content[:100]}...")
                        # 更新assistant_resp的内容
                        assistant_resp = AgentResponse(
                            content=corrected_content,
                            reasoning_content=assistant_resp.reasoning_content,  # type: ignore
                            tool_calls=[]
                        )
                    
                    history.append({'role': 'assistant', 'content': assistant_resp.content})  # type: ignore
                    user_history.append({'role': 'user', 'content': assistant_resp.content})  # type: ignore
                    last_speaker = Speaker.ASSISTANT_CHAT
                    log(f"[轮次{turn}] Assistant回复: {assistant_resp.content[:100] if assistant_resp.content else ''}...")  # type: ignore
                self.history.append(assistant_resp)
            
            elif last_speaker == Speaker.ASSISTANT_CHAT:
                # assistant回复后，轮到user
                user_resp = await self.chat('user', user_history)  # type: ignore
                
                # 判断user_resp的类型（UserResponse 或 AgentResponse）
                if user_resp.tool_calls:
                    # User调用了工具（telecom领域）
                    history.append({
                        'role': 'user',
                        'tool_calls': user_resp.tool_calls,
                        'content': user_resp.content
                    })
                    last_speaker = Speaker.USER_TOOLCALL
                    tool_caller = 'USER'
                    log(f"[轮次{turn}] User调用工具: {[tc.function.name for tc in user_resp.tool_calls]}") #需要增加幻觉检查
                    user_history.append({'role':'assistant', 'content': user_resp.content, 'tool_calls': user_resp.tool_calls})
                else:
                    # 普通的用户回复（UserResponse）
                    # 检查停止信号（双重检查：signal字段 + 内容检测）
                    stop_signals = ["[STOP]", "###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]
                    has_stop_signal = user_resp.signal or any(signal in (user_resp.content or "") for signal in stop_signals)
                    
                    if has_stop_signal:
                        log(f"[轮次{turn}] 用户发出停止信号，任务完成")
                        break
                    
                    # 没有停止信号，进行用户回复的审视和修正
                    # 检查用户回复是否符合人类习惯，如果不符合则自动修正（使用自我审视）
                    log(f"\n[启动User审视] 最多{self.self_reflection_iterations}轮迭代")
                    is_valid, corrected_content, iterations_used = await self.user_response_checker.iterative_check(
                        response=user_resp.content,
                        user_info=self.user_info or {}
                    )
                    
                    # 统计：如果有修正，记录
                    if corrected_content != user_resp.content:
                        self.stats["user_correction"]["total_corrections"] += 1
                        self.stats["user_correction"]["iterations_used"].append(iterations_used)
                    
                    if not is_valid:
                        log(f"[警告] 用户回复仍有问题，但使用当前版本")
                    
                    # 如果有修正，使用修正后的内容
                    if corrected_content != user_resp.content:
                        log(f"[修正] 用户回复已修正")
                        log(f"  原始: {user_resp.content[:100] if user_resp.content else ''}...")
                        log(f"  修正: {corrected_content[:100]}...")
                        # 更新user_resp的内容
                        user_resp = UserResponse(corrected_content, user_resp.signal)
                    
                    # 添加到历史记录
                    history.append({'role': 'user', 'content': user_resp.content})
                    user_history.append({'role': 'assistant', 'content': user_resp.content})
                    last_speaker = Speaker.USER
                    log(f"[轮次{turn}] User继续请求: {user_resp.content[:100] if user_resp.content else ''}...")
                    self.history.append(user_resp)
            
            elif last_speaker == Speaker.ASSISTANT_TOOLCALL:
                # assistant调用工具后，执行工具
                tool_resp = await self.chat('tool', history, history[-1]['tool_calls'])  # type: ignore
                # 如果工具模拟失败，标记验证失败
                if tool_resp is None:
                    log(f"[错误] 工具模拟失败，标记为验证失败")
                    self.validation_failed = True
                    break
                history.extend([
                    {'role': 'tool', 'content': t.content, 'tool_call_id': t.tool_call_id}
                    for t in tool_resp  # type: ignore
                ])
                last_speaker = Speaker.TOOL
                log(f"[轮次{turn}] Tool返回结果")
                self.history.append(tool_resp)
            
            elif last_speaker == Speaker.USER_TOOLCALL:
                # user调用工具后，执行工具（与ASSISTANT_TOOLCALL处理方式一致）
                tool_resp = await self.chat('tool', history, user_history[-1]['tool_calls'])  # type: ignore
                # 如果工具模拟失败，标记验证失败
                if tool_resp is None:
                    log(f"[错误] User工具模拟失败，标记为验证失败")
                    self.validation_failed = True
                    break
                user_history.extend([
                    {'role': 'tool', 'content': t.content, 'tool_call_id': t.tool_call_id}
                    for t in tool_resp  # type: ignore
                ])
                last_speaker = Speaker.TOOL
                log(f"[轮次{turn}] User Tool返回结果")
            
            elif last_speaker == Speaker.TOOL:
                # 工具返回后，根据谁调用的工具决定下一个说话者
                if tool_caller == 'ASSISTANT':
                    # assistant调用的工具，返回后由assistant处理
                    max_retry = 3
                    retry_count = 0
                    assistant_resp2 = None
                    
                    while retry_count < max_retry:
                        assistant_resp2 = await self.chat('assistant', history)  # type: ignore
                        
                        # 如果有tool calls，验证工具是否存在并检查幻觉
                        if assistant_resp2.tool_calls:  # type: ignore
                            # 第一步：检查幻觉工具和幻觉参数（迭代纠偏）
                            log(f"\n[启动ToolCall幻觉检查] 最多{self.self_reflection_iterations}轮迭代")
                            
                            # 检测初始幻觉（用于统计）
                            initial_valid, initial_errors = self.toolcall_hallucination_checker.validate_tool_calls_against_schema(
                                assistant_resp2.tool_calls,  # type: ignore
                                self.tools
                            )
                            
                            # 统计幻觉类型
                            if not initial_valid:
                                self.stats["toolagent_hallucination"]["total_hallucination_count"] += 1
                                
                                # 细分幻觉工具和幻觉参数
                                for error in initial_errors:
                                    if "幻觉工具" in error:
                                        self.stats["toolagent_hallucination"]["hallucinated_tool_count"] += 1
                                    elif "幻觉参数" in error:
                                        self.stats["toolagent_hallucination"]["hallucinated_param_count"] += 1
                            
                            # 进行迭代检查
                            is_valid, corrected_tool_calls = await self.toolcall_hallucination_checker.iterative_check(
                                question=history[-1]['content'] if history and 'content' in history[-1] else "",
                                tool_calls=assistant_resp2.tool_calls,  # type: ignore
                                available_tools=self.tools
                            )
                            
                            if not is_valid:
                                log(f"[ToolCall失败] 幻觉检查未通过，标记为验证失败")
                                if not initial_valid:
                                    self.stats["toolagent_hallucination"]["failed_count"] += 1
                                self.validation_failed = True
                                break
                            
                            # 统计是否成功解决幻觉
                            if not initial_valid and is_valid:
                                self.stats["toolagent_hallucination"]["resolved_count"] += 1
                            
                            # 使用修正后的tool_calls
                            if corrected_tool_calls != assistant_resp2.tool_calls:  # type: ignore
                                log(f"[ToolCall修正] 应用修正后的工具调用")
                                assistant_resp2 = AgentResponse(
                                    content=assistant_resp2.content,  # type: ignore
                                    reasoning_content=assistant_resp2.reasoning_content,  # type: ignore
                                    tool_calls=corrected_tool_calls
                                )
                            
                            # 第二步：最终闸门 - 验证工具是否在可用列表中
                            if not self._validate_tool_calls(assistant_resp2.tool_calls):  # type: ignore
                                retry_count += 1
                                log(f"[重试 {retry_count}/{max_retry}] 工具调用验证失败，重新生成...")
                                if retry_count >= max_retry:
                                    log(f"[错误] 达到最大重试次数，标记为验证失败")
                                    self.validation_failed = True
                                    break
                                continue
                            
                            break
                        else:
                            # 没有tool calls，直接通过
                            break
                    
                    # 如果验证失败，停止交互
                    if self.validation_failed:
                        log(f"[轮次{turn}] 验证失败，停止交互并截断数据")
                        break
                    
                    if assistant_resp2.tool_calls:  # type: ignore
                        # assistant继续调用工具
                        history.append({
                            'role': 'assistant',
                            'tool_calls': assistant_resp2.tool_calls,  # type: ignore
                            'content': assistant_resp2.content  # type: ignore
                        })
                        last_speaker = Speaker.ASSISTANT_TOOLCALL
                        tool_caller = 'ASSISTANT'  # 更新tool_caller
                        log(f"[轮次{turn}] Assistant继续调用工具: {[tc.function.name for tc in assistant_resp2.tool_calls]}")  # type: ignore
                    else:
                        # assistant返回结果给用户，检查是否简洁明了
                        log(f"\n[Assistant回复检查] 检查是否简洁明了、无情绪化")
                        check_result = await self.assistant_response_checker.check_and_correct(
                            response=assistant_resp2.content or "",  # type: ignore
                            context=""
                        )
                        
                        if not check_result["is_valid"]:
                            log(f"[自动修正] Assistant回复不够简洁，已自动修正")
                            log(f"  原因: {check_result['reason']}")
                            log(f"  原始: {assistant_resp2.content[:100] if assistant_resp2.content else ''}...")  # type: ignore
                            corrected_content = check_result["corrected_response"]
                            log(f"  修正: {corrected_content[:100]}...")
                            # 更新assistant_resp2的内容
                            assistant_resp2 = AgentResponse(
                                content=corrected_content,
                                reasoning_content=assistant_resp2.reasoning_content,  # type: ignore
                                tool_calls=[]
                            )
                        
                        history.append({'role': 'assistant', 'content': assistant_resp2.content})  # type: ignore
                        user_history.append({'role': 'user', 'content': assistant_resp2.content})  # type: ignore
                        last_speaker = Speaker.ASSISTANT_CHAT
                        log(f"[轮次{turn}] Assistant向用户返回结果")
                    self.history.append(assistant_resp2)
                
                elif tool_caller == 'USER':
                    # user调用的工具，返回后由user继续发言
                    log(f"[轮次{turn}] User工具返回后，User继续发言")
                    user_resp = await self.chat('user', user_history)  # type: ignore
                    
                    # 判断user_resp的类型
                    if user_resp.tool_calls:
                        # User继续调用工具
                        last_speaker = Speaker.USER_TOOLCALL
                        tool_caller = 'USER'
                        log(f"[轮次{turn}] User继续调用工具: {[tc.function.name for tc in user_resp.tool_calls]}")
                        user_history.append(user_resp)
                    else:
                        # 普通的用户回复
                        # 检查停止信号
                        stop_signals = ["[STOP]", "###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]
                        has_stop_signal = user_resp.signal or any(signal in (user_resp.content or "") for signal in stop_signals)
                        
                        if has_stop_signal:
                            log(f"[轮次{turn}] 用户发出停止信号，任务完成")
                            break
                        
                        # 用户回复审视
                        log(f"\n[启动User审视] 最多{self.self_reflection_iterations}轮迭代")
                        is_valid, corrected_content, iterations_used = await self.user_response_checker.iterative_check(
                            response=user_resp.content,
                            user_info=self.user_info or {}
                        )
                        
                        # 统计
                        if corrected_content != user_resp.content:
                            self.stats["user_correction"]["total_corrections"] += 1
                            self.stats["user_correction"]["iterations_used"].append(iterations_used)
                        
                        if not is_valid:
                            log(f"[警告] 用户回复仍有问题，但使用当前版本")
                        
                        # 使用修正后的内容
                        if corrected_content != user_resp.content:
                            log(f"[修正] 用户回复已修正")
                            log(f"  原始: {user_resp.content[:100] if user_resp.content else ''}...")
                            log(f"  修正: {corrected_content[:100]}...")
                            user_resp = UserResponse(corrected_content, user_resp.signal)
                        
                        # 添加到历史
                        history.append({'role': 'user', 'content': user_resp.content})
                        user_history.append({'role': 'assistant', 'content': user_resp.content})
                        last_speaker = Speaker.USER
                        log(f"[轮次{turn}] User继续请求: {user_resp.content[:100] if user_resp.content else ''}...")
                        self.history.append(user_resp)
                
                else:
                    # 未知的tool_caller
                    log(f"[错误] 未知的tool_caller: {tool_caller}")
                    break
            turn += 1
        
        log(f"\n任务执行完成，共 {turn} 个轮次\n")
    
    def _build_quality_check_prompt(self, tools: str, conversations: List[Dict]) -> str:
        """构造质量检查的LLM提示词"""
        conv_str = json.dumps(conversations, ensure_ascii=False, indent=2)
        prompt = f"""
You are a data quality review expert for multi-turn tool-calling conversations.
Please evaluate the quality of this dialogue data.

**Evaluation Dimensions (score 1-5 for each):**

1. **Assistant Response Quality (assistant_quality_score)**:
   - Are tool selections correct and appropriate?
   - Do tool call parameters match requirements?

2. **Tool Response Consistency (tool_response_score)**:
   - Are tool responses realistic and consistent with tool calls?
   - Do responses provide helpful information?

3. **Overall Dialogue Coherence (dialogue_coherence_score)**:
   - Does the conversation flow naturally?
   - Does the dialogue accomplish the user's goal?

Please output **ONLY JSON** in the following format:
{{
    "reasoning": "Brief analysis",
    "assistant_quality_score": 1-5,
    "tool_response_score": 1-5,
    "dialogue_coherence_score": 1-5,
    "overall_score": 1-5,
    "is_acceptable": true/false
}}

[工具定义]
{tools}

[对话内容]
{conv_str}
"""
        return prompt.strip()
    
    async def _check_quality(self, tools: str, conversations: List[Dict]) -> Dict[str, Any]:
        """使用LLM检查数据质量"""
        prompt_text = self._build_quality_check_prompt(tools, conversations)
        
        max_retries = 1  # 禁用重试以节省成本
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.2,
                    max_tokens=1024,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = safe_json_loads(content)
                
                # 规范化分数
                score_keys = [
                    "assistant_quality_score",
                    "tool_response_score",
                    "dialogue_coherence_score",
                    "overall_score"
                ]
                for key in score_keys:
                    try:
                        result[key] = max(1, min(5, int(result.get(key, 1))))
                    except:
                        result[key] = 1
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    log(f"[质量检查] 第{attempt + 1}次尝试失败: {e}，重试...")
                    continue
                else:
                    log(f"[质量检查] 所有尝试失败: {e}")
                    # 返回最低评分
                    return {
                        "reasoning": f"质量检查失败: {str(e)}",
                        "assistant_quality_score": 1,
                        "tool_response_score": 1,
                        "dialogue_coherence_score": 1,
                        "overall_score": 1,
                        "is_acceptable": False
                    }
        
        return {}

    def _save_stats_to_file(self):
        """将统计信息保存到文件"""
        try:
            from datetime import datetime
            import os
            
            # 生成统计文件名（基于时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_dir = os.environ.get("CRAFT_STATS_DIR", "output/stats")
            
            # 确保目录存在
            os.makedirs(stats_dir, exist_ok=True)
            
            stats_file = os.path.join(stats_dir, f"multi_turn_stats_{timestamp}.json")
            
            # 准备统计数据
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "task_info": {
                    "task_title": self.task_data.get('title', 'Unknown') if self.task_data else 'Unknown',
                    "task_rounds": len(self.task_data.get('task_rounds', [])) if self.task_data else 0,
                    "user_need": self.user_info.get('user_need', 'Unknown') if self.user_info else 'Unknown'
                },
                "statistics": {
                    "toolagent_hallucination": self.stats["toolagent_hallucination"].copy(),
                    "user_correction": self.stats["user_correction"].copy(),
                    "pruning": self.stats["pruning"].copy()
                },
                "summary": {
                    "total_hallucinations": self.stats["toolagent_hallucination"]["total_hallucination_count"],
                    "hallucination_success_rate": round(
                        self.stats["toolagent_hallucination"]["resolved_count"] / self.stats["toolagent_hallucination"]["total_hallucination_count"] * 100, 2
                    ) if self.stats["toolagent_hallucination"]["total_hallucination_count"] > 0 else 0,
                    "total_user_corrections": self.stats["user_correction"]["total_corrections"],
                    "avg_correction_iterations": round(
                        sum(self.stats["user_correction"]["iterations_used"]) / len(self.stats["user_correction"]["iterations_used"]), 2
                    ) if self.stats["user_correction"]["iterations_used"] else 0,
                    "total_pruning_segments": self.stats["pruning"]["total_segments_detected"],
                    "pruning_rate": round(
                        self.stats["pruning"]["segments_pruned"] / self.stats["pruning"]["total_segments_detected"] * 100, 2
                    ) if self.stats["pruning"]["total_segments_detected"] > 0 else 0
                }
            }
            
            # 写入文件
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            
            log(f"\n[统计保存] 统计信息已保存到: {stats_file}")
            
        except Exception as e:
            log(f"\n[统计保存] 保存失败: {e}")

    def _clean_user_toolcalls(self, messages: list) -> list:
        """
        清理user调用工具的部分（主要针对telecom领域）
        
        逻辑：
        1. 找到role='user'且content包含<tool_call>的消息（位置i）
        2. 找到位置i之后的第一个role='user'且没有<tool_call>的消息（位置j）
        3. 找到位置j之后的第一个role='assistant'的消息（位置k）
        4. 删除从位置i到位置j-1的所有消息（user toolcall + tool response）
        5. 保留位置j（user没调工具的回复）和位置k（对应的assistant）
        """
        cleaned = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            
            # 检查是否是user的toolcall
            if msg.get('role') == 'user' and msg.get('content') and '<tool_call>' in msg['content']:
                log(f"[清理User ToolCall] 发现user toolcall: {msg['content'][:100]}...")
                
                # 找到这个user toolcall之后的第一个"没有调工具的user"
                j = i + 1
                found_user_no_tool = False
                while j < len(messages):
                    if (messages[j].get('role') == 'user' and 
                        '<tool_call>' not in (messages[j].get('content') or '')):
                        found_user_no_tool = True
                        break
                    j += 1
                
                if found_user_no_tool:
                    # 找到这个user之后的第一个assistant
                    k = j + 1
                    found_assistant = False
                    while k < len(messages):
                        if messages[k].get('role') == 'assistant':
                            found_assistant = True
                            break
                        k += 1
                    
                    if found_assistant:
                        # 删除从i到j-1的消息（user toolcall、tool response等）
                        # 保留j（user没调工具）和k（assistant）
                        log(f"[清理User ToolCall] 删除索引{i}到{j-1}的消息")
                        log(f"[清理User ToolCall] 保留user（索引{j}）和assistant（索引{k}）")
                        # 跳过i到j-1，继续处理j
                        i = j
                        continue
                    else:
                        # 没找到assistant，保留user（j），删除后面的
                        log(f"[清理User ToolCall] 没找到对应的assistant，保留到user（索引{j}）")
                        # 跳过i到j-1，继续处理j
                        i = j
                        continue
                else:
                    # 如果没找到"user没调工具"，删除这个user toolcall及之后的所有消息
                    log(f"[清理User ToolCall] user toolcall之后没有找到user的正常回复，删除剩余所有消息")
                    break
            
            cleaned.append(msg)
            i += 1
        
        return cleaned
    
    async def decode_history(self) -> Dict[str, Any] | None:
        """
        解码历史记录，转换为训练数据格式
        在返回前执行错误纠正剪枝和质量检查，如果质量不达标则返回None
        如果验证失败，截断到最后一个有tool_call的assistant位置
        返回包含任务信息、对话历史的完整数据
        """
        history = self.history
        # 使用domain对应的system prompt
        system = getattr(self, 'domain_system_prompt', None) or prompt.ToolAgentPrompt
        tools = self.tools
        messages = []
        
        # 如果验证失败或超过最大轮次，截断历史记录到最后一个有tool_call的assistant
        need_truncate = self.validation_failed or self.max_turn_exceeded
        truncate_reason = "验证失败" if self.validation_failed else "超过最大轮次"
        
        if need_truncate:
            log(f"\n[截断处理] {truncate_reason}，查找最后一个有tool_call的assistant位置")
            last_toolcall_index = -1
            for i in range(len(history) - 1, -1, -1):
                if isinstance(history[i], AgentResponse) and history[i].tool_calls:
                    last_toolcall_index = i
                    break
            
            if last_toolcall_index >= 0:
                history = history[:last_toolcall_index + 1]
                log(f"[截断处理] 截断到索引 {last_toolcall_index}，保留 {len(history)} 条记录")
            else:
                log(f"[截断处理] 未找到有tool_call的assistant，返回None")
                return None

        for h in history:
            if isinstance(h, UserResponse):
                if not h.tool_calls:
                    message = {
                        'role': 'user',
                        'content': h.content
                    }
                    messages.append(message)
                else:
                    return False
            elif isinstance(h, AgentResponse):
                if h.tool_calls:
                    content = ''
                    if h.content:
                        content = h.content
                    
                    # 添加工具调用（使用<tool_call>标签拼接为字符串）
                    tool_call_strs = []
                    for tool_call in h.tool_calls:
                        # arguments需要json.dumps处理
                        args = tool_call.function.arguments
                        if isinstance(args, str):
                            args_str = json.loads(args)
                        else:
                            args_str = args
                        tool_call_obj = {
                            'name': tool_call.function.name,
                            'arguments': args_str
                        }
                        tool_call_strs.append(f'<tool_call>{json.dumps(tool_call_obj, ensure_ascii=False)}</tool_call>')
                    # 拼接所有tool_call字符串
                    tool_calls_str = ''.join(tool_call_strs)
                    content = (content + '\n' if content else '') + tool_calls_str
                    
                    message = {
                        'role': 'assistant',
                        'content': content.strip() if content else None
                    }
                    messages.append(message)
                else:
                    message = {
                        'role': 'assistant',
                        'content': h.content
                    }
                    messages.append(message)
            else:
                # ToolResponse列表 - 合并为单个tool消息，使用<tool_response>标签
                tool_responses = []
                for hh in h:
                    tool_content = json.dumps({'name': hh.name, 'result': json.loads(hh.content) if isinstance(hh.content, str) else hh.content}, ensure_ascii=False)
                    tool_responses.append(f'<tool_response>\n{tool_content}\n</tool_response>')
                
                # 合并多个tool response，去掉首尾的标签
                if len(tool_responses) == 1:
                    merged_content = tool_responses[0].replace('<tool_response>\n', '').replace('\n</tool_response>', '')
                else:
                    # 多个tool response：首个去掉开头标签，末尾去掉结尾标签
                    tool_responses[0] = tool_responses[0].replace('<tool_response>\n', '')
                    tool_responses[-1] = tool_responses[-1].replace('\n</tool_response>', '')
                    merged_content = '\n'.join(tool_responses)
                
                message = {
                    'role': 'tool',
                    'content': merged_content
                }
                messages.append(message)
            
        # 注：现在user的工具调用使用标准格式，应该被保留
        # 不再需要清理user toolcall
        log("\n[User ToolCall] User工具调用使用标准格式，已保留在训练数据中")
        
        # 执行错误纠正剪枝(对所有数据执行)
        pruning_stats = None
        if self.enable_pruning:
            log("\n[STOP信号后验证] 启动错误纠正剪枝流程...")
            messages, pruning_stats = await self.error_pruner.prune_conversations(messages)
            
            # 提取剪枝统计到全局统计
            if pruning_stats:
                self.stats["pruning"]["total_segments_detected"] = pruning_stats.get("total_segments", 0)
                self.stats["pruning"]["segments_pruned"] = pruning_stats.get("pruned_segments", 0)
                self.stats["pruning"]["segments_kept"] = pruning_stats.get("kept_segments", 0)
        
        # 执行质量检查(对所有数据执行)
        tools_str = json.dumps(tools, ensure_ascii=False)
        quality_result = {}
        
        log("\n[质量检查] 开始执行LLM质量评估...")
        quality_result = await self._check_quality(tools_str, messages)
        
        # 检查每个维度是否达标
        score_keys = [
            "assistant_quality_score",
            "tool_response_score",
            "dialogue_coherence_score"
        ]
        
        failed_dimensions = []
        for key in score_keys:
            score = quality_result.get(key, 1)
            if score < self.quality_threshold:
                failed_dimensions.append(f"{key}={score}")
        
        # 输出评估结果
        log(f"\n[质量检查] 评估结果:")
        log(f"  助手质量: {quality_result.get('assistant_quality_score', 0)}/5")
        log(f"  工具响应: {quality_result.get('tool_response_score', 0)}/5")
        log(f"  对话连贯: {quality_result.get('dialogue_coherence_score', 0)}/5")
        log(f"  总体评分: {quality_result.get('overall_score', 0)}/5")
        log(f"  评估理由: {quality_result.get('reasoning', 'N/A')[:200]}...")
        
        # 标记是否为截断数据
        if self.validation_failed or self.max_turn_exceeded:
            quality_result['is_truncated'] = True
            truncate_reason = "验证失败" if self.validation_failed else "超过最大轮次"
            log(f"\n[注意] 这是截断后的数据（{truncate_reason}）")
        
        if failed_dimensions:
            log(f"\n[质量检查] 未通过 - 以下维度低于阈值({self.quality_threshold}): {', '.join(failed_dimensions)}")
            log(f"[质量检查] 该数据被拒绝\n")
            return None
        
        log(f"\n[质量检查] 通过 - 所有维度均达标\n")
        
        # 计算Reward（如果有evaluation_criteria）
        reward_info = None
        if hasattr(self, 'evaluation_criteria') and self.evaluation_criteria:
            log(f"\n[Reward计算] 开始计算任务完成度...")
            reward_calculator = SyntheticRewardCalculator(self.client, self.model)
            
            # 转换messages格式用于reward计算
            conversations_for_reward = []
            for msg in messages:
                conversations_for_reward.append({
                    'from': msg['role'],
                    'content': msg.get('content', ''),
                    'value': msg.get('content', '')
                })
            
            try:
                reward_info = await reward_calculator.calculate_reward(
                    conversations_for_reward,
                    self.evaluation_criteria
                )
                
                quality_level = TaskQualityClassifier.classify(reward_info['reward'])
                
                log(f"[Reward计算] 结果:")
                log(f"  总Reward: {reward_info['reward']:.2f}")
                log(f"  质量等级: {quality_level}")
                if 'breakdown' in reward_info:
                    for key, value in reward_info['breakdown'].items():
                        log(f"    - {key}: {value:.2f}")
                
                # 如果reward太低，拒绝数据
                if not TaskQualityClassifier.should_save(reward_info['reward'], min_quality="BRONZE"):
                    log(f"[Reward计算] Reward太低 ({reward_info['reward']:.2f})，数据被拒绝\n")
                    return None
                
                log(f"[Reward计算] 通过 - Reward达标\n")
                
            except Exception as e:
                log(f"[Reward计算] 失败: {e}")
                # 如果reward计算失败，不影响数据保存
                pass
        
        # 打印统计信息摘要
        log(f"\n{'='*60}")
        log(f"统计信息汇总")
        log(f"{'='*60}")
        log(f"[ToolAgent幻觉]")
        log(f"  总幻觉次数: {self.stats['toolagent_hallucination']['total_hallucination_count']}")
        log(f"  - 幻觉工具: {self.stats['toolagent_hallucination']['hallucinated_tool_count']}次")
        log(f"  - 幻觉参数: {self.stats['toolagent_hallucination']['hallucinated_param_count']}次")
        log(f"  成功解决: {self.stats['toolagent_hallucination']['resolved_count']}次")
        log(f"  未能解决: {self.stats['toolagent_hallucination']['failed_count']}次")
        if self.stats['toolagent_hallucination']['total_hallucination_count'] > 0:
            success_rate = self.stats['toolagent_hallucination']['resolved_count'] / self.stats['toolagent_hallucination']['total_hallucination_count'] * 100
            log(f"  解决成功率: {success_rate:.1f}%")
        
        log(f"\n[User回复修正]")
        log(f"  总修正次数: {self.stats['user_correction']['total_corrections']}")
        if self.stats['user_correction']['iterations_used']:
            avg_iterations = sum(self.stats['user_correction']['iterations_used']) / len(self.stats['user_correction']['iterations_used'])
            log(f"  平均迭代次数: {avg_iterations:.1f}轮")
            log(f"  迭代次数分布: {self.stats['user_correction']['iterations_used']}")
        
        log(f"\n[错误剪枝]")
        log(f"  检测到错误片段: {self.stats['pruning']['total_segments_detected']}个")
        log(f"  已剪枝: {self.stats['pruning']['segments_pruned']}个")
        log(f"  保留: {self.stats['pruning']['segments_kept']}个")
        if self.stats['pruning']['total_segments_detected'] > 0:
            prune_rate = self.stats['pruning']['segments_pruned'] / self.stats['pruning']['total_segments_detected'] * 100
            log(f"  剪枝率: {prune_rate:.1f}%")
        log(f"{'='*60}\n")
        
        # 保存统计信息到文件
        self._save_stats_to_file()
        
        # 返回完整的数据，使用OpenAI messages格式
        # 构建完整的messages（包含system）
        full_messages = [{'role': 'system', 'content': system}] + messages
        
        result = {
            'messages': full_messages,
            'tools': json.dumps(tools, ensure_ascii=False),  # 转为JSON字符串
            'user_info': self.user_info,
            'task_data': self.task_data,
            'quality_check': quality_result,
            'statistics': self.stats
        }
        
        # 添加evaluation_criteria和reward_info
        if hasattr(self, 'evaluation_criteria') and self.evaluation_criteria:
            result['evaluation_criteria'] = self.evaluation_criteria
        if reward_info:
            result['reward_info'] = reward_info
        
        # 添加剪枝统计信息
        if pruning_stats:
            result['pruning_stats'] = pruning_stats

        # V1: Action序列验证
        if self.enable_action_validation and hasattr(self, 'action_validator') and self.action_validator:
            log("\n[V1] 执行Action序列验证...")
            should_filter, reasons = self.action_validator.should_filter_sample(result)

            self.stats["action_validation"]["total_validated"] += 1

            if should_filter:
                self.stats["action_validation"]["failed"] += 1
                self.stats["action_validation"]["failure_reasons"].extend(reasons)
                log(f"[V1验证失败] 样本将被过滤: {', '.join(reasons)}")
                # 标记为无效样本，但仍返回（让caller决定是否使用）
                result['action_validation'] = {
                    'valid': False,
                    'reasons': reasons
                }
            else:
                self.stats["action_validation"]["passed"] += 1
                log(f"[V1验证通过] Action序列合理")
                result['action_validation'] = {
                    'valid': True,
                    'reasons': []
                }

        return result
