import asyncio
import json
import random
import openai
from pathlib import Path
from ..prompts import prompt as local_prompt  # prompts目录下的prompt
from typing import Dict, List, Any, Tuple, Optional
from openai.types.chat import ChatCompletionMessageParam
from ..utils.logger import log
from ..utils.json_utils import clean_json_response, safe_json_loads

# 导入tau_prompt模块
from ..prompts import tau_prompt

# 获取三个域的system prompt
airline_system = tau_prompt.airline_system
retail_system = tau_prompt.retail_system
telecom_system = tau_prompt.telecom_system


class MultiTurnTaskGen:
    """多轮任务生成器 - 从三个固定类别(airline/retail/telecom)中采样"""

    _API_DIR = Path(__file__).resolve().parents[1] / "api"
    
    # 定义三个类别的配置
    DOMAIN_CONFIGS = {
        'airline': {
            'api_path': str(_API_DIR / 'airline_tool.json'),
            'system_prompt': airline_system
        },
        'retail': {
            'api_path': str(_API_DIR / 'retail_tool.json'),
            'system_prompt': retail_system
        },
        'telecom': {
            'api_path': str(_API_DIR / 'telecom_tool.json'),
            'system_prompt': telecom_system,
            'user_tools_path': str(_API_DIR / 'telecom_user_tool.json')
        }
    }
    
    def __init__(self, client, model, api_doc_path=None, domain: Optional[str] = None):
        """初始化多轮任务生成器
        
        Args:
            client: OpenAI客户端
            model: 模型名称
            api_doc_path: 兼容旧接口，但不再使用
            domain: 指定固定的领域（airline/retail/telecom），如果为None则随机选择
        """
        self.client = client
        self.model = model
        self.task_gen_prompt = local_prompt.MultiturnTaskPrompt
        self.user_info_prompt = local_prompt.UserInfoGenPrompt
        self.fixed_domain = domain  # 固定的领域
        
        # 验证domain参数
        if domain is not None and domain not in self.DOMAIN_CONFIGS:
            raise ValueError(f"Invalid domain: {domain}. Must be one of {list(self.DOMAIN_CONFIGS.keys())}")
        
        # 加载所有类别的API和对应的system prompt
        self.domain_data = {}
        for domain_name, config in self.DOMAIN_CONFIGS.items():
            # 加载API
            apis = self._load_apis(config['api_path'])
            # 获取对应的system prompt
            system_prompt = config['system_prompt']
            self.domain_data[domain_name] = {
                'apis': apis,
                'system_prompt': system_prompt
            }
        
        # 当前选中的类别
        self.current_domain = None
        self.current_system_prompt = None
    
    def _load_apis(self, api_path: str) -> List[Dict]:
        """加载API文件
        
        Args:
            api_path: API文件路径
            
        Returns:
            API列表
        """
        with open(api_path, 'r') as f:
            content = f.read().strip()
            # 文件是一个JSON数组
            tools = json.loads(content)
        
        # 提取function部分
        apis = []
        for tool in tools:
            if 'function' in tool:
                apis.append(tool['function'])
        
        return apis
    
    def sample_api(self) -> Tuple[List[Dict], str, str]:
        """从三个类别中选择一个（如果指定了fixed_domain则使用固定的），返回该类别的所有API和system prompt
        
        Returns:
            (apis, system_prompt, domain): API列表，对应的system prompt，选中的domain名称
        """
        # 如果指定了固定的domain，使用它；否则随机选择
        if self.fixed_domain:
            domain = self.fixed_domain
            log(f"[采样] 使用固定类别: {domain}")
        else:
            domain = random.choice(list(self.domain_data.keys()))
            log(f"[采样] 随机选中类别: {domain}")
        
        self.current_domain = domain
        
        domain_info = self.domain_data[domain]
        apis = domain_info['apis']
        system_prompt = domain_info['system_prompt']
        self.current_system_prompt = system_prompt
        
        log(f"[采样] 类别 {domain} 包含 {len(apis)} 个API")
        
        return apis, system_prompt, domain

    async def generate_user_info(self, sampled_apis, max_retries=2):
        """
        生成user信息：user有需求但不知道怎么解决，对某些信息不清楚，
        但可以通过已知信息调用工具找到
        """
        user_info_gen_messages: List[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': self.user_info_prompt},
            {'role': 'user', 'content': f"请根据以下可用的API工具，设计一个user的信息和需求场景。\n\n可用API工具：\n{json.dumps(sampled_apis, ensure_ascii=False)}"}
        ]
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    messages=user_info_gen_messages,
                    model=self.model,
                    temperature=0.8,
                    response_format={"type": "json_object"},
                    #extra_body={"chat_template_kwargs": {"thinking": True}}
                )
                
                content = response.choices[0].message.content
                user_info = safe_json_loads(content)
                
                # 验证user_info结构
                self._validate_user_info(user_info)
                
                return user_info
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt < max_retries - 1:
                    log(f"Generate user info attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    log(f"All {max_retries} attempts failed for user info. Last error: {e}")
                    raise
        
        return None
    
    def _validate_user_info(self, user_info):
        """验证user_info结构（支持更复杂的场景）"""
        required_fields = ['user_profile', 'known_info', 'unknown_info', 'user_need', 'difficulty_design']
        for field in required_fields:
            if field not in user_info:
                raise ValueError(f"Missing '{field}' field in user_info")
        
        # 验证unknown_info不为空（至少有1项）
        if not user_info['unknown_info'] or not user_info['unknown_info'].get('items'):
            raise ValueError("'unknown_info' must have at least 1 item")
        
        # 验证unknown_info items不超过3个（避免过于复杂）
        if len(user_info['unknown_info'].get('items', [])) > 5:
            raise ValueError("'unknown_info' should not have more than 3 items")
        
        # 验证complexity_level有效
        complexity = user_info.get('difficulty_design', {}).get('complexity_level', 'simple')
        if complexity not in ['simple', 'medium', 'complex']:
            user_info['difficulty_design']['complexity_level'] = 'simple'

    async def generate_task(self, max_retries=2):
        """
        生成多轮任务：先采样API，再生成user信息，最后生成任务
        
        Returns:
            (sampled_apis, user_info, task_data, domain, system_prompt)
        """
        # 第一步：采样API（从三个类别中选一个）
        sampled_apis, system_prompt, domain = self.sample_api()
        
        # 第二步：生成user信息
        user_info = await self.generate_user_info(sampled_apis, max_retries)
        if user_info is None:
            raise ValueError("Failed to generate user info")
        
        # 将domain信息添加到user_info中
        user_info['domain'] = domain
        
        # 第三步：基于user信息和API生成多轮任务
        task_gen_messages: List[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': self.task_gen_prompt},
            {'role': 'user', 'content': f"""请根据以下信息生成多轮任务。

Domain: {domain}

User信息：
{json.dumps(user_info, ensure_ascii=False)}

可用API工具：
{json.dumps(sampled_apis, ensure_ascii=False)}

Please generate a multi-turn task with the following requirements:
1. Difficulty level: Medium to Hard (中等到困难)
2. Task rounds: Design task rounds with dependencies between them
3. Dependencies: At least 1 round should depend on information from previous rounds
4. Use user's known_info as starting point
5. Help user get unknown_info through sequential tool calls
6. The task should be consistent with the {domain} domain context
7. Ensure the task requires logical reasoning and step-by-step problem solving
"""}
        ]
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    messages=task_gen_messages,
                    model=self.model,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                task_data = safe_json_loads(content)
                
                # 添加domain到task_data
                task_data['domain'] = domain
                
                # 验证任务结构
                self._validate_task_structure(task_data)
                
                return sampled_apis, user_info, task_data, domain, system_prompt
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt < max_retries - 1:
                    log(f"Generate task attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    log(f"All {max_retries} attempts failed for task. Last error: {e}")
                    raise
        
        return None, None, None, None, None
    
    def _validate_task_structure(self, task_data):
        """
        验证生成的任务结构是否符合规范（增强版，支持更复杂的任务）
        """
        # 检查必需字段
        if 'title' not in task_data:
            raise ValueError("Missing 'title' field in task data")
        
        if 'task_rounds' not in task_data:
            raise ValueError("Missing 'task_rounds' field in task data")
        
        # 验证任务轮次数量（1-5轮，允许更复杂的多轮任务）
        task_rounds = task_data['task_rounds']
        if not isinstance(task_rounds, list) or not (1 <= len(task_rounds) <= 5):
            raise ValueError(f"Task rounds must be a list with 1-5 items, got {len(task_rounds)} rounds")
        
        # 验证complexity_level（如果提供）
        if 'complexity_level' in task_data:
            if task_data['complexity_level'] not in ['simple', 'medium', 'complex']:
                task_data['complexity_level'] = 'medium'
        
        # 验证每个任务轮次的结构
        for i, round_data in enumerate(task_rounds):
            if 'round_index' not in round_data:
                raise ValueError(f"Missing 'round_index' in task round {i}")
            
            if 'user_goal' not in round_data:
                raise ValueError(f"Missing 'user_goal' in task round {i}")
            
            if 'tools_needed' not in round_data:
                raise ValueError(f"Missing 'tools_needed' in task round {i}")
            
            # 验证tools_needed数组
            tools_needed = round_data['tools_needed']
            if not isinstance(tools_needed, list):
                raise ValueError(f"'tools_needed' must be a list in task round {i}")
            
            # 允许tools_needed为空（有些轮次可能不需要调用工具，比如纯对话）
            # 但如果不为空，则验证每个工具的结构
            for j, tool in enumerate(tools_needed):
                if 'tool_name' not in tool:
                    raise ValueError(f"Missing 'tool_name' in tools_needed[{j}] of task round {i}")
        
        # 验证dependencies结构（如果提供）
        if 'dependencies' in task_data:
            dependencies = task_data['dependencies']
            if isinstance(dependencies, list):
                for dep in dependencies:
                    if 'from_round' not in dep or 'to_round' not in dep:
                        log(f"[验证警告] Dependency missing from_round or to_round, skipping")
                        continue
                    # 验证round索引在有效范围内
                    max_round = len(task_rounds)
                    if dep['from_round'] < 1 or dep['from_round'] > max_round:
                        raise ValueError(f"Invalid from_round {dep['from_round']} in dependency")
                    if dep['to_round'] < 1 or dep['to_round'] > max_round:
                        raise ValueError(f"Invalid to_round {dep['to_round']} in dependency")
    
    async def generate_evaluation_criteria(
        self,
        task_data: Dict,
        user_info: Dict,
        sampled_apis: List[Dict],
        max_retries: int = 2
    ) -> Optional[Dict]:
        """
        生成任务的评估标准
        
        Args:
            task_data: 任务数据
            user_info: 用户信息
            sampled_apis: 采样的API列表
            max_retries: 最大重试次数
        
        Returns:
            evaluation_criteria dict or None
        """
        criteria_messages: List[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': local_prompt.EvaluationCriteriaPrompt},
            {'role': 'user', 'content': f"""Generate evaluation criteria for the following task.

Task Data:
{json.dumps(task_data, ensure_ascii=False)}

User Information:
{json.dumps(user_info, ensure_ascii=False)}

Available APIs:
{json.dumps(sampled_apis, ensure_ascii=False)}

Generate comprehensive evaluation criteria that can be used to calculate a reward score.
"""}
        ]
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    messages=criteria_messages,
                    model=self.model,
                    temperature=0.5,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                criteria = safe_json_loads(content)
                
                # 验证必需字段
                if 'expected_actions' not in criteria:
                    criteria['expected_actions'] = []
                if 'success_conditions' not in criteria:
                    criteria['success_conditions'] = []
                if 'communicate_info' not in criteria:
                    criteria['communicate_info'] = []
                if 'reward_basis' not in criteria:
                    criteria['reward_basis'] = ['ACTION', 'COMMUNICATE']
                
                # 简要统计
                num_actions = len(criteria['expected_actions'])
                num_conditions = len(criteria['success_conditions'])
                log(f"[评估标准] 生成成功: {num_actions} actions, {num_conditions} conditions")
                
                # 详细打印每个action
                log(f"\n[期望动作详情]:")
                for i, action in enumerate(criteria['expected_actions'], 1):
                    # 类型检查：确保是字典
                    if isinstance(action, str):
                        log(f"  {i}. {action}")
                        continue
                    
                    tool_name = action.get('tool_name', 'Unknown')
                    params = action.get('required_params', [])
                    sequence = action.get('sequence_order', 'N/A')
                    round_assoc = action.get('round_association', 'N/A')
                    log(f"  {i}. {tool_name}")
                    log(f"     - 参数: {', '.join(params) if params else '无'}")
                    log(f"     - 顺序: {sequence} | 轮次: {round_assoc}")
                
                # 详细打印每个condition
                log(f"\n[成功条件详情]:")
                for i, cond in enumerate(criteria['success_conditions'], 1):
                    # 类型检查：确保是字典
                    if isinstance(cond, str):
                        log(f"  {i}. {cond}")
                        continue
                    
                    cond_type = cond.get('type', 'Unknown')
                    desc = cond.get('description', '')
                    target_round = cond.get('target_round', 'N/A')
                    impact = cond.get('failure_impact', 'N/A')
                    log(f"  {i}. [{cond_type}] {desc[:80]}{'...' if len(desc) > 80 else ''}")
                    log(f"     - 轮次: {target_round} | 影响: {impact}")
                
                return criteria
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt < max_retries - 1:
                    log(f"Generate evaluation criteria attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    log(f"All {max_retries} attempts failed for evaluation criteria. Last error: {e}")
                    # 返回默认的evaluation criteria
                    return {
                        'expected_actions': [],
                        'success_conditions': ["Task must be completed successfully"],
                        'communicate_info': [],
                        'reward_basis': ['ACTION', 'COMMUNICATE']
                    }
        
        return None
    
    async def generate(self):
        """
        生成完整的多轮任务数据，包括API、user信息、任务、user simulator prompt和evaluation criteria
        
        Returns:
            (sampled_apis, user_info, task_data, user_simulator_prompt, domain_system_prompt, evaluation_criteria)
            - sampled_apis: 采样的API列表
            - user_info: 用户信息
            - task_data: 任务数据
            - user_simulator_prompt: 用户模拟器的system prompt
            - domain_system_prompt: 该domain对应的agent system prompt
            - evaluation_criteria: 评估标准
        """
        # 生成任务
        sampled_apis, user_info, task_data, domain, domain_system_prompt = await self.generate_task()
        
        if sampled_apis is None:
            return None, None, None, None, None, None
        
        # 生成user simulator system prompt
        user_sim_messages: List[ChatCompletionMessageParam] = [
            {'role': 'system', 'content': local_prompt.UserSimulatorSystemPromptGen},
            {'role': 'user', 'content': f"""Please generate a system prompt for the user simulator based on the following information.

Domain: {domain}

User Information:
{json.dumps(user_info, ensure_ascii=False)}

Multi-turn Task Definition:
{json.dumps(task_data, ensure_ascii=False)}

Agent System Context (for reference only, do not include in user prompt):
{domain_system_prompt[:500]}...

Please generate a complete system prompt to guide the AI in playing this user role.
The user simulator should act as a realistic customer in the {domain} domain.
"""}
        ]
        
        response = await self.client.chat.completions.create(
            messages=user_sim_messages,
            model=self.model,
            temperature=0.7,
            extra_body = {'temperature':1.0}
        )
        
        user_simulator_prompt = response.choices[0].message.content
        
        # 生成evaluation criteria
        evaluation_criteria = await self.generate_evaluation_criteria(
            task_data, user_info, sampled_apis
        )
        
        return sampled_apis, user_info, task_data, user_simulator_prompt, domain_system_prompt, evaluation_criteria
