"""评测配置类"""
import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


def _expand_env(value):
    """Expand ${VAR} placeholders in YAML values."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
            lambda m: os.environ.get(m.group(1), m.group(2) or ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _redact_secret(value):
    """Keep generated summaries from persisting API credentials."""
    if not value or value in {"EMPTY", "your-api-key"}:
        return value
    return "<redacted>"


@dataclass
class ModelConfig:
    """模型配置"""
    base_url: str
    model_id: str
    max_tokens: int = 4096
    api_key: Optional[str] = None  # API密钥（可选，用于第三方 OpenAI-compatible API）
    extra_body: Optional[Dict[str, Any]] = None  # 额外参数，如 thinking 模式


@dataclass
class EvalConfig:
    """评测配置"""
    # 任务名称
    task_name: str

    # Agent模型配置
    agent: ModelConfig

    # User模型配置
    user: ModelConfig

    # 评测参数
    domains: List[str] = field(default_factory=lambda: ["airline", "retail", "telecom"])
    num_trials: int = 1
    num_runs: int = 1  # 每个榜测几次取平均
    max_concurrency: int = 64  # 每个eval task内部的并发数
    eval_batch_size: int = 15  # 同时运行的eval tasks数量（如15个tasks分批运行）
    num_tasks: Optional[int] = None  # None表示所有任务

    # 输出配置
    output_dir: str = "eval_results"

    # Synthetic testset配置
    synthetic_testset_dir: Optional[str] = None  # synthetic testset目录
    compute_extended_metrics: bool = True  # 是否计算扩展指标

    # 同时测评多种任务源
    also_test_native: bool = False  # 同时测试tau2-bench原生任务
    native_num_runs: int = 4  # 原生任务运行次数
    synthetic_num_runs: Optional[int] = None  # synthetic任务运行次数（None则使用num_runs）

    # 轨迹保存
    save_trajectories: bool = True  # 是否保存轨迹文件
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvalConfig":
        """从YAML文件加载配置"""
        with open(yaml_path, 'r') as f:
            data = _expand_env(yaml.safe_load(f))
        
        agent_config = ModelConfig(
            base_url=data['agent']['base_url'],
            model_id=data['agent']['model_id'],
            max_tokens=data['agent'].get('max_tokens', 4096),
            api_key=data['agent'].get('api_key'),
            extra_body=data['agent'].get('extra_body')
        )

        user_config = ModelConfig(
            base_url=data['user']['base_url'],
            model_id=data['user']['model_id'],
            max_tokens=data['user'].get('max_tokens', 4096),
            api_key=data['user'].get('api_key'),
            extra_body=data['user'].get('extra_body')
        )
        
        return cls(
            task_name=data['task_name'],
            agent=agent_config,
            user=user_config,
            domains=data.get('domains', ["airline", "retail", "telecom"]),
            num_trials=data.get('num_trials', 1),
            num_runs=data.get('num_runs', 1),
            max_concurrency=data.get('max_concurrency', 64),
            eval_batch_size=data.get('eval_batch_size', 15),
            num_tasks=data.get('num_tasks'),
            output_dir=data.get('output_dir', 'eval_results'),
            synthetic_testset_dir=data.get('synthetic_testset_dir'),
            compute_extended_metrics=data.get('compute_extended_metrics', True),
            also_test_native=data.get('also_test_native', False),
            native_num_runs=data.get('native_num_runs', 4),
            synthetic_num_runs=data.get('synthetic_num_runs'),
            save_trajectories=data.get('save_trajectories', True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_name': self.task_name,
            'agent': {
                'base_url': self.agent.base_url,
                'model_id': self.agent.model_id,
                'max_tokens': self.agent.max_tokens,
                'api_key': _redact_secret(self.agent.api_key),
                'extra_body': self.agent.extra_body
            },
            'user': {
                'base_url': self.user.base_url,
                'model_id': self.user.model_id,
                'max_tokens': self.user.max_tokens,
                'api_key': _redact_secret(self.user.api_key),
                'extra_body': self.user.extra_body
            },
            'domains': self.domains,
            'num_trials': self.num_trials,
            'num_runs': self.num_runs,
            'max_concurrency': self.max_concurrency,
            'eval_batch_size': self.eval_batch_size,
            'num_tasks': self.num_tasks,
            'output_dir': self.output_dir,
            'synthetic_testset_dir': self.synthetic_testset_dir,
            'compute_extended_metrics': self.compute_extended_metrics,
            'also_test_native': self.also_test_native,
            'native_num_runs': self.native_num_runs,
            'synthetic_num_runs': self.synthetic_num_runs,
            'save_trajectories': self.save_trajectories,
        }
