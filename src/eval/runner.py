"""评测运行器 - 支持并行评测"""
import os
import shutil
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import itertools
import multiprocessing

from .config import EvalConfig
from .metrics import compute_extended_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvalTask:
    """单个评测任务"""
    task_name: str
    domain: str
    run_id: int  # 第几次运行
    config: EvalConfig
    output_path: str
    timestamp: str
    task_source: str = "native"  # "native" 或 "synthetic"


@dataclass
class EvalResult:
    """评测结果"""
    task: EvalTask
    success: bool
    result_file: Optional[str] = None
    error: Optional[str] = None
    num_tasks: int = 0
    num_success: int = 0
    avg_reward: float = 0.0
    success_rate: float = 0.0
    # 新增4个扩展指标
    tool_call_accuracy: float = 0.0
    parameter_accuracy: float = 0.0
    conversation_efficiency: float = 0.0
    error_recovery_rate: float = 0.0


def run_single_eval(task: EvalTask) -> EvalResult:
    """运行单个评测任务（在子进程中执行）"""
    # 构建结果文件路径 - 根据task_source添加后缀
    source_suffix = f"_{task.task_source}" if task.task_source != "native" else ""
    source_tag = f"/{task.task_source}" if task.task_source != "native" else ""
    save_name = f"{task.task_name}_{task.timestamp}_{task.domain}{source_suffix}_run{task.run_id}"
    save_path = str(Path(task.output_path).resolve() / save_name)
    result_file = f"{save_path}.json"

    # 日志保存到 logs/ 子目录
    logs_dir = Path(task.output_path).resolve() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(logs_dir / f"{save_name}.log")

    print(f"[{task.domain}{source_tag}/run{task.run_id}] 开始评测... (日志: {log_file})")

    # 根据 task_source 选择运行方式
    if task.task_source == "synthetic":
        return _run_synthetic_eval(task, result_file, log_file, source_tag)
    else:
        return _run_native_eval(task, result_file, log_file, source_tag, save_path)


def _run_native_eval(task: EvalTask, result_file: str, log_file: str, source_tag: str, save_path: str) -> EvalResult:
    """运行原生任务评测（使用 subprocess 调用 tau2 run）"""
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', 'EMPTY')
        env['LITELLM_LOG'] = 'ERROR'
        env['PYTHONWARNINGS'] = 'ignore::UserWarning'

        # 构建tau2命令 - 使用绝对路径
        project_root = Path(__file__).parent.parent.parent.resolve()
        tau2_cmd = os.environ.get('TAU2_CMD') or str(project_root / '.venv' / 'bin' / 'tau2')
        if not Path(tau2_cmd).exists():
            tau2_cmd = shutil.which('tau2') or tau2_cmd
        tau2_bench_dir = os.environ.get('TAU2_BENCH_DIR') or str(project_root / 'tau2-bench')

        # 验证路径
        if not Path(tau2_cmd).exists():
            return EvalResult(
                task=task,
                success=False,
                error=f"tau2 command not found: {tau2_cmd}"
            )

        # 构建LLM参数
        agent_llm = f"openai/{task.config.agent.model_id}"
        user_llm = f"openai/{task.config.user.model_id}"

        agent_llm_args_dict = {
            "api_base": task.config.agent.base_url,
            "max_tokens": task.config.agent.max_tokens
        }
        # 添加 api_key (用于第三方 OpenAI-compatible API)
        if task.config.agent.api_key:
            agent_llm_args_dict["api_key"] = task.config.agent.api_key
        # 添加 extra_body (如 thinking 模式)
        if task.config.agent.extra_body:
            agent_llm_args_dict["extra_body"] = task.config.agent.extra_body
        agent_llm_args = json.dumps(agent_llm_args_dict)

        user_llm_args_dict = {
            "api_base": task.config.user.base_url,
            "max_tokens": task.config.user.max_tokens
        }
        # 添加 api_key (用于第三方API)
        if task.config.user.api_key:
            user_llm_args_dict["api_key"] = task.config.user.api_key
        if task.config.user.extra_body:
            user_llm_args_dict["extra_body"] = task.config.user.extra_body
        user_llm_args = json.dumps(user_llm_args_dict)

        # 构建命令
        cmd = [
            tau2_cmd, 'run',
            '--domain', task.domain,
            '--agent-llm', agent_llm,
            '--agent-llm-args', agent_llm_args,
            '--user-llm', user_llm,
            '--user-llm-args', user_llm_args,
            '--num-trials', str(task.config.num_trials),
            '--max-concurrency', str(task.config.max_concurrency),
            '--save-to', save_path
        ]

        if task.config.num_tasks:
            cmd.extend(['--num-tasks', str(task.config.num_tasks)])

        # 执行命令，将输出写入日志文件
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd,
                cwd=tau2_bench_dir,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=7200  # 2小时超时
            )

        if result.returncode != 0:
            # 进程失败，但检查是否有部分结果
            if os.path.exists(result_file):
                partial_result = _try_parse_result_file(task, result_file)
                if partial_result:
                    print(f"[{task.domain}{source_tag}/run{task.run_id}] 进程失败但有部分结果")
                    return partial_result

            # 读取日志文件获取错误信息
            error_msg = ""
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    error_msg = ''.join(lines[-20:])  # 最后20行
            print(f"[{task.domain}{source_tag}/run{task.run_id}] 评测失败，查看日志: {log_file}")
            return EvalResult(
                task=task,
                success=False,
                error=error_msg[:500] if error_msg else "Unknown error"
            )

        # 解析结果
        return _parse_result(task, result_file, source_tag)

    except subprocess.TimeoutExpired:
        partial_result = _try_parse_result_file(task, result_file)
        if partial_result:
            print(f"[{task.domain}{source_tag}/run{task.run_id}] 超时但有部分结果")
            return partial_result
        return EvalResult(task=task, success=False, error="Timeout after 2 hours")

    except Exception as e:
        partial_result = _try_parse_result_file(task, result_file)
        if partial_result:
            print(f"[{task.domain}{source_tag}/run{task.run_id}] 异常但有部分结果")
            return partial_result
        return EvalResult(task=task, success=False, error=str(e))


def _run_synthetic_eval(task: EvalTask, result_file: str, log_file: str, source_tag: str) -> EvalResult:
    """运行 synthetic 任务评测（使用 Python 直接调用 run_tasks）"""
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    try:
        # 设置环境变量
        os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', 'EMPTY')
        os.environ['LITELLM_LOG'] = 'ERROR'

        project_root = Path(__file__).parent.parent.parent.resolve()
        tau2_bench_dir = Path(os.environ.get('TAU2_BENCH_DIR', project_root / 'tau2-bench'))

        # 添加tau2-bench到path
        tau2_src = str(tau2_bench_dir / 'src')
        if tau2_src not in sys.path:
            sys.path.insert(0, tau2_src)

        from tau2.run import run_tasks
        from tau2.data_model.tasks import Task
        from tau2.evaluator.evaluator import EvaluationType

        # 加载synthetic任务
        synthetic_tasks_file = Path(task.config.synthetic_testset_dir) / f"synthetic_testset_{task.domain}.json"
        if not synthetic_tasks_file.exists():
            return EvalResult(
                task=task,
                success=False,
                error=f"Synthetic tasks file not found: {synthetic_tasks_file}"
            )

        with open(synthetic_tasks_file, 'r') as f:
            tasks_data = json.load(f)
        tasks_list = [Task.model_validate(t) for t in tasks_data]

        # 限制任务数
        if task.config.num_tasks:
            tasks_list = tasks_list[:task.config.num_tasks]

        print(f"[{task.domain}{source_tag}/run{task.run_id}] 加载了 {len(tasks_list)} 个synthetic任务")

        # 构建LLM参数
        agent_llm = f"openai/{task.config.agent.model_id}"
        user_llm = f"openai/{task.config.user.model_id}"
        llm_args_agent = {
            "api_base": task.config.agent.base_url,
            "max_tokens": task.config.agent.max_tokens
        }
        # 添加 api_key (用于第三方 OpenAI-compatible API)
        if task.config.agent.api_key:
            llm_args_agent["api_key"] = task.config.agent.api_key
        # 添加 extra_body (如 thinking 模式)
        if task.config.agent.extra_body:
            llm_args_agent["extra_body"] = task.config.agent.extra_body

        llm_args_user = {
            "api_base": task.config.user.base_url,
            "max_tokens": task.config.user.max_tokens
        }
        # 添加 api_key (用于第三方API)
        if task.config.user.api_key:
            llm_args_user["api_key"] = task.config.user.api_key
        if task.config.user.extra_body:
            llm_args_user["extra_body"] = task.config.user.extra_body

        # 运行评测，将输出重定向到日志文件
        with open(log_file, 'w') as log_f:
            with redirect_stdout(log_f), redirect_stderr(log_f):
                run_tasks(
                    domain=task.domain,
                    tasks=tasks_list,
                    agent="llm_agent",
                    user="user_simulator",
                    llm_agent=agent_llm,
                    llm_args_agent=llm_args_agent,
                    llm_user=user_llm,
                    llm_args_user=llm_args_user,
                    num_trials=task.config.num_trials,
                    max_steps=200,
                    max_errors=10,
                    save_to=result_file,
                    console_display=False,
                    evaluation_type=EvaluationType.ALL,
                    max_concurrency=task.config.max_concurrency,
                    seed=300 + task.run_id,
                    log_level="ERROR",
                )

        # 解析结果
        return _parse_result(task, result_file, source_tag)

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{task.domain}{source_tag}/run{task.run_id}] 评测失败: {e}")
        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f"\n\nError: {error_msg}")

        # 尝试恢复部分结果
        partial_result = _try_parse_result_file(task, result_file)
        if partial_result:
            print(f"[{task.domain}{source_tag}/run{task.run_id}] 评测异常但有部分结果")
            return partial_result

        return EvalResult(task=task, success=False, error=error_msg[:500])


def _parse_result(task: EvalTask, result_file: str, source_tag: str) -> EvalResult:
    """解析评测结果文件"""
    if not os.path.exists(result_file):
        return EvalResult(task=task, success=False, error="Result file not found")

    try:
        with open(result_file) as f:
            data = json.load(f)

        sims = data.get("simulations", [])
        rewards = [_get_reward(s) for s in sims]
        num_tasks = len(rewards)
        num_success = sum(1 for r in rewards if r > 0)
        avg_reward = sum(rewards) / num_tasks if num_tasks > 0 else 0.0
        success_rate = num_success / num_tasks * 100 if num_tasks > 0 else 0.0

        # 计算扩展指标
        try:
            extended = compute_extended_metrics(result_file, data.get("tasks", []))
            tool_call_accuracy = extended.tool_call_accuracy
            parameter_accuracy = extended.parameter_accuracy
            conversation_efficiency = extended.conversation_efficiency
            error_recovery_rate = extended.error_recovery_rate
        except Exception as e:
            print(f"[{task.domain}{source_tag}/run{task.run_id}] 计算扩展指标失败: {e}")
            tool_call_accuracy = 0.0
            parameter_accuracy = 0.0
            conversation_efficiency = 0.0
            error_recovery_rate = 0.0

        print(f"[{task.domain}{source_tag}/run{task.run_id}] 完成: {num_success}/{num_tasks} ({success_rate:.1f}%)")
        print(f"  工具调用准确率: {tool_call_accuracy:.1f}%, 参数准确率: {parameter_accuracy:.1f}%")
        print(f"  对话效率: {conversation_efficiency:.1f}%, 错误恢复率: {error_recovery_rate:.1f}%")

        return EvalResult(
            task=task,
            success=True,
            result_file=result_file,
            num_tasks=num_tasks,
            num_success=num_success,
            avg_reward=avg_reward,
            success_rate=success_rate,
            tool_call_accuracy=tool_call_accuracy,
            parameter_accuracy=parameter_accuracy,
            conversation_efficiency=conversation_efficiency,
            error_recovery_rate=error_recovery_rate,
        )
    except Exception as e:
        return EvalResult(task=task, success=False, error=f"Failed to parse result: {e}")


def _try_parse_result_file(task: 'EvalTask', result_file: str) -> Optional[EvalResult]:
    """尝试解析结果文件，返回EvalResult或None"""
    if not os.path.exists(result_file):
        return None

    try:
        with open(result_file) as f:
            data = json.load(f)
        sims = data.get("simulations", [])
        if not sims:
            return None

        rewards = [_get_reward(s) for s in sims]
        num_tasks = len(rewards)
        num_success = sum(1 for r in rewards if r > 0)
        avg_reward = sum(rewards) / num_tasks if num_tasks > 0 else 0.0
        success_rate = num_success / num_tasks * 100 if num_tasks > 0 else 0.0

        # 计算扩展指标
        try:
            extended = compute_extended_metrics(result_file, data.get("tasks", []))
            tool_call_accuracy = extended.tool_call_accuracy
            parameter_accuracy = extended.parameter_accuracy
            conversation_efficiency = extended.conversation_efficiency
            error_recovery_rate = extended.error_recovery_rate
        except Exception as e:
            print(f"[{task.domain}/run{task.run_id}] 从文件恢复时计算扩展指标失败: {e}")
            tool_call_accuracy = 0.0
            parameter_accuracy = 0.0
            conversation_efficiency = 0.0
            error_recovery_rate = 0.0

        print(f"[{task.domain}/run{task.run_id}] 从文件恢复结果: {num_success}/{num_tasks} ({success_rate:.1f}%)")

        return EvalResult(
            task=task,
            success=True,
            result_file=result_file,
            num_tasks=num_tasks,
            num_success=num_success,
            avg_reward=avg_reward,
            success_rate=success_rate,
            tool_call_accuracy=tool_call_accuracy,
            parameter_accuracy=parameter_accuracy,
            conversation_efficiency=conversation_efficiency,
            error_recovery_rate=error_recovery_rate,
        )
    except Exception:
        return None


def _get_reward(sim: dict) -> float:
    """从simulation中提取reward"""
    reward = sim.get("reward")
    if reward is not None:
        return reward
    reward_info = sim.get("reward_info", {})
    if reward_info:
        return reward_info.get("reward", 0) or 0
    return 0


class EvalRunner:
    """评测运行器"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / config.task_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """运行所有评测任务（并行）"""
        # 生成所有任务
        tasks = self._generate_tasks()
        total_tasks = len(tasks)

        # 统计各任务源的任务数
        native_tasks = [t for t in tasks if t.task_source == "native"]
        synthetic_tasks = [t for t in tasks if t.task_source == "synthetic"]

        print("=" * 60)
        print(f"评测任务: {self.config.task_name}")
        print("=" * 60)
        print(f"领域: {self.config.domains}")
        if native_tasks and synthetic_tasks:
            print(f"任务源: native ({len(native_tasks)}个) + synthetic ({len(synthetic_tasks)}个)")
            print(f"  - Native: {self.config.native_num_runs}次/领域")
            print(f"  - Synthetic: {self.config.synthetic_num_runs or self.config.num_runs}次/领域")
        elif native_tasks:
            print(f"任务源: native (tau2-bench原生)")
            print(f"每个领域运行次数: {self.config.num_runs}")
        else:
            print(f"任务源: synthetic")
            print(f"每个领域运行次数: {self.config.num_runs}")
        print(f"总任务数: {total_tasks}")
        print(f"最大并行任务数: {self.config.eval_batch_size}")
        print("=" * 60)

        # 使用固定数量的worker持续执行任务（始终保持eval_batch_size个任务在运行）
        results: List[EvalResult] = []
        max_workers = min(self.config.eval_batch_size, len(tasks))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            task_iter = iter(tasks)
            completed_count = 0

            # 初始提交 max_workers 个任务
            for task in itertools.islice(task_iter, max_workers):
                future = executor.submit(run_single_eval, task)
                futures[future] = task

            # 持续处理：每完成一个就补充一个新任务
            while futures:
                # 等待任意一个任务完成
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    task = futures.pop(future)
                    completed_count += 1
                    try:
                        result = future.result()
                        results.append(result)
                        # 实时提取并保存轨迹（如果启用）
                        if self.config.save_trajectories:
                            self._extract_single_trajectory(result)
                        print(f"  [{completed_count}/{total_tasks}] 完成: {task.domain}/{task.task_source}/run{task.run_id}")
                    except Exception as e:
                        results.append(EvalResult(
                            task=task,
                            success=False,
                            error=str(e)
                        ))
                        print(f"  [{completed_count}/{total_tasks}] 失败: {task.domain}/{task.task_source}/run{task.run_id} - {e}")

                    # 补充新任务
                    try:
                        next_task = next(task_iter)
                        new_future = executor.submit(run_single_eval, next_task)
                        futures[new_future] = next_task
                    except StopIteration:
                        pass  # 没有更多任务了

        # 汇总结果
        summary = self._aggregate_results(results)

        # 保存汇总 - scores字段使用紧凑格式
        summary_file = self.output_dir / f"summary_{self.timestamp}.json"
        self._save_summary(summary, summary_file)

        # 打印汇总
        self._print_summary(summary)

        return summary
    
    def _generate_tasks(self) -> List[EvalTask]:
        """生成所有评测任务（synthetic优先，然后native）"""
        tasks = []

        # 判断任务源模式
        has_synthetic = self.config.synthetic_testset_dir is not None
        has_native = not has_synthetic or self.config.also_test_native

        # 确定各任务源的运行次数
        native_runs = self.config.native_num_runs if has_native else 0
        synthetic_runs = (self.config.synthetic_num_runs or self.config.num_runs) if has_synthetic else 0

        # 先生成synthetic任务（优先执行）
        if has_synthetic and synthetic_runs > 0:
            for domain in self.config.domains:
                for run_id in range(1, synthetic_runs + 1):
                    task = EvalTask(
                        task_name=self.config.task_name,
                        domain=domain,
                        run_id=run_id,
                        config=self.config,
                        output_path=str(self.output_dir),
                        timestamp=self.timestamp,
                        task_source="synthetic"
                    )
                    tasks.append(task)

        # 再生成原生任务
        if has_native and native_runs > 0:
            for domain in self.config.domains:
                for run_id in range(1, native_runs + 1):
                    task = EvalTask(
                        task_name=self.config.task_name,
                        domain=domain,
                        run_id=run_id,
                        config=self.config,
                        output_path=str(self.output_dir),
                        timestamp=self.timestamp,
                        task_source="native"
                    )
                    tasks.append(task)

        return tasks
    
    def _save_summary(self, summary: Dict[str, Any], summary_file: Path):
        """保存汇总结果，scores字段使用紧凑格式"""
        # 检查是否是多源格式
        if "sources" in summary:
            # 多源格式，对每个 source 的 scores 使用紧凑格式
            self._save_multi_source_summary(summary, summary_file)
            return

        # 单源格式，保持原有的紧凑scores格式
        self._save_single_source_summary(summary, summary_file)

    def _save_multi_source_summary(self, summary: Dict[str, Any], summary_file: Path):
        """保存多源格式汇总，scores 使用紧凑格式"""
        # 对每个 source 的 scores 进行紧凑化处理
        for source_name, source_data in summary.get("sources", {}).items():
            if "scores" in source_data:
                source_data["scores"] = self._compact_scores(source_data["scores"])

        # 自定义 JSON encoder 来保持 scores 数组在一行
        class CompactScoresEncoder(json.JSONEncoder):
            def encode(self, obj):
                # 先用默认方式编码
                result = super().encode(obj)
                return result

        # 直接保存，然后后处理
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 后处理：将 scores 中的数组变成一行
        with open(summary_file, 'r') as f:
            content = f.read()

        # 使用正则表达式将多行数组变成一行
        import re
        # 匹配 scores 中的数组
        def compact_array(match):
            key = match.group(1)
            array_content = match.group(2)
            # 提取数字（包括负数和小数）
            numbers = re.findall(r'-?[\d.]+', array_content)
            return f'"{key}": [{", ".join(numbers)}]'

        # 匹配 scores 块中的数组（更宽泛的匹配）
        content = re.sub(
            r'"((?:retail|airline|telecom|avg|overall)_[a-z_]+)": \[\s*([^\]]+)\]',
            compact_array,
            content
        )

        with open(summary_file, 'w') as f:
            f.write(content)

    def _compact_scores(self, scores: Dict) -> Dict:
        """返回原始 scores，实际紧凑化在保存时处理"""
        return scores

    def _save_single_source_summary(self, summary: Dict[str, Any], summary_file: Path):
        """保存单源格式汇总"""
        # 分离scores字段
        scores = summary.pop("scores", {})

        # 先保存主体部分
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 读取并追加scores字段（紧凑格式）
        with open(summary_file, 'r') as f:
            content = f.read()

        # 移除最后的 }
        content = content.rstrip().rstrip('}').rstrip()

        # 构建scores字符串（每个字段一行，数组紧凑格式）
        scores_lines = []

        # 定义字段输出顺序
        ordered_keys = [
            # 主要分数
            "retail_score", "airline_score", "telecom_score",
            # 扩展指标
            "retail_tool_call_acc", "airline_tool_call_acc", "telecom_tool_call_acc",
            "retail_param_acc", "airline_param_acc", "telecom_param_acc",
            "retail_conv_eff", "airline_conv_eff", "telecom_conv_eff",
            "retail_err_recovery", "airline_err_recovery", "telecom_err_recovery",
            # 汇总
            "avg_score", "avg_per_run",
            "avg_tool_call_acc", "avg_param_acc", "avg_conv_eff", "avg_err_recovery",
            "overall_avg",
            "overall_tool_call_acc", "overall_param_acc", "overall_conv_eff", "overall_err_recovery"
        ]

        for key in ordered_keys:
            if key in scores:
                value = scores[key]
                if isinstance(value, list):
                    arr_str = "[" + ", ".join(str(v) for v in value) + "]"
                    scores_lines.append(f'"{key}": {arr_str}')
                else:
                    scores_lines.append(f'"{key}": {value}')

        # 每个字段换行
        scores_str = "{\n    " + ",\n    ".join(scores_lines) + "\n  }"

        # 追加scores字段
        content += f',\n  "scores": {scores_str}\n}}'

        with open(summary_file, 'w') as f:
            f.write(content)

        # 恢复scores到summary
        summary["scores"] = scores
    
    def _aggregate_results(self, results: List[EvalResult]) -> Dict[str, Any]:
        """汇总结果 - 支持按task_source分别汇总"""
        # 判断是否有多个task_source
        task_sources = set(r.task.task_source for r in results)
        has_multiple_sources = len(task_sources) > 1

        if has_multiple_sources:
            # 分别汇总native和synthetic
            summary = {
                "task_name": self.config.task_name,
                "timestamp": self.timestamp,
                "config": self.config.to_dict(),
                "sources": {}
            }
            for source in ["native", "synthetic"]:
                source_results = [r for r in results if r.task.task_source == source]
                if source_results:
                    source_summary = self._aggregate_single_source(source_results, source)
                    summary["sources"][source] = source_summary
            return summary
        else:
            # 单一task_source，保持原有格式
            source = list(task_sources)[0] if task_sources else "native"
            return self._aggregate_single_source(results, source)

    def _aggregate_single_source(self, results: List[EvalResult], source: str) -> Dict[str, Any]:
        """汇总单个task_source的结果"""
        summary = {
            "task_name": self.config.task_name,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "task_source": source,
            "domains": {},
            "overall": {
                "total_tasks": 0,
                "successful_tasks": 0,
                "avg_reward": 0.0,
                "success_rate": 0.0
            },
            "scores": {}  # 新增：简洁的分数汇总
        }

        # 按领域分组
        domain_results: Dict[str, List[EvalResult]] = {}
        for result in results:
            domain = result.task.domain
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)

        total_tasks = 0
        total_reward = 0.0
        total_success = 0

        for domain, domain_res in domain_results.items():
            runs = []
            domain_total_tasks = 0
            domain_total_reward = 0.0
            domain_total_success = 0

            # 按run_id排序
            domain_res_sorted = sorted(domain_res, key=lambda x: x.task.run_id)

            for res in domain_res_sorted:
                run_info = {
                    "run_id": res.task.run_id,
                    "success": res.success,
                    "result_file": res.result_file,
                    "num_tasks": res.num_tasks,
                    "num_success": res.num_success,
                    "avg_reward": round(res.avg_reward, 4),
                    "success_rate": round(res.success_rate, 2),
                    # 扩展指标
                    "tool_call_accuracy": round(res.tool_call_accuracy, 2),
                    "parameter_accuracy": round(res.parameter_accuracy, 2),
                    "conversation_efficiency": round(res.conversation_efficiency, 2),
                    "error_recovery_rate": round(res.error_recovery_rate, 2)
                }
                if res.error:
                    run_info["error"] = res.error
                runs.append(run_info)

                if res.success:
                    domain_total_tasks += res.num_tasks
                    domain_total_reward += res.avg_reward * res.num_tasks
                    domain_total_success += res.num_success

            # 计算领域平均值
            num_successful_runs = sum(1 for r in domain_res if r.success)
            if num_successful_runs > 0:
                avg_reward = domain_total_reward / domain_total_tasks if domain_total_tasks > 0 else 0
                avg_success_rate = domain_total_success / domain_total_tasks * 100 if domain_total_tasks > 0 else 0
            else:
                avg_reward = 0
                avg_success_rate = 0

            summary["domains"][domain] = {
                "runs": runs,
                "num_runs": len(runs),
                "successful_runs": num_successful_runs,
                "total_tasks": domain_total_tasks,
                "total_success": domain_total_success,
                "avg_reward": round(avg_reward, 4),
                "avg_success_rate": round(avg_success_rate, 2)
            }

        # 计算总体指标 - 三个榜的分数相加除以3（而不是所有题目的平均）
        valid_domains = [d for d, stats in summary["domains"].items() if stats["successful_runs"] > 0]
        num_valid_domains = len(valid_domains)

        if num_valid_domains > 0:
            # 计算各领域的平均reward和success_rate，然后取平均
            domain_avg_rewards = [summary["domains"][d]["avg_reward"] for d in valid_domains]
            domain_avg_success_rates = [summary["domains"][d]["avg_success_rate"] for d in valid_domains]

            # 总体指标 = 各领域分数之和 / 领域数
            overall_avg_reward = sum(domain_avg_rewards) / num_valid_domains
            overall_avg_success_rate = sum(domain_avg_success_rates) / num_valid_domains

            # 统计总任务数（仅供参考）
            total_tasks = sum(summary["domains"][d]["total_tasks"] for d in valid_domains)
            total_success = sum(summary["domains"][d]["total_success"] for d in valid_domains)

            summary["overall"]["num_domains"] = num_valid_domains
            summary["overall"]["total_tasks"] = total_tasks
            summary["overall"]["successful_tasks"] = total_success
            summary["overall"]["avg_reward"] = round(overall_avg_reward, 4)
            summary["overall"]["success_rate"] = round(overall_avg_success_rate, 2)

        # 生成简洁的分数汇总
        self._generate_scores_summary_for_source(summary, domain_results, source)

        return summary
    
    def _generate_scores_summary_for_source(self, summary: Dict[str, Any], domain_results: Dict[str, List[EvalResult]], source: str):
        """生成简洁的分数汇总格式（针对单个task_source）"""
        scores = {}
        # 根据source确定num_runs
        if source == "native":
            num_runs = self.config.native_num_runs if self.config.also_test_native else self.config.num_runs
        else:  # synthetic
            num_runs = self.config.synthetic_num_runs or self.config.num_runs

        # 固定领域顺序
        domain_order = ["retail", "airline", "telecom"]

        # 每个领域的分数列表（按run_id排序）
        for domain in domain_order:
            if domain in domain_results:
                sorted_results = sorted(domain_results[domain], key=lambda x: x.task.run_id)
                # 使用success_rate作为分数
                scores[f"{domain}_score"] = [
                    round(r.success_rate, 2) if r.success else 0.0
                    for r in sorted_results
                ]
                # 新增4个扩展指标
                scores[f"{domain}_tool_call_acc"] = [
                    round(r.tool_call_accuracy, 2) if r.success else 0.0
                    for r in sorted_results
                ]
                scores[f"{domain}_param_acc"] = [
                    round(r.parameter_accuracy, 2) if r.success else 0.0
                    for r in sorted_results
                ]
                scores[f"{domain}_conv_eff"] = [
                    round(r.conversation_efficiency, 2) if r.success else 0.0
                    for r in sorted_results
                ]
                scores[f"{domain}_err_recovery"] = [
                    round(r.error_recovery_rate, 2) if r.success else 0.0
                    for r in sorted_results
                ]
            else:
                scores[f"{domain}_score"] = [0.0] * num_runs
                scores[f"{domain}_tool_call_acc"] = [0.0] * num_runs
                scores[f"{domain}_param_acc"] = [0.0] * num_runs
                scores[f"{domain}_conv_eff"] = [0.0] * num_runs
                scores[f"{domain}_err_recovery"] = [0.0] * num_runs

        # 计算每次运行的三领域平均分（分数平均，不是题数平均）
        # 无论分数是多少，都要计入平均（三个领域分数相加除以3）
        avg_per_run = []
        for run_idx in range(num_runs):
            run_scores = []
            for domain in domain_order:
                domain_scores = scores.get(f"{domain}_score", [])
                if run_idx < len(domain_scores):
                    run_scores.append(domain_scores[run_idx])
            # 始终用3个领域计算平均，即使某个领域是0分
            if len(run_scores) == 3:
                avg_per_run.append(round(sum(run_scores) / 3, 2))
            elif run_scores:
                avg_per_run.append(round(sum(run_scores) / len(run_scores), 2))
            else:
                avg_per_run.append(0.0)

        # 计算各领域的平均分（avg_score）
        avg_score = []
        for domain in domain_order:
            domain_scores = scores.get(f"{domain}_score", [])
            if domain_scores:
                avg_score.append(round(sum(domain_scores) / len(domain_scores), 2))
            else:
                avg_score.append(0.0)

        # 计算总平均分（所有运行的avg_per_run的平均）
        if avg_per_run:
            overall_avg = round(sum(avg_per_run) / len(avg_per_run), 2)
        else:
            overall_avg = 0.0

        # 计算4个扩展指标的总平均
        for metric_suffix in ["tool_call_acc", "param_acc", "conv_eff", "err_recovery"]:
            metric_avg_per_run = []
            for run_idx in range(num_runs):
                run_metrics = []
                for domain in domain_order:
                    domain_metrics = scores.get(f"{domain}_{metric_suffix}", [])
                    if run_idx < len(domain_metrics):
                        run_metrics.append(domain_metrics[run_idx])
                if len(run_metrics) == 3:
                    metric_avg_per_run.append(round(sum(run_metrics) / 3, 2))
                elif run_metrics:
                    metric_avg_per_run.append(round(sum(run_metrics) / len(run_metrics), 2))
                else:
                    metric_avg_per_run.append(0.0)
            scores[f"avg_{metric_suffix}"] = metric_avg_per_run
            # 计算总平均
            if metric_avg_per_run:
                scores[f"overall_{metric_suffix}"] = round(sum(metric_avg_per_run) / len(metric_avg_per_run), 2)
            else:
                scores[f"overall_{metric_suffix}"] = 0.0

        scores["avg_score"] = avg_score
        scores["avg_per_run"] = avg_per_run
        scores["overall_avg"] = overall_avg

        summary["scores"] = scores
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印汇总结果"""
        # 检查是否是多源格式
        if "sources" in summary:
            self._print_multi_source_summary(summary)
            return

        # 单源格式
        self._print_single_source_summary(summary)

    def _print_multi_source_summary(self, summary: Dict[str, Any]):
        """打印多任务源汇总结果"""
        print("\n" + "=" * 60)
        print(f"评测汇总: {summary['task_name']}")
        print("=" * 60)

        for source, source_summary in summary["sources"].items():
            print(f"\n{'='*60}")
            print(f"任务源: {source.upper()}")
            print("=" * 60)
            self._print_single_source_summary(source_summary, skip_header=True)

    def _print_single_source_summary(self, summary: Dict[str, Any], skip_header: bool = False):
        """打印单任务源汇总结果"""
        if not skip_header:
            print("\n" + "=" * 60)
            print(f"评测汇总: {summary['task_name']}")
            print("=" * 60)

        for domain, stats in summary["domains"].items():
            print(f"\n{domain}:")
            for run in stats["runs"]:
                status = "✓" if run["success"] else "✗"
                if run["success"]:
                    print(f"  Run {run['run_id']}: {status} {run['num_success']}/{run['num_tasks']} ({run['success_rate']:.1f}%) avg={run['avg_reward']:.4f}")
                else:
                    print(f"  Run {run['run_id']}: {status} Error: {run.get('error', 'Unknown')[:50]}")
            print(f"  平均: {stats['avg_success_rate']:.1f}% (avg_reward={stats['avg_reward']:.4f})")

        print("\n" + "-" * 60)
        overall = summary["overall"]
        num_domains = overall.get("num_domains", len(summary["domains"]))
        print(f"Overall ({num_domains}个领域平均): {overall['success_rate']:.1f}% (avg_reward={overall['avg_reward']:.4f})")

        # 打印简洁的分数汇总格式
        scores = summary.get("scores", {})
        if scores:
            print("\n" + "-" * 60)
            print("分数汇总:")
            print("-" * 60)

            # 按固定顺序打印
            for domain in ["retail", "airline", "telecom"]:
                key = f"{domain}_score"
                if key in scores:
                    score_str = " ".join(f"{s:.2f}" for s in scores[key])
                    print(f"{key}: {score_str}")

            # 打印avg
            avg_per_run = scores.get("avg_per_run", [])
            overall_avg = scores.get("overall_avg", 0.0)
            if avg_per_run:
                first_run_avg = avg_per_run[0] if avg_per_run else 0.0
                print(f"avg: {first_run_avg:.2f} {overall_avg:.2f}")

        print("=" * 60)

    def _extract_single_trajectory(self, result: EvalResult):
        """实时提取单个评测结果的轨迹，按task组织保存（即使有错误也尝试保存）"""
        # 即使失败也尝试保存（可能有部分结果）
        if not result.result_file:
            print(f"  [轨迹] 跳过无结果文件任务")
            return

        # 检查结果文件是否存在
        if not Path(result.result_file).exists():
            print(f"  [轨迹] 结果文件不存在: {result.result_file}")
            return

        data_dir = Path(os.environ.get("CRAFT_TRAJECTORY_DIR", "data/trajectories"))
        print(f"  [轨迹] 开始提取: {result.result_file}")

        try:
            with open(result.result_file) as f:
                data = json.load(f)

            # 获取模型信息
            info = data.get("info", {})
            agent_info = info.get("agent_info", {})
            model_name = self.config.task_name

            # 提取轨迹
            tasks = data.get("tasks", [])
            simulations = data.get("simulations", [])
            task_map = {str(t.get("id")): t for t in tasks}

            domain = result.task.domain
            task_source = result.task.task_source
            # 目录结构: model/domain/task_source
            task_dir = data_dir / model_name / domain / task_source
            task_dir.mkdir(parents=True, exist_ok=True)

            saved_count = 0
            for sim in simulations:
                task_id = str(sim.get("task_id", ""))
                task_info = task_map.get(task_id, {})

                # 提取完整对话
                messages = sim.get("messages", [])
                conversation = []
                for msg in messages:
                    conv_msg = {
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                    }
                    if msg.get("tool_calls"):
                        conv_msg["tool_calls"] = msg.get("tool_calls")
                    if msg.get("tool_call_id"):
                        conv_msg["tool_call_id"] = msg.get("tool_call_id")
                    raw_data = msg.get("raw_data", {})
                    if raw_data and "reasoning_content" in raw_data:
                        conv_msg["reasoning_content"] = raw_data["reasoning_content"]
                    conversation.append(conv_msg)

                # 构建模型回复记录
                model_response = {
                    "model": model_name,
                    "model_full": agent_info.get("llm", ""),
                    "run_id": result.task.run_id,
                    "success": result.success,
                    "error": result.error,
                    "reward": sim.get("reward") or sim.get("reward_info", {}).get("reward", 0),
                    "conversation": conversation,
                }

                # 保存task文件 (每个run独立存储，文件名包含run_id)
                run_id = result.task.run_id
                task_file = task_dir / f"task_{task_id}_run{run_id}.json"
                task_entry = {
                    "task_info": {
                        "task_id": task_id,
                        "description": task_info.get("description", {}),
                        "user_scenario": task_info.get("user_scenario", {}),
                    },
                    "model_response": model_response,
                }

                with open(task_file, 'w', encoding='utf-8') as f:
                    json.dump(task_entry, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                saved_count += 1

            print(f"  [轨迹] {model_name}/{domain}/{task_source}: {saved_count} tasks saved")

        except Exception as e:
            import traceback
            print(f"  [轨迹] {result.result_file}: 提取失败 - {e}")
            traceback.print_exc()
