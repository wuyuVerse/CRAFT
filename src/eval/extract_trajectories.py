#!/usr/bin/env python3
"""
提取评测结果中的模型回复轨迹，按任务ID组织保存

用法:
    python -m src.eval.extract_trajectories eval_results/provider_a eval_results/provider_b

输出结构:
    data/trajectories/
    ├── by_task/                    # 按任务组织
    │   ├── airline/
    │   │   ├── task_0.json         # 包含所有模型对该任务的回复
    │   │   ├── task_1.json
    │   │   └── ...
    │   ├── retail/
    │   └── telecom/
    └── by_model/                   # 按模型组织
        ├── deepseek_v3/
        │   ├── airline.json
        │   ├── retail.json
        │   └── telecom.json
        └── ...
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def extract_agent_responses(simulation: Dict) -> List[Dict]:
    """从单个simulation中提取agent的回复"""
    messages = simulation.get("messages", [])
    agent_responses = []

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            response = {
                "turn": i,
                "content": msg.get("content"),
                "tool_calls": msg.get("tool_calls"),
            }
            # 添加raw_data中的额外信息（如thinking内容）
            raw_data = msg.get("raw_data", {})
            if raw_data:
                if "reasoning_content" in raw_data:
                    response["reasoning_content"] = raw_data["reasoning_content"]
                if "thinking" in raw_data:
                    response["thinking"] = raw_data["thinking"]
            agent_responses.append(response)

    return agent_responses


def extract_from_result_file(result_file: Path) -> Dict:
    """从单个结果文件中提取轨迹"""
    with open(result_file) as f:
        data = json.load(f)

    # 获取模型信息
    info = data.get("info", {})
    agent_info = info.get("agent_info", {})
    model_name = agent_info.get("llm", "unknown")

    # 提取任务和simulation
    tasks = data.get("tasks", [])
    simulations = data.get("simulations", [])

    # 构建task_id到task的映射
    task_map = {str(t.get("id")): t for t in tasks}

    trajectories = []
    for sim in simulations:
        task_id = str(sim.get("task_id", ""))
        task_info = task_map.get(task_id, {})

        trajectory = {
            "task_id": task_id,
            "task_description": task_info.get("description", {}),
            "user_scenario": task_info.get("user_scenario", {}),
            "reward": sim.get("reward") or sim.get("reward_info", {}).get("reward", 0),
            "agent_responses": extract_agent_responses(sim),
            "full_messages": sim.get("messages", []),
        }
        trajectories.append(trajectory)

    return {
        "model": model_name,
        "model_args": agent_info.get("llm_args", {}),
        "domain": result_file.stem.split("_")[-2] if "_" in result_file.stem else "unknown",
        "trajectories": trajectories,
    }


def process_eval_results(input_dirs: List[str], output_dir: str):
    """处理所有评测结果目录"""
    output_path = Path(output_dir)
    by_task_dir = output_path / "by_task"
    by_model_dir = output_path / "by_model"

    # 按任务组织的数据: {domain: {task_id: [{model, response}, ...]}}
    task_data = defaultdict(lambda: defaultdict(list))
    # 按模型组织的数据: {model_name: {domain: [trajectories]}}
    model_data = defaultdict(lambda: defaultdict(list))

    # 遍历所有输入目录
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"目录不存在: {input_path}")
            continue

        # 查找所有结果JSON文件
        for model_dir in input_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            print(f"处理模型: {model_name}")

            for result_file in model_dir.glob("*.json"):
                if result_file.name.startswith("summary"):
                    continue

                try:
                    extracted = extract_from_result_file(result_file)
                    domain = extracted.get("domain", "unknown")

                    # 尝试从文件名解析domain
                    parts = result_file.stem.split("_")
                    for d in ["airline", "retail", "telecom"]:
                        if d in parts:
                            domain = d
                            break

                    # 添加到按模型组织的数据
                    model_data[model_name][domain].extend(extracted["trajectories"])

                    # 添加到按任务组织的数据
                    for traj in extracted["trajectories"]:
                        task_id = traj["task_id"]
                        task_data[domain][task_id].append({
                            "model": model_name,
                            "model_full": extracted["model"],
                            "reward": traj["reward"],
                            "agent_responses": traj["agent_responses"],
                        })

                    print(f"  - {result_file.name}: {len(extracted['trajectories'])} trajectories")
                except Exception as e:
                    print(f"  - {result_file.name}: 错误 - {e}")

    # 保存按任务组织的数据
    print("\n保存按任务组织的轨迹...")
    for domain, tasks in task_data.items():
        domain_dir = by_task_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        for task_id, responses in tasks.items():
            task_file = domain_dir / f"task_{task_id}.json"
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "task_id": task_id,
                    "domain": domain,
                    "model_responses": responses,
                }, f, indent=2, ensure_ascii=False)

        print(f"  {domain}: {len(tasks)} tasks")

    # 保存按模型组织的数据
    print("\n保存按模型组织的轨迹...")
    for model_name, domains in model_data.items():
        model_dir = by_model_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for domain, trajectories in domains.items():
            domain_file = model_dir / f"{domain}.json"
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model_name,
                    "domain": domain,
                    "trajectories": trajectories,
                }, f, indent=2, ensure_ascii=False)

        print(f"  {model_name}: {len(domains)} domains")

    # 生成汇总统计
    summary = {
        "models": list(model_data.keys()),
        "domains": list(task_data.keys()),
        "task_counts": {d: len(tasks) for d, tasks in task_data.items()},
        "model_trajectory_counts": {
            m: {d: len(trajs) for d, trajs in domains.items()}
            for m, domains in model_data.items()
        }
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n汇总保存到: {summary_file}")
    print(f"共 {len(model_data)} 个模型, {sum(len(t) for t in task_data.values())} 个任务")


def main():
    parser = argparse.ArgumentParser(description='提取评测轨迹')
    parser.add_argument(
        'input_dirs',
        nargs='+',
        help='评测结果目录 (如 eval_results/provider_a eval_results/provider_b)'
    )
    parser.add_argument(
        '-o', '--output',
        default='data/trajectories',
        help='输出目录 (默认: data/trajectories)'
    )

    args = parser.parse_args()
    process_eval_results(args.input_dirs, args.output)


if __name__ == '__main__':
    main()
