"""
分析失败案例
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_failures(domain_file, domain_name):
    """分析失败案例"""
    data = load_json(domain_file)
    simulations = data.get('simulations', [])
    tasks = {task['id']: task for task in data.get('tasks', [])}
    
    failures = []
    
    for sim in simulations:
        reward = sim.get('reward_info', {}).get('reward', 0)
        
        if reward == 0:  # 失败的任务
            task_id = sim.get('task_id', 'unknown')
            task = tasks.get(task_id, {})
            
            failure_info = {
                'domain': domain_name,
                'task_id': task_id,
                'description': task.get('description', {}).get('purpose', 'No description'),
                'termination': sim.get('termination_reason', 'unknown'),
                'steps': len(sim.get('messages', [])),
                'missing_actions': [],
                'wrong_actions': [],
                'nl_failures': [],
            }
            
            # 分析 action 失败
            reward_info = sim.get('reward_info', {})
            action_checks = reward_info.get('action_checks', []) or []
            
            for check in action_checks:
                if check.get('score', 0) == 0:
                    action = check.get('action', {})
                    action_name = action.get('name', 'unknown')
                    action_args = action.get('arguments', {})
                    
                    if not check.get('found', True):
                        failure_info['missing_actions'].append({
                            'name': action_name,
                            'expected_args': action_args,
                        })
                    else:
                        failure_info['wrong_actions'].append({
                            'name': action_name,
                            'expected_args': action_args,
                            'reason': check.get('reason', 'unknown'),
                        })
            
            # 分析 NL assertion 失败
            nl_checks = reward_info.get('nl_assertions', []) or []
            
            for check in nl_checks:
                if check.get('score', 0) == 0:
                    failure_info['nl_failures'].append(check.get('assertion', 'unknown'))
            
            failures.append(failure_info)
    
    return failures

def generate_failure_report(eval_dir):
    """生成失败案例报告"""
    eval_path = Path(eval_dir)
    
    # 读取 summary
    summary_files = list(eval_path.glob('summary_*.json'))
    if not summary_files:
        print(f"No summary file found in {eval_dir}")
        return
    
    summary = load_json(summary_files[0])
    domains_data = summary.get('domains', {})
    
    all_failures = []
    
    for domain, domain_summary in domains_data.items():
        result_file = eval_path / domain_summary.get('result_file', '')
        if result_file.exists():
            failures = analyze_failures(result_file, domain)
            all_failures.extend(failures)
    
    # 生成报告
    report = []
    report.append(f"# {summary.get('task_name', 'unknown')} 失败案例分析")
    report.append("")
    report.append(f"**总失败任务数**: {len(all_failures)}")
    report.append("")
    
    # 按领域分组
    failures_by_domain = defaultdict(list)
    for failure in all_failures:
        failures_by_domain[failure['domain']].append(failure)
    
    # 统计失败原因
    missing_action_counter = Counter()
    wrong_action_counter = Counter()
    
    for failure in all_failures:
        for action in failure['missing_actions']:
            missing_action_counter[action['name']] += 1
        for action in failure['wrong_actions']:
            wrong_action_counter[action['name']] += 1
    
    report.append("## 失败原因统计")
    report.append("")
    
    if missing_action_counter:
        report.append("### 缺失的工具调用 (Top 10)")
        report.append("")
        for action, count in missing_action_counter.most_common(10):
            report.append(f"- `{action}`: {count} 次")
        report.append("")
    
    if wrong_action_counter:
        report.append("### 参数错误的工具调用 (Top 10)")
        report.append("")
        for action, count in wrong_action_counter.most_common(10):
            report.append(f"- `{action}`: {count} 次")
        report.append("")
    
    # 详细失败案例
    for domain in sorted(failures_by_domain.keys()):
        failures = failures_by_domain[domain]
        
        report.append(f"## {domain.capitalize()} 领域失败案例 ({len(failures)} 个)")
        report.append("")
        
        # 只展示前 10 个失败案例
        for i, failure in enumerate(failures[:10], 1):
            report.append(f"### 案例 {i}: {failure['task_id']}")
            report.append("")
            report.append(f"**描述**: {failure['description']}")
            report.append("")
            report.append(f"**终止原因**: {failure['termination']}")
            report.append(f"**步数**: {failure['steps']}")
            report.append("")
            
            if failure['missing_actions']:
                report.append("**缺失的工具调用**:")
                report.append("")
                for action in failure['missing_actions']:
                    report.append(f"- `{action['name']}`")
                    if action['expected_args']:
                        report.append(f"  - 期望参数: `{action['expected_args']}`")
                report.append("")
            
            if failure['wrong_actions']:
                report.append("**参数错误的工具调用**:")
                report.append("")
                for action in failure['wrong_actions']:
                    report.append(f"- `{action['name']}`")
                    if action['expected_args']:
                        report.append(f"  - 期望参数: `{action['expected_args']}`")
                    if action.get('reason'):
                        report.append(f"  - 错误原因: {action['reason']}")
                report.append("")
            
            if failure['nl_failures']:
                report.append("**NL Assertion 失败**:")
                report.append("")
                for assertion in failure['nl_failures']:
                    report.append(f"- {assertion}")
                report.append("")
            
            report.append("---")
            report.append("")
        
        if len(failures) > 10:
            report.append(f"*还有 {len(failures) - 10} 个失败案例未展示*")
            report.append("")
    
    return "\n".join(report)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_failures.py <eval_results_dir>")
        sys.exit(1)
    
    eval_dir = sys.argv[1]
    report = generate_failure_report(eval_dir)
    
    if report:
        # 保存报告
        output_file = Path(eval_dir) / "failure_analysis.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {output_file}")
        print("\n" + report[:5000])  # 只打印前 5000 字符
        if len(report) > 5000:
            print("\n... (truncated, see full report in file)")

if __name__ == "__main__":
    main()
