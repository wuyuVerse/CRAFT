"""
详细分析评测结果并生成报告
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_domain_detailed(domain_file):
    """详细分析单个领域的结果"""
    data = load_json(domain_file)
    simulations = data.get('simulations', [])
    
    stats = {
        'total_tasks': len(simulations),
        'successful_tasks': 0,
        'failed_tasks': 0,
        'missing_actions': defaultdict(int),  # 应该调用但没调用
        'wrong_parameters': defaultdict(int),  # 调用了但参数错误
        'termination_reasons': Counter(),
        'avg_steps': 0,
        'total_steps': 0,
        'max_steps_tasks': 0,
    }
    
    for sim in simulations:
        # 成功/失败统计
        reward = sim.get('reward_info', {}).get('reward', 0)
        if reward > 0:
            stats['successful_tasks'] += 1
        else:
            stats['failed_tasks'] += 1
        
        # 步数统计
        steps = len(sim.get('messages', []))
        stats['total_steps'] += steps
        
        # 终止原因
        termination = sim.get('termination_reason', 'unknown')
        stats['termination_reasons'][termination] += 1
        
        if termination == 'max_steps':
            stats['max_steps_tasks'] += 1
        
        # Action 失败分析
        reward_info = sim.get('reward_info', {})
        action_reward_info = reward_info.get('action_reward_info', {})
        action_checks = action_reward_info.get('action_checks', [])
        
        for check in action_checks:
            score = check.get('score', 0)
            action = check.get('action', {})
            action_name = action.get('name', 'unknown')
            
            if score == 0:
                # 判断是缺失还是参数错误
                # 如果 check 中有 'found' 字段且为 False，说明是缺失
                if not check.get('found', True):
                    stats['missing_actions'][action_name] += 1
                else:
                    stats['wrong_parameters'][action_name] += 1
    
    if stats['total_tasks'] > 0:
        stats['avg_steps'] = stats['total_steps'] / stats['total_tasks']
    
    return stats

def generate_detailed_report(eval_dir):
    """生成详细分析报告"""
    eval_path = Path(eval_dir)
    
    # 读取 summary
    summary_files = list(eval_path.glob('summary_*.json'))
    if not summary_files:
        print(f"No summary file found in {eval_dir}")
        return
    
    summary = load_json(summary_files[0])
    
    report = []
    report.append(f"# {summary.get('task_name', 'unknown')} 评测分析报告")
    report.append("")
    
    # Overall 统计表格
    report.append("## 总体表现")
    report.append("")
    report.append("| 领域 | 成功率 | 平均奖励 | 任务数 |")
    report.append("|------|--------|----------|--------|")
    
    domains_data = summary.get('domains', {})
    for domain, domain_summary in domains_data.items():
        success_rate = domain_summary.get('success_rate', 0)
        avg_reward = domain_summary.get('avg_reward', 0)
        num_tasks = domain_summary.get('num_tasks', 0)
        report.append(f"| {domain.capitalize()} | {success_rate:.1f}% | {avg_reward:.2f} | {num_tasks} |")
    
    overall = summary.get('overall', {})
    report.append(f"| **Overall** | **{overall.get('success_rate', 0):.1f}%** | **{overall.get('avg_reward', 0):.2f}** | **{overall.get('total_tasks', 0)}** |")
    report.append("")
    
    # 失败模式分析
    report.append("## 失败模式分析")
    report.append("")
    
    for domain, domain_summary in sorted(domains_data.items(), key=lambda x: x[1].get('success_rate', 100)):
        success_rate = domain_summary.get('success_rate', 0)
        failure_rate = 100 - success_rate
        
        if failure_rate < 5:
            continue  # 跳过表现很好的领域
        
        report.append(f"### {domain.capitalize()} 领域 ({failure_rate:.1f}% 失败率)")
        report.append("")
        
        result_file = eval_path / domain_summary.get('result_file', '')
        if result_file.exists():
            stats = analyze_domain_detailed(result_file)
            
            report.append("**主要问题:**")
            report.append("")
            
            # 缺失 Tool 调用
            if stats['missing_actions']:
                total_missing = sum(stats['missing_actions'].values())
                report.append(f"1. **缺失 Tool 调用 ({total_missing}次)** - 模型应该调用但没有调用")
                for action, count in sorted(stats['missing_actions'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    report.append(f"   - `{action}`: {count}次")
                report.append("")
            
            # 参数错误
            if stats['wrong_parameters']:
                total_wrong = sum(stats['wrong_parameters'].values())
                report.append(f"2. **参数错误 ({total_wrong}次)** - 调用了但参数不正确")
                for action, count in sorted(stats['wrong_parameters'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    report.append(f"   - `{action}`: {count}次")
                report.append("")
            
            # 其他问题
            if stats['max_steps_tasks'] > 0:
                report.append(f"3. **超时任务 ({stats['max_steps_tasks']}次)** - 达到最大步数限制")
                report.append("")
            
            report.append(f"**平均步数**: {stats['avg_steps']:.1f}")
            report.append("")
    
    # 改进建议
    report.append("## 改进建议")
    report.append("")
    
    for domain, domain_summary in sorted(domains_data.items(), key=lambda x: x[1].get('success_rate', 100)):
        success_rate = domain_summary.get('success_rate', 0)
        
        if success_rate >= 85:
            continue
        
        report.append(f"### {domain.capitalize()} 领域 (当前成功率: {success_rate:.1f}%)")
        report.append("")
        
        result_file = eval_path / domain_summary.get('result_file', '')
        if result_file.exists():
            stats = analyze_domain_detailed(result_file)
            
            # 针对性建议
            if stats['missing_actions']:
                report.append("**针对缺失 Tool 调用:**")
                report.append("")
                top_missing = sorted(stats['missing_actions'].items(), key=lambda x: x[1], reverse=True)[:3]
                for action, count in top_missing:
                    report.append(f"- 增加 `{action}` 的训练数据，确保模型在相关场景下能正确调用")
                    report.append(f"- 在 CoT 提示中强调 `{action}` 的使用时机")
                report.append("")
            
            if stats['wrong_parameters']:
                report.append("**针对参数错误:**")
                report.append("")
                top_wrong = sorted(stats['wrong_parameters'].items(), key=lambda x: x[1], reverse=True)[:3]
                for action, count in top_wrong:
                    report.append(f"- 为 `{action}` 添加参数验证的 CoT 提示")
                    report.append(f"- 增加 `{action}` 参数选择的训练样本")
                    report.append(f"- 将 `{action}` 加入弱点工具列表，优先采样")
                report.append("")
            
            if stats['max_steps_tasks'] > 0:
                report.append("**针对超时问题:**")
                report.append("")
                report.append(f"- 优化多步骤任务的执行效率")
                report.append(f"- 增加任务规划能力的训练")
                report.append("")
    
    # V2 架构优化建议
    report.append("## V2 架构优化建议")
    report.append("")
    report.append("基于当前评测结果，V2 架构应该重点关注:")
    report.append("")
    
    all_missing = defaultdict(int)
    all_wrong = defaultdict(int)
    
    for domain, domain_summary in domains_data.items():
        result_file = eval_path / domain_summary.get('result_file', '')
        if result_file.exists():
            stats = analyze_domain_detailed(result_file)
            for action, count in stats['missing_actions'].items():
                all_missing[action] += count
            for action, count in stats['wrong_parameters'].items():
                all_wrong[action] += count
    
    if all_missing:
        report.append("### 1. 弱点工具识别")
        report.append("")
        report.append("将以下工具加入 `WEAK_TOOLS` 列表:")
        report.append("```python")
        report.append("WEAK_TOOLS = {")
        for domain in domains_data.keys():
            domain_tools = []
            result_file = eval_path / domains_data[domain].get('result_file', '')
            if result_file.exists():
                stats = analyze_domain_detailed(result_file)
                domain_tools = [action for action, count in sorted(stats['missing_actions'].items(), key=lambda x: x[1], reverse=True)[:3]]
            if domain_tools:
                report.append(f'    "{domain}": {domain_tools},')
        report.append("}")
        report.append("```")
        report.append("")
    
    if all_wrong:
        report.append("### 2. CoT 提示增强")
        report.append("")
        report.append("为以下工具添加详细的 CoT 提示:")
        for action, count in sorted(all_wrong.items(), key=lambda x: x[1], reverse=True)[:5]:
            report.append(f"- `{action}`: 参数验证提示")
        report.append("")
    
    report.append("### 3. 对比反馈注入")
    report.append("")
    report.append("启用 `enable_contrast_feedback: true`，在 retry 时注入成功轨迹对比分析，帮助模型理解:")
    report.append("- 正确的工具调用顺序")
    report.append("- 正确的参数选择方法")
    report.append("- 关键的决策点")
    report.append("")
    
    return "\n".join(report)

def main():
    if len(sys.argv) < 2:
        print("Usage: python detailed_analysis.py <eval_results_dir>")
        sys.exit(1)
    
    eval_dir = sys.argv[1]
    report = generate_detailed_report(eval_dir)
    
    if report:
        # 保存报告
        output_file = Path(eval_dir) / "detailed_analysis_report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {output_file}")
        print("\n" + report)

if __name__ == "__main__":
    main()
