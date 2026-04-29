#!/usr/bin/env python3
"""
实验数据准备脚本
根据论文实验设计，准备五类实验数据：
1. 主实验 (main): v4_ds_v3 + synthetic_v1 全量混合
2. 消融实验 (ablation): 4个消融配置
3. Scaling实验: 从混合数据中抽取不同规模
4. 对比学习实验 (contrast): 不同训练数据组织方式
5. 复杂度分层实验: 按任务复杂度划分
"""

import json
import random
import os
from pathlib import Path
from collections import defaultdict

# 设置随机种子保证可复现
random.seed(42)

BASE_DIR = Path(os.environ.get("CRAFT_ROOT", Path.cwd()))
OUTPUT_DIR = Path(os.environ.get("CRAFT_SFT_OUTPUT_DIR", BASE_DIR / "sft" / "data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据源
V4_DS_V3 = Path(os.environ.get("CRAFT_MAIN_REAL_DATA", BASE_DIR / "output/iterative_learning/sft_data_all.jsonl"))
SYNTHETIC_V1 = Path(os.environ.get("CRAFT_MAIN_SYNTHETIC_DATA", BASE_DIR / "output/synthetic/sft_data_all.jsonl"))
ABLATION_DIR = Path(os.environ.get("CRAFT_ABLATION_DIR", BASE_DIR / "output/experiments/ablation"))
CONTRAST_DIR = Path(os.environ.get("CRAFT_CONTRAST_DIR", BASE_DIR / "output/experiments/contrast"))


def load_jsonl(filepath):
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """保存JSONL文件"""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved: {filepath} ({len(data)} entries)")


def prepare_main_experiment():
    """1. 主实验：混合 v4_ds_v3 + synthetic_v1"""
    print("\n" + "="*60)
    print("1. 主实验 (Full CRAFT)")
    print("="*60)

    exp_dir = OUTPUT_DIR / "main"
    exp_dir.mkdir(exist_ok=True)

    # 加载数据
    print("  Loading v4_ds_v3...")
    v4_data = load_jsonl(V4_DS_V3)
    print(f"    -> {len(v4_data)} entries")

    print("  Loading synthetic_v1...")
    syn_data = load_jsonl(SYNTHETIC_V1)
    print(f"    -> {len(syn_data)} entries")

    # 混合并打乱
    mixed_data = v4_data + syn_data
    random.shuffle(mixed_data)

    print(f"  Total mixed: {len(mixed_data)} entries")
    save_jsonl(mixed_data, exp_dir / "sft_data_all.jsonl")

    return mixed_data


def prepare_ablation_experiments():
    """2. 消融实验：4个消融配置"""
    print("\n" + "="*60)
    print("2. 消融实验")
    print("="*60)

    ablations = {
        "no_iterative": "− Iterative Learning",
        "no_contrast": "− Contrastive Analysis",
        "no_error_injection": "− Error Injection",
        "no_multi_agent": "− Multi-Agent Synthesis",
        "base_real_only": "Base (Real Data Only)"
    }

    for ablation_name, description in ablations.items():
        print(f"\n  {ablation_name}: {description}")

        exp_dir = OUTPUT_DIR / "ablation" / ablation_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 合并三个领域的数据
        all_data = []
        for domain in ["airline", "retail", "telecom"]:
            src_file = ABLATION_DIR / ablation_name / domain / "sft_data.jsonl"
            if src_file.exists():
                domain_data = load_jsonl(src_file)
                all_data.extend(domain_data)
                print(f"    {domain}: {len(domain_data)}")
            else:
                print(f"    {domain}: MISSING")

        random.shuffle(all_data)
        save_jsonl(all_data, exp_dir / "sft_data_all.jsonl")


def prepare_scaling_experiments(mixed_data):
    """3. Scaling实验：从混合数据中抽取不同规模"""
    print("\n" + "="*60)
    print("3. Scaling实验")
    print("="*60)

    # 规模设置
    scales = [500, 1000, 2000, 4000, 10000, 20000, 40000, 80000]

    exp_dir = OUTPUT_DIR / "scaling"
    exp_dir.mkdir(exist_ok=True)

    # 确保数据已打乱
    data_copy = mixed_data.copy()
    random.shuffle(data_copy)

    for scale in scales:
        if scale > len(data_copy):
            print(f"  scale_{scale}: SKIP (only {len(data_copy)} available)")
            continue

        sampled = data_copy[:scale]
        save_jsonl(sampled, exp_dir / f"sft_data_{scale}.jsonl")

    # 全量数据
    save_jsonl(data_copy, exp_dir / "sft_data_all.jsonl")


def prepare_contrast_experiments(mixed_data):
    """4. 对比学习实验：从真实实验输出加载数据

    数据来源: output/experiments/contrast/
    - only_correct: 仅正确轨迹（禁用错误注入）
    - correct_error_mix: 正确+失败轨迹混合（include_failed_in_sft）
    - correct_recovery_pair: 正确+错误恢复对（启用错误注入）
    """
    print("\n" + "="*60)
    print("4. 对比学习实验")
    print("="*60)

    exp_dir = OUTPUT_DIR / "contrast"
    exp_dir.mkdir(exist_ok=True)

    # 映射关系: 源目录 -> 目标文件名
    contrast_mapping = {
        "only_correct": "correct_only.jsonl",           # 仅正确轨迹
        "correct_error_mix": "mixed_correct_error.jsonl",  # 正确+错误混合
        "correct_recovery_pair": "full_contrastive.jsonl"  # 完整对比（正确+错误恢复对）
    }

    for src_name, dst_filename in contrast_mapping.items():
        src_file = CONTRAST_DIR / src_name / "sft_data_all.jsonl"
        dst_file = exp_dir / dst_filename

        print(f"\n  {src_name} -> {dst_filename}")

        if src_file.exists():
            data = load_jsonl(src_file)
            random.shuffle(data)
            save_jsonl(data, dst_file)
        else:
            print(f"    WARNING: Source not found: {src_file}")
            # 如果源文件不存在，跳过或使用备用逻辑
            continue


def generate_summary_table(mixed_data):
    """生成实验数据总结表"""
    print("\n" + "="*60)
    print("实验数据总结")
    print("="*60)

    summary = []

    # 1. 主实验
    main_file = OUTPUT_DIR / "main" / "sft_data_all.jsonl"
    if main_file.exists():
        count = sum(1 for _ in open(main_file))
        summary.append(("主实验 (Full CRAFT)", "main/sft_data_all.jsonl", count, "v4_ds_v3 + synthetic_v1 全量混合"))

    # 2. 消融实验
    for name in ["no_iterative", "no_contrast", "no_error_injection", "no_multi_agent", "base_real_only"]:
        ablation_file = OUTPUT_DIR / "ablation" / name / "sft_data_all.jsonl"
        if ablation_file.exists():
            count = sum(1 for _ in open(ablation_file))
            desc_map = {
                "no_iterative": "移除迭代学习机制",
                "no_contrast": "移除对比分析",
                "no_error_injection": "移除错误注入",
                "no_multi_agent": "移除多智能体合成",
                "base_real_only": "仅使用真实数据"
            }
            summary.append((f"消融: {name}", f"ablation/{name}/sft_data_all.jsonl", count, desc_map[name]))

    # 3. Scaling实验
    scales = [500, 1000, 2000, 4000, 10000, 20000, 40000, 80000, "all"]
    for scale in scales:
        filename = f"sft_data_{scale}.jsonl" if scale != "all" else "sft_data_all.jsonl"
        scale_file = OUTPUT_DIR / "scaling" / filename
        if scale_file.exists():
            count = sum(1 for _ in open(scale_file))
            summary.append((f"Scaling: {scale}", f"scaling/{filename}", count, f"从混合数据随机抽取{scale}条"))

    # 4. 对比学习实验
    contrast_files = [
        ("correct_only.jsonl", "仅正确轨迹（禁用错误注入）"),
        ("mixed_correct_error.jsonl", "正确+失败轨迹混合"),
        ("full_contrastive.jsonl", "正确+错误恢复对（完整对比）")
    ]
    for filename, desc in contrast_files:
        contrast_file = OUTPUT_DIR / "contrast" / filename
        if contrast_file.exists():
            count = sum(1 for _ in open(contrast_file))
            summary.append((f"对比: {filename.replace('.jsonl','')}", f"contrast/{filename}", count, desc))

    # 打印表格
    print("\n| 实验 | 数据文件 | 条目数 | 说明 |")
    print("|------|----------|--------|------|")
    for name, path, count, desc in summary:
        print(f"| {name} | {path} | {count:,} | {desc} |")

    # 保存到文件
    with open(OUTPUT_DIR / "README.md", 'w') as f:
        f.write("# SFT 实验数据\n\n")
        f.write("## 数据总结\n\n")
        f.write("| 实验 | 数据文件 | 条目数 | 说明 |\n")
        f.write("|------|----------|--------|------|\n")
        for name, path, count, desc in summary:
            f.write(f"| {name} | {path} | {count:,} | {desc} |\n")

        f.write("\n## 实验设计\n\n")
        f.write("### 1. 主实验 (Main)\n")
        f.write("- 数据: v4_ds_v3 + synthetic_v1 全量混合打乱\n")
        f.write("- 用途: 验证CRAFT框架整体性能\n\n")

        f.write("### 2. 消融实验 (Ablation)\n")
        f.write("- no_iterative: 移除迭代学习机制\n")
        f.write("- no_contrast: 移除对比分析\n")
        f.write("- no_error_injection: 移除错误注入\n")
        f.write("- no_multi_agent: 移除多智能体合成\n")
        f.write("- base_real_only: 仅使用真实数据（基线）\n\n")

        f.write("### 3. Scaling实验\n")
        f.write("- 从混合数据中随机抽取不同规模\n")
        f.write("- 规模: 500, 1K, 2K, 4K, 10K, 20K, 40K, 80K, 全量\n\n")

        f.write("### 4. 对比学习实验 (Contrast)\n")
        f.write("- correct_only: 仅正确轨迹（禁用错误注入）\n")
        f.write("- mixed_correct_error: 正确+失败轨迹混合\n")
        f.write("- full_contrastive: 正确+错误恢复对（完整对比）\n")

    print(f"\n  Summary saved to: {OUTPUT_DIR / 'README.md'}")


def main():
    print("="*60)
    print("SFT 实验数据准备")
    print("="*60)

    # 1. 主实验 - 获取混合数据
    mixed_data = prepare_main_experiment()

    # 2. 消融实验
    prepare_ablation_experiments()

    # 3. Scaling实验
    prepare_scaling_experiments(mixed_data)

    # 4. 对比学习实验
    prepare_contrast_experiments(mixed_data)

    # 生成总结表
    generate_summary_table(mixed_data)

    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == "__main__":
    main()
