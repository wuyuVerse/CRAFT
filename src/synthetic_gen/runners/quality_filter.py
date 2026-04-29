"""
Quality Filter

过滤低质量的合成数据
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List
from loguru import logger


class QualityFilter:
    """质量过滤器 - 过滤低质量的合成数据"""
    
    def __init__(
        self,
        min_turns: int = 3,              # 最少轮次
        max_turns: int = 30,             # 最多轮次
        min_tool_calls: int = 2,         # 最少工具调用
        max_hallucination_rate: float = 0.3,  # 最大幻觉率
        min_quality_score: int = 3,      # 最低质量分数
    ):
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.min_tool_calls = min_tool_calls
        self.max_hallucination_rate = max_hallucination_rate
        self.min_quality_score = min_quality_score
        
        logger.info(
            f"QualityFilter initialized: "
            f"turns=[{min_turns},{max_turns}], "
            f"min_tool_calls={min_tool_calls}, "
            f"max_hall_rate={max_hallucination_rate}"
        )
    
    def filter_sample(self, data: Dict) -> Tuple[bool, str]:
        """过滤单个样本"""
        messages = data.get("messages", [])
        stats = data.get("statistics", {})
        
        # 检查轮次
        num_turns = len([m for m in messages if m.get("role") == "user"])
        if num_turns < self.min_turns:
            return False, f"轮次太少: {num_turns} < {self.min_turns}"
        if num_turns > self.max_turns:
            return False, f"轮次太多: {num_turns} > {self.max_turns}"
        
        # 检查工具调用
        num_tool_calls = 0
        for m in messages:
            if m.get("role") == "assistant":
                if m.get("tool_calls"):
                    num_tool_calls += len(m.get("tool_calls", []))
        
        if num_tool_calls < self.min_tool_calls:
            return False, f"工具调用太少: {num_tool_calls} < {self.min_tool_calls}"
        
        # 检查幻觉率
        hall_stats = stats.get("toolagent_hallucination", {})
        total_hall = hall_stats.get("total_hallucination_count", 0)
        if num_tool_calls > 0:
            hall_rate = total_hall / num_tool_calls
            if hall_rate > self.max_hallucination_rate:
                return False, f"幻觉率太高: {hall_rate:.2f} > {self.max_hallucination_rate}"
        
        # 检查质量分数（如果有）
        quality_score = data.get("quality_score", 5)  # 默认5分
        if quality_score < self.min_quality_score:
            return False, f"质量分数太低: {quality_score} < {self.min_quality_score}"
        
        return True, "OK"
    
    def filter_file(self, input_file: str, output_file: str):
        """过滤整个文件"""
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        filtered = []
        rejected = []
        
        logger.info(f"Filtering file: {input_file}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    passed, reason = self.filter_sample(data)
                    
                    if passed:
                        filtered.append(data)
                    else:
                        rejected.append({"line": line_num, "data": data, "reason": reason})
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num}: {e}")
                    rejected.append({"line": line_num, "data": None, "reason": f"JSON parse error: {e}"})
        
        # 保存过滤后的数据
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in filtered:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 保存被拒绝的数据（用于分析）
        rejected_file = str(output_path).replace('.jsonl', '_rejected.jsonl')
        with open(rejected_file, 'w', encoding='utf-8') as f:
            for item in rejected:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        total = len(filtered) + len(rejected)
        pass_rate = len(filtered) / total * 100 if total > 0 else 0
        
        logger.info(
            f"过滤完成: 保留{len(filtered)}/{total} ({pass_rate:.1f}%) "
            f"-> {output_file}"
        )
        logger.info(f"被拒绝的样本保存到: {rejected_file}")
        
        # 保存统计信息
        stats_file = str(output_path).replace('.jsonl', '_filter_stats.json')
        stats = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "total_samples": total,
            "filtered_samples": len(filtered),
            "rejected_samples": len(rejected),
            "pass_rate": pass_rate,
            "rejection_reasons": {}
        }
        
        # 统计拒绝原因
        for item in rejected:
            reason = item["reason"].split(":")[0]  # 取原因的第一部分
            stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"过滤统计保存到: {stats_file}")
    
    def filter_directory(self, input_dir: str, output_dir: str, domains: List[str] = None):
        """过滤整个目录"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if domains is None:
            domains = ["airline", "retail", "telecom"]
        
        for domain in domains:
            input_file = input_path / domain / "synthetic_data.jsonl"
            output_file = output_path / domain / "synthetic_data_filtered.jsonl"
            
            if input_file.exists():
                self.filter_file(str(input_file), str(output_file))
            else:
                logger.warning(f"File not found: {input_file}")
        
        logger.info("Directory filtering completed")


# CLI接口
def main():
    """命令行接口"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Quality Filter")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--input", type=str, help="输入文件或目录")
    parser.add_argument("--output", type=str, help="输出文件或目录")
    parser.add_argument("--mode", type=str, choices=["file", "directory"], default="file", help="过滤模式")
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        filter_config = config.get('quality_filter', {})
    else:
        filter_config = {}
    
    # 创建filter
    quality_filter = QualityFilter(
        min_turns=filter_config.get('min_turns', 3),
        max_turns=filter_config.get('max_turns', 30),
        min_tool_calls=filter_config.get('min_tool_calls', 2),
        max_hallucination_rate=filter_config.get('max_hallucination_rate', 0.3),
        min_quality_score=filter_config.get('min_quality_score', 3),
    )
    
    # 执行过滤
    if args.mode == "file":
        quality_filter.filter_file(args.input, args.output)
    else:
        domains = config.get('domains', ["airline", "retail", "telecom"]) if args.config else None
        quality_filter.filter_directory(args.input, args.output, domains)


if __name__ == "__main__":
    main()
