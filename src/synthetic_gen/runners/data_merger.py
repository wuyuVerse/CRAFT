"""
Data Merger

合并真实任务数据和合成任务数据
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List
from loguru import logger


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


class DataMerger:
    """数据合并器 - 合并真实任务和合成任务数据"""
    
    def __init__(
        self,
        real_data_dir: str,
        synthetic_data_dir: str,
        output_dir: str,
        merge_ratio: Dict[str, float] = None,  # {"real": 0.6, "synthetic": 0.4}
        shuffle: bool = True,
        add_metadata: bool = True,
    ):
        self.real_data_dir = Path(real_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.output_dir = Path(output_dir)
        self.merge_ratio = merge_ratio or {"real": 0.5, "synthetic": 0.5}
        self.shuffle = shuffle
        self.add_metadata = add_metadata
        
        logger.info(
            f"DataMerger initialized: "
            f"real={real_data_dir}, synthetic={synthetic_data_dir}, "
            f"ratio={self.merge_ratio}"
        )
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """加载JSONL文件"""
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse line in {file_path}: {e}")
        
        return data
    
    def _convert_synthetic_format(self, data: Dict, domain: str) -> Dict:
        """转换合成数据格式以匹配真实任务格式"""
        # tau-bench-gen的格式:
        # {
        #   "messages": [...],
        #   "tools": [...],
        #   "statistics": {...},
        #   "user_info": {...},
        #   "task_data": {...}
        # }
        
        # 转换为统一格式
        converted = {
            "messages": data["messages"],
            "tools": json.dumps(data.get("tools", [])),  # 转换为JSON字符串
        }
        
        # 添加元数据
        if self.add_metadata:
            converted["metadata"] = {
                "source": "synthetic",
                "domain": domain,
                "statistics": data.get("statistics", {}),
                "user_info": data.get("user_info", {}),
                "task_data": data.get("task_data", {})
            }
        
        return converted
    
    def _add_real_metadata(self, data: Dict, domain: str) -> Dict:
        """为真实数据添加元数据"""
        if self.add_metadata and "metadata" not in data:
            data["metadata"] = {
                "source": "real",
                "domain": domain,
            }
        return data
    
    def merge_domain(self, domain: str):
        """合并单个领域的数据"""
        logger.info(f"Merging data for domain: {domain}")
        
        # 读取真实任务数据
        real_file = self.real_data_dir / domain / "sft_data.jsonl"
        real_data = self._load_jsonl(real_file)
        logger.info(f"Loaded {len(real_data)} real samples from {real_file}")
        
        # 读取合成任务数据
        synthetic_file = self.synthetic_data_dir / domain / "synthetic_data.jsonl"
        synthetic_data = self._load_jsonl(synthetic_file)
        logger.info(f"Loaded {len(synthetic_data)} synthetic samples from {synthetic_file}")
        
        if not real_data and not synthetic_data:
            logger.warning(f"No data found for domain {domain}")
            return
        
        # 转换合成数据格式
        synthetic_data = [self._convert_synthetic_format(d, domain) for d in synthetic_data]
        
        # 为真实数据添加元数据
        real_data = [self._add_real_metadata(d, domain) for d in real_data]
        
        # 按比例采样
        if real_data and synthetic_data:
            total_samples = len(real_data) + len(synthetic_data)
            real_count = int(total_samples * self.merge_ratio["real"])
            synthetic_count = int(total_samples * self.merge_ratio["synthetic"])
            
            real_sampled = random.sample(real_data, min(real_count, len(real_data)))
            synthetic_sampled = random.sample(synthetic_data, min(synthetic_count, len(synthetic_data)))
        elif real_data:
            real_sampled = real_data
            synthetic_sampled = []
        else:
            real_sampled = []
            synthetic_sampled = synthetic_data
        
        # 合并并打乱
        merged = real_sampled + synthetic_sampled
        if self.shuffle:
            random.shuffle(merged)
        
        # 保存
        output_file = self.output_dir / domain / "sft_data.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(
            f"{domain}: 合并完成 - "
            f"真实{len(real_sampled)} + 合成{len(synthetic_sampled)} = {len(merged)} "
            f"-> {output_file}"
        )
        
        # 保存统计信息
        stats_file = self.output_dir / domain / "merge_stats.json"
        stats = {
            "domain": domain,
            "real_samples": len(real_sampled),
            "synthetic_samples": len(synthetic_sampled),
            "total_samples": len(merged),
            "real_ratio": len(real_sampled) / len(merged) if merged else 0,
            "synthetic_ratio": len(synthetic_sampled) / len(merged) if merged else 0,
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def merge_all(self, domains: List[str] = None):
        """合并所有领域"""
        if domains is None:
            domains = ["airline", "retail", "telecom"]
        
        for domain in domains:
            self.merge_domain(domain)
        
        logger.info("All domains merged successfully")


# CLI接口
def main():
    """命令行接口"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Data Merger")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = _expand_env(yaml.safe_load(f))
    
    # 创建merger
    merger = DataMerger(
        real_data_dir=config['input']['real_data_dir'],
        synthetic_data_dir=config['input']['synthetic_data_dir'],
        output_dir=config['output']['dir'],
        merge_ratio=config.get('merge_ratio', {"real": 0.5, "synthetic": 0.5}),
        shuffle=config.get('shuffle', True),
        add_metadata=config.get('add_metadata', True),
    )
    
    # 合并所有领域
    domains = config.get('domains', ["airline", "retail", "telecom"])
    merger.merge_all(domains)


if __name__ == "__main__":
    main()
