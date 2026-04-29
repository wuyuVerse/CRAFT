"""
Synthetic Data Generation Runner

封装 tau-bench-gen 的 PlayGround，提供统一的接口
"""

import asyncio
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from asyncio import Lock, Semaphore

import openai
from loguru import logger

from ..core.playground import PlayGround
from ..utils.logger import log, setup_logger


class ClientPool:
    """OpenAI客户端池：为每个API创建客户端实例，支持请求级轮询负载均衡"""
    
    def __init__(self, api_pool: List[Dict]):
        self.api_pool = api_pool
        # 为每个API创建一个AsyncOpenAI客户端实例
        self.clients = [
            openai.AsyncOpenAI(api_key=api["api_key"], base_url=api["base_url"])
            for api in api_pool
        ]
        self.models = [api["model"] for api in api_pool]
        # 全局请求计数器，用于真正的请求级轮询负载均衡
        self._request_counter = 0
        self._counter_lock = Lock()
        # 统计每个API的使用次数
        self._usage_count = [0] * len(api_pool)
        log(f"客户端池已初始化: {len(self.clients)}个客户端")
    
    def get_client(self, index: int = None) -> tuple:
        """
        获取客户端和对应的模型
        
        Args:
            index: 指定客户端索引，如果为None则使用轮询
        
        Returns:
            (client, model)
        """
        if index is None:
            # 使用轮询方式分配（线程安全）
            index = self._request_counter % len(self.clients)
            self._request_counter += 1
        else:
            index = index % len(self.clients)
        
        self._usage_count[index] += 1
        return self.clients[index], self.models[index]
    
    def get_next_client(self) -> tuple:
        """
        获取下一个客户端（轮询方式），用于负载均衡
        
        Returns:
            (client, model, index)
        """
        index = self._request_counter % len(self.clients)
        self._request_counter += 1
        self._usage_count[index] += 1
        return self.clients[index], self.models[index], index
    
    def get_all_clients(self) -> List[tuple]:
        """
        获取所有客户端和模型的列表
        
        Returns:
            [(client, model), ...]
        """
        return list(zip(self.clients, self.models))
    
    def get_usage_stats(self) -> str:
        """获取客户端使用统计"""
        stats = "\n=== 客户端池使用统计 ==="
        for i, (api, count) in enumerate(zip(self.api_pool, self._usage_count)):
            stats += f"\nAPI {i+1} ({api['base_url']}): {count}次"
        return stats


class APIRouter:
    """API任务分发路由器 - 负载均衡"""
    
    def __init__(self, api_pool: List[Dict], max_concurrent_per_api: int = 100):
        self.api_pool = api_pool
        self.max_concurrent_per_api = max_concurrent_per_api
        # 为每个API创建独立的信号量控制并发
        self.semaphores = [Semaphore(max_concurrent_per_api) for _ in api_pool]
        self.api_usage_count = [0] * len(api_pool)  # 记录每个API的使用次数
        self.lock = Lock()
        
    async def get_api_config(self, task_id: int) -> tuple:
        """根据任务ID轮询分配API配置"""
        api_index = task_id % len(self.api_pool)
        semaphore = self.semaphores[api_index]
        api_config = self.api_pool[api_index]
        
        async with self.lock:
            self.api_usage_count[api_index] += 1
        
        return semaphore, api_config, api_index
    
    def get_stats(self) -> str:
        """获取API使用统计"""
        stats = "\n=== API使用统计 ==="
        for i, (config, count) in enumerate(zip(self.api_pool, self.api_usage_count)):
            stats += f"\nAPI {i+1} ({config['model']}): {count}次"
        return stats


class SyntheticRunner:
    """合成任务生成器 - 封装 tau-bench-gen 的 PlayGround"""
    
    def __init__(
        self,
        domain: str,                    # airline/retail/telecom
        num_samples: int,               # 生成样本数
        api_pool: List[Dict],           # API配置池
        output_dir: str,                # 输出目录
        max_turn: int = 30,             # 最大轮次
        quality_threshold: int = 4,     # 质量阈值
        max_concurrent_per_api: int = 100,  # 每个API最大并发数
        self_reflection_iterations: int = 1,  # 自我反思迭代次数
        enable_pruning: bool = False,   # 是否启用错误剪枝
        num_voting_agents: int = 3,     # 投票Agent数量
        enable_action_validation: bool = False,  # V1: 是否启用Action验证
        use_real_task_seed: bool = False,  # V1: 是否使用真实任务seed
    ):
        self.domain = domain
        self.num_samples = num_samples
        self.output_dir = Path(output_dir) / domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志（保存到output_dir/logs/）
        log_filename = f"synthetic_gen_{domain}_{Path(output_dir).name}.log"
        setup_logger(output_dir=output_dir, log_filename=log_filename)
        
        # 初始化客户端池和路由器
        self.client_pool = ClientPool(api_pool)
        self.router = APIRouter(api_pool, max_concurrent_per_api)
        
        # PlayGround配置
        self.max_turn = max_turn
        self.quality_threshold = quality_threshold
        self.self_reflection_iterations = self_reflection_iterations
        self.enable_pruning = enable_pruning
        self.num_voting_agents = num_voting_agents
        self.enable_action_validation = enable_action_validation  # V1
        self.use_real_task_seed = use_real_task_seed  # V1
        
        # 用于保存数据的锁
        self.lock = Lock()
        
        logger.info(
            f"SyntheticRunner initialized for {domain}: "
            f"{num_samples} samples, quality_threshold={quality_threshold}"
        )
    
    async def generate_single(self, task_id: int) -> Optional[Dict]:
        """生成单个合成任务"""
        semaphore, api_config, api_index = await self.router.get_api_config(task_id)
        
        async with semaphore:
            try:
                client, model = self.client_pool.get_client(api_index)
                
                playground = PlayGround(
                    client_pool=self.client_pool,
                    primary_client=client,
                    primary_model=model,
                    max_turn=self.max_turn,
                    quality_threshold=self.quality_threshold,
                    self_reflection_iterations=self.self_reflection_iterations,
                    enable_pruning=self.enable_pruning,
                    num_voting_agents=self.num_voting_agents,
                    domain=self.domain,  # 固定领域
                    enable_action_validation=self.enable_action_validation,  # V1
                    use_real_task_seed=self.use_real_task_seed  # V1
                )
                
                await playground.interact()
                data = await playground.decode_history()
                
                if data:
                    # 保存数据
                    async with self.lock:
                        output_file = self.output_dir / "synthetic_data.jsonl"
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                    log(f"[成功] 任务{task_id} (API{api_index+1}-{model}) 数据已保存")
                    return {"success": True, "stats": data.get('statistics', {})}
                else:
                    log(f"[失败] 任务{task_id} (API{api_index+1}-{model}) 数据未通过质量检查")
                    return {"success": False, "stats": None}
                    
            except Exception as e:
                log(f"[错误] 任务{task_id} (API{api_index+1}) 生成失败: {str(e)}")
                logger.error(f"Task {task_id} failed: {e}")
                return {"success": False, "stats": None}
    
    async def run(self):
        """批量生成合成任务"""
        log(f"开始生成 {self.num_samples} 条 {self.domain} 数据...\n")
        logger.info(f"Starting generation of {self.num_samples} samples for {self.domain}")
        
        # 为每个任务分配唯一ID
        tasks = [self.generate_single(task_id=i) for i in range(self.num_samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success_count = 0
        fail_count = 0
        exception_count = 0
        all_stats = []
        
        for r in results:
            if isinstance(r, Exception):
                exception_count += 1
                fail_count += 1
            elif isinstance(r, dict):
                if r.get("success"):
                    success_count += 1
                    if r.get("stats"):
                        all_stats.append(r["stats"])
                else:
                    fail_count += 1
            else:
                fail_count += 1
        
        log(f"\n{'='*60}")
        log(f"生成完成: 成功 {success_count}/{self.num_samples}, 失败 {fail_count}/{self.num_samples}")
        if exception_count > 0:
            log(f"异常数量: {exception_count}")
        log(self.router.get_stats())
        log(self.client_pool.get_usage_stats())
        log(f"{'='*60}\n")
        
        logger.info(
            f"Generation completed: {success_count}/{self.num_samples} successful, "
            f"{fail_count} failed, {exception_count} exceptions"
        )
        
        # 保存统计信息
        if all_stats:
            self._save_aggregated_stats(all_stats, success_count, self.num_samples)
        
        return success_count
    
    def _save_aggregated_stats(self, all_stats: List[Dict], success_count: int, total_count: int):
        """汇总并保存所有成功任务的统计信息"""
        from datetime import datetime
        
        try:
            # 汇总统计
            aggregated = {
                "toolagent_hallucination": {
                    "total_hallucination_count": 0,
                    "hallucinated_tool_count": 0,
                    "hallucinated_param_count": 0,
                    "resolved_count": 0,
                    "failed_count": 0
                },
                "user_correction": {
                    "total_corrections": 0,
                    "all_iterations": []
                },
                "pruning": {
                    "total_segments_detected": 0,
                    "segments_pruned": 0,
                    "segments_kept": 0
                }
            }
            
            # 累加每个任务的统计
            for stats in all_stats:
                # ToolAgent幻觉
                if "toolagent_hallucination" in stats:
                    th = stats["toolagent_hallucination"]
                    aggregated["toolagent_hallucination"]["total_hallucination_count"] += th.get("total_hallucination_count", 0)
                    aggregated["toolagent_hallucination"]["hallucinated_tool_count"] += th.get("hallucinated_tool_count", 0)
                    aggregated["toolagent_hallucination"]["hallucinated_param_count"] += th.get("hallucinated_param_count", 0)
                    aggregated["toolagent_hallucination"]["resolved_count"] += th.get("resolved_count", 0)
                    aggregated["toolagent_hallucination"]["failed_count"] += th.get("failed_count", 0)
                
                # User修正
                if "user_correction" in stats:
                    uc = stats["user_correction"]
                    aggregated["user_correction"]["total_corrections"] += uc.get("total_corrections", 0)
                    aggregated["user_correction"]["all_iterations"].extend(uc.get("iterations_used", []))
                
                # 剪枝
                if "pruning" in stats:
                    pr = stats["pruning"]
                    aggregated["pruning"]["total_segments_detected"] += pr.get("total_segments_detected", 0)
                    aggregated["pruning"]["segments_pruned"] += pr.get("segments_pruned", 0)
                    aggregated["pruning"]["segments_kept"] += pr.get("segments_kept", 0)
            
            # 计算汇总指标
            summary = {
                "total_tasks": total_count,
                "successful_tasks": success_count,
                "success_rate": round(success_count / total_count * 100, 2) if total_count > 0 else 0,
                "avg_hallucinations_per_task": round(
                    aggregated["toolagent_hallucination"]["total_hallucination_count"] / success_count, 2
                ) if success_count > 0 else 0,
                "hallucination_success_rate": round(
                    aggregated["toolagent_hallucination"]["resolved_count"] / aggregated["toolagent_hallucination"]["total_hallucination_count"] * 100, 2
                ) if aggregated["toolagent_hallucination"]["total_hallucination_count"] > 0 else 0,
                "avg_user_corrections_per_task": round(
                    aggregated["user_correction"]["total_corrections"] / success_count, 2
                ) if success_count > 0 else 0,
                "avg_correction_iterations": round(
                    sum(aggregated["user_correction"]["all_iterations"]) / len(aggregated["user_correction"]["all_iterations"]), 2
                ) if aggregated["user_correction"]["all_iterations"] else 0,
            }
            
            # 保存统计文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = self.output_dir / f"aggregated_stats_{timestamp}.json"
            
            aggregated_data = {
                "timestamp": datetime.now().isoformat(),
                "domain": self.domain,
                "task_summary": {
                    "total_tasks": total_count,
                    "successful_tasks": success_count,
                    "failed_tasks": total_count - success_count
                },
                "aggregated_statistics": aggregated,
                "summary_metrics": summary
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Aggregated statistics saved to: {stats_file}")
            
        except Exception as e:
            logger.error(f"Failed to save aggregated stats: {e}")
    
    @staticmethod
    def merge_all_domains(output_dir: str, domains: List[str]) -> dict:
        """
        合并所有domain的数据到一个文件
        
        Args:
            output_dir: 输出目录
            domains: 领域列表
            
        Returns:
            合并统计信息
        """
        from datetime import datetime
        
        output_path = Path(output_dir)
        merged_file = output_path / "merged_data.jsonl"
        
        total_count = 0
        domain_counts = {}
        
        log(f"\n{'='*60}")
        log(f"合并所有domain的数据")
        log(f"{'='*60}\n")
        
        with open(merged_file, 'w', encoding='utf-8') as out_f:
            for domain in domains:
                domain_file = output_path / domain / "synthetic_data.jsonl"
                
                if not domain_file.exists():
                    log(f"[警告] {domain} 数据文件不存在: {domain_file}")
                    domain_counts[domain] = 0
                    continue
                
                count = 0
                with open(domain_file, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + '\n')
                            count += 1
                
                domain_counts[domain] = count
                total_count += count
                log(f"✓ {domain:10s}: {count:5d} 条")
        
        log(f"\n{'='*60}")
        log(f"合并完成")
        log(f"{'='*60}")
        log(f"总计: {total_count} 条数据")
        log(f"输出文件: {merged_file}")
        log(f"文件大小: {merged_file.stat().st_size / 1024 / 1024:.1f} MB")
        log(f"{'='*60}\n")
        
        logger.info(f"Merged {total_count} samples from {len(domains)} domains to {merged_file}")
        
        # 保存统计信息
        stats = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_path),
            "merged_file": str(merged_file),
            "total_samples": total_count,
            "domain_counts": domain_counts,
            "domains": domains
        }
        
        stats_file = output_path / "merge_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        log(f"统计信息已保存到: {stats_file}\n")
        logger.info(f"Merge statistics saved to {stats_file}")

        return stats

    @staticmethod
    def convert_to_sft_format(output_dir: str, domains: List[str]) -> dict:
        """
        转换synthetic数据为SFT格式（只保留messages和tools字段）

        Args:
            output_dir: 输出目录
            domains: 领域列表

        Returns:
            转换统计信息
        """
        from datetime import datetime

        output_path = Path(output_dir)
        sft_file = output_path / "sft_data_all.jsonl"

        total_converted = 0
        total_skipped = 0
        domain_stats = {}

        log(f"\n{'='*60}")
        log(f"转换为SFT格式（只保留messages和tools）")
        log(f"{'='*60}\n")

        with open(sft_file, 'w', encoding='utf-8') as out_f:
            for domain in domains:
                domain_file = output_path / domain / "synthetic_data.jsonl"

                if not domain_file.exists():
                    log(f"[警告] {domain} 数据文件不存在: {domain_file}")
                    domain_stats[domain] = {"converted": 0, "skipped": 0}
                    continue

                converted = 0
                skipped = 0

                with open(domain_file, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        try:
                            data = json.loads(line)

                            # 检查action_validation，如果invalid则跳过
                            action_validation = data.get('action_validation', {})
                            if not action_validation.get('valid', True):
                                skipped += 1
                                continue

                            # 只保留messages和tools字段
                            sft_data = {
                                'messages': data['messages'],
                                'tools': data['tools']
                            }

                            out_f.write(json.dumps(sft_data, ensure_ascii=False) + '\n')
                            converted += 1

                        except Exception as e:
                            log(f"[错误] {domain} 第{line_num}行转换失败: {e}")
                            skipped += 1

                domain_stats[domain] = {"converted": converted, "skipped": skipped}
                total_converted += converted
                total_skipped += skipped
                log(f"✓ {domain:10s}: {converted:5d} 条转换, {skipped:5d} 条跳过")

        log(f"\n{'='*60}")
        log(f"转换完成")
        log(f"{'='*60}")
        log(f"成功: {total_converted} 条")
        log(f"跳过: {total_skipped} 条")
        log(f"输出文件: {sft_file}")
        log(f"文件大小: {sft_file.stat().st_size / 1024 / 1024:.1f} MB")
        log(f"{'='*60}\n")

        logger.info(f"Converted {total_converted} samples to SFT format, skipped {total_skipped}")

        # 保存统计信息
        stats = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_path),
            "sft_file": str(sft_file),
            "total_converted": total_converted,
            "total_skipped": total_skipped,
            "domain_stats": domain_stats,
            "domains": domains
        }

        stats_file = output_path / "sft_conversion_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        log(f"转换统计信息已保存到: {stats_file}\n")
        logger.info(f"SFT conversion statistics saved to {stats_file}")

        return stats


# CLI接口
async def main():
    """命令行接口"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Runner")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--domain", type=str, required=True, choices=["airline", "retail", "telecom"], help="领域")
    parser.add_argument("--num-samples", type=int, help="生成样本数（覆盖配置文件）")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取参数
    num_samples = args.num_samples or config['num_samples_per_domain'].get(args.domain, 100)
    api_pool = config['llm']['api_pool']
    output_dir = config['output']['dir']
    
    playground_config = config.get('playground', {})
    concurrency_config = config.get('concurrency', {})
    
    # 创建runner
    runner = SyntheticRunner(
        domain=args.domain,
        num_samples=num_samples,
        api_pool=api_pool,
        output_dir=output_dir,
        max_turn=playground_config.get('max_turn', 30),
        quality_threshold=playground_config.get('quality_threshold', 4),
        max_concurrent_per_api=concurrency_config.get('max_concurrent_per_api', 100),
        self_reflection_iterations=playground_config.get('self_reflection_iterations', 1),
        enable_pruning=playground_config.get('enable_pruning', False),
        num_voting_agents=playground_config.get('num_voting_agents', 3),
    )
    
    # 运行
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
