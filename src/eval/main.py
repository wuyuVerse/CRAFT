#!/usr/bin/env python3
"""评测主入口"""
import argparse
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.config import EvalConfig
from src.eval.runner import EvalRunner


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description='tau2-bench 并行评测工具')
    parser.add_argument(
        'config',
        type=str,
        help='评测配置文件路径 (YAML格式)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=None,
        help='覆盖配置中的num_runs参数（每个榜测几次）'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=None,
        help='覆盖配置中的domains参数'
    )
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=None,
        help='覆盖配置中的num_tasks参数（每个领域测试多少任务）'
    )
    parser.add_argument(
        '--max-concurrency',
        type=int,
        default=None,
        help='覆盖配置中的max_concurrency参数'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # 加载配置
    config = EvalConfig.from_yaml(args.config)

    # 覆盖参数
    if args.num_runs is not None:
        config.num_runs = args.num_runs
    if args.domains is not None:
        config.domains = args.domains
    if args.num_tasks is not None:
        config.num_tasks = args.num_tasks
    if args.max_concurrency is not None:
        config.max_concurrency = args.max_concurrency

    # 运行评测（EvalRunner 支持 native、synthetic 或两者同时）
    runner = EvalRunner(config)

    summary = runner.run()

    # 返回状态码
    overall = summary.get("overall", {})
    # 多源模式时检查 sources
    if "sources" in summary:
        # 任意一个 source 有成功即可
        any_success = any(
            s.get("overall", {}).get("total_tasks", 0) > 0
            for s in summary["sources"].values()
        )
        return 0 if any_success else 1

    total_tasks = overall.get("total_tasks", 0)
    if total_tasks and total_tasks > 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
