"""
错误库管理器

从评测结果中提取真实错误，持久化存储，用于错误注入。
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class ErrorDatabase:
    """错误库管理器"""
    
    def __init__(self, db_path: str = "eval_results/error_database.json"):
        self.db_path = Path(db_path)
        self.errors = self._load_database()
    
    def _load_database(self) -> dict:
        """加载错误库"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded error database from {self.db_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load error database: {e}")
                return {}
        return {}
    
    def save(self):
        """保存错误库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.errors, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved error database to {self.db_path}")
    
    def add_error(
        self, 
        domain: str, 
        tool: str, 
        error_msg: str,
        example: Optional[str] = None
    ):
        """
        添加新错误
        
        Args:
            domain: 领域名称
            tool: 工具名称
            error_msg: 错误消息模板
            example: 具体错误示例
        """
        if domain not in self.errors:
            self.errors[domain] = {}
        if tool not in self.errors[domain]:
            self.errors[domain][tool] = []
        
        # 查找是否已存在相同的错误模板
        for err in self.errors[domain][tool]:
            if err['error'] == error_msg:
                err['count'] += 1
                if example and example not in err['examples']:
                    err['examples'].append(example)
                return
        
        # 添加新错误
        self.errors[domain][tool].append({
            'error': error_msg,
            'count': 1,
            'examples': [example] if example else []
        })
    
    def get_errors(self, domain: str, tool: str) -> List[dict]:
        """
        获取工具的错误列表
        
        Returns:
            错误列表，每个错误包含 error, count, examples
        """
        return self.errors.get(domain, {}).get(tool, [])
    
    def get_all_tools(self, domain: str) -> List[str]:
        """获取领域中所有有错误的工具"""
        return list(self.errors.get(domain, {}).keys())
    
    def update_from_eval_results(self, eval_result_path: str):
        """
        从评测结果更新错误库
        
        Args:
            eval_result_path: 评测结果文件路径
        """
        try:
            with open(eval_result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load eval results from {eval_result_path}: {e}")
            return
        
        # 从文件名提取领域
        domain = self._extract_domain_from_path(eval_result_path)
        if not domain:
            logger.warning(f"Could not extract domain from {eval_result_path}")
            return
        
        error_count = 0
        
        # 遍历所有 simulations
        for sim in data.get('simulations', []):
            for msg in sim.get('messages', []):
                if msg.get('role') == 'tool':
                    content = msg.get('content', '')
                    try:
                        # 尝试解析为 JSON
                        if content.startswith('{'):
                            tool_data = json.loads(content)
                            result = tool_data.get('result', '')
                            tool_name = tool_data.get('name', 'unknown')
                        else:
                            # 直接是字符串
                            result = content
                            tool_name = 'unknown'
                        
                        # 检查是否是错误
                        if isinstance(result, str) and 'Error:' in result:
                            error_msg = result.replace('Error: ', '').strip()
                            
                            # 提取错误模板（将具体值替换为占位符）
                            error_template = self._extract_error_template(error_msg)
                            
                            # 添加到错误库
                            self.add_error(domain, tool_name, error_template, error_msg)
                            error_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to parse tool message: {e}")
                        continue
        
        logger.info(
            f"Updated error database from {eval_result_path}: "
            f"domain={domain}, errors={error_count}"
        )
    
    def _extract_domain_from_path(self, path: str) -> Optional[str]:
        """从文件路径提取领域名称"""
        path_lower = path.lower()
        if 'airline' in path_lower:
            return 'airline'
        elif 'retail' in path_lower:
            return 'retail'
        elif 'telecom' in path_lower:
            return 'telecom'
        return None
    
    def _extract_error_template(self, error_msg: str) -> str:
        """
        从具体错误消息提取模板
        
        例如：
        "User aarav_ahmed_6699 not found" -> "User {user_id} not found"
        "Reservation ZFA04Y not found" -> "Reservation {reservation_id} not found"
        """
        # 简单策略：保持原样，后续可以添加更智能的模板提取
        # 这里可以使用正则表达式识别常见模式
        
        import re
        
        # 替换常见的 ID 模式
        template = error_msg
        
        # 用户 ID: firstname_lastname_数字
        template = re.sub(r'\b[a-z]+_[a-z]+_\d+\b', '{user_id}', template)
        
        # 订单 ID: #W数字
        template = re.sub(r'#W\d+', '{order_id}', template)
        
        # 预订 ID: 6位大写字母
        template = re.sub(r'\b[A-Z]{6}\b', '{reservation_id}', template)
        
        # 航班号: HAT数字
        template = re.sub(r'HAT\d+', '{flight_number}', template)
        
        # 电话号码: P数字
        template = re.sub(r'P\d+', '{phone_number}', template)
        
        # 客户 ID: C数字
        template = re.sub(r'C\d+', '{customer_id}', template)
        
        # 产品 ID: 10位数字
        template = re.sub(r'\b\d{10}\b', '{product_id}', template)
        
        # 日期: YYYY-MM-DD
        template = re.sub(r'\d{4}-\d{2}-\d{2}', '{date}', template)
        
        # 数字金额
        template = re.sub(r'\b\d+\b', '{amount}', template)
        
        return template
    
    def get_statistics(self) -> dict:
        """获取错误库统计信息"""
        stats = {
            'total_domains': len(self.errors),
            'domains': {}
        }
        
        for domain, tools in self.errors.items():
            domain_stats = {
                'total_tools': len(tools),
                'total_errors': sum(len(errors) for errors in tools.values()),
                'total_occurrences': sum(
                    err['count'] 
                    for errors in tools.values() 
                    for err in errors
                ),
                'tools': {}
            }
            
            for tool, errors in tools.items():
                domain_stats['tools'][tool] = {
                    'error_count': len(errors),
                    'total_occurrences': sum(err['count'] for err in errors)
                }
            
            stats['domains'][domain] = domain_stats
        
        return stats
    
    def print_statistics(self):
        """打印错误库统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("ERROR DATABASE STATISTICS")
        print("="*60)
        print(f"Total domains: {stats['total_domains']}")
        
        for domain, domain_stats in stats['domains'].items():
            print(f"\n{domain.upper()} Domain:")
            print(f"  Total tools with errors: {domain_stats['total_tools']}")
            print(f"  Total unique errors: {domain_stats['total_errors']}")
            print(f"  Total error occurrences: {domain_stats['total_occurrences']}")
            
            print(f"\n  Top 10 tools by error count:")
            sorted_tools = sorted(
                domain_stats['tools'].items(),
                key=lambda x: x[1]['total_occurrences'],
                reverse=True
            )
            for i, (tool, tool_stats) in enumerate(sorted_tools[:10], 1):
                print(
                    f"    {i}. {tool}: "
                    f"{tool_stats['error_count']} unique errors, "
                    f"{tool_stats['total_occurrences']} occurrences"
                )
        
        print("\n" + "="*60)
