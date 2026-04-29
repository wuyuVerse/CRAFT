import json
import random
from typing import Dict, List, Any


class DataSampler:
    def __init__(self, data: Dict[str, List[Dict[str, Any]]]):
        """
        初始化数据采样器
        
        Args:
            data: 符合指定格式的数据
        """
        self.data = data
        self.categories = list(data.keys())
    
    def sample(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        采样数据：
        1. 随机选择2到10个类型
        2. 在每个被选类型中采样2到15个api
        3. 确保至少有一个类型采样不少于5个api
        
        Returns:
            采样后的数据
        """
        # 1. 随机选择2到10个类型
        num_categories = random.randint(1, min(3, len(self.categories)))
        selected_categories = random.sample(self.categories, num_categories)
        
        # 2. 初始化结果字典
        sampled_data = {}
        
        # 3. 确定哪个类别需要采样不少于5个api
        special_category_index = random.randint(0, len(selected_categories) - 1)
        special_category = selected_categories[special_category_index]
        
        # 4. 对每个选中的类别进行采样
        for i, category in enumerate(selected_categories):
            # 获取该类别中的所有api
            apis = self.data.get(category, [])
            
            # 如果api列表为空，跳过该类别
            if not apis:
                sampled_data[category] = []
                continue
            
            # 确定采样数量
            if category == special_category:
                # 特殊类别采样5到15个api
                # 如果该类别API数量不足5个，则全部采样
                if len(apis) < 5:
                    sample_count = len(apis)
                else:
                    sample_count = random.randint(5, min(10, len(apis)))
            else:
                # 普通类别采样2到15个api
                # 如果该类别API数量不足2个，则全部采样
                if len(apis) < 2:
                    sample_count = len(apis)
                else:
                    sample_count = random.randint(2, min(5, len(apis)))
            
            # 进行采样
            sampled_apis = random.sample(apis, sample_count)
            sampled_data[category] = sampled_apis
        
        return sampled_data
    
    def sample_multiple(self, num_samples: int) -> List[Dict[str, List[Dict[str, Any]]]]:
        """
        生成多个采样结果
        
        Args:
            num_samples: 采样次数
            
        Returns:
            采样结果列表
        """
        samples = []
        for _ in range(num_samples):
            samples.append(self.sample())
        return samples
