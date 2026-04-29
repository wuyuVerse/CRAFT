"""
ParameterEnricher

从tau2-bench数据库采样真实参数，确保生成的任务与数据库一致。
"""

import json
import os
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import tomli
    HAS_TOMLI = True
except ImportError:
    try:
        import tomllib as tomli  # Python 3.11+
        HAS_TOMLI = True
    except ImportError:
        HAS_TOMLI = False


@dataclass
class EnrichedParams:
    """丰富的参数集合"""
    # 通用
    user_id: str = ""
    first_name: str = ""
    last_name: str = ""
    email: str = ""

    # Airline
    confirmation: str = ""
    origin: str = ""
    destination: str = ""
    flight_date: str = ""
    cabin: str = ""
    flight_number: str = ""

    # Retail
    order_id: str = ""
    product_name: str = ""
    product_id: str = ""

    # Telecom
    phone_number: str = ""
    plan_name: str = ""
    plan_id: str = ""
    device_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，过滤空值"""
        return {k: v for k, v in self.__dict__.items() if v}


class ParameterEnricher:
    """
    参数丰富器

    从tau2-bench的db.json/db.toml中采样真实参数，
    确保生成的任务可以在tau2-bench中正确执行。
    """

    def __init__(self, tau2_data_dir: str = None):
        self.tau2_data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR", "tau2-bench/data/tau2")
        self._db_cache: Dict[str, Dict] = {}

    def _load_db(self, domain: str) -> Dict[str, Any]:
        """加载数据库"""
        if domain in self._db_cache:
            return self._db_cache[domain]

        db_json_path = os.path.join(self.tau2_data_dir, "domains", domain, "db.json")
        db_toml_path = os.path.join(self.tau2_data_dir, "domains", domain, "db.toml")

        db = {}

        # 优先加载 db.json
        if os.path.exists(db_json_path):
            with open(db_json_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
                logger.debug(f"Loaded db.json for {domain}")

        # 如果没有db.json或需要补充，加载db.toml
        if os.path.exists(db_toml_path):
            if HAS_TOMLI:
                with open(db_toml_path, 'rb') as f:
                    toml_db = tomli.load(f)
                    # 合并toml数据
                    for key, value in toml_db.items():
                        if key not in db:
                            db[key] = value
                    logger.debug(f"Loaded db.toml for {domain}")
            else:
                logger.warning(f"tomli not installed, cannot load db.toml for {domain}")

        self._db_cache[domain] = db
        return db

    def enrich_params(self, domain: str, pattern: Optional[Any] = None) -> EnrichedParams:
        """
        从数据库采样真实参数

        Args:
            domain: 领域名称
            pattern: 可选的任务模式，用于更精准的参数采样

        Returns:
            EnrichedParams对象
        """
        if domain == "airline":
            return self._enrich_airline_params(pattern)
        elif domain == "retail":
            return self._enrich_retail_params(pattern)
        elif domain == "telecom":
            return self._enrich_telecom_params(pattern)
        else:
            logger.warning(f"Unknown domain: {domain}")
            return EnrichedParams()

    def _enrich_airline_params(self, pattern: Optional[Any] = None) -> EnrichedParams:
        """采样航空领域参数"""
        db = self._load_db("airline")
        params = EnrichedParams()

        # 采样用户
        users = db.get("users", {})
        if users:
            user_id = random.choice(list(users.keys()))
            user = users[user_id]
            params.user_id = user_id
            params.first_name = user.get("name", {}).get("first_name", "")
            params.last_name = user.get("name", {}).get("last_name", "")
            params.email = user.get("email", "")

        # 采样预订
        reservations = db.get("reservations", {})
        if reservations:
            res_id = random.choice(list(reservations.keys()))
            res = reservations[res_id]
            params.confirmation = res_id
            params.origin = res.get("origin", "")
            params.destination = res.get("destination", "")
            params.cabin = res.get("cabin", "economy")

            # 从航班中获取日期
            flights = res.get("flights", [])
            if flights:
                params.flight_date = flights[0].get("date", "")
                params.flight_number = flights[0].get("flight_number", "")

        return params

    def _enrich_retail_params(self, pattern: Optional[Any] = None) -> EnrichedParams:
        """采样零售领域参数"""
        db = self._load_db("retail")
        params = EnrichedParams()

        # 采样用户
        users = db.get("users", {})
        if users:
            user_id = random.choice(list(users.keys()))
            user = users[user_id]
            params.user_id = user_id
            params.first_name = user.get("name", {}).get("first_name", "")
            params.last_name = user.get("name", {}).get("last_name", "")
            params.email = user.get("email", "")

            # 从用户的订单中采样
            user_orders = user.get("orders", [])
            if user_orders:
                params.order_id = random.choice(user_orders)

        # 如果用户没有订单，从全局订单中采样
        if not params.order_id:
            orders = db.get("orders", {})
            if orders:
                order_id = random.choice(list(orders.keys()))
                params.order_id = order_id
                order = orders[order_id]
                # 获取订单中的商品
                items = order.get("items", [])
                if items:
                    item = random.choice(items)
                    params.product_name = item.get("name", "")
                    params.product_id = item.get("product_id", "")

        # 采样产品
        products = db.get("products", {})
        if products and not params.product_name:
            product_id = random.choice(list(products.keys()))
            product = products[product_id]
            params.product_id = product_id
            params.product_name = product.get("name", "")

        return params

    def _enrich_telecom_params(self, pattern: Optional[Any] = None) -> EnrichedParams:
        """采样电信领域参数"""
        db = self._load_db("telecom")
        params = EnrichedParams()

        # 采样用户 (从user_db.toml)
        user_db_path = os.path.join(self.tau2_data_dir, "domains", "telecom", "user_db.toml")
        if os.path.exists(user_db_path) and HAS_TOMLI:
            with open(user_db_path, 'rb') as f:
                user_db = tomli.load(f)
                users = user_db.get("users", [])
                if users:
                    user = random.choice(users)
                    params.user_id = user.get("user_id", "")
                    params.first_name = user.get("first_name", "")
                    params.last_name = user.get("last_name", "")
                    params.email = user.get("email", "")
                    params.phone_number = user.get("phone_number", "")

        # 采样套餐
        plans = db.get("plans", [])
        if plans:
            plan = random.choice(plans)
            params.plan_id = plan.get("plan_id", "")
            params.plan_name = plan.get("name", "")

        # 采样设备
        devices = db.get("devices", [])
        if devices:
            device = random.choice(devices)
            params.device_id = device.get("device_id", "")

        return params

    def get_all_users(self, domain: str) -> List[Dict[str, Any]]:
        """获取所有用户"""
        db = self._load_db(domain)
        users = db.get("users", {})
        if isinstance(users, dict):
            return list(users.values())
        return users

    def get_all_reservations(self, domain: str) -> List[Dict[str, Any]]:
        """获取所有预订（仅airline）"""
        if domain != "airline":
            return []
        db = self._load_db(domain)
        reservations = db.get("reservations", {})
        if isinstance(reservations, dict):
            return list(reservations.values())
        return reservations

    def get_all_orders(self, domain: str) -> List[Dict[str, Any]]:
        """获取所有订单（仅retail）"""
        if domain != "retail":
            return []
        db = self._load_db(domain)
        orders = db.get("orders", {})
        if isinstance(orders, dict):
            return list(orders.values())
        return orders

    def get_statistics(self, domain: str) -> Dict[str, Any]:
        """获取数据库统计信息"""
        db = self._load_db(domain)

        stats = {
            "domain": domain,
            "tables": list(db.keys()),
        }

        for key, value in db.items():
            if isinstance(value, dict):
                stats[f"{key}_count"] = len(value)
            elif isinstance(value, list):
                stats[f"{key}_count"] = len(value)

        return stats


if __name__ == "__main__":
    # 测试代码
    enricher = ParameterEnricher()

    for domain in ["airline", "retail", "telecom"]:
        print(f"\n{'='*50}")
        print(f"Domain: {domain}")
        print(f"{'='*50}")

        # 统计信息
        stats = enricher.get_statistics(domain)
        print(f"Statistics: {stats}")

        # 采样参数
        for i in range(3):
            params = enricher.enrich_params(domain)
            print(f"\nSample {i+1}:")
            for key, value in params.to_dict().items():
                print(f"  {key}: {value}")
