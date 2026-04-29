"""
分析tau2-bench数据库的参数分布

从db.json/db.toml中提取真实的参数分布，用于生成任务时的参数采样。
"""
import json
import os
from collections import Counter
from typing import Dict, Any, Optional


class Tau2ParameterAnalyzer:
    """分析tau2数据库的参数分布"""

    def __init__(self, tau2_data_dir: str = None):
        self.tau2_data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR", "tau2-bench/data/tau2")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def analyze_parameter_space(self, domain: str) -> Dict[str, Any]:
        """
        分析领域参数空间

        Args:
            domain: 领域名称 (airline/retail/telecom)

        Returns:
            参数分布字典
        """
        if domain in self._cache:
            return self._cache[domain]

        db = self._load_database(domain)

        if domain == "airline":
            result = self._analyze_airline(db)
        elif domain == "retail":
            result = self._analyze_retail(db)
        elif domain == "telecom":
            result = self._analyze_telecom(db)
        else:
            result = {}

        self._cache[domain] = result
        return result

    def _load_database(self, domain: str) -> Dict:
        """
        加载db.json或db.toml

        Args:
            domain: 领域名称

        Returns:
            数据库内容
        """
        db_json = os.path.join(self.tau2_data_dir, "domains", domain, "db.json")
        db_toml = os.path.join(self.tau2_data_dir, "domains", domain, "db.toml")

        if os.path.exists(db_json):
            with open(db_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif os.path.exists(db_toml):
            try:
                import tomli
                with open(db_toml, 'rb') as f:
                    return tomli.load(f)
            except ImportError:
                print("Warning: tomli not installed, cannot load .toml files")
                print("Install with: pip install tomli")
                return {}
        else:
            print(f"Warning: No database file found for {domain}")
            return {}

    def _analyze_airline(self, db: Dict) -> Dict:
        """
        分析airline参数

        Args:
            db: 数据库内容

        Returns:
            参数分布
        """
        routes = []
        cabins = []
        passenger_counts = []

        # 从reservations提取 (注意: reservations可能是list)
        reservations = db.get("reservations", [])
        if isinstance(reservations, dict):
            reservations = list(reservations.values())

        for res in reservations:
            if not isinstance(res, dict):
                continue

            # 航线
            for flight in res.get("flights", []):
                if not isinstance(flight, dict):
                    continue
                origin = flight.get("origin")
                dest = flight.get("destination")
                if origin and dest:
                    routes.append((origin, dest))

            # 舱位
            cabin = res.get("cabin")
            if cabin:
                cabins.append(cabin)

            # 乘客数量
            passengers = res.get("passengers", [])
            if passengers:
                passenger_counts.append(len(passengers))

        # 提取用户信息
        users = db.get("users", {})
        if isinstance(users, list):
            users = {str(i): u for i, u in enumerate(users)}

        last_names = []
        first_names = []

        for user in users.values():
            if isinstance(user, dict):
                last_name = user.get("last_name")
                first_name = user.get("first_name")
                if last_name:
                    last_names.append(last_name)
                if first_name:
                    first_names.append(first_name)

        # 提取预订编号格式（用于生成）
        confirmation_numbers = [res.get("reservation_id") for res in reservations if isinstance(res, dict) and res.get("reservation_id")]

        return {
            "routes": Counter(routes),
            "cabins": Counter(cabins),
            "passenger_counts": Counter(passenger_counts),
            "last_names": Counter(last_names),
            "first_names": Counter(first_names),
            "confirmation_numbers": confirmation_numbers[:10],  # 保留一些示例
            "total_users": len(users),
            "total_reservations": len(reservations)
        }

    def _analyze_retail(self, db: Dict) -> Dict:
        """
        分析retail参数

        Args:
            db: 数据库内容

        Returns:
            参数分布
        """
        customers = db.get("customers", {})
        cities = []
        zip_codes = []
        names = []

        for customer in customers.values():
            if isinstance(customer, dict):
                city = customer.get("city")
                zip_code = customer.get("zip_code")
                name = customer.get("name")

                if city:
                    cities.append(city)
                if zip_code:
                    zip_codes.append(zip_code)
                if name:
                    names.append(name)

        # 订单信息
        orders = db.get("orders", {})
        statuses = []
        order_ids = []

        for order_id, order in orders.items():
            if isinstance(order, dict):
                status = order.get("status")
                if status:
                    statuses.append(status)
                order_ids.append(order_id)

        # 产品信息
        products = db.get("products", {})
        categories = []

        for product in products.values():
            if isinstance(product, dict):
                category = product.get("category")
                if category:
                    categories.append(category)

        return {
            "cities": Counter(cities),
            "zip_codes": Counter(zip_codes),
            "names": Counter(names),
            "order_statuses": Counter(statuses),
            "order_ids": order_ids[:10],  # 保留示例
            "product_categories": Counter(categories),
            "total_customers": len(customers),
            "total_orders": len(orders),
            "total_products": len(products)
        }

    def _analyze_telecom(self, db: Dict) -> Dict:
        """
        分析telecom参数

        Args:
            db: 数据库内容

        Returns:
            参数分布
        """
        # Plans
        plans = db.get("plans", [])
        plan_names = []
        plan_data_limits = []

        for plan in plans:
            if isinstance(plan, dict):
                name = plan.get("name")
                data_limit = plan.get("data_limit_gb")
                if name:
                    plan_names.append(name)
                if data_limit:
                    plan_data_limits.append(data_limit)

        # Devices
        devices = db.get("devices", [])
        device_names = []

        for device in devices:
            if isinstance(device, dict):
                name = device.get("name")
                if name:
                    device_names.append(name)

        # Customers
        customers = db.get("customers", [])
        customer_names = []

        for customer in customers:
            if isinstance(customer, dict):
                name = customer.get("name")
                if name:
                    customer_names.append(name)

        # Lines
        lines = db.get("lines", [])
        line_statuses = []
        phone_numbers = []

        for line in lines:
            if isinstance(line, dict):
                status = line.get("status")
                phone = line.get("phone_number")
                if status:
                    line_statuses.append(status)
                if phone:
                    phone_numbers.append(phone)

        return {
            "plan_names": Counter(plan_names),
            "plan_data_limits": Counter(plan_data_limits),
            "device_names": Counter(device_names),
            "customer_names": Counter(customer_names),
            "line_statuses": Counter(line_statuses),
            "phone_numbers": phone_numbers[:10],  # 保留示例
            "total_plans": len(plans),
            "total_devices": len(devices),
            "total_customers": len(customers),
            "total_lines": len(lines)
        }

    def print_summary(self, domain: str) -> None:
        """
        打印参数分布摘要

        Args:
            domain: 领域名称
        """
        params = self.analyze_parameter_space(domain)

        print(f"\n{'='*60}")
        print(f"参数分布分析 - {domain.upper()}")
        print(f"{'='*60}")

        if domain == "airline":
            print(f"\n航线数量: {len(params['routes'])}")
            print(f"舱位分布: {dict(params['cabins'])}")
            print(f"乘客数量分布: {dict(params['passenger_counts'])}")
            print(f"用户数量: {params['total_users']}")
            print(f"预订数量: {params['total_reservations']}")

            print(f"\n最常见航线 (前5):")
            for route, count in params['routes'].most_common(5):
                print(f"  {route[0]} -> {route[1]}: {count}")

        elif domain == "retail":
            print(f"\n城市数量: {len(params['cities'])}")
            print(f"订单状态分布: {dict(params['order_statuses'])}")
            print(f"产品类别数量: {len(params['product_categories'])}")
            print(f"客户数量: {params['total_customers']}")
            print(f"订单数量: {params['total_orders']}")

            print(f"\n最常见城市 (前5):")
            for city, count in params['cities'].most_common(5):
                print(f"  {city}: {count}")

        elif domain == "telecom":
            print(f"\n套餐数量: {params['total_plans']}")
            print(f"设备数量: {params['total_devices']}")
            print(f"客户数量: {params['total_customers']}")
            print(f"线路数量: {params['total_lines']}")

            print(f"\n套餐分布:")
            for plan, count in params['plan_names'].most_common():
                print(f"  {plan}: {count}")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    # 测试代码
    analyzer = Tau2ParameterAnalyzer()

    for domain in ["airline", "retail", "telecom"]:
        try:
            analyzer.print_summary(domain)
        except Exception as e:
            print(f"Error analyzing {domain}: {e}")
            import traceback
            traceback.print_exc()
