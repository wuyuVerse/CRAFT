"""
多样性增强策略模块
实现用户画像、任务复杂度分层、参数空间探索
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(Enum):
    """任务复杂度级别"""
    SIMPLE = "simple"      # 2-3个工具调用
    MEDIUM = "medium"      # 4-6个工具调用
    COMPLEX = "complex"    # 7+个工具调用


@dataclass
class UserPersona:
    """用户画像"""
    type: str
    communication_style: str
    tech_level: str
    urgency: str
    typical_requests: List[str]
    example_opening: str
    additional_traits: Dict[str, Any] = None


@dataclass
class TaskTemplate:
    """任务模板"""
    name: str
    complexity: ComplexityLevel
    required_tools: List[str]
    expected_turns: Tuple[int, int]
    tool_sequence: List[str]
    description: str


class UserPersonaLibrary:
    """用户画像库"""

    PERSONAS = [
        UserPersona(
            type="tech_savvy_professional",
            communication_style="terse and direct",
            tech_level="high",
            urgency="high",
            typical_requests=["quick status check", "efficient booking", "self-service preference"],
            example_opening="I need to book JFK to LAX tomorrow, economy, user ID alex_jones_123",
            additional_traits={"provides_all_info_upfront": True, "minimal_small_talk": True}
        ),
        UserPersona(
            type="elderly_traveler",
            communication_style="verbose and cautious",
            tech_level="low",
            urgency="low",
            typical_requests=["need detailed explanations", "multiple confirmations", "prefers step-by-step"],
            example_opening="Hello, I'm not very good with computers. I need some help booking a flight to visit my grandson...",
            additional_traits={"needs_reassurance": True, "prefers_detailed_responses": True}
        ),
        UserPersona(
            type="budget_conscious_student",
            communication_style="casual",
            tech_level="medium",
            urgency="medium",
            typical_requests=["cheapest options", "flexible dates", "baggage minimization"],
            example_opening="Hey, what's the cheapest way to get to Boston next month?",
            additional_traits={"price_sensitivity": "very high", "flexibility": "high"}
        ),
        UserPersona(
            type="business_traveler_with_status",
            communication_style="formal but efficient",
            tech_level="high",
            urgency="high",
            typical_requests=["multi-city bookings", "cabin upgrades", "last-minute changes"],
            example_opening="I need to modify my existing reservation R123XY to upgrade to business class.",
            additional_traits={"membership": "gold", "expects_priority": True}
        ),
        UserPersona(
            type="family_traveler",
            communication_style="detailed questions",
            tech_level="medium",
            urgency="medium",
            typical_requests=["group bookings", "special requests", "cancellation policies"],
            example_opening="Hi, I need to book flights for me, my wife, and our two children (ages 5 and 8)...",
            additional_traits={"passengers": "multiple", "concerns": ["child safety", "seat selection"]}
        ),
        UserPersona(
            type="first_time_user",
            communication_style="uncertain, many questions",
            tech_level="low",
            urgency="low",
            typical_requests=["how to book", "what information needed", "payment methods"],
            example_opening="I've never booked a flight online before. How do I get started?",
            additional_traits={"knowledge": "minimal", "needs_guidance": True}
        ),
        UserPersona(
            type="international_traveler",
            communication_style="formal with potential language nuances",
            tech_level="medium",
            urgency="medium",
            typical_requests=["multi-stop flights", "airline alliances", "baggage policies"],
            example_opening="I need to fly from Shanghai to New York, what are my options?",
            additional_traits={"concerns": ["visa", "international routing", "currency"]}
        ),
        UserPersona(
            type="last_minute_booker",
            communication_style="urgent and stressed",
            tech_level="medium",
            urgency="critical",
            typical_requests=["immediate availability", "any seat", "fastest route"],
            example_opening="URGENT: I need to fly out TONIGHT, any availability?",
            additional_traits={"flexibility": "low", "time_pressure": "extreme"}
        )
    ]

    @classmethod
    def get_random_persona(cls, weights: List[float] = None) -> UserPersona:
        """随机选择一个用户画像"""
        if weights is None:
            weights = [1/len(cls.PERSONAS)] * len(cls.PERSONAS)
        return random.choices(cls.PERSONAS, weights=weights)[0]


class TaskComplexityLibrary:
    """任务复杂度模板库"""

    TEMPLATES = {
        ComplexityLevel.SIMPLE: [
            TaskTemplate(
                name="query_reservation",
                complexity=ComplexityLevel.SIMPLE,
                required_tools=["get_reservation_details"],
                expected_turns=(3, 5),
                tool_sequence=["get_reservation_details"],
                description="查询预订详情"
            ),
            TaskTemplate(
                name="check_flight_status",
                complexity=ComplexityLevel.SIMPLE,
                required_tools=["get_reservation_details", "get_flight_status"],
                expected_turns=(4, 6),
                tool_sequence=["get_reservation_details", "get_flight_status"],
                description="检查航班状态"
            ),
        ],

        ComplexityLevel.MEDIUM: [
            TaskTemplate(
                name="book_round_trip",
                complexity=ComplexityLevel.MEDIUM,
                required_tools=["get_user_details", "search_direct_flight", "book_reservation"],
                expected_turns=(8, 12),
                tool_sequence=["get_user_details", "search_direct_flight", "search_direct_flight", "book_reservation"],
                description="预订往返航班"
            ),
            TaskTemplate(
                name="modify_and_add_baggage",
                complexity=ComplexityLevel.MEDIUM,
                required_tools=["get_user_details", "get_reservation_details", "search_direct_flight", "update_reservation_flights"],
                expected_turns=(10, 15),
                tool_sequence=["get_user_details", "get_reservation_details", "search_direct_flight", "update_reservation_flights"],
                description="修改预订并增加行李"
            ),
        ],

        ComplexityLevel.COMPLEX: [
            TaskTemplate(
                name="delay_rebooking_compensation",
                complexity=ComplexityLevel.COMPLEX,
                required_tools=["get_user_details", "get_reservation_details", "get_flight_status", "search_direct_flight", "update_reservation_flights", "send_certificate"],
                expected_turns=(15, 20),
                tool_sequence=["get_user_details", "get_reservation_details", "get_flight_status", "search_direct_flight", "update_reservation_flights", "send_certificate"],
                description="航班延误后改签并申请补偿"
            ),
        ]
    }

    @classmethod
    def get_random_template(cls, complexity: ComplexityLevel = None, weights: Dict[ComplexityLevel, float] = None) -> TaskTemplate:
        """随机选择任务模板"""
        if complexity:
            return random.choice(cls.TEMPLATES[complexity])

        if weights is None:
            weights = {ComplexityLevel.SIMPLE: 0.3, ComplexityLevel.MEDIUM: 0.5, ComplexityLevel.COMPLEX: 0.2}

        complexity_level = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        return random.choice(cls.TEMPLATES[complexity_level])


class ParameterSpaceExplorer:
    """参数空间探索器"""

    DATE_STRATEGIES = {
        "near_future": {"days_ahead": (1, 7), "weight": 0.2},
        "optimal_window": {"days_ahead": (14, 45), "weight": 0.5},
        "far_future": {"days_ahead": (90, 180), "weight": 0.3}
    }

    AIRLINE_ROUTES = {
        "popular_domestic": {
            "routes": [("JFK", "LAX"), ("LAX", "JFK"), ("ORD", "MIA"), ("MIA", "ORD"), ("SFO", "SEA"), ("SEA", "SFO")],
            "weight": 0.5
        },
        "regional": {
            "routes": [("BOS", "PHL"), ("PHL", "BOS"), ("DEN", "SLC"), ("SLC", "DEN")],
            "weight": 0.3
        },
        "less_common": {
            "routes": [("PDX", "BOS"), ("BOS", "PDX"), ("MSP", "PHX"), ("PHX", "MSP")],
            "weight": 0.2
        }
    }

    CABIN_SCENARIOS = {
        "budget": {"cabin": "basic_economy", "weight": 0.3},
        "standard": {"cabin": "economy", "weight": 0.5},
        "premium": {"cabin": "business", "weight": 0.2}
    }

    PASSENGER_COMBINATIONS = [
        {"count": 1, "weight": 0.60},
        {"count": 2, "weight": 0.20},
        {"count": 3, "weight": 0.10},
        {"count": 4, "weight": 0.07},
        {"count": 5, "weight": 0.03}
    ]

    @classmethod
    def generate_date(cls, strategy: str = None) -> str:
        """生成日期"""
        base_date = datetime.now()
        if strategy is None:
            strategy = random.choices(list(cls.DATE_STRATEGIES.keys()), weights=[s["weight"] for s in cls.DATE_STRATEGIES.values()])[0]

        days_range = cls.DATE_STRATEGIES[strategy]["days_ahead"]
        days_ahead = random.randint(days_range[0], days_range[1])
        target_date = base_date + timedelta(days=days_ahead)
        return target_date.strftime("%Y-%m-%d")

    @classmethod
    def generate_route(cls, category: str = None) -> Tuple[str, str]:
        """生成航线"""
        if category is None:
            category = random.choices(list(cls.AIRLINE_ROUTES.keys()), weights=[r["weight"] for r in cls.AIRLINE_ROUTES.values()])[0]
        return random.choice(cls.AIRLINE_ROUTES[category]["routes"])

    @classmethod
    def generate_cabin(cls, user_persona: UserPersona = None) -> str:
        """生成舱位"""
        if user_persona:
            if user_persona.type == "budget_conscious_student":
                return "basic_economy"
            elif user_persona.type == "business_traveler_with_status":
                return random.choice(["business", "economy"])

        cabin_choice = random.choices(list(cls.CABIN_SCENARIOS.keys()), weights=[c["weight"] for c in cls.CABIN_SCENARIOS.values()])[0]
        return cls.CABIN_SCENARIOS[cabin_choice]["cabin"]

    @classmethod
    def generate_passenger_count(cls, user_persona: UserPersona = None) -> int:
        """生成乘客数量"""
        if user_persona and user_persona.type == "family_traveler":
            return random.choices([3, 4, 5], weights=[0.4, 0.4, 0.2])[0]

        combo = random.choices(cls.PASSENGER_COMBINATIONS, weights=[c["weight"] for c in cls.PASSENGER_COMBINATIONS])[0]
        return combo["count"]


class DiversityEnhancer:
    """多样性增强器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.complexity_weights = self.config.get("complexity_weights", {
            ComplexityLevel.SIMPLE: 0.3,
            ComplexityLevel.MEDIUM: 0.5,
            ComplexityLevel.COMPLEX: 0.2
        })

    def generate_diverse_task_config(self, domain: str) -> Dict[str, Any]:
        """生成多样化任务配置"""
        persona = UserPersonaLibrary.get_random_persona()
        task_template = TaskComplexityLibrary.get_random_template(weights=self.complexity_weights)

        # 生成参数
        date_outbound = ParameterSpaceExplorer.generate_date()
        origin, destination = ParameterSpaceExplorer.generate_route()
        cabin = ParameterSpaceExplorer.generate_cabin(persona)
        passenger_count = ParameterSpaceExplorer.generate_passenger_count(persona)

        config = {
            "user_persona": {
                "type": persona.type,
                "communication_style": persona.communication_style,
                "tech_level": persona.tech_level,
                "urgency": persona.urgency,
                "example_opening": persona.example_opening
            },
            "task_template": {
                "name": task_template.name,
                "complexity": task_template.complexity.value,
                "expected_turns": task_template.expected_turns,
                "tool_sequence": task_template.tool_sequence
            },
            "parameters": {
                "domain": domain,
                "date_outbound": date_outbound,
                "origin": origin,
                "destination": destination,
                "cabin": cabin,
                "passenger_count": passenger_count
            }
        }

        return config

    def generate_batch_configs(self, domain: str, count: int) -> List[Dict[str, Any]]:
        """批量生成配置"""
        return [self.generate_diverse_task_config(domain) for _ in range(count)]
