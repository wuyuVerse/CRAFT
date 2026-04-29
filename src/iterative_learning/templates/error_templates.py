"""
错误消息模板

集中管理所有错误类型的模板。
"""

# 参数错误模板
ERROR_MESSAGE_TEMPLATES = {
    "parameter_error": [
        "Invalid {param}: '{wrong_value}' is not a valid format",
        "Parameter '{param}' validation failed: expected valid identifier, got '{wrong_value}'",
        "Error processing {param}: '{wrong_value}' does not match expected pattern",
    ],
}

# 各领域的业务逻辑错误模板
BUSINESS_LOGIC_ERRORS = {
    "airline": {
        "book_reservation": [
            "Payment amount does not add up, total price is {total}, but paid {paid}",
            "Not enough seats on flight {flight_number}",
            "Gift card balance is not enough",
            "Not enough balance in payment method {payment_id}",
        ],
        "update_reservation_flights": [
            "Flight {flight_number} not available on date {date}",
            "Certificate cannot be used to update reservation",
            "Gift card balance is not enough",
        ],
        "update_reservation_passengers": [
            "Number of passengers does not match",
        ],
    },
    "retail": {
        "return_delivered_order_items": [
            "Non-delivered order cannot be returned",
            "Payment method should be the original payment method",
        ],
        "exchange_delivered_order_items": [
            "Non-delivered order cannot be exchanged",
            "Insufficient gift card balance to pay for the price difference",
            "The number of items to be exchanged should match",
        ],
        "modify_pending_order_items": [
            "Non-pending order cannot be modified",
            "The number of items to be exchanged should match",
            "The new item id should be different from the old item id",
        ],
        "cancel_pending_order": [
            "Non-pending order cannot be cancelled",
            "Invalid reason",
        ],
    },
    "telecom": {
        "resume_line": [
            "Line must be suspended to resume",
        ],
        "send_payment_request": [
            "A bill is already awaiting payment for this customer",
        ],
    }
}

# 各领域的工具幻觉映射（真实工具 -> 常见幻觉工具）
TOOL_HALLUCINATIONS = {
    "airline": {
        "get_flight_status": ["check_flight_status", "flight_status", "get_flight_info"],
        "get_user_details": ["get_user_info", "get_customer_details", "user_details"],
        "get_reservation_details": ["get_booking_details", "reservation_info", "get_reservation"],
        "book_reservation": ["create_reservation", "make_booking", "book_flight"],
        "cancel_reservation": ["cancel_booking", "delete_reservation"],
    },
    "retail": {
        "find_user_id_by_email": ["get_user_by_email", "search_user_email", "lookup_user"],
        "find_user_id_by_name_zip": ["get_user_by_name", "search_user", "find_customer"],
        "get_order_details": ["get_order_info", "order_details", "get_order"],
        "get_product_details": ["get_product_info", "product_details", "get_item"],
        "return_delivered_order_items": ["return_order", "process_return", "return_items"],
        "cancel_pending_order": ["cancel_order", "delete_order"],
    },
    "telecom": {
        "get_customer_by_phone": ["find_customer", "search_customer", "get_customer"],
        "get_bills_for_customer": ["get_customer_bills", "get_billing_info", "list_bills"],
        "get_data_usage": ["check_data_usage", "data_usage", "get_usage"],
        "resume_line": ["activate_line", "enable_line", "start_line"],
    }
}

# 状态错误模板
STATE_ERROR_TEMPLATES = {
    "airline": [
        "Reservation cannot be modified in current state",
        "Flight has already departed, cannot make changes",
        "Booking is already cancelled",
    ],
    "retail": [
        "Non-pending order cannot be modified",
        "Non-delivered order cannot be returned",
        "Order has already been processed",
    ],
    "telecom": [
        "Line must be suspended to resume",
        "Account is not in active state",
        "Service cannot be modified while suspended",
    ],
}

# 状态错误关键词（用于从错误库中识别状态错误）
STATE_ERROR_KEYWORDS = [
    "cannot be", "must be", "not allowed", "invalid state",
    "suspended", "pending", "closed", "cancelled", "delivered",
    "non-pending", "non-delivered", "already"
]
