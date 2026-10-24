"""升堂 · 编排器

把六部制协议变成可执行的代码：中堂拆单、校验兜底、事件外发。
"""

from .llm import FastPathUnavailable, LLM, LLMConfig
from .orchestrator import Attempt, OrderRejected, Zhongtang
from .schema import (
    AUDITOR,
    DEPARTMENTS,
    Order,
    ParseError,
    Task,
    describe,
    parse_order,
    validate,
)

__all__ = [
    "AUDITOR",
    "DEPARTMENTS",
    "Attempt",
    "FastPathUnavailable",
    "LLM",
    "LLMConfig",
    "Order",
    "OrderRejected",
    "ParseError",
    "Task",
    "Zhongtang",
    "describe",
    "parse_order",
    "validate",
]

__version__ = "0.1.0"
