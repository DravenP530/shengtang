"""
升堂 · 报文模型与校验

这一层的职责：把模型的自由文本变成**可信的升堂单**，不合格就拒绝。

设计原则：
  1. 宽松解析，严格校验。模型爱包代码块、爱加解释，解析层要忍；但校验层一步不让。
  2. 校验只报「哪里不对」，不替模型改。改是模型的事，改不好就退回。
  3. 校验规则和协议文档一一对应，不藏私货。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- 常量

#: 可接执行任务的五司。刑部不在此列——它只审不做。
DEPARTMENTS: tuple[str, ...] = ("吏部", "户部", "礼部", "兵部", "工部")

#: 只审不做的司，出现在任务表里即为违规。
AUDITOR = "刑部"

MAX_TASKS = 5
MIN_TASKS = 1
MAX_TASKS_PER_DEPT = 2
MIN_DISTINCT_DEPTS = 3

#: 交付物里的空话，出现即视为没有交付物。
VAGUE_WORDS = ("看看", "研究一下", "了解一下", "跟进一下", "处理一下", "优化一下")


# ---------------------------------------------------------------- 模型


@dataclass
class Task:
    dept: str
    action: str
    deliverable: str

    def to_dict(self) -> dict[str, str]:
        return {"司": self.dept, "做什么": self.action, "交付物": self.deliverable}


@dataclass
class Order:
    goal: str
    tasks: list[Task] = field(default_factory=list)
    #: 模型原始输出，出问题时留着排查
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"目标": self.goal, "任务": [t.to_dict() for t in self.tasks]}


class ParseError(ValueError):
    """模型输出连 JSON 都算不上。"""


# ---------------------------------------------------------------- 解析


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出第一个 JSON 对象。

    容忍三种常见污染：代码块包裹、前后解释文字、首尾多余空白。
    """
    if not text or not text.strip():
        raise ParseError("模型返回为空")

    candidates: list[str] = []

    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))

    candidates.append(text)

    # 退而求其次：从第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for chunk in candidates:
        try:
            data = json.loads(chunk.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    raise ParseError("模型输出里找不到合法 JSON 对象")


def parse_order(text: str) -> Order:
    """把模型输出解析成 Order。只做结构转换，不做业务校验。"""
    data = extract_json(text)

    goal = data.get("目标") or data.get("goal") or ""
    raw_tasks = data.get("任务") or data.get("tasks") or []

    if not isinstance(raw_tasks, list):
        raise ParseError("「任务」必须是数组")

    tasks: list[Task] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise ParseError("「任务」里的每一项必须是对象")
        tasks.append(
            Task(
                dept=str(item.get("司") or item.get("dept") or "").strip(),
                action=str(item.get("做什么") or item.get("action") or "").strip(),
                deliverable=str(item.get("交付物") or item.get("deliverable") or "").strip(),
            )
        )

    return Order(goal=str(goal).strip(), tasks=tasks, raw=text)


# ---------------------------------------------------------------- 校验


def validate(order: Order) -> list[str]:
    """返回问题列表。空列表 = 通过。

    每条问题都要写成「模型看了就知道怎么改」的话，因为会被回灌给模型。
    """
    problems: list[str] = []

    # 1. 目标
    if not order.goal:
        problems.append("「目标」为空，必须用一句话写清要做什么")
    elif len(order.goal) > 60:
        problems.append(f"「目标」过长（{len(order.goal)} 字），压到 60 字以内")

    # 2. 任务条数
    n = len(order.tasks)
    if n < MIN_TASKS:
        problems.append(f"任务表为空，至少 {MIN_TASKS} 条")
    elif n > MAX_TASKS:
        problems.append(f"任务 {n} 条，超过上限 {MAX_TASKS} 条")

    # 3. 逐条检查
    for i, t in enumerate(order.tasks, 1):
        if not t.dept:
            problems.append(f"第 {i} 条缺「司」")
        elif t.dept == AUDITOR:
            problems.append(
                f"第 {i} 条派给了刑部。刑部只审不做，任何执行任务都不得派给刑部"
            )
        elif t.dept not in DEPARTMENTS:
            problems.append(
                f"第 {i} 条的司「{t.dept}」不在五司之内，只能是：{'、'.join(DEPARTMENTS)}"
            )

        if not t.action:
            problems.append(f"第 {i} 条缺「做什么」")

        if not t.deliverable:
            problems.append(f"第 {i} 条缺「交付物」。交付物必须是文件、结论或数字")
        elif any(w in t.deliverable for w in VAGUE_WORDS):
            problems.append(
                f"第 {i} 条的交付物「{t.deliverable}」太空泛，改成具体的产出"
            )

    # 4. 分配合理性
    counts: dict[str, int] = {}
    for t in order.tasks:
        if t.dept in DEPARTMENTS:
            counts[t.dept] = counts.get(t.dept, 0) + 1

    for dept, c in counts.items():
        if c > MAX_TASKS_PER_DEPT:
            problems.append(
                f"{dept} 被派了 {c} 条，同一司最多 {MAX_TASKS_PER_DEPT} 条，请分给其他司"
            )

    if n >= MIN_DISTINCT_DEPTS and len(counts) < MIN_DISTINCT_DEPTS:
        problems.append(
            f"只用了 {len(counts)} 个司（{'、'.join(counts) or '无'}），"
            f"{n} 条任务至少应覆盖 {MIN_DISTINCT_DEPTS} 个不同的司"
        )

    return problems


def is_valid(order: Order) -> bool:
    return not validate(order)


# ---------------------------------------------------------------- 汇总


def describe(order: Order) -> str:
    """把升堂单渲染成便于人读的文本（命令行演示用）。"""
    lines = [f"目标：{order.goal}", "任务："]
    for i, t in enumerate(order.tasks, 1):
        lines.append(f"  {i}. [{t.dept}] {t.action} → {t.deliverable}")
    return "\n".join(lines)
