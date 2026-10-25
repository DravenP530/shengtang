"""
升堂 · 附议

拆单之后、执行之前，五司各自表态。这一层解决的是「真表态」：

  写死的「附议」不是附议，是配音。真的附议必须满足三条：
    1. 每个司从自己的职掌看这单，而不是泛泛评论整张单（视角不同）
    2. 异议必须指明改哪一条、改成什么（否则没法落到单子上）
    3. 改完要重新表态；两轮不成上抛主公（协议原文，不许死循环）

三个不堵流程的决定：
  · 某个司没开口（模型报错／回话读不懂）→ 跳过它，不当成异议
  · 提了异议但没说清改哪条 → 降级为附议，并记下原因
  · 异议改完会让单子不合格 → 这条不改（宁可这轮白提，也不让坏单子往下走）
  · 一条也改不动 → 立刻上抛，不空转第二轮

刑部不参加附议：它只审不做，开工前没有立场。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .schema import DEPARTMENTS, Order, Task, extract_json, validate

#: 事件回调签名，与 orchestrator.EventSink 一致。
EventSink = Callable[[dict], None]

AGREE = "附议"
DISSENT = "异议"

#: 附议最多两轮。两轮还不成，按协议上抛主公拍板，不许无限吵。
MAX_ROUNDS = 2

_DIGITS = re.compile(r"\d+")
_AGREE_WORDS = ("附议", "同意", "无异议", "agree", "ok")
_DISSENT_WORDS = ("异议", "反对", "不同意", "dissent", "reject")


@dataclass
class Stance:
    """一个司的一次表态。"""

    dept: str
    verdict: str = AGREE
    reason: str = ""
    #: 改第几条，1 起；0 表示没指条
    target: int = 0
    change: str = ""
    deliverable: str = ""
    #: 异议三要素齐全才算有效
    valid: bool = False

    def to_dict(self) -> dict:
        return {
            "dept": self.dept,
            "verdict": self.verdict,
            "reason": self.reason,
            "target": self.target,
            "change": self.change,
            "deliverable": self.deliverable,
            "valid": self.valid,
        }


@dataclass
class Amendment:
    """一次真落到单子上的改单。"""

    #: 0 起的任务下标
    index: int
    by: str
    action: str
    deliverable: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "by": self.by,
            "action": self.action,
            "deliverable": self.deliverable,
            "reason": self.reason,
        }


@dataclass
class AttestResult:
    #: 附议之后的单子。passed 为假时它是最后一轮的版本，不可直接执行。
    order: Order
    passed: bool
    rounds: int
    stances: list[Stance] = field(default_factory=list)
    amendments: list[Amendment] = field(default_factory=list)

    @property
    def escalate(self) -> bool:
        """没达成共识，须上抛主公。"""
        return not self.passed

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "rounds": self.rounds,
            "order": self.order.to_dict(),
            "stances": [s.to_dict() for s in self.stances],
            "amendments": [a.to_dict() for a in self.amendments],
        }


# ---------------------------------------------------------------- 解析


def _pick(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _norm_verdict(text: str) -> str | None:
    """把模型的各种说法归一成附议／异议。判不出返回 None。"""
    low = (text or "").strip().lower()
    if not low:
        return None
    # 先判「无异议」——它含「异议」两个字，顺序不能反
    if "无异议" in low or "没有异议" in low:
        return AGREE
    for word in _DISSENT_WORDS:
        if word in low:
            return DISSENT
    for word in _AGREE_WORDS:
        if word in low:
            return AGREE
    return None


def _norm_target(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        found = _DIGITS.search(value)
        if found:
            return int(found.group())
    return 0


def parse_stance(raw: str, dept: str) -> Stance | None:
    """把模型的回话解析成表态。读不懂就返回 None（当作这一司没开口）。"""
    try:
        data = extract_json(raw)
    except Exception:  # noqa: BLE001 - 解析层要忍，读不懂是常态之一
        return None
    if not isinstance(data, dict):
        return None

    verdict = _norm_verdict(
        _pick(data, "表态", "立场", "结论", "verdict", "stance", "result")
    )
    if verdict is None:
        return None

    stance = Stance(dept=dept, verdict=verdict)
    stance.reason = _pick(data, "理由", "原因", "依据", "reason")
    stance.change = _pick(data, "改成什么", "怎么改", "改法", "新的做法", "change")
    stance.deliverable = _pick(data, "交付物改成", "新交付物", "交付物", "deliverable")
    stance.target = _norm_target(
        data.get("改第几条")
        or data.get("第几条")
        or data.get("条")
        or data.get("target")
        or data.get("index")
    )

    if verdict == DISSENT:
        stance.valid = stance.target >= 1 and bool(stance.change)
        if not stance.valid:
            # 要素不全的异议不算数：降级为附议，但把原话留着，前端能看出它想说没说到点
            stance.verdict = AGREE
            note = "提了异议，但没说清改哪条、改成什么"
            stance.reason = f"{note}（原话：{stance.reason}）" if stance.reason else note

    return stance


# ---------------------------------------------------------------- 改单


def _amend(order: Order, stances: list[Stance]) -> tuple[Order, list[Amendment]]:
    """把有效异议落到单子上。同一行只认第一个；改坏了这一条就不改。"""
    tasks = [Task(t.dept, t.action, t.deliverable) for t in order.tasks]
    touched: set[int] = set()
    applied: list[Amendment] = []

    for stance in stances:
        if stance.verdict != DISSENT or not stance.valid:
            continue
        idx = stance.target - 1
        if idx in touched or not (0 <= idx < len(tasks)):
            continue

        old = tasks[idx]
        action = stance.change.strip() or old.action
        deliverable = stance.deliverable.strip() or old.deliverable
        if action == old.action and deliverable == old.deliverable:
            continue

        candidate = Task(old.dept, action, deliverable)
        trial = Order(order.goal, tasks[:idx] + [candidate] + tasks[idx + 1 :])
        if validate(trial):
            # 改完反而不合格（交付物空了、司超载了……）→ 这一条不动
            continue

        tasks[idx] = candidate
        touched.add(idx)
        applied.append(
            Amendment(idx, stance.dept, action, deliverable, stance.reason)
        )

    if not applied:
        return order, []
    return Order(order.goal, tasks, order.raw), applied


# ---------------------------------------------------------------- 主流程


def _ask_one(
    llm,
    order: Order,
    dept: str,
    round_no: int,
    amendments: list[Amendment],
    emit: EventSink,
) -> Stance | None:
    """问一个司。它不开口也不拖垮整场附议。"""
    prompt = prompts.attest_prompt(order, dept, round_no, amendments)
    try:
        raw = llm.quick(prompt, max_tokens=400)
    except Exception as exc:  # noqa: BLE001
        emit(
            {
                "type": "attest/skip",
                "round": round_no,
                "dept": dept,
                "reason": f"模型没应：{exc}",
            }
        )
        return None

    stance = parse_stance(raw, dept)
    if stance is None:
        emit(
            {
                "type": "attest/skip",
                "round": round_no,
                "dept": dept,
                "reason": "没读懂它的回话",
            }
        )
    return stance


def run_attest(
    llm,
    order: Order,
    max_rounds: int = MAX_ROUNDS,
    on_event: EventSink | None = None,
    depts: tuple[str, ...] = DEPARTMENTS,
) -> AttestResult:
    """五司依次表态；有异议就改单，改完再表态。最多 max_rounds 轮。

    串行调而不是并发：本地只跑一个模型，并发只会排队还把端点压死；
    朝堂本来也是一个一个表态，串行反而对得上演出。
    """
    emit = on_event or (lambda _e: None)
    current = order
    all_stances: list[Stance] = []
    all_amendments: list[Amendment] = []

    for round_no in range(1, max_rounds + 1):
        emit({"type": "attest/round", "round": round_no})
        stances: list[Stance] = []

        for dept in depts:
            emit({"type": "attest/thinking", "round": round_no, "dept": dept})
            stance = _ask_one(llm, current, dept, round_no, all_amendments, emit)
            if stance is None:
                continue
            stances.append(stance)
            all_stances.append(stance)
            emit({"type": "attest/stance", "round": round_no, **stance.to_dict()})

        dissent = [s for s in stances if s.verdict == DISSENT]
        if not dissent:
            emit(
                {
                    "type": "attest/final",
                    "passed": True,
                    "escalate": False,
                    "rounds": round_no,
                    "order": current.to_dict(),
                    "amendments": [a.to_dict() for a in all_amendments],
                }
            )
            return AttestResult(current, True, round_no, all_stances, all_amendments)

        current, applied = _amend(current, dissent)
        if not applied:
            # 有异议却一条也落不下去（要素不全／改了就违规）→ 直接上抛，不空转
            emit(
                {
                    "type": "attest/final",
                    "passed": False,
                    "escalate": True,
                    "rounds": round_no,
                    "order": current.to_dict(),
                    "amendments": [a.to_dict() for a in all_amendments],
                    "reason": "有异议，但落不到单子上",
                }
            )
            return AttestResult(current, False, round_no, all_stances, all_amendments)

        all_amendments.extend(applied)
        for amendment in applied:
            emit({"type": "attest/amended", "round": round_no, **amendment.to_dict()})

    emit(
        {
            "type": "attest/final",
            "passed": False,
            "escalate": True,
            "rounds": max_rounds,
            "order": current.to_dict(),
            "amendments": [a.to_dict() for a in all_amendments],
            "reason": f"{max_rounds} 轮仍未达成共识",
        }
    )
    return AttestResult(current, False, max_rounds, all_stances, all_amendments)
