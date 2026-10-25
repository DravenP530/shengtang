"""
升堂 · 勘验

刑部只审不做。但「审」不能只看文件在不在——那是保安，不是刑部。

两层，缺一不可：

  客观层（程序算）  文件在不在、多大、HTML 有没有写到 </html>
  主观层（模型判）  这份东西算不算交了这活、有没有可复核的东西

判定方向定死：客观层不过就直接退，不必问模型（省一次调用，也不给它翻案的机会）；
客观层过了才让刑部看内容。

模型没应／回话读不懂时，按客观层放行，但如实标「刑部未应」。
少一次判断不能让好活白干；也不能假装刑部核过了。

退回的条目真补交一次：重新调那一司，带上退回理由，再核一遍。
补交只补一次——协议就是「退回补交」，没有第二轮补交。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts
from .execute import Check, Execution, strip_model_text, task_prompt
from .schema import Order, extract_json

EventSink = Callable[[dict], None]

#: 给刑部看的正文长度上限。再长它也不会细看，反而拖慢。
#: 但必须明确告诉它「这是节选」——否则它会把节选当成文件被截断，好活也判退（实测踩过）。
EXCERPT = 6000


@dataclass
class Verdict:
    index: int
    dept: str
    deliverable: str
    ok: bool
    reason: str = ""
    #: 客观层查出的事实，前端可以显示「文件在、12KB、完整页面」
    facts: list[str] = field(default_factory=list)
    #: 谁下的判：客观 / 客观+刑部 / 客观（刑部未应）
    source: str = "客观"
    #: 是否补交之后重判的
    retried: bool = False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "dept": self.dept,
            "deliverable": self.deliverable,
            "ok": self.ok,
            "reason": self.reason,
            "facts": self.facts,
            "source": self.source,
            "retried": self.retried,
        }


@dataclass
class Inspection:
    passed: bool
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def failed(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.ok]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# ---------------------------------------------------------------- 解析


def parse_verdict(raw: str) -> tuple[bool, str] | None:
    """把刑部的回话解析成 (过?, 理由)。读不懂返回 None。"""
    try:
        data = extract_json(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None

    word = ""
    for key in ("核", "结论", "结果", "verdict", "result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            word = value.strip()
            break
    if not word:
        return None

    low = word.lower()
    # 先判退：「不过」里也有「过」，顺序不能反
    if any(w in low for w in ("退", "驳", "不过", "不合格", "fail", "reject")):
        reason = ""
        for key in ("理由", "原因", "缺什么", "reason"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                reason = value.strip()
                break
        return False, reason or "刑部打退，没说理由"
    if any(w in low for w in ("过", "核", "pass", "ok")):
        return True, ""
    return None


# ---------------------------------------------------------------- 客观层


def audit(check: Check) -> list[str]:
    """程序能查的事实。查不出来的不猜，直接不写。"""
    facts: list[str] = []
    if check.path:
        path = Path(check.path)
        if path.is_file():
            facts.append(f"文件在：{path.name}，{path.stat().st_size} 字节")
            if path.suffix.lower() in (".html", ".htm"):
                text = path.read_text("utf-8", "replace")
                if "</html>" in text.lower():
                    facts.append("写到了 </html>，是完整页面")
                else:
                    facts.append("没写到 </html>，是半截页面")
        else:
            facts.append("说好的文件不在")
    elif check.note:
        facts.append(check.note)
    if check.body:
        facts.append(f"正文 {len(check.body)} 字符")
    if check.patched:
        facts.append("正文是被截断后补的闭合标签，尾部可能不全")
    return facts


# ---------------------------------------------------------------- 判一条


def _judge(llm, order: Order, check: Check, emit: EventSink) -> Verdict:
    task = order.tasks[check.index] if 0 <= check.index < len(order.tasks) else None
    deliverable = task.deliverable if task else check.deliverable
    action = task.action if task else ""
    facts = audit(check)

    # 客观层不过：执行阶段就没交出来，不必问模型
    if not check.ok:
        return Verdict(
            check.index, check.dept, deliverable, False,
            check.note or "没交出来", facts, "客观",
        )

    body = (check.body or "").strip()
    if not body:
        return Verdict(
            check.index, check.dept, deliverable, False,
            "交回来的正文是空的", facts, "客观",
        )

    prompt = prompts.verify_prompt(
        order.goal,
        check.dept,
        action,
        deliverable,
        facts,
        body[:EXCERPT],
        cut=len(body) > EXCERPT,
    )
    try:
        raw = llm.quick(prompt, max_tokens=300)
    except Exception as exc:  # noqa: BLE001 - 刑部没开口不能当成活没干
        emit({"type": "verify/skip", "index": check.index, "dept": check.dept, "reason": str(exc)})
        return Verdict(
            check.index, check.dept, deliverable, True,
            f"刑部没应（{exc}），按客观检查放行", facts, "客观（刑部未应）",
        )

    parsed = parse_verdict(raw)
    if parsed is None:
        emit({"type": "verify/skip", "index": check.index, "dept": check.dept, "reason": "回话读不懂"})
        return Verdict(
            check.index, check.dept, deliverable, True,
            "刑部回话读不懂，按客观检查放行", facts, "客观（刑部未应）",
        )

    ok, reason = parsed
    return Verdict(check.index, check.dept, deliverable, ok, reason, facts, "客观+刑部")


def _retry_one(llm, order: Order, index: int, reason: str) -> str:
    """让那一司补交。带上退回理由，否则它只会把原话再抄一遍。"""
    task = order.tasks[index]
    prompt = task_prompt(order, task.dept, task.action, task.deliverable)
    if reason:
        prompt += f"\n\n上一次交的被刑部打回，理由：{reason}\n这次必须补上这一点。"
    raw = llm.quick(prompt, max_tokens=12000)
    return strip_model_text(raw)


# ---------------------------------------------------------------- 主流程


def run_verify(
    llm,
    order: Order,
    execution: Execution,
    on_event: EventSink | None = None,
    allow_retry: bool = True,
    announce: bool = True,
) -> Inspection:
    """逐条核。退回的真补交一次，再核一遍。

    announce=False 用于主公下令「再补一轮」的第二次调用——
    前端已经处在勘验态，不必再播报一次开场。
    """
    emit = on_event or (lambda _e: None)
    if announce:
        emit({"type": "verify/start", "total": len(execution.checks)})

    verdicts: list[Verdict] = []
    for check in execution.checks:
        emit({"type": "verify/checking", "index": check.index, "dept": check.dept})
        verdict = _judge(llm, order, check, emit)
        verdicts.append(verdict)
        emit({"type": "verify/verdict", **verdict.to_dict()})

    if allow_retry:
        for pos, verdict in enumerate(list(verdicts)):
            if verdict.ok:
                continue
            emit(
                {
                    "type": "verify/retry",
                    "index": verdict.index,
                    "dept": verdict.dept,
                    "reason": verdict.reason,
                }
            )
            try:
                new_body = _retry_one(llm, order, verdict.index, verdict.reason)
            except Exception as exc:  # noqa: BLE001 - 补交失败就维持原判
                emit(
                    {
                        "type": "verify/retry-failed",
                        "index": verdict.index,
                        "dept": verdict.dept,
                        "reason": str(exc),
                    }
                )
                continue
            if len(new_body.strip()) < 20:
                emit(
                    {
                        "type": "verify/retry-failed",
                        "index": verdict.index,
                        "dept": verdict.dept,
                        "reason": "补回来的还是空的",
                    }
                )
                continue

            check = next(
                (c for c in execution.checks if c.index == verdict.index), None
            )
            if check is None:
                continue
            check.body = new_body
            # 只有「自己单独落盘的那一份」才回写文件；
            # 「已并入这一件成品」的回写会把别人的内容盖掉。
            if check.path and check.note == "成品已落盘":
                target = Path(check.path)
                if target.is_file():
                    target.write_text(new_body, "utf-8")

            again = _judge(llm, order, check, emit)
            again.retried = True
            verdicts[pos] = again
            emit({"type": "verify/verdict", **again.to_dict()})

    passed = bool(verdicts) and all(v.ok for v in verdicts)
    inspection = Inspection(passed, verdicts)
    emit({"type": "verify/final", **inspection.to_dict()})
    return inspection
