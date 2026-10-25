"""
升堂 · 汇总

中堂把这一场的全部记录收成一份结案文书。

它不是把前面的结果再念一遍——那没有意义。它要做的是：
  1. 对着主公的原需求说，这一单到底做成了什么（不是念任务清单）
  2. 把附议改单、勘验补交这些「过程里发生过的事」记下来
  3. 主动交代风险和没做的事
     ——协议铁律：交活必须写「风险/未做」，防止只报喜

模型写不出来时退回一份程序拼的文书。汇总不能因为模型没应就整场没有结论。

落盘位置不在成品夹：成品夹只留一件能打开的成品（硬规矩）。
结案文书写到 `.shengtang/结案/<编号>.md`，本机留存，不进仓库。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts
from .execute import Execution
from .schema import Order, extract_json
from .verify import Inspection

EventSink = Callable[[dict], None]


@dataclass
class Summary:
    #: 一句话结语，对着原需求说
    verdict: str = ""
    points: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    undone: list[str] = field(default_factory=list)
    #: 中堂 / 程序
    source: str = "中堂"
    #: 结案文书落盘路径，没落盘就是空
    path: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "points": self.points,
            "risks": self.risks,
            "undone": self.undone,
            "source": self.source,
            "path": self.path,
        }


# ---------------------------------------------------------------- 解析


def _text(data: dict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _items(data: dict, *keys: str) -> list[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


def parse_summary(raw: str) -> Summary | None:
    """读不懂就返回 None，由调用方退回程序版。"""
    try:
        data = extract_json(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None

    summary = Summary(
        verdict=_text(data, "结语", "一句话", "verdict"),
        points=_items(data, "结论", "各项", "points"),
        risks=_items(data, "风险", "risks"),
        undone=_items(data, "未做", "未完成", "undone"),
        source="中堂",
    )
    if not summary.verdict and not summary.points:
        return None
    return summary


def _ensure_retry_note(summary: Summary, inspection: Inspection | None) -> Summary:
    """模型爱把「补交过但没过」写成「未交」，像根本没试过。

    提示词里立了规矩它也不一定听，所以程序再兜一道：
    只要有一项补交过仍不合格，结案里就必须出现「补交」两个字。
    """
    if inspection is None:
        return summary
    retried_failed = [v for v in inspection.verdicts if v.retried and not v.ok]
    if not retried_failed:
        return summary

    joined = "".join(summary.undone) + "".join(summary.risks) + summary.verdict
    if "补交" in joined:
        return summary

    for verdict in retried_failed:
        reason = (verdict.reason or "").strip()
        if len(reason) > 40:
            reason = reason[:40] + "…"
        summary.undone.append(
            f"第 {verdict.index + 1} 条（{verdict.dept}）补交过一轮仍未过"
            + (f"：{reason}" if reason else "")
        )
    return summary


def fallback(order: Order, inspection: Inspection | None) -> Summary:
    """模型没应时的程序版。宁可干巴，也不能没有结论。"""
    points: list[str] = []
    undone: list[str] = []
    for verdict in inspection.verdicts if inspection else []:
        mark = "过" if verdict.ok else "退"
        line = f"第 {verdict.index + 1} 条（{verdict.dept}）{mark}"
        if verdict.retried:
            line += "（补交后过）" if verdict.ok else "（补交过，仍不合格）"
        if verdict.reason:
            line += f"：{verdict.reason}"
        points.append(line)
        if not verdict.ok:
            undone.append(
                f"第 {verdict.index + 1} 条补交过一轮仍未过"
                if verdict.retried
                else f"第 {verdict.index + 1} 条没过勘验"
            )

    risks = ["有任务没过刑部勘验，成品可能不完整"] if inspection and not inspection.passed else []
    return Summary(
        verdict=f"这一单已办完，共 {len(points)} 项，结论见下。",
        points=points,
        risks=risks,
        undone=undone,
        source="程序",
    )


# ---------------------------------------------------------------- 落盘


def render_markdown(
    summary: Summary,
    order_no: str,
    requirement: str,
    execution: Execution | None = None,
) -> str:
    lines = [
        f"# 结案 · {order_no}",
        "",
        f"**原需求**：{requirement}",
        "",
        f"**结语**：{summary.verdict}",
        "",
        "## 各项结论",
        "",
    ]
    lines += [f"- {item}" for item in summary.points] or ["- （无）"]
    lines += ["", "## 风险", ""]
    lines += [f"- {item}" for item in summary.risks] or ["- 无"]
    lines += ["", "## 未做", ""]
    lines += [f"- {item}" for item in summary.undone] or ["- 无"]
    if execution is not None:
        lines += ["", f"**成品**：{execution.product}"]
    lines.append("")
    lines.append(f"> 结案文书由{summary.source}写成。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程


def run_summary(
    llm,
    order: Order,
    requirement: str,
    execution: Execution | None = None,
    inspection: Inspection | None = None,
    attest: dict | None = None,
    out_dir: Path | str | None = None,
    order_no: str | None = None,
    on_event: EventSink | None = None,
) -> Summary:
    """中堂写结案。模型没应就退回程序版，绝不交一份空白结论。"""
    emit = on_event or (lambda _e: None)
    emit({"type": "summary/start"})

    summary: Summary | None = None
    try:
        raw = llm.quick(
            prompts.summary_prompt(requirement, order, execution, inspection, attest),
            max_tokens=1200,
        )
        summary = parse_summary(raw)
    except Exception as exc:  # noqa: BLE001 - 汇总失败不能拖垮整场
        emit({"type": "summary/skip", "reason": str(exc)})

    if summary is None:
        summary = fallback(order, inspection)
        emit({"type": "summary/fallback"})
    else:
        # 模型爱把「补交过但没过」写成「未交」，程序再兜一道
        summary = _ensure_retry_note(summary, inspection)

    if out_dir is not None:
        number = order_no or (execution.order_no if execution else "未编号")
        target = Path(out_dir) / f"{number}.md"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                render_markdown(summary, number, requirement, execution), "utf-8"
            )
            summary.path = str(target)
        except OSError as exc:  # noqa: BLE001 - 写不下就只在前端显示
            emit({"type": "summary/skip", "reason": f"结案文书没写成：{exc}"})

    emit({"type": "summary/final", **summary.to_dict()})
    return summary
