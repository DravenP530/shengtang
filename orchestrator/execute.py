"""
升堂 · 执行

附议定了之后，五司把交付物写成文件。刑部不参与写，只核对文件在不在。

产出只落在一个成品夹里，不另建案卷，不放说明和草稿：

    成品/<能打开的成品>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .schema import DEPARTMENTS, Order

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_CASE_ID = re.compile(r"^\d{8}-\d{2}$")
#: 这些是过程纸，不是主公要打开的那一件成品。
_SIDE_PAPER = ("分镜", "说明", "报告", "风格", "脚本", "文档", "关键帧", "测试")
_FENCE = re.compile(r"```(?:html|markdown|md|text)?\s*(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)

#: 单份交付物的上限，防止模型把案卷写成一本厚书。
MAX_BYTES = 200_000


@dataclass
class Check:
    index: int
    dept: str
    deliverable: str
    path: str
    ok: bool
    note: str


@dataclass
class Execution:
    order_no: str
    case_dir: Path
    product: Path
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def to_dict(self) -> dict:
        name = self.product.name if self.product.is_file() else ""
        url = f"/成品/{name}" if name else ""
        return {
            "orderNo": self.order_no,
            "caseDir": str(self.case_dir),
            "productPath": str(self.product),
            "productUrl": url,
            "ok": self.ok,
            "checks": [
                {
                    "index": c.index,
                    "dept": c.dept,
                    "deliverable": c.deliverable,
                    "path": c.path,
                    "ok": c.ok,
                    "note": c.note,
                }
                for c in self.checks
            ],
        }


def allocate_case_id(root: Path, today: str | None = None) -> str:
    """当天已有 01 就用 02，不覆盖上一场。"""
    day = today or datetime.now().strftime("%Y%m%d")
    n = 1
    while (root / f"{day}-{n:02d}").exists():
        n += 1
        if n > 99:
            raise RuntimeError("今天的案卷编号用尽")
    return f"{day}-{n:02d}"


def safe_stem(name: str) -> str:
    stem = _UNSAFE.sub("", name).strip().strip(".")
    stem = stem.replace("..", "")
    return stem[:40] or "交付物"


def strip_model_text(text: str) -> str:
    """去掉思考过程和代码围栏，只留文件正文。"""
    raw = _THINK.sub("", text or "")
    blocks = [m.group(1).strip() for m in _FENCE.finditer(raw) if m.group(1).strip()]
    if blocks:
        return max(blocks, key=len)
    return raw.strip()


def choose_filename(deliverable: str, body: str, used: set[str]) -> str:
    stem = safe_stem(deliverable)
    lower = stem.lower()
    head = body.lstrip()[:40].lower()
    if "." in stem and lower.rsplit(".", 1)[-1] in {"html", "htm", "md", "txt", "css", "js"}:
        name = stem
    elif head.startswith("<!doctype") or head.startswith("<html") or head.startswith("<style"):
        name = stem + ".html" if not lower.endswith(".html") else stem
    else:
        name = stem + ".md"
    base, dot, ext = name.rpartition(".")
    if not dot:
        base, ext = name, ""
    candidate = name
    n = 2
    while candidate in used:
        candidate = f"{base}-{n}.{ext}" if ext else f"{base}-{n}"
        n += 1
    used.add(candidate)
    return candidate


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write(root: Path, path: Path, text: str) -> None:
    if not _inside(root, path):
        raise RuntimeError(f"拒绝写出案卷：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")[:MAX_BYTES]
    path.write_bytes(data)


def is_side_paper(action: str, deliverable: str) -> bool:
    text = action + deliverable
    return any(word in text for word in _SIDE_PAPER)


def is_complete_page(body: str) -> bool:
    return "</html>" in body.lower()


def _task_prompt(order: Order, dept: str, action: str, deliverable: str) -> str:
    visual = any(word in (action + deliverable) for word in ("动画", "页面", "原型", "demo", "HTML", "html"))
    shape = (
        "输出一份能单独打开的完整 HTML，从 <!DOCTYPE html> 写起，"
        "样式和动画都写在这一份里，不要外链，必须写到 </html> 结束。"
        if visual
        else "输出 Markdown 正文。"
    )
    return (
        f"你是升堂的{dept}，只负责这一条，不要做别人的活。\n"
        f"目标：{order.goal}\n"
        f"你要做：{action}\n"
        f"交付物：{deliverable}\n"
        "只输出文件正文，不要开场白，不要解释。\n"
        f"{shape}\n"
    )


def execute_order(
    llm,
    order: Order,
    requirement: str,
    case_root: Path,
    on_event=None,
    order_no: str | None = None,
) -> Execution:
    """五司按单写文件。能打开的成品直接进成品夹，说明和案卷都不落盘。"""
    product_dir = case_root.resolve()
    product_dir.mkdir(parents=True, exist_ok=True)
    number = order_no or allocate_case_id(product_dir)
    if order_no and not _CASE_ID.match(number):
        raise RuntimeError(f"编号不合法：{number}")

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    emit({"type": "execute/start", "orderNo": number, "caseDir": str(product_dir)})

    used: set[str] = set()
    checks: list[Check] = []
    htmls: list[tuple[str, str]] = []

    for index, task in enumerate(order.tasks):
        dept = task.dept
        if dept not in DEPARTMENTS:
            checks.append(Check(index, dept, task.deliverable, "", False, "此司不接执行"))
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": False, "note": "此司不接执行"})
            continue
        emit({"type": "execute/working", "index": index, "dept": dept})
        try:
            raw = llm.quick(_task_prompt(order, dept, task.action, task.deliverable), max_tokens=6000)
            body = strip_model_text(raw)
        except Exception as exc:  # noqa: BLE001
            body = ""
            fail_note = str(exc)
        else:
            fail_note = ""
        if len(body.strip()) < 20:
            note = fail_note or "交回来的是空的"
            checks.append(Check(index, dept, task.deliverable, "", False, note))
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": False, "note": note})
            continue
        if is_side_paper(task.action, task.deliverable):
            checks.append(Check(index, dept, task.deliverable, "", True, "说明不进成品夹"))
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": True, "note": "说明不进成品夹"})
            continue
        filename = choose_filename(task.deliverable, body, used)
        if not filename.lower().endswith((".html", ".htm")) or not is_complete_page(body):
            note = "写到一半就停了，不进成品夹" if filename.lower().endswith((".html", ".htm")) else "说明不进成品夹"
            ok = not filename.lower().endswith((".html", ".htm"))
            checks.append(Check(index, dept, task.deliverable, "", ok, note))
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": ok, "note": note})
            continue
        htmls.append((filename, body))
        checks.append(Check(index, dept, task.deliverable, "", True, "待收成一件成品"))
        emit({"type": "execute/task", "index": index, "dept": dept, "ok": True, "note": "待收成一件成品"})

    product: Path = product_dir
    if htmls:
        filename, body = max(htmls, key=lambda item: len(item[1]))
        for old in product_dir.iterdir():
            if old.is_file():
                old.unlink()
        product = product_dir / filename
        _write(product_dir, product, body)
        for item in checks:
            if item.note != "待收成一件成品":
                continue
            item.path = str(product)
            item.note = "成品已落盘" if safe_stem(item.deliverable) == filename.split(".")[0] else "已并入这一件成品"
    result = Execution(number, product_dir, product, checks)
    emit({"type": "execute/final", **result.to_dict()})
    return result
