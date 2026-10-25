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

#: 成品要挑「最像成品」的那一件。只按长度挑会选错——
#: 实测踩过：礼部交了个 7868 字符的「文案清单」HTML（内容其实是一整个动画页），
#: 把兵部 3291 字符的动画页挤掉，结果结案说「动画已交付」，成品却是任务清单。
#:
#: 所以分两步：先排除「纸」（清单/说明/文案/报告），再在剩下的里看是不是视觉类，最后比长度。
_VISUAL_HINTS = ("动画", "页面", "原型", "demo", "网页", "场景", "可视化", "小游戏")
_PAPER_HINTS = ("清单", "说明", "文案", "报告", "文档", "名册", "任务表", "升堂单")


def _product_rank(item: tuple[str, str, str]) -> tuple[int, int, int]:
    """纸类排最后；同为视觉类才比长度。"""
    _filename, body, hint = item
    paper = 1 if any(word in hint for word in _PAPER_HINTS) else 0
    visual = 1 if any(word in hint for word in _VISUAL_HINTS) else 0
    return (-paper, visual, len(body))


@dataclass
class Check:
    index: int
    dept: str
    deliverable: str
    path: str
    ok: bool
    note: str
    #: 交回来的正文。只留在内存里给刑部核，不进 to_dict——正文动辄几万字符，
    #: 发给前端没有意义，还会把 SSE 撑爆。
    body: str = ""
    #: 正文是被 max_tokens 截断后补的闭合标签。刑部要知道这件事。
    patched: bool = False


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
                    "patched": c.patched,
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


def close_partial_html(body: str) -> tuple[str, bool]:
    """模型被 max_tokens 截断时，补个收尾，让它至少能打开。

    只补闭合标签，不编内容。补过之后页面尾部可能缺，但浏览器能渲染——
    比交一个打不开的半截文件强，也比整条白干强。补过这件事要记下来，
    刑部有权知道这份东西的结尾是拼上去的。
    """
    low = body.lower()
    if "</html>" in low:
        return body, False
    if "<html" not in low and "<!doctype" not in low:
        return body, False
    tail = ""
    if "</body>" not in low:
        tail += "\n</body>"
    return body + tail + "\n</html>", True


def task_prompt(order: Order, dept: str, action: str, deliverable: str) -> str:
    visual = any(word in (action + deliverable) for word in ("动画", "页面", "原型", "demo", "HTML", "html"))
    #: 交付物里要说明/步骤的，光让模型「不写说明」就永远交不出来（实测踩过）
    wants_doc = any(
        word in deliverable for word in ("说明", "步骤", "用法", "指南", "文档", "清单")
    )
    if visual:
        shape = (
            "输出一份能单独打开的完整 HTML，从 <!DOCTYPE html> 写起，"
            "样式和动画都写在这一份里，不要外链，必须写到 </html> 结束。"
            "篇幅控制在 200 行以内。"
        )
        if wants_doc:
            shape += (
                "交付物里还要说明或操作步骤的，就在页面末尾加一节写出来"
                "（怎么打开、怎么操作、支持哪些浏览器），不要只说「详见文档」。"
            )
        else:
            shape += "不写注释、不写多余说明——写不完就等于没交。"
    else:
        shape = "输出 Markdown 正文，不超过 400 字。"
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
    #: (文件名, 正文, 这条任务要做什么)——第三项用来判断哪一件更像成品
    htmls: list[tuple[str, str, str]] = []

    for index, task in enumerate(order.tasks):
        dept = task.dept
        if dept not in DEPARTMENTS:
            checks.append(
                Check(index, dept, task.deliverable, "", False, "此司不接执行", "")
            )
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": False, "note": "此司不接执行"})
            continue
        emit({"type": "execute/working", "index": index, "dept": dept})
        try:
            # 12000 而不是 6000：6000 写不完一页带样式和动画的 HTML，
            # 实测会被硬截断，永远到不了 </html>，那条必然判失败。
            raw = llm.quick(task_prompt(order, dept, task.action, task.deliverable), max_tokens=12000)
            body = strip_model_text(raw)
        except Exception as exc:  # noqa: BLE001
            body = ""
            fail_note = str(exc)
        else:
            fail_note = ""
        if len(body.strip()) < 20:
            note = fail_note or "交回来的是空的"
            checks.append(Check(index, dept, task.deliverable, "", False, note, body))
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": False, "note": note})
            continue
        # 交付物里提到「说明」但正文其实是一整个页面时，不能当纸丢掉——
        # 否则「HTML + 使用说明」这种活，交出来的页面会凭空消失
        body_is_html = body.lstrip().lower().startswith(("<!doctype", "<html", "<style"))
        if is_side_paper(task.action, task.deliverable) and not body_is_html:
            checks.append(
                Check(index, dept, task.deliverable, "", True, "说明不进成品夹", body)
            )
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": True, "note": "说明不进成品夹"})
            continue
        filename = choose_filename(task.deliverable, body, used)
        is_html = filename.lower().endswith((".html", ".htm"))
        patched = False
        if is_html:
            body, patched = close_partial_html(body)
        if is_html and not is_complete_page(body):
            # 连 <html 都没有，补也补不成页面
            note = "写到一半就停了，不进成品夹"
            checks.append(Check(index, dept, task.deliverable, "", False, note, body))
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": False, "note": note})
            continue
        if not is_html:
            checks.append(
                Check(index, dept, task.deliverable, "", True, "说明不进成品夹", body)
            )
            emit({"type": "execute/task", "index": index, "dept": dept, "ok": True, "note": "说明不进成品夹"})
            continue
        htmls.append((filename, body, task.action + task.deliverable))
        checks.append(
            Check(index, dept, task.deliverable, "", True, "待收成一件成品", body, patched)
        )
        emit(
            {
                "type": "execute/task",
                "index": index,
                "dept": dept,
                "ok": True,
                "note": "结尾被截断，已补闭合标签" if patched else "待收成一件成品",
                "patched": patched,
            }
        )

    product: Path = product_dir
    if htmls:
        filename, body, _hint = max(htmls, key=_product_rank)
        for old in product_dir.iterdir():
            if old.is_file():
                old.unlink()
        product = product_dir / filename
        _write(product_dir, product, body)
        # 只有真正成为成品的那一条记文件路径；其余如实写「已并入」。
        # 全都记成同一路径会让刑部误以为每条都落了这个文件。
        stem = filename.rsplit(".", 1)[0]
        for item in checks:
            if item.note != "待收成一件成品":
                continue
            if safe_stem(item.deliverable) == stem:
                item.path = str(product)
                item.note = "成品已落盘"
            else:
                item.note = "已并入这一件成品"
    result = Execution(number, product_dir, product, checks)
    emit({"type": "execute/final", **result.to_dict()})
    return result
