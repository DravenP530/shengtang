"""
升堂 · 中堂（拆单）

这一版只实现状态机的第一态：拆单。
但兜底逻辑是完整的——因为它决定了整条流水线可不可信。

兜底分五层，从外到内：
  0. 澄清兜底  需求有歧义 → 先停下来问主公，把回答回灌进提示词
               （模型说无疑问、调用失败、主公跳过，都按原需求继续，绝不堵流程）
  1. 解析兜底  模型输出不是 JSON → 当成「不合格」，进重试，不让异常往上冒
  2. 校验兜底  结构化但违规（派活给刑部、没交付物、司重复）→ 把问题回灌让它改
  3. 重试兜底  最多 N 轮；每轮都换新 session，避免上一轮的错被记住
  4. 终局兜底  全部失败 → 返回 None 并抛「拆单失败」，由上层显示「请示主公」
               ——绝不用一份不合格的单子往下走

第 0 层和第 4 层是一体两面：宁可停下来问人，也不要假装拆好了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import prompts
from .llm import FastPathUnavailable, LLM
from .schema import Order, ParseError, parse_order, validate

#: 事件回调签名：收到一个 dict，前端据此演朝堂的某一拍。
EventSink = Callable[[dict], None]


class OrderRejected(RuntimeError):
    """重试用尽仍未得到合格的升堂单。"""

    def __init__(self, attempts: list["Attempt"]) -> None:
        self.attempts = attempts
        last = attempts[-1].problems if attempts else ["未知原因"]
        super().__init__("拆单失败：" + "；".join(last))


@dataclass
class Attempt:
    """一次拆单尝试的完整记录，出问题时能复盘。"""

    index: int
    raw: str = ""
    order: Order | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def parse_question(raw: str) -> str | None:
    """从澄清输出里抠出问题。抠不出来（无疑问／跑题／长篇大论）就返回 None。

    宁可漏问，不可错问：返回 None 只是少一次交互，
    返回一句垃圾问题却会浪费主公的时间，还显得系统不靠谱。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        # 剥掉围栏：首行可能是 ```json，末行是 ```，都不是问题
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    for line in text.splitlines():
        q = line.strip()
        # 去掉「1. 」「- 」这类列表头，但不动以数字开头的问题本身
        if len(q) > 2 and q[0] in "-*" and q[1] == " ":
            q = q[2:].strip()
        elif len(q) > 3 and q[0].isdigit() and q[1] in ".、)" and q[2] == " ":
            q = q[3:].strip()
        q = q.strip("「」『』“”\"' 　").strip()
        if not q:
            continue
        if q.lower() in ("json", "json5", "text"):
            continue
        if "无疑问" in q or q.startswith(("{", "[")):
            return None
        if len(q) > 150:
            return None
        return q
    return None


class Zhongtang:
    """中堂。目前只负责拆单，后续会扩到收附议、汇总。"""

    def __init__(
        self,
        llm: LLM,
        max_attempts: int = 3,
        on_event: EventSink | None = None,
        fast: bool = True,
    ) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self.on_event = on_event or (lambda _e: None)
        self.fast = fast
        self.attempts: list[Attempt] = []

    # ------------------------------------------------------------ 内部

    def _emit(self, type_: str, **data: object) -> None:
        self.on_event({"type": type_, **data})

    def _generate(self, prompt: str) -> str:
        """拆单是纯生成任务：默认走裸调，快得多。配置不全时退回 agent 回合。"""
        if self.fast:
            try:
                return self.llm.quick(prompt)
            except FastPathUnavailable as exc:
                self._emit("order/fast-unavailable", reason=str(exc))
        return self.llm.chat(prompt)

    def _try_once(self, index: int, prompt: str) -> Attempt:
        attempt = Attempt(index=index)
        self._emit("order/attempt", index=index)

        try:
            attempt.raw = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001 - 模型层任何异常都算这一轮失败
            attempt.problems = [f"模型调用失败：{exc}"]
            self._emit("order/rejected", index=index, problems=attempt.problems)
            return attempt

        try:
            attempt.order = parse_order(attempt.raw)
        except ParseError as exc:
            attempt.problems = [f"输出不是合法 JSON：{exc}"]
            self._emit("order/rejected", index=index, problems=attempt.problems)
            return attempt

        attempt.problems = validate(attempt.order)
        if attempt.problems:
            self._emit("order/rejected", index=index, problems=attempt.problems)
        else:
            self._emit("order/accepted", index=index, order=attempt.order.to_dict())
        return attempt

    # ------------------------------------------------------------ 对外

    def issue_order(
        self,
        requirement: str,
        clarifier: Callable[[str], str] | None = None,
    ) -> Order:
        """拆单。成功返回 Order，失败抛 OrderRejected。

        clarifier 是可选的「问人」通道：传了它，拆单前先问模型
        「这道需求有什么非问不可的问题」，有就问主公，把回答回灌进提示词。
        不传（CLI、单元测试）就按原需求直接拆。
        """
        requirement = (requirement or "").strip()
        if not requirement:
            raise ValueError("需求不能为空")

        if clarifier is not None:
            requirement = self._clarify(requirement, clarifier)

        self.attempts = []
        previous_raw = ""

        for i in range(1, self.max_attempts + 1):
            if i == 1:
                prompt = prompts.order_prompt(requirement)
            else:
                prompt = prompts.repair_prompt(
                    requirement,
                    previous_raw,
                    self.attempts[-1].problems,
                )

            attempt = self._try_once(i, prompt)
            self.attempts.append(attempt)

            if attempt.ok and attempt.order is not None:
                return attempt.order

            previous_raw = attempt.raw

        self._emit("order/failed", attempts=len(self.attempts))
        raise OrderRejected(self.attempts)

    def _clarify(self, requirement: str, clarifier: Callable[[str], str]) -> str:
        """第 0 层兜底：先问人，再拆单。

        澄清是增益，不是门槛——模型说「无疑问」、调用炸了、主公跳过，
        全都按原需求继续拆。宁可不问，也不能把流程卡死。
        """
        try:
            raw = self._generate(prompts.clarify_prompt(requirement))
            question = parse_question(raw)
            if not question:
                return requirement

            answer = (clarifier(question) or "").strip()
            if not answer:
                return requirement

            self._emit("order/clarified", question=question, answer=answer)
            return f"{requirement}\n【主公补充】{answer}"
        except Exception:  # noqa: BLE001 - 澄清失败绝不堵拆单
            return requirement
