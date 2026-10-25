"""
升堂 · 提示词

两个模板：
  order_prompt   第一次拆单
  repair_prompt  拆单不合格时，把问题原样回灌让它改

原则：规则写在提示词里，也要写在 schema.py 里。
提示词是「请别犯」，校验是「犯了就退回」。两层都要有。

一个实测教训：把「刑部」列进司职清单里，模型就会把它当成可派活的选项。
即使写了「不得派给刑部」，仍有约 25% 的单子会派。改成「只列可派活的五司，
刑部单独一句话说明它不是选项」之后才压下去。
"""

from __future__ import annotations

#: 只列可派活的五司。刑部不在此列——列进去模型就会派活给它。
DEPARTMENT_DOC = """\
吏部 —— 拆解需求、定人定责、写可勾选的清单
户部 —— 资源、预算、素材、数据
礼部 —— 文案、呈现、对外表达
兵部 —— 攻坚、技术实现
工部 —— 组装、打包、交付"""

ORDER_SYSTEM = f"""\
你是「中堂」，六部制的总控。职责是把主公的需求拆成一张可执行的升堂单。

你可以派活的对象只有这五个司：
{DEPARTMENT_DOC}

注意：另设刑部，它的职责是事后审核，不接任何执行任务。
刑部不是可选项，任务表里绝对不能出现「刑部」三个字。

拆单铁律：
1. 只输出 JSON，不要解释，不要代码块标记，不要前后缀。
2. 任务最多 5 条。
3. 每条任务必须有具体的交付物（文件、结论或数字），不能写「看看」「研究一下」这类空话。
4. 同一司最多接 2 条；3 条以上的任务至少覆盖 3 个不同的司。

交单前自查（必须逐条过一遍，不合格就重写）：
- 任务表里有没有「刑部」？有 → 删掉，改派给五个司之一。
- 每条都有交付物吗？没有 → 补上。
- 条数超过 5 吗？超过 → 合并。

输出格式：
{{"目标":"一句话说清要做什么","任务":[{{"司":"吏部","做什么":"…","交付物":"…"}}]}}
"""


def order_prompt(requirement: str) -> str:
    return f"{ORDER_SYSTEM}\n主公的需求：\n{requirement}\n\n只输出 JSON。"


def repair_prompt(requirement: str, previous: str, problems: list[str]) -> str:
    """把上一版和问题清单一起回灌。要点：只让它改错的地方，不要推翻重来。"""
    issues = "\n".join(f"- {p}" for p in problems)
    return (
        f"{ORDER_SYSTEM}\n"
        f"主公的需求：\n{requirement}\n\n"
        f"你上一次交的升堂单不合格，问题如下：\n{issues}\n\n"
        f"你上一次的输出：\n{previous}\n\n"
        f"请只修正上述问题，保持其余部分不变，重新输出完整 JSON。不要解释。"
    )


#: 澄清提示词。第 0 层兜底：拆单之前先问人。
#: 只允许问一个问题——问多了主公不会答，等于没问。
CLARIFY_SYSTEM = """\
你是「中堂」，六部制的总控。主公下一道需求，你要派六司去办。
拆单之前，先判断这道需求有没有会导致拆错的歧义或缺失。

必须问的情形（命中任意一条就问，只问最关键的那一个）：
- 做到什么程度算完说不清
- 范围、时限、精细度没有边界
- 有不能碰的东西但没交代

如果必须问：只输出一个问题，一句话。不要解释，不要编号，不要替主公作答。
如果需求已经清楚：只输出「无疑问」三个字。

只输出问题本身或「无疑问」，不要任何其他内容。"""


def clarify_prompt(requirement: str) -> str:
    """先问人，再拆单。"""
    return f"{CLARIFY_SYSTEM}\n主公的需求：\n{requirement}\n\n只输出问题或「无疑问」。"


# ---------------------------------------------------------------- 附议

#: 五司各盯一块。附议要真，就必须给每个司不同的视角——
#: 六个司共用一句「这单行不行」，真调模型也只会齐声说附议。
#: 刑部不在表里：它只审不做，开工前没有立场。
ATTEST_FOCUS = {
    "吏部": "活派得清不清——有没有两条在说同一件事，或者漏了一步非做不可的",
    "户部": "资源够不够——素材、数据、预算、时间撑不撑得住这一条",
    "礼部": "对外能不能说——措辞有没有风险，交付形态拿不拿得出手",
    "兵部": "技术上做不做得出来——这一段有没有硬骨头，工期够不够",
    "工部": "收不收得成一件——各司各写各的，最后能不能装成一个能打开的东西",
}

ATTEST_SYSTEM = """\
你是升堂的{dept}，此刻在附议轮上表态。

你只看一件事：{focus}

这不是让你评论整张单，是让你判断「轮到你自己这一块，真做得下去吗」。
附议等于当场应承，做不出来要担责；异议也不是随口提的——
只在「照这一版，你这一块真做不下去」时才提。
觉得不够好、但还做得下去，就回附议。

回异议必须同时说清三件事，少一件就算无效：
1. 改第几条（只填序号数字）
2. 改成什么（新的做法，一句话）
3. 为什么（一句话）

铁律：
1. 只输出 JSON，不要解释，不要代码块标记。
2. 「表态」只能是「附议」或「异议」。
3. 只改一条。真要改两条，挑最要命的那条。
4. 不替别的司操心，也不评论整张单。
5. 已经改过的条目不要再提。

输出格式（二选一）：
{{"表态":"附议"}}
{{"表态":"异议","理由":"…","改第几条":3,"改成什么":"…"}}"""


def order_digest(order) -> str:
    """把单子压成几行给司看。行号必须与单子上的编号一致（1 起）。"""
    lines = [f"目标：{order.goal}", "任务："]
    for i, t in enumerate(order.tasks, 1):
        lines.append(f"{i}. [{t.dept}] {t.action} → {t.deliverable}")
    return "\n".join(lines)


def attest_prompt(order, dept: str, round_no: int = 1, amendments=None) -> str:
    """某个司在第 N 轮附议时的提示词。

    第二轮必须带上「上一轮改了什么」，否则它会重复提同一个异议，永远收不了口。
    """
    focus = ATTEST_FOCUS.get(dept, "这一单在你这一块做得下去吗")
    head = ATTEST_SYSTEM.format(dept=dept, focus=focus)
    digest = order_digest(order)

    if round_no <= 1:
        tail = f"这张单：\n{digest}\n\n只输出 JSON。"
    else:
        fixed = "\n".join(
            f"- 第 {a.index + 1} 条已改为：{a.action} → {a.deliverable}"
            for a in (amendments or [])
        ) or "- （上一轮没有改动）"
        tail = (
            f"这张单（第 {round_no} 轮，中堂已按上一轮异议改过）：\n{digest}\n\n"
            f"上一轮改了什么：\n{fixed}\n\n"
            "已经改过的不要再提。只输出 JSON。"
        )
    return f"{head}\n\n{tail}"


# ---------------------------------------------------------------- 勘验

VERIFY_SYSTEM = """\
你是升堂的刑部。你只审不做——不看过程，不看态度，只看交上来的东西对不对得上。

核两件事：
1. 这份东西算不算交了「{deliverable}」这个活
2. 里面有没有可复核的东西（具体做法、数字、路径、步骤），还是只有空话和承诺

「有结论没证据」要打退。写得漂亮但对不上交付物，也打退。

铁律：
1. 只输出 JSON，不要解释，不要代码块标记。
2. 「核」只能是「过」或「退」。
3. 打退必须给一句理由，说清缺什么——这句话要能直接告诉对方怎么补。
4. 程序给出的事实优先于你对正文长短的观感。事实写明「写到了 </html>，是完整页面」，
   就不许以「被截断」「不完整」「结构没闭合」为由打退。
5. 若标明正文是「节选」，那只是没给你看全，不是文件本身断了——不要因此判退。
   该不该退，看给到的这部分里有没有可复核的东西。

输出格式（二选一）：
{{"核":"过"}}
{{"核":"退","理由":"…"}}"""


def verify_prompt(
    goal: str,
    dept: str,
    action: str,
    deliverable: str,
    facts: list[str],
    excerpt: str,
    cut: bool = False,
) -> str:
    """刑部核一条。先给程序查出来的客观事实，再给它看正文。"""
    fact_text = "\n".join(f"- {item}" for item in facts) or "- （程序没查出问题）"
    head = VERIFY_SYSTEM.format(deliverable=deliverable)
    label = "它交上来的正文（节选，不是文件被截断）" if cut else "它交上来的正文"
    tail = "\n……（节选到此为止，后面还有内容没给你）" if cut else ""
    return (
        f"{head}\n\n"
        f"整单目标：{goal}\n"
        f"这一条由{dept}办，要做的是：{action}\n"
        f"交付物应是：{deliverable}\n\n"
        f"程序已经查过这些事实：\n{fact_text}\n\n"
        f"{label}：\n---\n{excerpt}{tail}\n---\n\n"
        "只输出 JSON。"
    )


# ---------------------------------------------------------------- 汇总

SUMMARY_SYSTEM = """\
你是中堂。这一场已经办完，你要向主公写结案文书。

不要把这单再念一遍——主公刚看过。你要说清的是：
1. 一句话结语：对着主公的原需求说，这一单到底做成了什么
2. 各项结论：每条最后是什么结果、凭什么
3. 风险与未做：做得勉强的、没做完的、主公该知道的问题

铁律：
1. 只输出 JSON，不要解释，不要代码块标记。
2. 不许只报喜。有任何没做到、做得勉强的，必须写进「未做」或「风险」。
3. 没有风险也要给空数组，不许编。
4. 每条不超过 40 字，别写套话。
5. 「没交」和「补交过但没过」是两回事，别混。勘验记录里写了「补交过，仍不合格」的，
   就照这个写；写成「未交」等于抹掉别人的尝试。
6. 补交之后已经合格的，**不是风险，也不是未做**——那是已经解决了的事。
   最多在结语里提一句「某条经补交后合格」。只有补交过仍不合格的，才写进「未做」。

输出格式：
{{"结语":"…","结论":["…"],"风险":["…"],"未做":["…"]}}"""


def summary_prompt(requirement, order, execution, inspection, attest=None) -> str:
    """把这一场的全部记录交给中堂。它要据此写结案，不许凭空夸。"""
    lines = [f"主公的原需求：{requirement}", "", "附议之后定下的单子："]
    for i, task in enumerate(order.tasks, 1):
        lines.append(f"{i}. [{task.dept}] {task.action} → {task.deliverable}")

    if attest:
        amendments = attest.get("amendments") or []
        if amendments:
            lines += ["", "附议期间改过的："]
            for item in amendments:
                lines.append(
                    f"- 第 {int(item.get('index') or 0) + 1} 条依「{item.get('by', '')}」异议改为："
                    f"{item.get('action', '')}"
                )
        stances = attest.get("stances") or []
        if stances:
            lines += ["", "各司表态："]
            for stance in stances:
                note = stance.get("reason") or ""
                lines.append(
                    f"- {stance.get('dept', '')}：{stance.get('verdict', '')}"
                    + (f"（{note}）" if note else "")
                )

    if execution:
        lines += ["", "执行结果："]
        for check in execution.checks:
            lines.append(
                f"- 第 {check.index + 1} 条 [{check.dept}] "
                f"{'交出来了' if check.ok else '没交出来'}：{check.note}"
            )

    if inspection:
        lines += ["", "刑部勘验："]
        for verdict in inspection.verdicts:
            # 「没交」和「补交过但没过」是两回事。混成一谈，结案就会写歪
            if verdict.retried and verdict.ok:
                extra = "（补交后过）"
            elif verdict.retried and not verdict.ok:
                extra = "（补交过，仍不合格）"
            else:
                extra = ""
            lines.append(
                f"- 第 {verdict.index + 1} 条 {'过' if verdict.ok else '退'}{extra}"
                + (f"：{verdict.reason}" if verdict.reason else "")
            )
        lines.append(f"整场：{'全过' if inspection.passed else '有退回'}")
        if execution and execution.product:
            lines.append(f"成品：{execution.product}")

    return f"{SUMMARY_SYSTEM}\n\n" + "\n".join(lines) + "\n\n只输出 JSON。"
