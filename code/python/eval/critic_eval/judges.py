"""
The three adversarial judges for the critic auto-eval harness.

Each judge inspects (query, sources, draft) from ONE angle and its job is to
FIND FAULTS the critic should have caught — not to give a score. A judge that
finds nothing is what lets the critic "pass" that dimension.

Design (see docs/specs/critic-eval-plan.md §2):
- 評審 1 grounding    : is every concrete claim actually supported by a source?
- 評審 2 fabrication  : are the cited sources real / matching? any invented facts?
- 評審 3 logic        : does the conclusion follow from the premises, no jumps?

Prompt language: Traditional Chinese (matches reasoning/prompts/*.py).
Code comments: English.

Each builder returns a prompt string; the runner feeds it to the LLM with
response_schema=JudgeVerdict. The optional critic_verdict lets a judge focus on
what the critic PASSed; leave it None during Q3 calibration (judge vs human on
the draft alone).
"""

from typing import List, Optional
from core.prompts import generate_boundary_token, wrap_content_with_boundary
from eval.critic_eval.schema import Source


def _format_sources(sources: List[Source]) -> str:
    """Render sources as a numbered, boundary-isolated block (SEC-6)."""
    lines = [f"[{s.id}] {s.text}" for s in sources]
    body = "\n".join(lines) if lines else "（無來源）"
    boundary = generate_boundary_token()
    return wrap_content_with_boundary(body, boundary)


def _shared_header(query: str, sources: List[Source], draft: str,
                   critic_verdict: Optional[str]) -> str:
    """Common context block shared by all three judges."""
    isolated_draft = wrap_content_with_boundary(draft, generate_boundary_token())
    verdict_line = ""
    if critic_verdict:
        verdict_line = (
            f"\n## Critic 對這篇草稿的判定\n\n"
            f"Critic 給了「{critic_verdict}」。你的工作是檢查這個判定有沒有放過該抓的問題。\n"
        )
    return f"""## 使用者的問題

{query}

## 可用來源（草稿只能根據這些內容，不得引用來源以外的資訊）

{_format_sources(sources)}

## 待審草稿

{isolated_draft}
{verdict_line}"""


# ---------------------------------------------------------------------------
# 評審 1 — grounding：有沒有根據
# ---------------------------------------------------------------------------
def build_grounding_judge_prompt(query: str, sources: List[Source], draft: str,
                                 critic_verdict: Optional[str] = None) -> str:
    role = """你是嚴格的「根據性審查員」。你的唯一任務是找出草稿裡**缺乏來源根據**的具體說法。

判準：草稿中每一個具體主張（數字、日期、人名、機構、事件、因果、比較、評價）都必須能在「可用來源」裡對得上。只要一句話講了來源沒有的具體內容，就是問題。

**這是你的守備範圍（很重要）**：**評價性／程度性的描述**——例如「表現穩健」「獲利能力強勁」「創下歷史新高」「前景看好」「競爭力強」——只要來源沒有寫、或來源撐不起這個程度，就是**根據性問題，屬於你**。這類「加了一個沒根據的評價詞」是你要抓的重點，不要以為它是別人的事。

重要立場：
- 你的預設是**挑毛病**，不是打分數。逐句掃描，寧可嚴格。
- 每找到一個問題，必須舉出**具體證據**：抄下草稿原句、指出去哪條來源查、來源裡實際上寫了什麼（或根本沒寫）。
- 只抓「根據性」問題（來源沒有卻寫了，含沒根據的評價詞）。引用編號對不對得上（假引用）、以及推論跳步，不是你的守備範圍。
- 真的每句都有根據，才回 found_issue=false。
"""
    tail = """
## 輸出

回傳符合 JudgeVerdict schema 的 JSON：dimension 固定填 "grounding"；把每個沒根據的說法放進 issues（quote=草稿原句，reason=為何沒根據，evidence_check=對照了哪條來源、來源實際內容）。
"""
    return role + "\n" + _shared_header(query, sources, draft, critic_verdict) + tail


# ---------------------------------------------------------------------------
# 評審 2 — fabrication：有沒有編造 / 假引用
# ---------------------------------------------------------------------------
def build_fabrication_judge_prompt(query: str, sources: List[Source], draft: str,
                                   critic_verdict: Optional[str] = None) -> str:
    role = """你是嚴格的「編造與引用查核員」。你的唯一任務是找出草稿裡**編造的內容**與**對不上的引用**。

要抓兩類問題：
1. 假引用：草稿標了 [N]，但編號 N 的來源根本不存在，或內容跟該句講的對不上（張冠李戴）。
2. 憑空編造：草稿講了一個**具體事實**（明確的數字、被引號框起來的引述、可查證的事件／紀錄），但**任何**一條來源都沒有這個內容。

**守備範圍界線（很重要，避免和「根據性」評審撞在一起）**：
- 你抓的是「無中生有的具體事實」——憑空冒出來的數字、假的引述、捏造的事件或紀錄。
- 若只是**沒根據的評價詞或程度誇大**（如「表現穩健」「創下歷史新高」而來源沒寫），那是「根據性」評審的守備範圍，**不要**由你重複開火；除非它偽裝成一個可查證的具體事實（例如捏造一個確切數字、或標了對不上的引用編號）才算你的。

重要立場：
- 你的預設是**挑毛病**，不是打分數。
- 每找到一個問題，必須舉證：抄下草稿原句與它標的引用編號、指出該編號來源實際寫什麼、說明為何對不上或為何是憑空捏造。
- 只抓「編造／假引用」問題。單純措辭抽象、沒根據的評價詞、邏輯跳步都不是你的守備範圍。
- 引用都真實對得上、沒有捏造，才回 found_issue=false。
"""
    tail = """
## 輸出

回傳符合 JudgeVerdict schema 的 JSON：dimension 固定填 "fabrication"；每個問題放進 issues（quote=草稿原句，reason=假引用還是編造、為什麼，evidence_check=該引用編號來源的實際內容）。
"""
    return role + "\n" + _shared_header(query, sources, draft, critic_verdict) + tail


# ---------------------------------------------------------------------------
# 評審 3 — logic：邏輯有沒有跳步
# ---------------------------------------------------------------------------
def build_logic_judge_prompt(query: str, sources: List[Source], draft: str,
                             critic_verdict: Optional[str] = None) -> str:
    role = """你是嚴格的「邏輯審查員」。你的唯一任務是找出草稿裡**推論跳步**的地方。

判準：草稿裡有一個**明確的推論結構**（用「因此／所以／導致／顯示出／可見」把一個前提句推到一個更強的結論），而該推論缺了中間環節時才算問題。要抓的是：
- 前提只支持 A，卻直接跳到更強的結論 B（過度推論）。
- 因果宣稱缺中間環節（「因為 X 所以 Y」但沒交代 X 如何導致 Y）。
- 以偏概全、把個案講成通則、把相關講成因果。

**守備範圍界線（很重要，避免越界誤抓）**：
- 單純的**評價性／程度性形容詞或修飾語**（如「表現穩健」「獲利能力強勁」「創歷史新高」「前景看好」）本身**不是**邏輯跳步——那是「有沒有根據」評審的守備範圍。
- 一句話只是**加了一個沒根據的評價詞**、但**沒有**前提→結論的推理結構（沒有「因此／所以」這類推論動作）時，**不要**當成跳步、**不要**開火。
- 你只在草稿**真的做了一次推論動作**、而那一步接不起來時才報。

重要立場：
- 你的預設是**挑毛病**，不是打分數。
- 每找到一個跳步，必須舉證：抄下草稿的結論句、指出它依賴哪個前提、說明中間缺了什麼環節。
- 只抓「邏輯跳步」。內容有沒有根據、引用真不真、評價詞有沒有依據，都不是你的守備範圍（那是另外兩位評審的事）。
- 推論都接得起來、或草稿根本沒有推論結構，才回 found_issue=false。
"""
    tail = """
## 輸出

回傳符合 JudgeVerdict schema 的 JSON：dimension 固定填 "logic"；每個跳步放進 issues（quote=草稿結論句，reason=缺了什麼推論環節，evidence_check=它依賴的前提／來源說到哪為止）。
"""
    return role + "\n" + _shared_header(query, sources, draft, critic_verdict) + tail


# Registry the runner iterates over. dimension -> builder.
JUDGES = {
    "grounding": build_grounding_judge_prompt,
    "fabrication": build_fabrication_judge_prompt,
    "logic": build_logic_judge_prompt,
}
