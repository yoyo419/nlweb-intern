"""報告行文紀律 + 報告骨架 registry。

單一定義來源：DR (`build_compose_prompt`) 與 LR (`build_section_compose_prompt`)
兩條產文路徑共用同一份文字，不各寫一套（改一處兩邊同時生效）。

紀律取捨：
- **只給正向舉例**。不放「錯誤示範」的完整句子——弱模型會把反例字串鸚鵡學舌
  進正文（既有 v8 regression 教訓）。要擋的字改以「禁用詞表」形式給，
  詞表同時被 tests/unit/reasoning/test_prose_discipline.py 當機械斷言用。
- 報告骨架（shape）是**使用者可選**的，不寫死一種：記者要倒三角，
  永續報告書撰寫者要的是「外界關切 × 公司作為」對照表。
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# 禁用詞表 —— prompt 與測試共用同一份（改詞表 = 同時改 prompt 與斷言）
# ---------------------------------------------------------------------------

# 軟墊詞 / 空洞副詞：句子拿掉語意不變，只是拖長
BANNED_HEDGE_WORDS: List[str] = [
    "其實", "某種程度上", "值得注意的是", "坦白說", "不得不說",
    "眾所周知", "不言而喻", "無庸置疑",
    "事實上", "基本上", "值得一提的是", "毋庸置疑",
]

# 導覽句 / 碎形摘要 / 標示結論：讀者看得到自己在哪，不需要被預告與總結。
# 「如前所述」「前文提到」一併列入：回頭指前文＝把講過的事再講一次。
BANNED_NAVIGATION_WORDS: List[str] = [
    "總結來說", "綜上所述", "綜合以上", "講到最後", "接下來將說明",
    "以下將探討", "本節將介紹", "首先我們來看",
    "總而言之", "簡而言之", "本章將", "如前所述", "如上所述", "前文提到",
]

# 假條件句：沒有排除任何可能性
BANNED_FAKE_CONDITIONALS: List[str] = [
    "取決於", "視情況而定", "依需求不同", "因人而異",
    "因地制宜", "視個案而定",
]

# 模糊量詞：拿來假裝有數字。證據沒有數字就誠實說沒有
BANNED_VAGUE_QUANTIFIERS: List[str] = [
    "相當數量", "為數不少", "數量龐大", "面積廣大", "不在少數",
    "大幅提升", "大幅增加", "顯著提升", "與日俱增",
]

# 以下四類來自 CEO 中文寫作規格（taiwan-news-ai-search
# memory/reference_chinese_writing_spec.md）的「機械層」掃描項。

# 稻草人句型：警告讀者別做一件沒人打算做錯的事。要寫的是這件事的用途
BANNED_STRAWMAN: List[str] = [
    "不這樣做", "若忽略", "一旦忽略", "若忽視", "少了這一步",
]

# 替讀者數數：預告「有幾點」，數量本身不是資訊時只是多一句
BANNED_COUNTING_FOR_READER: List[str] = [
    "以下幾點", "以下幾個", "有以下", "下列幾", "幾個面向",
    "三大面向", "兩大面向", "可分為兩", "可分為三",
]

# 無主詞斷言：說法沒有掛信源。誰說的就寫誰，找不到人就是證據不足
BANNED_SUBJECTLESS_CLAIMS: List[str] = [
    "一般認為", "普遍認為", "多數人認為", "有人認為", "有觀點認為",
    "有專家指出", "有學者指出", "外界普遍", "各界看法不一", "據了解", "據悉",
]

# 替讀者的專業下定義：收攏語氣，替讀者決定他的工作長什麼樣
BANNED_PRESUMING_READER: List[str] = [
    "不外乎", "就那幾類", "大家都知道", "眾所皆知",
]

BANNED_WORD_CATEGORIES: List[List[str]] = [
    BANNED_HEDGE_WORDS,
    BANNED_NAVIGATION_WORDS,
    BANNED_FAKE_CONDITIONALS,
    BANNED_VAGUE_QUANTIFIERS,
    BANNED_STRAWMAN,
    BANNED_COUNTING_FOR_READER,
    BANNED_SUBJECTLESS_CLAIMS,
    BANNED_PRESUMING_READER,
]

ALL_BANNED_WORDS: List[str] = [w for cat in BANNED_WORD_CATEGORIES for w in cat]


def _fmt_words(words: List[str]) -> str:
    return "、".join(words)


PROSE_DISCIPLINE_BLOCK = f"""
---

## 行文紀律（每一句都要做工）

文筆乾淨不等於不是廢話。以下每一條砍的都是「讀起來通順但不做工」的句子。

**1. 讀完這句，讀者的下一個動作有沒有改變？** 沒有就不要寫。只描述現象而不導向
   判斷或動作的句子一律刪掉，或改寫成讓判斷接得上。

**2. 抽象詞後面立刻接具體物。** 抽象詞是索引不是結論——寫「影響」就說是哪一個
   影響，寫「原因」就說是哪一個原因。承認限制之後接怎麼處理，不要停在限制本身。

**3. 不硬塞數字。** 這裡需不需要數字先問一次；不需要就改成不帶數字的寫法。
   證據裡沒有的門檻、比例、金額**一律不得發明**——寧可寫「本研究資料未涵蓋此項」。

**4. 禁用詞表**（出現即視為不合格，改寫成直說）：
   - 軟墊詞（拿掉語意不變）：{_fmt_words(BANNED_HEDGE_WORDS)}
   - 導覽與總結句（小標自己會導覽；回頭指前文就是在重講）：{_fmt_words(BANNED_NAVIGATION_WORDS)}
   - 假條件句（沒有排除任何可能）：{_fmt_words(BANNED_FAKE_CONDITIONALS)}
   - 模糊量詞（證據有數字就寫數字，沒有就說沒有）：{_fmt_words(BANNED_VAGUE_QUANTIFIERS)}
   - 稻草人句型（沒人打算做錯的事不必警告，直接寫它的用途）：{_fmt_words(BANNED_STRAWMAN)}
   - 替讀者數數（數量本身是資訊才講數量，否則直接列）：{_fmt_words(BANNED_COUNTING_FOR_READER)}
   - 無主詞斷言（論述掛信源，寫出是誰說的；來源沒寫是誰就是證據不足）：{_fmt_words(BANNED_SUBJECTLESS_CLAIMS)}
   - 替讀者下定義（不替讀者決定他的工作或處境長什麼樣）：{_fmt_words(BANNED_PRESUMING_READER)}

**5. 標題給理由，不只給動作。** 把該節最有力的資訊提到標題。標題用陳述句，
   不用否定句（讀者從否定句讀不出該改用什麼）。

**6. 一件事講完就走。** 同一個論點不要換十種說法重講；一段話的第二三句不要
   只是把第一句換句話說；同一個比喻不要用超過兩次。

**7. 句子縮短，讓標點分擔。** 四個子句以上的長句拆開。連續三四句長度與結構
   一樣就是機器味，長句後面接短句。

**8. 虛詞瘦身。** 的、也、將、就、在、便、了、於、一、而——拿掉句子還成立就拿掉。

**9. 中英不混雜。** 該詞在臺灣本來就用英文（API、ERP、POS、RE100、CBAM）才保留原文，
   其餘一律用中文說法。

**10. 一律繁體中文。**

正向示範（照這個密度寫）：
- 「環評門檻加嚴後，六大環團在協商前一天發出聯合聲明，理由是擴大禁建區會誤傷
  已完成饋線登記的案場。」
- 「正因為讀不出來，這一步不能靠人工核對，要給名詞表讓機器比對。」
"""


# ---------------------------------------------------------------------------
# 報告骨架 registry
# ---------------------------------------------------------------------------

SHAPE_NEWS = "news_inverted_pyramid"
SHAPE_RESEARCH = "research_report"
SHAPE_ESG_MATRIX = "esg_matrix"

DEFAULT_REPORT_SHAPE = SHAPE_NEWS

_SHAPE_SPECS: Dict[str, str] = {
    SHAPE_NEWS: """
## 報告骨架：新聞寫作（倒三角）

**第一段是導言（lead）**，兩到三句寫完：誰、做了什麼、結果是什麼、憑什麼這樣說。
結論放在第一句，不要鋪陳背景再收斂。

**第二段是核心段（nut graf）**：為什麼這件事現在成立、這份報告最倚重哪幾篇來源。

**其後倒三角遞減**：先寫最重要的發現，再寫脈絡與成因，再寫對立說法，最後寫限制。
讀者只讀前三分之一就離開時，帶走的必須是最重要的那幾件事。

**對立說法要指名**：誰主張什麼、根據是什麼，兩方都給版面，不用一句籠統的分歧帶過。

**版面規則**：章內小標一律用 markdown `###` 開頭，不要寫成裸文字行（裸文字行匯出後
沒有落點，整章讀起來是一片流水帳）。正文第一行直接開始寫，不要把章名再寫一次
（章名由系統排版，重複會變成標題出現兩次）。
""",
    SHAPE_RESEARCH: """
## 報告骨架：研究報告

## 研究摘要

[核心發現 - 2-3 句話]

### 官方/主流觀點

[官方/主流來源的資訊，附來源標註]

### 輿情觀察

> ⚠️ 以下內容來自社群討論，尚未經官方證實

[社群/網路來源的資訊，加註警語]

### 觀點落差

[若有矛盾，明確列出]

### 建議後續關注

[可追蹤的發展方向]
""",
    SHAPE_ESG_MATRIX: """
## 報告骨架：關切事項對照

把「外界在意的事」與「已經做過的事」放在同一個畫面上對齊，用 markdown 表格：

| 外界關切（誰提出、何時） | 對應作為（做了什麼、時程） | 佐證來源 | 落差 |
| --- | --- | --- | --- |

**紀律**：
- 每一列的「外界關切」與「對應作為」都必須各自有來源支撐；只有一邊有來源時，
  另一邊寫「本研究資料未涵蓋」，不可用推測填滿。
- 表格之後用文字說明落差最大的兩到三列，說明落差是什麼、缺哪一筆資料才能判斷。
""",
}

# 使用者輸入 → 正規 shape key 的別名（Stage 4 free text 容錯）
_SHAPE_ALIASES: Dict[str, str] = {
    "news": SHAPE_NEWS,
    "新聞": SHAPE_NEWS,
    "新聞稿": SHAPE_NEWS,
    "倒三角": SHAPE_NEWS,
    "inverted_pyramid": SHAPE_NEWS,
    "research": SHAPE_RESEARCH,
    "研究報告": SHAPE_RESEARCH,
    "discovery": SHAPE_RESEARCH,
    "esg": SHAPE_ESG_MATRIX,
    "永續": SHAPE_ESG_MATRIX,
    "對照表": SHAPE_ESG_MATRIX,
}


def normalize_report_shape(raw: str) -> str:
    """使用者/Stage 4 給的 shape 字串 → registry key。

    不認得的值**不 silent fallback**：呼叫端負責 log。此處回 DEFAULT 並由
    `is_known_report_shape` 讓呼叫端能判斷是否走了降級。
    """
    key = (raw or "").strip().lower()
    if key in _SHAPE_SPECS:
        return key
    if key in _SHAPE_ALIASES:
        return _SHAPE_ALIASES[key]
    return DEFAULT_REPORT_SHAPE


def is_known_report_shape(raw: str) -> bool:
    key = (raw or "").strip().lower()
    return key in _SHAPE_SPECS or key in _SHAPE_ALIASES


def build_shape_block(shape: str) -> str:
    """回傳該 shape 的結構規格區塊。未知 shape → 預設骨架（呼叫端須 log）。"""
    return _SHAPE_SPECS[normalize_report_shape(shape)]


def report_shapes() -> List[str]:
    return list(_SHAPE_SPECS.keys())


# ---------------------------------------------------------------------------
# 可回溯 / 缺口 / 並排欄位 —— 三個 output 契約的 prompt 說明
# ---------------------------------------------------------------------------

TRACEABILITY_BLOCK = """
---

## 可回溯到「句」（每個引用要附來源逐字片段）

每個 `{cite:N}` 都要在 `citations` 陣列對應一筆 `{"evidence_id": N, "quote": "..."}`。

`quote` 規則：
- 從該來源摘要裡**逐字複製** 10 到 40 個字，是真正支撐這句話的那一段原文。
- 逐字，不改寫、不濃縮、不補標點、不加引號、不接省略號。
- 系統會檢查 quote 是否真的出現在該來源摘要裡；對不上就丟棄，讀者會失去跳轉能力。
- 該來源摘要為空、或找不到能逐字對應的片段 → `quote` 留空字串，不要硬湊。
"""

OPEN_GAPS_BLOCK = """
---

## 缺什麼就說缺什麼

寫完本章後，把「本章答不了的問題」填進 `open_gaps`。判準是**這一章的論述真的
需要、而現有資料拿不到**的那一筆，不是泛泛的延伸題目。

每筆包含：
- `question`：具體到可以拿去查的一個問題。
- `why_missing`：為什麼現有資料答不了（沒有這個年份、沒有這個機關的版本、
  只有二手轉述沒有原文…）。
- `where_to_look`：`internal`（站內語料再查）/ `web`（外部公開資料）/
  `manual`（要人去問或調閱）三選一。

紀律：**不要**用現有的某一段來充數說「拿這段也可以」。答不了就寫答不了。
沒有缺口就給空陣列，不要為了填欄位而編。
"""

COMPARISON_FIELDS_BLOCK_TEMPLATE = """
---

## 並排比較的欄位必須一致

本章要並排比較多個案例。先固定欄位，每個案例都填同一組：

{fields}

用 markdown 表格呈現，每個案例一列。某案例的某個欄位查不到 → 該格寫「查無」，
**不可**換一組欄位講第二個案例（欄位不一致就比不出來）。
"""


def build_comparison_fields_block(fields: List[str]) -> str:
    if not fields:
        return ""
    listed = "\n".join(f"- {f}" for f in fields)
    return COMPARISON_FIELDS_BLOCK_TEMPLATE.format(fields=listed)


# ---------------------------------------------------------------------------
# 產物層 lint —— prompt 擋不住的由 code 抓出來
# ---------------------------------------------------------------------------

def lint_prose(text, context_chars=18):
    """掃產出的散文有沒有禁用詞，回 [(詞, 上下文)]。

    存在理由：純 prompt 紀律鎖不住模型（實測 gpt-5.1 仍會漏「其實」）。
    這支是產物層的機械防線，replay harness、Stage 6 與 unit test 共用同一份，
    命中不改稿也不吞掉——列出來讓人看得到改哪一句。
    """
    hits = []
    for word in ALL_BANNED_WORDS:
        start = 0
        while True:
            i = (text or "").find(word, start)
            if i < 0:
                break
            lo = max(0, i - context_chars)
            hi = min(len(text), i + len(word) + context_chars)
            hits.append((word, text[lo:hi]))
            start = i + len(word)
    return hits


def strip_duplicate_title(title, content):
    """剝掉章節正文開頭那行重複的章名。

    章名由系統排 `## `，模型常在正文再寫一次，匯出後標題連著出現兩次。
    prompt 已經講了（見骨架的版面規則），這裡是擋不住時的 code 層防線。
    只比對第一行，且要求去掉 markdown 標記後與章名完全相同才剝 —— 不做模糊比對，
    避免誤刪一句開場白。
    """
    if not content or not title:
        return content
    lines = content.split("\n")
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return content
    first = lines[idx].strip().lstrip("#").strip().strip("*").strip()
    if first == title.strip():
        rest = lines[idx + 1:]
        while rest and not rest[0].strip():
            rest.pop(0)
        return "\n".join(rest)
    return content

