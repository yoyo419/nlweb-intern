"""報告頭尾的兩張表：來源倚重統計，以及這份報告答不了什麼。

為什麼要有這支：既有 references 清單（orchestrator._build_references_block）是
平鋪的，讀者看不出這份報告到底靠哪幾篇撐起來。這裡把「被引用次數」與
「跨章覆蓋數」算出來，讓報告開頭能先講清楚它倚重誰。

計數來源優先序（**不 silent fallback**，回傳 `counting_mode` 讓呼叫端可 log）：
1. `section["citation_counts"]` —— 由 orchestrator 在 render `{cite:N}` placeholder
   時就地統計，是真正的「出現次數」。
2. `section["sources_used"]` —— 舊 session / citation_format="none" / guard 失敗章
   沒有 counts，退回「每章每來源記 1 次」。此時次數等於章數，報告會標明。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

COUNTING_MODE_EXACT = "exact"          # 由 {cite:N} 出現次數統計
COUNTING_MODE_PER_SECTION = "per_section"  # 退回 sources_used，每章每來源記 1
COUNTING_MODE_MIXED = "mixed"          # 兩者混用（部分章有 counts，部分沒有）

_CITE_PLACEHOLDER_RE = re.compile(r"\{cite:(\d+)\}")


def count_cite_placeholders(content: str) -> Dict[int, int]:
    """數 section_content 裡每個 `{cite:N}` 出現幾次。

    必須在 citation render **之前**呼叫——render 之後 placeholder 已變成
    `[1]` / 上標 / APA 字串，數不回來。
    """
    counts: Dict[int, int] = {}
    for m in _CITE_PLACEHOLDER_RE.finditer(content or ""):
        eid = int(m.group(1))
        if eid == 0:          # {cite:0} 是「無引用」marker，不計入來源統計
            continue
        counts[eid] = counts.get(eid, 0) + 1
    return counts


def _normalize_counts(raw: Any) -> Dict[int, int]:
    """written_sections 經 JSON round-trip 後 key 會變 str，統一轉回 int。"""
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in raw.items():
        try:
            eid = int(k)
            n = int(v)
        except (TypeError, ValueError):
            continue
        if eid != 0 and n > 0:
            out[eid] = out.get(eid, 0) + n
    return out


def prior_source_chapters(written_sections: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    """evidence_id → 引用過該來源的章號（0-based，依 written_sections 順序）。

    給 writer 下一章用：前面章節已交代過的來源，本章不要從頭再講一次。
    與 compute_source_reliance 同一套判準：有 citation_counts 用它，沒有退回 sources_used。
    """
    out: Dict[int, List[int]] = {}
    for idx, section in enumerate(written_sections or []):
        eids = set(_normalize_counts(section.get("citation_counts")))
        if not eids:
            for eid in section.get("sources_used") or []:
                try:
                    eid_int = int(eid)
                except (TypeError, ValueError):
                    continue
                if eid_int != 0:
                    eids.add(eid_int)
        for eid in sorted(eids):
            out.setdefault(eid, []).append(idx)
    return out


def compute_source_reliance(
    written_sections: List[Dict[str, Any]],
    evidence_pool: Optional[Dict[int, Any]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """統計每個 evidence 被引用幾次、撐住哪幾章。

    Args:
        written_sections: state.written_sections（dict list）。
        evidence_pool: eid -> EvidencePoolEntry（或 dict）。用來補標題/網域；
            None 時只回數字。

    Returns:
        (rows, counting_mode)
        rows 依「引用次數 desc → 跨章數 desc → eid asc」排序，每筆：
            evidence_id, citation_count, section_count, section_titles,
            title, source_domain, url, in_pool
    """
    exact_sections = 0
    fallback_sections = 0
    counts: Dict[int, int] = {}
    sections_by_eid: Dict[int, List[str]] = {}

    for idx, section in enumerate(written_sections or []):
        title = section.get("title") or f"第 {idx + 1} 章"
        per_section = _normalize_counts(section.get("citation_counts"))
        if per_section:
            exact_sections += 1
        else:
            # 退回 sources_used：每章每來源記 1 次
            per_section = {}
            for eid in section.get("sources_used") or []:
                try:
                    eid_int = int(eid)
                except (TypeError, ValueError):
                    continue
                if eid_int != 0:
                    per_section[eid_int] = 1
            if per_section:
                fallback_sections += 1
        for eid, n in per_section.items():
            counts[eid] = counts.get(eid, 0) + n
            sections_by_eid.setdefault(eid, [])
            if title not in sections_by_eid[eid]:
                sections_by_eid[eid].append(title)

    if exact_sections and fallback_sections:
        mode = COUNTING_MODE_MIXED
    elif fallback_sections:
        mode = COUNTING_MODE_PER_SECTION
    else:
        mode = COUNTING_MODE_EXACT

    rows: List[Dict[str, Any]] = []
    for eid, n in counts.items():
        entry = (evidence_pool or {}).get(eid)
        rows.append({
            "evidence_id": eid,
            "citation_count": n,
            "section_count": len(sections_by_eid.get(eid, [])),
            "section_titles": sections_by_eid.get(eid, []),
            "title": _attr(entry, "title", ""),
            "source_domain": _attr(entry, "source_domain", ""),
            "url": _attr(entry, "url", ""),
            "in_pool": entry is not None,
        })

    rows.sort(key=lambda r: (-r["citation_count"], -r["section_count"], r["evidence_id"]))
    return rows, mode


def _attr(entry: Any, name: str, default: str = "") -> str:
    if entry is None:
        return default
    if isinstance(entry, dict):
        return entry.get(name) or default
    return getattr(entry, name, None) or default


def _format_chapters(titles: List[str], order: List[str]) -> str:
    """章節清單 → 「第 1、3 章（共 2 章）」。整串章名會把表格撐到讀不動。"""
    if not titles:
        return "—"
    nums = []
    for t in titles:
        if t in order:
            nums.append(order.index(t) + 1)
    if not nums:
        return f"共 {len(titles)} 章"
    nums.sort()
    listed = "、".join(str(n) for n in nums[:6])
    tail = "…" if len(nums) > 6 else ""
    return f"第 {listed}{tail} 章（共 {len(nums)} 章）"


def render_reliance_block(
    rows: List[Dict[str, Any]],
    counting_mode: str,
    top_n: int = 5,
    chapter_order: Optional[List[str]] = None,
) -> str:
    """報告開頭的「本報告最倚重的來源」區塊（markdown）。

    chapter_order: 章節標題依報告順序排列，用來把章名換成章號。
    rows 為空 → 回空字串（呼叫端跳過此區塊）。
    """
    if not rows:
        return ""
    chapter_order = chapter_order or []

    top = rows[:top_n]
    lines: List[str] = ["", "## 本報告最倚重的來源", ""]
    lines.append("| 來源 | 引用次數 | 撐住哪幾章 |")
    lines.append("| --- | --- | --- |")
    for r in top:
        title = r["title"] or f"（來源 {r['evidence_id']} 已不在資料池）"
        domain = f"（{r['source_domain']}）" if r["source_domain"] else ""
        link = f"[[{r['evidence_id']}] {title}]({r['url']})" if r["url"] else f"[{r['evidence_id']}] {title}"
        # 章節名整串列出會把表格撐到讀不動（實測四章就爆寬）；改列章號 + 章數。
        chapters = _format_chapters(r["section_titles"], chapter_order)
        lines.append(f"| {link}{domain} | {r['citation_count']} | {chapters} |")

    lines.append("")
    total_cites = sum(r["citation_count"] for r in rows)
    lines.append(
        f"全文共 {total_cites} 次引用，落在 {len(rows)} 個來源上；"
        f"前 {len(top)} 個來源佔 {sum(r['citation_count'] for r in top)} 次。"
    )
    if counting_mode == COUNTING_MODE_PER_SECTION:
        lines.append(
            "（本次為每章每來源計 1 次的估計值——這份報告的章節沒有留下逐處引用位置。）"
        )
    elif counting_mode == COUNTING_MODE_MIXED:
        lines.append(
            "（部分章節沒有逐處引用位置，該部分以每章每來源計 1 次估算。）"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 逐字引句校驗（可回溯到「句」）
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub("", s or "")


def verify_quote(quote: str, snippet: str) -> bool:
    """quote 是否真的逐字出現在該來源摘要裡（正規化空白後比對）。

    空 quote 一律回 False（呼叫端據此降級為裸 URL，不是錯誤）。
    """
    q = _norm(quote)
    if len(q) < 4:
        return False
    return q in _norm(snippet)


def filter_verified_quotes(
    citations: List[Any],
    evidence_pool: Dict[int, Any],
) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
    """把 writer 給的 citations 逐一驗過，回 (通過的 eid->quote, 被丟棄的清單)。

    同一 eid 有多個 quote 時取第一個通過的。被丟棄的**必須**由呼叫端 log
    （CLAUDE.md：不可 silent fail）。
    """
    kept: Dict[int, str] = {}
    dropped: List[Tuple[int, str]] = []
    for c in citations or []:
        eid = _cit_attr(c, "evidence_id")
        quote = _cit_attr(c, "quote") or ""
        if eid is None or not quote:
            continue
        if eid in kept:
            continue
        entry = evidence_pool.get(eid)
        snippet = _attr(entry, "snippet", "")
        if verify_quote(quote, snippet):
            kept[eid] = quote
        else:
            dropped.append((eid, quote))
    return kept, dropped


def _cit_attr(c: Any, name: str):
    if isinstance(c, dict):
        return c.get(name)
    return getattr(c, name, None)


# ---------------------------------------------------------------------------
# 「這份報告答不了什麼」 —— writer 逐章填的 open_gaps 彙整
# ---------------------------------------------------------------------------

WHERE_TO_LOOK_ZH = {
    "internal": "站內語料再查",
    "web": "外部公開資料",
    "manual": "需人工詢問或調閱",
}


def render_open_gaps_block(written_sections: List[Dict[str, Any]]) -> str:
    """彙整各章 open_gaps → 報告尾的「這份報告答不了什麼」。

    去重依 question 字面（同一缺口常被相鄰章節各填一次）。
    全部章節都沒填 → 回空字串，呼叫端跳過此段。
    Stage 6 與 tools/report_replay.py 共用同一份，避免兩套排版漂移。
    """
    seen = set()
    rows = []
    for idx, section in enumerate(written_sections or []):
        for gap in section.get("open_gaps") or []:
            q = (gap.get("question") or "").strip()
            if not q or q in seen:
                continue
            seen.add(q)
            rows.append((section.get("title") or f"第 {idx + 1} 章", gap))
    if not rows:
        return ""

    lines = ["", "---", "", "## 這份報告答不了什麼", ""]
    lines.append("| 缺的那一筆 | 為什麼現有資料答不了 | 去哪裡拿 | 出現在 |")
    lines.append("| --- | --- | --- | --- |")
    for chapter, gap in rows:
        raw_where = gap.get("where_to_look") or ""
        where = WHERE_TO_LOOK_ZH.get(raw_where, raw_where or "—")
        why = (gap.get("why_missing") or "—").replace("|", "｜")
        q = (gap.get("question") or "").replace("|", "｜")
        lines.append(f"| {q} | {why} | {where} | {chapter} |")
    lines.append("")
    return "\n".join(lines)
