# -*- coding: utf-8 -*-
"""報告行文 replay harness —— 改 prompt、看整篇報告長怎樣，重複跑而不重複燒錢。

為什麼是 replay 而不是跑完整 pipeline：行文改的是 writer 那一層，
retrieval / ranking / BAB 迴圈跟「文章讀起來像不像新聞」無關，卻佔掉九成成本與時間，
而且需要 Docker + PostgreSQL 才跑得起來。這支只重放真正產文的那幾個 prompt。
（沿用 docs/e2etest.md「來源發布日期錨定」那次的 prompt-level replay 作法。）

一次跑的成本 = 章數 × 1 次 writer 呼叫。加上 prompt-hash 快取後，
只改組裝邏輯（表格排版、缺口區塊）重跑是零成本。

用法：
    # 先看 prompt 長怎樣，不呼叫 LLM，不花錢
    python tools/report_replay.py --fixture <fixture.json> --dry-run

    # 真的產一篇（章數可限，預設全部）
    python tools/report_replay.py --fixture <fixture.json> --chapters 3

    # 改了組裝邏輯想重看報告，但不想重寫章節
    python tools/report_replay.py --fixture <fixture.json> --cache-only

fixture 格式（JSON）：
    {
      "research_question": "...",
      "report_shape": "news_inverted_pyramid",
      "chapters": [{"title": "...", "outline": "...", "comparison_fields": ["..."]}],
      "evidence_pool": {"1": {"url","title","snippet","source_domain","published_at","source"}}
    }
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.citation_stats import (  # noqa: E402
    compute_source_reliance,
    count_cite_placeholders,
    filter_verified_quotes,
    render_open_gaps_block,
    render_reliance_block,
)
from reasoning.prompts.prose_discipline import (  # noqa: E402
    DEFAULT_REPORT_SHAPE,
    lint_prose,
    strip_duplicate_title,
)
from reasoning.prompts.writer import WriterPromptBuilder  # noqa: E402
from reasoning.schemas_live import (  # noqa: E402
    EvidencePoolEntry,
    LiveWriterSectionOutput,
)

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_replay_out")


class _StubHandler:
    """WriterAgent 只從 handler 取 query_params / site；replay 不需要真 handler。"""

    def __init__(self) -> None:
        self.query_params: Dict[str, Any] = {}
        self.site = "replay"


# ---------------------------------------------------------------------------
# fixture 載入
# ---------------------------------------------------------------------------

def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        fx = json.load(f)
    for key in ("research_question", "chapters", "evidence_pool"):
        if key not in fx:
            raise ValueError(f"fixture 缺 {key!r} 欄位：{path}")
    if not fx["chapters"]:
        raise ValueError(f"fixture 的 chapters 是空的：{path}")
    return fx


def build_evidence_pool(raw: Dict[str, Any]) -> Dict[int, EvidencePoolEntry]:
    """fixture 的 evidence_pool → EvidencePoolEntry dict（key 轉 int）。

    snippet 全空時直接報錯：沒有原文的 pool 產不出有依據的報告，
    讓它失敗比產一篇看起來像樣、其實無所本的東西好。
    """
    pool: Dict[int, EvidencePoolEntry] = {}
    for k, v in raw.items():
        eid = int(k)
        data = dict(v)
        data.setdefault("evidence_id", eid)
        pool[eid] = EvidencePoolEntry(**data)
    if not any((e.snippet or "").strip() for e in pool.values()):
        raise ValueError(
            "evidence_pool 每一筆的 snippet 都是空的 —— 沒有原文可依據，"
            "replay 會產出無所本的報告。請補 snippet 後再跑。"
        )
    return pool


def render_evidence_view(pool: Dict[int, EvidencePoolEntry], budget: int = 12000) -> str:
    """全 pool grounding 視圖（對齊 writer prompt 的 writer_evidence_view 通道）。"""
    lines: List[str] = []
    used = 0
    for eid in sorted(pool):
        e = pool[eid]
        date = f"，{e.published_at} 發布" if getattr(e, "published_at", None) else ""
        line = (
            f"[{eid}] {e.title}（{e.source_domain}{date}）\n"
            f"{(e.snippet or '').strip()}\n"
        )
        if used + len(line) > budget:
            lines.append("[來源清單已達字數上限，其餘從略]")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 快取：同一段 prompt 只付一次錢
# ---------------------------------------------------------------------------

def cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def cache_get(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def cache_put(cache_dir: str, key: str, payload: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"{key}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def compose_one(
    writer,
    builder: WriterPromptBuilder,
    chapter: Dict[str, Any],
    idx: int,
    total: int,
    pool: Dict[int, EvidencePoolEntry],
    evidence_view: str,
    shape: str,
    prev_summary: str,
    cache_dir: str,
    dry_run: bool,
    cache_only: bool,
) -> Dict[str, Any]:
    """寫一章：組 prompt →（快取 miss 才）呼叫 LLM → 校驗 quote → 統計引用次數。"""
    citations = sorted(pool.keys())
    prompt = builder.build_section_compose_prompt(
        section_title=chapter["title"],
        section_outline=chapter.get("outline", ""),
        relevant_findings=evidence_view,
        analyst_citations=citations,
        citation_format="numeric",
        evidence_lookup=pool,
        writer_evidence_view=evidence_view,
        current_chapter_index=idx,
        previous_chapter_summary=prev_summary,
        report_shape=shape,
        comparison_fields=chapter.get("comparison_fields"),
    )
    key = cache_key(prompt)

    with open(os.path.join(cache_dir, f"prompt_{idx}_{key}.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    if dry_run:
        print(f"  [{idx + 1}/{total}] {chapter['title']}：prompt {len(prompt)} 字（dry-run，未呼叫 LLM）")
        return {}

    cached = cache_get(cache_dir, key)
    if cached is not None:
        print(f"  [{idx + 1}/{total}] {chapter['title']}：快取命中（未付費）")
        raw = cached
    elif cache_only:
        print(f"  [{idx + 1}/{total}] {chapter['title']}：快取沒有，--cache-only 跳過")
        return {}
    else:
        print(f"  [{idx + 1}/{total}] {chapter['title']}：呼叫 writer LLM…")
        result, _retries, _fallback = await writer.call_llm_validated(
            prompt=prompt, response_schema=LiveWriterSectionOutput, level="high",
        )
        raw = result.model_dump()
        cache_put(cache_dir, key, raw)

    section = LiveWriterSectionOutput(**raw)

    # 逐句 quote 校驗（與 orchestrator._render_section_citations 同一支函式）
    kept, dropped = filter_verified_quotes(section.citations, pool)
    if dropped:
        print(
            f"      quote 校驗未通過 {len(dropped)} 筆（該引用降級為裸連結）："
            + "; ".join(f"[{eid}] {q[:16]!r}" for eid, q in dropped[:3])
        )

    counts = count_cite_placeholders(section.section_content)
    content = section.section_content
    for eid in sorted(counts, reverse=True):
        content = content.replace("{cite:%d}" % eid, f"[{eid}]")

    return {
        "section_index": idx,
        "title": section.section_title or chapter["title"],
        "content": strip_duplicate_title(section.section_title or chapter["title"], content),
        "sources_used": section.sources_used,
        "citation_counts": counts,
        "citation_quotes": kept,
        "open_gaps": [g.model_dump() for g in section.open_gaps],
        "narration": section.narration,
        "chapter_summary": section.chapter_summary,
        "confidence_level": section.confidence_level,
    }


def assemble(
    fx: Dict[str, Any],
    sections: List[Dict[str, Any]],
    pool: Dict[int, EvidencePoolEntry],
) -> str:
    """組裝整篇報告（與 orchestrator._run_stage_6 同順序：標題 → 倚重表 → 正文 → 缺口 → 參考文獻）。"""
    parts: List[str] = [f"# {fx['research_question']}\n"]

    rows, mode = compute_source_reliance(sections, pool)
    block = render_reliance_block(
        rows, mode, chapter_order=[s["title"] for s in sections]
    )
    if block:
        parts.append(block)

    for sec in sections:
        parts.append(f"## {sec['title']}\n")
        parts.append(sec["content"])
        parts.append("")
        if sec.get("narration"):
            parts.append(f"> 撰寫說明：{sec['narration']}\n")

    gaps = render_open_gaps_block(sections)
    if gaps:
        parts.append(gaps)

    cited = {eid for sec in sections for eid in (sec.get("citation_counts") or {})}
    if cited:
        parts += ["", "---", "", "## 參考文獻", ""]
        for eid in sorted(cited, key=int):
            e = pool.get(int(eid))
            if e is None:
                parts.append(f"[{eid}] （來源遺失：此 ID 不在資料池內）")
                continue
            parts.append(f"[{eid}] {e.title} — {e.source_domain} {e.url}")

    uncited = [eid for eid in sorted(pool) if eid not in {int(c) for c in cited}]
    if uncited:
        parts += ["", "---", "", "## 研究時搜尋到、報告未引用的資料", ""]
        for eid in uncited:
            e = pool[eid]
            parts.append(f"[{eid}] {e.title} — {e.source_domain} {e.url}")

    return "\n".join(parts)


async def main() -> int:
    ap = argparse.ArgumentParser(description="報告行文 replay harness")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--chapters", type=int, default=0, help="只寫前 N 章（0 = 全部）")
    ap.add_argument("--shape", default=None, help="報告骨架；預設讀 fixture，再預設 news_inverted_pyramid")
    ap.add_argument("--dry-run", action="store_true", help="只組 prompt 不呼叫 LLM（免費）")
    ap.add_argument("--cache-only", action="store_true", help="只用快取，缺的章節跳過（免費）")
    ap.add_argument("--tag", default="run", help="輸出檔名前綴")
    args = ap.parse_args()

    fx = load_fixture(args.fixture)
    pool = build_evidence_pool(fx["evidence_pool"])
    shape = args.shape or fx.get("report_shape") or DEFAULT_REPORT_SHAPE
    chapters = fx["chapters"][: args.chapters] if args.chapters > 0 else fx["chapters"]

    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(args.out, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"fixture：{args.fixture}")
    print(f"骨架：{shape}｜章數：{len(chapters)}｜evidence：{len(pool)} 筆")

    builder = WriterPromptBuilder()
    writer = None
    if not args.dry_run:
        from reasoning.agents.writer import WriterAgent
        writer = WriterAgent(_StubHandler(), timeout=180)

    evidence_view = render_evidence_view(pool)

    sections: List[Dict[str, Any]] = []
    prev_summary = ""
    for idx, ch in enumerate(chapters):
        sec = await compose_one(
            writer, builder, ch, idx, len(chapters), pool, evidence_view,
            shape, prev_summary, cache_dir, args.dry_run, args.cache_only,
        )
        if not sec:
            continue
        sections.append(sec)
        prev_summary = sec.get("chapter_summary") or ""

    if not sections:
        print("沒有產出任何章節（dry-run 或快取全 miss）。")
        return 0

    report = assemble(fx, sections, pool)
    md_path = os.path.join(args.out, f"{args.tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(args.out, f"{args.tag}.sections.json"), "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    rows, mode = compute_source_reliance(sections, pool)
    print(f"\n報告：{md_path}（{len(report)} 字）")
    print(f"引用統計（{mode}）：")
    for r in rows[:5]:
        print(f"  [{r['evidence_id']}] {r['citation_count']} 次 / {r['section_count']} 章 — {r['title'][:40]}")

    total_quotes = sum(len(s.get("citation_quotes") or {}) for s in sections)
    print(f"逐字引句：{total_quotes} 個來源附上可跳轉的原文片段（全部通過逐字校驗）")

    # 行文 lint：prompt 擋不住的在這裡現形，下一輪就知道要調哪條規則
    hits = lint_prose("\n".join(s["content"] for s in sections))
    if hits:
        tally: Dict[str, int] = {}
        for w, _ctx in hits:
            tally[w] = tally.get(w, 0) + 1
        print(f"\n行文 lint 命中 {len(hits)} 處：" + "、".join(f"{w}×{n}" for w, n in tally.items()))
        for w, ctx in hits[:8]:
            print(f"  {w}：…{ctx}…")
    else:
        print("行文 lint：0 命中")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
