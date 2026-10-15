# -*- coding: utf-8 -*-
"""從真實 prod session 的 evidence URL 抓回內文，補成可 replay 的 fixture。

為什麼要這支：`fixtures/real_energy_policy_state.json` 是真的 prod session，
但脫敏時只留了 URL / 標題，snippet 全空——沒有原文就產不出有依據的報告。
這支照 URL 把內文抓回來填進 snippet，讓 report_replay.py 有東西可 ground。

抓不到的條目會被丟掉並列出來（不 silent skip）。整份 fixture 只需要建一次，
之後改 prompt 重跑都吃同一份。

用法：
    python tools/build_replay_fixture.py --out reasoning/live_research/fixtures/report_style_demo.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_STATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reasoning", "live_research", "fixtures", "real_energy_policy_state.json",
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

MAX_SNIPPET = 1200


def clean(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_body(url: str) -> str:
    import httpx
    import trafilatura

    with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": UA}) as c:
        r = c.get(url)
        r.raise_for_status()
        html = r.text
    body = trafilatura.extract(html, include_comments=False, include_tables=False)
    return clean(body or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 筆（0 = 全部）")
    args = ap.parse_args()

    with open(SRC_STATE, encoding="utf-8") as f:
        state = json.load(f)
    src_pool: Dict[str, Any] = state["evidence_pool"]
    cm = json.loads(state["context_map_json"])

    items = sorted(src_pool.items(), key=lambda kv: int(kv[0]))
    if args.limit:
        items = items[: args.limit]

    pool: Dict[str, Any] = {}
    failed: List[str] = []
    for eid, entry in items:
        url = entry.get("url") or ""
        if not url.startswith("http"):
            failed.append(f"[{eid}] 非 http URL：{url}")
            continue
        try:
            body = fetch_body(url)
        except Exception as e:  # noqa: BLE001 — 逐條回報，不整批中斷
            failed.append(f"[{eid}] 抓取失敗 {type(e).__name__}: {e} — {url}")
            continue
        if len(body) < 200:
            failed.append(f"[{eid}] 內文過短（{len(body)} 字），可能是付費牆或動態頁 — {url}")
            continue
        pool[eid] = {
            "evidence_id": int(eid),
            "title": entry.get("title", ""),
            "url": url,
            "source_domain": entry.get("source_domain") or urlparse(url).netloc,
            "snippet": body[:MAX_SNIPPET],
            "author": entry.get("author", ""),
            "year": entry.get("year", ""),
            "published_at": entry.get("published_at"),
            "source": "internal",
        }
        print(f"  [{eid}] OK {len(body)} 字 — {entry.get('title', '')[:36]}")

    if not pool:
        print("一筆都沒抓到，fixture 不寫出。")
        for f_ in failed:
            print("  " + f_)
        return 1

    fixture = {
        "_source": state.get("_source", ""),
        "_note": "evidence snippet 由 tools/build_replay_fixture.py 依原 URL 重新抓取",
        "research_question": cm.get("research_question", ""),
        "report_shape": "news_inverted_pyramid",
        "chapters": [
            {"title": t["name"], "outline": t["description"]}
            for t in cm.get("topics", [])
            if t.get("relevance") == "core" and t.get("evidence_ids")
        ][:4],
        "evidence_pool": pool,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=2)

    print(f"\n寫出 {args.out}：{len(pool)} 筆 evidence，{len(fixture['chapters'])} 章")
    if failed:
        print(f"未納入 {len(failed)} 筆：")
        for f_ in failed:
            print("  " + f_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
