"""DR 報告表頭「分析來源數」與 confidence_level 的計數來源測試。

症狀（CEO 回報）：報告內文引用了十個來源（[1]..[10]），表頭卻寫「**分析來源數：** 1」。

根因：`_generate_final_report()` 用 `len(results)` 當來源數，但
`DeepResearchOrchestrator.run_research()` / `run_research_rerun()` 一律回**單一元素**的 list
（`_format_result()` 把整份報告包成一個 Item），故 `len(results)` 恆為 1，
與報告實際分析／引用的來源數毫無關係。同一根因也讓 `_calculate_confidence()`
（`len(results)>=5 → High`）永遠落在 'Low'。

修法：兩者都改讀 orchestrator 寫進 `schema_object` 的權威值——
`total_sources_analyzed`（= `len(context)`）與 `confidence`（= Critic status 映射）。

測試策略沿用 test_deep_research_interrupt.py 的 `__new__` bare handler pattern
（不走重量級 super().__init__，只填這條路徑用到的 attr）。
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from methods.deep_research import DeepResearchHandler  # noqa: E402


def _bare_handler(query="台灣半導體產業現況"):
    h = DeepResearchHandler.__new__(DeepResearchHandler)
    h.query = query
    h.research_mode = "discovery"
    return h


def _report_item(description="# 報告本文\n\n主要發現 [1][2][3]。", **schema):
    """模擬 orchestrator._format_result() 回的單一元素 list。"""
    return [{
        "@type": "Item",
        "url": "internal://system/discovery/q",
        "name": "深度研究報告：q",
        "description": description,
        "schema_object": {"@type": "ResearchReport", **schema},
    }]


NO_TEMPORAL = {"is_temporal_query": False}


def _source_count_line(markdown):
    """從報告 markdown 抓「分析來源數」的數字；沒有這行回 None。"""
    m = re.search(r"\*\*分析來源數：\*\*\s*(\d+)", markdown)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- 分析來源數

def test_source_count_reads_total_sources_analyzed_not_len_results():
    """核心迴歸：單一元素 results + total_sources_analyzed=10 → 表頭必須寫 10（不是 1）。"""
    results = _report_item(total_sources_analyzed=10)
    h = _bare_handler()

    report = h._generate_final_report(results, NO_TEMPORAL)

    assert len(results) == 1, "前提：orchestrator 回單一元素 list（本測試的意義所在）"
    assert _source_count_line(report) == 10, \
        f"分析來源數應為 10（total_sources_analyzed），實得：{_source_count_line(report)}"


def test_source_count_falls_back_to_non_empty_sources_used():
    """缺 total_sources_analyzed 時降級數 sources_used 的非空 URL（缺號以空字串佔位不算）。"""
    results = _report_item(sources_used=["https://a", "", "https://c", "  ", "private://d"])
    h = _bare_handler()

    report = h._generate_final_report(results, NO_TEMPORAL)

    assert _source_count_line(report) == 3, \
        f"應只數 3 個非空 URL，實得：{_source_count_line(report)}"


def test_source_count_line_omitted_when_no_authoritative_value():
    """no-data / error item 無任何來源欄位 → 不印假數字（整行省略），而非退回 len(results)。"""
    results = [{"description": "# 查無相關資料", "schema_object": {"@type": "NoDataReport"}}]
    h = _bare_handler()

    report = h._generate_final_report(results, NO_TEMPORAL)

    assert "分析來源數" not in report, "取不到權威來源數時不可印數字（會是假的 1）"
    assert "# 深度研究報告：" in report, "其餘報告結構仍應完整"
    assert "# 查無相關資料" in report, "本文仍須輸出"


def test_source_count_zero_is_printed_not_treated_as_missing():
    """total_sources_analyzed=0 是有效值（真的零來源），不可被當成「缺值」而省略整行。"""
    results = _report_item(total_sources_analyzed=0)
    h = _bare_handler()

    report = h._generate_final_report(results, NO_TEMPORAL)

    assert _source_count_line(report) == 0, "0 是權威值，應照印"


def test_source_count_ignores_bool_true():
    """bool 是 int 的 subclass——total_sources_analyzed=True 不可被讀成 1。"""
    results = _report_item(total_sources_analyzed=True, sources_used=["https://a", "https://b"])
    h = _bare_handler()

    report = h._generate_final_report(results, NO_TEMPORAL)

    assert _source_count_line(report) == 2, \
        f"True 應被排除、降級數 sources_used=2，實得：{_source_count_line(report)}"


def test_temporal_range_still_rendered():
    """對照組：修改沒把時間範圍那行弄丟。"""
    results = _report_item(total_sources_analyzed=7)
    h = _bare_handler()

    report = h._generate_final_report(
        results,
        {"is_temporal_query": True, "start_date": "2026-01-01", "end_date": "2026-06-30"},
    )

    assert _source_count_line(report) == 7
    assert "**時間範圍：** 2026-01-01 至 2026-06-30" in report


# ---------------------------------------------------------------- confidence

def test_confidence_reads_schema_object_not_len_results():
    """同根因：單一元素 results 也要能回 High（舊版 len(results)==1 → 恆 Low）。"""
    results = _report_item(confidence="High")
    h = _bare_handler()

    assert h._calculate_confidence(results) == "High"


def test_confidence_medium_passthrough():
    results = _report_item(confidence="Medium")
    h = _bare_handler()

    assert h._calculate_confidence(results) == "Medium"


def test_confidence_degrades_to_low_when_field_missing():
    """無 confidence 欄位（no-data / error item）→ 保守回 Low（降級，非宣稱高信心）。"""
    results = [{"description": "# 查無相關資料", "schema_object": {"@type": "NoDataReport"}}]
    h = _bare_handler()

    assert h._calculate_confidence(results) == "Low"


# ---------------------------------------------------------------- helper 直測

def test_count_analyzed_sources_returns_none_for_empty_results():
    """斷線早退 results=[] → None（呼叫端不印該行；persist gate 另以 len(results)==0 判斷）。"""
    assert DeepResearchHandler._count_analyzed_sources([]) is None
