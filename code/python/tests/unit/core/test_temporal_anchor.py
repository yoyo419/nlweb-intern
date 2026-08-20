"""core.temporal_anchor 單元測試（發布日期錨定）。

修的症狀：新聞內文的「今年」指的是該報導**發布那年**（2025），但答案輸出寫成
當前年（2026）。錨定的年份換算由 code 做，這裡鎖住換算規則本身。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from core.temporal_anchor import (  # noqa: E402
    TEMPORAL_ANCHOR_RULE,
    anchor_note,
    annotate_relative_years,
    published_year,
)


class TestPublishedYear:

    def test_iso_datetime(self):
        assert published_year("2025-09-12T08:00:00Z") == 2025

    def test_date_only_and_slash_and_year_only(self):
        assert published_year("2025-09-12") == 2025
        assert published_year("2025/09/12") == 2025
        assert published_year("2025") == 2025

    def test_unknown_shapes_return_none(self):
        # 抽不到就回 None（不可猜成今年 —— 猜錯就是本 bug 本身）
        for bad in (None, "", "   ", "Unknown", "n.d.", "9999-01-01", "1800-01-01"):
            assert published_year(bad) is None, bad


class TestAnnotateRelativeYears:

    def test_this_year_resolves_to_publication_year(self):
        out = annotate_relative_years("今年流感疫苗10月開打", "2025-09-12", now_year=2026)
        assert "今年（2025年）" in out

    def test_all_year_level_terms(self):
        out = annotate_relative_years(
            "今年成長、去年持平、前年下滑、明年看好、後年待觀察、本年持穩", "2025-05-01", now_year=2026
        )
        assert "今年（2025年）" in out
        assert "去年（2024年）" in out
        assert "前年（2023年）" in out
        assert "明年（2026年）" in out
        assert "後年（2027年）" in out
        assert "本年（2025年）" in out

    def test_same_year_source_is_untouched(self):
        # 發布年 == 當前年 → LLM 用「今天」換算剛好正確，不加噪音
        text = "今年流感疫苗10月開打"
        assert annotate_relative_years(text, "2026-01-02", now_year=2026) == text

    def test_unknown_date_is_untouched(self):
        text = "今年流感疫苗10月開打"
        for bad in (None, "", "Unknown"):
            assert annotate_relative_years(text, bad, now_year=2026) == text

    def test_fiscal_year_term_not_broken(self):
        # 「今年度」是會計年度用語，插進去會變「今年（2025年）度」
        out = annotate_relative_years("今年度預算", "2025-05-01", now_year=2026)
        assert out == "今年度預算"

    def test_idempotent_no_double_annotation(self):
        once = annotate_relative_years("今年開打", "2025-09-12", now_year=2026)
        twice = annotate_relative_years(once, "2025-09-12", now_year=2026)
        assert once == twice == "今年（2025年）開打"

    def test_empty_text(self):
        assert annotate_relative_years("", "2025-09-12", now_year=2026) == ""


class TestAnchorNote:

    def test_note_carries_the_resolved_year(self):
        note = anchor_note("2025-09-12T08:00:00Z", now_year=2026)
        assert "今年" in note and "2025" in note
        # 預設不重印日期（多數標頭本來就印了）
        assert "2025-09-12" not in note

    def test_note_with_date_when_header_has_none(self):
        note = anchor_note("2025-09-12T08:00:00Z", include_date=True, now_year=2026)
        assert "2025-09-12" in note
        assert "今年" in note and "2025" in note

    def test_note_empty_when_same_year_or_unknown(self):
        assert anchor_note("2026-03-01", now_year=2026) == ""
        assert anchor_note("", now_year=2026) == ""
        assert anchor_note(None, now_year=2026) == ""


class TestRuleText:

    def test_rule_states_the_anchor(self):
        # prompt 端第二道防線被清空 / 改寫掉時要炸（純 prompt 規則沒東西鎖它）
        assert "發布日期" in TEMPORAL_ANCHOR_RULE
        assert "今年" in TEMPORAL_ANCHOR_RULE
        assert "不可用今天的日期換算" in TEMPORAL_ANCHOR_RULE
