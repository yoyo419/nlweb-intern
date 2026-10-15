"""報告行文紀律的機械防線。

純改 prompt 沒有東西鎖它 —— 下一個人動 prompt 時，這裡是唯一會叫的東西。
斷言分三層：
1. prompt 層：紀律區塊、骨架、可回溯與缺口契約有沒有真的進到送給 LLM 的字串。
2. 產物層：`assert_prose_clean` 對「產出的中文散文」掃禁用詞，replay harness 與
   人工審稿都能拿來用。
3. 統計層：引用次數與逐字引句校驗的行為。

詞表本身定義在 reasoning/prompts/prose_discipline.py，測試從那裡 import ——
改詞表只需改一處，prompt 與斷言會同時跟著動。
"""

import re

import pytest

from reasoning.citation_stats import (
    COUNTING_MODE_EXACT,
    COUNTING_MODE_MIXED,
    COUNTING_MODE_PER_SECTION,
    compute_source_reliance,
    count_cite_placeholders,
    filter_verified_quotes,
    render_open_gaps_block,
    render_reliance_block,
    verify_quote,
)
from reasoning.prompts.prose_discipline import (
    ALL_BANNED_WORDS,
    BANNED_WORD_CATEGORIES,
    DEFAULT_REPORT_SHAPE,
    DEFAULT_SECTION_CHAR_CAP,
    build_shape_block,
    measure_section_chars,
    lint_prose,
    is_known_report_shape,
    strip_duplicate_title,
    normalize_report_shape,
    report_shapes,
)
from reasoning.prompts.writer import WriterPromptBuilder


def assert_prose_clean(text, where=""):
    """對產出的散文掃禁用詞。命中就把命中的詞與上下文一起吐出來。

    掃描本體是 prose_discipline.lint_prose —— replay harness、Stage 6 與這裡
    共用同一支，避免三套判準漂移。
    """
    hits = lint_prose(text)
    assert not hits, (
        f"{where} 出現禁用詞 {len(hits)} 處：\n"
        + "\n".join(f"{w}：…{ctx}…" for w, ctx in hits[:10])
    )


def _section_prompt(**kwargs):
    builder = WriterPromptBuilder()
    base = dict(
        section_title="第一章",
        section_outline="大綱",
        relevant_findings="發現 A",
        analyst_citations=[1, 2],
        citation_format="numeric",
    )
    base.update(kwargs)
    return builder.build_section_compose_prompt(**base)


@pytest.mark.unit
class TestProseDisciplineInPrompt:
    """行文紀律必須真的進到送給 LLM 的字串裡。"""

    def test_lr_section_prompt_carries_every_banned_word(self):
        prompt = _section_prompt()
        missing = [w for w in ALL_BANNED_WORDS if w not in prompt]
        assert not missing, f"LR writer prompt 沒帶到禁用詞：{missing}"

    def test_lr_prompt_carries_each_banned_category_heading(self):
        prompt = _section_prompt()
        for heading in (
            "軟墊詞", "導覽與總結句", "假條件句", "模糊量詞",
            "稻草人句型", "替讀者數數", "無主詞斷言", "替讀者下定義",
        ):
            assert heading in prompt, f"缺分類標頭：{heading}"

    def test_banned_categories_are_disjoint_and_nonempty(self):
        for c in BANNED_WORD_CATEGORIES:
            assert c, "禁用詞分類不可為空"
        assert len(ALL_BANNED_WORDS) == len(set(ALL_BANNED_WORDS)), "禁用詞重複"

    def test_prompt_has_no_full_sentence_negative_examples_of_banned_words(self):
        """禁用詞只能以詞表出現，不可包在完整的錯誤示範句裡。

        理由：弱模型會把 prompt 裡的錯誤示範整句抄進正文（既有 v8 regression）。
        判準是禁用詞後面不可緊接中文句號——詞表是頓號分隔，示範句才會有句號。
        """
        prompt = _section_prompt()
        for word in ALL_BANNED_WORDS:
            for m in re.finditer(re.escape(word), prompt):
                tail = prompt[m.end():m.end() + 12]
                assert "。" not in tail.split("、")[0], (
                    f"禁用詞 {word!r} 疑似出現在完整示範句中：…{prompt[m.start():m.end() + 20]}…"
                )

    def test_dr_compose_prompt_carries_prose_discipline(self):
        from reasoning.schemas import CriticReviewOutput

        builder = WriterPromptBuilder()
        review = CriticReviewOutput(
            status="PASS",
            critique=(
                "草稿每個論點都掛得上來源，沒有發現超出證據範圍的推論；"
                "引用編號與來源對得起來，時間範圍也沒有跑掉，可以進入編輯階段。"
            ),
            mode_compliance="符合",
            suggestions=[],
        )
        prompt = builder.build_compose_prompt(
            analyst_draft="草稿",
            critic_review=review,
            analyst_citations=[1],
            mode="discovery",
            user_query="q",
            suggested_confidence="High",
        )
        assert "行文紀律" in prompt
        missing = [w for w in ALL_BANNED_WORDS if w not in prompt]
        assert not missing, f"DR writer prompt 沒帶到禁用詞：{missing}"


@pytest.mark.unit
class TestReportShapeRegistry:
    """骨架是可選的：記者要倒三角，永續報告書撰寫者要對照表。"""

    def test_default_shape_is_news_inverted_pyramid(self):
        assert DEFAULT_REPORT_SHAPE == "news_inverted_pyramid"

    def test_every_registered_shape_renders_nonempty_block(self):
        for shape in report_shapes():
            assert build_shape_block(shape).strip(), f"{shape} 骨架是空的"

    def test_news_shape_demands_lead_and_nut_graf(self):
        block = build_shape_block("news_inverted_pyramid")
        assert "導言" in block and "核心段" in block and "倒三角" in block

    def test_esg_shape_puts_concern_and_action_in_one_table(self):
        block = build_shape_block("esg_matrix")
        assert "外界關切" in block and "對應作為" in block

    def test_chinese_aliases_resolve(self):
        assert normalize_report_shape("新聞") == "news_inverted_pyramid"
        assert normalize_report_shape("研究報告") == "research_report"
        assert normalize_report_shape("永續") == "esg_matrix"

    def test_unknown_shape_is_reported_not_silently_accepted(self):
        assert is_known_report_shape("news_inverted_pyramid") is True
        assert is_known_report_shape("不存在的骨架") is False
        assert normalize_report_shape("不存在的骨架") == DEFAULT_REPORT_SHAPE

    def test_first_chapter_gets_lead_instruction_and_later_chapters_do_not(self):
        first = _section_prompt(current_chapter_index=0)
        later = _section_prompt(current_chapter_index=2)
        assert "本章是第 1 章" in first
        assert "本章是第 3 章" in later
        assert "不要重寫一次開場" in later


@pytest.mark.unit
class TestTraceabilityContract:
    """從結論走得回出處：每個引用要附來源逐字片段。"""

    def test_prompt_requires_verbatim_quote_per_citation(self):
        prompt = _section_prompt()
        # 斷言在紀律區塊本身，不是 JSON 範例裡的欄位說明 —— 範例即使留著，
        # 少了規則區塊模型也不知道 quote 要逐字。
        assert "可回溯到「句」" in prompt
        assert "不改寫、不濃縮" in prompt
        assert "系統會檢查 quote 是否真的出現在該來源摘要裡" in prompt
        assert '"quote"' in prompt

    def test_no_quote_requirement_when_citation_format_is_none(self):
        prompt = _section_prompt(citation_format="none")
        assert "可回溯到「句」" not in prompt

    def test_verify_quote_accepts_verbatim_and_rejects_paraphrase(self):
        snippet = "六大環團發出聯合聲明，指擴大禁建區恐矯枉過正。"
        assert verify_quote("擴大禁建區恐矯枉過正", snippet) is True
        assert verify_quote("環團認為禁建範圍太大", snippet) is False

    def test_verify_quote_ignores_whitespace_differences(self):
        assert verify_quote("擴大 禁建區", "指擴大禁建區恐矯枉過正") is True

    def test_verify_quote_rejects_too_short_and_empty(self):
        assert verify_quote("", "任何內容") is False
        assert verify_quote("環團", "環團發出聲明") is False

    def test_filter_verified_quotes_keeps_good_and_reports_bad(self):
        pool = {1: {"snippet": "環評門檻應加嚴，屋頂優先。"}, 2: {"snippet": "無關內容"}}
        kept, dropped = filter_verified_quotes(
            [
                {"evidence_id": 1, "quote": "環評門檻應加嚴"},
                {"evidence_id": 2, "quote": "這句話不在來源裡"},
            ],
            pool,
        )
        assert kept == {1: "環評門檻應加嚴"}
        assert [eid for eid, _ in dropped] == [2]


@pytest.mark.unit
class TestOpenGapsContract:
    """缺什麼就說缺什麼，不用現有段落充數。"""

    def test_prompt_requires_open_gaps_field(self):
        prompt = _section_prompt()
        assert "open_gaps" in prompt
        assert "where_to_look" in prompt

    def test_prompt_forbids_substituting_an_existing_passage(self):
        assert "充數" in _section_prompt()

    def test_gap_block_dedupes_same_question_across_chapters(self):
        sections = [
            {"title": "第一章", "open_gaps": [
                {"question": "2025 年裝置容量", "why_missing": "無官方版本", "where_to_look": "web"}]},
            {"title": "第二章", "open_gaps": [
                {"question": "2025 年裝置容量", "why_missing": "無官方版本", "where_to_look": "web"}]},
        ]
        block = render_open_gaps_block(sections)
        assert block.count("2025 年裝置容量") == 1
        assert "外部公開資料" in block

    def test_gap_block_empty_when_no_gaps(self):
        assert render_open_gaps_block([{"title": "第一章", "open_gaps": []}]) == ""


@pytest.mark.unit
class TestSourceReliance:
    """哪一篇被引用最多次 —— 報告開頭那張表的算法。"""

    def test_counts_repeat_citations_within_one_chapter(self):
        assert count_cite_placeholders("a{cite:1}b{cite:1}c{cite:2}") == {1: 2, 2: 1}

    def test_cite_zero_is_not_a_source(self):
        assert count_cite_placeholders("a{cite:0}b{cite:3}") == {3: 1}

    def test_ranking_puts_most_cited_first(self):
        sections = [
            {"title": "第一章", "citation_counts": {1: 3, 2: 1}},
            {"title": "第二章", "citation_counts": {2: 1, 3: 5}},
        ]
        rows, mode = compute_source_reliance(sections, {})
        assert mode == COUNTING_MODE_EXACT
        assert [r["evidence_id"] for r in rows] == [3, 1, 2]
        assert rows[0]["citation_count"] == 5
        assert rows[2]["section_count"] == 2

    def test_falls_back_to_per_section_counting_and_says_so(self):
        sections = [
            {"title": "第一章", "sources_used": [1, 2]},
            {"title": "第二章", "sources_used": [2]},
        ]
        rows, mode = compute_source_reliance(sections, {})
        assert mode == COUNTING_MODE_PER_SECTION
        assert rows[0]["evidence_id"] == 2 and rows[0]["citation_count"] == 2
        block = render_reliance_block(rows, mode)
        assert "估計值" in block, "退回估算時必須讓讀者看得到，不可假裝是精確次數"

    def test_mixed_mode_when_only_some_chapters_have_counts(self):
        sections = [
            {"title": "第一章", "citation_counts": {1: 3}},
            {"title": "第二章", "sources_used": [2]},
        ]
        _rows, mode = compute_source_reliance(sections, {})
        assert mode == COUNTING_MODE_MIXED

    def test_json_roundtrip_string_keys_still_count(self):
        """written_sections 落 JSONB 後 key 會變 str，不可因此漏算。"""
        sections = [{"title": "第一章", "citation_counts": {"1": 2, "2": 1}}]
        rows, mode = compute_source_reliance(sections, {})
        assert mode == COUNTING_MODE_EXACT
        assert rows[0]["evidence_id"] == 1 and rows[0]["citation_count"] == 2

    def test_table_carries_title_and_chapters(self):
        sections = [{"title": "光電修法", "citation_counts": {7: 2}}]
        pool = {7: {"title": "環評門檻應加嚴", "source_domain": "udn.com", "url": "https://x/y"}}
        rows, mode = compute_source_reliance(sections, pool)
        block = render_reliance_block(rows, mode, chapter_order=["光電修法"])
        assert "本報告最倚重的來源" in block
        assert "環評門檻應加嚴" in block
        assert "第 1 章" in block
        assert "https://x/y" in block

    def test_empty_sections_render_nothing(self):
        rows, _mode = compute_source_reliance([], {})
        assert render_reliance_block(rows, COUNTING_MODE_EXACT) == ""


@pytest.mark.unit
class TestProseCleanHelper:
    """assert_prose_clean 自己要有牙，否則產物層等於沒驗。"""

    def test_passes_on_clean_prose(self):
        assert_prose_clean("環團在協商前一天發出聯合聲明，理由是禁建區會誤傷既有案場。")

    @pytest.mark.parametrize("word", ALL_BANNED_WORDS)
    def test_catches_every_banned_word(self, word):
        with pytest.raises(AssertionError):
            assert_prose_clean(f"這件事{word}很重要。")


@pytest.mark.unit
class TestReliancTableReadability:
    """整串章名會把表格撐到讀不動；章號才讀得動。"""

    def test_chapters_render_as_numbers_not_full_titles(self):
        titles = ["國家能源轉型與再生能源目標", "農地與漁塭的空間與土地使用管制"]
        sections = [
            {"title": titles[0], "citation_counts": {1: 2}},
            {"title": titles[1], "citation_counts": {1: 1}},
        ]
        rows, mode = compute_source_reliance(sections, {})
        block = render_reliance_block(rows, mode, chapter_order=titles)
        assert "第 1、2 章（共 2 章）" in block
        assert titles[0] not in block, "章名不該整串出現在表格裡"


@pytest.mark.unit
class TestLintIsWiredIntoExport:
    """產物層防線要真的接在匯出流程上，不是只有測試檔在用。"""

    def test_stage_6_logs_lint_hits(self):
        import inspect
        from reasoning.live_research import orchestrator as orch

        src = inspect.getsource(orch.LiveResearchOrchestrator._run_stage_6)
        assert "lint_prose(full_report)" in src, "Stage 6 沒跑行文 lint"
        assert "logger.warning" in src, "lint 命中沒有留下訊息就是 silent fail"

    def test_lint_reports_position_context_not_just_the_word(self):
        hits = lint_prose("開發商其實早就完成饋線登記。")
        assert hits and "饋線登記" in hits[0][1], "lint 要給上下文，否則看不出改哪一句"


@pytest.mark.unit
class TestSectionContentNormalization:
    """章名由系統排版；正文再寫一次會變成標題出現兩次。"""

    def test_strips_repeated_title_line(self):
        assert strip_duplicate_title("國家能源轉型", "國家能源轉型\n\n正文開始") == "正文開始"

    def test_strips_repeated_title_written_as_markdown_heading(self):
        assert strip_duplicate_title("國家能源轉型", "## 國家能源轉型\n正文") == "正文"

    def test_keeps_first_line_that_merely_starts_with_the_title(self):
        content = "國家能源轉型的目標訂在 2050 年。\n正文"
        assert strip_duplicate_title("國家能源轉型", content) == content

    def test_news_shape_requires_markdown_subheadings(self):
        block = build_shape_block("news_inverted_pyramid")
        assert "###" in block, "小標寫成裸文字行，匯出後整章沒有落點"


def _outline(target=0):
    from reasoning.schemas_live import BookOutline, ChapterPlan
    return BookOutline(
        chapters=[
            ChapterPlan(chapter_index=0, title="一", brief="b", target_word_count=target, role="intro"),
            ChapterPlan(chapter_index=1, title="二", brief="b", target_word_count=target, role="body"),
        ],
        overall_arc="arc",
    )


@pytest.mark.unit
class TestSectionLength:
    """CEO 8/29 看 replay 成品：「超級長」。沒指定字數時要有上限；有指定時不打架。"""

    def test_default_cap_when_no_word_target(self):
        prompt = _section_prompt()
        assert "## 篇幅" in prompt
        assert f"{DEFAULT_SECTION_CHAR_CAP} 字為上限" in prompt

    def test_cap_is_not_a_target_to_fill(self):
        assert "上限不是要湊滿的目標" in _section_prompt()

    def test_no_default_cap_when_outline_sets_word_target(self):
        prompt = _section_prompt(book_outline=_outline(2500), current_chapter_index=0)
        assert "## 篇幅" not in prompt
        assert "本章目標字數" in prompt

    def test_default_cap_when_outline_has_no_word_target(self):
        prompt = _section_prompt(book_outline=_outline(0), current_chapter_index=1)
        assert "## 篇幅" in prompt

    def test_no_default_cap_on_revision(self):
        prompt = _section_prompt(revise_instruction="擴寫第二段", prior_section_content="舊稿")
        assert "## 篇幅" not in prompt

    def test_measure_ignores_cite_marks_layout_and_whitespace(self):
        assert measure_section_chars("### 小標\n\n環團聲明{cite:3}[4]。 ") == len("小標環團聲明。")

