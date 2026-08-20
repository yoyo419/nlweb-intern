"""來源發布日期錨定 —— 各「來源文字 → LLM prompt」出口的機械防線。

症狀：2025 年的報導寫「今年流感疫苗 10 月開打」，系統答案寫成「2026 年」。
根因：prompt 另一端注入「今天是 YYYY-MM-DD」，而多數渲染點根本沒把該來源的
發布日期送進去（LR analyst 搜尋結果、LR/DR writer evidence 清單皆無日期）
→ LLM 把來源內文的相對年份掛到「今天」。

本檔逐一鎖住每個出口：來源日期有送進 prompt、且內文相對年份已被 code 換算標註。
純 prompt 規則沒有東西鎖它，所以規則字串本身也各鎖一條。
"""

import os
import sys
import types

from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.temporal_anchor import TEMPORAL_ANCHOR_RULE  # noqa: E402

# 固定用「發布年 ≠ 當前年」的素材（那正是會出錯的情況）
PAST_DATE = "2020-09-12"
ARTICLE_TEXT = "今年流感疫苗10月開打，去年為9月。"
EXPECTED_THIS_YEAR = "今年（2020年）"
EXPECTED_LAST_YEAR = "去年（2019年）"


# ===== DR：analyst / critic / writer 共用的 formatted_context =====


class TestDRFormatContextShared:

    def _orch(self):
        from reasoning.orchestrator import DeepResearchOrchestrator

        orch = MagicMock()
        orch.logger = MagicMock()
        orch._format_context_shared = (
            DeepResearchOrchestrator._format_context_shared.__get__(orch)
        )
        orch._get_current_time_header = MagicMock(return_value="")
        return orch

    def test_snippet_relative_year_is_anchored_to_publication_date(self):
        orch = self._orch()
        formatted, source_map = orch._format_context_shared([
            {
                "url": "https://news.example/1",
                "title": "流感疫苗開打",
                "site": "中央社",
                "description": ARTICLE_TEXT,
                "datePublished": PAST_DATE + "T08:00:00Z",
            }
        ])
        assert EXPECTED_THIS_YEAR in formatted
        assert EXPECTED_LAST_YEAR in formatted
        assert PAST_DATE in formatted           # 標頭仍帶發布日期
        assert len(source_map) == 1

    def test_missing_date_leaves_text_untouched(self):
        orch = self._orch()
        formatted, _ = orch._format_context_shared([
            {"url": "u", "title": "t", "site": "s", "description": ARTICLE_TEXT}
        ])
        assert "今年（" not in formatted     # 猜年份比不標更糟

    def test_current_time_header_carries_the_rule(self):
        from reasoning.orchestrator import DeepResearchOrchestrator

        orch = MagicMock()
        orch.logger = MagicMock()
        header = DeepResearchOrchestrator._get_current_time_header.__get__(orch)()
        assert TEMPORAL_ANCHOR_RULE in header


class TestDRCriticReferenceSheet:

    def test_reference_sheet_is_anchored(self):
        from reasoning.orchestrator import DeepResearchOrchestrator

        orch = MagicMock()
        orch.logger = MagicMock()
        orch.source_map = {
            3: {
                "url": "https://news.example/1",
                "title": "流感疫苗開打",
                "site": "中央社",
                "description": ARTICLE_TEXT,
                "datePublished": PAST_DATE,
            }
        }
        orch._extract_item_fields = (
            DeepResearchOrchestrator._extract_item_fields.__get__(orch)
        )
        orch._extract_item_date = (
            DeepResearchOrchestrator._extract_item_date.__get__(orch)
        )
        sheet = DeepResearchOrchestrator._build_critic_reference_sheet.__get__(orch)([3])
        assert EXPECTED_THIS_YEAR in sheet
        assert PAST_DATE in sheet

    def test_extract_item_date_handles_row_tuple(self):
        import json

        from reasoning.orchestrator import DeepResearchOrchestrator

        orch = MagicMock()
        orch.logger = MagicMock()
        extract = DeepResearchOrchestrator._extract_item_date.__get__(orch)
        row = ("u", json.dumps({"datePublished": PAST_DATE}), "t", "site")
        assert extract(row) == PAST_DATE
        assert extract(("u", "not-json", "t", "site")) == ""   # 壞 schema → 不猜
        assert extract(None) == ""


class TestDRWriterEvidenceLookup:

    def test_lookup_carries_date_and_anchored_snippet(self):
        from reasoning.orchestrator import DeepResearchOrchestrator

        orch = MagicMock()
        orch.logger = MagicMock()
        orch._extract_item_fields = (
            DeepResearchOrchestrator._extract_item_fields.__get__(orch)
        )
        orch._extract_item_date = (
            DeepResearchOrchestrator._extract_item_date.__get__(orch)
        )
        state = types.SimpleNamespace(
            analyst_citations=[1],
            source_map={
                1: {
                    "url": "https://news.example/1",
                    "title": "流感疫苗開打",
                    "site": "中央社",
                    "description": ARTICLE_TEXT,
                    "datePublished": PAST_DATE,
                }
            },
        )
        lookup = DeepResearchOrchestrator._build_writer_evidence_lookup.__get__(orch)(state)
        assert lookup[1]["date"] == PAST_DATE
        assert EXPECTED_THIS_YEAR in lookup[1]["snippet"]

    def test_compose_prompt_shows_date_and_rule(self):
        from reasoning.prompts.writer import WriterPromptBuilder

        block = WriterPromptBuilder()._render_evidence_lookup_block({
            1: {
                "title": "流感疫苗開打",
                "site": "中央社",
                "url": "https://news.example/1",
                "snippet": EXPECTED_THIS_YEAR + "流感疫苗10月開打",
                "date": PAST_DATE,
            }
        })
        assert PAST_DATE in block
        assert EXPECTED_THIS_YEAR in block
        assert TEMPORAL_ANCHOR_RULE in block

    def test_lookup_without_date_renders_as_before(self):
        from reasoning.prompts.writer import WriterPromptBuilder

        block = WriterPromptBuilder()._render_evidence_lookup_block({
            1: {"title": "t", "site": "s", "url": "u", "snippet": "內容"}
        })
        # 規則行以外不出現「發布」字樣（缺日期 → 渲染與修改前一致）
        assert "發布" not in block.split(TEMPORAL_ANCHOR_RULE)[0]


# ===== LR：搜尋結果 → analyst，evidence 視圖 → writer / critic =====


class TestLRSearchResultLine:

    def test_line_carries_date_and_anchored_text(self):
        from reasoning.live_research.loop_engine import _format_search_result_line

        line = _format_search_result_line(
            7, "流感疫苗開打", ARTICLE_TEXT, "https://news.example/1", PAST_DATE
        )
        assert "[7]" in line
        assert PAST_DATE in line
        assert EXPECTED_THIS_YEAR in line
        assert "https://news.example/1" in line

    def test_line_without_date_is_unchanged_shape(self):
        from reasoning.live_research.loop_engine import _format_search_result_line

        line = _format_search_result_line(7, "標題", ARTICLE_TEXT, "u", None)
        assert line == "[7] 標題" + chr(10) + ARTICLE_TEXT + chr(10) + "URL: u" + chr(10)


class TestLRGroundingEvidenceView:

    def _entry(self, published_at):
        from reasoning.schemas_live import EvidencePoolEntry

        return EvidencePoolEntry(
            evidence_id=1,
            title="流感疫苗開打",
            url="https://news.example/1",
            source_domain="cna.com.tw",
            snippet=ARTICLE_TEXT,
            iteration_origin=1,
            published_at=published_at,
        )

    def test_view_is_anchored(self):
        from reasoning.schemas_live import render_grounding_evidence_view

        view = render_grounding_evidence_view(
            chapter_eids=[1],
            evidence_usage={},
            evidence_pool={1: self._entry(PAST_DATE)},
            prior_grounded_entities=[],
        )
        assert PAST_DATE in view
        assert EXPECTED_THIS_YEAR in view

    def test_view_without_date_untouched(self):
        from reasoning.schemas_live import render_grounding_evidence_view

        view = render_grounding_evidence_view(
            chapter_eids=[1],
            evidence_usage={},
            evidence_pool={1: self._entry(None)},
            prior_grounded_entities=[],
        )
        assert "今年（" not in view


class TestLRWriterEvidenceBlock:

    def test_section_prompt_evidence_block_is_anchored(self):
        from reasoning.prompts.writer import WriterPromptBuilder
        from reasoning.schemas_live import EvidencePoolEntry

        entry = EvidencePoolEntry(
            evidence_id=1,
            title="流感疫苗開打",
            url="https://news.example/1",
            source_domain="cna.com.tw",
            snippet=ARTICLE_TEXT,
            iteration_origin=1,
            published_at=PAST_DATE,
        )
        prompt = WriterPromptBuilder().build_section_compose_prompt(
            section_title="第一章",
            section_outline="大綱",
            relevant_findings="發現",
            analyst_citations=[1],
            evidence_lookup={1: entry},
        )
        assert PAST_DATE in prompt
        assert EXPECTED_THIS_YEAR in prompt
        assert TEMPORAL_ANCHOR_RULE in prompt


# ===== 搜尋（generate / summarize）：request.answers =====


class TestRequestAnswersAnchoring:

    def test_answers_are_anchored_per_article(self):
        from core.prompts import get_prompt_variable_value

        handler = MagicMock()
        handler.final_ranked_answers = [
            {
                "url": "https://news.example/1",
                "site": "中央社",
                "name": "流感疫苗開打",
                "ranking": {"description": ARTICLE_TEXT, "score": 90},
                "schema_object": {"datePublished": PAST_DATE + "T08:00:00Z"},
            },
            {
                "url": "https://news.example/2",
                "site": "中央社",
                "name": "無日期",
                "ranking": {"description": ARTICLE_TEXT, "score": 80},
                "schema_object": {},
            },
        ]
        value = get_prompt_variable_value("request.answers", handler)
        assert EXPECTED_THIS_YEAR in value           # 有日期 → 錨定
        assert value.count("今年（") == 1             # 無日期那筆不猜年份
        assert value.startswith("[") and value.endswith("]")


# ===== list 模式：ranking 產出的卡片摘要 =====


class TestRankingPromptAnchoring:
    """卡片摘要是使用者直接讀到的字，也是 summarize 的素材 —— 送進 ranking LLM 的
    報導內文必須先錨定，否則摘要會把 2020 年的「今年」寫成當前年。"""

    @staticmethod
    def _ranking():
        import asyncio
        import threading

        from core.ranking import Ranking

        class _H:
            required_item_type = None
            generate_mode = "list"
            query_params = {}
            query = "流感疫苗"
            site = "test_site"
            item_type = "Item"

            def __init__(self):
                self.connection_alive_event = threading.Event()
                self.connection_alive_event.set()
                self.pre_checks_done_event = asyncio.Event()
                self.pre_checks_done_event.set()
                self.final_ranked_answers = []

        r = object.__new__(Ranking)
        r.ranking_type = Ranking.REGULAR_TRACK
        r.ranking_type_str = "REGULAR_TRACK"
        r.handler = _H()
        r.level = "low"
        r.items = []
        r.num_results_sent = 0
        r.rankedAnswers = []
        r._sent_title_keys = set()
        return r

    def _run(self, schema_json):
        import asyncio
        import json
        from unittest.mock import patch

        import core.ranking as rmod
        from core.ranking import Ranking

        captured = {}

        async def _ask(prompt, ans_struc, level=None, query_params=None):
            captured["prompt"] = prompt
            return {"score": 80, "subject_match": "same", "description": "ok"}

        item = {"url": "https://news.example/1", "title": "流感疫苗開打", "site": "中央社",
                "schema_json": schema_json, "retrieval_scores": {}, "vector": None}
        with patch.object(rmod, "ask_llm", _ask),              patch.object(Ranking, "get_ranking_prompt",
                          lambda self: ("{item.description}", {"score": "0-100", "description": "d"})):
            asyncio.run(self._ranking().rankItem(item))
        return captured["prompt"]

    def test_article_text_is_anchored_before_ranking(self):
        import json

        prompt = self._run(json.dumps(
            {"description": ARTICLE_TEXT, "datePublished": PAST_DATE}, ensure_ascii=False))
        assert EXPECTED_THIS_YEAR in prompt

    def test_missing_date_leaves_text_untouched(self):
        import json

        prompt = self._run(json.dumps({"description": ARTICLE_TEXT}, ensure_ascii=False))
        assert "今年（" not in prompt

    def test_broken_schema_json_degrades_without_crash(self):
        from core.ranking import _item_date_published

        assert _item_date_published("not-json") == ""
        assert _item_date_published(None) == ""


class TestBothRankItemCallSitesAnchor:
    """同批同病種：list 模式（core/ranking.py）與 generate 模式
    （methods/generate_answer.py）各有一個 rankItem，兩個都是同一個出口。

    structural 檢查——generate_answer.rankItem 要一整個 NLWebHandler 才跑得起來，
    行為由上面 core/ranking.py 那組代表；這條負責「有人把某個呼叫點刪掉就會紅」。
    """

    def test_both_rank_item_bodies_call_the_anchor(self):
        import ast

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for relpath in ("core/ranking.py", "methods/generate_answer.py"):
            src = open(os.path.join(base, relpath), encoding="utf-8").read()
            found = False
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "rankItem":
                    assert "annotate_relative_years" in ast.dump(node), (
                        relpath + "::rankItem 沒做發布日期錨定 —— 該出口漏接"
                    )
                    found = True
                    break
            assert found, relpath + " 找不到 rankItem"


# ===== prompts.xml：產文 prompt 的規則（第二道防線） =====


class TestPromptsXmlRule:

    TARGETS = [
        "SynthesizePromptForGenerate",
        "SummarizeResultsPrompt",
        "DescriptionPromptForGenerate",
        "RankingPromptForGenerate",
        "RankingPrompt",
    ]

    def _prompt_bodies(self):
        from xml.etree import ElementTree as ET

        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "prompts.xml")
        )
        tree = ET.parse(path)
        ns = "{http://nlweb.ai/base}"
        bodies = {}
        for node in tree.iter(ns + "Prompt"):
            ref = node.get("ref")
            body = node.findtext(ns + "promptString") or ""
            bodies.setdefault(ref, []).append(body)
        return bodies

    def test_every_answer_writing_prompt_states_the_rule(self):
        bodies = self._prompt_bodies()
        for ref in self.TARGETS:
            assert ref in bodies, ref + " 不在 prompts.xml"
            for body in bodies[ref]:
                # 中英兩區措辭不同，鎖「以該報導自己的發布日期為基準」這個語意錨點
                assert ("該篇報導自己的發布日期" in body
                        or "OWN publication date" in body), ref
