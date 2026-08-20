"""票 2026-07-28-f Fix 3：DR compose 接 evidence_lookup（治假引用——斷點 3）。

Writer 在標準 / plan 兩條 compose 路徑都看得到每個白名單 [N] 的真實
title/site/URL/snippet + 「內文必須真的由該來源支持」紀律行；
evidence_lookup=None 時 prompt 與現狀完全相同（backward compat）。
"""

import json
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reasoning.prompts.writer import WriterPromptBuilder
from reasoning.orchestrator import DeepResearchOrchestrator
from reasoning.research_state import ResearchState


BLOCK_HEADER = "白名單 ID ↔ 真實來源對照"

LOOKUP = {
    1: {
        "title": "[維基百科] 李登輝",
        "site": "Wikipedia",
        "url": "https://zh.wikipedia.org/wiki/李登輝",
        "snippet": "李登輝，中華民國政治人物……",
    },
    3: {
        "title": "邱啟新談都市規劃",
        "site": "自由時報",
        "url": "https://news.example/1",
        "snippet": "台大副教授邱啟新表示……",
    },
}


def _critic_review(status="PASS"):
    r = MagicMock()
    r.status = status
    r.mode_compliance = "OK"
    r.critique = "內容合理"
    r.suggestions = []
    r.logical_gaps = []
    r.source_issues = []
    return r


def _plan():
    p = MagicMock()
    p.outline = "## 第一章：背景"
    p.estimated_length = 2000
    p.key_arguments = ["論點A"]
    return p


# === Prompt builder 渲染 ===


class TestComposePromptEvidenceBlock:

    def test_standard_compose_renders_lookup(self):
        b = WriterPromptBuilder()
        p = b.build_compose_prompt(
            analyst_draft="草稿內容",
            critic_review=_critic_review(),
            analyst_citations=[1, 3],
            mode="discovery",
            user_query="q",
            suggested_confidence="High",
            evidence_lookup=LOOKUP,
        )
        assert BLOCK_HEADER in p
        assert "邱啟新談都市規劃" in p
        assert "[維基百科] 李登輝" in p
        assert "https://news.example/1" in p
        assert "必須真的由該來源摘要支持" in p

    def test_standard_compose_none_is_backward_compatible(self):
        b = WriterPromptBuilder()
        p = b.build_compose_prompt(
            analyst_draft="草稿內容",
            critic_review=_critic_review(),
            analyst_citations=[1, 3],
            mode="discovery",
            user_query="q",
            suggested_confidence="High",
        )
        assert BLOCK_HEADER not in p

    def test_plan_compose_renders_lookup(self):
        """prod 主路徑（plan_and_write=true → build_compose_prompt_with_plan）。"""
        b = WriterPromptBuilder()
        p = b.build_compose_prompt_with_plan(
            analyst_draft="草稿內容",
            analyst_citations=[1, 3],
            plan=_plan(),
            evidence_lookup=LOOKUP,
        )
        assert BLOCK_HEADER in p
        assert "邱啟新談都市規劃" in p
        assert "必須真的由該來源摘要支持" in p

    def test_plan_compose_none_is_backward_compatible(self):
        b = WriterPromptBuilder()
        p = b.build_compose_prompt_with_plan(
            analyst_draft="草稿內容",
            analyst_citations=[1, 3],
            plan=_plan(),
        )
        assert BLOCK_HEADER not in p


# === Agent 透傳 ===


class TestComposePassesLookup:

    @pytest.mark.asyncio
    async def test_standard_path_passes_lookup(self):
        from reasoning.agents.writer import WriterAgent

        writer = WriterAgent(handler=MagicMock(), timeout=10)
        writer.call_llm_validated = AsyncMock(return_value=(MagicMock(), 0, False))
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return "PROMPT"

        with patch.object(
            writer.prompt_builder, "build_compose_prompt", side_effect=spy
        ):
            await writer.compose(
                analyst_draft="d",
                critic_review=_critic_review(),
                analyst_citations=[1, 3],
                mode="discovery",
                user_query="q",
                evidence_lookup=LOOKUP,
            )
        assert captured["evidence_lookup"] is LOOKUP

    @pytest.mark.asyncio
    async def test_plan_path_passes_lookup(self):
        from reasoning.agents.writer import WriterAgent

        writer = WriterAgent(handler=MagicMock(), timeout=10)
        writer.call_llm_validated = AsyncMock(return_value=(MagicMock(), 0, False))
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return "PROMPT"

        with patch.object(
            writer.prompt_builder, "build_compose_prompt_with_plan", side_effect=spy
        ):
            await writer.compose(
                analyst_draft="d",
                critic_review=_critic_review(),
                analyst_citations=[1, 3],
                mode="discovery",
                user_query="q",
                plan=_plan(),
                evidence_lookup=LOOKUP,
            )
        assert captured["evidence_lookup"] is LOOKUP

    @pytest.mark.asyncio
    async def test_no_lookup_backward_compatible(self):
        """既有 caller（不帶 evidence_lookup）不 break。"""
        from reasoning.agents.writer import WriterAgent

        writer = WriterAgent(handler=MagicMock(), timeout=10)
        mock_result = MagicMock()
        writer.call_llm_validated = AsyncMock(return_value=(mock_result, 0, False))
        result = await writer.compose(
            analyst_draft="d",
            critic_review=_critic_review(),
            analyst_citations=[1],
            mode="discovery",
            user_query="q",
        )
        assert result is mock_result


# === Orchestrator lookup builder ===


class TestBuildWriterEvidenceLookup:

    def _orch(self):
        orch = MagicMock()
        orch.logger = MagicMock()
        orch._extract_item_fields = (
            DeepResearchOrchestrator._extract_item_fields.__get__(orch)
        )
        # 發布日期錨定：lookup 會抽 datePublished，需綁真方法（否則拿到 MagicMock）
        orch._extract_item_date = (
            DeepResearchOrchestrator._extract_item_date.__get__(orch)
        )
        orch._build_writer_evidence_lookup = (
            DeepResearchOrchestrator._build_writer_evidence_lookup.__get__(orch)
        )
        return orch

    def _state(self, **overrides):
        defaults = dict(query="q", mode="discovery", items=[])
        defaults.update(overrides)
        return ResearchState(**defaults)

    def test_shape_aware_dict_and_row_and_phantom_skip(self):
        orch = self._orch()
        row = [
            "https://news.example/2",
            json.dumps({"description": "邱啟新出席公聽會指出……"}, ensure_ascii=False),
            "邱啟新出席公聽會",
            "聯合報",
        ]
        state = self._state(
            analyst_citations=[1, 2, 9],  # 9 不在 source_map（防禦性跳過）
            source_map={
                1: {"url": "https://news.example/1", "title": "邱啟新談都市規劃",
                    "site": "自由時報", "description": "台大副教授邱啟新表示……"},
                2: row,
            },
        )
        lookup = orch._build_writer_evidence_lookup(state)
        assert set(lookup) == {1, 2}
        assert lookup[1]["title"] == "邱啟新談都市規劃"
        assert lookup[1]["site"] == "自由時報"
        assert lookup[1]["url"] == "https://news.example/1"
        assert lookup[2]["title"] == "邱啟新出席公聽會"
        assert lookup[2]["site"] == "聯合報"
        assert lookup[2]["url"] == "https://news.example/2"
        assert "公聽會指出" in lookup[2]["snippet"]

    def test_snippet_truncated_to_200(self):
        orch = self._orch()
        state = self._state(
            analyst_citations=[1],
            source_map={1: {"url": "u", "title": "T", "site": "S",
                            "description": "很長" * 300}},
        )
        lookup = orch._build_writer_evidence_lookup(state)
        assert len(lookup[1]["snippet"]) == 200


# === _phase_writer 接線 ===


class TestPhaseWriterWiring:

    @pytest.mark.asyncio
    async def test_phase_writer_passes_lookup_to_compose(self):
        orch = MagicMock()
        orch.logger = MagicMock()
        orch._check_connection = MagicMock()
        orch._send_progress = AsyncMock()
        orch._emit_phase_event = AsyncMock()
        orch._extract_item_fields = (
            DeepResearchOrchestrator._extract_item_fields.__get__(orch)
        )
        # 發布日期錨定：lookup 會抽 datePublished，需綁真方法（否則拿到 MagicMock）
        orch._extract_item_date = (
            DeepResearchOrchestrator._extract_item_date.__get__(orch)
        )
        orch._build_writer_evidence_lookup = (
            DeepResearchOrchestrator._build_writer_evidence_lookup.__get__(orch)
        )
        final_report = MagicMock()
        final_report.sources_used = [1]  # ⊆ analyst_citations → 不觸發 guard 修正
        orch.writer = MagicMock()
        orch.writer.compose = AsyncMock(return_value=final_report)

        state = ResearchState(
            query="q", mode="discovery", items=[],
            draft="草稿內容",
            review=_critic_review(),
            response=MagicMock(),
            analyst_citations=[1],
            source_map={1: {"url": "u", "title": "T", "site": "S",
                            "description": "D"}},
            iteration_logger=MagicMock(),
            tracer=None,
        )

        with patch("reasoning.orchestrator.CONFIG") as cfg:
            cfg.reasoning_params.get.return_value = {}  # features={} → plan_and_write=False
            result = await DeepResearchOrchestrator._phase_writer(orch, state)

        kwargs = orch.writer.compose.await_args.kwargs
        # date 欄位為發布日期錨定所加（source_map 無 datePublished → 空字串）
        assert kwargs["evidence_lookup"] == {
            1: {"title": "T", "site": "S", "url": "u", "snippet": "D", "date": ""}
        }
        assert result.final_report is final_report
