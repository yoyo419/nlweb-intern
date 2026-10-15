"""
Writer Agent - Final report formatting for the Actor-Critic system.
"""

import time
from typing import TYPE_CHECKING, Dict, Any, List, Optional

if TYPE_CHECKING:
    from reasoning.schemas_live import EvidencePoolEntry
from reasoning.agents.base import BaseReasoningAgent
from reasoning.schemas import WriterComposeOutput, CriticReviewOutput
from reasoning.prompts.writer import WriterPromptBuilder


class WriterAgent(BaseReasoningAgent):
    """
    Writer Agent responsible for formatting final reports.

    The Writer takes approved drafts and formats them into polished,
    well-structured reports with proper citations and formatting.
    """

    def __init__(self, handler, timeout: int = 45):
        """
        Initialize Writer Agent.

        Args:
            handler: Request handler with LLM configuration
            timeout: Timeout in seconds for LLM calls
        """
        super().__init__(
            handler=handler,
            agent_name="writer",
            timeout=timeout,
            max_retries=3
        )
        self.prompt_builder = WriterPromptBuilder()

    async def plan(
        self,
        analyst_draft: str,
        critic_review: CriticReviewOutput,
        user_query: str,
        target_length: int = 2000
    ):
        """
        Generate outline plan for long-form report (Phase 3).

        Args:
            analyst_draft: The Analyst's draft
            critic_review: Critic's feedback
            user_query: Original user query
            target_length: Target word count (default 2000)

        Returns:
            WriterPlanOutput with outline and key arguments
        """
        from reasoning.schemas_enhanced import WriterPlanOutput

        # RSN-8: Use WriterPromptBuilder instead of inline prompt duplication
        prompt = self.prompt_builder.build_plan_prompt(
            analyst_draft=analyst_draft,
            critic_review=critic_review,
            user_query=user_query,
            target_length=target_length
        )

        result, retry_count, fallback_used = await self.call_llm_validated(
            prompt=prompt,
            response_schema=WriterPlanOutput,
            level="high"  # Use high quality for planning
        )

        # Log TypeAgent metrics for analytics
        self.logger.debug(f"TypeAgent metrics (plan): retries={retry_count}, fallback={fallback_used}")

        self.logger.info(f"Plan generated: {len(result.key_arguments)} key arguments, est. {result.estimated_length} words")
        return result

    async def compose(
        self,
        analyst_draft: str,
        critic_review: CriticReviewOutput,
        analyst_citations: List[int],
        mode: str,
        user_query: str,
        plan = None,  # Optional WriterPlanOutput from plan() method (Phase 3)
        evidence_lookup: Optional[Dict[int, Dict[str, str]]] = None,
    ) -> WriterComposeOutput:
        """
        Compose final report, optionally using pre-generated plan.

        Args:
            analyst_draft: Draft content from Analyst
            critic_review: Review from Critic with validated schema
            analyst_citations: Whitelist of citation IDs from Analyst (防幻覺機制)
            mode: Research mode (strict, discovery, monitor)
            user_query: Original user query
            plan: Optional WriterPlanOutput from plan() method (Phase 3)
            evidence_lookup: 白名單 ID → 真實來源明細（title/site/url/snippet）。
                Writer prompt 看到 [N] 的真實內容，防內文與引用脫鉤（票 2026-07-28-f）。
                None 時 fallback 現狀（只看白名單範圍）。

        Returns:
            WriterComposeOutput with validated schema
        """
        # Build suggested confidence level based on Critic status
        suggested_confidence = self.prompt_builder.map_status_to_confidence(critic_review.status)

        if plan:
            # RSN-8: Use WriterPromptBuilder instead of inline prompt duplication
            compose_prompt = self.prompt_builder.build_compose_prompt_with_plan(
                analyst_draft=analyst_draft,
                analyst_citations=analyst_citations,
                plan=plan,
                evidence_lookup=evidence_lookup,
            )
        else:
            # Standard mode (existing prompt)
            compose_prompt = self.prompt_builder.build_compose_prompt(
                analyst_draft=analyst_draft,
                critic_review=critic_review,
                analyst_citations=analyst_citations,
                mode=mode,
                user_query=user_query,
                suggested_confidence=suggested_confidence,
                evidence_lookup=evidence_lookup,
            )

        # Call LLM with validation
        result, retry_count, fallback_used = await self.call_llm_validated(
            prompt=compose_prompt,
            response_schema=WriterComposeOutput,
            level="high"
        )

        # Log TypeAgent metrics for analytics
        self.logger.debug(f"TypeAgent metrics (compose): retries={retry_count}, fallback={fallback_used}")

        return result

    async def compose_section(
        self,
        section_title: str,
        section_outline: str,
        relevant_findings: str,
        analyst_citations: List[int],
        style_features=None,  # Optional StyleAnalysisOutput
        format_spec: Optional[str] = None,
        context_map_summary: Optional[str] = None,
        citation_format: Optional[str] = None,
        evidence_lookup: Optional[Dict[int, 'EvidencePoolEntry']] = None,  # noqa: F821
        writer_evidence_view: Optional[str] = None,   # P2 W7：全 pool grounding 視圖（W5 組）
        is_chapter_override: bool = False,
        book_outline: Optional['BookOutline'] = None,  # noqa: F821
        current_chapter_index: int = 0,
        previous_chapter_summary: str = "",
        special_elements_for_chapter: Optional[List[Dict[str, str]]] = None,
        # Plan: lr-user-voice-container-and-4-fixes (Fix I-1)
        revise_instruction: Optional[str] = None,
        prior_section_content: Optional[str] = None,
        # Track A (sprint 2026-05-28) Task 5: entity grounding guard auto-rewrite
        ungrounded_entities_revision: Optional[List[str]] = None,
        # Track A (sprint 2026-05-28) Task 7: cross-chapter coherence
        prior_used_entities: Optional[List[str]] = None,
        # B (Cayenne cross-section): synthesis 章注入所有前章摘要（不只 entity 名稱）
        all_prior_chapter_summaries: Optional[List[str]] = None,
        # Track E (sprint 2026-05-28) E5: 強制時間約束 BINDING block
        time_constraint: Optional['TimeRange'] = None,  # noqa: F821
        # 模塊5 Task 5 (calibration 通道 B): per-chapter evidence 充分度信號透傳
        evidence_sufficiency: Optional[str] = None,
        # Task 3 (DR-parity): KG 摘要透傳給 prompt builder（全圖背景脈絡，非引用來源）。
        knowledge_graph: Optional['KnowledgeGraph'] = None,  # noqa: F821
        # 報告行文改版（2026-08-28）：骨架 registry + 並排比較欄位透傳。
        report_shape: Optional[str] = None,
        comparison_fields: Optional[List[str]] = None,
        # 篇幅與跨章重複（2026-10-15）：前面章節已用過的來源 → 章號。
        prior_source_chapters: Optional[Dict[int, List[int]]] = None,
    ) -> 'LiveWriterSectionOutput':  # noqa: F821
        """
        在 Live Research 分段模式下撰寫單一章節。

        Args:
            section_title: 章節標題
            section_outline: 本章節的大綱
            relevant_findings: 與本章節相關的發現
            analyst_citations: 引用 ID 白名單
            style_features: StyleAnalysisOutput（可選）
            format_spec: 格式規格（由使用者在 Stage 4 指定）
            context_map_summary: ContextMap 摘要（可選）
            citation_format: 引用格式 enum 'author_year'/'numeric'/'footnote'/'none'。
                若為 None，會自動從 style_features.citation_format 取用（fallback: 'numeric'）。
            evidence_lookup: 白名單 ID 對應的真實 evidence dict（解決 phantom citation）。
                Writer prompt 看到 [N] → 真實 title/URL/snippet，避免亂填。
                None 時 fallback 為現狀（只看白名單範圍，不看 URL）。

        Returns:
            LiveWriterSectionOutput with section content and metadata
        """
        from reasoning.schemas_live import LiveWriterSectionOutput

        prompt = self.prompt_builder.build_section_compose_prompt(
            section_title=section_title,
            section_outline=section_outline,
            relevant_findings=relevant_findings,
            analyst_citations=analyst_citations,
            style_features=style_features,
            format_spec=format_spec,
            context_map_summary=context_map_summary,
            citation_format=citation_format,
            evidence_lookup=evidence_lookup,
            writer_evidence_view=writer_evidence_view,   # P2 W7：透傳全 pool 視圖
            is_chapter_override=is_chapter_override,
            book_outline=book_outline,
            current_chapter_index=current_chapter_index,
            previous_chapter_summary=previous_chapter_summary,
            special_elements_for_chapter=special_elements_for_chapter,
            # Plan: lr-user-voice-container-and-4-fixes (Fix I-1)
            revise_instruction=revise_instruction,
            prior_section_content=prior_section_content,
            # Track A (sprint 2026-05-28) Task 5
            ungrounded_entities_revision=ungrounded_entities_revision,
            # Track A (sprint 2026-05-28) Task 7
            prior_used_entities=prior_used_entities,
            # B (Cayenne cross-section): synthesis 章前章摘要注入
            all_prior_chapter_summaries=all_prior_chapter_summaries,
            # Track E (sprint 2026-05-28) E5
            time_constraint=time_constraint,
            # 模塊5 Task 5 (calibration 通道 B)
            evidence_sufficiency=evidence_sufficiency,
            # Task 3 (DR-parity): KG 摘要透傳
            knowledge_graph=knowledge_graph,
            # 報告行文改版（2026-08-28）
            report_shape=report_shape,
            comparison_fields=comparison_fields,
            prior_source_chapters=prior_source_chapters,
        )

        _llm_start = time.perf_counter()
        self.logger.info(
            f"[LIVE RESEARCH] Writer LLM call start: schema=LiveWriterSectionOutput "
            f"section={section_title!r} prompt_len={len(prompt)} timeout={self.timeout}s"
        )
        try:
            result, retry_count, fallback_used = await self.call_llm_validated(
                prompt=prompt,
                response_schema=LiveWriterSectionOutput,
                level="high"
            )
        except Exception as e:
            _llm_elapsed = time.perf_counter() - _llm_start
            self.logger.error(
                f"[LIVE RESEARCH] Writer LLM call failed: elapsed={_llm_elapsed:.2f}s "
                f"section={section_title!r} error={type(e).__name__}: {e}"
            )
            raise  # 不 silent swallow

        _llm_elapsed = time.perf_counter() - _llm_start
        self.logger.info(
            f"[LIVE RESEARCH] Writer LLM call done: elapsed={_llm_elapsed:.2f}s "
            f"section={section_title!r} retries={retry_count} fallback={fallback_used} "
            f"output_type={type(result).__name__}"
        )

        self.logger.debug(
            f"TypeAgent metrics (compose_section '{section_title}'): "
            f"retries={retry_count}, fallback={fallback_used}"
        )
        self.logger.info(
            f"Section composed: '{section_title}', "
            f"sources={result.sources_used}, confidence={result.confidence_level}"
        )

        return result

