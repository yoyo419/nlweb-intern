# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Deep Research Handler - Full Handler for Reasoning Module

This handler extends NLWebHandler to reuse all infrastructure (retrieval, temporal detection, etc.)
while adding multi-agent reasoning capabilities.

Future implementation will include:
- DeepResearchOrchestrator
- Analyst, Critic, Writer Agents
- Actor-Critic Loop
- Multi-tier source filtering
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from core.baseHandler import NLWebHandler
from misc.logger.logging_config_helper import get_configured_logger
from reasoning.prompts.clarification import build_clarification_prompt

logger = get_configured_logger("deep_research_handler")


class DeepResearchHandler(NLWebHandler):
    """
    Full handler for deep research mode.

    Inherits retrieval/ranking infrastructure from NLWebHandler.
    Adds multi-agent reasoning with mode detection (strict/discovery/monitor).
    """

    def __init__(self, query_params, http_handler):
        """
        Initialize Deep Research Handler.

        Args:
            query_params: Query parameters from API request
            http_handler: HTTP streaming wrapper
        """
        # Call parent constructor - sets up all infrastructure
        super().__init__(query_params, http_handler)

        self.research_mode = None  # Will be set in prepare()

        # Phase KG: Extract enable_kg parameter from query_params, fallback to config
        from core.config import CONFIG
        config_enable_kg = CONFIG.reasoning_params.get("features", {}).get("knowledge_graph_generation", False)
        enable_kg_param = query_params.get('enable_kg', None)
        if enable_kg_param is not None:
            self.enable_kg = enable_kg_param in [True, 'true', 'True', '1']
        else:
            self.enable_kg = config_enable_kg

        # Stage 5: Extract enable_web_search parameter (default: False)
        enable_web_search_param = query_params.get('enable_web_search', 'false')
        self.enable_web_search = enable_web_search_param in [True, 'true', 'True', '1']

        # Task 6: Non-blocking research support
        self._research_task: Optional[asyncio.Task] = None
        self._soft_interrupt_event = asyncio.Event()

        # 層3（B1/B5）server-side persist：session UUID + 新建旗標 + 前端當前 session id
        #   在 __init__ 初始化，避免 clarification 早退路徑 AttributeError。
        self.dr_session_id = None
        self._dr_session_is_new = False
        from core.utils.utils import get_param
        self.loaded_session_id = get_param(query_params, "loaded_session_id", str, "")

        logger.info("DeepResearchHandler initialized")
        logger.info(f"  Query: {self.query}")
        logger.info(f"  Site: {self.site}")
        logger.info(f"  Enable KG: {self.enable_kg}")
        logger.info(f"  Enable Web Search: {self.enable_web_search}")

        # Future: Initialize Orchestrator
        # self.orchestrator = DeepResearchOrchestrator(...)

    async def runQuery(self):
        """
        Main entry point for query execution.
        Follows standard handler pattern.
        """
        logger.info(f"[DEEP RESEARCH] Starting query execution for: {self.query}")

        try:
            # Call parent prepare() - gets retrieval, temporal detection, etc.
            await self.prepare()

            if self.query_done:
                logger.info("[DEEP RESEARCH] Query done prematurely")
                return self.return_value

            # Execute deep research
            await self.execute_deep_research()

            self.return_value["conversation_id"] = self.conversation_id
            logger.info(f"[DEEP RESEARCH] Query execution completed")

            return self.return_value

        except Exception as e:
            logger.error(f"[DEEP RESEARCH] Error in runQuery: {e}", exc_info=True)
            raise

    async def prepare(self):
        """
        Run pre-checks and retrieval.
        Extends parent prepare() to add mode detection and clarification check.
        """
        # Call parent prepare() - handles:
        # - Decontextualization
        # - Query rewrite
        # - Tool selection (will be skipped due to generate_mode)
        # - Memory retrieval
        # - Temporal detection
        # - Vector search with date filtering
        await super().prepare()

        # Phase 4: Check if clarification is needed
        # This happens after temporal detection, so we can check the results
        needs_clarification = await self._check_clarification_needed()
        if needs_clarification:
            # Clarification request sent, early return
            # Set return_value to indicate clarification is pending
            self.return_value.update({
                'answer': '',  # Empty answer - clarification needed
                'status': 'clarification_pending',
                'message': 'Waiting for user clarification'
            })
            logger.info("[DEEP RESEARCH] Clarification required, waiting for user input")
            return

        # Additional: Detect research mode
        self.research_mode = await self._detect_research_mode()
        logger.info(f"[DEEP RESEARCH] Mode detected: {self.research_mode.upper()}")

    async def _detect_research_mode(self) -> str:
        """
        Get research mode from frontend request.

        Returns:
            'strict' | 'discovery' | 'monitor'
        """
        # Use frontend-specified mode
        if 'research_mode' in self.query_params:
            user_mode = self.query_params['research_mode']
            if user_mode in ['strict', 'discovery', 'monitor']:
                logger.info(f"[DEEP RESEARCH] Using frontend mode: {user_mode}")
                return user_mode

        # Default if not specified
        logger.info("[DEEP RESEARCH] No mode specified, using default: discovery")
        return 'discovery'

    async def execute_deep_research(self):
        """
        Execute deep research using reasoning orchestrator.
        If reasoning module disabled, falls back to mock implementation.

        When nonblocking_research=true AND composable_pipeline=true, the research
        runs as a named asyncio.Task that can be cancelled via _research_task.cancel().
        The HTTP connection stays open (we still await the task) but the task can be
        interrupted by soft_interrupt_event or client disconnect.
        """
        from core.config import CONFIG

        # Access pre-filtered items from parent's prepare()
        items = self.final_retrieved_items

        logger.info(f"[DEEP RESEARCH] Executing {self.research_mode} mode")
        logger.info(f"[DEEP RESEARCH] Retrieved items: {len(items)}")

        # Get temporal context from parent
        temporal_context = self._get_temporal_context()
        logger.info(f"[DEEP RESEARCH] Temporal context: {temporal_context}")

        # 層3（B5）：決定這次寫哪個 session UUID（優先採用前端當前 session）。
        # 放在 run_research 之前——clarification pending 早退路徑不會進到 execute_deep_research
        # （prepare() 設 query_done → runQuery 早 return），故不會誤建 row。
        self.dr_session_id = await self._create_dr_session()
        # 層3（B1）：只有「新建 row」時才 run 前 emit UUID handshake——
        # 採用現有 session 時 UUID=前端已知的當前 session，不需推。
        # （照抄 LR live_research.py:134-151 範本；用 self.http_handler，非 wrapper。）
        if getattr(self, "_dr_session_is_new", False) and self.http_handler is not None:
            try:
                await self.http_handler.write_stream({
                    "message_type": "deep_research_session_created",
                    "session_id": self.dr_session_id,
                })
                logger.info(f"[DEEP RESEARCH] Sent session_created event (new row): {self.dr_session_id}")
            except Exception as e:
                logger.warning(f"[DEEP RESEARCH] Could not send session_created event: {e}")

        # Feature flag check
        if not CONFIG.reasoning_params.get("enabled", False):
            logger.info("[DEEP RESEARCH] Reasoning module disabled, using mock implementation")
            results = self._generate_mock_results(items, temporal_context)
        else:
            # Import and run orchestrator
            logger.info("[DEEP RESEARCH] Reasoning module enabled, using orchestrator")
            from reasoning.orchestrator import DeepResearchOrchestrator

            orchestrator = DeepResearchOrchestrator(handler=self)

            # Check if non-blocking mode is enabled
            # Requires BOTH composable_pipeline=true AND nonblocking_research=true
            enable_composable = CONFIG.reasoning_params.get("features", {}).get(
                "composable_pipeline", False
            )
            enable_nonblocking = CONFIG.reasoning_params.get("features", {}).get(
                "nonblocking_research", False
            )

            if enable_composable and enable_nonblocking:
                # Non-blocking: wrap in named task for cancellation support.
                # The HTTP connection stays open (we await the task), but
                # the task can be .cancel()'d from on_disconnect or soft interrupt.
                logger.info("[DEEP RESEARCH] Non-blocking mode: creating research task")
                self._research_task = asyncio.create_task(
                    orchestrator.run_research(
                        query=self.query,
                        mode=self.research_mode,
                        items=items,
                        temporal_context=temporal_context,
                        enable_kg=self.enable_kg,
                        enable_web_search=self.enable_web_search,
                    ),
                    name=f"research_{self.conversation_id}"
                )
                # W2 fix: add done callback to catch exceptions that would
                # otherwise be silently swallowed if await is interrupted
                self._research_task.add_done_callback(self._on_research_complete)

                try:
                    results = await self._research_task
                except asyncio.CancelledError:
                    logger.info("[DEEP RESEARCH] Research task cancelled")
                    # W1: 中斷 ≠ 完成。原本落到下方 results=[] 會 fabricate 一份空的
                    # 「成功」報告送給前端（silent fail）。改為明確標記 interrupted +
                    # 主動 SSE 告知，並提前 return，不往下送假報告。
                    self.return_value.update({
                        'status': 'interrupted',
                        'answer': '',
                        'confidence_level': 'Low',
                        'methodology_note': 'Deep Research 已中斷（未完成，無報告）',
                        'sources_used': [],
                        'items': [],
                    })
                    await self._send_research_interrupted()
                    return
                finally:
                    self._research_task = None
            else:
                # Legacy blocking path (default)
                results = await orchestrator.run_research(
                    query=self.query,
                    mode=self.research_mode,
                    items=items,
                    temporal_context=temporal_context,
                    enable_kg=self.enable_kg,
                    enable_web_search=self.enable_web_search,
                )

        # Send results using parent's message sender
        from core.schemas import create_assistant_result
        create_assistant_result(results, handler=self, send=True)

        logger.info(f"[DEEP RESEARCH] Sent {len(results)} results to frontend")

        # Generate final report for api.py
        final_report = self._generate_final_report(results, temporal_context)

        # Extract source URLs from schema_object (set by orchestrator)
        source_urls = []
        for item in results:
            schema_obj = item.get('schema_object', {})
            if 'sources_used' in schema_obj:
                source_urls.extend(schema_obj['sources_used'])

        # Update return_value with structured response
        self.return_value.update({
            'answer': final_report,
            'confidence_level': self._calculate_confidence(results),
            'methodology_note': f'Deep Research ({self.research_mode} mode)',
            'sources_used': source_urls,  # Use actual source URLs, not report URL
            'items': results  # Include full results with schema_object for Phase 4
        })

        logger.info(f"[DEEP RESEARCH] Updated return_value with final report")

        # [R8 BLOCKER2] report_obj 組裝（含 snake→camel 轉換）抽成 helper `_build_research_report_obj`——
        #   轉換本身（results[0].schema_object 的 snake_case → report_obj 的 camelCase）是實質步驟、必須有 test 覆蓋。
        report_obj = self._build_research_report_obj(final_report, source_urls, results)
        # 層3：把報告一次性 persist（B2 空覆蓋防護 + B3 讀回驗證都在 _persist 內；傳入 results 供 gate 判）
        await self._persist_research_report(self.dr_session_id, report_obj, results)
        # B1 冗餘：讓 api.py final envelope 也帶回 server UUID（非唯一送達點，run 前 event 才是主送達）
        self.return_value["dr_session_id"] = self.dr_session_id

    def _on_research_complete(self, task: asyncio.Task):
        """
        Callback when background research task completes or fails.

        W2 fix: Without this callback, if the handler itself raises before
        `await task`, the task's exception would be silently swallowed by asyncio
        ("Task exception was never retrieved"). This ensures exceptions are always
        logged and error messages are pushed to the user via SSE.
        """
        try:
            exc = task.exception()
            if exc:
                logger.error(
                    f"[DEEP RESEARCH] Background research task failed: {exc}",
                    exc_info=exc
                )
                # Push error to user via SSE (best-effort, non-blocking)
                asyncio.create_task(self._send_research_error(exc))
        except asyncio.CancelledError:
            # Normal cancellation (client disconnect or soft interrupt) -- not an error
            logger.info(f"[DEEP RESEARCH] Research task cancelled: {task.get_name()}")
        except asyncio.InvalidStateError:
            # Task not done yet -- shouldn't happen in done callback, but be safe
            pass

    async def _send_research_error(self, exc: Exception):
        """Push a research error message to the user via SSE (best-effort)."""
        try:
            if hasattr(self, 'message_sender') and self.message_sender:
                error_message = {
                    "message_type": "research_error",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                await self.message_sender.send_message(error_message)
                logger.info("[DEEP RESEARCH] Error message sent to frontend via SSE")
        except Exception as send_err:
            # SSE send itself failed -- connection probably dead, just log
            logger.warning(
                f"[DEEP RESEARCH] Failed to send error to frontend: {send_err}"
            )

    async def _send_research_interrupted(self):
        """Push a research-interrupted notice to the user via SSE (best-effort).

        W1: 中斷 ≠ 完成。不可送出空的 final_result 假裝成功（silent fail）。此處主動
        通知前端研究被中斷、未完成，前端據此顯示中斷狀態而非空報告。
        """
        try:
            if hasattr(self, 'message_sender') and self.message_sender:
                await self.message_sender.send_message({
                    "message_type": "research_interrupted",
                    "message": "Deep Research 已中斷，未產生完整報告。",
                })
                logger.info("[DEEP RESEARCH] Interrupted notice sent to frontend via SSE")
        except Exception as send_err:
            logger.warning(
                f"[DEEP RESEARCH] Failed to send interrupted notice: {send_err}"
            )

    def _get_temporal_context(self) -> Dict[str, Any]:
        """
        Package temporal metadata for reasoning module.
        Reuses parent's temporal detection.

        Returns:
            Dictionary with temporal information including user_selected flag
            for BINDING constraint in Analyst prompt.
        """
        # Check if temporal parsing was done
        temporal_range = getattr(self, 'temporal_range', None)

        context = {
            'is_temporal_query': temporal_range.get('is_temporal', False) if temporal_range else False,
            'method': temporal_range.get('method') if temporal_range else 'none',
            'start_date': temporal_range.get('start_date') if temporal_range else None,
            'end_date': temporal_range.get('end_date') if temporal_range else None,
            'start': temporal_range.get('start_date') if temporal_range else None,  # Alias for orchestrator
            'end': temporal_range.get('end_date') if temporal_range else None,  # Alias for orchestrator
            'relative_days': temporal_range.get('relative_days') if temporal_range else None,
            'current_date': datetime.now().strftime("%Y-%m-%d"),
            # NEW: User-selected time range from clarification (BINDING constraint)
            'user_selected': temporal_range.get('user_selected', False) if temporal_range else False,
            'user_choice_label': temporal_range.get('user_choice_label', '') if temporal_range else ''
        }

        if context['user_selected']:
            logger.info(f"[DEEP RESEARCH] User-selected time constraint: {context['user_choice_label']} ({context['start_date']} to {context['end_date']})")

        return context

    def _generate_mock_results(self, items: list, temporal_context: Dict) -> list:
        """
        Generate mock results for testing.
        Will be replaced by Orchestrator output.

        Args:
            items: Retrieved and filtered items from parent handler
            temporal_context: Temporal metadata

        Returns:
            List of result items in standard format
        """
        mode_descriptions = {
            'strict': 'High-accuracy fact-checking with Tier 1/2 sources only',
            'discovery': 'Comprehensive exploration across multiple sources and perspectives',
            'monitor': 'Gap detection between official statements and public sentiment'
        }

        return [{
            "@type": "Item",
            "url": f"internal://system/{self.research_mode}",
            "name": f"[MOCK] Deep Research Result - {self.research_mode.upper()} Mode",
            "site": "讀豹系統",
            "siteUrl": "internal",
            "score": 95,
            "description": (
                f"This is a placeholder result from deep_research handler.\n\n"
                f"**Query:** {self.query}\n\n"
                f"**Mode:** {self.research_mode} - {mode_descriptions.get(self.research_mode, 'Unknown')}\n\n"
                f"**Items Retrieved:** {len(items)} articles\n\n"
                f"**Temporal Context:**\n"
                f"- Is Temporal: {temporal_context['is_temporal_query']}\n"
                f"- Method: {temporal_context['method']}\n"
                f"- Date Range: {temporal_context.get('start_date', 'N/A')} to {temporal_context.get('end_date', 'N/A')}\n\n"
                f"**Future:** This will be replaced by DeepResearchOrchestrator output with:\n"
                f"- Analyst Agent findings\n"
                f"- Critic Agent validation\n"
                f"- Writer Agent synthesis\n"
                f"- Actor-Critic loop iterations"
            ),
            "schema_object": {
                "@type": "Article",
                "headline": f"Deep Research: {self.research_mode} Mode",
                "description": "Mock implementation - infrastructure testing",
                "mode": self.research_mode,
                "temporal_detected": temporal_context['is_temporal_query'],
                "num_items_retrieved": len(items)
            }
        }]

    @staticmethod
    def _count_analyzed_sources(results: list) -> Optional[int]:
        """回「這份報告實際分析了幾個來源」；取不到權威值時回 None（呼叫端據此不印假數字）。

        ⚠️ **不可用 `len(results)`**：`orchestrator.run_research()` / `run_research_rerun()`
        一律回**單一元素**的 list（`_format_result()` 只把整份報告包成一個 Item），
        故 `len(results)` 恆為 1，與報告內文的 `[1]..[N]` 引用毫無關係
        （CEO 回報症狀：報告引用了十個來源，「分析來源數」卻寫 1）。

        權威值取用順序：
          1. `schema_object["total_sources_analyzed"]`——orchestrator 以 `len(context)` 算的
             「送進 Analyst 的來源數」，也就是 citation ID 1..N 的 N（source_map 與 context 1:1）。
          2. `schema_object["sources_used"]`——citation ID → URL 陣列，缺號以空字串佔位，
             故降級時取「非空元素數」。
        兩者皆無（no-data / error / mock item 沒有這些欄位）→ None。
        """
        for item in results:
            schema_obj = item.get('schema_object') or {}
            total = schema_obj.get('total_sources_analyzed')
            # bool 是 int 的 subclass，明確排除避免 True → 1
            if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                return total

        for item in results:
            schema_obj = item.get('schema_object') or {}
            sources = schema_obj.get('sources_used')
            if isinstance(sources, list):
                return sum(1 for url in sources if isinstance(url, str) and url.strip())

        return None

    def _generate_final_report(self, results: list, temporal_context: Dict) -> str:
        """
        Generate a final markdown report from research results.

        Args:
            results: List of NLWeb Item dicts from research
            temporal_context: Temporal metadata

        Returns:
            Markdown-formatted final report
        """
        # Extract descriptions from results (which contain the actual content)
        descriptions = [item.get('description', '') for item in results]

        # Build final report
        report_parts = [
            f"# 深度研究報告：{self.query}",
        ]

        # 「分析來源數」必須來自 orchestrator 的權威計數，不是回傳 Item 數（見 _count_analyzed_sources）
        source_count = self._count_analyzed_sources(results)
        if source_count is None:
            # no-silent-fail：拿不到權威來源數就不印數字（寧可少一行也不可印錯），但要留 log
            logger.warning(
                "[DEEP RESEARCH] 無法取得分析來源數（schema_object 缺 total_sources_analyzed "
                f"與 sources_used，results={len(results)} 筆）→ 略過「分析來源數」欄位"
            )
        else:
            report_parts.append(f"\n**分析來源數：** {source_count}")

        # Add temporal context if applicable
        if temporal_context.get('is_temporal_query'):
            date_range = f"{temporal_context.get('start_date', 'N/A')} 至 {temporal_context.get('end_date', 'N/A')}"
            report_parts.append(f"\n**時間範圍：** {date_range}")

        report_parts.append("\n---\n")

        # Add research findings
        for desc in descriptions:
            report_parts.append(desc)
            report_parts.append("\n")

        return "\n".join(report_parts)

    async def _create_dr_session(self) -> str:
        """決定「這次報告寫哪個 session UUID」，回 UUID。

        B5：DR 語意=「在當前 session 裡研究」，故**優先採用前端當前 session**：
          - 前端有傳 loaded_session_id 且經 get_session 驗證屬當前 user/org → 用它（不建新，
            不分裂側欄）。self._dr_session_is_new = False（→ B1：不需 emit handshake）。
          - loaded_session_id 缺失/無效/非本人 → create_session 建新，
            self._dr_session_is_new = True（→ B1：run 前須 emit deep_research_session_created）。
        未登入 / DB 故障 → fallback bare UUID，pipeline 不被 block（no-silent-fail：fallback 時留 log）。
        """
        fallback_id = str(uuid.uuid4())
        self._dr_session_is_new = False  # 預設；真的 create 時才設 True
        user_id = getattr(self, "user_id", "") or ""
        org_id = getattr(self, "org_id", "") or ""
        if not user_id or not org_id:
            logger.info(f"[DEEP RESEARCH] No user/org ID, using bare UUID without DB session (key={fallback_id})")
            return fallback_id

        from core.session_service import SessionService
        service = SessionService()

        # B5：優先採用前端當前 session（loaded_session_id 驗過屬本人）
        # loaded_session_id 由前端 DR 請求新增帶入（Task 8 Step 0），經 get_param 讀出。
        loaded_sid = getattr(self, "loaded_session_id", "") or ""
        if loaded_sid:
            try:
                # F23：get_session owner 不匹配回 None → 只有屬本 user/org 才採用
                existing = await service.get_session(loaded_sid, user_id, org_id)
                if existing is not None:
                    logger.info(f"[DEEP RESEARCH] Adopting current session (no new row): {loaded_sid}")
                    return loaded_sid
                else:
                    logger.info(f"[DEEP RESEARCH] loaded_session_id not owned/found, will create new: {loaded_sid}")
            except Exception as e:
                logger.warning(f"[DEEP RESEARCH] get_session validate failed, will create new: {e}")

        # 沒給有效 loaded_session_id → 建新 row
        try:
            result = await service.create_session(
                user_id=user_id,
                org_id=org_id,
                title=f"深度研究：{self.query[:50]}",
            )
            session_id = result["id"]
            self._dr_session_is_new = True  # 新建 → B1 需 emit UUID handshake
            logger.info(f"[DEEP RESEARCH] Created server session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"[DEEP RESEARCH] Failed to create DB session, using bare UUID: {e}")
            return fallback_id

    def _build_research_report_obj(self, final_report: str, source_urls: list, results: list) -> dict:
        """[R8 BLOCKER2] 組 research_report persist dict——含 graph/chain/KG 的 snake→camel 轉換。

        抽 helper 的理由：snake→camel 轉換是實質步驟（前端 reload 讀 camelCase、schema_object 是
        snake_case），必須有 test 覆蓋轉換本身，不能只測「存已對的物件」（假綠）。

        來源 = results[0].schema_object（**snake_case**：argument_graph / reasoning_chain_analysis /
        knowledge_graph，orchestrator.py:2085/2089/2092 塞、kg-spec §2.1）。
        目標 report_obj = **camelCase**（前端 reload top-level restore news-search.js:2720-2721 讀
        session.researchReport.argumentGraph/.chainAnalysis、:2867 讀 serverReport.knowledgeGraph，
        且 hydrate :2602 整包搬不轉 key → 必須存 camelCase 才讀得出）。
        """
        _schema_obj = results[0].get("schema_object", {}) if results else {}
        _argument_graph = _schema_obj.get("argument_graph")            # list（node dict 陣列）或 None
        _reasoning_chain = _schema_obj.get("reasoning_chain_analysis")  # dict 或 None
        _knowledge_graph = _schema_obj.get("knowledge_graph")          # dict {entities, relationships, metadata} 或 None

        report_obj = {
            "report": final_report,
            "sources": source_urls,
            "query": self.query,
            "timestamp": int(time.time() * 1000),
            # [R4 C3] 唯一 write marker：persist 讀回驗證比對此值（非 timestamp），消除同毫秒撞縫。
            # 只是內部驗證欄位，前端 render 不讀（讀 report/sources/query/timestamp），多此欄位無害。
            "persist_marker": str(uuid.uuid4()),
        }
        # [R6 BLOCKER2] graph/chain 轉 camelCase（snake schema_object → camel report_obj）。無值則不塞
        # （保持與前端「? [...] : null」相容——restore 分支 undefined 走 null 分支）。
        if _argument_graph:
            report_obj["argumentGraph"] = _argument_graph      # camel，對齊 :2720 session.researchReport.argumentGraph
        if _reasoning_chain:
            report_obj["chainAnalysis"] = _reasoning_chain      # camel，對齊 :2721 session.researchReport.chainAnalysis
        # [R7 BLOCKER1] KG 轉 camelCase `knowledgeGraph`——存進**既有 research_report JSONB**（零 DB migration，
        #   [verified] search_sessions 無獨立 knowledge_graph 欄位）。三處 key 一致（存==hydrate 搬==reload 讀，
        #   皆 camel `knowledgeGraph`）。shape = dict {entities,relationships,metadata}（displayKnowledgeGraph 直接吃）。
        if _knowledge_graph:
            report_obj["knowledgeGraph"] = _knowledge_graph    # camel，對齊 (c) serverReport.knowledgeGraph / (b) hydrate 搬
        # Bug 1：把 rerun 所需的精簡輸入狀態（phase 1 完成後 orchestrator 掛上）塞進
        # research_report 內層，供 server 重啟後 rerun cache miss 時從 DB 重建。
        # 無子集（如 rerun 自身、或非 composable 路徑）→ 不塞（不塞 None 空殼）。
        # [verified] rerun 走 execute_rerun 不呼叫 _persist_research_report → 不覆蓋 research_report、
        # 原內層 rerunState 天然保留（update_session 整欄覆蓋只發生在主 DR run persist 時）。
        _rerun_subset = getattr(self, "_rerun_state_subset", None)
        if _rerun_subset:
            report_obj["rerunState"] = _rerun_subset
        return report_obj

    async def _persist_research_report(self, session_id: str, report_obj: dict, results: list) -> bool:
        """把報告寫進 search_sessions.research_report。回 True=已驗證寫入，False=skip/寫失敗。

        B2/F24 空覆蓋防護：**看研究是否實質成功（results 非空），不看 markdown 字串空不空**。
          斷線早退 run_research→[] 時 _generate_final_report([]) 回「非空空洞骨架」——len(results)==0 才是斷線的權威訊號。
        B3/F22/F23 silent-fail 防護 + [R4 C3] 唯一 write marker 比對：update_session 假成功（execute
          後不讀 rowcount，owner 不匹配仍回 True）——故 update 後 get_session **讀回驗證**。**[R3 should-fix 2]**
          不只比 truthiness——要比對讀回的 `research_report["persist_marker"] == report_obj["persist_marker"]`
          才算真寫入這一份；不符 → loud error + 回 False（不可 silent 說 Persisted）。
        """
        # B2：實質成功 gate——results 空（斷線早退）則 skip
        if not results:
            logger.info(f"[DEEP RESEARCH] results empty (disconnect early-exit?), skip persist "
                        f"(avoid empty-overwrite) session={session_id}")
            return False
        if not report_obj or not (report_obj.get("report") or "").strip():
            # 冗餘保險：research 成功但 report 竟空（不預期）——仍 skip 不覆蓋
            logger.warning(f"[DEEP RESEARCH] results non-empty but report string empty, skip persist "
                           f"session={session_id}")
            return False
        user_id = getattr(self, "user_id", "") or ""
        org_id = getattr(self, "org_id", "") or ""
        if not user_id or not org_id:
            logger.info(f"[DEEP RESEARCH] No user/org, skip persist session={session_id}")
            return False
        try:
            from core.session_service import SessionService
            service = SessionService()
            # F2：update_session(session_id, user_id, org_id, updates) — 4 參數！
            await service.update_session(session_id, user_id, org_id,
                                         {"research_report": report_obj})
            # B3：讀回驗證（update_session 回 True 是假成功，不可信）。F23：get_session 回 snake_case dict，
            # research_report 已反序列化為 dict；owner 不匹配 → get_session 回 None。
            verify = await service.get_session(session_id, user_id, org_id)
            persisted = verify.get("research_report") if verify else None
            if not persisted or (isinstance(persisted, dict) and not (persisted.get("report") or "").strip()):
                logger.error(f"[DEEP RESEARCH] persist readback verify FAILED (row not updated / owner mismatch?) "
                             f"session={session_id}")
                return False
            # [R3 should-fix 2 / R4 C3]：只比非空不夠——DB 若本就有舊非空 report、這次 UPDATE 命中 0 rows，
            # 讀回會撈到「舊的非空 report」→ 誤判寫成功。故比對**唯一 write marker**（persist_marker，UUID），
            # 確認撈回的正是這次寫的那份。
            expected_marker = report_obj.get("persist_marker")
            got_marker = persisted.get("persist_marker") if isinstance(persisted, dict) else None
            if expected_marker is not None and got_marker != expected_marker:
                logger.error(f"[DEEP RESEARCH] persist readback marker MISMATCH "
                             f"(readback persist_marker={got_marker} != this-write {expected_marker}; "
                             f"UPDATE likely hit 0 rows, stale report served) session={session_id}")
                return False
            logger.info(f"[DEEP RESEARCH] Persisted + readback-verified (marker matched) research_report "
                        f"session={session_id}")
            return True
        except Exception as e:
            logger.error(f"[DEEP RESEARCH] Failed to persist research_report session={session_id}: {e}")
            return False

    def _calculate_confidence(self, results: list) -> str:
        """
        取這份報告的信心水準（'High' / 'Medium' / 'Low'）。

        以 Writer/Critic 算出的值為準——`schema_object["confidence"]`
        （= `final_report.confidence_level`，由 Critic status PASS/WARN/REJECT 映射）。

        ⚠️ 與「分析來源數」同一個 root cause：舊版用 `len(results)` 當「結果數」跑啟發式
        （>=5 High / >=2 Medium / else Low），但 `run_research()` 恆回單一元素 list，
        該啟發式永遠落在 'Low'，與實際審查結果無關。

        Args:
            results: List of research result items

        Returns:
            Confidence level: 'High', 'Medium', or 'Low'
        """
        for item in results:
            schema_obj = item.get('schema_object') or {}
            confidence = schema_obj.get('confidence')
            if isinstance(confidence, str) and confidence.strip():
                return confidence.strip()

        # no-silent-fail：no-data / error / mock item 沒有 confidence 欄位——
        # 保守回 'Low'（不宣稱高信心），但要留 log 表明這是降級值而非真實評估。
        logger.warning(
            "[DEEP RESEARCH] schema_object 無 confidence 欄位"
            f"（results={len(results)} 筆）→ 降級回報 'Low'"
        )
        return 'Low'

    async def _check_clarification_needed(self) -> bool:
        """
        Check if query needs clarification before proceeding with research.

        Single LLM call to detect all ambiguities (time, scope, entity).
        Returns conversational clarification questions.

        Returns:
            True if clarification was sent (early return needed)
            False if no clarification needed (proceed normally)
        """
        # Check if clarification should be skipped (user already selected an option)
        if self.query_params.get('skip_clarification') == 'true':
            logger.info("[DEEP RESEARCH] Skipping clarification check (user already clarified)")
            return False

        # Single LLM call to detect all ambiguities
        questions = await self._detect_all_ambiguities()

        if questions:
            # Format as multi-dimensional parallel clarification
            clarification_data = {
                "query": self.query,
                "questions": questions,
                "instruction": "為了精準搜尋，請選擇以下條件",
                "submit_label": "開始搜尋"
            }

            # Send to frontend (render in conversation)
            await self._send_clarification_request(clarification_data)

            self.query_done = True
            return True

        return False

    async def _detect_all_ambiguities(self) -> list:
        """
        Single LLM call to detect all ambiguities (time, scope, entity).

        Uses extracted prompt builder for cleaner separation of concerns.

        Returns:
            List of question dicts, each with options. Empty list if no ambiguities.
        """
        from core.llm import ask_llm, LLMError

        # Get temporal context for rule-based time ambiguity check
        temporal_range = getattr(self, 'temporal_range', None)
        has_time_ambiguity = self._check_time_ambiguity_rules(temporal_range)

        # Build prompt using extracted prompt builder (P1.3)
        prompt = build_clarification_prompt(
            query=self.query,
            temporal_range=temporal_range,
            has_time_ambiguity=has_time_ambiguity,
        )

        response_structure = {
            "questions": [
                {
                    "clarification_type": "string - time | scope | entity",
                    "question": "string - 對話式問題",
                    "required": "boolean - 必須為 true",
                    "options": [
                        {
                            "label": "string - 選項文字",
                            "intent": "string - 系統內部標籤",
                            "query_modifier": "string - 用於組合查詢的修飾詞（空字串表示全面性選項）",
                            "is_comprehensive": "boolean - 可選，標記為全面性選項"
                        }
                    ]
                }
            ]
        }

        try:
            response = await ask_llm(
                prompt,
                response_structure,
                level="low",
                query_params=self.query_params,
                max_length=1536  # Increased for multiple questions
            )

            # LLMError sentinel（falsy dict）→ response.get 會回 [] 偽裝成「無歧義」，
            # 把 LLM 故障 silent 吞掉。明確分型：故障降級（仍 proceed without
            # clarification）但留 error 級訊息，不偽裝成正常結果。
            if isinstance(response, LLMError):
                logger.error(
                    f"[AMBIGUITY] Detection degraded ({response.error_kind}): "
                    f"LLM call failed, proceeding without clarification questions"
                )
                return []

            questions = response.get('questions', [])

            if questions:
                # Add question_id and option IDs
                for i, q in enumerate(questions, 1):
                    q['question_id'] = f"q{i}"
                    # Add option IDs (1a, 1b, 1c...)
                    for j, opt in enumerate(q.get('options', []), 1):
                        opt['id'] = f"{i}{chr(96+j)}"  # 1a, 1b, 1c...

                logger.info(f"[AMBIGUITY] Detected {len(questions)} ambiguities")
                return questions
            else:
                logger.info("[AMBIGUITY] No ambiguities detected")
                return []

        except Exception as e:
            logger.error(f"[AMBIGUITY] Detection failed: {e}", exc_info=True)
            return []

    async def execute_rerun(self, original_query_id: str, kg_edits_json: str, session_id: str = ""):
        """Execute selective re-run with KG edits.

        Builds a modified query by appending KG edit instructions to the original query,
        then runs phases 2-4 of the composable pipeline using cached context from phase 1.

        Args:
            original_query_id: query_id from the original deep research run
            kg_edits_json: serialized JSON string of KG edits from frontend
            session_id: 前端帶的 PG session UUID（Bug 1：記憶體 cache miss 時用它讀 DB
                research_report 的 rerunState 重建；空字串 = 前端未載入 session，僅走記憶體 cache）

        Raises:
            ValueError: if no cached state for original_query_id
            RuntimeError: if composable_pipeline feature flag is disabled
        """
        from core.config import CONFIG

        # Feature flag gate
        enable_composable = CONFIG.reasoning_params.get("features", {}).get(
            "composable_pipeline", False
        )
        if not enable_composable:
            raise RuntimeError("Selective re-run requires composable_pipeline=true")

        # Parse edit_summary for the query template
        try:
            kg_edits = json.loads(kg_edits_json) if isinstance(kg_edits_json, str) else kg_edits_json
            edit_summary = kg_edits.get('edit_summary', {})
        except (json.JSONDecodeError, AttributeError):
            edit_summary = {}

        # Build modified query using the template from kg-editing-spec.md
        summary_lines = []
        if edit_summary.get('nodes_added', 0):
            summary_lines.append(f"- 新增節點：{edit_summary['nodes_added']} 個")
        if edit_summary.get('nodes_deleted', 0):
            summary_lines.append(f"- 刪除節點：{edit_summary['nodes_deleted']} 個")
        if edit_summary.get('nodes_modified', 0):
            summary_lines.append(f"- 修改節點：{edit_summary['nodes_modified']} 個")
        if edit_summary.get('edges_added', 0):
            summary_lines.append(f"- 新增關係：{edit_summary['edges_added']} 個")
        if edit_summary.get('edges_deleted', 0):
            summary_lines.append(f"- 刪除關係：{edit_summary['edges_deleted']} 個")
        if edit_summary.get('edges_modified', 0):
            summary_lines.append(f"- 修改關係：{edit_summary['edges_modified']} 個")

        summary_text = "\n".join(summary_lines) if summary_lines else "- （變更摘要不可用）"

        modified_query = (
            f"{self.query}\n\n"
            f"【使用者知識圖譜修改】\n"
            f"使用者根據知識圖譜進行了以下修改，請以此為前提重新分析：\n"
            f"{summary_text}\n\n"
            f"【修改後的知識圖譜（JSON）】\n"
            f"{kg_edits_json}\n\n"
            f"請假設使用者的修改為正確前提，據此重新分析。"
            f"如果你的分析結果與使用者修改有衝突，"
            f"仍以使用者的修改為前提分析，但標註 ⚠️ 說明 evidence 與使用者判斷的差異。"
        )

        logger.info(f"[DEEP RESEARCH RERUN] Starting rerun for query_id={original_query_id}")

        from reasoning.orchestrator import (
            DeepResearchOrchestrator,
            get_cached_research_state,
            restore_rerun_state_from_report,
        )
        orchestrator = DeepResearchOrchestrator(handler=self)

        # Bug 1：記憶體 cache miss（server 重啟/TTL/LRU）→ 用 session UUID 讀 DB research_report 重建
        restored_state = None
        if get_cached_research_state(original_query_id) is None and session_id:
            user_id = getattr(self, "user_id", "") or ""
            org_id = getattr(self, "org_id", "") or ""
            if user_id and org_id:
                from core.session_service import SessionService
                svc = SessionService()
                row = await svc.get_session(session_id, user_id, org_id)   # owner 綁定，不跨 user
                if row:
                    _candidate = restore_rerun_state_from_report(row.get("research_report"))
                    # [R3 修訂 S2-new] session↔query_id 對齊驗證：一個 session 可跑多次 DR，
                    # research_report 只存最後一次的 rerunState。若前端 stale（KG 編輯用的 query_id
                    # 與 session 最後一次 DR 不符）→ DB fallback 會拿錯 rerunState → 張冠李戴。
                    # 不匹配（含舊資料缺 query_id → None ≠ original_query_id）→ 當作無有效 rerunState、
                    # 不放行（放行會用錯研究狀態）。memory cache 路徑用 query_id 當 key 無此縫。
                    if _candidate and _candidate.get("query_id") == original_query_id:
                        restored_state = _candidate
                        # cache miss 時 api.py 的 research_mode 已 fallback 成 'discovery'（讀不到
                        # cache）——用重建 state 的真實 mode 對齊，避免 methodology_note 顯示錯 mode。
                        if _candidate.get("mode"):
                            self.research_mode = _candidate["mode"]
                        logger.info(f"[RERUN] cache miss → DB fallback 重建成功 session={session_id}")
                    elif _candidate:
                        logger.warning(
                            f"[RERUN] cache miss + DB rerunState 的 query_id="
                            f"{_candidate.get('query_id')!r} 與請求 original_query_id={original_query_id!r} "
                            f"不符（session 跑過多次 DR / 前端 stale / 舊資料缺 query_id）→ 判無效、回落"
                            f" session={session_id}")
                    else:
                        logger.warning(f"[RERUN] cache miss + DB research_report 無有效 rerunState session={session_id}")

        results = await orchestrator.run_research_rerun(
            original_query_id=original_query_id,
            modified_query=modified_query,
            restored_state=restored_state,
        )

        # Send results via existing SSE mechanism (same path as execute_deep_research)
        from core.schemas import create_assistant_result
        create_assistant_result(results, handler=self, send=True)

        logger.info(f"[DEEP RESEARCH RERUN] Sent {len(results)} results to frontend")

        # Update return_value with structured response
        final_report = self._generate_final_report(results, self._get_temporal_context())
        source_urls = []
        for item in results:
            schema_obj = item.get('schema_object', {})
            if 'sources_used' in schema_obj:
                source_urls.extend(schema_obj['sources_used'])

        self.return_value.update({
            'answer': final_report,
            'confidence_level': self._calculate_confidence(results),
            'methodology_note': f'Deep Research Rerun ({self.research_mode or "discovery"} mode)',
            'sources_used': source_urls,
            'items': results,
            'is_rerun': True,
        })

    def _check_time_ambiguity_rules(self, temporal_range) -> bool:
        """
        Rule-based time ambiguity check (preserving existing logic).
        Returns True if time ambiguity detected.

        Args:
            temporal_range: Temporal parsing result from parent handler

        Returns:
            True if time clarification needed, False otherwise
        """
        # Check 1: Explicit time parsing issues
        if temporal_range is None:
            logger.info("[TIME RULES] Time parsing failed, needs clarification")
            return True
        elif temporal_range.get('confidence', 1.0) < 0.7:
            logger.info(f"[TIME RULES] Low confidence parsing ({temporal_range.get('confidence')}), needs clarification")
            return True

        # Check 2: Semantic temporal ambiguity
        elif not temporal_range.get('is_temporal', False):
            query_lower = self.query.lower()
            temporal_ambiguity_indicators = [
                # Political figures and their policies
                ('蔡英文', ['政策', '兩岸', '外交', '立場', '主張']),
                ('賴清德', ['政策', '兩岸', '外交', '立場', '主張']),
                ('馬英九', ['政策', '兩岸', '外交', '立場', '主張']),
                # Events that span time or evolve
                ('發展', None),
                ('趨勢', None),
                ('演變', None),
                ('變化', None),
            ]

            for entity, context_words in temporal_ambiguity_indicators:
                if entity in query_lower:
                    if context_words:
                        if any(word in query_lower for word in context_words):
                            logger.info(f"[TIME RULES] Semantic ambiguity: '{entity}' with context")
                            return True
                    else:
                        logger.info(f"[TIME RULES] Semantic ambiguity: '{entity}'")
                        return True

        return False

    async def _send_clarification_request(self, clarification_data: dict):
        """
        Send clarification request to frontend via SSE.

        Args:
            clarification_data: Clarification options from ClarificationAgent
        """
        try:
            # Use inherited message_sender from NLWebHandler
            if not hasattr(self, 'message_sender'):
                logger.error("[DEEP RESEARCH] message_sender not available")
                return

            message_data = {
                "message_type": "clarification_required",
                "clarification": clarification_data,
                "query": self.query
            }

            await self.message_sender.send_message(message_data)

            logger.info("[DEEP RESEARCH] Clarification request sent to frontend")

        except Exception as e:
            logger.error(f"[DEEP RESEARCH] Failed to send clarification request: {e}", exc_info=True)
