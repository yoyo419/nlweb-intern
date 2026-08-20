"""
BABLoopEngine — B->A->B' 可複用迴圈引擎。

核心引擎，Stage 1（全域迴圈）和 Stage 2（per-section 聚焦迴圈）都用。

流程：
  Phase 0: build initial B (ContextMap)
  Loop:
    Phase 1: derive A (search plan) from B
    Phase 2: execute A (retrieval)
    Phase 3: mini-reasoning (Analyst + Critic + Consistency Monitor)
    Phase 4: update B -> B'
    Check: is_stable? → break
    Check: consistency pause? → break
    Check: max_iterations? → break
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from core.retriever import search as retriever_search  # Track E (sprint 2026-05-28): module-level for monkeypatch testability
from core.temporal_anchor import annotate_relative_years, anchor_note
from misc.logger.logging_config_helper import get_configured_logger
from reasoning.agents.associator import AssociatorAgent
from reasoning.live_research import lr_copy
from reasoning.live_research.sse_emit import emit_sse
from reasoning.schemas_live import (
    ContextMap,
    ConsistencyReview,
    EvidencePoolEntry,
    context_map_to_summary,
)


def _format_search_result_line(
    evidence_id: int, title: str, desc: str, url: str, published_at: Optional[str]
) -> str:
    """搜尋結果餵給 analyst 的單行渲染（含發布日期錨定）。

    抽成 module-level helper 是為了可被單測直接驗（原本埋在 _execute_search 內，
    沒有機械防線鎖住「日期有送進去」）。

    修的症狀：原本這行只送 [id] 標題 / 內文 / URL，**完全沒有發布日期**，
    analyst 只看得到內文 + 別處注入的「今天是 YYYY-MM-DD」→ 2025 年報導寫的
    「今年」被寫成當前年。改：標頭附發布日期與錨定註記，內文相對年份就地標絕對年份。
    """
    snippet = annotate_relative_years(desc[:500], published_at)
    date_part = f" ({published_at})" if published_at else ""
    return (
        f"[{evidence_id}] {title}{date_part}{anchor_note(published_at)}\n"
        f"{snippet}\nURL: {url}\n"
    )


def _extract_domain(url: str) -> str:
    """從 URL 取 domain（不含 www.）。空 URL 或 parse 失敗回傳空字串。"""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""


# 外部來源（web/wiki）無標題補標題（low-tier LLM 從 snippet 生成）。
# Google CSE API 無 title 時填字串 "No Title"；空標題 / 此 sentinel 觸發補標題。
_NO_TITLE_SENTINEL = "No Title"
# 每個 BAB run 最多用 LLM 補幾筆無標題 evidence（per-run 計數器，超過用 source_domain）。
# module 常數方便調。
TITLE_BACKFILL_CAP = 8
# 補標題 LLM call timeout（秒）— low-tier 小工作，短 timeout。
_TITLE_BACKFILL_TIMEOUT = 10

logger = get_configured_logger("live_research.loop_engine")


class BABLoopEngine:
    """
    B->A->B' 可複用迴圈引擎。

    使用方式：
        engine = BABLoopEngine(associator=..., handler=..., max_iterations=3)
        context_map = await engine.run_loop(query="...", focus_topic_ids=None)
    """

    def __init__(
        self,
        associator: AssociatorAgent,
        handler: Any,
        max_iterations: int = 3,
        enable_consistency_monitor: bool = True,
        dry_run: bool = False,
        seed_evidence_pool: Optional[Dict[int, EvidencePoolEntry]] = None,
        seed_counter: int = 0,
    ):
        """
        Args:
            associator: AssociatorAgent instance
            handler: Request handler (for SSE, connection check, retrieval access)
            max_iterations: B->A->B' loop 最大迭代次數
            enable_consistency_monitor: 是否跑 Consistency Monitor
            dry_run: True = use fixture results, skip LLM calls
            seed_evidence_pool: 從先前 stage（如 Stage 1）累積的 evidence pool，
                注入後本 engine 從此基礎繼續累積（Stage 2 跨 engine 共用 ID）
            seed_counter: 從先前 stage 累積的 _evidence_counter 最大值，
                本 engine 從 seed_counter+1 開始分配新 ID
        """
        self.associator = associator
        self.handler = handler
        self.max_iterations = max_iterations
        self.enable_consistency_monitor = enable_consistency_monitor
        self.dry_run = dry_run

        # State exposed for orchestrator to access
        self.initial_context_map: Optional[ContextMap] = None
        self.executed_searches: List[str] = []
        self.consistency_review: Optional[ConsistencyReview] = None
        self.paused_by_consistency: bool = False
        # plan: lr-disconnect-midstage-persist（D-7，R1 AR blocker 消化）— 是否因
        # client offline cooperative break 提早退出（vs 正常收斂 return）。caller
        # （_run_stage_2）靠這個訊號分辨「這個 topic 真的跑完了」還是「跑到一半被
        # 打斷」，避免把半套研究誤標 completed_sections（見 D-7 完整論證）。
        # per-run 重置於 run_loop 入口（對齊本行下方既有 paused_by_consistency 的
        # per-run-reset pattern，非只在 __init__ 設一次）。
        self.stopped_early: bool = False
        # 降級旁白 per-run dedup flags（FIX-3：__init__ 兜底 + run_loop 入口重置共用一個
        # helper，避免兩處各自展開 + 註解互相交叉引用）。語意見 _reset_per_run_dedup_flags。
        self._reset_per_run_dedup_flags()

        # Evidence pool — 跨 iteration 累積，caller 透過 engine.evidence_pool 取出持久化
        self.evidence_pool: Dict[int, EvidencePoolEntry] = (
            dict(seed_evidence_pool) if seed_evidence_pool else {}
        )
        self._evidence_counter: int = seed_counter
        # URL → evidence_id 反查表（去重用，避免同一 URL 重複收錄）
        self._url_to_id: Dict[str, int] = {
            entry.url: eid
            for eid, entry in self.evidence_pool.items()
            if entry.url
        }
        # 當前 BAB iteration（_execute_search 寫進 evidence.iteration_origin 用，debug）
        self._current_iteration: int = 0

        # 票 2026-07-28-m：tier6 進池相關性 gate 已判過的 eid（跨輪不重判，只判增量）。
        # 🔧 R2（B-R2-2）：初始化 = seed 池的全部 eid 快照——Stage 2 per-topic 建新 engine
        # 時 seed_evidence_pool 帶入前 stage/topic 舊池（orchestrator.py:2308 一帶），這批
        # 在先前 engine 已判過（或屬 internal 本就不判）；不預載會讓每個 topic 把歷史
        # tier6 全池重判一次（N topics = N 次重複燒 LLM），且前面 topic 已引用的 tier6
        # 可能被後面 gate 刪掉（跨 engine dangling citation）。預載全 seed eid（不只
        # web/wiki）行為等價且最簡——internal eid 本就不進 digest，多載無害。
        # 🔧 R3（B-R3-1 裁決）：seed 批**一律**豁免重判——含前 engine gate fail-open
        # 未判成的批。理由：seed 批可能已被前面 topic 引用（fail-open = 沒判成 ≠ 沒被
        # 引用），本 engine 重判刪除 = dangling citation 照樣發生；dangling 防護優先於
        # fail-open 重判完整性。fail-open 重判契約因此收窄為「同 engine 內」（Step 1.4）；
        # 跨 engine 殘漏由下游 C-1 / grounding 三層 / publish gate 兜底（風險表 b2'）。
        # 🔧 R4（SF-R4-1）：豁免契約是**通則**——涵蓋所有 seed_evidence_pool 進入路徑，
        # 不限 Stage 2 per-topic。現行兩條 caller 鏈（R4 親驗）：①Stage 2 per-topic 迴圈
        # 直建 engine（live_research/orchestrator.py:2308；Stage 2 offline resume 走同一
        # 迴圈重入，無獨立 callsite）②recollect 退回 Stage 1（:4630 → _run_stage_1
        # 簽名 :1256 → :1345 轉傳本建構子）。未來新增 seed caller 自動繼承同一豁免。
        self._relevance_gate_judged_ids: set = set(self.evidence_pool.keys())
        # 🔧 R1：gate query（run_loop 入口注入實值；__init__ 先給 None 供直呼 method 的
        # unit test 不 AttributeError——接線 getattr 亦有 None default，此為雙保險）。
        self._relevance_gate_query: Optional[str] = None

        # Track A (sprint 2026-05-28): caller (orchestrator) injects state for
        # evidence_usage indexing — Analyst argument_graph 索引進 state.evidence_usage。
        # 默認 None: caller 未注入 → 不索引（test code / dry-run 沿舊行為）。
        self.state: Optional[Any] = None
        # 當前 BAB topic id (Stage 2 per-topic loop) — caller (orchestrator) 注入
        # 給 GroundedClaim.source_topic 用; default "" → GroundedClaim 用 "global"。
        self._current_topic_id: str = ""

        # Track F (sprint 2026-05-28) I-3: caller (orchestrator) 在 Stage 1 / Stage 2
        # BAB invoke 入口分別 inject `_current_stage`（"stage_1" / "stage_2"）。
        # default "stage_1" 對未注入 case fallback（backward-compat, 但 audit data
        # 對 Stage 2 entry 會誤標 stage_1 → 必須由 caller 補 inject 才正確）。
        # Gemini R4 F-MIN-1: 未來 Stage 3+ 必同步 (a) Literal type 加新值、
        # (b) caller chain sweep + inject。本 sprint 只 cover Stage 1+2。
        self._current_stage: str = "stage_1"

        if dry_run:
            self._setup_dry_run()

    def _reset_per_run_dedup_flags(self) -> None:
        """重置所有降級旁白 per-run dedup flag（每個降級場景一次性旁白去重）。

        兩處呼叫：
          - `__init__` 兜底——直呼降級分支所在 method（如 unit test 直呼
            `_run_mini_reasoning` / `_run_gap_routing_phase` / `_process_gap_resolutions_lr`）
            時，except 內檢查 flag 不可 AttributeError（在 except body 內 raise 會往外
            拋，破壞 non-fatal 語意）。
          - `run_loop` 入口——**per-run 重置才是正確語意**：engine instance 被重用（多
            次 run_loop）時，放 __init__-only 會讓第二次 run 的降級旁白被永久靜音。

        新增降級 flag 時只在此處加一行，兩個呼叫點自動覆蓋。
        """
        self._consistency_degraded_narrated = False
        self._mini_reasoning_degraded_narrated = False   # O5-B
        self._gap_routing_degraded_narrated = False       # O5-C
        self._gap_routing_cap_narrated = False            # C3: gap routing 外部呼叫 cap 旁白
        self._revise_degraded_narrated = False            # M-5: revise 降級旁白
        self._search_required_degraded_narrated = False   # M-5: SEARCH_REQUIRED 補搜降級旁白
        self._kg_merge_degraded_narrated = False          # O5a-(2): KG merge 降級旁白
        self._retrieval_error_degraded_narrated = False   # 檢索出錯（例外跳過該筆查詢）降級旁白
        # 外部來源無標題補標題 per-run cap 計數器（每 run 最多 TITLE_BACKFILL_CAP 次 LLM
        # call，超過用 source_domain）。per-run 重置 = engine 重用時下一輪重新配額。
        self._title_backfill_count = 0
        self._relevance_gate_narrated = False   # 票 -m: tier6 剔除透明化旁白 per-run dedup

    def _setup_dry_run(self):
        """Override methods with fixtures for dry-run."""
        async def mock_search(seeds):
            logger.info("[BAB LOOP] Dry-run: returning fixture search results")
            return (
                "[1] 台灣光電發展現況\n近年光電裝置容量大幅成長...\nURL: https://example.com/1\n",
                {"1": {"name": "台灣光電發展現況", "url": "https://example.com/1"}},
            )

        async def mock_mini(cm, results):
            logger.info("[BAB LOOP] Dry-run: skipping mini-reasoning")
            return True  # s5-4: dry_run 視為成功輪 → run_loop 照 emit bab_phase3 completed

        async def mock_consistency(current, initial):
            return ConsistencyReview(
                drift_level="none",
                drift_description="方向一致（dry-run）",
                dubao_voice_message="",
                recommended_action="continue",
            )

        self._execute_search = mock_search
        self._run_mini_reasoning = mock_mini
        self._run_consistency_check = mock_consistency

    async def run_loop(
        self,
        query: str,
        initial_context: Optional[str] = None,
        user_prior_knowledge: Optional[str] = None,
        focus_topic_ids: Optional[List[str]] = None,
        existing_context_map: Optional[ContextMap] = None,
        existing_initial_map: Optional[ContextMap] = None,
        prior_executed_searches: Optional[List[str]] = None,
    ) -> ContextMap:
        """
        執行 B->A->B' 迴圈。

        plan: lr-disconnect-midstage-persist — 例外退出契約：例外退出（soft-interrupt /
        LLM fail）時 caller 應從 engine.evidence_pool + engine.executed_searches 讀已累積
        成果，不可假設 return 值。

        Args:
            query: 研究問題
            initial_context: 初始 retrieval 結果（Phase 0 用）
            user_prior_knowledge: 使用者提供的先備知識
            focus_topic_ids: 聚焦的 topic IDs（Stage 2 per-section 用）
            existing_context_map: 已有的 ContextMap（Stage 2 跳過 Phase 0）
            existing_initial_map: 已有的 initial ContextMap（Stage 2 用）
            prior_executed_searches: 之前已執行的搜尋

        Returns:
            最新版本的 ContextMap
        """
        self.executed_searches = list(prior_executed_searches or [])
        self.paused_by_consistency = False
        # plan: lr-disconnect-midstage-persist（D-7）— per-run 重置，理由同 __init__ 註解。
        self.stopped_early = False
        # per-run dedupe flags 重置（FIX-3：與 __init__ 共用 helper）。**per-run 重置**
        # 是正確語意——engine instance 被重用（多次 run_loop）時，第二次 run 的降級旁白
        # 不會被永久靜音。見 _reset_per_run_dedup_flags docstring。
        self._reset_per_run_dedup_flags()
        # 🔧 R1：gate query 唯一注入點——接線讀 self._relevance_gate_query。
        # 不能靠 state.research_question（LiveResearchStageState 無此欄位，R1 已驗）。
        # 🔧 R2（B-R2-2）：只注入 query，不 reset _relevance_gate_judged_ids——那會把
        # __init__ 預載的 seed 快照清空（seed 重判問題復發）。judged 集合生命週期 =
        # engine 生命週期（與 evidence_pool 一致，跨 run 保留）。
        self._relevance_gate_query = query

        # Phase 0: 建立或複用 initial B
        if existing_context_map is not None:
            context_map = existing_context_map
            self.initial_context_map = existing_initial_map or existing_context_map.model_copy(deep=True)
        else:
            await self._emit_narration("開始建立研究結構...")
            build_output = await self.associator.build_context_map(
                query=query,
                initial_context=initial_context,
                user_prior_knowledge=user_prior_knowledge,
            )
            context_map = build_output.context_map
            self.initial_context_map = context_map.model_copy(deep=True)
            await self._emit_narration(build_output.narration)
            await self._emit_phase("bab_phase0", "completed")

        # B->A->B' Loop
        # plan: lr-disconnect-midstage-persist — try/finally 兜底：無論正常收斂 /
        # cooperative offline break / soft-interrupt raise / LLM 例外，退出時 caller
        # 都能從 engine.evidence_pool（instance attr，本迴圈持續 mutate）撈已累積 evidence。
        # 這是「斷線進度不蒸發」的最後一道：raise 路徑下 `updated_map = await run_loop()`
        # 賦值不會發生，但 caller 改讀 engine.evidence_pool（Task 3）就拿得到。
        try:
            for iteration in range(self.max_iterations):
                logger.info(f"[BAB LOOP] Iteration {iteration + 1}/{self.max_iterations}")
                # plan: lr-disconnect-midstage-persist — cooperative stop：client 純斷線時
                # _check_connection 回 "offline"（不 raise），break 出迴圈把當前累積的
                # context_map / evidence_pool 帶回 caller（_run_stage_2）落盤，不蒸發。
                # soft-interrupt（使用者主動停）仍會在 _check_connection 內 raise。
                #
                # D-7（R1 AR blocker 消化）：break 前設 self.stopped_early = True——這是
                # caller（_run_stage_2）分辨「這個 topic 正常收斂完成」vs「跑到一半被
                # offline 打斷」的唯一依據。沒有這個旗標，caller 會把半套研究誤標
                # completed_sections（見 D-7 完整論證）。正常收斂（is_stable 或跑滿
                # max_iterations）不會經過這個 break，stopped_early 維持 run_loop 入口
                # 重置的 False。
                if self._check_connection() == "offline":
                    self.stopped_early = True
                    logger.info(
                        f"[BAB LOOP] Client offline detected at iteration {iteration + 1}; "
                        f"cooperative stop (stopped_early=True) — returning accumulated "
                        f"context_map + {len(self.evidence_pool)} evidence entries for "
                        f"caller to persist WITHOUT marking topic as completed"
                    )
                    break
                # 標記當前 iteration 給 _execute_search 寫進 EvidencePoolEntry.iteration_origin
                self._current_iteration = iteration + 1

                # Phase 1: 從 B 推導 A
                await self._emit_phase("bab_phase1", "started")
                derive_output = await self._run_derive_phase(
                    context_map, self.executed_searches, focus_topic_ids
                )
                await self._emit_narration(derive_output.narration)
                await self._emit_phase("bab_phase1", "completed")

                # Phase 2: 執行 A (retrieval)
                await self._emit_phase("bab_phase2", "started")
                formatted_results, new_source_map = await self._execute_search(derive_output.search_seeds)
                for seed in derive_output.search_seeds:
                    self.executed_searches.append(seed.query)
                await self._emit_phase("bab_phase2", "completed")

                # Phase 3: Mini-Reasoning (optional — Analyst + Critic on new evidence)
                # bab_phase3 進度事件 + 預期管理 narration：mini-reasoning 是 BAB loop 最耗時的
                # LLM 段（Analyst high model + Critic + 可能 gap routing）。沿 #8 長 LLM call 前
                # 推進度 pattern（phase1/phase4 已有），補 phase3 進度，避免前端最長窗口零進度。
                # emit 放 run_loop 包夾處而非 _run_mini_reasoning 內部：dry_run 會整個替換
                # _run_mini_reasoning，放內部 dry_run emit 不到；放這裡 dry_run 也會 emit。
                # s5-4 收斂（O5-B/O5-C 耦合判定，見 plan）：
                # - early-skip 輪（gate False，檢索空手）完全不 emit phase3 事件 —
                #   沒有資料可分析，不對 user 謊稱「正在深入分析這批資料」。
                # - mini 失敗輪（O5-B 降級，回傳 False）不 emit completed —
                #   降級旁白「已先略過⋯我會繼續往下進行」就是該輪收尾，緊接的
                #   bab_phase4 started 標記邊界；「完成」與「已先略過」並列矛盾。
                if self._has_mini_reasoning_input(formatted_results):
                    await self._emit_phase("bab_phase3", "started")
                    await self._emit_narration("正在深入分析這批資料、交叉檢驗論點...")
                    mini_ok = await self._run_mini_reasoning(context_map, formatted_results)
                    if mini_ok:
                        await self._emit_phase("bab_phase3", "completed")
                else:
                    # early-skip 輪：保留呼叫維持行為等價（內部 early return + 既有 log）
                    await self._run_mini_reasoning(context_map, formatted_results)

                # 票 2026-07-28-m（🔧 R5 插點）：每輪 Phase 3 後、refine 前，無條件過
                # tier6 相關性 gate——單一插點涵蓋兩類批：gap 批（本輪 mini 內尾端進池）
                # 與 direct-web 批（Phase 2 進池）此刻都已在池、都未被 refine 消費。
                # ⛔ 不可搬進 _run_gap_routing_phase（條件式：無 gap_resolutions 輪不跑）。
                # 🔧 R1：query 讀 self._relevance_gate_query（run_loop 入口注入；
                # LiveResearchStageState 無 research_question 欄位，R1 已驗）。
                _gate_query = getattr(self, "_relevance_gate_query", None)
                if _gate_query:
                    # gate 自身例外不可炸掉 BAB 主迴圈；cancel 先放行（N1）。
                    from reasoning.orchestrator_base import ResearchCancelledError
                    try:
                        await self._relevance_gate_evidence_pool(_gate_query)
                    except ResearchCancelledError:
                        raise  # soft-interrupt 立即停——穿迴圈 finally 冒泡出 run_loop
                    except Exception as e:
                        logger.warning(
                            f"[LR RELEVANCE-GATE] gate 執行失敗（non-fatal，池不動）：{e}",
                            exc_info=True,
                        )
                else:
                    # 不可 silent skip（CLAUDE.md 紀律）：query 缺 = 接線未注入，明確告警。
                    logger.warning(
                        "[LR RELEVANCE-GATE] _relevance_gate_query 未注入，tier6 相關性 gate 跳過本輪"
                    )

                # Phase 4: 更新 B -> B'
                await self._emit_phase("bab_phase4", "started")
                # #8: 長 LLM call 前推進度 narration
                await self._emit_narration("正在根據新資料更新研究結構...")
                refine_output = await self.associator.refine_context_map(
                    current_context_map=context_map,
                    initial_context_map=self.initial_context_map,
                    retrieval_results=formatted_results,
                    focus_topic_ids=focus_topic_ids,
                )
                context_map = refine_output.updated_context_map
                await self._emit_narration(refine_output.narration)
                await self._emit_phase("bab_phase4", "completed")

                # Consistency Monitor (always-on)
                if self.enable_consistency_monitor:
                    # #8: 長 LLM call 前推進度 narration
                    await self._emit_narration("正在檢查研究方向的一致性...")
                    self.consistency_review = await self._run_consistency_check(
                        context_map, self.initial_context_map
                    )

                    # Track F F2 (sprint 2026-05-28): 持久化 consistency drift log
                    # spec §9.2 自標未實現的補完；F-AMB-3 LOCKED: 每輪都 append
                    # （drift_level=none 也 append，audit trail 完整）。
                    # I-3: stage 欄位區分 Stage 1 (global) / Stage 2 (per-topic) invoke
                    # — caller (orchestrator) 透過 self._current_stage attribute inject。
                    if self.state is not None:
                        try:
                            from reasoning.schemas_live import ConsistencyDriftEntry
                            entry = ConsistencyDriftEntry(
                                stage=getattr(self, "_current_stage", "stage_1"),
                                iteration=self._current_iteration,
                                topic_id=self._current_topic_id or "",
                                drift_level=self.consistency_review.drift_level,
                                drift_description=(
                                    self.consistency_review.drift_description or ""
                                ),
                                recommended_action=(
                                    self.consistency_review.recommended_action
                                ),
                                monitor_degraded=getattr(
                                    self.consistency_review, "monitor_degraded", False
                                ),  # O5-A: 降級旗標傳入 audit entry
                            )
                            self.state.consistency_drift_log.append(entry.model_dump())
                        except Exception as e:
                            # secondary defense: 不阻塞 BAB loop
                            logger.warning(
                                f"[BAB LOOP F2] consistency drift log append failed "
                                f"(non-fatal): {type(e).__name__}: {e}"
                            )

                    if self.consistency_review.dubao_voice_message:
                        await self._emit_narration(self.consistency_review.dubao_voice_message)
                    elif getattr(self.consistency_review, "monitor_degraded", False):
                        # O5-A: 降級必有 user-facing 訊息（CLAUDE.md「不可 silent fail」）。
                        # round-3（Gemini critical）：consistency check 每輪（for iteration in
                        # range(max_iterations)）都跑，持久失敗會每輪 emit → 高頻迴圈訊息轟炸。
                        # per-run 只提示一次（flag 在 run_loop 入口初始化，見 Step 3b）。
                        await self._narrate_once("_consistency_degraded_narrated", "（一致性監控暫時無法使用，研究仍會繼續，但這一輪沒有做方向一致性檢查。）")
                    if self.consistency_review.recommended_action == "pause_confirm":
                        self.paused_by_consistency = True
                        logger.info("[BAB LOOP] Paused by Consistency Monitor")
                        break

                # 穩定性判定
                if refine_output.is_stable:
                    logger.info(f"[BAB LOOP] Stable after iteration {iteration + 1}")
                    break
        finally:
            # instance state 一致性：_current_iteration 停在最後跑到的輪次。
            # evidence_pool / executed_searches 已是 instance attr，天然保留。
            logger.info(
                f"[BAB LOOP] run_loop exiting: evidence_pool={len(self.evidence_pool)}, "
                f"executed_searches={len(self.executed_searches)}, "
                f"last_iteration={self._current_iteration}"
            )

        return context_map

    async def _run_derive_phase(self, context_map, executed_searches, focus_topic_ids=None):
        # #8: 長 LLM call 前推進度 narration（避免靜默窗口 → SSE idle 斷 + 黑屏）。
        # 抽 helper 的唯一目的:讓「emit 在 derive 之前」由 production code 保證,
        # 單測可直接驗(GPT #2:不可讓 test 自己 replay 順序)。
        await self._emit_narration("正在推導下一輪要查的資料方向...")
        return await self.associator.derive_search_plan(
            context_map=context_map,
            executed_searches=executed_searches,
            focus_topic_ids=focus_topic_ids,
        )

    @staticmethod
    def _normalize_item(item) -> dict:
        """將 list/tuple 格式的搜尋結果轉為 dict。

        postgres_client.search() 回傳 [url, schema_str, title, source, ?vector]
        google_search_client 回傳 (url, schema_json, title, site, [])
        """
        if isinstance(item, dict):
            return item
        # list/tuple format
        schema_str = item[1] if len(item) > 1 else "{}"
        try:
            schema_obj = json.loads(schema_str) if isinstance(schema_str, str) else (schema_str or {})
        except (json.JSONDecodeError, TypeError):
            schema_obj = {}
        # FIX-3 (Cayenne #10, sprint 2026-05-28): 抽 author 進 normalized item，
        # 讓 EvidencePoolEntry 能填 author → APA inline citation 不再「(來源不明, n.d.)」。
        # schema.org author 可為 dict({"@type":"Person","name":...}) / list / str
        # （沿 qdrant._parse_schema 既有 pattern）。
        author_data = schema_obj.get('author', '')
        if isinstance(author_data, dict):
            author = author_data.get('name', '') or ''
        elif isinstance(author_data, list) and author_data:
            author = (
                author_data[0].get('name', '')
                if isinstance(author_data[0], dict)
                else str(author_data[0])
            )
        elif isinstance(author_data, str):
            author = author_data
        else:
            author = ''
        return {
            'url': item[0] if len(item) > 0 else '',
            'name': item[2] if len(item) > 2 else '',
            'title': item[2] if len(item) > 2 else '',
            'source': item[3] if len(item) > 3 else '',
            'description': schema_obj.get('description') or schema_obj.get('articleBody', ''),
            'snippet': schema_obj.get('description') or schema_obj.get('articleBody', ''),
            'datePublished': schema_obj.get('datePublished', ''),  # Track E (sprint 2026-05-28)
            'author': author,  # FIX-3 (Cayenne #10): APA citation metadata
        }

    async def _execute_search(self, search_seeds) -> Tuple[str, Dict]:
        """
        執行搜尋計畫。

        使用現有 retriever_search（internal）和 GoogleSearchClient（web）。

        Returns:
            (formatted_results_str, source_map_dict)
        """
        # Track E (sprint 2026-05-28): 從 state.time_constraint 構造 datePublished
        # filter 與 query suffix（沿 DR baseHandler.py:518-532 模式）。
        # N-1: None bound 不過濾；N-4: raw_phrase 優先 fallback 才用日期。
        # N-8: kwarg name 固定用 `filters=`，禁 `search_filters=` 別名。
        search_filters: List[Dict] = []
        query_suffix: str = ""
        if self.state is not None and getattr(self.state, "time_constraint", None) is not None:
            tc = self.state.time_constraint
            if tc.start_date:
                search_filters.append(
                    {"field": "datePublished", "operator": "gte", "value": tc.start_date}
                )
            if tc.end_date:
                search_filters.append(
                    {"field": "datePublished", "operator": "lte", "value": tc.end_date}
                )
            # 給 query rewriter / search engine 提示（N-4: raw_phrase 優先）
            if tc.raw_phrase:
                query_suffix = f"（{tc.raw_phrase}）"
            elif tc.start_date and tc.end_date:
                query_suffix = f"（{tc.start_date[:4]} 至 {tc.end_date[:4]}）"
            elif tc.start_date:
                query_suffix = f"（{tc.start_date[:4]} 之後）"
            elif tc.end_date:
                query_suffix = f"（{tc.end_date[:4]} 之前）"
            if search_filters:
                logger.info(
                    f"[BAB LOOP][Track E] datePublished filter active: {search_filters}; "
                    f"query suffix: {query_suffix!r}"
                )

        all_items = []
        # 🔧 票 2026-07-28-m Task 0.5：direct/fallback web items 的 URL 集合——入庫時據此
        # 標 source="web"（修既存錯標 internal bug：web 與 internal 混 extend 進 all_items
        # 後無法靠內容區分；_normalize_item 輸出的 'source' key 是 google 第 4 欄 site
        # 網域名，與 EvidencePoolEntry.source 的來源分類同名不同義，不可拿來判定）。
        web_urls: set = set()
        # Track C C3 (F-9 根解 2026-05-28): keyword white-list for intl/transnational queries.
        # 沿 C-AMB-3 / C-AMB-4 紀律段同一份 white-list（class-level constant），
        # 確保 Associator prompt 紀律與 fallback gate 同步。
        # C-MIN-1 (Gemini R4 2026-05-29): 英文 keyword 必須 case-insensitive 比對
        # (user 可能輸入小寫 'oecd' / 'paris agreement')。
        INTL_KEYWORDS = (
            "德國", "丹麥", "日本", "美國", "歐洲", "荷蘭", "英國", "法國",
            "義大利", "瑞典", "挪威", "印度", "中國", "韓國", "新加坡", "東南亞",
            "IEA", "IRENA", "CBAM", "Paris Agreement", "IPCC", "OECD",
            "聯合國", "WTO",
        )
        FALLBACK_THRESHOLD = 3

        for seed in search_seeds:
            seed_items_count = 0  # F-9 per-seed local count (不 cross-seed 累計)
            try:
                if seed.source_strategy in ("internal", "both"):
                    # NOTE: deliberately NO handler= here. Per-loop retrieval (num_results=5,
                    # web-augmented via Track C) must NOT trigger the low-relevance/low-keyword
                    # warning — those fire only on the initial prepare() in-corpus retrieval
                    # (research-start baseline). Passing handler= would re-emit warnings every
                    # loop (5 results always < KEYWORD_HIT_MIN) and mislead when web augments.
                    items = await retriever_search(
                        query=seed.query + query_suffix,  # Track E: query suffix
                        site=getattr(self.handler, 'site', 'all'),
                        num_results=5,
                        query_params=getattr(self.handler, 'query_params', None),
                        filters=search_filters or None,  # Track E (N-8): kwarg name `filters=`
                    )
                    seed_items_count += len(items or [])
                    all_items.extend(items or [])

                # Track C C2: enable_web_search toggle gate. Default False → web path inactive
                # (backward-compat for LR sessions that didn't ask for web search).
                # Source: docs/decisions.md:87-90 active decision「Web search 只在 Reasoning
                # gap resolution 中使用」— LR direct web path 仍受 toggle 控制。
                if seed.source_strategy in ("web", "both") and getattr(
                    self.handler, "enable_web_search", False
                ):
                    web_items = await self._execute_web_search(seed.query)
                    seed_items_count += len(web_items)
                    web_urls.update(  # 🔧 R1：記 web url，入庫時標 source="web"
                        (it[0] if isinstance(it, (list, tuple)) else it.get("url", ""))
                        for it in web_items
                    )
                    all_items.extend(web_items)

            except Exception as e:
                logger.warning(f"[BAB LOOP] Search failed for '{seed.query}': {e}")
                # 2026-06-20 prod：embedding 雙 provider 同失敗 → retriever_search 拋例外
                # → 該 seed 完全沒撈到資料卻只有 log（silent skip），user 以為有補到資料。
                # 降級必有 user-facing 訊息（CLAUDE.md 不可 silent fail）。
                # 持久性故障會每 seed × 每 iteration 重複觸發 → per-run 只播一次
                # （flag 登記於 _reset_per_run_dedup_flags）。log 每次照記，只旁白 dedup。
                # 與 _search_required_degraded_narrated（補搜「無結果」）語義不同：
                # 這裡是「檢索出錯（例外）」，不可混用同一 flag / 文案。
                await self._narrate_once(
                    "_retrieval_error_degraded_narrated",
                    lr_copy.RETRIEVAL_ERROR_DEGRADED_NARRATION,
                )

            # Track C C3 (F-9 根解 2026-05-28): 站內空集合 + 國際 keyword 雙閘 fallback web gate.
            # 觸發條件 (AND)：
            #   1. seed 原 source_strategy=="internal" (沒跑過 web)
            #   2. *本 seed* 回 < 3 條 (per-seed count filter — 不 cross-seed 累計)
            #   3. seed.query 含「非台灣 / 國際 / 跨國」keyword (keyword filter)
            #   4. self.handler.enable_web_search==True (toggle)
            # 單純 count<3 不觸發 — 避免「純台灣議題站內偶然空也打 Google」(Reward Hack 補丁)。
            # 獨立於上方 try/except — fallback 失敗有自己的 warning，不被外層 swallow。
            seed_was_internal_only = seed.source_strategy == "internal"
            seed_returned_low = seed_items_count < FALLBACK_THRESHOLD
            # C-MIN-1 case-insensitive 比對
            _query_lower = seed.query.lower()
            query_has_intl_keyword = any(kw.lower() in _query_lower for kw in INTL_KEYWORDS)
            if (
                seed_was_internal_only
                and seed_returned_low
                and query_has_intl_keyword
                and getattr(self.handler, "enable_web_search", False)
            ):
                logger.info(
                    f"[BAB LOOP][Track C C3 fallback] seed internal returned {seed_items_count} items "
                    f"(<{FALLBACK_THRESHOLD}) for seed '{seed.query}' (intl keyword matched), "
                    f"triggering fallback web search"
                )
                try:
                    fallback_web_items = await self._execute_web_search(seed.query)
                    web_urls.update(  # 🔧 R1：fallback web 同標
                        (it[0] if isinstance(it, (list, tuple)) else it.get("url", ""))
                        for it in fallback_web_items
                    )
                    all_items.extend(fallback_web_items)
                except Exception as e:
                    logger.warning(f"[BAB LOOP][Track C C3 fallback] web search failed: {e}")
            elif seed_was_internal_only and seed_returned_low and not query_has_intl_keyword:
                logger.debug(
                    f"[BAB LOOP][Track C C3 fallback] seed internal low ({seed_items_count} items) "
                    f"for seed '{seed.query}' but no intl keyword matched — skipping fallback"
                )

        # Normalize: retrieval returns list/tuple, not dict
        all_items = [self._normalize_item(item) for item in all_items]

        # Format results using orchestrator's shared formatter
        if not all_items:
            return ("（未找到相關結果）", {})

        # 全局累積編號：跨 iteration 唯一遞增；同 URL 去重沿用既有 evidence_id
        formatted_lines = []
        iteration_source_map: Dict[int, dict] = {}

        # Track E (sprint 2026-05-28): time_constraint 過濾 helper。
        # N-1: None bound 不過濾；N-2: published_at 缺 (None/empty) 走 fallback 不過濾；
        # N-3: 只取前 10 字元（YYYY-MM-DD）做比對，不做 timezone normalize。
        tc = self.state.time_constraint if (self.state is not None and getattr(self.state, "time_constraint", None) is not None) else None

        def _is_in_time_range(date_only: Optional[str]) -> bool:
            """date_only=None/empty → True（不過濾）。"""
            if not date_only:
                return True
            if tc.start_date and date_only < tc.start_date:
                return False
            if tc.end_date and date_only > tc.end_date:
                return False
            return True

        filtered_count = 0

        for item in all_items:
            url = item.get('url', '') or ''
            title = item.get('name', item.get('title', '')) or ''
            desc = item.get('description', item.get('snippet', '')) or ''
            raw_published = item.get('datePublished', '') or ''
            # Track E (N-3): 只取前 10 字元（YYYY-MM-DD）；空字串 → None
            published_at: Optional[str] = raw_published[:10] if raw_published else None

            # Track E: 範圍外 evidence 跳過入庫（N-2: published_at 缺則 fallback 不過濾）
            if tc is not None and published_at and not _is_in_time_range(published_at):
                filtered_count += 1
                logger.debug(
                    f"[BAB LOOP][Track E] Filtered out-of-range item: "
                    f"published={published_at!r} not in [{tc.start_date}, {tc.end_date}] — {url}"
                )
                continue

            if url and url in self._url_to_id:
                # 同 URL 已收錄 → 沿用既有 evidence_id（去重）
                evidence_id = self._url_to_id[url]
            else:
                self._evidence_counter += 1
                evidence_id = self._evidence_counter
                if url:
                    self._url_to_id[url] = evidence_id
                self.evidence_pool[evidence_id] = EvidencePoolEntry(
                    evidence_id=evidence_id,
                    title=title,
                    url=url,
                    source_domain=_extract_domain(url),
                    snippet=desc[:300],
                    iteration_origin=self._current_iteration,
                    published_at=published_at,  # Track E (sprint 2026-05-28) Option A
                    # FIX-3 (Cayenne #10): 填 author → APA inline 不再「來源不明」。
                    # year 由 render 從 published_at 取年份（避免重複 source-of-truth）。
                    author=item.get('author', '') or '',
                    source="web" if url and url in web_urls else "internal",  # 🔧 R1
                )

            formatted_lines.append(
                _format_search_result_line(evidence_id, title, desc, url, published_at)
            )
            iteration_source_map[evidence_id] = item

        if filtered_count > 0 and tc is not None:
            logger.info(
                f"[BAB LOOP][Track E] Filtered {filtered_count} out-of-range items "
                f"(constraint: start={tc.start_date}, end={tc.end_date})"
            )

        return ("\n".join(formatted_lines), iteration_source_map)

    async def _execute_web_search(self, query: str) -> list:
        """執行 web search（Google Custom Search）。
        num_results 從 LR 專屬 config key tier_6.web_search.max_results_lr 讀取，
        default 直接 8，**不 fallback 到共用 max_results**（與 DR 完全解耦）。
        CEO 決策③：max_results_lr 設為 8（原 hard-code 3 → 補上游資料不足）。
        DR 走自己的 reasoning/orchestrator.py 讀 max_results=5，與此處兩鍵互不 fallback。
        reviewer R-G1：移除 fallback 耦合，避免日後調 DR 的 max_results 連帶影響 LR。
        """
        try:
            from retrieval_providers.google_search_client import GoogleSearchClient
            from core.config import CONFIG
            tier_6_config = CONFIG.reasoning_params.get("tier_6", {})
            web_config = tier_6_config.get("web_search", {})
            # LR 專屬 key，default 8，不 fallback 到共用 max_results（與 DR 解耦）
            num_results = web_config.get("max_results_lr", 8)
            client = GoogleSearchClient()
            # 票 2026-07-28-k：透傳 query_id → client 端既有 `if query_id:` gated
            # tier_6_enrichment 落表（google_search_client.py:219-238）生效。
            # detach 後背景 task 續跑時 handler（closure 持有）仍在，query_id 仍有效。
            results = await client.search_all_sites(
                query,
                num_results=num_results,
                query_id=getattr(self.handler, "query_id", None),
            )
            logger.debug(f"[BAB LOOP] Web search '{query}': num_results={num_results}, got={len(results or [])}")
            return results or []
        except Exception as e:
            logger.warning(f"[BAB LOOP] Web search failed: {e}")
            return []

    # Evidence Sufficiency Narration（reviewer R-G2：比例制，不寫死 magic number）
    # 閾值用「實際取得 evidence 數 / 理論最大數」的百分比判斷，而非固定筆數。
    # 理由：search 數已從 3→8（max_results_lr），固定筆數閾值在未來再調 search 量時會失真。
    #   理論最大 = num_results (max_results_lr=8) × query 數 (BAB 每 run 3 query) = 24
    # 比例常數（依現況推導，標明依據；可調）：
    #   - THIN：< 25%（≈ 24 筆中不到 6 筆 → 偏少。原固定 thin=5 筆 ≈ 20.8%，取整為 25% 邊界）
    #   - CRITICAL：< 10%（≈ 24 筆中不到 2.4 筆 → 嚴重不足。原固定 critical=2 筆 ≈ 8.3%，取整為 10%）
    EVIDENCE_THIN_RATIO = 0.25       # < 25% 理論最大 → 偏少
    EVIDENCE_CRITICAL_RATIO = 0.10   # < 10% 理論最大 → 嚴重不足
    BAB_QUERIES_PER_RUN = 3          # BAB 每 run 派生的 search query 數（理論最大數的乘數）

    async def emit_evidence_sufficiency_narration(self) -> None:
        """BAB 完成後評估 evidence pool 充分度，emit SSE narration。

        判斷標準（reviewer R-G2：比例制）：
        令 theoretical_max = max_results_lr × BAB_QUERIES_PER_RUN，
        ratio = pool_size / theoretical_max。
        - ratio < EVIDENCE_CRITICAL_RATIO：嚴重不足，明告 user
        - ratio < EVIDENCE_THIN_RATIO：偏少，溫和提示
        - ratio >= EVIDENCE_THIN_RATIO：不 emit（正常）

        用比例而非固定筆數：search 量（max_results_lr）未來再調時閾值自動跟著縮放，
        不會因為 hard-coded 筆數而失真。

        目的：透明度 — user 知道 LR 拿到多少資料（純前端 SSE 顯示）。
        注意：narration 只走 SSE 給 user 看，**不進 writer 的 prompt context**
        （writer context 由 evidence_pool 經 outline_planner / _write_section 注入，
        與本 narration 是兩條獨立通道；見決策 C）。
        """
        from core.config import CONFIG
        pool_size = len(self.evidence_pool)
        # 理論最大數 = LR 每次 search 抓取上限 × BAB query 數
        web_config = CONFIG.reasoning_params.get("tier_6", {}).get("web_search", {})
        num_results = web_config.get("max_results_lr", 8)
        theoretical_max = max(1, num_results * self.BAB_QUERIES_PER_RUN)  # 防除零
        ratio = pool_size / theoretical_max
        logger.info(
            f"[BAB LOOP] Evidence pool after BAB: {pool_size}/{theoretical_max} "
            f"(ratio={ratio:.2f})"
        )

        if ratio < self.EVIDENCE_CRITICAL_RATIO:
            msg = (
                f"（注意：本次研究蒐集到的資料來源極為有限（{pool_size} 筆），"
                "結論可靠性較低，建議擴大搜尋範圍或開啟更多資料來源。）"
            )
            await self._emit_narration(msg)
            logger.warning(
                f"[BAB LOOP] Evidence pool critically thin: {pool_size} entries "
                f"(ratio={ratio:.2f} < {self.EVIDENCE_CRITICAL_RATIO})"
            )

        elif ratio < self.EVIDENCE_THIN_RATIO:
            msg = (
                f"（本次研究蒐集到 {pool_size} 筆資料，資料量偏少，"
                "報告內容可能較為有限，部分議題可能無法深入分析。）"
            )
            await self._emit_narration(msg)
            logger.info(
                f"[BAB LOOP] Evidence pool thin: {pool_size} entries "
                f"(ratio={ratio:.2f} < {self.EVIDENCE_THIN_RATIO})"
            )
        else:
            logger.info(
                f"[BAB LOOP] Evidence pool sufficient: {pool_size} entries "
                f"(ratio={ratio:.2f})"
            )

    # ========================================================================
    # Track C C4 (sprint 2026-05-28): gap_resolutions routing port from DR.
    # 移植自 reasoning/orchestrator.py:1864-1989 _process_gap_resolutions，
    # 只 handle 4 類 (LLM_KNOWLEDGE / WIKIPEDIA / WEB_SEARCH / INTERNAL_SEARCH)。
    # Stock/weather/company 6 類 LR 明示砍 — log skip 不 raise (fail-loud-with-info).
    # ========================================================================

    async def _process_gap_resolutions_lr(self, gap_resolutions: list) -> None:
        """Process Analyst gap_resolutions in LR BAB context (Track C C4).

        結果建 EvidencePoolEntry(source=...) 入 self.evidence_pool。
        不 re-run mini-reasoning Analyst (C-AMB-2 拍板：一次性消費)。

        Toggle gates:
        - enable_gap_enrichment=False → 整個 method early return
        - enable_web_search=False → WEB_SEARCH gap log skip (其他三類仍跑)

        F-6 紀律: 只寫 self.evidence_pool (engine dict)，不寫 state — state 上
        沒 evidence_pool dict attr (只有 evidence_pool_json: str 序列化欄位)，
        merge 由 orchestrator 統一處理。
        """
        if not getattr(self.handler, "enable_gap_enrichment", False):
            logger.info("[LR gap routing] enable_gap_enrichment=False, skipping all gaps")
            return

        from reasoning.schemas_enhanced import GapResolutionType

        SUPPORTED_IN_LR = {
            GapResolutionType.LLM_KNOWLEDGE,
            GapResolutionType.WIKIPEDIA,
            GapResolutionType.WEB_SEARCH,
            GapResolutionType.INTERNAL_SEARCH,
        }

        # C3 (2026-06-11): per-run 外部呼叫 cap（default-on 安全配套）。CEO「gap_enrichment
        # 全開不拆」不動 — cap 是「單 run 外部呼叫次數上限」非分類開關，4 類仍全開。
        # 防 Analyst 異常輸出大量 gap 時 Wikipedia / Google CSE 呼叫失控燒 API 額度 + 拉延遲。
        # 呼叫時讀（非 module-import 常數）→ unit test monkeypatch CONFIG 可蓋到。
        from core.config import CONFIG
        gap_external_cap = (
            CONFIG.reasoning_params.get("tier_6", {})
            .get("gap_routing", {})
            .get("max_external_calls_per_run", 6)
        )
        external_calls_made = 0

        for gap in gap_resolutions:
            res = gap.resolution
            if res not in SUPPORTED_IN_LR:
                # 明示砍：stock / weather / company → fail-loud-with-info
                res_value = res.value if hasattr(res, "value") else str(res)
                logger.info(
                    f"[LR gap routing] Skipping {res_value} "
                    f"— not supported in LR (financial/weather/company APIs are DR-only)"
                )
                continue

            if res == GapResolutionType.LLM_KNOWLEDGE:
                self._add_llm_knowledge_evidence(gap)

            elif res == GapResolutionType.WIKIPEDIA:
                # C3: cap 計數加在真正打外部之前（WIKIPEDIA 無其他 gate，直接 guard）。
                if external_calls_made >= gap_external_cap:
                    logger.info(
                        f"[LR gap routing] external call cap ({gap_external_cap}) reached, "
                        f"skipping WIKIPEDIA gap '{gap.search_query}'"
                    )
                    await self._narrate_gap_cap_once()
                    continue
                external_calls_made += 1
                await self._execute_wikipedia_searches_lr([gap])

            elif res == GapResolutionType.WEB_SEARCH:
                if not getattr(self.handler, "enable_web_search", False):
                    logger.info(
                        f"[LR gap routing] WEB_SEARCH gap '{gap.search_query}' "
                        f"skipped (enable_web_search=False)"
                    )
                    continue
                if not gap.search_query:
                    logger.info("[LR gap routing] WEB_SEARCH gap has empty search_query, skipping")
                    continue
                # C3: cap 計數加在 enable_web_search gate + 空 query 檢查通過後、真打外部之前。
                # 被 gate / 空 query / cap 跳過的 gap 不消耗額度（skip 不燒 budget）。
                if external_calls_made >= gap_external_cap:
                    logger.info(
                        f"[LR gap routing] external call cap ({gap_external_cap}) reached, "
                        f"skipping WEB_SEARCH gap '{gap.search_query}'"
                    )
                    await self._narrate_gap_cap_once()
                    continue
                external_calls_made += 1
                web_items = await self._execute_web_search(gap.search_query)
                # F-11 紀律: _execute_web_search 回傳 list of tuples (來自
                # GoogleSearchClient.search_all_sites)，必須先過 _normalize_item
                # 轉成 dict 才能 .get() 取 url/title/snippet。
                for item in web_items:
                    await self._add_external_evidence(self._normalize_item(item), source="web")

            elif res == GapResolutionType.INTERNAL_SEARCH:
                # internal search 已由 BAB main loop _execute_search 處理 — gap routing no-op
                logger.info(
                    f"[LR gap routing] INTERNAL_SEARCH gap '{gap.search_query}' "
                    f"pass-through (handled by BAB main loop)"
                )

    async def _narrate_gap_cap_once(self) -> None:
        """C3: gap routing 外部呼叫達 per-run cap 時 emit 一次 user-facing 旁白。

        CLAUDE.md「不可 silent fail」紀律：cap 跳過 gap 不可無聲略過，使用者
        得知「補強查證已達本輪上限」。per-run dedup（_gap_routing_cap_narrated）
        防同一輪多個 gap 撞 cap 時重複轟炸旁白。
        """
        await self._narrate_once("_gap_routing_cap_narrated", "這一輪的外部補強查證已達上限 — 為控制查證次數與時間，"
            "剩餘的外部資料補強（維基／網路）先略過，研究照常繼續。")

    def _add_llm_knowledge_evidence(self, gap) -> None:
        """LLM_KNOWLEDGE gap → 建 virtual doc 進 evidence_pool (Track C C4).

        F-6 紀律: 只寫 self.evidence_pool (engine dict)，沿既有 _execute_search
        line 378 pattern。state.evidence_pool 在 LiveResearchStageState 上不存在
        (state 有 evidence_pool_json: str 序列化欄位，由 orchestrator BAB run
        loop 結束時統一 serialize)。
        """
        from reasoning.schemas_live import EvidencePoolEntry

        topic = gap.topic or gap.gap_type.replace(" ", "_")
        urn = f"urn:llm:knowledge:{topic}"
        # 避免重複建 — 若 URN 已存在 self._url_to_id，沿用既有 eid
        if urn in self._url_to_id:
            return
        self._evidence_counter += 1
        eid = self._evidence_counter
        self._url_to_id[urn] = eid
        entry = EvidencePoolEntry(
            evidence_id=eid,
            title=f"AI 背景知識：{gap.gap_type}",
            url=urn,
            source_domain="llm_knowledge",
            snippet=f"[Tier 6 | llm_knowledge] {gap.llm_answer or ''}"[:300],
            iteration_origin=self._current_iteration,
            source="llm_knowledge",
            # F-12 紀律 (Track C × Track E orthogonal): llm_knowledge 無 publish 概念
            published_at=None,
        )
        self.evidence_pool[eid] = entry
        logger.info(f"[LR gap routing] Added llm_knowledge evidence eid={eid} urn={urn}")

    async def _execute_wikipedia_searches_lr(self, gaps: list) -> None:
        """Wikipedia API call + 結果建 EvidencePoolEntry(source='wiki') (Track C C4).

        仿 DR `orchestrator.py:2319-2363` _execute_wikipedia_searches，但 result
        結構直接寫進 evidence_pool（不走 DR current_context + source_map pattern）。

        F-2 dual-guard 紀律: 同時測 module-level WIKIPEDIA_AVAILABLE (ImportError
        defensive) + client.is_available() (CONFIG yaml tier_6.wikipedia.enabled 開關)，
        避免 import 副作用觸發。
        """
        try:
            from retrieval_providers.wikipedia_client import WikipediaClient, WIKIPEDIA_AVAILABLE
        except ImportError:
            logger.info("[LR gap routing] wikipedia_client import failed, skipping wiki gaps")
            return

        if not WIKIPEDIA_AVAILABLE:
            logger.info("[LR gap routing] wikipedia library not installed, skipping wiki gaps")
            return

        client = WikipediaClient()
        if not client.is_available():
            logger.info(
                "[LR gap routing] WikipediaClient disabled "
                "(tier_6.wikipedia.enabled=false in config), skipping wiki gaps"
            )
            return

        for gap in gaps:
            query = gap.search_query or (gap.api_params.get("query") if gap.api_params else None)
            if not query:
                logger.info("[LR gap routing] WIKIPEDIA gap has no query, skipping")
                continue
            try:
                results = await client.search(
                    query, query_id=getattr(self.handler, "query_id", None)
                )
            except Exception as e:
                logger.warning(f"[LR gap routing] Wikipedia search failed for '{query}': {e}")
                continue
            for result in (results or []):
                if not isinstance(result, dict):
                    continue
                url = result.get("link", "")
                if not url:
                    continue
                await self._add_external_evidence({
                    "url": url,
                    "title": result.get("title", "Wikipedia"),
                    "snippet": f"[Tier 6 | encyclopedia] {result.get('snippet', '')}",
                }, source="wiki")

    async def _relevance_gate_evidence_pool(self, query: str) -> None:
        """票 2026-07-28-m：tier6（web/wiki）進池相關性 gate。

        挪用 DR gate（reasoning/orchestrator.py._relevance_gate_source_pool，票 -f）的
        判定核心（reasoning.relevance_gate_core），適配 LR evidence_pool 形狀：
        - 只判 tier6（source in {web, wiki}）且本輪新進、未判過的 eid（跨輪不重判；
          🔧 R2 seed_evidence_pool 帶入的舊池 eid 已在 __init__ 預標已判——Stage 2
          per-topic 新 engine 不重判、不刪前 stage/topic 舊池）。
        - 部分不相關 → 從 evidence_pool 刪 eid + 同步 _url_to_id（LR 池 = 唯一真相）。
        - 全不相關且刪光後池真空 → 下游 C-1 章級 gate（reasoning/live_research/orchestrator.py:5833
          `not evidence_pool` 🔧 R1）自然走誠實查無（LR 不需 early_return 機制）。
        - fail-open（🔧 R1 / 🔧 R3 收窄）：core 判定失敗回 None → 全保留 **且不標 judged**
          （**同 engine 內**下輪重判，避免 transient failure 讓該批在本 engine 永久豁免；
          誤殺真證據代價高於留噪音）。**跨 engine seed 邊界一律豁免不重判**——fail-open
          批帶進下個 engine 後不再重判（可能已被前 topic 引用，重判刪除 = dangling；
          裁決全文見 Step 1.3 __init__ 註解）。
        - 剔除必 emit 旁白（CLAUDE.md 不可 silent fail；per-run dedup 防轟炸）。

        成本：每輪 tier6 有新進才 +1 次 low-tier 批次 call；無新 tier6 → skip 零成本。
        LR 是 CSE 大戶，故只判增量 + 批次判全批，不逐 source call。
        """
        from reasoning.relevance_gate_core import (
            judge_irrelevant_source_ids,
            RELEVANCE_GATE_DIGEST_CHAR_BUDGET,
        )

        # 1. 挑本輪待判的 tier6 eid（未判過的 web/wiki）
        digest_lines = []
        judged_ids = []
        used_chars = 0
        for eid in sorted(self.evidence_pool):
            entry = self.evidence_pool[eid]
            if entry.source not in ("web", "wiki"):
                continue
            if eid in self._relevance_gate_judged_ids:
                continue
            line = f"[{eid}] {entry.source_domain} - {entry.title}：{(entry.snippet or '')[:150]}"
            if used_chars + len(line) > RELEVANCE_GATE_DIGEST_CHAR_BUDGET:
                logger.warning(
                    f"[LR RELEVANCE-GATE] digest 達字數上限，[{eid}] 起未送判（fail-open 保留）"
                )
                break
            digest_lines.append(line)
            judged_ids.append(eid)
            used_chars += len(line) + 1

        if not judged_ids:
            return  # 無新 tier6 → skip（零成本）

        # 2. 委派共用核心（DR/LR 同一份 prompt + clamp + fail-open）
        irrelevant = await judge_irrelevant_source_ids(
            query=query,
            digest="\n".join(digest_lines),
            judged_ids=judged_ids,
            query_params=getattr(self.handler, "query_params", {}),
            logger=logger,
            log_prefix="LR RELEVANCE-GATE",
        )
        # 🔧 R1（SF1）：fail-open（core 回 None）→ 全保留且**不標 judged**，同 engine 內
        # 下輪自動重判（🔧 R3：重判範圍限本 engine；跨 engine seed 邊界一律豁免，
        # 裁決見 __init__ seed 預載註解）。core 已印 warning。
        if irrelevant is None:
            return

        # 拿到正常判定（非 None）後才標記已判——本批確實被判過了，下輪不重判。
        self._relevance_gate_judged_ids.update(judged_ids)

        if not irrelevant:  # set() = 全相關（與 fail-open None 語義分明）
            logger.info(
                f"[LR RELEVANCE-GATE] {len(judged_ids)} 筆 tier6 判定相關，池不動"
            )
            return

        # 3. 濾池：從 evidence_pool 刪 eid + 同步 _url_to_id
        for eid in irrelevant:
            entry = self.evidence_pool.pop(eid, None)
            if entry is not None and entry.url in self._url_to_id:
                del self._url_to_id[entry.url]
        logger.warning(
            f"[LR RELEVANCE-GATE] 剔除 {len(irrelevant)} 筆不相關 tier6 來源："
            f"{sorted(irrelevant)}"
        )

        # 4. 透明化旁白（不可 silent fail；per-run 只提示一次）
        await self._narrate_once(
            "_relevance_gate_narrated",
            "（部分外部補強來源與查詢主體關聯薄弱，已從證據中排除，研究照常繼續。）",
        )

    async def _add_external_evidence(self, item: dict, source: str) -> None:
        """共用 helper：把外部來源 item (web / wiki) 寫進 evidence_pool (Track C C4).

        item 預期 keys: url, title, snippet (已 normalized)。
        URL 去重沿用既有 self._url_to_id 機制。

        無標題補標題 (2026-06-17): Google CSE 無 title 時填 "No Title"，空標題 /
        sentinel 進池前用 low-tier LLM 從 snippet 生成簡潔中文標題（見
        _resolve_external_title）。async 化以 await LLM call；兩 caller
        (_process_gap_resolutions_lr web 路徑 / _execute_wikipedia_searches_lr)
        皆為 async，已加 await。

        F-6 紀律: 只寫 self.evidence_pool (engine dict)。
        F-12 紀律 (Track C × Track E orthogonal): wiki/web 不填 publish date
        (Wikipedia 只有 revision date 不是 content publish date；Google CSE
        無可靠 publish metadata)，落 Track E _is_in_time_range fallback 不過濾。
        """
        from reasoning.schemas_live import EvidencePoolEntry

        url = item.get("url", "") or ""
        if not url:
            return
        if url in self._url_to_id:
            return  # 已存在 → 不重複建（沿既有 eid）

        snippet = (item.get("snippet", "") or "")[:300]
        source_domain = _extract_domain(url)
        title = await self._resolve_external_title(
            raw_title=item.get("title", "") or "",
            snippet=snippet,
            source_domain=source_domain,
        )

        self._evidence_counter += 1
        eid = self._evidence_counter
        self._url_to_id[url] = eid
        entry = EvidencePoolEntry(
            evidence_id=eid,
            title=title,
            url=url,
            source_domain=source_domain,
            snippet=snippet,
            iteration_origin=self._current_iteration,
            source=source,
            published_at=None,  # F-12: wiki/web 不填 publish date
        )
        self.evidence_pool[eid] = entry
        logger.info(f"[LR gap routing] Added {source} evidence eid={eid} url={url[:60]}")

    async def _resolve_external_title(
        self, raw_title: str, snippet: str, source_domain: str
    ) -> str:
        """外部來源 entry title 決議：有正常標題用原值，無標題補標題 (2026-06-17)。

        觸發補標題：raw_title 為空字串 OR 等於 "No Title"（Google CSE fallback）。
        補標題策略（CEO 拍板，順序）：
          1. snippet 也空 → 直接用 source_domain（沒內文餵 LLM 沒意義，不呼叫 LLM）。
          2. 本 run 已達 TITLE_BACKFILL_CAP → 直接用 source_domain（不呼叫 LLM，省錢）。
          3. 否則 low-tier LLM 從 snippet 生成簡潔中文標題；計數器 +1。
             LLM 失敗 / timeout / 空回應 → 降級 source_domain + log（不可 silent fail）。
        """
        if raw_title and raw_title != _NO_TITLE_SENTINEL:
            return raw_title  # 有正常標題 → 不動（省錢）

        # 無內文 → LLM 無從生成，直接 domain（不燒錢）
        if not snippet:
            return source_domain

        # per-run cap：超過上限不再呼叫 LLM
        if self._title_backfill_count >= TITLE_BACKFILL_CAP:
            return source_domain

        self._title_backfill_count += 1
        from core.llm import ask_llm
        from reasoning.schemas_live import GeneratedTitle

        prompt = (
            "以下是一篇外部來源（網路 / 維基百科）的內文摘要，但這篇來源沒有標題。\n"
            "請根據摘要內容，生成一個簡潔的繁體中文標題（不超過 30 字），"
            "概括這篇來源的主題。\n"
            "禁止使用「相關報導」「新聞」「文章」等泛化詞，標題要具體點出主題。\n\n"
            f"內文摘要：\n{snippet}"
        )
        try:
            response = await ask_llm(
                prompt,
                GeneratedTitle.model_json_schema(),
                level="low",
                query_params=getattr(self.handler, "query_params", {}),
                timeout=_TITLE_BACKFILL_TIMEOUT,
            )
            generated = (GeneratedTitle.model_validate(response).title or "").strip()
            if generated:
                return generated
            # 空回應 → 降級（fail-loud-with-info，不留空標題）
            logger.warning(
                f"[LR title backfill] LLM returned empty title, "
                f"degrading to source_domain='{source_domain}'"
            )
            return source_domain
        except Exception as e:
            # 失敗 / timeout → 降級 source_domain（不可 silent fail）
            logger.warning(
                f"[LR title backfill] LLM title generation failed "
                f"({type(e).__name__}: {e}), degrading to source_domain='{source_domain}'"
            )
            return source_domain

    @staticmethod
    def _merge_knowledge_graph(state_kg, new_kg):
        """Track D D1 (sprint 2026-05-28): merge new_kg 進 state_kg。

        D-AMB-2 LOCKED 2026-05-28:
        - entity dedup by name.lower().strip()
        - relationship dedup by (remapped_src, str(rel_type), remapped_tgt) triple
        - evidence_ids set union (不保留重複)
        - 沿用 existing entity_id (new_kg entity_id 被丟棄)
        - relationship 的 source/target entity_id remap 到 existing entity_id

        Args:
            state_kg: 當前累積的 KnowledgeGraph (None 表示尚未啟動 → fast path)
            new_kg: Analyst 新輸出的 KnowledgeGraph

        Returns:
            合併後新的 KnowledgeGraph instance (不 mutate 原 state_kg)

        N-5: empty new_kg → fast return；
        N-6: name normalization 用 .lower().strip()，不做更複雜處理；
        N-7: dangling relationship 由 DR validate_relationships() 自動 filter。
        """
        from reasoning.schemas_enhanced import KnowledgeGraph

        # N-5: empty new_kg no-op (info log 由 caller 決定)
        if not new_kg.entities and not new_kg.relationships:
            return state_kg if state_kg is not None else new_kg

        # state=None fast path — fresh start, 不跑 dedup logic (留給後續 iteration)
        if state_kg is None:
            return new_kg

        # fix-up round 1 I-3 / R2-C1 / Gemini R4 I-3: deep copy state_kg
        # 防 in-place mutation 副作用 — merge 過程 mutate evidence_ids 直接改 state_kg
        # 內 reference，若 merge 過程 exception raise，state 進入半 merged 狀態
        # (fail-unsafe)。deep_copy 後改 working copy，return 新 instance 是 fail-safe path。
        state_kg = state_kg.model_copy(deep=True)

        name_to_eid: Dict[str, str] = {
            e.name.lower().strip(): e.entity_id
            for e in state_kg.entities
        }

        merged_entities = list(state_kg.entities)
        new_eid_to_existing: Dict[str, str] = {}

        for new_e in new_kg.entities:
            norm_name = new_e.name.lower().strip()
            if not norm_name:
                continue  # skip 空 name entity (防 Analyst hallucinate)
            if norm_name in name_to_eid:
                # dedup: merge evidence_ids 進 existing entity
                existing_eid = name_to_eid[norm_name]
                existing = next(
                    e for e in merged_entities if e.entity_id == existing_eid
                )
                existing.evidence_ids = sorted(
                    set(existing.evidence_ids) | set(new_e.evidence_ids)
                )
                new_eid_to_existing[new_e.entity_id] = existing_eid
            else:
                # new entity
                merged_entities.append(new_e)
                name_to_eid[norm_name] = new_e.entity_id

        # Relationship triple dedup
        existing_triples: set = {
            (r.source_entity_id, str(r.relation_type), r.target_entity_id)
            for r in state_kg.relationships
        }
        merged_relationships = list(state_kg.relationships)

        for new_r in new_kg.relationships:
            # remap new entity_id → existing entity_id (若 entity 被 dedup)
            src = new_eid_to_existing.get(new_r.source_entity_id, new_r.source_entity_id)
            tgt = new_eid_to_existing.get(new_r.target_entity_id, new_r.target_entity_id)
            rtype_str = str(new_r.relation_type)
            triple = (src, rtype_str, tgt)
            if triple in existing_triples:
                # dedup: merge evidence_ids
                existing = next(
                    r for r in merged_relationships
                    if r.source_entity_id == src
                    and str(r.relation_type) == rtype_str
                    and r.target_entity_id == tgt
                )
                existing.evidence_ids = sorted(
                    set(existing.evidence_ids) | set(new_r.evidence_ids)
                )
            else:
                # remap & add
                new_r.source_entity_id = src
                new_r.target_entity_id = tgt
                merged_relationships.append(new_r)
                existing_triples.add(triple)

        # N-7: dangling relationship 由 KnowledgeGraph.validate_relationships
        # 自動 filter (Pydantic field_validator)
        return KnowledgeGraph(
            entities=merged_entities,
            relationships=merged_relationships,
        )

    @staticmethod
    def _has_mini_reasoning_input(formatted_results) -> bool:
        """Phase 3 是否有可分析輸入。

        s5-4 收斂：run_loop 的 phase3 emit gate 與 _run_mini_reasoning 開頭
        early-skip 用同一判準、單一定義 — sentinel 字串不得各寫一份（drift
        即 gate 失效）。sentinel 來源：_execute_search 空手時回傳
        「（未找到相關結果）」。
        """
        return bool(formatted_results) and formatted_results != "（未找到相關結果）"

    def _build_analyst_research_kwargs(self, research_question, enriched_context, cm_summary):
        """Task 15 (behavior-preserving): 兩處 analyst.research(...) kwargs 完全相同，
        抽共用 builder 防 drift。value-expression 與兩 call site 逐欄一致：
        query=context_map.research_question / formatted_context=enriched_context /
        mode="discovery" / enable_live_research=True / context_map_summary=cm_summary /
        enable_web_search=getattr(self.handler, "enable_web_search", False) / enable_kg=True。
        """
        return {
            "query": research_question,
            "formatted_context": enriched_context,
            "mode": "discovery",
            "enable_live_research": True,
            "context_map_summary": cm_summary,
            "enable_web_search": getattr(self.handler, "enable_web_search", False),
            "enable_kg": True,
        }

    async def _run_mini_reasoning(self, context_map, formatted_results):
        """
        Mini-Reasoning：對新 evidence 跑 Analyst + Critic。

        使用現有 AnalystAgent.research() + CriticAgent.review()，
        注入 ContextMap summary 作為額外 context（live research 模式）。
        失敗為 non-fatal，catch Exception 後 log warning 並繼續。

        Track A (sprint 2026-05-28) — Task 1 + Task 6:
        Analyst argument_graph 跑完 + Critic status 接到 → 索引進
        state.evidence_usage[eid] 每個 evidence_id 一筆 GroundedClaim:
        - status="REJECT" → 入庫並標 critic_status="REJECT"（forensic trail，
          Task 3 render 層 filter 不入 writer prompt），同步 append rejected_claims_log
        - status="WARN" → 入庫，confidence 降為 "low" + critic_status="WARN"
          + entry 帶 from_warned_critic_review=True tag
        - status="PASS"（或 default）→ 正常索引 + critic_status="PASS"

        Returns:
            bool — 本體成功 True / early-skip 或整段失敗 False（s5-4：run_loop
            據此條件 emit bab_phase3 completed）。
        """
        if not self._has_mini_reasoning_input(formatted_results):
            logger.info("[BAB LOOP] No results for mini-reasoning, skipping")
            return False

        from reasoning.agents.analyst import AnalystAgent
        from reasoning.agents.critic import CriticAgent
        # 函式內 import（對齊本檔既有 function-local CONFIG pattern :578/620/697，
        # 並支援 unit test monkeypatch CONFIG）。LR 沿用 DR config key
        # (analyst_timeout=300 / critic_timeout=120)；config key 已存在，
        # fallback 120 僅在「忘了帶 config」時兜底（對齊 base.py:168 / analyst.py / critic.py）。
        from core.config import CONFIG

        analyst = AnalystAgent(
            self.handler,
            timeout=CONFIG.reasoning_params.get("analyst_timeout", 120),
        )
        critic = CriticAgent(
            self.handler,
            timeout=CONFIG.reasoning_params.get("critic_timeout", 120),
        )

        # Inject ContextMap summary into analyst context (prepended)
        cm_summary = context_map_to_summary(context_map)
        enriched_context = f"{cm_summary}\n\n---\n\n{formatted_results}"

        # Track C C4 (F-4 紀律 2026-05-28): analyst_output 必須在 try 外初始化，
        # 否則 try 內 raise 時後續 gap routing block (try 外) 會 NameError。
        analyst_output = None

        # s5-4: mini-reasoning 本體成敗狀態（outer try 成功 True / 整段失敗 False）。
        # gap routing 失敗（O5-C）不翻轉它（耦合判定 2 — 分析本體已成功）。
        mini_ok = True

        try:
            # Analyst pass — use actual parameter name: formatted_context
            # Track C C4 (F-7 fix 2026-05-28): 接通 enable_web_search 傳遞鏈，
            # 讓 Analyst prompt _build_mandatory_precheck 段在
            # enable_web_search=True AND CONFIG.gap_knowledge_enrichment=True
            # 同時成立時 inject → Analyst 才會主動標 gap_resolutions。
            # Track D D1 (sprint 2026-05-28): enable_kg=True 啟用 KG 生成。
            # D-CEO-Q6 LOCKED 預設 ON — LR 永遠跑 KG (LR 是重度研究模式)。
            # Analyst.research(enable_kg=True) → 走 AnalystResearchOutputLive
            # schema (inherit AnalystResearchOutputEnhancedKG, 含 knowledge_graph
            # field) + 注入 KG instruction prompt block → LLM 輸出 knowledge_graph。
            # Task 15: kwargs 與 Task2 re-run site 共用 _build_analyst_research_kwargs。
            analyst_output = await analyst.research(
                **self._build_analyst_research_kwargs(
                    context_map.research_question, enriched_context, cm_summary
                )
            )
            logger.info("[BAB LOOP] Mini-reasoning: Analyst pass complete")

            # Task 2 (DR-parity SEARCH_REQUIRED): Analyst 喊站內資料不足 → 補搜站內 evidence → 重跑 Analyst。
            # 補搜走 BAB 既有 _execute_search path（不新建檢索器），上限 1 次（與外層 BAB iteration 隔離）。
            # 邊界：此處補的是 Analyst 頂層 status=SEARCH_REQUIRED（即時補救），與 gap_resolutions
            # INTERNAL_SEARCH no-op（交給下一輪 Associator）不同層、不重複。
            # 可引用性（CEO 2026-06-12 拍板）：補搜到的新 evidence 經 _execute_search side-effect
            # 寫入 self.evidence_pool（:540）→ BAB 結束後 serialize 進 state.evidence_pool_json
            # （orchestrator :938/1668）→ outline planner deserialize 全 pool 做 planned_evidence_ids
            # 分配（orchestrator :3471）→ render_grounded_narrative 餵 writer findings = writer 可引用。
            analyst_output, enriched_context = await self._handle_search_required_supplementary(
                analyst_output, enriched_context, context_map, cm_summary, analyst
            )

            # Critic pass (if analyst produced a non-empty draft) — Track A Task 6:
            # Critic status (CriticReviewOutput.status: Literal["PASS","WARN","REJECT"])
            # 決定 Analyst claims 的 critic_status 標記。
            critic_status = "PASS"  # default (Critic 沒跑 / 沒 draft → PASS)
            if analyst_output and hasattr(analyst_output, 'draft') and analyst_output.draft:
                critic_output = await critic.review(
                    draft=analyst_output.draft,
                    query=context_map.research_question,
                    mode="discovery",
                    formatted_context=enriched_context,
                    enable_live_research=True,
                )
                # CriticReviewOutput.status 是 schema 真實欄位（不是 verdict）
                critic_status = str(
                    getattr(critic_output, 'status', 'PASS')
                ).upper()
                if critic_status not in ("PASS", "WARN", "REJECT"):
                    critic_status = "PASS"
                logger.info(
                    f"[BAB LOOP] Mini-reasoning: Critic status={critic_status}"
                )

            # Task 1 (DR-parity revise loop): Critic REJECT → analyst.revise() 重寫該批推論 → re-review。
            # 上限 1 輪（非 DR 的 3 輪 — LR mini-reasoning per-topic 內嵌，外層 BAB max_iterations 已疊乘）。
            # 退出：revise 後 PASS/WARN → 用 revised output 走正常索引；仍 REJECT 或 revise 失敗 → 維持
            # 既有 REJECT 入庫 forensic + render 過濾。critic REJECT 才進此迴圈。
            MAX_REVISE = 1
            revise_count = 0
            while (
                critic_status == "REJECT"
                and revise_count < MAX_REVISE
                and analyst_output is not None
                and getattr(analyst_output, "draft", None)
            ):
                revise_count += 1
                try:
                    revised = await analyst.revise(
                        original_draft=analyst_output.draft,
                        review=critic_output,
                        formatted_context=enriched_context,
                        query=context_map.research_question,
                        enable_live_research=True,  # B9/C1: match LR research() schema (keep KG type)
                        enable_kg=True,             # B9/C1: match LR research() schema (keep KG type)
                    )
                except Exception as e:
                    # revise 這一步失敗 → 降級旁白（不可 silent fail），退回原 REJECT 入庫路徑。
                    logger.warning(
                        f"[BAB LOOP] Task1 revise failed (non-fatal): {e}", exc_info=True
                    )
                    await self._narrate_once("_revise_degraded_narrated", lr_copy.MINI_REASONING_REVISE_DEGRADED_NARRATION)
                    break  # 退回原 analyst_output 的 REJECT 入庫
                # revised 沒 draft → 無法 re-review，退回原路徑
                if not getattr(revised, "draft", None):
                    logger.info("[BAB LOOP] Task1 revise produced empty draft; keeping original REJECT")
                    break
                # M-4（AR round1 雙家收斂 IH S-4 / CX SF#4）：re-review 的 critic.review **必須**
                # 也納入內層 try/except。否則 re-review 拋例外會冒泡到 outer except 被吞成
                # 通用「Mini-reasoning failed」旁白（違反不可 silent fail：顯示的是整段失敗而非
                # revise/re-review 失敗），且會中斷後續本應正常入庫的 forensic trail + KG merge。
                # re-review 失敗 → 維持 critic_status 原值（原 REJECT）+ emit 降級旁白 + break，
                # 讓迴圈外的索引/merge 用「原 analyst_output + 原 REJECT」正常跑完。
                try:
                    re_review = await critic.review(
                        draft=revised.draft,
                        query=context_map.research_question,
                        mode="discovery",
                        formatted_context=enriched_context,
                        enable_live_research=True,
                    )
                except Exception as e:
                    logger.warning(
                        f"[BAB LOOP] Task1 re-review failed (non-fatal): {e}", exc_info=True
                    )
                    await self._narrate_once("_revise_degraded_narrated", lr_copy.MINI_REASONING_REVISE_DEGRADED_NARRATION)
                    break  # 退回原 analyst_output / 原 critic_status 的 REJECT 入庫
                new_status = str(getattr(re_review, "status", "PASS")).upper()
                if new_status not in ("PASS", "WARN", "REJECT"):
                    new_status = "PASS"
                logger.info(
                    f"[BAB LOOP] Task1 revise #{revise_count}: critic re-review status={new_status}"
                )
                # 採用 revised 結果（不論 PASS/WARN/REJECT 都以 revised 為準 — 它是更新後的推論）
                analyst_output = revised
                critic_output = re_review
                critic_status = new_status
                # PASS/WARN 達成 → 退出迴圈；仍 REJECT 且未達上限 → 迴圈續（此處 MAX_REVISE=1 即停）

            self._index_argument_graph(analyst_output, critic_status)

            await self._merge_kg_from_analyst(analyst_output, critic_status)
        except Exception as e:
            # s3-3 review（in-house N2 + Gemini）：對齊 Track C except，補 stack trace
            #（非 LLM 類程式錯誤如 GroundedClaim 建構失敗，無 trace 極難 debug）。
            logger.warning(
                f"[BAB LOOP] Mini-reasoning failed (non-fatal): {e}", exc_info=True
            )
            # O5-B: 降級必有 user-facing 訊息（CLAUDE.md 不可 silent fail）。
            # mini-reasoning（Analyst+Critic，最耗時推理）整段失敗時，本輪
            # evidence_usage 不被索引，下游某章可能落 blocked_no_evidence；若不告知，
            # user 只看到「資料不足」而誤以為真的沒資料（實為 LLM 推理出狀況）。
            # s3-3（三家收斂）：per-run 只提示一次，防持續性失敗（429 貫穿 run）
            # 每輪轟炸——照同檔 consistency 先例（_consistency_degraded_narrated）。
            # log 每輪照記，只有 user-facing 旁白 dedup。
            await self._narrate_once("_mini_reasoning_degraded_narrated", "這一輪的深入分析遇到狀況、已先略過，"
                "稍後若有章節顯示資料不足，可能與此有關。我會繼續往下進行。")
            # s5-4: 本體失敗 → run_loop 不 emit bab_phase3 completed（耦合判定 1）。
            mini_ok = False

        # Track C C4 (F-4 紀律 2026-05-28): post-mini-reasoning gap routing.
        # 必須在 outer try/except 之外獨立執行 — 若放 outer try 內，gap routing 內任何
        # 例外都會被上方 outer except swallow 成 "Mini-reasoning failed (non-fatal)"
        # warning，違反 CLAUDE.md「不可 silent fail」紀律（gap routing 失敗應該有自己
        # 明確訊息，不可借用 mini-reasoning 的 warning 蓋掉）。
        # Toggle gate 在 _process_gap_resolutions_lr 內判定
        # (enable_gap_enrichment / enable_web_search)。
        if (
            analyst_output
            and hasattr(analyst_output, "gap_resolutions")
            and analyst_output.gap_resolutions
        ):
            await self._run_gap_routing_phase(analyst_output)

        # s5-4: 回傳值只反映 mini-reasoning 本體成敗（outer try）。gap routing 失敗
        # （O5-C，_run_gap_routing_phase 自帶降級旁白）不翻轉 mini_ok（耦合判定 2 —
        # 分析本體已成功，completed 語意保真）。
        return mini_ok

    def _index_argument_graph(self, analyst_output, critic_status) -> None:
        """Index Analyst argument_graph into state.evidence_usage (Track A T1+T6).
        REJECT 也入庫保留 forensic trail（Task 3 render 層 filter）。
        in-place 寫 self.state.evidence_usage / rejected_claims_log；不回傳。
        critic_status 與 analyst_output 由 caller 顯式傳入（跨 block mutable local）。"""
        if (
            self.state is not None
            and analyst_output
            and hasattr(analyst_output, 'argument_graph')
            and analyst_output.argument_graph
        ):
            from reasoning.schemas_live import GroundedClaim
            for node in analyst_output.argument_graph:
                rtype = getattr(node, 'reasoning_type', 'induction')
                # LogicType enum → str
                rtype_str = rtype.value if hasattr(rtype, 'value') else str(rtype)
                raw_conf = str(getattr(node, 'confidence', 'medium'))
                # WARN / REJECT 兩種情況都把 confidence 視為 low
                # (REJECT 降級或保留視 metadata 用途; critic_status 為主鍵)
                eff_conf = (
                    "low" if critic_status in ("WARN", "REJECT") else raw_conf
                )
                eff_critic_status = critic_status  # PASS / WARN / REJECT
                for eid in (getattr(node, 'evidence_ids', None) or []):
                    # T1 Fix 2: C-4 invariant #3 — eid 必須在 evidence_pool 中
                    # 若 Analyst 幻覺出不存在的 eid，skip 並 warning（不 raise）
                    if eid not in self.evidence_pool:
                        logger.warning(
                            f"[BAB invariant #3] Analyst hallucinated eid={eid} not in evidence_pool "
                            f"(topic={self._current_topic_id}, status={critic_status}); skipping indexing"
                        )
                        continue
                    gc = GroundedClaim(
                        claim=node.claim,
                        reasoning_type=rtype_str,
                        confidence=eff_conf,
                        source_topic=self._current_topic_id or "global",
                        source_iteration=self._current_iteration,
                        critic_status=eff_critic_status,
                        # T1 Fix 1: 正式欄位取代 dict key inject，確保 model_validate round-trip 不 drop
                        from_warned_critic_review=(critic_status == "WARN"),
                    )
                    entry = gc.model_dump()
                    self.state.evidence_usage.setdefault(eid, []).append(entry)
            logger.info(
                f"[BAB LOOP] Indexed argument_graph (critic={critic_status}): "
                f"{len(analyst_output.argument_graph)} nodes, "
                f"total eids={len(self.state.evidence_usage)}"
            )
            # Gemini C-1: REJECT batch 同步 append rejected_claims_log
            # (metadata trace — oncall 可直接撈某次 reject batch)
            if critic_status == "REJECT":
                if not hasattr(self.state, "rejected_claims_log") or \
                        self.state.rejected_claims_log is None:
                    self.state.rejected_claims_log = []
                self.state.rejected_claims_log.append({
                    "topic_id": self._current_topic_id or "global",
                    "iteration": self._current_iteration,
                    "claim_count": len(analyst_output.argument_graph),
                    "evidence_ids": sorted({
                        eid for n in analyst_output.argument_graph
                        for eid in (getattr(n, 'evidence_ids', None) or [])
                    }),
                    "reason": "critic_status_reject",
                })
                logger.warning(
                    f"[BAB LOOP] Critic REJECT — claims indexed with "
                    f"critic_status='REJECT' (forensic trail); will be "
                    f"filtered by render_grounded_narrative"
                )

    async def _merge_kg_from_analyst(self, analyst_output, critic_status) -> None:
        """Merge analyst KG into self.state.knowledge_graph (Track D D1).
        Critic REJECT → 跳過（KG 不入庫）。KG merge 失敗為 non-fatal：log + per-run
        降級旁白（不可 silent fail）原樣保留在內。analyst_output / critic_status 由
        caller 顯式傳入（跨 block mutable local）。"""
        # Track D D1 (sprint 2026-05-28): merge analyst KG into state.knowledge_graph
        # D-AMB-4 LOCKED: Critic REJECT → 跳過 KG merge (KG 不入庫；跟 Track A T6
        # evidence_usage REJECT 入庫 forensic trail 紀律刻意不同 — KG 是 LLM 生成
        # structured data, REJECT 表示推斷整段不可信, 入庫只是噪音)
        # fix-up round 1 I-5: critic_status 變數由 Track A T6 上方 block 設定
        # (default "PASS" + normalize uppercase) — 不可改名 / refactor 必須同步 Track D
        # N-5: empty new_kg → _merge_knowledge_graph 內部 fast return (no-op)
        if (
            self.state is not None
            and critic_status != "REJECT"
            and analyst_output
            and hasattr(analyst_output, 'knowledge_graph')
            and analyst_output.knowledge_graph
            and (
                analyst_output.knowledge_graph.entities
                or analyst_output.knowledge_graph.relationships
            )
        ):
            try:
                new_kg = analyst_output.knowledge_graph
                merged = self._merge_knowledge_graph(
                    self.state.knowledge_graph,
                    new_kg,
                )
                self.state.knowledge_graph = merged
                logger.info(
                    f"[BAB LOOP][Track D] KG merged: "
                    f"+{len(new_kg.entities)} entities, "
                    f"+{len(new_kg.relationships)} relationships; "
                    f"state total: {len(merged.entities)} entities, "
                    f"{len(merged.relationships)} relationships"
                )
                # Stop-and-Report #2 (plan v6 §9): 大 KG warning
                if len(merged.entities) > 100:
                    logger.warning(
                        f"[BAB LOOP][Track D] state.knowledge_graph.entities "
                        f"count exceeded 100 ({len(merged.entities)}); KG render "
                        f"layout may break (kg-spec §11). Consider review Analyst "
                        f"KG instruction tightness."
                    )
            except Exception as e:
                logger.warning(
                    f"[BAB LOOP][Track D] KG merge failed (non-fatal): {e}"
                )
                # O5a-(2): 降級必有 user-facing 訊息（CLAUDE.md 不可 silent fail）。
                # _run_mini_reasoning 每輪由 run_loop 呼叫；KG merge 持續性失敗
                #（如 429 貫穿 run）會每輪觸發 → per-run 只提示一次防轟炸
                #（照同檔 _mini_reasoning_degraded_narrated / _gap_routing_degraded_narrated
                # 先例）。log 每輪照記，只有 user-facing 旁白 dedup。
                await self._narrate_once("_kg_merge_degraded_narrated", lr_copy.KG_MERGE_DEGRADED_NARRATION)

    async def _handle_search_required_supplementary(
        self, analyst_output, enriched_context, context_map, cm_summary, analyst
    ):
        """Analyst status=SEARCH_REQUIRED → 補搜站內 evidence → 重跑 Analyst 一次（上限 1）。
        補搜後 enriched_context (append) 與 analyst_output (replace) 被更新，**顯式回傳**
        (analyst_output, enriched_context) 給 caller 續用（跨 block mutable local，不可靠閉包）。
        補搜失敗 / 無結果的降級旁白（不可 silent fail）原樣保留在內。"""
        if (
            analyst_output is not None
            and str(getattr(analyst_output, "status", "")).upper() == "SEARCH_REQUIRED"
            and getattr(analyst_output, "new_queries", None)
        ):
            # M-10（AR CX SF#5 / Q5）：consumer 層硬限 — 去重 + 過濾空字串 + cap 3 條
            # （即使 Analyst prompt 要求 1-3，runtime 仍須兜底防 LLM 吐超量 / 空 query）。
            _seen = set()
            _capped_queries = []
            for q in analyst_output.new_queries:
                # AR R2 Codex nit：先 strip 再去重（否則 " query" 與 "query" 被當不同條）
                qn = (q or "").strip()
                if qn and qn not in _seen:
                    _seen.add(qn)
                    _capped_queries.append(qn)
                if len(_capped_queries) >= 3:
                    break
            if not _capped_queries:
                logger.info("[BAB LOOP] Task2 SEARCH_REQUIRED but new_queries empty after dedup; skipping")
            else:
                logger.info(
                    f"[BAB LOOP] Task2 SEARCH_REQUIRED: Analyst requested "
                    f"{len(_capped_queries)} secondary queries (capped from {len(analyst_output.new_queries)})"
                )
                # 把 capped queries 包成 _execute_search 吃的 seed-like 物件（讀 .query / .source_strategy）
                from types import SimpleNamespace
                secondary_seeds = [
                    SimpleNamespace(query=q, source_strategy="internal")
                    for q in _capped_queries
                ]
                # 註：secondary_formatted, _ = ... 丟棄回傳 source_map 是**安全的** —
                # _execute_search 已 side-effect 寫 self.evidence_pool（:540）。不需手動 merge source_map。
                try:
                    secondary_formatted, _ = await self._execute_search(secondary_seeds)
                except Exception as e:
                    logger.warning(
                        f"[BAB LOOP] Task2 secondary search failed (non-fatal): {e}",
                        exc_info=True,
                    )
                    secondary_formatted = "（未找到相關結果）"
                if secondary_formatted and secondary_formatted != "（未找到相關結果）":
                    # 補到 evidence → 重組 context 重跑 Analyst 一次（上限 1 次）
                    enriched_context = (
                        f"{enriched_context}\n\n--- 補充資料 ---\n\n{secondary_formatted}"
                    )
                    # Task 15: 與首次 Analyst pass 共用 builder（kwargs 完全相同）。
                    analyst_output = await analyst.research(
                        **self._build_analyst_research_kwargs(
                            context_map.research_question, enriched_context, cm_summary
                        )
                    )
                    logger.info("[BAB LOOP] Task2: Analyst re-run after secondary search complete")
                    # AR R2 Codex should-fix：第二次仍非 DRAFT_READY 或 draft 空 → 不可 silent no-op。
                    # 補搜有結果但 Analyst 重跑後仍判資料不足 / 沒寫出 draft（→ critic 不跑、該批
                    # 推論落空），須 forensic log + user-facing 降級旁白（不可 silent fail）。
                    _rerun_status = str(getattr(analyst_output, "status", "")).upper()
                    _rerun_draft_len = len(getattr(analyst_output, "draft", None) or "")
                    if _rerun_status != "DRAFT_READY" or _rerun_draft_len == 0:
                        logger.warning(
                            f"[BAB LOOP] Task2: secondary search re-run still inconclusive "
                            f"(status={_rerun_status!r}, draft_len={_rerun_draft_len}, "
                            f"queries={_capped_queries}); continuing with existing evidence"
                        )
                        await self._narrate_once("_search_required_degraded_narrated", lr_copy.SEARCH_REQUIRED_DEGRADED_NARRATION)
                else:
                    # 無結果 → 降級旁白（不可 silent fail），用原 analyst_output（draft 空 → critic 不跑）
                    await self._narrate_once("_search_required_degraded_narrated", lr_copy.SEARCH_REQUIRED_DEGRADED_NARRATION)
        return analyst_output, enriched_context

    async def _run_gap_routing_phase(self, analyst_output) -> None:
        """Track C C4：post-mini-reasoning gap routing，獨立 except + 降級 narration。

        gap routing 把 Analyst 標出的 gap 補進 evidence_pool。失敗時 non-fatal
        （不擋 BAB 主流程），但必須 emit user-facing narration（CLAUDE.md 不可
        silent fail）—— 否則使用者拿到的報告少了補強證據卻完全無感知。
        s3-4 收斂：旁白 per-run dedup（防 429 貫穿 run 每輪轟炸），log 每輪照記。
        """
        logger.info(
            f"[BAB LOOP][Track C] Processing "
            f"{len(analyst_output.gap_resolutions)} gap resolutions"
        )
        try:
            await self._process_gap_resolutions_lr(analyst_output.gap_resolutions)
        except Exception as e:
            # 獨立 except — 明確標 [Track C] 來源 + non-fatal（不擋 BAB 主流程）
            # exc_info=True 保留（既有 code 已有，helper 化不可丟失）
            logger.warning(
                f"[BAB LOOP][Track C] gap routing failed (non-fatal): {e}",
                exc_info=True,
            )
            await self._narrate_once("_gap_routing_degraded_narrated", "這一輪的補強查證沒能完成 — 用外部資料（維基／網路／"
                "背景知識）補上資料缺口時出了狀況，這部分先略過，"
                "研究照常繼續。")

    async def _run_consistency_check(
        self, current_map: ContextMap, initial_map: ContextMap
    ) -> ConsistencyReview:
        """
        執行 Consistency Monitor。

        使用已實作的 ConsistencyPromptBuilder + Critic agent。
        """
        from reasoning.prompts.consistency import ConsistencyPromptBuilder
        from reasoning.schemas_live import ConsistencyReview as CR

        builder = ConsistencyPromptBuilder()
        prompt = builder.build_consistency_check_prompt(
            current_map_summary=context_map_to_summary(current_map),
            initial_map_summary=context_map_to_summary(initial_map),
            recent_events=self.executed_searches[-5:]  # Last 5 searches as events
        )

        # Use handler's LLM infrastructure
        from core.llm import ask_llm
        try:
            response = await ask_llm(
                prompt,
                CR.model_json_schema(),
                level="low",
                query_params=getattr(self.handler, 'query_params', {}),
            )
            return CR.model_validate(response)
        except Exception as e:
            logger.warning(f"[BAB LOOP] Consistency check failed: {e}")
            return CR(
                drift_level="none",
                drift_description="一致性檢查失敗，預設為無漂移",
                dubao_voice_message="",
                recommended_action="continue",
                monitor_degraded=True,
            )

    def _check_connection(self):
        """檢查客戶端連線狀態，回傳 cooperative-stop 訊號。

        plan: lr-disconnect-midstage-persist — 語義分流（root cause 修復）：
        - **client offline（純斷線）**：回 "offline"（**不 raise**）。舊行為 raise
          會把「api_routes 刻意要背景跑完」的 task 自己殺掉（自相矛盾），且走 error
          路徑無 finally persist → Stage 2 累積 evidence 全蒸發。改回訊號讓 run_loop
          有序收場（break + return 已累積 context_map/evidence_pool，caller persist）。
        - **soft interrupt（使用者主動打字中斷研究）**：仍 raise ResearchCancelledError
          （使用者要立刻停，語義與純斷線不同）。
        - **在線**：回 None。

        3-signal 檢查順序與 OrchestratorBase._check_connection 對齊。
        """
        from reasoning.orchestrator_base import ResearchCancelledError

        wrapper = getattr(self.handler, 'http_handler', None)
        event = getattr(self.handler, 'connection_alive_event', None)

        # Signal 3 先判：soft interrupt = 使用者主動打斷 → raise（立刻停）。
        # 放最前面確保「使用者主動停」優先於「純斷線 cooperative」語義。
        soft_interrupt = getattr(self.handler, '_soft_interrupt_event', None)
        if soft_interrupt and soft_interrupt.is_set():
            raise ResearchCancelledError("User interrupted BAB loop (soft)")

        # Signal 1: wrapper connection_alive flag → offline（cooperative，不 raise）
        if wrapper and not wrapper.connection_alive:
            # 同步清除 event，確保下游 offline 檢查一致（沿舊行為）
            if event and event.is_set():
                event.clear()
            return "offline"

        # Signal 2: handler connection_alive_event → offline（cooperative，不 raise）
        if event and not event.is_set():
            return "offline"

        return None

    async def _emit_narration(self, text: str):
        """推送讀豹旁白到前端。"""
        if not text:
            return
        await emit_sse(self.handler, {
            "message_type": "live_research_narration",
            "text": text,
        })

    async def _narrate_once(self, flag_name, narration_text):
        """per-run 只發一次的降級旁白：用 flag_name 去重。
        降級訊息必須明確發出（不可 silent fail），但同一 run 不重複洗版。
        flag_name 屬 per-run dedup 旗標族（_reset_per_run_dedup_flags 在 run_loop
        入口重置 → engine 池化重用時第二次 run 不會被永久靜音，見 F2）。
        新增旗標時務必同步登記進 _reset_per_run_dedup_flags。"""
        if getattr(self, flag_name, False):
            return
        setattr(self, flag_name, True)
        await self._emit_narration(narration_text)

    async def _emit_phase(self, phase_name: str, status: str):
        """推送 phase 進度事件到前端。"""
        await emit_sse(self.handler, {
            "message_type": "research_phase",
            "phase": phase_name,
            "status": status,
        })
