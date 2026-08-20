"""
Live Research Pydantic schemas.

Defines ContextMap (Master B), AssociatorOutput variants, ConsistencyReview,
StyleAnalysis, LiveWriterSectionOutput, and AnalystResearchOutputLive
for the Live Research pipeline.

All field descriptions are in Traditional Chinese.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from core.temporal_anchor import annotate_relative_years, anchor_note
from misc.logger.logging_config_helper import get_configured_logger

logger = get_configured_logger("reasoning.schemas_live")


# ============================================================================
# Track A (LR DR-parity sprint 2026-05-28) — grounding schema additions
# ============================================================================

class InvariantViolation(Exception):
    """codex Imp-1 (Track A sprint 2026-05-28)：schema invariant fail-loud exception
    （test / CI 模式專用）。

    Runtime（prod / dev）違反 invariant 只 log warning + caller mark `guard_failed`；
    test / CI fixture 透過 `LR_STRICT_INVARIANTS=1` env var 升為 raise，確保
    invariant violation 在 CI 必 fail（不可被 silent log 蓋掉）。
    """
    pass


class GroundedClaim(BaseModel):
    """單一 Analyst 推理 claim 與其支持 evidence_id 的綁定（Track A 2026-05-28）。

    每筆 GroundedClaim 對應 Analyst argument_graph 一個 node 對某個
    evidence_id 的 grounding 關係（一個 node 有 N 個 evidence_ids 會
    展開為 N 筆 GroundedClaim，分別 append 進 state.evidence_usage[eid]）。

    本欄位由 loop_engine._run_mini_reasoning 寫入，由 Stage 5 chapter
    writer 經 render_grounded_narrative 讀取。

    Gemini Critical 拍板（2026-05-28）：REJECT claim 必須**入庫並標記**
    `critic_status="REJECT"` 保留 forensic trail（3 個月後 oncall 查 DB 才能
    區分「Analyst 沒抽 / 檢索壞 / Critic 全殺」）；由 Task 3
    `render_grounded_narrative` 在 presentation 層 filter 不入 writer prompt。
    雙層職責清楚：source = data（保留全部，含 REJECT），render = presentation（過濾）。
    """
    claim: str = Field(..., description="Analyst argument_graph node 的 claim 文字")
    reasoning_type: str = Field(..., description="induction / deduction / abduction")
    confidence: str = Field(..., description="high / medium / low")
    source_topic: str = Field(..., description="哪個 BAB topic（或全域）的 Analyst 產生")
    source_iteration: int = Field(..., description="哪輪 BAB iteration 產生（1-based）")
    # addendum I-5 + Gemini C-1: Critic verdict 接回（Task 6 寫入），Track F1 可二次審查 WARN
    critic_status: Literal["PASS", "WARN", "REJECT"] = Field(
        default="PASS",
        description="該 claim 對應 BAB iteration 的 Critic status（PASS / WARN / REJECT）",
    )
    # T1 schema review Fix 1 (2026-05-28): 正式欄位取代 entry dict key-inject
    # 讓 model_validate / model_dump round-trip 不會 silent drop WARN 來源 marker。
    # True 代表此 claim 由 Critic WARN 降級（confidence 被壓為 low）產生，
    # 與「node 本來就 low confidence」可區分，Track F1 / consistency monitor 可據此做二次審查。
    from_warned_critic_review: bool = Field(
        default=False,
        description="此 claim 是否因 Critic WARN 降級而索引（True = 降級來源，False = 正常 PASS 或 REJECT）",
    )


class BlockedSection(BaseModel):
    """body chapter 在 _write_section 入口被 deterministic gate 攔下時的 structured output。

    addendum C-1 (Track A sprint 2026-05-28)：取代「empty evidence scope → 自由寫 prose」
    的 fail mode；走此 path 不呼叫 writer LLM，直接由 orchestrator 構造此結構放入
    written_sections。Stage 6 export 偵測有 `require_review=True` chapter → SSE 提醒
    user + final report header 警告 banner。
    """
    chapter_index: int = Field(..., description="0-based chapter index")
    title: str = Field(..., description="chapter title from outline")
    status: Literal["blocked_no_evidence", "guard_failed"] = Field(
        ...,
        description="blocked_no_evidence = 入口空 evidence；guard_failed = 第二次重寫仍 ungrounded",
    )
    content: str = Field(..., description="user-visible blocked 文字（含原因與建議行動）")
    require_review: bool = Field(
        default=True, description="Stage 6 export 偵測此 flag 為 True 時提醒 user"
    )


# ============================================================================
# Track F (LR DR-parity sprint 2026-05-28) — Critic 擴充 / Consistency Monitor / CoV-lite
# ============================================================================

class ClaimLevelIssue(BaseModel):
    """F1 critic flag 出的 single claim-level issue（claim-level fabrication 之一）。

    對應 Track F §2 fabrication enum 6 類 + other：
    numeric / temporal / causal / comparative / predictive / evaluative / other。
    """

    claim_type: Literal[
        "numeric", "temporal", "causal", "comparative",
        "predictive", "evaluative", "other",
    ] = Field(..., description="claim 類型（對應 Track F §2 fabrication enum）")
    claim_text: str = Field(..., description="critic 從 section 抽出的 claim 原文片段（≤ 100 字）")
    severity: Literal["reject", "warn"] = Field(
        ...,
        description="reject = block publish 走 status=critic_rejected；warn = 加 marker 不 block",
    )
    explanation: str = Field(
        default="",
        description="critic 解釋為何 flag，給 user-facing 也給 audit",
    )


class CriticSectionReview(BaseModel):
    """F1 per-section Critic publish gate 的輸出（Track F sprint 2026-05-28）。

    每個 section 寫完 + Track A T5 entity guard 跑完後，跑 F1 critic call 產生此 review。
    review 結果寫進 state.critic_section_reviews[section_index]，audit / Stage 6 export
    detection / frontend banner 都讀此欄位。

    F-AMB-2 LOCKED: REJECT → content 替換 blocked 文字 + status=critic_rejected；
                    WARN → 保留 content + critic_note marker。
    """

    section_index: int = Field(..., description="0-based section index（對齊 written_sections）")
    verdict: Literal["PASS", "WARN", "REJECT"] = Field(
        ..., description="critic 整體 verdict",
    )
    claim_issues: List[ClaimLevelIssue] = Field(
        default_factory=list,
        description="F1 critic flag 出的 claim-level issues（可多筆）",
    )
    overall_explanation: str = Field(
        default="",
        description="critic 對整 section 的 verdict 解釋（給 user 看的 narrative，≤ 200 字）",
    )
    # F3 CoV-lite 整合（同 publish gate call 後寫入；CoV 未跑 → None）
    cov_verification_summary: Optional[Dict] = Field(
        default=None,
        description=(
            "F3 CoV-lite verification 結果摘要（verified_count / unverified_count / "
            "contradicted_count / results）；None = F3 未跑或失敗"
        ),
    )


class ConsistencyDriftEntry(BaseModel):
    """F2 Consistency Monitor 持久化 entry（spec §9.2 自標未實現的補完）。

    loop_engine 每輪 BAB 後跑 _run_consistency_check 後 append 一筆 entry。
    F-AMB-3 LOCKED: 每輪都 append（drift_level=none 也 append，audit trail 完整）。

    **I-3 紀律**（adversarial review round 1, 2026-05-28）：
    `iteration` 編號是 per-invoke 而非全 session（Stage 1 + Stage 2 per-topic
    invoke 各有自己的 max_iterations 內部循環）。`(stage, topic_id, iteration)`
    三元組才是 audit unique key。本 schema 加 `stage` 欄位明示。
    """

    stage: Literal["stage_1", "stage_2"] = Field(
        default="stage_1",
        description=(
            "I-3: BAB 跑在 Stage 1（global associator）還是 Stage 2（per-topic loop）。"
            "Stage 2 per-topic invoke 各自有 max_iterations 內部循環，"
            "(stage, topic_id, iteration) 才是 audit unique key（避免 iteration overlap）。"
        ),
    )
    iteration: int = Field(..., description="BAB iteration 序號（1-based，per-invoke 而非 session）")
    topic_id: str = Field(default="", description="哪個 BAB topic（Stage 2 per-topic loop）；空 = Stage 1 global")
    drift_level: Literal["none", "minor", "moderate", "major"] = Field(
        ..., description="本輪 BAB 與初版 ContextMap 偏移程度"
    )
    drift_description: str = Field(default="", description="critic 對 drift 的描述")
    recommended_action: Literal["continue", "pause_confirm"] = Field(
        ..., description="continue = 繼續 BAB；pause_confirm = 暫停請 user 確認"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="entry 寫入時間（ISO 8601 string）",
    )
    monitor_degraded: bool = Field(
        default=False,
        description=(
            "O5-A: True 表示本 entry 由 fallback 路徑產生（LLM 呼叫失敗降級）；"
            "False = 正常 LLM 回應。用於事後區分 drift_level=none 是正常 vs 降級。"
        )
    )


# ============================================================================
# ContextMap sub-models
# ============================================================================

class ContextMapTopic(BaseModel):
    """研究結構圖（Context Map）中的單一議題/領域節點。"""

    topic_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="議題名稱，例如：'德國綠能社區共有模式'")
    domain: str = Field(..., description="領域分類，例如：'能源政策'")
    description: str = Field(default="", description="此議題在研究中的角色簡述")
    relevance: Literal["core", "supporting", "peripheral"] = Field(
        default="supporting",
        description="此議題對研究問題的核心程度"
    )
    evidence_ids: List[int] = Field(default_factory=list, description="支持此議題的引用 ID")
    confidence: Literal["high", "medium", "low"] = "medium"


class ContextMapRelation(BaseModel):
    """研究結構圖中兩個議題之間的關係。"""

    relation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_topic_id: str = Field(..., description="來源議題 UUID")
    target_topic_id: str = Field(..., description="目標議題 UUID")
    relation_type: Literal[
        "causes", "enables", "prevents", "contradicts",
        "supports", "part_of", "precedes", "analogous_to"
    ] = Field(..., description="關係類型")
    description: str = Field(default="", description="此關係的說明")
    evidence_ids: List[int] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class ContextMapSearchSeed(BaseModel):
    """從研究結構圖推導出的搜尋計畫種子 — '要找什麼、去哪找、為什麼找'。"""

    seed_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(..., description="具體搜尋 query")
    target_topic_id: str = Field(..., description="此搜尋服務的議題")
    rationale: str = Field(..., description="為何需要此搜尋")
    source_strategy: Literal["internal", "web", "both"] = Field(
        default="both",
        description="搜尋來源策略"
    )
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["pending", "executed", "exhausted"] = "pending"


class ContextMapDelta(BaseModel):
    """單一版本 delta — 記錄從第 N 版到第 N+1 版的變更內容。"""

    from_version: int
    to_version: int
    added_topics: List[str] = Field(default_factory=list, description="新增的 topic_id")
    removed_topics: List[str] = Field(default_factory=list, description="移除的 topic_id")
    modified_topics: List[str] = Field(default_factory=list, description="修改的 topic_id")
    added_relations: List[str] = Field(default_factory=list, description="新增的 relation_id")
    removed_relations: List[str] = Field(default_factory=list, description="移除的 relation_id")
    reason: str = Field(..., description="此次精煉的原因")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# ContextMap (Master B)
# ============================================================================

class ContextMap(BaseModel):
    """
    主控 B — Session 級別的持久性知識結構。

    驅動 B->A->B' 迴圈：Associator 建立初始 B，
    從 B 推導搜尋計畫 A，擷取結果將 B 精煉為 B'。
    """

    research_question: str = Field(..., description="使用者的原始研究問題")
    working_hypothesis: str = Field(default="", description="目前的工作假設結論方向")
    topics: List[ContextMapTopic] = Field(default_factory=list)
    relations: List[ContextMapRelation] = Field(default_factory=list)
    followup_questions: List[str] = Field(default_factory=list, description="潛在的後續問題")
    search_seeds: List[ContextMapSearchSeed] = Field(
        default_factory=list, description="搜尋計畫種子"
    )
    version: int = Field(default=0)
    revision_history: List[ContextMapDelta] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_refined_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Track E (LR DR-parity sprint 2026-05-28) — Temporal BINDING schema
# ============================================================================

class TimeRange(BaseModel):
    """user 對時間範圍的訴求，由 LR Stage 1 intent parser 抽出，
    存進 state.time_constraint。

    Track E（sprint 2026-05-28）新欄位。BAB retrieval 用此作 datePublished
    filter、writer prompt 用此注入 BINDING 區段。

    紀律：
    - start_date / end_date 各自 Optional。None bound 不過濾（user 只說
      「2024 後」→ start="2024-01-01", end=None）。
    - 空字串 normalize 為 None（serializer 友善）。
    - user_selected=True 代表 user 在 Stage 1 dialog 明確選擇 → writer
      BINDING block 強度升級為 STRICT（user_selected=False 系統自動抽 →
      MEDIUM 措辭）。
    """

    start_date: Optional[str] = Field(
        default=None,
        description="ISO 8601 date string (YYYY-MM-DD)；None = 無起始邊界",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="ISO 8601 date string (YYYY-MM-DD)；None = 無結束邊界",
    )
    raw_phrase: str = Field(
        default="",
        description="user 原話片段（如「2024 之後」「最近三年」），writer prompt 顯示給 LLM 看",
    )
    user_selected: bool = Field(
        default=False,
        description=(
            "user 在 dialog 明確選擇 → True；自動抽 → False。"
            "True 時 writer BINDING block 強度升級為 STRICT。"
        ),
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def _validate_iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        from datetime import date
        try:
            date.fromisoformat(v)
        except ValueError as e:
            raise ValueError(
                f"start_date/end_date must be ISO 8601 (YYYY-MM-DD), got {v!r}: {e}"
            )
        return v


# ============================================================================
# Stage 1 Dialog Loop schemas（4/30 新增 — dialog loop 補完）
# ============================================================================

class ContextMapRevisionOperation(BaseModel):
    """單一 ContextMap mutation 指令 — 由 LLM intent parser 產生。

    聲明式（declarative）：每個 operation 是「動詞 + 目標」結構，
    apply 階段用 dispatch table 處理。所有欄位都是 optional，因為
    不同 op_type 用到不同 fields；驗證在 mutation engine 中做。
    """

    op_type: Literal[
        "merge_topics",
        "split_topic",
        "add_topic",
        "remove_topic",
        "rename_topic",
        "change_relevance",
        "change_description",
        "reframe_structure",
    ] = Field(..., description="操作類型")
    # merge_topics
    source_topic_ids: List[str] = Field(
        default_factory=list, description="merge 的來源 topic_ids（>= 2）"
    )
    merged_name: str = Field(default="", description="merge / rename 後的新名稱")
    # split_topic
    split_from_topic_id: str = Field(default="", description="split 的來源 topic_id")
    split_into: List[dict] = Field(
        default_factory=list,
        description="split 結果：[{name: str, description: str, evidence_ids: List[int]}]",
    )
    # add_topic
    new_topic_name: str = Field(default="")
    new_topic_description: str = Field(default="")
    new_topic_relevance: Literal["core", "supporting", "peripheral"] = "core"
    new_topic_evidence_ids: List[int] = Field(default_factory=list)
    # remove_topic / rename_topic / change_relevance / change_description
    target_topic_id: str = Field(default="")
    new_relevance: Literal["core", "supporting", "peripheral"] = "supporting"
    new_description: str = Field(default="")
    new_name: str = Field(default="")
    # reframe_structure (UX-9)
    new_chapters: List[dict] = Field(
        default_factory=list,
        description=(
            "reframe_structure 的新章節清單："
            "[{name: str, description: str (optional), "
            "relevance: 'core'|'supporting'|'peripheral' (default 'core'), "
            "word_target: int (optional, 0=未指定)}]。"
            "Replace All semantics — 套用後 cm.topics 全清空後依此重建。"
            "word_target：user 若在 reframe 句中為該章指定字數（如「前言~500、"
            "國內~2500」）就填入；沒指定填 0 / 省略。"
        ),
    )
    new_research_question: str = Field(
        default="",
        description=(
            "reframe_structure 時，optional 覆寫 ContextMap.research_question。"
            "空字串 = 保留原 research_question。"
        ),
    )
    proposal_markdown: str = Field(
        default="",
        description=(
            "reframe_structure 的 detail-rich confirm proposal markdown（D-6 spec）。"
            "LLM intent parser 對 reframe action 必填；其他 op_type 留空。"
            "Stored in state.pending_reframe_json，confirm round 用於 re-emit checkpoint。"
        ),
    )


class Stage1ParsedIntent(BaseModel):
    """LLM 對 Stage 1 user reply 的解析結果。"""

    action: Literal["confirm", "adjust"] = Field(
        ..., description="confirm = 採用目前結構；adjust = 要修改"
    )
    operations: List[ContextMapRevisionOperation] = Field(
        default_factory=list,
        description="action=adjust 時的 mutation 清單；action=confirm 或無實質訴求時為空",
    )
    summary: str = Field(
        default="",
        description="繁體中文一句話摘要 user 訴求，回填到 ContextMapDelta.reason",
    )
    clarifying_question: str = Field(
        default="",
        description=(
            "當 LLM 無法把 user reply mapping 到任何 op_type、且 reply 也不是純 confirm 時，"
            "在此填一句**繁體中文問句**追問 user 想法（例如「你說『太細』是希望整段刪掉，"
            "還是降為輔助議題？」）。orchestrator 收到此欄位非空 + operations=[] 時走澄清 dialog 分支，"
            "把問句以 narration + checkpoint proposal 形式 emit 給 user。"
            "純 confirm / 明確 mutation ops / 完全 ask_llm 失敗（return None）時留空。"
        ),
    )
    # Stage 1 reframe 時 user 順帶指定的引用格式 / 總字數（與 Stage 4 平行）。
    # 過去 Stage 1 reframe path 只抓 new_chapters，APA / 字數被吃掉（C4 / C8 缺口）。
    citation_style: Optional[Literal[
        "author_year", "numeric", "footnote", "none"
    ]] = Field(
        default=None,
        description=(
            "user 在 reply 中順帶提到的引用格式偏好（與 reframe / 結構訴求同句也要抽）。"
            "「APA」「（作者, 年份）」「哈佛」→ 'author_year'；"
            "「[1]」「數字編號」「IEEE」→ 'numeric'；"
            "「腳註」「footnote」「上標」→ 'footnote'；"
            "「不要引用」→ 'none'；沒提 → null。"
        ),
    )
    total_word_count: Optional[int] = Field(
        default=None,
        description=(
            "user 在 reply 中提到的整份報告總字數 budget（中文字數整數）。"
            "如「總共約 7000 字」→ 7000。沒提 → null。"
            "若 user 只給各章字數沒給總數，可留 null（各章字數放 new_chapters[i].word_target）。"
        ),
    )
    # Track E (sprint 2026-05-28): user 在 Stage 1 reply 順帶提到時間訴求 → 抽出
    time_range_extracted: Optional[TimeRange] = Field(
        default=None,
        description=(
            "若 user 在 Stage 1 reply 提到時間範圍，抽出 TimeRange；"
            "由 orchestrator handler 寫進 state.time_constraint。"
            "user 沒提 → None（既有 confirm / adjust 流程不受影響）。"
        ),
    )


# ============================================================================
# Stage 4 / Stage 5 Intent schemas（4/30 新增 — friendly redirect）
# ============================================================================

class Stage4Action(str, Enum):
    """Stage 4 user reply intent classification action values（改動 6：enum 化）。"""
    auto_continue = "auto_continue"      # user 表達「你決定」「都可以」
    format_spec = "format_spec"          # 純格式偏好
    structure_change = "structure_change"  # 結構性訴求（誤入 stage 4）
    mixed = "mixed"                      # 同時含結構 + 格式


# ============================================================================
# Stage 4 TypeAgent typed sub-models (Plan: lr-typeagent-refactor, 2026-05-19)
# ============================================================================
# CEO 拍板 OQ-2：純 typed schema + few-shot 強化，**不**加 _ELEMENT_KEYWORDS
# validator 兜底。LLM 若 mis-classify chapter vs element → 加更多 few-shot，
# 不靠 keyword heuristic（違反 TypeAgent 紀律）。
# `type` Literal 強制 LLM output schema 對齊，instructor backend 自動 reject + retry。


class ChapterSpec(BaseModel):
    """單一章節 outline spec — 取代 Dict[str, str] 鬆散字典。

    `type` Literal 強制 LLM 對齊「敘事章節」channel，與 SpecialElementSpec 互斥語意
    由 typed schema 保證（不靠 prompt 紀律「請不要 X」）。
    """
    type: Literal["narrative_chapter"] = "narrative_chapter"
    name: str = Field(..., min_length=1, description="章節標題（user 原文，不增刪不重排）")
    description: str = Field(default="", description="該章描述（optional）")
    relevance: Literal["core", "supporting", "peripheral"] = "core"


class SpecialElementSpec(BaseModel):
    """單一 special element spec — 取代 Dict[str, str]。

    `type` Literal enum 強制 LLM output 對齊；非 enum 值 schema-side reject。
    """
    type: Literal["table", "list", "chart", "diagram", "code_block"] = Field(
        ..., description="element 類型"
    )
    target_chapter: str = Field(
        default="",
        description="user 指定章節（與 cm.topics 名稱比對），空字串 = unspecified",
    )
    description: str = Field(default="", description="user 自然語言描述")

    # === transient（classifier 語意判斷輸出，用完即丟、不持久化）===
    # LLM 對「非字面 exact/唯一」的 target 判語意對應到哪一章（OQ-6 靈魂）。
    # dispatch 讀這兩欄位路由（clear→確認 / uncertain→問），持久化前經 serializer strip 掉（B3）。
    resolved_chapter_title: Optional[str] = Field(
        default=None,
        description="LLM 判 target 語意對應的章名（空/None = 判不出）。transient，不持久化。",
    )
    resolution_confidence: Optional[Literal["clear", "uncertain"]] = Field(
        default=None,
        description="LLM 語意判斷信心：clear=明確對應 / uncertain=判不準。transient，不持久化。",
    )


class InitialChapterSpec(BaseModel):
    """初始 query 抽取**專用**章節 submodel（與 Stage 4 ChapterSpec 隔離）。

    AR round 1 B1：既有 ChapterSpec 無 word_target，但下游 outline planner
    (prompts/outline_planner.py:88-93) 與 _extract_chapters_from_ops
    (orchestrator.py:462-465) 真消費逐章字數。故初始抽取章節需要能帶 word_target。
    不擴 ChapterSpec（避免 Stage 4 channel regression），新增此專用 submodel。
    """
    name: str = Field(..., min_length=1, description="章節標題（user 原文，不增刪不重排）")
    description: str = Field(default="", description="該章描述（optional）")
    word_target: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "user 為**這一章**明說的目標字數（「第一章 2000 字」→ 2000）。"
            "沒為該章指定 → null。"
        ),
    )


class InitialFormatSpec(BaseModel):
    """初始 query 格式 spec 抽取結果（Stage 1 進場一次性 LLM 抽取）。

    使用者**初始** prompt 內嵌的格式需求（章節架構 / 各章字數 / 總字數 /
    引用格式 / 特殊元素）抽成結構化欄位。所有欄位 optional / 預設空 —
    抽不到 = 該欄位 null / 空 list = 現行 LLM 自由發揮行為（保守 default 紀律）。

    chapters 用專用 InitialChapterSpec（含 word_target）；special_elements 重用既有
    SpecialElementSpec typed submodel（與 Stage 4 同 channel，下游消費機制完全現成）。
    """

    chapters: List[InitialChapterSpec] = Field(
        default_factory=list,
        description=(
            "user 初始 prompt 指定的章節架構（typed）。name = 章節標題（user 原文，"
            "不增刪不重排）；description / word_target optional。沒指定章節 → 空 list。"
        ),
    )
    total_word_count: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "user 初始 prompt 提到的整份報告總字數（中文字數整數）。"
            "「總共約 7000 字」→ 7000、「七千字左右」→ 7000。沒提 → null。"
        ),
    )
    citation_style: Optional[Literal[
        "author_year", "numeric", "footnote", "none"
    ]] = Field(
        default=None,
        description=(
            "user 初始 prompt 提到的引用格式：「APA」「（作者, 年份）」「哈佛」→ "
            "author_year；「[1]」「數字編號」「IEEE」→ numeric；「腳註」「footnote」→ "
            "footnote；「不要引用」→ none；沒提 → null。"
        ),
    )
    special_elements: List[SpecialElementSpec] = Field(
        default_factory=list,
        description=(
            "user 初始 prompt 明確指定的特殊格式 element（typed）。"
            "type ∈ {table, list, chart, diagram, code_block}；"
            "target_chapter 空字串 = unspecified；無特殊格式訴求 → 空 list。"
        ),
    )

    def has_meaningful_spec(self) -> bool:
        """True 表示 user 確實指定了至少一項格式需求（→ 觸發 checkpoint 確認）。

        全空 → False → proposal 不變、不問、走現行自由發揮。
        """
        return bool(
            self.chapters
            or self.total_word_count is not None
            or self.citation_style is not None
            or self.special_elements
        )


class Stage4Intent(BaseModel):
    """Stage 4 user reply intent classification（LLM 分類結果）。

    TypeAgent 紀律（Plan: lr-typeagent-refactor 2026-05-19）：
    - new_chapters / special_elements 為兩條獨立 typed channel。
    - `type` Literal 強制 LLM 對齊，instructor backend auto-reject + retry。
    - **不**用 keyword heuristic 兜底（OQ-2 CEO 拍板，違反 TypeAgent 紀律）。
    """
    intent: Stage4Action = Field(..., description="主要意圖類型")
    format_spec_extracted: str = Field(
        default="",
        description="當 intent=mixed 時，從 user 訊息抽出的格式偏好部分；其他 intent 留空",
    )
    raw_message: str = Field(default="", description="原始 user 訊息（debug / log 用）")
    special_elements: List[SpecialElementSpec] = Field(
        default_factory=list,
        description=(
            "user 在 Stage 4 reply 明確指定的特殊格式 element 列表（typed）。"
            "type ∈ {table, list, chart, diagram, code_block}；target_chapter 空字串 = unspecified；"
            "description 為 user 自然語言描述。無特殊格式訴求時為空 list。"
        ),
    )
    new_chapters: List[ChapterSpec] = Field(
        default_factory=list,
        description=(
            "intent=structure_change / mixed 時，從 user_message 抽出的章節 outline（typed）。"
            "name = 章節標題（user 原文，不增刪不重排）；description / relevance 為 optional。"
            "special_elements 與 new_chapters 為兩條獨立 typed channel — `type` Literal 強制紀律。"
        ),
    )
    # Plan: lr-user-voice-container-and-4-fixes (Fix B)：user 拍板的引用格式 enum
    citation_style_extracted: Optional[Literal[
        "author_year", "numeric", "footnote", "none"
    ]] = Field(
        default=None,
        description=(
            "從 user_message 抽出的引用格式偏好。"
            "user 講「APA」「（作者, 年份）」「哈佛」→ 'author_year'；"
            "user 講「[1]」「數字編號」「IEEE」→ 'numeric'；"
            "user 講「腳註」「上標」「footnote」→ 'footnote'；"
            "user 明確說「不要引用」「不標來源」→ 'none'；"
            "user 沒提引用 → null。"
        ),
    )


# ============================================================================
# Stage4Response TypeAgent typed action enum dispatcher
# Plan: lr-typeagent-refactor (2026-05-19, CEO 拍板 OQ-1: 完全取代舊
# `_parse_stage_4_intent` — 沒 backward compat tax，沒 production user)
# ============================================================================


class Stage4ConfirmTarget(str, Enum):
    """Stage 4 confirm round target — 區分 user 在 confirm 什麼。"""
    reframe = "reframe"        # confirm 接受 reframe 提案
    format = "format"          # confirm 接受 format dialog
    both = "both"              # 兩個 pending 都 confirm


class Stage4ResponseAction(str, Enum):
    """Stage 4 user reply typed action — TypeAgent dispatcher 嚴格路由。

    10-action enum 取代既有「reply 內容自由解析 + pending_* flag 推斷」混合邏輯。
    """
    confirm_reframe = "confirm_reframe"          # 接受既有 reframe 提案
    confirm_format = "confirm_format"            # 接受 format dialog
    confirm_both = "confirm_both"                # 兩個 pending 都 confirm
    cancel_reframe = "cancel_reframe"            # 拒絕 reframe，回原結構
    adjust_chapters = "adjust_chapters"          # 改 chapter outline（reframe 提案不對）
    adjust_format = "adjust_format"              # 改 format spec
    add_special_element = "add_special_element"  # 純補 table/list/figure
    new_structure_request = "new_structure_request"  # 全新結構訴求（非 adjust）
    auto_continue = "auto_continue"              # 「你決定」
    unclear = "unclear"                          # 模糊，需 clarify


class Stage4StructuralPayload(BaseModel):
    """Stage4Response.structural_content — action ∈ {adjust_chapters, new_structure_request}."""
    new_chapters: List[ChapterSpec] = Field(..., min_length=1)
    summary: str = Field(default="")


class Stage4FormatPayload(BaseModel):
    """Stage4Response.format_content — action ∈ {adjust_format, add_special_element}.

    target_word_count（Blocker A root fix, 2026-05-19）：user 在 Stage 4 reply
    指定的中文總字數（typed int）。LLM 解析「五千字」→ 5000、「七千字左右」→ 7000、
    「三千多字」→ 3000。null = user 沒提字數 → outline planner 回每章 target=0
    （writer 自決長度，不硬塞 default）。dispatcher 寫進 state.user_voice.target_word_count。
    """
    format_spec_extracted: str = Field(default="")
    citation_style_extracted: Optional[Literal[
        "author_year", "numeric", "footnote", "none"
    ]] = None
    special_elements: List[SpecialElementSpec] = Field(default_factory=list)
    target_word_count: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "user 拍板的中文總字數（int >= 1）。「五千字」→ 5000、「七千字左右」→ 7000、"
            "「三千多字」→ 3000。user 沒提 → null。"
            "舊 fixture 沒此欄位 → default None backward compat。"
        ),
    )


class Stage4Response(BaseModel):
    """Stage 4 user reply 經 LLM 分類後的 typed response — dispatcher 用此路由。

    TypeAgent 紀律：dispatcher 依 `action` 嚴格路由，**不**靠「reply 內容自由解析
    + flag 推斷」混合邏輯。互斥語意（confirm_target / structural_content /
    format_content / clarifying_question）由 @model_validator 強制。
    """
    action: Stage4ResponseAction = Field(
        ..., description="主要 action — dispatcher 嚴格依此路由"
    )
    confirm_target: Optional[Stage4ConfirmTarget] = Field(
        default=None,
        description=(
            "action ∈ {confirm_reframe, confirm_format, confirm_both} 時必填。"
            "其他 action 留 None。"
        ),
    )
    structural_content: Optional[Stage4StructuralPayload] = Field(
        default=None,
        description=(
            "action ∈ {adjust_chapters, new_structure_request} 時必填，含 new_chapters。"
        ),
    )
    format_content: Optional[Stage4FormatPayload] = Field(
        default=None,
        description=(
            "action ∈ {adjust_format, add_special_element} 時必填，"
            "含 format_spec_extracted / citation_style / special_elements。"
        ),
    )
    clarifying_question: str = Field(
        default="",
        description=(
            "action='unclear' 時必填繁體中文問句；其他 action 留空。"
            "Blocker C (2026-05-19) root fix：欄位語意上 None == \"\" == 無需澄清，"
            "schema-side coerce None → \"\"（LLM 對非 'unclear' action 偶爾仍"
            "output null clarifying_question，原本 Pydantic str 嚴格 reject 整個"
            "Stage4Response，導致 _save_state 沒跑完）。unclear 契約仍由 "
            "@model_validator 強制非空（coerce 後若仍空字串照樣 reject）。"
        ),
    )

    @field_validator("clarifying_question", mode="before")
    @classmethod
    def _coerce_null_clarifying_question(cls, v):
        """Blocker C：LLM 對非 'unclear' action 偶爾 output null clarifying_question。
        欄位語意上 None == "" == 無需澄清，coerce 為 ""。
        unclear 契約由 @model_validator after 仍會檢查空字串並 reject。"""
        if v is None:
            return ""
        return v

    @model_validator(mode="after")
    def _enforce_action_payload_contract(self) -> "Stage4Response":
        """互斥語意 schema-side enforce — instructor backend reject + retry 不合規 output。"""
        a = self.action
        if a in (
            Stage4ResponseAction.confirm_reframe,
            Stage4ResponseAction.confirm_format,
            Stage4ResponseAction.confirm_both,
        ):
            if self.confirm_target is None:
                raise ValueError(
                    f"action={a.value} 要求 confirm_target 非 None"
                )
        if a in (
            Stage4ResponseAction.adjust_chapters,
            Stage4ResponseAction.new_structure_request,
        ):
            if self.structural_content is None:
                raise ValueError(
                    f"action={a.value} 要求 structural_content 非 None"
                )
        if a == Stage4ResponseAction.adjust_format:
            if self.format_content is None:
                raise ValueError(
                    f"action={a.value} 要求 format_content 非 None"
                )
        if a == Stage4ResponseAction.unclear:
            if not self.clarifying_question.strip():
                raise ValueError("action='unclear' 要求 clarifying_question 非空")
        return self


class ClarificationRequest(BaseModel):
    """通用「不確定就澄清」typed request（設計文件 §3）。

    收斂 LR 三條同型 clarification spine（Stage 1 empty-ops / Stage 4 unclear /
    R2 表格章節指涉）。對齊既有 Stage4Response action='unclear' 的「必填問句 +
    validator 非空」藍本（schemas_live.py:784-786）。
    """
    question: str = Field(..., description="繁體中文澄清問句（可含選項枚舉）")
    stage: int = Field(..., description="拋出點的 stage（決定 re-emit 哪個 checkpoint）")

    @field_validator("question")
    @classmethod
    def _question_nonempty(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("ClarificationRequest.question 不可為空")
        return v


# ============================================================================
# ContextMap utility functions
# ============================================================================

_RELEVANCE_ORDER = {"core": 0, "supporting": 1, "peripheral": 2}
_RELEVANCE_LABELS = {"core": "核心議題", "supporting": "支持議題", "peripheral": "周邊議題"}


def _render_context_map_topics(topics: List["ContextMapTopic"]) -> List[str]:
    """純格式化：把已 filter 的 topic list 依 relevance 分組組成 markdown 行 list。

    呼叫端負責 filter（extract 限縮到 section，summary 用全部）；本 helper 只做
    格式化。回傳 list of lines（含每組結尾的空字串分隔元素），呼叫端用
    `lines.extend(...)` 接回，與抽取前逐行 append 的輸出位元一致。空 list → 回 []
    （呼叫端的 skip 決策仍在呼叫端，本 helper 不吸收）。
    """
    lines: List[str] = []
    for relevance in ["core", "supporting", "peripheral"]:
        group = [t for t in topics if t.relevance == relevance]
        if not group:
            continue
        lines.append(f"### {_RELEVANCE_LABELS[relevance]}")
        for t in group:
            # Plan: lr-user-voice-container-and-4-fixes (Fix D, 2026-05-18 audit)
            # 改 narrative count form — 給 LLM grounding 訊息（「真有依據」），
            # 但不給可直接抄寫的 [1, 2, 3] literal（消除段末「來源: [1] [2]」dump 觸發點）。
            evidence_str = (
                f"，{len(t.evidence_ids)} 個來源支持" if t.evidence_ids else ""
            )
            desc = f": {t.description}" if t.description else ""
            lines.append(
                f"- **{t.name}** ({t.domain}){desc} (confidence: {t.confidence}{evidence_str})"
            )
        lines.append("")
    return lines


def _render_context_map_relations(
    relations: List["ContextMapRelation"],
    topic_name_map: Dict[str, str],
) -> List[str]:
    """純格式化：把已 filter 的 relation list 組成「### 關係」markdown 行 list。

    呼叫端負責 filter 與「relations 為空時是否 skip 整段」的決策；本 helper 只在
    收到非空 list 時格式化（含結尾空字串分隔元素），空 list → 回 []（不吸收 skip
    決策）。呼叫端用 `lines.extend(...)` 接回。
    """
    if not relations:
        return []
    lines: List[str] = ["### 關係"]
    for rel in relations:
        src = topic_name_map.get(rel.source_topic_id, "?")
        tgt = topic_name_map.get(rel.target_topic_id, "?")
        desc = f": {rel.description}" if rel.description else ""
        lines.append(f"- {src} --{rel.relation_type}--> {tgt}{desc}")
    lines.append("")
    return lines


def context_map_extract_for_section(
    context_map: ContextMap,
    section_topic_ids: List[str],
) -> str:
    """
    Extract relevant topics and relations for a specific section from ContextMap.

    Filters to section_topic_ids plus their direct relation neighbors.
    No truncation — ContextMap is small (10-20 topics), no token management needed.
    """
    lines: List[str] = []
    lines.append(f"## 研究結構 (v{context_map.version})")
    lines.append(f"研究問題: {context_map.research_question}")
    if context_map.working_hypothesis:
        lines.append(f"工作假設: {context_map.working_hypothesis}")
    lines.append("")

    # Build neighbor set: any topic connected via a relation to the focus set
    focus_set = set(section_topic_ids)
    neighbor_ids: set[str] = set()
    for rel in context_map.relations:
        if rel.source_topic_id in focus_set:
            neighbor_ids.add(rel.target_topic_id)
        if rel.target_topic_id in focus_set:
            neighbor_ids.add(rel.source_topic_id)

    include_ids = focus_set | neighbor_ids
    topics = [t for t in context_map.topics if t.topic_id in include_ids]
    topics.sort(key=lambda t: _RELEVANCE_ORDER.get(t.relevance, 1))

    # Group by relevance (filter 已在上方完成 — 餵 filtered topics 給共用 render)
    lines.extend(_render_context_map_topics(topics))

    # Relations where both endpoints are in the included set (filter 在 caller)
    topic_name_map = {t.topic_id: t.name for t in context_map.topics}
    include_ids_set = {t.topic_id for t in topics}
    relevant_rels = [
        r for r in context_map.relations
        if r.source_topic_id in include_ids_set and r.target_topic_id in include_ids_set
    ]
    lines.extend(_render_context_map_relations(relevant_rels, topic_name_map))

    # Pending search seeds (up to 5 to keep token budget manageable)
    pending = [s for s in context_map.search_seeds if s.status == "pending"]
    if pending:
        lines.append("### 待查")
        for s in pending[:5]:
            lines.append(f"- [{s.query}]: {s.rationale} (priority: {s.priority})")
        lines.append("")

    return "\n".join(lines)


def context_map_to_summary(context_map: ContextMap) -> str:
    """
    Generate a complete Markdown summary of the full ContextMap.

    Unlike context_map_extract_for_section(), this includes ALL topics and
    relations without filtering. Used for Stage 1 and other scenarios
    that need the full picture.
    """
    lines: List[str] = []
    lines.append(f"## 研究結構 (v{context_map.version})")
    lines.append(f"研究問題: {context_map.research_question}")
    if context_map.working_hypothesis:
        lines.append(f"工作假設: {context_map.working_hypothesis}")
    lines.append("")

    # All topics grouped by relevance (summary 不 filter — 全部餵給共用 render)
    topics = sorted(context_map.topics, key=lambda t: _RELEVANCE_ORDER.get(t.relevance, 1))
    lines.extend(_render_context_map_topics(topics))

    # All relations (summary 不 filter — 全部 relations 餵給共用 render)
    topic_name_map = {t.topic_id: t.name for t in context_map.topics}
    lines.extend(_render_context_map_relations(context_map.relations, topic_name_map))

    # Pending search seeds
    pending = [s for s in context_map.search_seeds if s.status == "pending"]
    if pending:
        lines.append("### 待查")
        for s in pending:
            lines.append(f"- [{s.query}]: {s.rationale} (priority: {s.priority})")
        lines.append("")

    # Followup questions
    if context_map.followup_questions:
        lines.append("### 後續問題")
        for q in context_map.followup_questions:
            lines.append(f"- {q}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Associator output schemas
# ============================================================================

class AssociatorBuildOutput(BaseModel):
    """AssociatorAgent.build_context_map() 的輸出 — 建立初始 B。"""

    context_map: ContextMap
    narration: str = Field(
        ...,
        description="給使用者看的自然語言說明：說明研究方向的布局與原因。禁止使用任何欄位名稱（topics、relations、confidence、v0 等），只用自然語言。使用繁體中文。"
    )


class AssociatorDeriveOutput(BaseModel):
    """AssociatorAgent.derive_search_plan() 的輸出 — 從 B 推導 A。"""

    search_seeds: List[ContextMapSearchSeed] = Field(
        ..., description="推導出的搜尋計畫"
    )
    narration: str = Field(
        ...,
        description="給使用者看的自然語言說明：說明接下來要找什麼資料以及為什麼這樣優先排序。禁止使用任何欄位名稱（search_seeds、target_topic_id、confidence 等），只用自然語言。使用繁體中文。"
    )


class AssociatorRefineOutput(BaseModel):
    """AssociatorAgent.refine_context_map() 的輸出 — 將 B 更新為 B'。"""

    updated_context_map: ContextMap
    delta: ContextMapDelta
    is_stable: bool = Field(
        ...,
        description="True 表示結構已足夠穩定，可以退出迴圈"
    )
    narration: str = Field(
        ...,
        description="給使用者看的自然語言說明：說明這輪搜尋後發現了什麼、調整了什麼方向，以及研究是否需要繼續。禁止使用任何欄位名稱（is_stable、delta、topics、relations、confidence、v0/v1 等），只用自然語言。使用繁體中文。"
    )


# ============================================================================
# Consistency Monitor schema (Critic extension)
# ============================================================================

class ConsistencyReview(BaseModel):
    """Critic 一致性檢查的輸出 — 偵測主控 B 漂移。"""

    drift_level: Literal["none", "minor", "moderate", "major"] = Field(
        ...,
        description="目前 B 相對於初始 B 的漂移程度"
    )
    drift_description: str = Field(
        ...,
        description="具體說明漂移了什麼以及為何重要"
    )
    dubao_voice_message: str = Field(
        ...,
        description="用讀豹語氣的自然語言敘述，用於對話中（繁體中文）"
    )
    recommended_action: Literal["continue", "pause_confirm", "refine_master_b", "abort"] = Field(
        ...,
        description="Orchestrator 應採取的行動"
    )
    affected_topics: List[str] = Field(
        default_factory=list,
        description="受漂移影響的 topic_id"
    )
    monitor_degraded: bool = Field(
        default=False,
        description="True 表示一致性監控 LLM 呼叫失敗已降級；"
                    "消費端據此 emit 降級旁白（非 LLM 自評欄位）"
    )


# ============================================================================
# Style Analysis schemas (Stage 3)
# ============================================================================

class StyleFeature(BaseModel):
    """單一提取出的文筆特徵。"""

    dimension: str = Field(..., description="例如：'句式結構'、'用詞層次'、'段落節奏'")
    observation: str = Field(..., description="在範本中觀察到的內容")
    instruction: str = Field(..., description="給 Writer 遵循的具體指令")


class StyleAnalysisOutput(BaseModel):
    """對使用者提供的寫作範本進行風格分析的輸出。"""

    features: List[StyleFeature] = Field(
        ..., min_length=1, max_length=10,
        description="提取出的文筆特徵（sparse 範本 1 個亦合法；prod blocker fix 2026-05-30，min_length 3→1）"
    )
    overall_tone: str = Field(..., description="整體語氣摘要，例如：'學術嚴謹但不枯燥'")
    sample_quality_note: str = Field(
        default="",
        description="關於範本品質或限制的備註"
    )
    citation_format: Literal["author_year", "numeric", "footnote", "none"] = Field(
        default="numeric",
        description=(
            "引用格式偏好（從 user 描述/範本中萃取，分類為以下離散值之一）："
            "'author_year' = APA 風格（作者, 年份）；"
            "'numeric' = 數字編號 [N]；"
            "'footnote' = 腳註編號 ¹²³；"
            "'none' = 不需要引用標記。"
            "說明：此欄位設計為 enum 而非自由文字，避免 Writer 把樣式描述當字面字串輸出。"
        )
    )
    input_is_writing_sample: bool = Field(
        default=True,
        description=(
            "LLM 對輸入本質的判斷：True = 這是一段寫作範本（可正常抽特徵）；"
            "False = 這其實是調整指令 / meta 指令 / 閒聊，不是可供分析的寫作範本。"
            "預設 True（安全方向：絕大多數輸入是範本）；只有 LLM 明確判定非範本時才回 False。"
            "orchestrator 讀此訊號決定是否降級（不可 silent fail）。"
        )
    )


class StyleInputNotASampleError(Exception):
    """Style Analysis 偵測到輸入不是寫作範本（input_is_writing_sample=False）。

    由 orchestrator._run_style_analysis raise，呼叫端負責優雅降級成 user-facing
    narration（不可 silent fail），並保留既有 style_features_json 不被覆蓋。
    注意：與 _run_style_analysis 回 None（LLM 系統失敗，S2-2 soft-fail）是
    兩個獨立通道——sentinel = 語意判定非範本；None = LLM 掛了。
    """


# ============================================================================
# Live Writer Section output (Stage 5 per-section writing)
# ============================================================================

class CitationInline(BaseModel):
    """單一 inline citation — Writer LLM output structured data。

    TypeAgent 紀律（Plan: lr-typeagent-refactor 2026-05-19 Target 3，CEO 拍板 OQ-5
    立刻 strict）：Writer LLM 只 output evidence_id + 在 section_content 用
    `{cite:N}` placeholder 標記；orchestrator post-process 從 EvidencePoolEntry
    metadata lookup 取 author/year，依 user_voice.citation_style render 為對應字串。
    """
    evidence_id: int = Field(
        ..., description="必須 ∈ analyst_citations 白名單；hallucination guard 兜底"
    )


class LiveWriterSectionOutput(BaseModel):
    """Writer 在 Live Research 模式下撰寫單一章節的輸出。"""

    section_title: str
    section_content: str = Field(..., description="此章節的 Markdown 內容（含 {cite:N} placeholder）")
    sources_used: List[int] = Field(default_factory=list)
    confidence_level: Literal["High", "Medium", "Low"] = "Medium"
    narration: str = Field(
        default="",
        description="Writer 敘述：撰寫此章節時做了哪些決策"
    )
    methodology_note: Optional[str] = Field(
        default=None,
        description=(
            "撰寫方法論註記。Hallucination Guard 自動修正時會在此欄位 append "
            "「[自動修正：...]」紀錄，給 user transparent。Writer LLM 通常留 None。"
        )
    )
    chapter_summary: str = Field(
        default="",
        description=(
            "本章 50 字摘要（Plan 4 Phase 3：供下章 writer prompt 注入 "
            "previous_chapter_summary）。Writer LLM 在 compose_section 同時 output，"
            "避免另呼一次 LLM。舊 session restore 時無此欄位 → 空字串（writer 第一段"
            "或舊 written_sections 取 .get() 自然處理）。"
        ),
    )
    citations: List[CitationInline] = Field(
        default_factory=list,
        description=(
            "本章 inline citations structured data（TypeAgent Target 3，2026-05-19）。"
            "Writer LLM output 此 list + 在 section_content 用 {cite:N} placeholder "
            "標記事實後的引用位置。orchestrator post-process 依 user_voice.citation_style "
            "取 EvidencePoolEntry author/year 替換 placeholder 為對應字串。"
            "舊 fixture / 過渡期 LLM 沒 output → default empty（backward compat）。"
        ),
    )
    # addendum C-2 (Track A sprint 2026-05-28): section status enum
    # drafted = writer 正常寫完；
    # guard_failed = entity guard rewrite 仍 fail → content 替換為 blocked 文字；
    # blocked_no_evidence = _write_section 入口空 evidence scope → 不呼叫 writer LLM；
    # accepted = 通過 Critic / user accept；
    # critic_rejected = Track F F1 per-section critic publish gate REJECT (sprint 2026-05-28)
    #                   content 已替換為 blocked 文字以避免 claim-level fabrication 傳播。
    status: Literal[
        "drafted", "guard_failed", "blocked_no_evidence", "accepted", "critic_rejected"
    ] = Field(
        default="drafted",
        description=(
            "section 狀態（addendum C-2 / Track A 2026-05-28 + Track F sprint 2026-05-28）。"
            "drafted = writer 正常寫完；"
            "blocked_no_evidence = _write_section 入口空 evidence scope；"
            "guard_failed = entity guard rewrite 仍 fail；"
            "accepted = 通過 Critic / user accept；"
            "critic_rejected = Track F F1 per-section critic publish gate REJECT "
            "(content 已替換 blocked 文字以避免 claim-level fabrication 傳播)。"
        ),
    )

    def validate_sources_against_plan(
        self, planned_evidence_ids: List[int],
        allowed_evidence_ids: Optional[List[int]] = None,   # P2 W8：全 pool 容許集
    ) -> None:
        """addendum C-4 invariant check：sources_used ⊆ allowed ∪ {0}（0 = no-citation marker）。

        P2 W8（§0 #16）：全局 evidence 模型下合法引用集 = 全 evidence pool（非只該章
        planned）。`allowed_evidence_ids` 提供時用全 pool；未提供 → 向後相容退回
        planned ∪ {0}（避免未來把此 invariant wire 進 prod 時變成隱性白名單擋合法 citation）。

        codex Imp-1 雙模式紀律（CEO 2026-05-28 拍板）：
        - Runtime（prod / dev）：違反不 raise（避免阻塞 user pipeline），用
          logger.warning 告警 + 由 caller 將該 section 標 `status="guard_failed"`
          + Stage 6 narration 用 hallucination_corrected flag 推送
        - Test / CI fixture：偵測 `LR_STRICT_INVARIANTS=1` env var → raise
          `InvariantViolation`（test 故意傳 invalid sources_used → 必須 fail-loud）
        """
        base = allowed_evidence_ids if allowed_evidence_ids is not None else planned_evidence_ids
        allowed = set(base) | {0}
        violations = [s for s in self.sources_used if s not in allowed]
        if not violations:
            return
        msg = (
            f"[SCHEMA INVARIANT] sources_used violation in section "
            f"{self.section_title!r}: ids {violations} not in planned "
            f"(planned={planned_evidence_ids})"
        )
        if os.environ.get("LR_STRICT_INVARIANTS"):
            # test / CI 模式：fail-loud
            raise InvariantViolation(msg)
        # runtime 模式：log warning + caller 自行 mark guard_failed
        logger.warning(msg)


# ============================================================================
# Plan 4: Two-stage writer — Outline Planner schemas
# ============================================================================

class ChapterPlan(BaseModel):
    """Plan 4 Phase 1: 單章 writer 計劃 — 由 outline planner LLM 規劃。

    Outline planner 把整書章節 (來自 Plan 2 _resolve_chapter_source) 規劃成
    LLM-assisted 結構：每章 brief / target_word_count / planned_evidence_ids /
    transition_hint / role。Writer prompt 在 Phase 3 注入「全書 outline」+
    「前章摘要」block，避免章節間銜接弱、重複、結論章不知道怎麼收。
    """

    chapter_index: int = Field(..., description="0-based 章節索引")
    title: str = Field(..., description="章節標題（對齊 format_specs.chapters[i].name）")
    brief: str = Field(..., description="50-100 字章節 brief（這章要講什麼、定位）")
    target_word_count: int = Field(default=0, description="預定字數，0 表示未指定")
    planned_evidence_ids: List[int] = Field(
        default_factory=list,
        description=(
            "預定本章引用哪些 evidence_ids（LLM-assisted allocation）。"
            "升級 Plan 2 Phase 3 的 union-to-first 簡化策略 — 那個降為 skeleton fallback only。"
        ),
    )
    transition_hint: str = Field(
        default="",
        description="50 字「如何承接上章、引出下章」hint。第一章可為空。",
    )
    role: Literal["intro", "body", "conclusion"] = Field(
        ..., description="章節角色：intro 鋪陳、body 深入、conclusion 收尾"
    )

    # Track A (sprint 2026-05-28) addendum C-4: planned_evidence_ids ⊆ evidence_pool.keys()
    # 透過 ValidationInfo context (caller 在 model_validate 時傳
    # context={"evidence_pool_keys": set(pool.keys())}). 沒提供 context → skip
    # (test / dry-run / direct construction 容錯).
    @model_validator(mode="after")
    def _validate_planned_evidence_subset(self, info: ValidationInfo) -> "ChapterPlan":
        keys = (info.context or {}).get("evidence_pool_keys") if info.context else None
        if keys is None:
            return self
        invalid = [eid for eid in self.planned_evidence_ids if eid not in keys]
        if invalid:
            raise ValueError(
                f"planned_evidence_ids contains ids not in evidence_pool: {invalid} "
                f"(chapter='{self.title}', allowed_keys_count={len(keys)})"
            )
        return self

    # Gemini Critical C-2 紅隊 #2 (sprint 2026-05-28): role + chapter_index 雙重一致
    # 防 LLM 把 body 章節標 intro 繞 Task 3 deterministic gate.
    # 注意: total_chapters 在此處不可知; BookOutline._validate_chapters_role_index_consistency
    # 在 outline 層為每個 chapter 再驗一次帶 total 的 case.
    # 此處只能檢查不需 total 的 case: role==intro 必 index==0; role==body 不可
    # index==0 (除非 single-chapter book, 但 single-chapter 無法在此判斷, 留給 outline 層).
    @model_validator(mode="after")
    def _validate_role_position(self) -> "ChapterPlan":
        if self.role == "intro" and self.chapter_index != 0:
            raise ValueError(
                f"ChapterPlan role/index inconsistency: role='intro' but "
                f"chapter_index={self.chapter_index} (intro must be chapter_index==0)"
            )
        # conclusion 與 body 需要 total_chapters 才能驗; 留 BookOutline level
        return self


class BookOutline(BaseModel):
    """Plan 4 Phase 1: 全書 outline — outline planner LLM 一次性產出。

    Stage 5 開頭 outline planner 跑一次，存進 state.book_outline_json。
    Section writer loop 讀此 outline 知道整本書結構 (current chapter 是第 N / 共 M)，
    並引用 ChapterPlan.planned_evidence_ids 做 per-chapter evidence allocation。
    """

    chapters: List[ChapterPlan] = Field(..., min_length=1)
    overall_arc: str = Field(default="", description="100 字全書論述軌跡 (intro → body → conclusion)")
    redundancy_warnings: List[str] = Field(
        default_factory=list,
        description="outline planner 發現的重疊風險警告（'第2、第3章都會碰到X，請writer注意分工'）",
    )

    # Gemini Critical C-2 紅隊 #2 (sprint 2026-05-28): BookOutline-level
    # role + index 一致 (含 conclusion + last-position + body+index 0 single-chapter
    # 邊緣 case)。schema 層在 test/CI 必 raise; runtime double-check 在 Task 3
    # _is_intro_or_conclusion 用 log warning + 視 body 走 gate (寬容策略避免 prod crash)。
    @model_validator(mode="after")
    def _validate_chapters_role_index_consistency(self) -> "BookOutline":
        n = len(self.chapters)
        for ch in self.chapters:
            if ch.role == "intro" and ch.chapter_index != 0:
                raise ValueError(
                    f"BookOutline role inconsistency: chapter {ch.title!r} role='intro' "
                    f"but chapter_index={ch.chapter_index}"
                )
            if ch.role == "conclusion" and ch.chapter_index != n - 1:
                raise ValueError(
                    f"BookOutline role inconsistency: chapter {ch.title!r} "
                    f"role='conclusion' but chapter_index={ch.chapter_index} "
                    f"!= last (total={n})"
                )
            if ch.role == "body" and ch.chapter_index == 0 and n > 1:
                raise ValueError(
                    f"BookOutline role inconsistency: chapter {ch.title!r} role='body' "
                    f"but chapter_index==0 (intro position; total={n})"
                )
        return self


# ============================================================================
# Task 4: AnalystResearchOutputLive (Analyst output with optional narration)
# ============================================================================

# ============================================================================
# Task 5: CriticReviewOutputLive (Critic output with narration_transition)
# ============================================================================

# Import base class (schemas_enhanced does not import schemas_live, so no circular dep)
from reasoning.schemas_enhanced import CriticReviewOutputEnhancedCoV as _CriticCoV


class CriticReviewOutputLive(_CriticCoV):
    """
    Critic 輸出，附加 Live Research 敘述轉折欄位。

    繼承自 CriticReviewOutputEnhancedCoV（包含 cov_verification 等），
    額外增加 narration_transition 欄位供 Live Research 模式使用。

    narration_transition 由 Critic 在發現問題時主動填寫，
    空字串表示不需要轉折（Analyst 表現正常）。

    用於 enable_live_research=True 時的 schema 選擇。
    """

    narration_transition: str = Field(
        default="",
        description="讀豹語氣的敘述轉折訊息（空字串 = 不需要轉折，Analyst 表現正常）"
    )


# ============================================================================
# Task 4: AnalystResearchOutputLive (Analyst output with optional narration)
# ============================================================================

# Import base class (schemas_enhanced does not import schemas_live, so no circular dep)
from reasoning.schemas_enhanced import AnalystResearchOutputEnhancedKG as _AnalystKG


class AnalystResearchOutputLive(_AnalystKG):
    """
    Analyst 輸出，附加 Live Research narration 欄位。

    繼承自 AnalystResearchOutputEnhancedKG（包含 argument_graph、knowledge_graph、
    gap_resolutions 等），額外增加選填的 narration 欄位。

    用於 enable_live_research=True 時的 schema 選擇。
    Narration 由 orchestrator SSE 處理（CEO Review 2026-04-13），
    但此欄位保留供 Analyst 選填使用。
    """

    narration: Optional[str] = Field(
        default=None,
        description="Live Research 敘述：Analyst 做了什麼以及為什麼（繁體中文）"
    )


# ============================================================================
# Evidence Pool — References master list 持久化（LR 報告底部參考文獻來源）
# ============================================================================

class EvidencePoolEntry(BaseModel):
    """單一 evidence 來源 — references master list 反查表的 entry。

    跨 BAB iteration 累積的全局 evidence pool，Stage 6 匯出報告時組合
    references master list（「## 參考文獻」段落）。Writer prompt 也透過
    evidence_lookup 子集看到真實 URL/title，避免 phantom citation。
    """

    evidence_id: int = Field(..., description="全局唯一 evidence ID（跨 BAB iteration 累積）")
    title: str = Field(default="", description="文章標題（從 retrieval item 取）")
    url: str = Field(default="", description="文章 URL")
    source_domain: str = Field(default="", description="來源網域，例如：'reuters.com'、'cna.com.tw'")
    snippet: str = Field(default="", description="文章摘要前 300 字（給 Writer prompt 看）")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="這筆 evidence 被收錄的時間（ISO 8601）"
    )
    iteration_origin: int = Field(
        default=0,
        description="這筆 evidence 是在第幾輪 BAB iteration 抓到的（debug 用）"
    )
    # TypeAgent Target 3 (2026-05-19): typed citation metadata
    # 由 retrieval pipeline / indexing 填入。空字串 = metadata 缺，render 階段
    # fallback 為 source_domain / 'n.d.' 並在 methodology_note 明示。
    author: str = Field(
        default="",
        description="文章 author（retrieval metadata；空字串 = 缺，render fallback 為 source_domain）",
    )
    year: str = Field(
        default="",
        description="文章年份字串（retrieval metadata；空字串 = 缺，render fallback 為 'n.d.'）",
    )
    # Track E (sprint 2026-05-28, E-AMB-3 LOCKED Option A): evidence article 發布時間
    # 由 loop_engine._execute_search 在入庫時填入（從 retrieval item.schema_json 抽
    # datePublished 前 10 字元 YYYY-MM-DD），state.time_constraint filter 用此欄位
    # 過濾範圍外 evidence。空 / None = retrieval 無此 metadata → fallback 不過濾。
    # 邊界紀律：新增 Optional 欄位 ≠ 修改既有欄位，不視為動 Track A frozen schema 結構。
    published_at: Optional[str] = Field(
        default=None,
        description=(
            "ISO 8601 date string（YYYY-MM-DD）；"
            "None = retrieval 無此 metadata（落 fallback 不過濾）"
        ),
    )
    # Track C (sprint 2026-05-28): evidence 來源分類，writer prompt + references render 用。
    # default="internal" 保 backward-compat：所有 Track A 既有 evidence_pool 條目 load 後自動標 internal。
    # 邊界紀律：新增帶 default 的 Literal 欄位 ≠ 修改既有欄位，不視為動 Track A frozen schema 結構
    # （沿 Track A addendum C-4 lemma / Track E E-AMB-3 拍板先例）。
    source: Literal["internal", "web", "wiki", "llm_knowledge"] = Field(
        default="internal",
        description=(
            "evidence 來源類型："
            "'internal' = 站內 corpus retrieval；"
            "'web' = Google CSE web search；"
            "'wiki' = Wikipedia API；"
            "'llm_knowledge' = Analyst gap_resolutions LLM_KNOWLEDGE virtual doc"
        ),
    )
    # P2 全局 evidence 模型（W1，§0 #13）：evidence→建議章節正向 N:M 軟標註。
    # 由 OutlinePlanner 反轉現有 per-chapter allocation 回填（orchestrator outline stage）。
    # default_factory=list 保 backward-compat（舊 session evidence_pool_json 無此 key → 空 list）。
    suggested_chapters: List[int] = Field(
        default_factory=list,
        description=(
            "建議此 evidence 適用的章節 index（0-based，可多章 = N:M 正向標記）。"
            "空 list = 無特定建議（全章皆可，當低優先 tier）。"
            "**軟性提示**：writer 讀全 pool，此欄只影響排序優先序，不是白名單。"
        ),
    )


class GeneratedTitle(BaseModel):
    """外部來源（web/wiki）無標題時，low-tier LLM 從 snippet 生成的標題。

    Google CSE API 無 title 時填字串 "No Title"；空標題 / "No Title" 進
    evidence_pool 前用 low-tier LLM 從 snippet 生成簡潔中文標題（loop_engine
    `_add_external_evidence`）。失敗 / 空回應由 caller 降級為 source_domain。
    """

    title: str = Field(
        ...,
        description="簡潔的繁體中文標題（≤30 字），概括 snippet 主題；禁用「相關報導」「新聞」等泛化詞。",
    )


class GeneratedReportTitle(BaseModel):
    """LR 最終報告 H1 標題：Stage 6 組 H1 前，low-tier LLM 依 research_question
    + 各章標題/摘要生成的有質感報告標題（取代直用 user 原始 query 當 H1）。

    與 GeneratedTitle 的區別：GeneratedTitle 是「外部來源缺標題時補來源標題」
    （web/wiki backfill）；本 schema 是「整份報告的 H1 標題」。兩者勿混用。
    失敗 / 空回應由 caller（orchestrator._generate_report_title）降級為 research_question。
    """

    title: str = Field(
        ...,
        description=(
            "整份研究報告的繁體中文標題（≤40 字），需具體、有資訊量、概括報告主旨；"
            "禁用「研究報告」「分析」「探討」等泛化空詞當開頭套語，直接點出主題與核心張力。"
        ),
    )


GROUNDING_VIEW_CHAR_BUDGET = 12000   # R2：evidence view 字元硬上限（可調常數）
GROUNDING_VIEW_MAX_PRIOR = 40        # R7：prior grounded entity 上限


def _render_claim_line(gc: Dict, prefix: str) -> str:
    """渲染單行 claim bullet（全形括號），供 narrative / evidence view 共用。

    兩 caller 唯一差異是 prefix「推論」有無：
    - render_grounded_narrative 傳 prefix="推論" → `- 推論（rtype，conf）：claim`
    - render_grounding_evidence_view 傳 prefix=""  → `- （rtype，conf）：claim`

    WARN tag 規則共用：critic_status=="WARN" → dash 後、prefix 前內嵌
    `[confidence: low | critic_status: WARN] `。括號固定全形（兩 caller 一致）。
    REJECT 過濾 / empty-skip / title-snippet 處理留各 caller，不在此 helper。
    """
    rtype = gc.get("reasoning_type", "")
    conf = gc.get("confidence", "")
    claim = gc.get("claim", "")
    tag = ""
    if gc.get("critic_status", "PASS") == "WARN":
        tag = "[confidence: low | critic_status: WARN] "
    return f"- {tag}{prefix}（{rtype}，{conf}）：{claim}"


def render_grounded_narrative(
    chapter_eids: List[int],
    evidence_usage: Dict[int, List[Dict]],
    evidence_pool: Dict[int, "EvidencePoolEntry"],
    priority_eids: Optional[List[int]] = None,           # P2 W4：優先渲染（本章 planned/suggested）
    char_budget: int = GROUNDING_VIEW_CHAR_BUDGET,        # P2 W4：防全 pool 爆窗（對齊 12000）
) -> str:
    """把該章 evidence_ids 對應的 GroundedClaim 渲染為 writer 可讀的 markdown findings
    (Track A Task 3, sprint 2026-05-28)。

    Gemini Critical 拍板 (2026-05-28): REJECT 在 source (Task 6) 入庫保留 forensic
    trail，在 render (此處) 過濾不入 writer prompt。雙層職責清楚：source = data
    (保留全部含 REJECT)，render = presentation (過濾 REJECT、WARN 標 low confidence)。
    三個月後 oncall 可從 DB evidence_usage 直接看 critic_status 分佈，區分
    Analyst / 檢索 / Critic 三類問題。

    紀律：
    - evidence_pool 沒對應 entry (或 title/snippet 都空) → 跳過該 eid
    - chapter_eids 無對應 evidence_usage 條目 → 跳過 (writer 端走「資料不足」路徑)
    - REJECT claim 一律過濾不渲染 (critic_status="REJECT")
    - WARN claim 渲染但在每筆 narrative 行首明標
      `[confidence: low | critic_status: WARN]` (Gemini Imp-1, 搭配 Task 4 writer
      prompt 紀律降語氣)
    - PASS claim 正常渲染
    - 該 eid 所有 claim 都被 filter (如整批 REJECT) → 視同空，整個 eid block 跳過
    - 全部跳過 → 回空字串 (caller 用空字串判斷「該章無 grounded 證據」)

    Output 範例:
        ### [1] T1（snippet1）
        - 推論（induction，high）：claim-A
        ### [2] T2（snippet2）
        - [confidence: low | critic_status: WARN] 推論（deduction，low）：claim-B
    """
    # P2 W4：排序 — priority_eids（依傳入序）優先 → 其餘 eid 升冪。
    # priority_eids=None → 退回現況（純 chapter_eids 升冪，無 budget 截斷即傳大值）。
    _unique = list(dict.fromkeys(chapter_eids))  # 去重保序（不直接 sort，先取 unique）
    if priority_eids:
        _prio = [e for e in priority_eids if e in _unique]
        _rest = sorted(e for e in _unique if e not in set(_prio))
        ordered_eids = _prio + _rest
    else:
        ordered_eids = sorted(set(_unique))

    parts: List[str] = []
    used_chars = 0
    truncated = False
    for eid in ordered_eids:
        entry = evidence_pool.get(eid)
        if entry is None:
            continue
        title = getattr(entry, "title", "") or ""
        snippet = (getattr(entry, "snippet", "") or "")[:200]
        if not (title or snippet):
            continue
        raw_claims = evidence_usage.get(eid) or []
        if not raw_claims:
            continue
        # Gemini C-1: filter REJECT claims at render layer (source 已入庫保留 forensic)
        renderable = [
            c for c in raw_claims
            if c.get("critic_status", "PASS") != "REJECT"
        ]
        if not renderable:
            # 整批 claim 都是 REJECT → 該 eid block 跳過 (writer 看不到)
            continue
        # P2 W4：組這一 eid 的 block，先量字數再決定是否超 budget（不 silent 截斷）。
        block_lines = [f"### [{eid}] {title}（{snippet}）"]
        for gc in renderable:
            # Gemini Imp-1: WARN 在 narrative 行首明標, 搭配 Task 4 writer prompt 紀律降語氣
            # narrative 傳 prefix="推論"（含推論前綴）；WARN tag 規則共用於 helper。
            block_lines.append(_render_claim_line(gc, prefix="推論"))
        block = "\n".join(block_lines)
        # P2 W4：char_budget 防全 pool 爆窗。priority 已先進，被截的是相關度最低的。
        if used_chars + len(block) > char_budget and parts:
            truncated = True
            break
        parts.append(block)
        used_chars += len(block) + 1  # +1 為 join 的換行
    if truncated:
        # 不可 silent fail：明示截斷（比照 render_grounding_evidence_view）。
        parts.append(f"[narrative 已達 budget {char_budget} 字元上限，其餘 evidence 見 grounding 視圖]")
    return "\n".join(parts)


def render_grounding_evidence_view(
    chapter_eids: List[int],
    evidence_usage: Dict[int, List[Dict]],
    evidence_pool: Dict[int, "EvidencePoolEntry"],
    prior_grounded_entities: List[str],
    analyst_citations: Optional[List[int]] = None,
    char_budget: int = GROUNDING_VIEW_CHAR_BUDGET,
    current_chapter_index: Optional[int] = None,   # P2 W5：suggested_chapters 含本章 → 升 tier
) -> str:
    """grounding 判讀專用的「良好資料來源」視圖（CEO 方向：給 LLM 完整 context 判 grounding）。

    與 render_grounded_narrative 的差異：
    - **不截斷個別 snippet**（grounding 判讀要看到完整 evidence 文字，才判得出同義改寫）。
    - **R2 整體 context budget cap**：全 pool 內按優先序裝到 `char_budget` 上限，超出截斷／
      捨棄（**不是無限制全 pool**）。優先序：①本章 analyst_citations 對應 evidence →
      ②有實際 claim 的 evidence → ③prior grounded entities lexical-overlap 的 evidence →
      ④其餘 pool evidence。
    - 末尾附「前章已 grounded 的 entity 清單」（R7 限量：lexical overlap + 上限 N）。
    - 與 render_grounded_narrative 同口徑 filter REJECT claim（source 已入庫保留 forensic）。

    Returns:
        str: writer/checker 可讀的 evidence 視圖 markdown（budget 內）。全空回 ""。
    """
    cited = set(analyst_citations or [])
    all_eids = set(chapter_eids) | set(evidence_pool.keys())

    # P2 W5：suggested_chapters 含 current_chapter_index 的 eid 升 tier ①b（cited 之後、有 claim 之前）。
    # current_chapter_index=None → suggested_here 空集 → 行為與現況完全一致（Critic 呼叫端不傳）。
    suggested_here = set()
    if current_chapter_index is not None:
        suggested_here = {
            eid for eid, e in evidence_pool.items()
            if current_chapter_index in (getattr(e, "suggested_chapters", []) or [])
        }

    def _renderable_claims(eid: int) -> List[Dict]:
        return [
            c for c in (evidence_usage.get(eid) or [])
            if c.get("critic_status", "PASS") != "REJECT"
        ]

    # R2 優先序分桶（tier 1=最高）：①本章 citation → ①b suggested_here → ②有 claim
    # → ③prior-overlap → ④其餘
    prior_terms = [e for e in (prior_grounded_entities or []) if e]

    def _priority(eid: int) -> int:
        if eid in cited:
            return 1
        if eid in suggested_here:   # P2 W5：tier ①b（建議本章）
            return 2
        if _renderable_claims(eid):
            return 3
        entry = evidence_pool.get(eid)
        text = (getattr(entry, "title", "") or "") + (getattr(entry, "snippet", "") or "")
        if any(t and t in text for t in prior_terms):
            return 4
        return 5

    ordered = sorted(all_eids, key=lambda e: (_priority(e), e))

    parts: List[str] = []
    used = 0
    for eid in ordered:
        entry = evidence_pool.get(eid)
        if entry is None:
            continue
        title = getattr(entry, "title", "") or ""
        snippet = getattr(entry, "snippet", "") or ""  # 個別不截斷
        if not (title or snippet):
            continue
        # 發布日期錨定（core.temporal_anchor）：writer / critic 兩邊都吃這個視圖，
        # 原本完全不帶發布日期 → 來源內文的「今年」會被當成「今天那年」。
        # 年份由 code 換算後就地標註，並在標頭補發布日期。
        published_at = getattr(entry, "published_at", None)
        snippet = annotate_relative_years(snippet, published_at)
        _date_part = f"（{published_at}）" if published_at else ""
        block_lines = [f"### [{eid}] {title}{_date_part}{anchor_note(published_at)}"]
        if snippet:
            block_lines.append(snippet)
        for gc in _renderable_claims(eid):
            # evidence view 傳 prefix=""（無「推論」前綴）；WARN tag 規則共用於 helper。
            block_lines.append(_render_claim_line(gc, prefix=""))
        block = "\n".join(block_lines)
        # R2：超 budget 即停（已按優先序排序，後面是低優先 evidence，捨棄）
        if used + len(block) + 1 > char_budget:
            parts.append(
                f"\n[grounding view 已達 context budget {char_budget} 字元上限，"
                f"低優先 evidence 已省略]"
            )
            break
        parts.append(block)
        used += len(block) + 1
    body = "\n".join(parts)

    # R7：prior entities 限量（lexical overlap 本章 evidence + 上限 N，取最近）
    body_text = body
    overlap_prior = [e for e in prior_terms if e and e in body_text]
    rest_prior = [e for e in prior_terms if e not in overlap_prior]
    selected_prior = (overlap_prior + rest_prior)[-GROUNDING_VIEW_MAX_PRIOR:]
    if selected_prior:
        prior_block = (
            "\n\n## 前章已確立（grounded）的具體 entity（同義改寫對照用）\n"
            + "\n".join(f"- {e}" for e in selected_prior)
        )
        # prior block 也納入 budget：超出則不附（evidence 比 prior 重要）
        if used + len(prior_block) <= char_budget or not body:
            body = (body + prior_block) if body else prior_block.lstrip()
    return body


def serialize_evidence_pool(pool: Dict[int, EvidencePoolEntry]) -> str:
    """將 evidence_pool dict 序列化為 JSON 字串（存進 LiveResearchStageState）。

    JSON keys 必須是 str（int 在 JSON 不合法），讀取時再 cast 回 int。
    使用 ensure_ascii=False 保留中文字，方便人類 review DB row。
    """
    return json.dumps(
        {str(eid): entry.model_dump() for eid, entry in pool.items()},
        ensure_ascii=False,
    )


def deserialize_evidence_pool(s: str) -> Dict[int, EvidencePoolEntry]:
    """從 JSON 字串還原 evidence_pool dict。空字串回傳空 dict（兼容舊 DB row）。

    F-1 (full-scan 批7) fail-closed 防護：corrupt state（半寫入 checkpoint /
    DB 損壞 / 舊 schema drift）不該炸整條 pipeline，而要與 DR 側「corrupt 資料
    回退、非 500」的紀律對齊。

    - 整份 JSON 損壞（JSONDecodeError / 非 dict）→ log warning + 回空 dict，
      讓上游 caller（orchestrator Stage 5/6）走既有的判空回退（blocked /
      checkpoint），而不是往上炸 _run_stage_5 / Stage 6。
    - 單筆 entry 型別變形（ValidationError）或非數字 key（int() ValueError）→
      per-entry skip 該筆 + log warning，其餘 entry 正常還原（單筆壞不炸全池）。

    絕不 silent fail：每個被跳過的壞資料都留 warning，方便排查。
    """
    if not s:
        return {}

    try:
        raw = json.loads(s)
    except (ValueError, TypeError) as e:
        # json.loads 對壞 JSON 拋 JSONDecodeError（ValueError 子類）。
        logger.warning(
            f"[evidence_pool] deserialize failed: corrupt JSON, returning empty pool "
            f"(fail-closed); error={type(e).__name__}: {e}"
        )
        return {}

    if not isinstance(raw, dict):
        logger.warning(
            f"[evidence_pool] deserialize: top-level JSON is not a dict "
            f"(got {type(raw).__name__}), returning empty pool (fail-closed)"
        )
        return {}

    pool: Dict[int, EvidencePoolEntry] = {}
    for k, v in raw.items():
        try:
            pool[int(k)] = EvidencePoolEntry.model_validate(v)
        except (ValidationError, ValueError, TypeError) as e:
            # int(k) 非數字 key → ValueError；model_validate 型別變形 →
            # ValidationError。單筆壞跳過，不炸全池。
            logger.warning(
                f"[evidence_pool] skipping corrupt entry key={k!r}: "
                f"{type(e).__name__}: {e}"
            )
            continue
    return pool
