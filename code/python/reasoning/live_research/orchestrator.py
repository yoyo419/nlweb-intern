"""
LiveResearchOrchestrator — 6-Stage 對話驅動研究流程控制器。

Chat-based 架構：每則使用者訊息是獨立 HTTP request。
Session 追蹤 stage 狀態（LiveResearchStageState）。

6 Stages:
  1. 建立研究結構（B->A->B' loop）
  2. Per-Section 資料策略 + 蒐集
  3. 寫作準備（Style Analysis dialogue loop）
  4. 格式確認
  5. 分段輸出
  6. 匯出

Entry points:
  - start(query) → 從頭開始，進入 Stage 1
  - continue_from_checkpoint(state, user_message) → 從 checkpoint 繼續
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from misc.logger.logging_config_helper import get_configured_logger
from core.config import CONFIG
from reasoning.orchestrator_base import OrchestratorBase
from reasoning.agents.associator import AssociatorAgent
from reasoning.live_research.loop_engine import BABLoopEngine
from reasoning.live_research import lr_copy
from reasoning.live_research.sse_emit import emit_sse
from reasoning.live_research.stage_state import LiveResearchStageState
from core.llm import ask_llm
from reasoning.schemas_live import (
    BookOutline,
    ChapterPlan,
    ContextMap,
    ContextMapDelta,
    ContextMapRevisionOperation,
    ContextMapTopic,
    EvidencePoolEntry,
    GeneratedReportTitle,
    StyleAnalysisOutput,
    StyleInputNotASampleError,
    Stage4Response,
    Stage4StructuralPayload,
    Stage4FormatPayload,
    LiveWriterSectionOutput,
    context_map_to_summary,
    deserialize_evidence_pool,
    serialize_evidence_pool,
)

logger = get_configured_logger("live_research.orchestrator")


# Stage 4 confirmation round 關鍵字（user 第二輪「OK / 好」確認已記下的 format_specs）
CONFIRMATION_KEYWORDS = [
    "OK", "ok", "Ok", "好", "好的", "確認", "對", "沒問題", "就這樣", "可以", "go", "Go", "GO",
]

# UX-9: reframe confirm round 用 — 取消 keywords（user 想 abort reframe，回到 incremental path）
CANCEL_KEYWORDS = [
    "取消", "算了", "不要", "不用", "再想想", "cancel", "Cancel", "CANCEL", "nope", "no", "No", "NO",
]

# Stage 6 未完成章節偵測集合 + banner 中文原因映射（O4+O4-C 合併版集中）。
# 紀律：兩者必須同步維護 — 偵測集合每個 status 都要有中文原因，
# 否則 reason fallback 會把「未完成」顯示給 user（不再漏 raw status code）。
# 同步性由 test_live_orchestrator.py::
# test_stage6_problematic_statuses_have_reason_zh 把關。
_PROBLEMATIC_STATUSES = ("blocked_no_evidence", "guard_failed", "critic_rejected")
_PROBLEMATIC_REASON_ZH = {
    "blocked_no_evidence": "資料不足",
    "guard_failed": "驗證失敗",
    "critic_rejected": "查核未通過",
}

# LR 報告標題生成 LLM timeout（秒）。標題不需重推理，短 timeout 即可，
# 逾時降級 research_question（plan: lr-report-title-generation）。
_REPORT_TITLE_TIMEOUT = 15
# 生成標題長度上限（字），超長截斷（AR P2：LLM 可能回超長，schema description 只是軟約束）。
_REPORT_TITLE_MAX_LEN = 40


def _looks_like_confirmation(msg: str) -> bool:
    """Stage 4 mixed path 後的 confirmation 判定。

    判定條件（從嚴）：
    - 訊息 strip 後長度 ≤ 10 字元
    - **且** 訊息（剝掉常見標點）完全等於某個 CONFIRMATION_KEYWORDS

    這樣避免「OK 但我想加表格」這類含 keyword 但實際是 spec 修改的 case
    被誤判為 confirmation。
    """
    msg = msg.strip()
    if not msg:
        return False
    if len(msg) > 10:
        return False
    # 剝掉常見標點 / 空白後比對是否完全等於某個 keyword
    stripped = msg.strip(" .,!?!?。，、~～").strip()
    if not stripped:
        return False
    return any(stripped == kw or stripped.lower() == kw.lower() for kw in CONFIRMATION_KEYWORDS)


def _looks_like_cancel(msg: str) -> bool:
    """UX-9: reframe confirm round 的 cancel 判定（同 confirmation 從嚴策略）。"""
    msg = msg.strip()
    if not msg:
        return False
    if len(msg) > 10:
        return False
    stripped = msg.strip(" .,!?!?。，、~～").strip()
    if not stripped:
        return False
    return any(stripped == kw or stripped.lower() == kw.lower() for kw in CANCEL_KEYWORDS)


# FIX-5 (Cayenne #7, 2026-05-29)：reframe confirm round 的 confirm 關鍵字 shortcut。
# 純 confirm 動詞（含 compound「確認。進入寫作。」）。比舊 _looks_like_confirmation 寬鬆
# —— 用 substring 命中（compound 句仍中），但比 LLM classifier 嚴格（只在「短訊息 + 含
# confirm 動詞 + 不含任何 adjust/cancel 訊號」時才直接判 confirm）。
_CONFIRM_PROCEED_KEYWORDS = (
    "確認", "進入寫作", "開始寫", "開始撰寫", "沒問題", "就這樣", "確定",
    "ok", "okay", "go", "好的", "可以了",
)
# adjust/cancel 訊號：出現任一即不走 shortcut，交回 LLM classifier 細判
# （避免誤攔「確認後幫我把第二章改短」這類 confirm+adjust 混合句）。
_CONFIRM_SHORTCUT_VETO_KEYWORDS = (
    "改", "修", "調整", "換", "加", "新增", "刪", "移除", "合併", "拆", "重組",
    "但", "不過", "然後", "幫我", "把第", "章", "段", "字數", "格式",
    "取消", "算了", "不要", "不用", "再想想", "等等", "cancel", "no", "nope",
)


def _looks_like_confirm_proceed_shortcut(msg: str) -> bool:
    """FIX-5：判斷 reframe pending 回覆是否為「純 confirm（含 compound）」可走 shortcut。

    命中條件（全部成立）：
    - 訊息 strip 後非空、長度 ≤ 20 字元（compound「確認。進入寫作。」仍在範圍內）
    - 含至少一個 confirm 動詞（substring，compound 句也中）
    - **不含**任何 adjust/cancel veto 訊號（confirm+adjust 混合 → 不攔，交 LLM 細判）

    回 True → caller 直接走 confirm path，不打 LLM intent parse（省成本 + 防誤判）。

    DP-12 reconcile（2026-06-02，meta-intent 窄版 plan）：此處刻意用 **substring**
    （非 Stage 5 的 frozenset 完全匹配），因為 reframe confirm 常見 compound 句
    「確認。進入寫作。」用 frozenset 整句完全匹配會漏（整句 != 「確認」）。abort 類
    （算了/取消/不要/不用）已在 _CONFIRM_SHORTCUT_VETO_KEYWORDS → 命中 veto → 不走
    shortcut，交 LLM classifier 細判（→ cancel，spec §4.3.4）。因此 DP-12 與 Stage 5
    frozenset 是「不同語意層」，**不統一成同一結構**：reframe 要容 compound confirm，
    Stage 5 要嚴防帶內容句誤觸匯出。盲目對齊成 frozenset 會回退 compound confirm bug。
    """
    s = (msg or "").strip()
    if not s or len(s) > 20:
        return False
    lower = s.lower()
    if not any(kw.lower() in lower for kw in _CONFIRM_PROCEED_KEYWORDS):
        return False
    if any(kw.lower() in lower for kw in _CONFIRM_SHORTCUT_VETO_KEYWORDS):
        return False
    return True


# Bug #14 root fix (取代 substring + veto 枚舉 reward hack)：Stage 5 的 continue /
# export shortcut 只是「省 LLM 成本」的優化。fall through 到 LLM intent parse 永遠安全
# （LLM 會正確分類 revise/continue/done）。因此 shortcut 不該用 substring「猜」訊息是不是
# 純確認，只在訊息正規化後「整句完全等於」某個純確認/匯出詞時才命中，其餘一律交 LLM。
# 正規化 = strip + 去頭尾標點/空白；比對用 set 完全匹配（含 lower-case 英文）。這樣
# 「好像哪裡怪」「不錯,繼續」「第2段還沒完成」這類帶內容句必然 fall through，不再有
# substring 漏洞，也不需枚舉 veto 詞。
_CONTINUE_SHORTCUT_KEYWORDS = frozenset({
    "好", "好的", "ok", "繼續", "下一段", "下一章", "next", "go",
    "繼續寫", "接著寫", "往下寫",
})
_EXPORT_SHORTCUT_KEYWORDS = frozenset({
    "匯出", "export", "完成", "結束", "下載", "下一階段", "下一個階段",
    # abort guardrail follow-up（CEO reframe 2026-06-02）：abort checkpoint 給「接受/繼續編輯」。
    # 「接受」= 確認匯出，exact-match 才命中（「接受但第2段再短一點」整句 != 「接受」→ fall through）。
    "接受",
})
# 正規化剝除的頭尾標點/空白（沿用 _looks_like_confirmation 同一組）
_SHORTCUT_STRIP_CHARS = " .,!?！？。，、~～"


def _normalize_shortcut_msg(msg: str) -> str:
    """正規化使用者訊息供 shortcut 完全匹配：strip + 去頭尾標點 + lower。"""
    s = (msg or "").strip().strip(_SHORTCUT_STRIP_CHARS).strip()
    return s.lower()


def _looks_like_continue_shortcut(msg: str) -> bool:
    """正規化後整句完全等於某個純確認詞 → 命中 continue shortcut（直接寫下一段）。

    完全匹配（非 substring）：「好」命中，「好像哪裡怪」「不錯,繼續」不命中（fall
    through 到 LLM）。空字串 / 純標點不命中。
    """
    return _normalize_shortcut_msg(msg) in _CONTINUE_SHORTCUT_KEYWORDS


def _looks_like_export_shortcut(msg: str) -> bool:
    """正規化後整句完全等於某個純匯出詞 → 命中 export shortcut（直接進 Stage 6）。

    完全匹配：「完成」命中，但「第2段還沒完成」「完成度不夠」不命中（整句 != 「完成」，
    fall through 到 LLM）。
    """
    return _normalize_shortcut_msg(msg) in _EXPORT_SHORTCUT_KEYWORDS


# === Stage 5 recollect confirm parsers（plan: lr-stage5-backward-recollect, A/B/K）===
# B（confirm 詞收斂，3方共識）：**不收單字「要/是/好」**（過寬 → 誤觸不可逆刪章，B 原罪）。
# K（Codex+in-house）：但純 exact-match 又太窄 —— 「OK。」「好，開始吧」「確認，請重新蒐集」
# 等自然短肯定句不在白名單 → 落 substantive → 可能被下游 _parse_revision_intent 重 parse
# 成 recollect → 二次 consent loop（user 已確認卻被再問一次）。平衡點：**多字明確肯定詞
# 命中，單字「要/是/好」不命中（避免 B 原罪復發），含實質修改名詞的長句走 substantive**。
# 注意：刻意只放**多字**確認詞 + 「好的」（非裸「好」）。「好，開始吧」靠「開始」/「吧」命中，
# 不靠裸「好」；裸「好」「要」「是」單獨出現 → 不命中 → 交 _classify_meta_intent / fall through。
_RECOLLECT_CONFIRM_TOKENS = frozenset({
    "確認", "確定", "好的", "開始", "沒問題", "ok", "yes", "可以", "沒錯", "對的",
})
# 出現以下任一「實質修改名詞」即視為含修改訴求 → 不當純確認（走 substantive fall through）。
# 涵蓋 user 在 consent round 順帶提修改的常見詞（改章節 / 加內容 / 換方向 / 補面向）。
_RECOLLECT_REVISE_MARKERS = (
    "段", "章", "節", "改", "加", "增", "刪", "換", "調整", "修改", "重寫",
    "方向", "面向", "經濟", "政治", "標題", "字數", "風格",
)
# 短肯定句長度上限（中文）：超過此長度即使含確認詞也視為「夾帶實質訴求」走 substantive，
# 避免長句被誤判純確認（K 平衡點，刻意保守偏短）。
_RECOLLECT_CONFIRM_MAX_LEN = 12


def _strip_affirmative_punct(msg: str) -> str:
    """strip 前後標點 / 空白（含全形標點），保留中間字判長度。"""
    import re
    return re.sub(r"^[\s，。！、,.!？?~～]+|[\s，。！、,.!？?~～]+$", "", msg.strip())


def _looks_like_recollect_confirm(msg: str) -> bool:
    """含確認 token 的 bounded affirmative parser（四段式 recollect confirm 段1 用，K）。

    True 條件（全部成立）：strip 標點/空白後 (1) 不含任何實質修改名詞（_RECOLLECT_REVISE_MARKERS）
    且 (2) 長度 ≤ _RECOLLECT_CONFIRM_MAX_LEN 且 (3) 至少含一個確認 token。
    → 「確認」「OK。」「好，開始吧」「可以，請重新蒐集」命中；
       「改第3段」「資料還是不夠，連經濟面也查」不命中（含修改名詞 / 過長）→ fall through。

    刻意保守：含確認 token 才命中，無 token 的自然肯定句（「好，那就重新蒐集吧」「是的」）
    在此**不命中** —— 交段3 `_looks_like_bounded_affirmative_shape`（先過 abort 分類器後）兜底。
    """
    stripped = _strip_affirmative_punct(msg)
    low = stripped.lower()
    if not low:
        return False
    # (1) 含實質修改名詞 → 不是純確認，走 substantive（避免吞掉 user 順帶提的修改訴求）
    if any(marker in stripped for marker in _RECOLLECT_REVISE_MARKERS):
        return False
    # (2) 過長 → 視為夾帶實質內容，保守走 substantive
    if len(stripped) > _RECOLLECT_CONFIRM_MAX_LEN:
        return False
    # (3) 至少含一個確認 token（去掉標點後逐 token 比對 + 子字串命中短肯定詞）
    if low in _RECOLLECT_CONFIRM_TOKENS:
        return True
    return any(tok in low for tok in _RECOLLECT_CONFIRM_TOKENS)


def _looks_like_bounded_affirmative_shape(msg: str) -> bool:
    """無 token 的 bounded affirmative「形狀」parser（K Round 4，四段式 recollect confirm
    段3 用，in-house R3 終驗修）。

    **必須在 abort 分類器（_classify_meta_intent==ABORT）判定為非 abort 後才呼叫** ——
    本函式刻意**不依賴確認 token 白名單**（解決「好，那就重新蒐集吧」「是的」「行」「成」
    「麻煩了」這類無 token 自然肯定句漏接），因此單靠它無法區分「算了」（同為無 marker 短句）
    與「好」；abort 區分交給先行的 _classify_meta_intent，本函式只負責「非 abort 的短肯定形狀」。

    True 條件（全部成立）：strip 標點/空白後 (1) 非空 且 (2) 不含任何實質修改名詞
    （_RECOLLECT_REVISE_MARKERS）且 (3) 長度 ≤ _RECOLLECT_CONFIRM_MAX_LEN。
    → 「好，那就重新蒐集吧」「是的」「行」「成」「麻煩了」「嗯」命中（在 consent gate 內 = 確認）；
       「改第3段」「資料還是不夠，連經濟面也查」不命中（含修改 marker / 過長）→ 段4 fall through。

    為何安全（不復發 B 過寬）：B 原罪是「裸『要/好』被 confirm 白名單收進**一般** Stage 5
    路徑」。本兜底**只在 pending_recollect_confirmation==True 的 consent gate 內**被呼叫
    （user 剛被問「確認要重新蒐集嗎？」），且 abort 已先攔（「算了」走不到這），含修改 marker
    的句子被 (2) 排除（「改第3段」絕不誤觸刪章）。consent gate 內非 abort 的短肯定句語意明確
    = 確認，非泛用放行。
    """
    stripped = _strip_affirmative_punct(msg)
    if not stripped:
        return False
    # (1) 含實質修改名詞 → 不是純確認，留給段4 substantive fall through（B 原罪防護）
    if any(marker in stripped for marker in _RECOLLECT_REVISE_MARKERS):
        return False
    # (2) 過長 → 視為夾帶實質內容，保守走段4 substantive
    if len(stripped) > _RECOLLECT_CONFIRM_MAX_LEN:
        return False
    # 非空 + 無 marker + 短 → 在 consent gate 內（且已排除 abort）視為確認形狀
    return True


def _count_chapter_words(content: str) -> int:
    """章節字數 = 剝除 {cite:N} 引用 placeholder 後的字元數。

    設計（lr-chapter-word-budget plan 設計細節 1）：
    - 中文無空白分詞，字元數即近似字數（學術慣例）。
    - 在 citation render 之前計算，content 仍是統一的 {cite:N}，剝一個 regex 即可，
      不必處理 render 後 4 種變體（[N] / 上標 / (作者, 年) / 空）長度不一。
    - 標點不剝除（過度工程；target_word_count 本就含標點的粗估目標）。
    """
    import re
    return len(re.sub(r"\{cite:\d+\}", "", content or ""))


# mock_bab fixture 目錄（單點切換；rollback = 換回 "lr_mock_bab_real"，舊目錄保留）。
# 現行：Cayenne 綠能命題（prod session 8e1db658-3bac-4071-a4f3-cdcb53e8c162，2026-07-15，
# 567 筆 evidence / 20 topics / 3 章：前言・國際案例分析・結論 / 40 id・172 claims）。
_MOCK_BAB_FIXTURE_DIRNAME = "lr_mock_bab_cayenne_2026_07"


def _resolve_target_chapter_layer1(target: str, chapter_names):
    """R2 第一層 code 短路（不另起 LLM call、復用既有 classifier 結果）：只處理「明擺著的」——
    exact / 唯一 substring / 空。

    回傳：
    - "" → 空 target（= user 明說全章/每章都要，全章注入 sentinel；B2：classifier 只在「明說全章」時填空）
    - 章名原文 → exact 相等 或 唯一 substring 命中（明確，不採 LLM 語意、不問）
    - None → 非 exact 非唯一（語意指涉 / 序數 / 多候選 / 對不到）→ **放手交第二層 LLM 判語意**

    禁 hardcode 章名映射：純比對傳入 chapter_names。序數/語意指涉在此**不猜**，回 None 交 LLM。
    """
    s = (target or "").strip()
    if not s:
        return ""
    names = [(n or "").strip() for n in (chapter_names or [])]
    for n in names:  # exact
        if n and s == n:
            return n
    cands = [n for n in names if n and (s in n or n in s)]  # 唯一 substring
    if len(cands) == 1:
        return cands[0]
    return None  # 交第二層 LLM


def _serialize_special_element_for_state(elem):
    """B3 集中 serializer：所有寫入 state.format_specs["special_elements"] 的路徑一律走此，
    強制排除 transient 語意判斷欄位，保持持久化標準 shape 與今日一致（OQ-4）。

    **持久化標準 shape（拍板）= 固定三欄** `{type, target_chapter, description}`，
    缺欄位補空字串（None → ""）。不多欄、不少欄——與今日 special_elements dict shape 逐字一致。

    elem：SpecialElementSpec 或 dict。回持久化用純三欄 dict。
    """
    d = elem.model_dump() if hasattr(elem, "model_dump") else dict(elem)
    return {
        "type": d.get("type", ""),
        "target_chapter": d.get("target_chapter", "") or "",
        "description": d.get("description", "") or "",
    }


def _diagnose_unmatched_special_element_targets(all_special_elements, chapter_titles):
    """report-level no-silent-fail 後衛：outline 定案後，找出「對不到任何章」的
    special_element target（exact 比對 finalized 章名）。只在有 outline 時判。
    Returns：對不到任何章的 target 字串 list（去重保序），供 caller emit 一次 narration。
    SF2 註：本 helper 只讀 elem["target_chapter"]、回傳字串 list，**不把 elem 傳給下游**，
    故不需 serializer sanitize（transient 欄位不會外洩到 writer / narration）。
    """
    if not all_special_elements or not chapter_titles:
        return []
    titles = [(t or "").strip() for t in chapter_titles]
    unmatched = []
    seen = set()
    for elem in all_special_elements:
        if not isinstance(elem, dict):
            continue
        target = (elem.get("target_chapter") or "").strip()
        if not target or target in seen:
            continue
        if target not in titles:  # exact 對不到任何 finalized 章名
            unmatched.append(target)
            seen.add(target)
    return unmatched


def _is_intro_or_conclusion(book_outline, idx: int) -> bool:
    """Track A Task 3 (sprint 2026-05-28) — Gemini Critical 紅隊 #2 runtime
    double-check (defense in depth)。

    即使 Task 2 schema validator 已 enforce role/index 一致，runtime 此處仍
    cross-validate role 與 idx 雙重 match。

    紅隊 #2 場景：LLM 把 body 章節標 role="intro" 想繞 Task 3 deterministic gate。
    - schema validator (Task 2) 在 test/CI 模式必 raise (fail-loud)
    - runtime 此處不信「role 單獨」也不信「index 單獨」，要求 role + idx 雙重一致
      才回 True；不一致 → log warning + 視同 body 走 gate
      (保守，不 raise，不讓 prod LLM 偶發 crash)

    codex C-1 v2 紀律：預設值改為**保守 = False = 當作 body = 走 gate**。
    - book_outline=None → 回 False (不知道 outline 不能假設 intro/conclusion；
      當作 body chapter 走 deterministic gate); 不可回 True (會 bypass gate);
      不可 raise (runtime 不要 crash)
    - 取不到 chapter / 取不到 role enum → 同樣回 False = 走 gate
    """
    if book_outline is None:
        # codex C-1 v2: 不知道 outline → 當作 body → 走 gate
        return False
    try:
        ch = book_outline.chapters[idx]
        role = getattr(ch, "role", "") or ""
        n = len(book_outline.chapters)
        # Gemini 紅隊 #2 defense in depth: role + index 雙重一致才回 True
        if role == "intro" and idx == 0:
            return True
        if role == "conclusion" and idx == n - 1:
            return True
        # role 與 idx 不一致 (LLM 幻覺把 body 標 intro，或把第 2 章標 intro)
        # → 不信 role，視同 body 走 gate
        # R1 reviewer I-4 fix (sprint 2026-05-28): 升 ERROR (SRE 可監控頻率)。
        # schema_validator (BookOutline._validate_chapters_role_index_consistency)
        # 是 fail-loud raise; 此處能執行到表示 model_construct() bypass 或直接
        # raw constructor 創建 — 不正常路徑，需 SRE 注意。紀律保持: 仍 return False
        # (不 raise / 不 crash runtime), 視同 body chapter 走 C-1 gate。
        if role in ("intro", "conclusion"):
            logger.error(
                f"[LIVE RESEARCH] CHAPTER ROLE INCONSISTENCY: chapter idx={idx} "
                f"role={role!r} but position invalid (n={n}); schema_validator "
                f"should have caught this. Falling back to body chapter (C-1 gate "
                f"applies). This indicates model_construct() bypass or direct "
                f"object creation."
            )
            return False
    except (IndexError, AttributeError):
        # codex C-1 v2: 取不到 chapter / role → 當作 body → 走 gate
        return False
    return False


async def _extract_entities_from_section(
    section_content: str,
    handler: Any,
) -> List[str]:
    """Track A Task 7 (sprint 2026-05-28) — cheap LLM helper:
    從 section content 抽出具體 entity (國家 / 城市 / 地名 / 機構 / 法規 / 人名 / 風場名)。

    紀律:
    - 抽具體名詞, 不抽抽象概念
    - 重複的只列一次 (dedupe 保 order)
    - LLM call 失敗 → 回 [] + log warning (不阻塞 pipeline)
    """
    from core.llm import ask_llm

    prompt = (
        "從下列段落抽出**具體 entity** (國家名 / 城市名 / 地名 / 機構名 / 法規名 / "
        "人名 / 風場名):\n"
        "- 抽具體名詞, 不要抽抽象概念\n"
        "- 重複的只列一次\n\n"
        f"段落:\n{section_content}\n\n"
        "回傳 JSON: {\"entities\": [\"e1\", \"e2\", ...]}"
    )
    schema = {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
        "required": ["entities"],
    }
    try:
        resp = await ask_llm(
            prompt, schema, level="low",
            query_params=getattr(handler, "query_params", {}),
            max_length=1024, timeout=10,
        )
    except Exception as e:
        logger.warning(
            f"[LIVE RESEARCH] _extract_entities_from_section failed (non-fatal): "
            f"{type(e).__name__}: {e}"
        )
        return []
    ents = (resp or {}).get("entities") or []
    if not isinstance(ents, list):
        return []
    # dedupe 保 order
    seen: set = set()
    out: List[str] = []
    for x in ents:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ──── 窄版 meta-intent 前置分類 helper（CEO 拍板 2026-06-01，#6/#14/#16 同源病）────
# 設計：可重用的輕量 LLM 分類器，辨識「流程指令」vs「實質內容」。**只接 Stage 3 入口 +
# Stage 5 abort 兩個點**，不普遍套所有 stage（避免過度工程）。純確認類（好/OK/繼續/匯出）
# 由既有 frozenset shortcut 即時放行，不打此 helper。
META_INTENT_SKIP = "skip_use_default"     # 「用預設/跳過/不提供/不用了」= 不想給內容、用預設
META_INTENT_ABORT = "abort_cancel"        # 「算了/取消/不要了/放棄/先停」= 想中止當前流程
META_INTENT_SUBSTANTIVE = "substantive"   # 實質內容（範本/修改訴求/混合句）→ 交既有路徑處理

# Stage 3「重新提供範本」按鈕的 sentinel user_message。
# 前端按鈕點擊時送此確切字串（非 LLM 可生成的自然語句），後端在 round-2 入口
# 比對到它即識別為「使用者明確要求重新提供範本」的手勢，清空既有分析、重問新範本。
# 不靠 LLM intent 判斷（明確手勢 → 明確訊號）。
STAGE3_NEW_SAMPLE_SENTINEL = "__LR_STAGE3_NEW_SAMPLE__"


async def _classify_meta_intent(user_message: str, handler: Any) -> Optional[str]:
    """輕量 meta-intent 分類（low-model LLM）。

    回傳：
    - META_INTENT_SKIP / META_INTENT_ABORT / META_INTENT_SUBSTANTIVE 之一
    - None：LLM API 失敗（空回應 / 例外 / 未知）→ caller 必須 fail-loud
      （沿用 #21：narration 用系統端文案，不可預設成某類意圖蒙混、不可怪 user）

    紀律：
    - **混合句**（同時含 meta 訊號與實質內容，如「確認。然後第3段加案例」）→ substantive
      （本窄版只處理純 meta 指令；混合句交既有 _parse_revision_intent / style analysis）。
    - **唯一例外 = abort**：明確 abort 詞（算了/取消/不要了/放棄/先停）優先回 abort_cancel，
      因為誤匯出（不可逆）代價最高，err toward NOT exporting（寧可多問一次）。
    - 未知 category → 保守回 substantive（不誤攔正常輸入）。
    """
    from core.llm import ask_llm

    prompt = f"""你是一個「流程指令 vs 實質內容」分類器。使用者正在回覆系統的某個 checkpoint。

使用者訊息：
{user_message}

判斷這則訊息屬於哪一類，回傳 JSON：

- category:
  * "skip_use_default"（使用者表達**不想提供內容、要用預設或跳過**，例如：
      「用預設就好」/「用預設的學術風格」/「跳過」/「不提供」/「不用了」/「沒有範本」/
      「你決定風格就好」。注意：只在「拒絕提供 / 要用預設」時用，不是「接受某結果」。）
  * "abort_cancel"（使用者表達**想中止 / 放棄當前流程**，例如：
      「算了」/「取消」/「不要了」/「放棄」/「先停」/「不做了」/「停掉」。
      這類訊號優先級最高 —— 只要含明確放棄/中止意圖就歸此類。）
  * "substantive"（使用者**提供了實質內容或實質訴求**，包含：一段文筆範本、
      一個具體修改要求（「第3段太短」）、或**混合句**（既確認又提訴求，
      如「確認，然後第3段加案例」）。凡是帶實質內容的一律歸此類。）

- reason: 簡述判斷原因（繁體中文）

紀律：
- 「算了」「取消」「放棄」這類**明確中止詞** → abort_cancel（最高優先，誤判代價最高）。
- 帶任何實質內容 / 具體訴求 / 範本文字 → substantive（混合句也歸 substantive）。
- skip_use_default 僅適用於「明確拒絕提供 + 要用預設 / 跳過」，不含 abort、不含實質內容。
"""
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": [
                META_INTENT_SKIP, META_INTENT_ABORT, META_INTENT_SUBSTANTIVE,
            ]},
            "reason": {"type": "string"},
        },
        "required": ["category", "reason"],
    }
    try:
        response = await ask_llm(
            prompt,
            schema,
            level="low",
            query_params=getattr(handler, "query_params", {}),
            max_length=2048,
        )
    except Exception as e:
        logger.warning(
            f"[LIVE RESEARCH] _classify_meta_intent failed (None, fail-loud): "
            f"{type(e).__name__}: {e}"
        )
        return None
    if not response:
        logger.warning("[LIVE RESEARCH] _classify_meta_intent: ask_llm returned empty (None)")
        return None
    # Unwrap schema-wrapped response: {type, properties, required} → properties dict
    if "category" not in response and "properties" in response and isinstance(response["properties"], dict):
        response = response["properties"]
    category = response.get("category")
    if category in (META_INTENT_SKIP, META_INTENT_ABORT, META_INTENT_SUBSTANTIVE):
        return category
    # 未知 category → 保守當 substantive（不誤攔正常輸入；abort 靠專屬詞，model 該抓到）
    logger.warning(
        f"[LIVE RESEARCH] _classify_meta_intent: unknown category {category!r}, "
        f"defaulting to substantive"
    )
    return META_INTENT_SUBSTANTIVE


def _section_dict(
    section_output,
    section_index: int,
    entities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Track A Task 7 + addendum I-1 (sprint 2026-05-28): 統一構造 written_sections
    dict (避免 3 個 mutation 點漏 key — Stage 5 main append / revise in-place /
    revise else-branch append)。

    必含 keys: section_index, title, content, sources_used, confidence_level,
              chapter_summary, entities, status (addendum C-2)
    """
    return {
        "section_index": section_index,
        "title": section_output.section_title,
        "content": section_output.section_content,
        "sources_used": section_output.sources_used,
        "confidence_level": section_output.confidence_level,
        "chapter_summary": getattr(section_output, "chapter_summary", "") or "",
        "entities": entities or [],  # Track A Task 7
        # addendum C-2: section status
        "status": getattr(section_output, "status", "drafted") or "drafted",
        # mock_bab E2E fix (2026-05-29): F1 critic WARN marker 存在 methodology_note，
        # 之前漏帶此 key → marker 序列化即丟，前端永遠收不到。補上讓資料送出。
        "methodology_note": getattr(section_output, "methodology_note", "") or "",
    }


def _extract_chapters_from_ops(operations) -> list:
    """Plan 2 Phase 4: 從 ContextMapRevisionOperation list 抽出第一個非空
    new_chapters 欄位，轉成 format_specs.chapters 格式 [{name, outline}, ...]。

    CEO 決策：reuse reframe parser 的 new_chapters 輸出（不另開 parser path）。
    即使 LLM 沒 dispatch reframe_structure op_type，只要任一 op 帶 new_chapters，
    仍能 fallback 把 chapter list 寫進 state.format_specs["chapters"]。

    Schema 對應：
    - new_chapters[i].name → format_specs.chapters[i].name
    - new_chapters[i].description (optional) → format_specs.chapters[i].outline
    - new_chapters[i].word_target (optional, >0) → format_specs.chapters[i].word_target
      （C8 fix：user 在 reframe 句中為各章指定的字數，過去被丟棄）
    - relevance 不轉（chapter override 沒 relevance 概念，沿用全 core）

    Args:
        operations: List[ContextMapRevisionOperation]

    Returns:
        List[Dict[str, Any]] — [] 表示沒任何 op 帶 new_chapters。
    """
    for op in operations:
        chapters = getattr(op, "new_chapters", None) or []
        if not chapters:
            continue
        extracted = []
        for spec in chapters:
            if not isinstance(spec, dict):
                continue
            name = (spec.get("name") or "").strip()
            if not name:
                continue
            outline = (spec.get("description") or "").strip()
            entry = {"name": name, "outline": outline}
            wt = spec.get("word_target")
            if isinstance(wt, int) and wt > 0:
                entry["word_target"] = wt
            extracted.append(entry)
        if extracted:
            return extracted
    return []


# ──────────────────────────────────────────────────────────────────────
# LR evidence aggregate cap (chapter-0 context bomb fix, 與 keystone f172d9b 綁定)
# 借 DR _format_context_shared 的 char-budget 累加思路 (reasoning/orchestrator.py:114-181)，
# 但**選誰留下**的策略不同：DR 的 items 已 relevance-pre-sorted 可純截尾；LR 的
# analyst_citations 是多 topic 交錯、無 retrieval score，純頭部截尾 = deterministic
# topic starvation（系統性丟掉排序在後的 topic 全部證據）→ 兩個獨立外部模型
# (Gemini+ChatGPT) 收斂反對。改：
#   1. planned_evidence_ids ∩ pool 全保留（LLM 已做 topic assignment，各 topic 平衡）
#   2. remaining（不在 planned 的）用 stratified 均勻抽樣補位，橫跨整個檢索時間軸
#      （evidence_id 升冪 = 檢索時間軸，loop_engine.py:516-517 _evidence_counter 單調遞增），
#      不頭部截斷 → 不 starve 任何 topic
#   3. 照 planned-then-stratified 序動態累加 snippet 字數至 MAX_EVIDENCE_CHARS（主 binding
#      cap）；MAX_EVIDENCE_ITEMS 只當「絕對不超過幾筆」的次要安全閥
# MAX_EVIDENCE_CHARS 推導見 plan「設計建議」段（目標 model=gpt-5.1 / fallback 128K + writer
# max_tokens=16384 reservation 反推 ≈ 10K tokens ≈ 20000 char，並對齊 DR MAX_TOTAL_CHARS）。
# ──────────────────────────────────────────────────────────────────────
MAX_EVIDENCE_CHARS = 20000     # 主 binding cap：餵 writer+critic 的 evidence 文字總字數上限
MAX_EVIDENCE_ITEMS = 80        # 次要 backstop：絕對不超過幾筆。放寬到 80（非 40）才能讓
                               # char budget 先觸發 —— per-item snippet[:200] 估 ~303 字，
                               # 40*303≈12120<20000（N 會先 bind 使 budget 虛設），
                               # 80*303≈24240>20000（char budget 先 drop，N 只當極短 snippet 洪水 backstop）。
_EVIDENCE_OVERHEAD_PER_ITEM = 150   # 每筆 marker + title + url + 換行估算
                                    # （150 非 100：writer evidence 行新增發布日期與
                                    # 「今年＝YYYY 年」錨定註記約 45 字，見
                                    # prompts/writer.py evidence_block 與 core/temporal_anchor.py）
_EVIDENCE_SNIPPET_CHARS = 200       # 與下游 writer prompts/writer.py:808 的 (snippet)[:200] 對齊

# P2 W9（SF1 / §3.1）：per-chapter evidence 充分度門檻（抽 module 常數，不留 inline 魔術數字）。
EVIDENCE_THIN_CHAPTER_CITATIONS = 2  # 全 pool 量 <= 此值 → thin；可調


def _compute_chapter_sufficiency(analyst_citations, evidence_pool):
    """P2 W9（SF1）：用「全 pool 有料量」判 critical / thin / ok（非 analyst_citations 量）。

    全局 evidence 模型下 writer 讀全 pool，analyst_citations 空不代表沒 evidence。
    pool 完全空 → critical；pool 量 <= EVIDENCE_THIN_CHAPTER_CITATIONS → thin；否則 ok。
    （intro/conclusion 章的 'ok' 覆寫由 caller 在 _is_intro_or_conclusion 處理，保留既有。）
    """
    _pool_count = len(evidence_pool or {})
    if _pool_count == 0:
        return "critical"
    if _pool_count <= EVIDENCE_THIN_CHAPTER_CITATIONS:
        return "thin"
    return "ok"


def _stratified_sample(items, needed):
    """從 items（已排序）均勻抽 needed 筆，橫跨頭中尾（不頭部截斷）。

    stride = len(items)/needed；取每段「中點」index = min(int((i+0.5)*stride), n-1)。
    用 (i+0.5) 而非 i 避免左偏 —— `int(i*stride)` 第 0 筆永遠取 index 0、且永遠拿不到
    最後一個 index（尾段 topic 系統性少抽）。(i+0.5)*stride 取每段中點 → 頭尾都覆蓋。
    deterministic（同輸入同輸出，可測）。needed >= len → 全回。
    """
    n = len(items)
    if needed >= n:
        return list(items)
    if needed <= 0:
        return []
    stride = n / needed
    picked = []
    seen = set()
    for i in range(needed):
        idx = min(int((i + 0.5) * stride), n - 1)
        if idx not in seen:
            seen.add(idx)
            picked.append(items[idx])
    return picked


def _item_chars(entry):
    """單筆 evidence 估算字數（與下游 writer evidence_block 對齊）。

    ⚠️ future note：此估算與真 writer prompt format 綁定 —— snippet[:200] 截斷對齊
    `prompts/writer.py:808`、overhead=100 估 marker/title/url/換行。**未來若改 writer
    evidence_block 的 format（snippet 截長度、加欄位、改 marker）→ 必須同步改此處
    `_EVIDENCE_SNIPPET_CHARS` / `_EVIDENCE_OVERHEAD_PER_ITEM` / 本函式**，否則 cap 估算
    與真實送進 prompt 的字數脫鉤、char budget 失準。
    """
    snippet = (getattr(entry, "snippet", "") or "")[:_EVIDENCE_SNIPPET_CHARS]
    title = getattr(entry, "title", "") or ""
    return len(snippet) + len(title) + _EVIDENCE_OVERHEAD_PER_ITEM


def _cap_evidence_citations(citations, evidence_pool, planned_evidence_ids=None):
    """選 evidence：planned 優先（受 budget，保底 ≥1）+ remaining stratified 補位 + char budget 主 cap。

    P2 全局 evidence 模型（W2）：cap 後的 list 是「優先 tier 名單」（analyst_citations）——
    決定 W5 writer 視圖排序與 budget 內誰先進，**不再等於 writer 唯一可見集**（後者 =
    全 evidence_pool，見 W3）。本函式演算法不變，只是消費語意從「切割可用集」改「決定優先序」。

    一處 cap，writer 的 evidence_lookup（從 analyst_citations 建）受 char budget 限。
    注意（模塊1 A.2 / 43bd5c61, 2026-06-09 起）：critic 的 chapter_evidence_text 已改
    「全 evidence pool 視圖」，受 R2 GROUNDING_VIEW_CHAR_BUDGET 管轄，與本 cap 不同源
    （不再「同步受限」）；兩鏈各有 cap，F1/F3 仍不爆窗。

    防 topic starvation（兩家外部模型收斂結論）：
      - planned_evidence_ids（LLM 對該章做的 topic assignment）∩ pool **優先保留**，
        各 topic 都有代表，不抽樣不頭部截斷；但**仍受 char budget**（保底至少 1 筆，
        防 80 筆 planned 自身爆窗 —— Round-2 Must-Fix #1）。
      - remaining（analyst_citations 扣掉 planned）用 stratified 均勻抽樣補位，
        橫跨整個檢索時間軸（evidence_id 升冪），**不頭部截斷**；超 budget 的單筆用
        `continue` 跳過（非 `break`），保住尾段 topic 覆蓋（Round-2 Must-Fix #2）。
      - char budget（MAX_EVIDENCE_CHARS）為主 binding cap：照 planned-then-stratified
        序動態累加 snippet 字數至上限；MAX_EVIDENCE_ITEMS 為次要安全閥。

    Args:
        citations: analyst_citations（list[int]，可能亂序，內部排序去重）
        evidence_pool: Dict[int, EvidencePoolEntry]（phantom id 自動略過）
        planned_evidence_ids: 該章 planned_evidence_ids（list[int] / None）；
            None 或空（如 chapter-0 raw union）→ 全部 citations 當 remaining 做抽樣。

    Returns:
        capped citations（list[int]，evidence_id 升冪，已套選擇策略 + char + N 上限）
    """
    if not citations:
        return []
    pool = evidence_pool or {}

    # 只保留 pool 內存在的 ID（phantom guard），去重 + 依 evidence_id 升冪（= 時間軸）
    valid = sorted({eid for eid in citations if eid in pool})
    if not valid:
        return []

    planned_set = {eid for eid in (planned_evidence_ids or []) if eid in pool}
    # planned 依時間軸排序（保留全部 planned∩valid）
    planned_kept = [eid for eid in valid if eid in planned_set]
    remaining = [eid for eid in valid if eid not in planned_set]

    # ── Step 1: planned 全保留（受 char budget；保底至少留 1 筆 grounding 不消失）──
    # ⚠️ 修正（兩家外部模型 Round-2 收斂 Must-Fix #1）：planned 不可「全保留不檢查
    #   char budget」。realistic 下 ~80 筆 planned × ~300 字 = ~24000 > 20000 →
    #   仍會 context_length_exceeded（爆窗保證被打破）。故 planned 也受 char budget，
    #   但用 `selected and ...` 守住「至少留 1 筆」（第 1 筆即使超標也保留 → grounding
    #   不會整章消失）。其餘 planned 一樣受 budget。
    selected = []
    cumulative = 0
    for eid in planned_kept:
        ic = _item_chars(pool[eid])
        # 保底：至少留 1 筆 planned（grounding 不消失）；之後 planned 也受 char budget
        if selected and cumulative + ic > MAX_EVIDENCE_CHARS:
            break
        if len(selected) >= MAX_EVIDENCE_ITEMS:
            break
        selected.append(eid)
        cumulative += ic

    # ── Step 2: remaining 用 stratified 均勻抽樣補位（橫跨時間軸，不頭部截斷）──
    # 先估還能補幾筆（用平均 item 字數粗估抽樣目標數，再精確累加）
    if remaining and cumulative < MAX_EVIDENCE_CHARS and len(selected) < MAX_EVIDENCE_ITEMS:
        avg_item = max(1, (cumulative // max(1, len(selected))) if selected
                       else _EVIDENCE_SNIPPET_CHARS + _EVIDENCE_OVERHEAD_PER_ITEM)
        budget_slots = (MAX_EVIDENCE_CHARS - cumulative) // avg_item
        item_slots = MAX_EVIDENCE_ITEMS - len(selected)
        needed = max(0, min(budget_slots, item_slots, len(remaining)))
        sampled = _stratified_sample(remaining, needed)
        # ⚠️ 修正（Round-2 收斂 Must-Fix #2，Gemini sharp catch）：超 char budget 的單筆
        #   用 `continue` 跳過，**不可 `break`**。sampled 是 stratified 橫跨各 topic 的清單；
        #   若 avg_item 低估、cumulative 中途撞頂就 `break` → 把抽樣清單後半（= 後面 topic）
        #   全砍 → topic starvation 借屍還魂。改 `continue`：跳過「這一筆超標」但繼續檢查後面
        #   （後面可能較短能塞），保住尾段 topic 覆蓋。MAX_EVIDENCE_ITEMS 那條仍用 `break`
        #   （筆數硬上限，超過就沒必要再掃）。
        for eid in sampled:
            ic = _item_chars(pool[eid])
            if len(selected) >= MAX_EVIDENCE_ITEMS:
                break
            if selected and cumulative + ic > MAX_EVIDENCE_CHARS:
                continue   # 不 break：跳過超標單筆，讓後面（可能較短）的尾段 topic 仍有機會
            selected.append(eid)
            cumulative += ic

    dropped = len(valid) - len(selected)
    if dropped > 0:
        logger.warning(
            f"[LIVE RESEARCH][EVIDENCE CAP] {len(valid)} 筆 → 保留 {len(selected)} 筆 "
            f"(planned {len(planned_kept)} 全留 + stratified 補位; "
            f"~{cumulative} 字 / MAX_CHARS={MAX_EVIDENCE_CHARS} / MAX_ITEMS={MAX_EVIDENCE_ITEMS})，"
            f"drop {dropped} 筆（remaining stratified 未抽中者，非頭部截斷）"
        )
    # 輸出依 evidence_id 升冪（穩定、與下游 sorted(analyst_citations) 一致）
    return sorted(selected)


class LiveResearchOrchestrator(OrchestratorBase):
    """6-Stage 對話驅動研究流程控制器。"""

    def __init__(self, handler: Any, dry_run: bool = False):
        """
        Args:
            handler: LiveResearchHandler (inherits from DeepResearchHandler)
            dry_run: True = use mock agents, skip LLM calls
        """
        # Base class sets self.handler and self.logger
        super().__init__(handler)

        # Live Research-specific attributes
        self.features = CONFIG.reasoning_params.get("features", {})
        self.dry_run = dry_run or self.features.get("live_research_dry_run", False)
        self.mock_bab = self.features.get("live_research_mock_bab", False)
        if self.mock_bab:
            logger.info("[LIVE RESEARCH] mock_bab mode: Stage 1+2 use fixture ContextMap, Stage 3-6 use real LLM")

        if self.dry_run:
            self._setup_dry_run_agents()
        else:
            # Normal initialization
            associator_timeout = CONFIG.reasoning_params.get("analyst_timeout", 90)
            self.associator = AssociatorAgent(handler, timeout=associator_timeout)

        # Loop engine config
        self.max_bab_iterations = self.features.get("live_research_max_bab_iterations", 3)

        # Track F (sprint 2026-05-28) S-1: CriticAgent 單例 lazy holder
        # 多章共用同一 instance（LLM client / token counter / log context 統一）。
        # 走 critic_agent property accessor (下方) 自動 lazy init。
        self._critic_agent = None

        # D-2026-06-11 決策1: guard/grounding 故障即時旁白 per-run dedup flags
        # （照 loop_engine 三個 *_degraded_narrated flag 先例）。orchestrator
        # instance 每個 user request 重建（methods/live_research.py L196/L349），
        # instance 層 flag 即 per-run —— LLM 故障常貫穿整個 run，Stage 5 多章
        # 連續退化只播一次旁白；各章退化仍由 methodology note 逐章標示。
        # 各 flag 語意不同（GCU 退化 vs guard 其他環節故障 vs 發布審查故障 vs 抽取故障），不可共用。
        self._grounding_unavailable_narrated = False
        self._guard_error_narrated = False
        # R2（2026-07）：Stage 5 outline 定案後 special_element target 對不到章的 per-run 後衛旁白。
        self._special_element_unmatched_narrated = False
        # Task 1: 發布審查（第三層 publish gate）自身故障的 degrade-and-narrate per-run dedup。
        self._publish_gate_unavailable_narrated = False
        # Task 3: 抽取層（grounding 第 1 步 candidate 抽取）LLM 故障旁白。
        # pending 由 guard callback 同步 set（emit 是 async，callback 只能 set flag）；
        # narrated 確保 per-run 旁白只播一次（三 callsite 共用 emit helper）。
        self._grounding_extraction_failed_narrated = False
        self._grounding_extraction_failed_pending = False

        # 離線防呆燒錢上限（plan: lr-sse-reconnect-resume, 2026-06-15）：per-call guard。
        # orchestrator instance 每個 user request（continue/start）重建 → 此 flag 即
        # per-continue-call。確保「離線跨 checkpoint 計數」一次 continue 只 +1，即使同一
        # call 內穿越多個 durable boundary（如 _handle_stage_5_response → _run_stage_5）。
        self._offline_advance_counted_this_call = False

    @property
    def critic_agent(self) -> "Any":
        """Track F (sprint 2026-05-28) S-1: CriticAgent 單例（N 章共用）。

        Lazy init 避免 import circular（critic.py → schemas_live → orchestrator）。
        多章 publish gate call 共用一個 CriticAgent instance — LLM client /
        token counter / log context 一致。
        """
        if self._critic_agent is None:
            from reasoning.agents.critic import CriticAgent
            # LR 沿用 DR config key (critic_timeout=120)；fallback 120
            # 對齊 base.py:168 / critic.py，僅 config key 缺失時兜底。
            self._critic_agent = CriticAgent(
                self.handler,
                timeout=CONFIG.reasoning_params.get("critic_timeout", 120),
            )
        return self._critic_agent

    def _setup_dry_run_agents(self):
        """Dry-run mode: mock agents that return fixtures without LLM calls."""
        from unittest.mock import AsyncMock, MagicMock
        from reasoning.schemas_live import (
            ContextMapTopic,
            ContextMapSearchSeed,
            ContextMapDelta,
            AssociatorBuildOutput,
            AssociatorDeriveOutput,
            AssociatorRefineOutput,
            ConsistencyReview,
        )

        # Mock AssociatorAgent
        self.associator = MagicMock()

        mock_map = ContextMap(
            research_question="台灣綠能發展衝突",
            topics=[
                ContextMapTopic(
                    topic_id="t1",
                    name="土地使用衝突",
                    domain="能源政策",
                    relevance="core",
                    description="光電與農地爭議",
                ),
                ContextMapTopic(
                    topic_id="t2",
                    name="社區參與機制",
                    domain="治理",
                    relevance="core",
                    description="居民反對與溝通",
                ),
                ContextMapTopic(
                    topic_id="t3",
                    name="電網整合",
                    domain="基礎設施",
                    relevance="supporting",
                    description="再生能源併網挑戰",
                ),
            ],
            version=0,
            working_hypothesis="台灣綠能發展需要在土地使用和社區接受度間找到平衡",
        )

        self.associator.build_context_map = AsyncMock(
            return_value=AssociatorBuildOutput(
                context_map=mock_map,
                narration="建立了初始研究結構，涵蓋土地使用、社區參與和電網整合三個面向",
            )
        )

        self.associator.derive_search_plan = AsyncMock(
            return_value=AssociatorDeriveOutput(
                search_seeds=[
                    ContextMapSearchSeed(
                        query="台灣光電農地衝突",
                        target_topic_id="t1",
                        rationale="核心議題",
                        priority="high",
                    ),
                ],
                narration="計畫搜尋土地使用相關資料",
            )
        )

        refined_map = mock_map.model_copy(deep=True)
        refined_map.version = 1
        self.associator.refine_context_map = AsyncMock(
            return_value=AssociatorRefineOutput(
                updated_context_map=refined_map,
                delta=ContextMapDelta(from_version=0, to_version=1, reason="加入搜尋結果"),
                is_stable=True,
                narration="研究結構已穩定",
            )
        )

        logger.info("[LIVE RESEARCH] Dry-run mode: using mock agents")

    async def start(self, query: str, initial_items: list = None) -> LiveResearchStageState:
        """
        開始新研究。進入 Stage 1 並跑到第一個 checkpoint。

        Args:
            query: 使用者的研究問題
            initial_items: 初始 retrieval items（from handler.prepare()）

        Returns:
            LiveResearchStageState at Stage 1 checkpoint
        """
        logger.info(f"[LIVE RESEARCH] Starting: {query[:80]}...")

        # 離線跨 checkpoint 計數 per-call guard reset（plan: lr-sse-reconnect-resume）。
        self._offline_advance_counted_this_call = False

        state = LiveResearchStageState()
        state.advance_to_stage(1)

        state = await self._run_stage_1(state, query, initial_items)
        return state

    async def continue_from_checkpoint(
        self,
        state: LiveResearchStageState,
        user_message: str = "",
        auto_continue: bool = False,
        nav_action: str = "",   # ""=正常前進；"back_one"=退一階；"restart"=回 Stage 1
    ) -> LiveResearchStageState:
        """
        從 checkpoint 繼續研究。

        Args:
            state: 當前 LiveResearchStageState（從 session 讀取）
            user_message: 使用者的回覆訊息
            auto_continue: True = 使用者選了「你決定就好」
            nav_action: backward navigation 動作（"" / "back_one" / "restart"）

        Returns:
            更新後的 LiveResearchStageState（可能是下一個 checkpoint 或完成）
        """
        current_stage = state.current_stage

        if state.stage_status != "checkpoint":
            logger.warning(f"[LIVE RESEARCH] continue called but not at checkpoint: {state.stage_status}")
            return state

        logger.info(
            f"[LIVE RESEARCH] Continue from Stage {current_stage} "
            f"(auto={auto_continue}, nav='{nav_action}', msg='{user_message[:50]}')"
        )

        # ── Backward navigation: restart 確認 consume（plan: lr-backward-nav）──
        # 上一輪 restart 已 emit confirm prompt（set pending_restart_confirmation=True）。
        # 這輪 user 回覆（無 nav_action，但 pending flag 在）→ 在 nav 路由 + forward 路由
        # 之前消費。B1 紅線：含 meta is None fail-loud 分支（抄 export gate :5713-5723），
        # 置於 _looks_like_bounded_affirmative_shape 兜底之前。
        # 回傳 None = 段4 substantive，未消費 → fall through 到下方 forward 正常處理
        # 使用者實質訴求（不漏使用者任何一句話）。
        if getattr(state, "pending_restart_confirmation", False) and user_message.strip():
            consumed = await self._consume_restart_confirmation(state, user_message)
            if consumed is not None:
                return consumed

        # ── Backward navigation: nav_action 路由（plan: lr-backward-nav）──
        # nav_action 在 forward 路由前攔截。back_one/restart 走 reset_to_stage 後
        # emit 通用通知 checkpoint（不打 LLM），return；不走下方 forward +1 路徑。
        if nav_action == "back_one":
            return await self._navigate_back_one(state)
        if nav_action == "restart":
            return await self._navigate_restart(state, user_message, auto_continue)

        # Stage 3 特殊處理：dialogue loop 可能多輪（style analysis 確認）
        if current_stage == 3:
            state = await self._handle_stage_3_response(state, user_message, auto_continue)
            if state.stage_status == "checkpoint":
                # 仍在 dialogue loop 中，不 advance
                return state
            # Stage 3 完成，進 Stage 4
            state = await self._run_stage_4(state)
            return state

        # Stage 5 特殊處理：段落修改 loop 可能多輪
        if current_stage == 5:
            state = await self._handle_stage_5_response(state, user_message, auto_continue)
            if state.stage_status == "checkpoint":
                return state
            state = await self._run_stage_6(state)
            return state

        # 其他 stages：處理使用者回饋然後進入下一個 Stage
        next_stage = current_stage + 1

        if next_stage == 2:
            state = await self._handle_stage_1_response(state, user_message, auto_continue)
            if state.stage_status == "checkpoint":
                # Stage 1 dialog loop 仍在進行（user 改了 ContextMap，等下一輪 confirm）
                return state
            state = await self._run_stage_2(state)
        elif next_stage == 3:
            # plan: lr-consistency-pause-teeth（AR R2 BLOCKER 1）——pending_reframe
            # confirm round dispatch。Stage 2 drift adjust（下方 drift 分流的 adjust
            # 分支）emit reframe proposal 時已設 pending_reframe_json + 清 stage2_drift_paused，
            # 下一輪 user 確認 reframe 時 current_stage=2/next_stage==3。這裡必須先接
            # pending_reframe（與 _handle_stage_1_response 的 Step 0 dispatch 同款），
            # 否則會掉進下方 else 走 _handle_stage_2_response，把 user 的「OK」當 Stage 2
            # feedback 記下、reframe 永不 apply（R2 BLOCKER 1 根因）。此檢查必須在
            # stage2_drift_paused 檢查之前（adjust 已清 drift flag、狀態改由 pending 接管）。
            # Python 語意：if 命中則同塊 elif/else 絕不執行——pending_reframe 分支自己
            # 處理後續（回 checkpoint 就 persist+return；否則離開 if/elif/else block 依
            # 後續既有程式碼處理，不執行同塊 else）。
            if state.pending_reframe_json:
                state = await self._handle_pending_reframe(
                    state, user_message, target_stage=2
                )
                if state.stage_status == "checkpoint":
                    await self._persist_checkpoint_boundary(state)
                    return state
            # plan: lr-consistency-pause-teeth（AR R1 blocker）——drift checkpoint 分流。
            # stage2_drift_paused=True 代表本 Stage 2 checkpoint 是一致性漂移暫停，
            # banner 問了 user「繼續還是調整」，回覆要分類意圖後分流（不能只當 feedback
            # 記下來走 _handle_stage_2_response，那樣「調整方向」會被吞掉照原方向推進）。
            elif state.stage2_drift_paused:
                state = await self._handle_drift_checkpoint_response(
                    state, user_message, auto_continue
                )
                # drift 分流內部決定是否推進 / 回補 / 轉 reframe。若仍在 checkpoint
                # （adjust 轉 reframe pending，或解不出 reframe re-prompt），return 等下一輪。
                if state.stage_status == "checkpoint":
                    await self._persist_checkpoint_boundary(state)
                    return state
            else:
                state = await self._handle_stage_2_response(state, user_message, auto_continue)
                state = await self._run_stage_3(state)
        elif next_stage == 5:
            state = await self._handle_stage_4_response(state, user_message, auto_continue)
            if state.stage_status == "checkpoint":
                # Stage 4 redirect 等 user 重新回覆格式偏好
                return state
            state = await self._run_stage_5(state)
        elif next_stage > 6:
            state.complete_stage()
            logger.info("[LIVE RESEARCH] All stages complete")
        else:
            logger.warning(f"[LIVE RESEARCH] Unexpected next_stage={next_stage}")

        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    # ──── Backward Navigation（plan: lr-backward-nav, CEO 拍板 2026-06-19）────

    # 中文 stage label（user-facing；不含術語）
    _NAV_STAGE_LABELS = {
        1: "研究主題與架構",
        2: "資料蒐集",
        3: "文筆設定",
        4: "報告格式與大綱",
        5: "章節撰寫",
    }

    async def _navigate_back_one(self, state):
        """退回上一階段（Stage N→N-1）。reset_to_stage(N-1) + emit 通用通知 checkpoint。"""
        cur = state.current_stage
        if cur <= 1:
            # 邊界：Stage 1 無更早 stage → narration + 維持原 checkpoint
            await self._emit_narration("已經在最開始的階段了，沒有可以再退回的步驟。")
            await self._emit_checkpoint(stage=cur, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)
            return state
        target = cur - 1
        state.reset_to_stage(target)
        await self._emit_stage_change(target)  # 前端據此自動清更晚 stage section cards (#6)
        proposal = lr_copy.NAV_BACK_NOTICE
        state.set_checkpoint(proposal)
        await self._emit_checkpoint(stage=target, proposal=proposal)
        await self._persist_checkpoint_boundary(state)
        return state

    async def _navigate_restart(self, state, user_message, auto_continue):
        """Full restart（回 Stage 1，#3 復用 evidence）。#4 兩段式：先 emit confirm。"""
        # 第一輪：set pending flag + emit confirm prompt（章節未清）
        state.pending_restart_confirmation = True
        proposal = lr_copy.NAV_RESTART_CONFIRM_PROMPT
        state.set_checkpoint(proposal)
        await self._emit_checkpoint(stage=state.current_stage, proposal=proposal)
        await self._persist_checkpoint_boundary(state)
        return state

    async def _consume_restart_confirmation(self, state, user_message):
        """#4 restart 確認 consume：上一輪已 emit restart confirm，這輪 user 回答。

        沿 _handle_stage_5_response recollect consent gate 形態（token 確認 / abort 取消 /
        bounded affirmative 兜底 / substantive fall-through），**並補上 export gate
        :5713-5723 的 meta is None fail-loud 分支**（recollect gate 缺，export gate 有）。
        """
        state.pending_restart_confirmation = False  # 一律先清（這輪已消費）
        msg_norm = user_message.strip()
        # 段1：含確認 token 的快路徑（不打 LLM）→ 確認重新規劃。
        if _looks_like_recollect_confirm(msg_norm):
            logger.info("[LIVE RESEARCH] nav restart confirmed by user (token)")
            return await self._do_restart(state)
        # 段2：非 token → 打 abort 分類器。
        meta = await _classify_meta_intent(user_message, self.handler)
        # ── B1 fail-loud（抄 orchestrator.py:5713-5723）：meta is None = LLM 故障。
        #    必在 abort / bounded-affirmative 判定**之前**攔截。絕不放行清章節，
        #    停原地問確認、發系統端旁白；不可 silent fail（#21）、不可怪 user。
        if meta is None:
            logger.warning(
                "[LIVE RESEARCH] nav restart-confirm meta-intent classify failed "
                "(None) — stay at checkpoint, NOT clearing sections"
            )
            await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
            # 停在原 checkpoint（章節未清、stage 未退）；下一輪 user 再回覆。
            # 重設 pending flag 讓 user 仍可在系統恢復後再確認一次（不誤觸刪章）。
            state.pending_restart_confirmation = True
            state.set_checkpoint(lr_copy.NAV_RESTART_CONFIRM_PROMPT)
            await self._emit_checkpoint(
                stage=state.current_stage, proposal=state.checkpoint_prompt
            )
            await self._persist_checkpoint_boundary(state)
            return state
        if meta == META_INTENT_ABORT:
            # 取消 → 回原 checkpoint，不動章節
            logger.info("[LIVE RESEARCH] nav restart cancelled by user (abort)")
            await self._emit_narration("好的，那就維持現在的內容，不重新規劃。")
            state.set_checkpoint(state.checkpoint_prompt)
            await self._emit_checkpoint(
                stage=state.current_stage, proposal=state.checkpoint_prompt
            )
            await self._persist_checkpoint_boundary(state)
            return state
        # 段3：非 abort 的無 token 短肯定句 → 確認（與 recollect gate 段3 同形態）。
        #     meta 已保證非 None（上方 fail-loud 已攔），此處才安全兜底。
        if _looks_like_bounded_affirmative_shape(msg_norm):
            logger.info(
                "[LIVE RESEARCH] nav restart confirmed by user "
                f"(bounded affirmative shape, no token, meta={meta})"
            )
            return await self._do_restart(state)
        # 段4：其餘 substantive（如「改第3段」「再多查經濟面」）→ 不重新規劃，回傳 None
        # 讓 caller fall through 到下方既有 forward dispatch 正常路由處理 user 實質訴求
        # （「不漏使用者任何一句話」鐵律）。pending flag 已於本方法開頭清除。
        logger.info(
            "[LIVE RESEARCH] nav restart pending-confirm got substantive reply "
            f"(meta={meta}) — fall through to normal forward dispatch"
        )
        return None

    async def _do_restart(self, state):
        """確認後執行 restart：reset_to_stage(1) + emit Stage 1 通知 checkpoint（#3 復用 evidence）。"""
        state.reset_to_stage(1)
        await self._emit_stage_change(1)
        proposal = lr_copy.NAV_RESTART_NOTICE
        state.set_checkpoint(proposal)
        await self._emit_checkpoint(stage=1, proposal=proposal)
        await self._persist_checkpoint_boundary(state)
        return state

    # ──── Stage 1: 建立研究結構 ────────────────────────────────

    async def _run_stage_1(
        self, state: LiveResearchStageState, query: str, initial_items: list = None,
        seed_evidence_pool: dict = None, seed_counter: int = 0,
    ) -> LiveResearchStageState:
        """Stage 1: 跑 BABLoopEngine 建立研究結構。

        seed_evidence_pool / seed_counter：recollect 退回補搜時注入既有 pool 當 seed，
        engine 從 seed_counter+1 起分配新 ID 疊加（對齊 Stage 2 :1656-1703 寫法）。
        forward 首跑（Stage 1 一般進場）兩者為 None/0 = 行為不變。
        """
        # online substantive advance → 重置離線計數（plan 3d；離線 auto-advance 不 reset）。
        self._maybe_reset_offline_counters(state)
        await self._emit_stage_change(1)
        await self._emit_narration("開始分析研究主題，建立研究結構，可能會蒐集好幾分鐘，請耐心等候。")

        # mock_bab: 從 fixture 載入真實 ContextMap，跳過 BAB loop 的 LLM 呼叫
        if self.mock_bab:
            context_map = self._load_mock_bab_fixture()
            initial_map = context_map.model_copy(deep=True)
            state.context_map_json = context_map.model_dump_json()
            state.initial_context_map_json = initial_map.model_dump_json()
            state.executed_searches = ["[mock_bab] fixture loaded"]
            # 載入 fixture 內的 evidence_pool（mock 跟真實 cascade 對齊）
            state.evidence_pool_json = self._load_mock_evidence_pool_fixture()
            logger.info(
                f"[LIVE RESEARCH] mock_bab evidence_pool loaded: "
                f"{len(deserialize_evidence_pool(state.evidence_pool_json))} entries"
            )
            # 載入 fixture 內的 evidence_usage（chapter-override writer 必需）
            # chapter-override（fixture 章數，現行 Cayenne 3 章）路徑 writer 硬依賴
            # state.evidence_usage；缺此 → body 章「[本章資料不足]」空轉，over-block 測不到。
            state.evidence_usage = self._load_mock_evidence_usage_fixture()
            total_claims = sum(len(v) for v in state.evidence_usage.values())
            logger.info(
                f"[LIVE RESEARCH] mock_bab evidence_usage loaded: "
                f"{len(state.evidence_usage)} evidence ids, {total_claims} grounded claims"
            )
            # 載入 fixture 內的 book_outline（fixture 章數，現行 Cayenne 3 章）：
            # (a) 序列化進 state.book_outline_json → Stage 5 idempotent guard 看到已存在
            #     → skip 重 plan（省 LLM cost）；
            # (b) 同步寫進 format_specs["chapters"] → _resolve_chapter_source 走
            #     chapter-override 路徑（fixture 章數），不 fallback core_topics。
            mock_book_outline = self._load_mock_book_outline_fixture()
            state.book_outline_json = mock_book_outline.model_dump_json()
            if not state.format_specs:
                state.format_specs = {}
            state.format_specs["chapters"] = [
                {"name": ch.title, "outline": ch.brief}
                for ch in mock_book_outline.chapters
            ]
            logger.info(
                f"[LIVE RESEARCH] mock_bab book_outline loaded: "
                f"{len(mock_book_outline.chapters)} chapters → "
                f"state.book_outline_json + format_specs.chapters synced"
            )

            # 設計鎖定：mock_bab 路徑以 fixture book_outline 為 ground truth，
            # **不跑**初始格式抽取（_maybe_extract_initial_format 僅真實路徑呼叫），
            # 避免抽取覆蓋 E2E fixture 章節。regression 守門見
            # test_mock_bab_initial_format_uses_fixture_not_extraction。
            #
            # 刻意不同步（AR round 2 nit）：mock_bab proposal 由 ContextMap topics
            # 生成（_context_map_to_outline）；writer override chapters 來自
            # fixture book_outline — 兩者語意不同（研究結構概覽 vs 分章 override），
            # 非 bug，見 E2E-3 說明。

            outline = self._context_map_to_outline(context_map)
            proposal = f"## 研究結構提案\n\n{outline}\n\n這是我整理的研究結構，你覺得如何？需要調整嗎？"
            state.set_checkpoint(proposal)
            await self._emit_checkpoint(
                stage=1,
                proposal=proposal,
                context_map_summary=context_map_to_summary(context_map),
            )
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        # Format initial items if available
        initial_context = None
        if initial_items:
            initial_context = self._format_initial_items(initial_items)

        # Run B->A->B' loop
        engine = BABLoopEngine(
            associator=self.associator,
            handler=self.handler,
            max_iterations=self.max_bab_iterations,
            enable_consistency_monitor=self.features.get(
                "live_research_consistency_monitor", True
            ),
            dry_run=self.dry_run,
            seed_evidence_pool=seed_evidence_pool,
            seed_counter=seed_counter,
        )
        # Track A (sprint 2026-05-28): enable evidence_usage indexing
        # — Stage 1 全域 BAB loop 沒對應特定 topic, source_topic 默認 "global"
        engine.state = state
        # Track F (sprint 2026-05-28) I-3: Stage 1 global BAB invoke 標 stage_1
        # 給 ConsistencyDriftEntry audit log 用 — (stage, topic_id, iteration)
        # 三元組才是 unique audit key（Stage 2 per-topic invoke 各自有
        # max_iterations 內部循環，iteration overlap）。
        engine._current_stage = "stage_1"
        context_map = await engine.run_loop(
            query=query,
            initial_context=initial_context,
        )

        # 存 state
        state.context_map_json = context_map.model_dump_json()
        state.initial_context_map_json = engine.initial_context_map.model_dump_json()
        state.executed_searches = engine.executed_searches

        # 持久化 evidence pool（references master list 來源）
        # F (Codex #1+#8): defensive merge —— 以本輪 caller 傳入的 seed 為底，
        # engine 結果疊加。即使 engine 補搜全失敗、內部把 evidence_pool 重建成比
        # seed 少，也不丟失既有 seed（:94 只保證起點含 seed，不保證終點 ⊇ seed）。
        # forward 首跑（seed_evidence_pool=None）→ seed_base={} → 行為與原樣相同。
        seed_base = dict(seed_evidence_pool) if seed_evidence_pool else {}
        final_pool = {**seed_base, **engine.evidence_pool}
        if len(final_pool) < len(seed_base):
            # 理論上 dict-merge 後不可能小於 seed_base（merge 是 union）；此 log 為
            # forensic 兜底，若觸發代表 seed_base 本身異常（非 silent fail 紀律）。
            logger.warning(
                f"[LIVE RESEARCH] Stage 1 evidence merge anomaly: "
                f"final={len(final_pool)} < seed={len(seed_base)}"
            )
        state.evidence_pool_json = serialize_evidence_pool(final_pool)
        logger.info(
            f"[LIVE RESEARCH] Stage 1 evidence_pool persisted: "
            f"{len(final_pool)} entries"
        )

        # 產出提案
        # plan: lr-consistency-pause-teeth（掛載點 B，Stage 1）——BAB 若因研究方向漂移
        # 暫停（engine.paused_by_consistency），proposal 換成 drift banner 明說方向可能
        # 偏了、問要繼續還是調整，而非把半穩定 context_map 當正常「研究結構提案」端給
        # user（context_map + evidence_pool 已在上方 :1330-1348 persist，資料完整性無虞）。
        if engine.paused_by_consistency:
            proposal = self._build_drift_pause_proposal(engine.consistency_review)
            logger.info(
                "[LIVE RESEARCH] Stage 1 paused by Consistency Monitor "
                "(drift_level=%s); emitting drift banner checkpoint",
                getattr(engine.consistency_review, "drift_level", "?"),
            )
        else:
            outline = self._context_map_to_outline(context_map)
            proposal = f"## 研究結構提案\n\n{outline}\n\n這是我整理的研究結構，你覺得如何？需要調整嗎？"
            # 初始 query 格式 spec 抽取（傳輸層）：把 user 初始 prompt 內嵌的
            # 章節 / 字數 / 引用格式 / 特殊元素抽成結構化欄位、落進既有下游欄位，
            # 並在此 checkpoint 跟 user 確認一次。抽不到 → 零變化（不問、不落庫）。
            # dry_run 不打真 LLM。漂移暫停不做格式抽取（結構尚未穩定，抽取無意義）。
            if not self.dry_run:
                proposal = await self._maybe_extract_initial_format(
                    query, state, proposal
                )

        state.set_checkpoint(proposal)

        # P0 #5: build evidence_list from all topics in context_map
        all_evidence: list = []
        for _topic in context_map.topics:
            all_evidence.extend(self._build_topic_evidence_list(_topic, engine.evidence_pool))

        # 推送 checkpoint event
        # evidence_total 用 final_pool 完整筆數（顯示層 all_evidence 只是 per-topic 子集，
        # 前端據 total 標示「節選 vs 總量」，與下游 narration 同源）。
        await self._emit_checkpoint(
            stage=1,
            proposal=proposal,
            context_map_summary=context_map_to_summary(context_map),
            evidence_list=all_evidence,
            evidence_total=len(final_pool),
        )

        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    async def _maybe_extract_initial_format(
        self, query: str, state, proposal: str
    ) -> str:
        """抽初始 query 格式 spec、落庫、回傳（可能附確認句的）proposal。

        不可 silent fail（AR B3）：抽取 LLM 出錯 → 記 WARN **+ 發 user-visible 旁白**
        （降級為現行自由發揮，但告知 user 格式需求這次沒被解析），不中斷 Stage 1。
        空 spec → 原 proposal 不變（這是「user 沒指定格式」的正常情形，非故障，不旁白）。
        """
        try:
            spec = await self.associator.extract_initial_format_spec(query)
        except Exception as e:
            logger.warning(
                f"[LIVE RESEARCH] initial format spec extraction failed "
                f"(降級為自由發揮): {e}"
            )
            # AR B3：no-silent-fail — 故障≠user 需求被接受，補即時旁白告知。
            await self._emit_narration(
                lr_copy.INITIAL_FORMAT_EXTRACTION_FAILED_NARRATION
            )
            return proposal

        if spec is None or not spec.has_meaningful_spec():
            return proposal

        self._apply_initial_format_spec(state, spec)

        chapter_names = [c.name for c in spec.chapters]
        special = [
            {"type": e.type, "target_chapter": e.target_chapter}
            for e in spec.special_elements
        ]
        confirmation = lr_copy.initial_format_confirmation_line(
            chapter_names=chapter_names,
            total_word_count=spec.total_word_count,
            citation_style=spec.citation_style,
            special_elements=special,
        )
        return proposal + confirmation

    async def _handle_stage_1_response(
        self, state: LiveResearchStageState, user_message: str, auto_continue: bool
    ) -> LiveResearchStageState:
        """處理 Stage 1 checkpoint 的使用者回覆 — dialog loop。

        流程：
        0. UX-9：state.pending_reframe_json 非空 → reframe confirm round dispatch
           （confirm / cancel / adjust 三分支）
        1. auto_continue / 空訊息 → 直接 advance
        2. LLM intent parse 失敗連續 3 次 → 強制 advance（避免無限 loop）
        3. action=confirm → advance
        4. action=adjust + operations 為空 → narration + advance（user 沒實質訴求）
        5. action=adjust + operations 含 reframe_structure → emit detail-rich
           confirm proposal + set pending_reframe_json，**不立即 apply**（UX-9 D-1）
        6. action=adjust + operations 非空（incremental） → apply mutation → 重產 outline → 保持 checkpoint
           6a. mutation 後 ContextMap 變空 / 中途 exception → narration + 保持 checkpoint，
               不 mutate state.context_map_json（transactional safety）
        """
        # ── Step 0: UX-9 reframe confirm round dispatch ────────────
        if state.pending_reframe_json:
            # #8 fix: auto_continue means the user explicitly delegated the
            # decision ("讀豹決定"). With a pending reframe already proposed,
            # this is an authorised confirm — apply pending and advance.
            # This does NOT violate §4.3.6: that rule forbids silent advance
            # when there is no pending and LLM self-generated an op. Here the
            # user has already seen the proposal and is authorising it.
            # Implementation: pass user_message="OK" to route through
            # _looks_like_confirm_proceed_shortcut ("OK" is in
            # _CONFIRM_PROCEED_KEYWORDS) — confirmed confirm path without
            # extra LLM call. This borrows the existing confirm shortcut path
            # to avoid duplicating logic (auto_continue in pending = user-
            # authorised confirm).
            if auto_continue:
                logger.info(
                    "[LIVE RESEARCH] Stage 1: auto_continue with pending reframe "
                    "— treating as confirm (§4.3.6 compliant: user-authorised)"
                )
                return await self._handle_pending_reframe(
                    state, user_message="OK", target_stage=1
                )
            return await self._handle_pending_reframe(
                state, user_message, target_stage=1
            )

        if auto_continue or not user_message.strip():
            logger.info("[LIVE RESEARCH] Stage 1: auto-continue with current structure")
            state.failed_intent_parse_count = 0
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        context_map = ContextMap.model_validate_json(state.context_map_json)
        intent = await self._parse_stage_1_intent(user_message, context_map)

        # Track E (sprint 2026-05-28): user 在 Stage 1 reply 順帶提到時間訴求
        # → 寫進 state.time_constraint（N-6 紀律：state 是 single source of truth）
        if intent is not None and intent.time_range_extracted is not None:
            state.time_constraint = intent.time_range_extracted
            logger.info(
                "[LR Stage 1] time_constraint set from intent: "
                "start=%s, end=%s, raw=%r, user_selected=%s",
                intent.time_range_extracted.start_date,
                intent.time_range_extracted.end_date,
                intent.time_range_extracted.raw_phrase,
                intent.time_range_extracted.user_selected,
            )

        if intent is None:
            # #20 改善：intent is None = LLM API 失敗（系統端），非「user 講不清」。
            # 過去不分青紅皂白 failed_intent_parse_count += 1 + 3 次 force advance，
            # 會在 API 掛掉時把 user silent 推進到下一 stage（沒錢時的真實情境）。
            # 修：系統端文案 + 不累積計數（不餵 force-advance 安全閥）+ 重 emit checkpoint
            # 等 user 重試，不 force advance。
            #
            # 註：force-advance 安全閥（failed_intent_parse_count >= 3）保留供未來「真的反覆
            # parse 出但無有效 op」情境使用；目前 parser 對「真模糊」是回 dict（走下方
            # action=adjust + empty ops path），不會回 None，故 None 不該餵此計數。
            logger.warning(
                "[LIVE RESEARCH] Stage 1 intent parse returned None (LLM API fail), "
                "keep checkpoint, do not bump fail count / force advance"
            )
            await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
            # Bug fix 2026-05-16：必須重新 emit checkpoint，否則前端 reply UI 已被
            # continueLiveResearch 隱藏，user 無法再 reply。proposal 維持 state 既有 checkpoint
            # (ContextMap 未 mutate)，僅作 retry hint 用。
            await self._emit_checkpoint(
                stage=1, proposal=state.checkpoint_prompt
            )
            return state

        state.failed_intent_parse_count = 0  # 成功 parse → reset

        if intent.action == "confirm":
            logger.info("[LIVE RESEARCH] Stage 1: user confirmed structure")
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        # action == "adjust"
        if not intent.operations:
            # 兩條子路徑：clarifying_question 有 → 走澄清 dialog；無 → 既有「沒實質訴求 advance」
            clarifying = (intent.clarifying_question or "").strip()
            if clarifying:
                # 澄清 dialog 分支（empty-ops clarification dialog plan）—— 收編到共用
                # _emit_clarification helper（設計文件 §3，行為零漂移）。
                from reasoning.schemas_live import ClarificationRequest
                logger.info(
                    f"[LIVE RESEARCH] Stage 1: empty ops + clarifying_question — "
                    f"emit clarification checkpoint (summary='{intent.summary}', "
                    f"question='{clarifying[:60]}')"
                )
                return await self._emit_clarification(
                    ClarificationRequest(question=clarifying, stage=1), state
                )

            # 模糊但完全無實質訴求（既有 path 保留）
            logger.info(
                f"[LIVE RESEARCH] Stage 1: adjust intent but no operations "
                f"(summary='{intent.summary}'), treating as confirm"
            )
            await self._emit_narration("沒問題，目前的結構直接用。")
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        # UX-9: reframe_structure → 不立即 apply，存 pending + emit detail-rich confirm
        # 邊界規則（plan）：reframe 不可與 incremental 同陣列；若 LLM 違規，取第一個 reframe
        reframe_ops = [op for op in intent.operations if op.op_type == "reframe_structure"]
        if reframe_ops:
            if len(intent.operations) > 1:
                logger.warning(
                    f"[LIVE RESEARCH] Stage 1: reframe_structure + other ops 共 "
                    f"{len(intent.operations)} 個，採第一個 reframe，其餘忽略"
                )
            # C4 / C8 fix：Stage 1 reframe 句中順帶的引用格式 / 總字數，在 emit proposal
            # 前就寫進 state（持久化），這樣跨 confirm round 不會遺失。過去只抓
            # new_chapters，APA / 字數被吃掉 → writer 拿不到 → 內文全 [N] + 字數失準。
            self._apply_stage1_format_prefs(state, intent)
            return await self._emit_reframe_proposal(
                state, reframe_ops[0], context_map, intent.summary, target_stage=1
            )

        # Apply mutation
        # 改動 4 + 改動 5：pre-mutation snapshot 用於 removed_name_map 構建，
        # 因為 removed_topics 已從 mutated_cm 移除，需從 pre snapshot 取 name。
        context_map_pre_snapshot = context_map  # 此為 mutation 前 parse 出的物件
        logger.info(
            f"[LIVE RESEARCH] Stage 1: applying {len(intent.operations)} revision operations"
        )
        mutated_cm, delta, warnings = _apply_context_map_revisions(
            context_map, intent.operations, intent.summary
        )

        if mutated_cm is None:
            # 改動 3 + 改動 5：empty / exception abort path
            # caller 不取用任何 mutation 中途 cm 物件，僅保持 state.context_map_json 不變
            first_warning = warnings[0] if warnings else "未知錯誤"
            logger.warning(f"[LIVE RESEARCH] Stage 1: mutation rejected/abort — {warnings}")
            if "至少要保留" in first_warning:
                await self._emit_narration(
                    "至少要保留一個研究主題。請告訴我你想保留哪些議題？"
                )
            else:
                await self._emit_narration(
                    f"修改失敗：{first_warning}。請重新提供建議。"
                )
            # Bug fix 2026-05-16：必須重新 emit checkpoint，否則前端 reply UI 卡住。
            # 保持原 checkpoint state（context_map_json 未 mutate），讓 user 再 reply。
            await self._emit_checkpoint(
                stage=1, proposal=state.checkpoint_prompt
            )
            return state

        # Update state
        state.context_map_json = mutated_cm.model_dump_json()

        # 改動 4：構建 post-mutation name_map 與 removed_name_map（pre-mutation snapshot）
        outline = self._context_map_to_outline(mutated_cm)
        post_name_map = {t.topic_id: t.name for t in mutated_cm.topics}
        removed_name_map = {
            t.topic_id: t.name for t in context_map_pre_snapshot.topics
            if t.topic_id in delta.removed_topics
        }
        delta_summary = self._format_delta_summary(delta, post_name_map, removed_name_map)
        # 改動 1：warnings 不串接進 narration，僅 backend log（_apply_context_map_revisions 已 log）

        proposal = (
            f"## 研究結構提案（已根據你的建議調整）\n\n"
            f"本輪變更：{delta_summary}\n\n"
            f"{outline}\n\n"
            f"這次的結構你覺得如何？還需要再調整嗎？"
        )
        state.set_checkpoint(proposal)
        await self._emit_checkpoint(
            stage=1,
            proposal=proposal,
            context_map_summary=context_map_to_summary(mutated_cm),
        )
        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    def _format_delta_summary(
        self,
        delta,
        post_name_map: dict,      # 改動 4：post-mutation cm.topics 的 {id: name}
        removed_name_map: dict,   # 改動 4：pre-mutation snapshot 提供 removed topics 的 {id: name}
    ) -> str:
        """產生人話的 delta 摘要字串。

        UX-9: reframe_structure 場景由 caller 偵測 `is_reframe`，
        改用「整體重組為 N 章：A/B/C/...」格式。
        """
        # UX-9: reframe 場景（全砍重建 → pre 全在 removed，post 全在 added）
        # heuristic：removed_topics + added_topics 都很多且 modified=空
        if (
            delta.added_topics
            and delta.removed_topics
            and not delta.modified_topics
            and len(delta.added_topics) >= 2
            and len(delta.removed_topics) == len(removed_name_map)  # 全 pre 都被砍
        ):
            new_names = [post_name_map.get(tid, "?") for tid in delta.added_topics]
            return f"整體重組為 {len(new_names)} 章：{' / '.join(new_names)}"

        parts = []
        if delta.added_topics:
            names = [post_name_map.get(tid, "?") for tid in delta.added_topics]
            parts.append(f"新增 {len(names)} 個議題（{', '.join(names)}）")
        if delta.removed_topics:
            removed_names = [removed_name_map.get(tid, "?") for tid in delta.removed_topics]
            parts.append(f"移除 {len(removed_names)} 個議題（{', '.join(removed_names)}）")
        if delta.modified_topics:
            modified_names = [post_name_map.get(tid, "?") for tid in delta.modified_topics]
            parts.append(f"調整 {len(modified_names)} 個議題內容（{', '.join(modified_names)}）")
        return "、".join(parts) if parts else "微調"

    # ──── UX-9: Reframe confirm round helpers ─────────────────

    def _apply_stage1_format_prefs(self, state, intent) -> None:
        """把 Stage 1 reframe 句中順帶的引用格式 / 總字數寫進 state（C4 / C8 fix）。

        過去 Stage 1 reframe path 只抓 new_chapters，user 同句說的「APA」「總共 7000 字」
        被丟棄 → user_voice.citation_style 為 None → writer 用 numeric → 內文全 [N]；
        且總字數沒進 outline planner → 各章字數失準。

        與 Stage 4 path（line ~1561）對稱：citation_style 寫 user_voice、
        target_word_count 同時寫 user_voice + mirror 進 format_specs（outline planner
        prompt 從 format_specs.target_word_count 讀 budget）。

        per-chapter word_target 在 reframe apply 時由 _extract_chapters_from_ops 帶進
        format_specs.chapters，這裡只處理 top-level 偏好。
        """
        citation_style = getattr(intent, "citation_style", None)
        if citation_style is not None:
            state.user_voice.citation_style = citation_style
            logger.info(
                f"[LIVE RESEARCH] Stage 1 reframe: citation_style={citation_style} "
                f"captured → user_voice (C4 fix)"
            )
        total_wc = getattr(intent, "total_word_count", None)
        if isinstance(total_wc, int) and total_wc >= 1:
            state.user_voice.target_word_count = total_wc
            state.format_specs = dict(state.format_specs or {})
            state.format_specs["target_word_count"] = total_wc
            logger.info(
                f"[LIVE RESEARCH] Stage 1 reframe: total_word_count={total_wc} "
                f"captured → user_voice + format_specs (C8 fix)"
            )

    def _apply_initial_format_spec(self, state, spec) -> None:
        """把初始 query 抽出的 InitialFormatSpec 落進既有 format_specs / user_voice。

        沿用既有下游欄位形狀（零新 schema 欄位）：
        - spec.chapters → format_specs["chapters"]（{name, outline[, word_target]}）
          形狀對齊 _extract_chapters_from_ops 輸出 → _resolve_chapter_source 直接 honor。
        - spec.total_word_count → user_voice.target_word_count + mirror
          format_specs["target_word_count"]（對齊 _apply_stage1_format_prefs C8 寫法）。
        - spec.citation_style → user_voice.citation_style。
        - spec.special_elements → format_specs["special_elements"]
          （{type, target_chapter, description}，對齊 Stage 4 寫法）。

        每項抽不到（空 / None）→ 不寫該欄位（維持現行行為，保守 default）。
        時序紀律：本 helper 在 Stage 1 第一次 checkpoint **之前**呼叫；後續 user
        reply 的 reframe / Stage 4 dispatch 後寫覆蓋（user 最新意圖優先）。
        """
        if spec.chapters:
            chapters = []
            for ch in spec.chapters:
                entry = {"name": ch.name, "outline": ch.description or ""}
                # AR B1：InitialChapterSpec.word_target 真帶出（下游 outline planner
                # per_chapter_targets 消費）。None → 不寫該 key（同 _extract_chapters_from_ops）。
                wt = ch.word_target
                if isinstance(wt, int) and wt > 0:
                    entry["word_target"] = wt
                chapters.append(entry)
            state.format_specs = dict(state.format_specs or {})
            state.format_specs["chapters"] = chapters
            logger.info(
                f"[LIVE RESEARCH] initial format spec: {len(chapters)} chapters "
                f"→ format_specs.chapters"
            )

        if spec.total_word_count is not None:
            state.user_voice.target_word_count = spec.total_word_count
            state.format_specs = dict(state.format_specs or {})
            state.format_specs["target_word_count"] = spec.total_word_count
            logger.info(
                f"[LIVE RESEARCH] initial format spec: total_word_count="
                f"{spec.total_word_count} → user_voice + format_specs"
            )

        if spec.citation_style is not None:
            state.user_voice.citation_style = spec.citation_style
            logger.info(
                f"[LIVE RESEARCH] initial format spec: citation_style="
                f"{spec.citation_style} → user_voice"
            )

        if spec.special_elements:
            state.format_specs = dict(state.format_specs or {})
            state.format_specs["special_elements"] = [
                _serialize_special_element_for_state(e) for e in spec.special_elements
            ]
            logger.info(
                f"[LIVE RESEARCH] initial format spec: "
                f"{len(spec.special_elements)} special_elements → format_specs"
            )

    async def _emit_reframe_proposal(
        self,
        state: LiveResearchStageState,
        reframe_op: ContextMapRevisionOperation,
        context_map: ContextMap,
        summary: str,
        target_stage: int,
    ) -> LiveResearchStageState:
        """UX-9 Task 2.5：emit detail-rich reframe proposal + set pending_reframe_json。

        Not immediately apply — wait for user confirm in next round.

        Args:
            target_stage: 1 = Stage 1 entry；4 = Stage 4 mixed path entry
                         （兩條 path 都呼叫此 helper，保持 state.current_stage）

        Behavior:
        - 把 reframe_op JSON-serialize 存 state.pending_reframe_json
        - 用 op.proposal_markdown 作為 checkpoint proposal（detail-rich, D-6）
        - 若 proposal_markdown 為空（LLM 沒填）→ fallback 用簡單 list
        - emit checkpoint, state.current_stage 不變（target_stage 給 emit 用）
        """
        if not reframe_op.new_chapters:
            # 防呆：LLM 給了 reframe op 但沒給 chapters，落 narration 回 checkpoint
            logger.warning(
                "[LIVE RESEARCH] reframe op 缺 new_chapters，退回 narration "
                f"(target_stage={target_stage})"
            )
            await self._emit_narration(
                "我看到你想整體重組結構，但新章節清單還不夠清楚。"
                "可以再說明一下要幾章、每章標題大概是什麼嗎？"
            )
            await self._emit_checkpoint(
                stage=target_stage, proposal=state.checkpoint_prompt
            )
            return state

        # Detail-rich proposal markdown（D-6）— LLM 填的優先，沒填 fallback
        proposal = reframe_op.proposal_markdown.strip()
        if not proposal:
            logger.info(
                "[LIVE RESEARCH] reframe op proposal_markdown 為空，"
                "fallback 用 chapter name list"
            )
            chapter_lines = []
            for i, spec in enumerate(reframe_op.new_chapters, 1):
                if not isinstance(spec, dict):
                    continue
                name = spec.get("name", "?")
                desc = spec.get("description", "")
                line = f"### 第 {i} 章：{name}"
                if desc:
                    line += f"\n- {desc}"
                chapter_lines.append(line)
            new_rq = reframe_op.new_research_question or context_map.research_question
            proposal = (
                f"## 我準備重組為 {len(chapter_lines)} 章：\n\n"
                + "\n\n".join(chapter_lines)
                + f"\n\n**整體研究問題**：{new_rq}\n\n"
                + "確認這個結構嗎？或者哪一段要調整？"
            )

        # 存 pending — JSON-serialize 整個 reframe_op
        state.pending_reframe_json = reframe_op.model_dump_json()
        # Bug 2 (2026-05-18) root-fix：proposal markdown 寫獨立 field，**不再**
        # mutate `state.checkpoint_prompt`。以前 set_checkpoint(proposal) 把
        # reframe markdown 覆寫進 checkpoint_prompt，導致 Stage 4 entry confirm
        # path re-emit checkpoint 時拿到污染後的 prompt → 重複出現 reframe 提案。
        # 現在 checkpoint_prompt 保持為原 stage 的 prompt（Stage 1 outline 或
        # Stage 4 format dialog），不會被 reframe 提案污染。
        state.pending_reframe_proposal_markdown = proposal
        state.stage_status = "checkpoint"
        state.last_updated_at = datetime.now().isoformat()
        # state.current_stage 不變（caller 已經是 1 或 4，保持）
        logger.info(
            f"[LIVE RESEARCH] reframe proposal emitted "
            f"(target_stage={target_stage}, chapters={len(reframe_op.new_chapters)}, "
            f"summary={summary!r})"
        )
        await self._emit_checkpoint(
            stage=target_stage,
            proposal=proposal,
            context_map_summary=context_map_to_summary(context_map),
        )
        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    async def _handle_pending_reframe(
        self,
        state: LiveResearchStageState,
        user_message: str,
        target_stage: int,
    ) -> LiveResearchStageState:
        """UX-9 Task 2.5：處理 reframe confirm round 的 user reply。

        三分支：
        - confirm 短訊息（OK / 好 / 確認 ...） → 從 pending 取 op apply → clear pending
          - Stage 1 entry：advance to Stage 2
          - Stage 4 entry：reframe 完套用後保持 Stage 4（state.current_stage 不變），
            re-emit 既有 Stage 4 checkpoint（user 繼續處理格式偏好）
        - cancel 短訊息（取消 / 算了 ...） → clear pending → re-emit 原 stage checkpoint
        - 其他訊息 → 視為新訴求：clear pending + re-parse intent（recursive call
          進入正常 Stage 1 mutation 路徑，可能解出新 reframe 或 incremental）
        """
        if not state.pending_reframe_json:
            # 不應該到這（caller 已 check），保險起見 fallback
            logger.warning("[LIVE RESEARCH] _handle_pending_reframe called without pending")
            return state

        # FIX-5 (Cayenne #7, 2026-05-29)：confirm 關鍵字 shortcut — 在 LLM intent parse
        # 之前先攔「純 confirm（含 compound 如「確認。進入寫作。」）」。
        # 根因：compound confirm 句被 low-model classifier fallback 成 adjust → 又進
        # reframe 介面。對齊 Stage 5 export/continue keyword shortcut 的 pattern：
        # 短訊息 + 含 confirm 動詞 + 無 adjust/cancel veto 訊號 → 直接判 confirm，
        # 不打 LLM（省成本 + 防誤判）。confirm+adjust 混合句（含 veto 詞）仍交 LLM 細判。
        if _looks_like_confirm_proceed_shortcut(user_message):
            logger.info(
                f"[LIVE RESEARCH] reframe confirm keyword shortcut hit "
                f"({user_message.strip()[:40]!r}) — skip LLM classify (FIX-5)"
            )
            confirm_intent = "confirm"
        else:
            # R1 (2026-05-16)：用 LLM-based classifier 取代 _looks_like_confirmation /
            # _looks_like_cancel keyword exact-match（後者對「OK 就這樣」這類複合
            # confirm 句型過嚴）。CEO 拍板 — 不可 hardcode keyword。
            confirm_intent = await self._classify_confirmation_intent(user_message)

        # === Confirm path ===
        if confirm_intent == "confirm":
            logger.info(
                f"[LIVE RESEARCH] reframe confirmed by user "
                f"(target_stage={target_stage})"
            )
            try:
                reframe_op = ContextMapRevisionOperation.model_validate_json(
                    state.pending_reframe_json
                )
            except Exception as e:
                logger.warning(
                    f"[LIVE RESEARCH] pending_reframe_json deserialize fail: {e}"
                )
                state.pending_reframe_json = ""
                state.pending_reframe_proposal_markdown = ""
                await self._emit_narration(
                    "重組資料讀取失敗，先用目前結構繼續。"
                )
                state.complete_stage()
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state

            context_map = ContextMap.model_validate_json(state.context_map_json)
            pre_snapshot = context_map  # for removed_name_map
            mutated_cm, delta, warnings = _apply_context_map_revisions(
                context_map, [reframe_op],
                f"使用者確認整體重組（{len(reframe_op.new_chapters)} 章）",
            )
            if mutated_cm is None:
                first_warning = warnings[0] if warnings else "未知錯誤"
                logger.warning(
                    f"[LIVE RESEARCH] reframe apply rejected: {warnings}"
                )
                state.pending_reframe_json = ""
                state.pending_reframe_proposal_markdown = ""
                await self._emit_narration(
                    f"重組失敗：{first_warning}。請重新提供建議。"
                )
                # 退回 stage checkpoint
                await self._emit_checkpoint(
                    stage=target_stage, proposal=state.checkpoint_prompt
                )
                # plan: lr-pending-reframe-persist — apply 被拒後已清 pending，
                # boundary persist 落地（同型漏，ts=1·4 無 caller 兜底）。
                await self._persist_checkpoint_boundary(state)
                return state

            state.context_map_json = mutated_cm.model_dump_json()
            state.pending_reframe_json = ""
            state.pending_reframe_proposal_markdown = ""

            # P0-3 fix (2026-05-19, spec §4.7.6 reframe → writer 接線):
            # reframe apply 後同步 state.format_specs["chapters"]。沒這個，
            # _resolve_chapter_source 走 core_topics fallback → writer 寫舊
            # ContextMap N 章而非 user reframe 的 5 章。v15 Cayenne real persona
            # E2E P0-3 root fix。
            extracted_chapters = _extract_chapters_from_ops([reframe_op])
            if extracted_chapters:
                if not state.format_specs:
                    state.format_specs = {}
                state.format_specs["chapters"] = extracted_chapters
                logger.info(
                    f"[LIVE RESEARCH] reframe → format_specs.chapters synced: "
                    f"{len(extracted_chapters)} chapters (P0-3 fix)"
                )

            if target_stage == 1:
                # Stage 1 entry：reframe 套完即視為 user 認可結構 → advance to Stage 2
                logger.info(
                    "[LIVE RESEARCH] reframe applied (Stage 1 entry), "
                    "advancing past Stage 1"
                )
                # Bug fix 2026-05-16：Stage 1 entry confirm path 必須 emit
                # acknowledge narration，否則 user 體感是 silent jump to Stage 2。
                # 對齊 Stage 4 entry confirm path 既有「結構已重組」narration（line
                # 781-784），確保 Stage 1 / Stage 4 設計對稱。
                chapter_names = [
                    c.get("name", "") for c in reframe_op.new_chapters
                    if c.get("name")
                ]
                chapter_list = "、".join(chapter_names)
                await self._emit_narration(
                    f"好，就用這 {len(chapter_names)} 章結構"
                    f"（{chapter_list}）。"
                    f"接下來進入下一階段，蒐集每段所需的資料。"
                )
                state.complete_stage()
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state

            if target_stage == 2:
                # plan: lr-consistency-pause-teeth（AR R1 blocker）——Stage 2 drift adjust
                # 確認 reframe：reframe 後全新 core topics 全未 completed → 回 Stage 2
                # 重跑蒐集。suppress_consistency_pause=True 防 reframe 後立刻又漂移暫停
                # soft-lock（作用域＝這一趟）。
                chapter_names = [
                    c.get("name", "") for c in reframe_op.new_chapters if c.get("name")
                ]
                await self._emit_narration(
                    f"好，就用這 {len(chapter_names)} 章結構（{'、'.join(chapter_names)}）"
                    f"重新蒐集資料。"
                )
                state.stage2_drift_paused = False
                # AR R2 NIT：reframe apply 後是全新研究結構、所有 topic 未完成。
                # ContextMapTopic.topic_id 用 UUID 預設，新 topic 大概率不撞舊
                # completed_sections 的 ID，但為明確計清空——避免任何殘留舊 ID 讓
                # _run_stage_2 的 `if topic.topic_id in completed_sections: continue`
                # 誤跳過新結構的 topic。
                state.completed_sections = []
                return await self._run_stage_2(state, suppress_consistency_pause=True)

            # Stage 4 entry：reframe 套完，保持 Stage 4 等格式 reply
            # re-emit 原 Stage 4 checkpoint（user 繼續處理格式偏好）
            outline = self._context_map_to_outline(mutated_cm)
            post_name_map = {t.topic_id: t.name for t in mutated_cm.topics}
            removed_name_map = {
                t.topic_id: t.name for t in pre_snapshot.topics
                if t.topic_id in delta.removed_topics
            }
            delta_summary = self._format_delta_summary(
                delta, post_name_map, removed_name_map
            )
            await self._emit_narration(
                f"結構已重組（{delta_summary}）。\n\n{outline}\n\n"
                "現在回到格式偏好確認 — 你需要表格 / 列表 / 引用格式怎麼處理？"
            )
            # 維持 Stage 4 既有 checkpoint
            await self._emit_checkpoint(
                stage=4, proposal=state.checkpoint_prompt
            )
            # plan: lr-pending-reframe-persist — confirm target_stage=4 套 reframe 後
            # 清 pending + 換 context_map，boundary persist 落地（幽靈 pending 根解）。
            await self._persist_checkpoint_boundary(state)
            return state

        # === Cancel path ===
        if confirm_intent == "cancel":
            logger.info(
                f"[LIVE RESEARCH] reframe canceled by user "
                f"(target_stage={target_stage})"
            )
            state.pending_reframe_json = ""
            state.pending_reframe_proposal_markdown = ""

            if target_stage == 2:
                # plan: lr-consistency-pause-teeth（AR R2 BLOCKER 2）——Stage 2 drift adjust
                # 走到 reframe proposal 後 user 選 cancel。**不 re-emit 原 drift banner**：
                # adjust 分支已把 stage2_drift_paused 清成 False，若照既有 cancel path
                # re-emit state.checkpoint_prompt（＝原 drift banner），下一輪 user 回覆會
                # 走 next_stage==3 的 else 分支（drift flag 已 False、pending 已清）→
                # _handle_stage_2_response，banner 承諾「繼續或調整」但分流已丟＝dead banner。
                # 修法：drift 情境對 reframe 選 cancel 語意＝「不想這樣重組」＝「照原方向
                # 繼續」→ 直接回補未完成 core topic（與 continue 分支 cancel→繼續映射一致），
                # 閉環、不留 dead banner。suppress 防 reframe/回補反覆 soft-lock。
                # AR R3 SHOULD-FIX 2：防禦性清 stage2_drift_paused。正常路徑 adjust 分支
                # emit reframe 前已清此 flag，但萬一持久化 / 舊 state 讓 pending_reframe_json
                # 與 stage2_drift_paused 同時為 True（pending 在 caller 三層分流贏、被 cancel
                # 清掉），下方防呆 fallback 若帶著 stale drift flag return，下一輪同一
                # checkpoint 回覆又會誤走 drift 分流。此處在 narration / 回補前顯式清成 False，
                # 消滅這修復本要根除的 routing flag 殘留（低機率、但正是靶心）。
                state.stage2_drift_paused = False
                await self._emit_narration(
                    "好的，保留目前的研究方向，繼續補齊還沒查完的資料。"
                )
                if self._has_incomplete_core_topics(state):
                    return await self._run_stage_2(state, suppress_consistency_pause=True)
                # plan: lr-pending-reframe-persist — 防呆 return（無未完成 topic）自身
                # 補 persist（caller 有兜底、此為防禦；idempotent，不依賴 incoming stage_status）。
                await self._persist_checkpoint_boundary(state)
                return state   # 防呆：無未完成 topic → 讓 caller 走正常推進（drift flag 已清）

            # 既有：其他 target_stage（1 / 4）維持 re-emit 原 stage checkpoint（不動）
            await self._emit_narration(
                "好的，已取消整體重組，先保留目前結構。"
                "你可以給更具體的小修建議（例如「合併第 1 章和第 3 章」），"
                "或是不調整直接進入下一階段。"
            )
            # Re-emit 原 stage checkpoint（保持 user input）
            await self._emit_checkpoint(
                stage=target_stage, proposal=state.checkpoint_prompt
            )
            # plan: lr-pending-reframe-persist — cancel target_stage 1/4 清 pending 後
            # boundary persist 落地（否則下一輪 confirm 把已取消 reframe apply 進去）。
            await self._persist_checkpoint_boundary(state)
            return state

        # === Per-chapter edit path（FIX-4 / Cayenne #4）====================
        # user reply 不是 confirm 也不是 cancel — 先看是不是**針對 pending 提案中
        # 單一章節的微調**（「第 3 章改成只談丹麥德國」「國外案例那章把智利拿掉」）。
        # 過去這類訴求落入下方 adjust fall-through → 只回「我沒判斷該怎麼處理」→
        # user 卡 loop（Cayenne 三輪重打「拿掉智利案例」）。
        # 解析出單章 → mutate pending_reframe_op.new_chapters[i].description →
        # re-emit 更新後提案（**不整份 replace-all**），其他章不動。
        # 解析不出明確單章（None）→ fall through 回既有 adjust narration（safe）。
        try:
            pending_op_for_edit = ContextMapRevisionOperation.model_validate_json(
                state.pending_reframe_json
            )
        except Exception as e:
            logger.warning(
                f"[LIVE RESEARCH] per-chapter edit: pending deserialize fail: {e} "
                f"→ skip per-chapter path"
            )
            pending_op_for_edit = None

        if pending_op_for_edit is not None and pending_op_for_edit.new_chapters:
            edit = await self._parse_per_chapter_reframe_edit(
                user_message, pending_op_for_edit.new_chapters
            )
            if edit is not None:
                idx = edit["chapter_index"]
                new_desc = edit["new_description"]
                chapter = pending_op_for_edit.new_chapters[idx]
                chapter_name = chapter.get("name", f"第 {idx + 1} 章")
                old_desc = chapter.get("description", "")
                # mutate 單章 description（其他章 untouched）
                chapter["description"] = new_desc
                pending_op_for_edit.new_chapters[idx] = chapter
                # proposal_markdown 是 confirm round re-emit 用的 detail-rich 文字；
                # 單章改完讓 _emit_reframe_proposal 重新生 fallback（清掉舊 markdown，
                # 避免 re-emit 出來的提案仍顯示舊章描述）。
                pending_op_for_edit.proposal_markdown = ""
                state.pending_reframe_json = pending_op_for_edit.model_dump_json()
                logger.info(
                    f"[LIVE RESEARCH] reframe per-chapter edit applied: "
                    f"chapter[{idx}] {chapter_name!r} desc "
                    f"{old_desc[:30]!r} → {new_desc[:30]!r} (target_stage={target_stage})"
                )
                # 明確 narration：只有第 X 章被改，其他章不動
                await self._emit_narration(
                    f"好，只調整「第 {idx + 1} 章：{chapter_name}」，"
                    f"改成：{new_desc}\n"
                    f"其他章節維持不變。下面是更新後的提案 — "
                    f"確認就回「OK」，或繼續指定某一章要改。"
                )
                # re-emit 更新後提案（重建 pending checkpoint，保留 pending）
                context_map = ContextMap.model_validate_json(state.context_map_json)
                return await self._emit_reframe_proposal(
                    state, pending_op_for_edit, context_map,
                    summary=f"單章微調：第 {idx + 1} 章 {chapter_name}",
                    target_stage=target_stage,
                )

        # === Adjust path：user reply 不是 confirm 也不是 cancel，也不是單章微調。
        # P0-2 fix (2026-05-19, spec §4.3.6):
        # (a) narration 不引用 LLM-generated 數字當 user 訴求
        # (b) 不可 silent advance — 保留 pending reframe + re-emit checkpoint 等
        #     user 明示 OK / 取消 / 重給結構
        # 舊行為（已移除）：clear pending + recursive call → user reframe 訴求被
        # 丟給下一個 LLM 解析而訊息流上無痕跡，且 recursive 內 silent advance Stage 2。
        # v15 Cayenne real persona E2E P0-2 root fix。
        logger.info(
            f"[LIVE RESEARCH] reframe adjust reply received "
            f"(target_stage={target_stage}); keeping pending + re-emit checkpoint"
        )
        await self._emit_narration(
            "我看到你的回覆，但還沒判斷該怎麼處理 — 看起來不是「確認」也不是「取消」剛才的重組提案。\n"
            "請選一個：\n"
            "(1) 用剛才那版重組 — 回覆「OK」或「確認」\n"
            "(2) 不要重組 — 回覆「取消」\n"
            "(3) 提供新的整體結構訴求（例如「想要 6 章：X、Y、Z、...」）"
        )
        # Re-emit pending reframe checkpoint（保留 pending，讓 user 看到提案再選）
        await self._emit_checkpoint(
            stage=target_stage,
            proposal=state.pending_reframe_proposal_markdown or state.checkpoint_prompt,
        )
        return state

    # ──── Stage 2: Per-Section 資料策略 + 蒐集 ─────────────────

    async def _run_stage_2(
        self, state: LiveResearchStageState, *, suppress_consistency_pause: bool = False
    ) -> LiveResearchStageState:
        """Stage 2: 對每個 section 執行 focused B->A->B' loop。

        suppress_consistency_pause: True 時本趟關閉一致性監控（回補未完成 topic 那一趟用，
        防持續漂移每輪反覆暫停 soft-lock）。作用域＝這一次呼叫（函式參數、不寫 state、
        不持久化），下次任何 forward 進 Stage 2 恢復預設 False。全域 feature flag
        live_research_consistency_monitor 不受影響。（plan: lr-consistency-pause-teeth）

        engine constructor 的 enable_consistency_monitor 接線於本函式建 BABLoopEngine 處
        （suppress → False；否則沿全域 flag）。
        """
        self._maybe_reset_offline_counters(state)  # online substantive advance → reset（plan 3d）
        state.advance_to_stage(2)
        await self._emit_stage_change(2)
        await self._emit_narration("接下來進入資料蒐集階段，每個主題都需要搜尋並分析資料，可能會蒐集好幾分鐘，請耐心等候。")

        # mock_bab: 跳過 per-section BAB loop，直接用 Stage 1 的 ContextMap
        if self.mock_bab:
            logger.info("[LIVE RESEARCH] mock_bab: skipping Stage 2 BAB loops, using existing ContextMap")
            context_map = ContextMap.model_validate_json(state.context_map_json)
            core_topics = [t for t in context_map.topics if t.relevance == "core"]
            state.completed_sections = [t.topic_id for t in core_topics]

            # FIX-7 #8 補強 (2026-05-29): mock_bab 也是真實會產報告的 code path
            # （evidence 用 fixture），Stage 2 checkpoint 前一樣 emit consolidation
            # summary，讓 (a) UX 與 real-BAB 路徑一致，(b) 本地 mock_bab E2E 驗得到 #8。
            # context_map 與 evidence_pool 都由 Stage 1 mock_bab 從 fixture 持久化進
            # state（21 topics + evidence_pool），結構與 real path 同 ContextMap schema。
            mock_total_evidence = len(deserialize_evidence_pool(state.evidence_pool_json))
            await self._emit_stage2_consolidation(context_map, mock_total_evidence)

            proposal = "所有段落的資料都蒐集完了。需要補充哪個部分嗎？還是可以進入寫作準備？"
            state.set_checkpoint(proposal)
            # P0 #5: evidence_list for mock_bab path (pool already persisted from Stage 1 mock)
            _mb_pool = deserialize_evidence_pool(state.evidence_pool_json)
            _mb_evidence: list = []
            for _topic in context_map.topics:
                _mb_evidence.extend(self._build_topic_evidence_list(_topic, _mb_pool))
            # evidence_total 與 consolidation narration 同源 mock_total_evidence。
            await self._emit_checkpoint(
                stage=2, proposal=proposal, evidence_list=_mb_evidence,
                evidence_total=mock_total_evidence,
            )
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        context_map = ContextMap.model_validate_json(state.context_map_json)
        initial_map = ContextMap.model_validate_json(state.initial_context_map_json)

        # 對每個核心 topic group 執行 focused loop
        core_topics = [t for t in context_map.topics if t.relevance == "core"]

        # 跨 engine 累積 evidence_pool：從 state（Stage 1 已持久化）讀起點，每個
        # engine 跑完後 merge 回 existing_pool，counter 跟著遞增（§6.3 選項 1）。
        existing_pool = deserialize_evidence_pool(state.evidence_pool_json)
        counter = max(existing_pool.keys()) if existing_pool else 0

        for topic in core_topics:
            if topic.topic_id in state.completed_sections:
                continue

            # plan: lr-disconnect-midstage-persist — topic 開始前檢查 offline cap
            # （對齊 Stage 5 _run_stage_5 每段前檢查 :4562-4573）。這是給「topic 都還
            # 沒開始跑、但已經離線」的情境用（例如 resume 後重新進入某 topic 前 client
            # 仍未重連）；「topic 執行中被打斷」改由下方 engine.stopped_early 判斷
            # （D-7），兩者是互補、不重疊的兩個檢查點。達上限 → 標 capped + persist
            # 當下累積 + return（bounded burn，不無人看管燒錢）。未達 → 跑這個 topic。
            # 不呼叫 _check_connection（不靠 raise，D-6）。
            #
            # ⚠ SHOULD-FIX 4（R1 AR，誠實聲明不修）：這個 cap 檢查只**讀**
            # _offline_cap_reached(state)，不像 _persist_checkpoint_boundary
            # （:4850-4854）那樣做「increment → 判 capped」的計數動作。這代表如果
            # topic 一路都是「開始前檢查」這條路徑觸發 cap（而非跑完到
            # _persist_checkpoint_boundary 那個 durable boundary），
            # offline_checkpoint_advances 不會被 increment——Stage 2 的
            # offline-cap-early-return 只有 wall-clock 上限（state.offline_since
            # 累積時間）這條路徑實質生效，checkpoint-advance 上限不會被這裡觸及。
            # 這是已知限制，不在本次修復範圍內修正（需要更動 _persist_checkpoint_boundary
            # 的計數語意，影響面超出 Stage 2 斷線修復）。
            alive = getattr(self.handler, 'connection_alive_event', None)
            offline = alive is not None and not alive.is_set()
            if offline:
                self._mark_offline_since(state)
                if self._offline_cap_reached(state):
                    logger.warning(
                        f"[LIVE RESEARCH] Offline cap reached at Stage 2 topic "
                        f"'{topic.topic_id}'; stopping (reason={state.offline_cap_reason})"
                    )
                    state.offline_capped = True
                    # 落盤當下已累積（下方 topic 完成路徑同款 flush，這裡提早 return 前補一次）
                    state.context_map_json = context_map.model_dump_json()
                    state.evidence_pool_json = serialize_evidence_pool(existing_pool)
                    await self._persist_progress(state)
                    return state

            await self._emit_narration(f"開始蒐集「{topic.name}」相關的資料...")

            engine = BABLoopEngine(
                associator=self.associator,
                handler=self.handler,
                max_iterations=2,  # Per-section 迴圈較短
                # suppress_consistency_pause=True（回補趟）→ 強制關閉一致性監控，防
                # 持續漂移 soft-lock；否則沿全域 feature flag（plan: lr-consistency-pause-teeth）。
                enable_consistency_monitor=(
                    False if suppress_consistency_pause
                    else self.features.get("live_research_consistency_monitor", True)
                ),
                dry_run=self.dry_run,
                seed_evidence_pool=existing_pool,
                seed_counter=counter,
            )
            # Track A (sprint 2026-05-28): enable evidence_usage indexing
            # — Stage 2 per-topic loop, GroundedClaim.source_topic 帶 topic_id
            engine.state = state
            engine._current_topic_id = topic.topic_id
            # Track F (sprint 2026-05-28) I-3: Stage 2 per-topic BAB invoke 標 stage_2
            # 給 ConsistencyDriftEntry audit log 用 — (stage, topic_id, iteration)
            # 三元組才是 unique audit key（同一 topic 內 max_iterations=2 內部循環）。
            engine._current_stage = "stage_2"
            updated_map = await engine.run_loop(
                query=context_map.research_question,
                focus_topic_ids=[topic.topic_id],
                existing_context_map=context_map,
                existing_initial_map=initial_map,
                prior_executed_searches=state.executed_searches,
            )

            context_map = updated_map
            # Fix (2026-05-27): engine 以 prior_executed_searches=state.executed_searches
            # seed（loop_engine.py:154 複製全部 prior），跑完 engine.executed_searches
            # 已是「prior + 本 topic 新增」的累積 superset。此處必須 **assign**（覆蓋）而非
            # extend，否則每個 topic 都把全部 prior 再 append 一次 → 指數級重複累積
            # （5 topics 後 ~2^5 倍：prod 觀察 911 筆，實際 unique evidence 僅 23）。
            # 對齊 _run_stage_1 line 387 既有的 assignment pattern。
            state.executed_searches = list(engine.executed_searches)
            # engine.evidence_pool 已含本 topic 累積（正常收斂完整、或 stopped_early
            # 時的部分累積）；merge 後即為跨 engine 累積 superset（engine 已 dedup，
            # 直接覆蓋）。不論 stopped_early 與否都要 merge——即使只跑到一半，已經
            # 蒐集到的 evidence 也不該浪費（下一輪 resume 用這個當 seed 續跑）。
            existing_pool = engine.evidence_pool
            counter = engine._evidence_counter

            # plan: lr-disconnect-midstage-persist（D-7，R1 AR blocker 消化）——
            # 每個 topic 完成即落盤 + persist（root cause 修復核心）。舊 code 只在
            # loop 外 :2202-2203 更新一次 → 中途斷線全蒸發。改成每 topic 邊界把
            # context_map + evidence_pool 落進 state 並 _persist_progress（save fail
            # log+raise，不 silent）。
            #
            # 關鍵分流（D-7）：completed_sections 只在「正常收斂完成」時 append。
            # engine.stopped_early == True 代表這個 topic 是被 offline cooperative
            # break 打斷（Task 1），並非真正跑完——如果無條件 append，resume 時
            # `if topic.topic_id in state.completed_sections: continue` 會永久跳過
            # 這個實際沒跑完的 topic（BLOCKER B-1）。
            state.context_map_json = context_map.model_dump_json()
            state.evidence_pool_json = serialize_evidence_pool(existing_pool)

            # plan: lr-consistency-pause-teeth（選項乙）——offline 中斷（stopped_early）
            # 與一致性漂移暫停（paused_by_consistency）共用同一套「不標 completed + persist
            # + return」中斷生命週期。兩 flag 語意正交：stopped_early=純 offline、
            # paused_by_consistency=純漂移，此處 or 只共用資料完整性行為；下游處置
            # （offline 靜默 bounded-burn vs 漂移彈 drift banner 問 user）在 return 前按
            # 各自 flag 分流（Task 3 加 drift banner emit）。不設 stopped_early=True（不走
            # 選項甲）避免漂移借用 offline 的 _mark_offline_since wall-clock 副作用。
            if engine.stopped_early or engine.paused_by_consistency:
                if engine.paused_by_consistency:
                    # plan: lr-consistency-pause-teeth（掛載點 B）——漂移暫停：不是離線，
                    # 不記 offline_since（避免污染 offline wall-clock cap）。emit drift
                    # banner checkpoint + set_checkpoint 停在 Stage 2 等 user 確認方向。
                    # user 回覆由 continue_from_checkpoint next_stage==3 分支分流：
                    # continue → 回補未完成 core topic（Task 7）；adjust → 走 reframe
                    # 重設研究方向（Task 6，AR R1 blocker 消化）。stage2_drift_paused
                    # flag 讓 continue 分支知道「這是 drift checkpoint、要先分類 user 意圖」，
                    # 與正常 Stage 2 收斂 checkpoint 區分（後者不需 continue/adjust 分流）。
                    logger.warning(
                        f"[LIVE RESEARCH] Stage 2 topic '{topic.topic_id}' paused by "
                        f"Consistency Monitor (drift_level="
                        f"{getattr(engine.consistency_review, 'drift_level', '?')}); "
                        f"persisting accumulated evidence WITHOUT marking completed, "
                        f"emitting drift banner checkpoint"
                    )
                    await self._persist_progress(state)
                    proposal = self._build_drift_pause_proposal(engine.consistency_review)
                    state.set_checkpoint(proposal)
                    # AR R1 blocker 消化：標記本 checkpoint 為 drift pause，讓
                    # continue_from_checkpoint 走 continue/adjust 分流（Task 6）。
                    state.stage2_drift_paused = True
                    # AR R1 nit 消化：drift checkpoint 帶 evidence metadata，對齊正常
                    # Stage 2 checkpoint（:2317 附近）——沿用 _build_topic_evidence_list
                    # 逐 topic 建 evidence_list + evidence_total=len(existing_pool)。
                    # user 在 drift banner 上一併看到「已蒐集到 N 筆」的實況。
                    _drift_evidence: list = []
                    if state.evidence_pool_json:
                        _drift_pool = deserialize_evidence_pool(state.evidence_pool_json)
                        for _t in context_map.topics:
                            _drift_evidence.extend(
                                self._build_topic_evidence_list(_t, _drift_pool)
                            )
                    await self._emit_checkpoint(
                        stage=2, proposal=proposal,
                        evidence_list=_drift_evidence,
                        evidence_total=len(existing_pool),
                    )
                    await self._persist_checkpoint_boundary(state)
                    return state

                # 純 offline 中斷（既有 D-7 / SF-1 行為，不改）
                logger.warning(
                    f"[LIVE RESEARCH] Stage 2 topic '{topic.topic_id}' interrupted "
                    f"mid-execution by offline cooperative stop; persisting "
                    f"accumulated evidence WITHOUT marking completed (resume will "
                    f"re-enter this topic using accumulated evidence as seed)"
                )
                # plan: lr-disconnect-midstage-persist（R2 AR SHOULD-FIX 1，SF-1）——
                # topic 執行到一半才第一次偵測到離線時（state.offline_since 此前為
                # None），這裡也要記 wall-clock cap 的起點，跟上方「topic 開始前」
                # 那個檢查點（本函式迴圈頂端 `if offline: self._mark_offline_since(state)`
                # 那段）一致，否則要等下一次 resume 重新進入這個 topic、走到「topic
                # 開始前」檢查點時才會補寫，offline_since 起點會被推遲（若使用者在
                # 兩次呼叫之間有過短暫重連又斷線，wall-clock cap 的計時基準會因此
                # 失真）。_mark_offline_since 內部已有「已有值不覆寫」保護（既有件
                # A），這裡多呼叫一次是安全的 no-op（若已設過）。
                # 注意：這裡**只**補 wall-clock 起點記錄，不做
                # _persist_checkpoint_boundary 那套 checkpoint-advance increment 邏輯
                # ——那是 SHOULD-FIX 4（R1）誠實聲明範圍內的另一個限制，兩者相關但不同：
                # SF-1 修的是「offline_since 起點沒被寫」，SHOULD-FIX 4 講的是
                # 「checkpoint-advance 計數不會被這條路徑 increment」。修 SF-1 不等於
                # 修掉 SHOULD-FIX 4，SHOULD-FIX 4 的限制在這個分支依然成立。
                self._mark_offline_since(state)
                await self._persist_progress(state)
                # 提早收場：不繼續跑後續 topic（已離線，等同觸發了 D-3 的 cap 情境；
                # 不需要再另外判一次 _offline_cap_reached 才停——這個 topic 已經被
                # 打斷就代表當下已離線，該收場，語意上與 D-3 的「開始前 cap」互補
                # 而非重複）。
                return state

            state.completed_sections.append(topic.topic_id)
            await self._persist_progress(state)

            await self._emit_narration(f"「{topic.name}」的資料蒐集完成。")

        # Evidence Sufficiency Narration（模塊5 Task 2，通道 A）
        # BAB 全部 topic 跑完後評估 evidence pool 充分度，emit SSE narration 給 user
        # （純前端透明度，不進 writer prompt）。engine 為最後一輪 loop 的實例，其
        # evidence_pool 即跨 engine 累積 merge 後的完整 pool（line 1620 existing_pool = engine.evidence_pool）。
        # core_topics 為空時 engine 未綁定，跳過（與既有 existing_pool 取用一致）。
        if core_topics:
            await engine.emit_evidence_sufficiency_narration()

        # 更新 state
        state.context_map_json = context_map.model_dump_json()
        state.evidence_pool_json = serialize_evidence_pool(existing_pool)
        logger.info(
            f"[LIVE RESEARCH] Stage 2 evidence_pool after cross-engine merge: "
            f"{len(existing_pool)} entries (max id={counter})"
        )

        # FIX-7b (2026-05-29): Stage 2 完成後 emit consolidation summary。
        # Cayenne #8 根因：Stage 2 結束缺「最終研究地圖」，user 只看到一坨碎念後反問
        # 「所以現在的結構是啥」。這個 narration 是給 user 看的主訊息，格式化呈現：
        #   - 核心主題列點 + 各主題 evidence 筆數
        #   - 非 internal narration，用既有 _emit_narration 機制推送
        await self._emit_stage2_consolidation(context_map, len(existing_pool))

        proposal = "所有段落的資料都蒐集完了。需要補充哪個部分嗎？還是可以進入寫作準備？"
        state.set_checkpoint(proposal)

        # P0 #5: build evidence_list from all topics using persisted evidence_pool
        _stage2_evidence: list = []
        if state.evidence_pool_json:
            _stage2_pool = deserialize_evidence_pool(state.evidence_pool_json)
            for _topic in context_map.topics:
                _stage2_evidence.extend(
                    self._build_topic_evidence_list(_topic, _stage2_pool)
                )
        else:
            logger.warning(
                "[LIVE RESEARCH] Stage 2 checkpoint: evidence_pool_json empty, "
                "evidence_list will be empty in SSE payload"
            )

        # evidence_total 與 _emit_stage2_consolidation 的「共蒐集到 N 筆」同源 len(existing_pool)。
        await self._emit_checkpoint(
            stage=2, proposal=proposal, evidence_list=_stage2_evidence,
            evidence_total=len(existing_pool),
        )

        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    async def _emit_stage2_consolidation(self, context_map: ContextMap, total_evidence: int) -> None:
        """FIX-7b (2026-05-29): Stage 2 完成後 emit「最終研究地圖」consolidation summary。

        整理 ContextMap 中的核心主題清單 + 各主題 evidence 筆數，讓 user 在 Stage 2
        checkpoint 前一眼看懂研究結構是什麼（解決 Cayenne #8「所以現在的結構是啥」）。

        格式設計（保守版）：
          - 標題行說明總 evidence 筆數
          - 核心主題列點（name + domain + evidence 筆數）
          - 輔助主題列點（若有）
          - 尾行引導 user 進入下一步
        """
        core_topics = [t for t in context_map.topics if t.relevance == "core"]
        supporting_topics = [t for t in context_map.topics if t.relevance == "supporting"]

        lines = []
        lines.append(
            f"✅ 資料蒐集完成！共蒐集到 {total_evidence} 筆相關資料。以下是目前的研究地圖："
        )
        lines.append("")

        if core_topics:
            lines.append("**核心研究主題：**")
            for t in core_topics:
                evidence_count = len(t.evidence_ids)
                evidence_note = f"（{evidence_count} 筆資料）" if evidence_count > 0 else "（資料較少）"
                desc_note = f"：{t.description}" if t.description else ""
                lines.append(f"• {t.name}（{t.domain}）{evidence_note}{desc_note}")
        else:
            lines.append("（尚未確認核心主題）")

        if supporting_topics:
            lines.append("")
            lines.append("**輔助參考主題：**")
            for t in supporting_topics:
                evidence_count = len(t.evidence_ids)
                evidence_note = f"（{evidence_count} 筆資料）" if evidence_count > 0 else ""
                lines.append(f"• {t.name}（{t.domain}）{evidence_note}")

        if context_map.working_hypothesis:
            lines.append("")
            lines.append(f"**目前工作假設：** {context_map.working_hypothesis}")

        lines.append("")
        lines.append("如果研究方向符合預期，可以繼續進入寫作準備；若需要補充特定面向，請告訴我。")

        consolidation_text = "\n".join(lines)
        await self._emit_narration(consolidation_text)
        logger.info(
            f"[LIVE RESEARCH] Stage 2 consolidation emitted: "
            f"{len(core_topics)} core topics, {len(supporting_topics)} supporting topics, "
            f"{total_evidence} total evidence"
        )

    def _has_incomplete_core_topics(self, state: LiveResearchStageState) -> bool:
        """Stage 2 是否還有未完成的 core topic（漂移暫停 / 中途中斷留下的）。

        core topic 不在 completed_sections = 未查完。

        AR R1 SHOULD-FIX 2（no silent fail）：context_map_json 解析失敗**不放行 Stage 3**。
        Stage 2 checkpoint 必有合法 context_map，解析失敗＝資料毀損，靜默回 False 會讓壞
        context_map 放行進 Stage 3（放行不是保守）。改成 logger.exception 記錄 + 回 True，
        讓流程走回補分支不推 Stage 3；回補的 _run_stage_2 會再讀 context_map，真毀損會在
        那裡 fail-loud（不吞掉錯誤）。（plan: lr-consistency-pause-teeth）
        """
        from reasoning.schemas_live import ContextMap
        try:
            cm = ContextMap.model_validate_json(state.context_map_json)
        except Exception:
            logger.exception(
                "[LIVE RESEARCH] _has_incomplete_core_topics: context_map_json "
                "解析失敗（資料毀損）— 回 True 不放行 Stage 3，交回補路徑處理"
            )
            return True
        core_ids = [t.topic_id for t in cm.topics if t.relevance == "core"]
        return any(tid not in state.completed_sections for tid in core_ids)

    async def _handle_drift_checkpoint_response(
        self, state: LiveResearchStageState, user_message: str, auto_continue: bool
    ) -> LiveResearchStageState:
        """AR R1 blocker：Stage 2 drift banner checkpoint 的 user 回覆分流。

        banner 問「要照目前方向繼續，還是要調整回原本的重點？」——分類 user 意圖：
        - confirm / cancel（「繼續」「就這樣」「不要調整」）→ 清 drift flag、回補未完成
          core topic（_run_stage_2 suppress，防 soft-lock）；補齊後 _run_stage_2 走正常
          收斂設 checkpoint，下一輪才進 Stage 3。
        - adjust（帶新方向訴求）→ 解 reframe op 走既有 reframe confirm round（target_stage=2）。
          解不出 reframe op → 不 silent advance，re-emit drift checkpoint 請 user 更明確。
        - auto_continue / 空訊息 → 視為 confirm（user 選「你決定就好」= 照現況繼續回補）。

        （plan: lr-consistency-pause-teeth）
        """
        if auto_continue or not user_message.strip():
            intent = "confirm"
        else:
            intent = await self._classify_confirmation_intent(user_message)

        # === adjust：走 reframe 重設研究方向 ===
        if intent == "adjust":
            context_map = ContextMap.model_validate_json(state.context_map_json)
            parsed = await self._parse_stage_1_intent(user_message, context_map)
            reframe_ops = (
                [op for op in parsed.operations if op.op_type == "reframe_structure"]
                if parsed is not None else []
            )
            if reframe_ops:
                logger.info(
                    "[LIVE RESEARCH] Stage 2 drift adjust → reframe "
                    f"({len(reframe_ops[0].new_chapters)} chapters)"
                )
                state.stage2_drift_paused = False  # 改由 pending_reframe_json 接管狀態
                return await self._emit_reframe_proposal(
                    state, reframe_ops[0], context_map,
                    summary="使用者於漂移暫停要求調整研究方向",
                    target_stage=2,
                )
            # 解不出明確 reframe 訴求 → 不 silent advance（no silent fail）
            logger.info(
                "[LIVE RESEARCH] Stage 2 drift adjust 但未解出 reframe op — re-prompt"
            )
            await self._emit_narration(
                "了解你想調整方向。可以具體說明要調整成什麼重點嗎？"
                "例如想聚焦哪些主題、或整體改成幾個章節。"
            )
            # 保留 stage2_drift_paused=True，re-emit drift checkpoint 等 user 再說
            await self._emit_checkpoint(stage=2, proposal=state.checkpoint_prompt)
            return state

        # === confirm / cancel：照現況繼續、回補未完成 core topic ===
        logger.info(
            f"[LIVE RESEARCH] Stage 2 drift {intent} → continue, backfill incomplete topics"
        )
        state.stage2_drift_paused = False
        if self._has_incomplete_core_topics(state):
            return await self._run_stage_2(state, suppress_consistency_pause=True)
        # 無未完成 topic（理論上 drift 必有，防呆）→ drift 分支內顯式推進 Stage 3
        # 閉環，不依賴 fall-through（AR R3 SHOULD-FIX 1）。
        state = await self._run_stage_3(state)
        return state

    async def _handle_stage_2_response(self, state, user_message, auto_continue):
        """處理 Stage 2 checkpoint 的 user feedback。

        Plan: lr-user-voice-container-and-4-fixes (Fix I-2)
        - Empty / auto_continue：直接 complete_stage，不寫 feedback、不 emit narration
        - 非空訊息：把 feedback 寫進 state.user_voice.stage2_feedback（accumulate）+
          emit 誠實 narration（CEO OQ 1 拍板：繁體中文 user-friendly + 不撒謊
          「已記錄」這種 unverified claim）
        """
        if auto_continue or not user_message.strip():
            logger.info("[LIVE RESEARCH] Stage 2: auto-continue")
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        feedback_text = user_message.strip()
        logger.info(
            f"[LIVE RESEARCH] Stage 2: user feedback captured: {feedback_text[:100]}"
        )
        # Fix I-2: write to user_voice (CEO OQ 3：保留 round forward-compat，目前固定 "0")
        state.user_voice.stage2_feedback.append({
            "round": "0",
            "text": feedback_text,
        })
        # CEO OQ 1：繁體中文 + user-friendly + 不撒謊
        # 「記下來」≠「已記錄」（避免 unverified claim）；「寫稿階段會盡量採用」誠實
        # 反映 user_voice.stage2_feedback 目前無自動 consumer 但 Stage 5 user 可針對
        # 段落要求補充內容（writer 走 Fix I-1 revise path 確實能採用）。
        await self._emit_narration(
            "謝謝你的建議，我已經把它記下來，寫稿階段會盡量採用。"
        )
        state.complete_stage()
        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    # ──── Stage 3: 寫作準備 ────────────────────────────────────

    async def _run_stage_3(self, state: LiveResearchStageState) -> LiveResearchStageState:
        """Stage 3: Style Analysis dialogue loop。"""
        self._maybe_reset_offline_counters(state)  # online substantive advance → reset（plan 3d）
        state.advance_to_stage(3)
        await self._emit_stage_change(3)

        proposal = (
            "接下來進入寫作準備。你需要我幫忙試寫嗎？\n\n"
            "如果需要，可以提供一段你的文筆範本（貼一段你寫過的段落），"
            "我會分析文筆特徵來調整寫作風格。\n\n"
            "不提供也沒關係，我會用預設的學術寫作風格。"
        )
        state.set_checkpoint(proposal)
        await self._emit_checkpoint(stage=3, proposal=proposal)

        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    async def _handle_stage_3_response(self, state, user_message, auto_continue):
        """處理 Stage 3 回覆 — 支援 Style Analysis 多輪確認。"""
        if auto_continue or not user_message.strip():
            logger.info("[LIVE RESEARCH] Stage 3: skip style analysis")
            state.style_features_json = ""
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        # 判斷是否是對現有分析的確認/修正
        if state.style_features_json:
            # 「重新提供範本」按鈕（明確手勢）：前端送 sentinel user_message。
            # 設計鎖定（2026-06-05）：覆蓋整份分析這個不可逆動作鎖在明確按鈕之後，
            # 不靠 LLM intent。比對 sentinel → 清空既有分析 + 重問新範本。清空後，
            # 使用者下一則訊息因 `state.style_features_json` 為空，自然落入下方第一輪
            # else 入口，經 _classify_meta_intent 守門 + full _run_style_analysis +
            # o7 input-guard 重新分析（DRY：不在此重貼一份守門邏輯）。
            if user_message.strip() == STAGE3_NEW_SAMPLE_SENTINEL:
                logger.info(
                    "[LIVE RESEARCH] Stage 3: '重新提供範本' button → clear analysis, reprompt."
                )
                await self._emit_narration(
                    "好的，我們重來。請再貼一段你的文筆範本，我會重新分析。"
                )
                state.style_features_json = ""
                reprompt = (
                    "請提供一段你的文筆範本（貼一段你寫過的段落），我會重新分析文筆特徵。\n\n"
                    "不提供也沒關係，回覆「用預設就好」我會用預設的學術寫作風格。"
                )
                state.set_checkpoint(reprompt)
                # 重問 checkpoint **不**帶 show_new_sample_button：此時 style_features_json
                # 已空，沒有分析可覆蓋；按鈕本身就是「再給一次範本」，重複無意義。
                await self._emit_checkpoint(stage=3, proposal=reprompt)
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state  # 回到索取範本的 checkpoint，等使用者貼新範本

            # 已有分析結果 — 這是使用者對分析的回覆
            # 用 LLM intent parsing 判斷（CEO 決策：文筆分析是細緻的東西，keyword matching 不可接受）
            intent = await self._parse_style_confirmation_intent(user_message, state.style_features_json)
            if intent is None:
                # #21：intent-parse LLM API 失敗（系統端）→ 不可 silent confirm 推進、
                # 吞掉 user 可能的 adjust/redo 訴求。emit 系統端文案 + 保持 Stage 3
                # checkpoint（對齊 Stage 1 reframe / Stage 5 的 None 分流）。
                logger.warning(
                    "[LIVE RESEARCH] Stage 3 round-2 confirmation intent LLM call "
                    "failed (None), stay at Stage 3 checkpoint"
                )
                await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
                checkpoint_prompt = state.checkpoint_prompt or (
                    "這份文筆分析準確嗎？需要調整的話告訴我。"
                )
                state.set_checkpoint(checkpoint_prompt)
                await self._emit_checkpoint(stage=3, proposal=checkpoint_prompt)
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state  # 停原地，等 user 再回
            # round-3（Gemini）：移除 .get("action","confirm") 預設。intent 非 None 時
            # parser 已保證 action 合法（見 Task 2 Step 4 驗證），直接取；拿不到就是
            # parser 該回 None 沒回的 bug，寧可 KeyError 炸出來也不要 silent confirm。
            action = intent["action"]

            if action == "confirm":
                logger.info("[LIVE RESEARCH] Stage 3: style analysis confirmed")
                state.complete_stage()
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state
            else:
                # round-2 對話回饋只有兩種：confirm（上面）/ adjust（這裡）。
                # 設計鎖定（2026-06-05）：移除 conversational redo——「重新提供範本」
                # 這個覆蓋整份分析的不可逆動作改鎖在前端明確按鈕（sentinel 路徑），
                # 不再從對話文字觸發整碗重抽。intent parser schema 已收緊為
                # ["confirm","adjust"]，action 非 confirm 即 adjust，走 merge
                # （reconcile：保留未提及維度，只更新使用者點到的維度）。
                # #6 fix: merge user adjustment into existing analysis.
                # Do NOT re-analyse user_message as a new writing sample — that
                # produces 1 feature from sparse input and discards all others.
                # Instead feed (existing analysis + user request) to LLM and ask
                # it to produce a revised analysis preserving untouched dimensions.
                await self._emit_narration("了解，我來調整分析...")
                style_output = await self._run_style_analysis_merge(
                    state.style_features_json, user_message
                )
                state.style_features_json = style_output.model_dump_json()

                feedback_text = (
                    f"調整後的分析。整體語氣：{style_output.overall_tone}。\n\n"
                    f"文筆特徵：\n"
                )
                for f in style_output.features:
                    feedback_text += f"- **{f.dimension}**：{f.observation} → {f.instruction}\n"
                feedback_text += "\n這次準確嗎？"

                state.set_checkpoint(feedback_text)
                await self._emit_checkpoint(
                    stage=3, proposal=feedback_text, show_new_sample_button=True
                )
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state  # 保持 checkpoint，等下一輪確認

        else:
            # 首次回覆 — 先過窄版 meta-intent helper 判斷「流程指令 vs 實質範本」
            # （#16：原本無條件假設 user_message=文筆範本，把「用預設就好」誤當範本分析）。
            meta = await _classify_meta_intent(user_message, self.handler)
            if meta is None:
                # 不可 silent fail（#21）：LLM 失敗 → 系統端文案，不假設意圖、不分析、不推進。
                logger.warning(
                    "[LIVE RESEARCH] Stage 3 meta-intent classify failed (None), stay at checkpoint"
                )
                await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
                checkpoint_prompt = state.checkpoint_prompt or (
                    "請提供你的文筆範本，或回覆「用預設就好」。"
                )
                state.set_checkpoint(checkpoint_prompt)
                await self._emit_checkpoint(stage=3, proposal=checkpoint_prompt)
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state  # 停原地，等 user 再回
            if meta in (META_INTENT_SKIP, META_INTENT_ABORT):
                # 「用預設/跳過/不提供」或「算了不弄範本」→ 用預設學術風格往下
                # （Stage 3 無不可逆動作，abort 在此 = 不想提供範本 = 用預設）。
                logger.info(
                    f"[LIVE RESEARCH] Stage 3: meta-intent={meta} → skip style analysis, use default"
                )
                state.style_features_json = ""
                state.complete_stage()
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state
            # META_INTENT_SUBSTANTIVE → 實質範本（或混合句）→ 跑 Style Analysis
            if self.features.get("live_research_style_analysis", True):
                await self._emit_narration("收到你的文筆範本，讓我分析一下...")
                try:
                    style_output = await self._run_style_analysis(user_message)
                except StyleInputNotASampleError:
                    # O7: LLM 判定輸入是指令/閒聊而非範本（語意降級）→ 不假裝分析、
                    # 不覆蓋狀態。與下方 None guard（LLM 系統失敗，S2-2）是獨立通道。
                    logger.info(
                        "[LIVE RESEARCH] Stage 3 first reply: input judged not a "
                        "writing sample; degrading and holding at checkpoint."
                    )
                    await self._emit_narration(lr_copy.STYLE_INPUT_NOT_SAMPLE_FIRST_NARRATION)
                    checkpoint_prompt = state.checkpoint_prompt or (
                        "請提供你的文筆範本，或回覆「用預設就好」。"
                    )
                    state.set_checkpoint(checkpoint_prompt)
                    await self._emit_checkpoint(stage=3, proposal=checkpoint_prompt)
                    await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                    return state  # 停原地，不覆蓋 style_features_json
                if style_output is None:
                    # LLM 整個回空 → 不可 silent fail，也不寫半成品 style_features_json：
                    # 系統端文案 + 重 emit Stage 3 checkpoint + 停原地等 user 重試。
                    # 對齊上方 meta-intent 回 None 的 soft-fail pattern。
                    logger.warning(
                        "[LIVE RESEARCH] Stage 3 style analysis returned None "
                        "(LLM empty), stay at checkpoint for retry"
                    )
                    await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
                    checkpoint_prompt = state.checkpoint_prompt or (
                        "請提供你的文筆範本，或回覆「用預設就好」。"
                    )
                    state.set_checkpoint(checkpoint_prompt)
                    await self._emit_checkpoint(stage=3, proposal=checkpoint_prompt)
                    await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                    return state  # 停原地，等 user 再回
                state.style_features_json = style_output.model_dump_json()

                # 提案分析結果
                feedback_text = (
                    f"分析完成。整體語氣：{style_output.overall_tone}。\n\n"
                    f"觀察到的文筆特徵：\n"
                )
                for f in style_output.features:
                    feedback_text += f"- **{f.dimension}**：{f.observation} → {f.instruction}\n"
                feedback_text += "\n準確嗎？需要調整的話告訴我。"

                state.set_checkpoint(feedback_text)
                # 首輪成功分析後的 checkpoint 帶 show_new_sample_button=True：
                # 此後使用者已有一份分析，「重新提供範本」按鈕才有意義。降級 / soft-fail
                # 分支（請貼範本）不帶按鈕——此時還沒有分析可覆蓋。
                await self._emit_checkpoint(
                    stage=3, proposal=feedback_text, show_new_sample_button=True
                )
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state  # 等確認

        state.complete_stage()
        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    async def _run_style_analysis(self, sample_text: str) -> Optional[StyleAnalysisOutput]:
        """執行 Style Analysis（deferred import）。

        回傳 None = LLM 整個回空（high+low 兩級都失敗）。caller 必須檢查 None
        並走 soft-fail（emit lr_copy.LLM_UNAVAILABLE_NARRATION + 重 emit checkpoint），
        不可 silent fail。對齊 _classify_meta_intent 回 None 的 caller pattern。
        """
        if self.dry_run:
            from reasoning.schemas_live import StyleFeature
            return StyleAnalysisOutput(
                features=[
                    StyleFeature(dimension="句式結構", observation="多用短句", instruction="保持簡潔清晰"),
                    StyleFeature(dimension="用詞層次", observation="學術但易讀", instruction="使用精準術語"),
                    StyleFeature(dimension="段落節奏", observation="論點層次分明", instruction="每段一個核心論點"),
                ],
                overall_tone="學術嚴謹但不枯燥",
                sample_quality_note="dry-run fixture",
            )

        from reasoning.prompts.style_analysis import StyleAnalysisPromptBuilder
        from core.llm import ask_llm

        builder = StyleAnalysisPromptBuilder()
        # Note: StyleAnalysisPromptBuilder.build_style_analysis_prompt uses 'writing_sample'
        prompt = builder.build_style_analysis_prompt(writing_sample=sample_text)

        # Use a flat schema instead of model_json_schema() to avoid LLM echoing
        # the complex Pydantic schema structure (with $defs, title, etc.)
        flat_schema = {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    # minItems 與 StyleAnalysisOutput.features min_length=1 對齊
                    # （杜絕「LLM 看鬆 schema、validator 嚴」的根本不一致；prod blocker fix 2026-05-30）。
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {"type": "string"},
                            "observation": {"type": "string"},
                            "instruction": {"type": "string"},
                        },
                        "required": ["dimension", "observation", "instruction"],
                    },
                },
                "overall_tone": {"type": "string"},
                "sample_quality_note": {"type": "string"},
                # citation_format: 引用格式偏好（enum，由 Stage 3 prompt 引導 LLM 分類）
                # 設計為離散 enum 而非自由文字，避免 Writer 把樣式描述當字面字串輸出
                "citation_format": {
                    "type": "string",
                    "enum": ["author_year", "numeric", "footnote", "none"],
                },
                "input_is_writing_sample": {"type": "boolean"},
            },
            "required": ["features", "overall_tone"],
        }

        # Style analysis 是從文本提取特徵（extraction task），不需 high-level model
        response = await ask_llm(
            prompt,
            flat_schema,
            level="low",
            query_params=getattr(self.handler, 'query_params', {}),
            max_length=4096,
        )
        if not response:
            # 不可 silent fail，也不 raise 炸穿整條 LR pipeline：回 None，
            # 由 _handle_stage_3_response 走 soft-fail（emit 系統端文案 + 重 emit
            # checkpoint + 停原地等 user 重試）。對齊 _classify_meta_intent 回 None
            # 與 Stage 1 intent parse 回 None 的既有 pattern。
            self.logger.warning(
                "[LIVE RESEARCH] Style analysis: LLM returned empty response "
                "on both levels; returning None for caller soft-fail."
            )
            return None
        # Unwrap schema-wrapped response if needed: {type, properties, required} → extract properties
        if "features" not in response and "properties" in response:
            inner = response.get("properties", {})
            if isinstance(inner, dict) and "features" in inner:
                response = inner

        # O7: input-type 守門。LLM 判定輸入不是寫作範本（是指令/閒聊）→ raise sentinel，
        # 由呼叫端優雅降級（不可 silent fail），避免把短句當新範本整碗重抽。
        # 預設 True（缺欄位 = 當範本，不改正常路徑行為）。
        # 與上方 None 回傳（LLM 系統失敗，S2-2）是兩個獨立通道。
        if response.get("input_is_writing_sample", True) is False:
            self.logger.warning(
                "Style analysis: LLM judged input is NOT a writing sample "
                "(input_is_writing_sample=False); raising sentinel for caller to degrade."
            )
            raise StyleInputNotASampleError(
                "輸入被判定為非寫作範本（調整指令/閒聊）"
            )

        # sparse 防呆（prod blocker fix 2026-05-30）：sparse / 極短範本下 LLM 可能
        # 回**空 features（0 個）**。schema min_length=1 會對此硬炸 ValidationError，
        # 進而中斷整條 LR。不可 silent fail，但可優雅降級（CLAUDE.md）：塞一個最小
        # 合理的 fallback StyleFeature + 在 sample_quality_note 明確標示已降級。
        # 注意：1 個（含）以上 feature 走正常 validate（min_length 已改 1）。
        features = response.get("features")
        if not features:  # None 或 空 list
            from reasoning.schemas_live import StyleFeature

            degraded_note = "範本較短，可提取的文筆特徵有限，已採用通用寫作指引補足。"
            self.logger.warning(
                "Style analysis: LLM returned empty features for sparse sample; "
                "degrading gracefully to a generic fallback feature."
            )
            await self._emit_narration(
                "你提供的範本比較短，我能歸納出的文筆特徵有限，"
                "寫稿時我會以通用的清晰、嚴謹語氣為主。"
            )
            return StyleAnalysisOutput(
                features=[
                    StyleFeature(
                        dimension="整體風格",
                        observation="範本篇幅較短，僅能觀察到偏向清晰、直接的表達。",
                        instruction="以清晰、嚴謹、條理分明的語氣撰寫，避免冗詞與過度修飾。",
                    )
                ],
                overall_tone=response.get("overall_tone") or "清晰嚴謹",
                sample_quality_note=degraded_note,
                citation_format=response.get("citation_format", "numeric"),
            )

        return StyleAnalysisOutput.model_validate(response)

    async def _run_style_analysis_merge(
        self,
        existing_features_json: str,
        adjustment_message: str,
    ) -> StyleAnalysisOutput:
        """#6 fix: merge user adjustment into existing style analysis.

        Instead of re-analysing user_message as a new writing sample (which
        produces 1 feature from a sparse input and discards the rest), feed
        both the existing analysis AND the user's request to LLM and ask it
        to produce a revised analysis that:
        - preserves every dimension the user did not mention
        - updates only the dimension(s) the user explicitly addressed

        Falls back to the sparse-safe behaviour if LLM fails.
        """
        from core.llm import ask_llm
        from reasoning.schemas_live import StyleFeature

        if self.dry_run:
            # In dry_run, parse existing and return it unchanged (no LLM call).
            # The adjust branch in _handle_stage_3_response will re-emit the
            # checkpoint with the existing analysis, which is the correct
            # dry-run behaviour.
            return StyleAnalysisOutput.model_validate_json(existing_features_json)

        prompt = f"""你是文筆分析專家。使用者對一份已完成的文筆分析提出了微調訴求。

## 現有文筆分析
{existing_features_json}

## 使用者的調整訴求
{adjustment_message}

## 你的任務

輸出一份**修訂版文筆分析**，規則如下：
1. 使用者**沒有提到**的維度：原樣保留，observation 和 instruction 不變。
2. 使用者**明確指出**需要改的維度：只更新該維度的 observation / instruction，反映使用者的訴求。
3. 如果使用者的訴求暗示新增一個維度（現有分析沒有這個維度），可以加入。
4. **不可刪除**使用者沒有提到的既有維度。
5. overall_tone：維持不變，除非使用者明確要求改整體語氣。
6. citation_format：只在使用者明確提到引用格式時才修改；否則保留原值。

紀律：這是基於舊分析的**局部修改**，不是重新分析一份新文本。輸出的 features 數量必須 ≥ 現有 features 數量（除非合併了維度）。
"""

        flat_schema = {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {"type": "string"},
                            "observation": {"type": "string"},
                            "instruction": {"type": "string"},
                        },
                        "required": ["dimension", "observation", "instruction"],
                    },
                },
                "overall_tone": {"type": "string"},
                "sample_quality_note": {"type": "string"},
                "citation_format": {
                    "type": "string",
                    "enum": ["author_year", "numeric", "footnote", "none"],
                },
            },
            "required": ["features", "overall_tone"],
        }

        response = await ask_llm(
            prompt,
            flat_schema,
            level="low",
            query_params=getattr(self.handler, "query_params", {}),
            max_length=4096,
        )

        if not response:
            logger.warning(
                "[LIVE RESEARCH] _run_style_analysis_merge: LLM returned empty; "
                "returning existing analysis unchanged"
            )
            return StyleAnalysisOutput.model_validate_json(existing_features_json)

        # Unwrap schema-wrapped response if needed
        if "features" not in response and "properties" in response:
            inner = response.get("properties", {})
            if isinstance(inner, dict) and "features" in inner:
                response = inner

        # sparse fallback — same guard as _run_style_analysis
        features = response.get("features")
        if not features:
            logger.warning(
                "[LIVE RESEARCH] _run_style_analysis_merge: LLM returned empty "
                "features; returning existing analysis unchanged"
            )
            return StyleAnalysisOutput.model_validate_json(existing_features_json)

        return StyleAnalysisOutput.model_validate(response)

    async def _parse_style_confirmation_intent(self, user_message: str, style_features_json: str) -> Optional[dict]:
        """用 LLM 判斷使用者對文筆分析的回覆意圖。

        CEO 決策：文筆分析是非常細緻的東西，不可以用 keyword matching。

        Returns:
            意圖 dict（含 'action': confirm/adjust）；LLM API 失敗（空回應 /
            exception）或回壞 dict（缺 action / action 不在 enum）時回 None — caller
            須視為系統端失敗，emit lr_copy.LLM_UNAVAILABLE_NARRATION + 保持 Stage 3
            checkpoint，不可 silent confirm（#21；與 Stage 1/4/5 None 分流一致）。
        """
        if self.dry_run:
            return {"action": "confirm", "reason": "dry-run: always confirm"}

        from core.llm import ask_llm

        prompt = f"""你是一個意圖分析器。使用者正在確認一份文筆分析結果。

文筆分析結果：
{style_features_json}

使用者回覆：
{user_message}

判斷使用者的意圖，回傳 JSON：

- action（只有兩種，沒有第三種）：
  * "confirm"（user 明確且只表達接受分析正確、沒有提出任何調整，例如：
      「準確」/「對」/「沒問題」/「就是這樣」/「OK 繼續」/「分析得很準」）
  * "adjust"（user 提出任何形式的修改、補充、不滿或方向調整，例如：
      「第三項講得不夠準，應該是 X」/「整體 OK 但句式那項換成 Y」/
      「降低正式程度」/「再多一點口語感」/「不太對」/「換個方向試試」/
      「整體再嚴謹一點」）。**只要 user 表達了任何想改的訴求，一律歸 adjust**，
      由下游以「在現有分析上局部調整」處理，保留未提及的維度。

- adjustments: 如果 action 是 adjust，列出需要調整的維度和新指示（陣列）
- reason: 簡述判斷原因（繁體中文）

紀律：
- 任何指出特定維度錯誤、或要求微調具體面向的訊息 → adjust（不是 confirm）
- 「不對」「不太準」「整體再 X 一點」「換個方向」這類**對話式不滿**也 → adjust。
  系統會在現有分析上做局部調整（reconcile），不會整碗重抽。使用者若真的想換掉
  整份範本重來，介面上有獨立的「重新提供範本」按鈕負責，**不由你判斷**。
- confirm 僅適用於：user 完全沒提任何修改、純粹表達接受
- 如果分類錯誤（例如把 adjust 誤判為 confirm），caller 會直接 advance 階段、
  把 user 的調整訴求吃掉 — 寧可偏 adjust 也不要錯分 confirm
"""
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["confirm", "adjust"]},
                "adjustments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {"type": "string"},
                            "new_instruction": {"type": "string"},
                        },
                    },
                },
                "reason": {"type": "string"},
            },
            "required": ["action", "reason"],
        }
        try:
            response = await ask_llm(
                prompt,
                schema,
                level="low",
                query_params=getattr(self.handler, 'query_params', {}),
                max_length=4096,
            )
            if not response:
                # #21 紀律：LLM API 失敗（空回應）是系統端問題，不可偽造 confirm
                # silently 推進、吞掉 user 的 adjust/redo 訴求。回 None，由 caller
                # emit lr_copy.LLM_UNAVAILABLE_NARRATION + 保持 Stage 3 checkpoint
                # （對齊 Stage 1/4/5 的 None 分流）。
                logger.warning("[LIVE RESEARCH] _parse_style_confirmation_intent: ask_llm returned empty (None, fail-loud)")
                return None
            # Unwrap schema-wrapped response: {type, properties, required} → properties dict
            if "action" not in response and "properties" in response and isinstance(response["properties"], dict):
                response = response["properties"]
            # round-3（Gemini critical）：LLM 可能回「合法 JSON 但缺 action / action 不在
            # enum」（如 {"reason":"..."}）或非 dict 型態（如 list）。此時不可讓 caller 的
            # 取值 fall 回 confirm（silent-confirm 換條路重現）。一律當 parse-fail 回 None，
            # 由 caller None 分支處理（對齊空/exception）。isinstance 檢查同時防 list 型態
            # response.get 噴 AttributeError。
            if not isinstance(response, dict) or response.get("action") not in ("confirm", "adjust"):
                logger.warning(
                    f"[LIVE RESEARCH] _parse_style_confirmation_intent: invalid/missing "
                    f"action in response, treat as parse-fail (None): {response!r}"
                )
                return None
            return response
        except Exception as e:
            # #21 紀律：API exception → None（系統端），不偽造 confirm。
            logger.warning(f"[LIVE RESEARCH] _parse_style_confirmation_intent failed: {e}")
            return None

    async def _parse_stage_1_intent(
        self, user_message: str, context_map: ContextMap
    ):
        """用 LLM 解析使用者對 Stage 1 ContextMap 提案的回覆意圖。

        Returns:
            Stage1ParsedIntent 物件；LLM 失敗 / schema validate fail 時回 None
            （caller 視為「沒看懂」，emit fallback narration）。
        """
        from reasoning.schemas_live import Stage1ParsedIntent

        if self.dry_run:
            return Stage1ParsedIntent(action="confirm", operations=[], summary="dry-run: confirm")

        from reasoning.prompts.stage1_revision import Stage1RevisionPromptBuilder

        builder = Stage1RevisionPromptBuilder()
        prompt = builder.build_intent_parse_prompt(user_message, context_map)

        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["confirm", "adjust"]},
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op_type": {"type": "string", "enum": [
                                "merge_topics", "split_topic", "add_topic",
                                "remove_topic", "rename_topic",
                                "change_relevance", "change_description",
                                "reframe_structure",
                            ]},
                            "source_topic_ids": {"type": "array", "items": {"type": "string"}},
                            "merged_name": {"type": "string"},
                            "split_from_topic_id": {"type": "string"},
                            "split_into": {"type": "array"},
                            "new_topic_name": {"type": "string"},
                            "new_topic_description": {"type": "string"},
                            "new_topic_relevance": {"type": "string", "enum": [
                                "core", "supporting", "peripheral"
                            ]},
                            "new_topic_evidence_ids": {"type": "array", "items": {"type": "integer"}},
                            "target_topic_id": {"type": "string"},
                            "new_relevance": {"type": "string", "enum": [
                                "core", "supporting", "peripheral"
                            ]},
                            "new_description": {"type": "string"},
                            "new_name": {"type": "string"},
                            # reframe_structure (UX-9)
                            "new_chapters": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "relevance": {"type": "string", "enum": [
                                            "core", "supporting", "peripheral"
                                        ]},
                                        "word_target": {
                                            "type": "integer",
                                            "description": (
                                                "user 為該章指定的字數（如「前言~500」→ 500）；"
                                                "沒指定填 0。"
                                            ),
                                        },
                                    },
                                },
                            },
                            "new_research_question": {"type": "string"},
                            "proposal_markdown": {"type": "string"},
                        },
                        "required": ["op_type"],
                    },
                },
                "summary": {"type": "string"},
                "clarifying_question": {
                    "type": "string",
                    "description": (
                        "當 user reply 無法 mapping 到任何 op_type 且不是純 confirm 時，"
                        "在此填繁體中文問句針對 user 訴求具體追問。"
                        "純 confirm / 明確 ops 時留空字串。"
                    ),
                },
                "citation_style": {
                    "type": ["string", "null"],
                    "enum": ["author_year", "numeric", "footnote", "none", None],
                    "description": (
                        "user 順帶提到的引用格式：「APA」「（作者, 年份）」→ author_year；"
                        "「[1]」「數字編號」「IEEE」→ numeric；「腳註」→ footnote；"
                        "「不要引用」→ none；沒提 → null。"
                    ),
                },
                "total_word_count": {
                    "type": ["integer", "null"],
                    "description": (
                        "user 提到的整份報告總字數（如「總共約 7000 字」→ 7000）；沒提 → null。"
                    ),
                },
                # Track E (sprint 2026-05-28, N-7 三方同步): time_range_extracted
                # 必須同步出現在 Pydantic Stage1ParsedIntent + prompt 描述 + 此處 inline schema
                # 三處；少改任一邊 ask_llm structured output 不要求該欄位 → LLM 永遠不抽
                # → Stage 1 拿不到 time_range_extracted（假綠燈）。
                "time_range_extracted": {
                    "type": ["object", "null"],
                    "description": (
                        "若 user reply 順帶提到時間訴求（如「2024 後」「最近三年」「2020-2023」），"
                        "抽出 dict：start_date(YYYY-MM-DD 或 null) / end_date(YYYY-MM-DD 或 null) / "
                        "raw_phrase(user 原話片段) / user_selected(Stage 1 dialog 一律 true)；"
                        "user reply 沒提時間 → null"
                    ),
                    "properties": {
                        "start_date": {"type": ["string", "null"]},
                        "end_date": {"type": ["string", "null"]},
                        "raw_phrase": {"type": "string"},
                        "user_selected": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["action"],
        }

        try:
            response = await ask_llm(
                prompt,
                schema,
                level="low",
                query_params=getattr(self.handler, "query_params", {}),
                max_length=4096,
            )
            if not response:
                logger.warning("[LIVE RESEARCH] Stage 1 intent parse: ask_llm returned empty")
                return None
            # Unwrap schema-wrapped response: {type, properties, required} → properties dict
            if "action" not in response and "properties" in response \
                    and isinstance(response["properties"], dict):
                response = response["properties"]
            intent = Stage1ParsedIntent.model_validate(response)
            return intent
        except Exception as e:
            logger.warning(f"[LIVE RESEARCH] Stage 1 intent parse failed: {e}")
            return None

    async def _classify_confirmation_intent(self, user_message: str) -> str:
        """R1 (2026-05-16)：LLM-based confirmation intent classifier。

        取代 `_looks_like_confirmation` / `_looks_like_cancel` 兩個 keyword exact-match
        helper，因為它們對「OK 就這樣」「沒問題就這樣」這類複合 confirm 句型過嚴。
        CEO 拍板用 LLM 解析 — 不可 hardcode keyword list。

        Args:
            user_message: 使用者 reply 訊息

        Returns:
            "confirm" — user 明確接受目前 pending 訴求
            "cancel"  — user 想取消 pending 訴求，回到原狀
            "adjust"  — user 帶新訴求 / 不明確（safe default — 不誤套用）

        LLM 失敗 → 回 "adjust"（safe default，避免誤套用後不可逆 reframe）。

        dry_run 模式：short 訊息且包含經典 confirm/cancel keyword → 直接判定（純 unit
        test 加速 / 不打 LLM）。production 永遠走 LLM 解析。
        """
        msg = (user_message or "").strip()
        if not msg:
            # 空訊息 — 視為 adjust（caller 可能用 auto_continue 處理）
            return "adjust"

        # dry_run 加速分支：unit test 不要花 LLM cost
        # 只對「明顯」短句做 keyword fallback；其他訊息仍交給 caller 處理（adjust）
        if self.dry_run:
            stripped = msg.strip(" .,!?!?。，、~～").strip()
            confirm_kws_lower = {"ok", "好", "好的", "確認", "對", "沒問題", "就這樣", "可以", "go"}
            cancel_kws_lower = {"取消", "算了", "不要", "不用", "再想想", "cancel", "no", "nope"}
            if len(stripped) <= 10:
                if stripped.lower() in confirm_kws_lower:
                    return "confirm"
                if stripped.lower() in cancel_kws_lower:
                    return "cancel"
            return "adjust"

        # Production：LLM-based 分類
        prompt = (
            "你是一個意圖分類器。系統剛剛向使用者提出了一個提案（例如：「將研究結構重組為 5 章」），"
            "現在使用者回了一句訊息。請判斷使用者的意圖是哪一類：\n\n"
            f"使用者訊息：\n\"\"\"{msg}\"\"\"\n\n"
            "請回傳 JSON：\n\n"
            "- intent:\n"
            "  * \"confirm\"：使用者明確表達接受 / 同意目前提案，沒有要修改。\n"
            "    例如：「OK」「好」「就這樣」「OK 就這樣」「OK 確認」「沒問題就這樣」"
            "「好的，這樣可以」「嗯，可以這樣」「確認」「Sure」「行」\n"
            "  * \"cancel\"：使用者想取消這次提案，不要套用，回到上一狀態。\n"
            "    例如：「取消」「算了」「不要重組」「再想想」「先不要」「nope」\n"
            "  * \"adjust\"：使用者有新訴求 — 要改提案內容、或丟出新方向、或無法判斷的模糊回覆。\n"
            "    例如：「改成 3 章 A/B/C」「再加一章」「APA 引用、每段 500 字」"
            "「不對，我想要…」「等等，我重想一下」\n\n"
            "只回傳 JSON，不要其他文字。"
        )
        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["confirm", "cancel", "adjust"],
                },
            },
            "required": ["intent"],
        }
        try:
            response = await ask_llm(
                prompt,
                schema,
                level="low",
                query_params=getattr(self.handler, "query_params", {}),
                max_length=512,
            )
            if not response:
                logger.warning(
                    "[LIVE RESEARCH] _classify_confirmation_intent: "
                    "ask_llm returned empty, fallback adjust"
                )
                return "adjust"
            # Unwrap schema-wrapped response
            if "intent" not in response and "properties" in response \
                    and isinstance(response["properties"], dict):
                response = response["properties"]
            intent = response.get("intent", "adjust")
            if intent not in ("confirm", "cancel", "adjust"):
                logger.warning(
                    f"[LIVE RESEARCH] _classify_confirmation_intent: "
                    f"unknown intent {intent!r}, fallback adjust"
                )
                return "adjust"
            logger.info(
                f"[LIVE RESEARCH] _classify_confirmation_intent: "
                f"msg={msg[:40]!r} → {intent!r}"
            )
            return intent
        except Exception as e:
            logger.warning(
                f"[LIVE RESEARCH] _classify_confirmation_intent failed: {e}, "
                f"fallback adjust (safe default — 不誤套用)"
            )
            return "adjust"

    async def _parse_per_chapter_reframe_edit(
        self, user_message: str, new_chapters: List[dict]
    ) -> Optional[dict]:
        """FIX-4 (Cayenne #4)：解析 reframe pending 期間 user 對**單一章節**的微調。

        針對「第 3 章改成只談丹麥德國」「國外案例那章把智利拿掉」這類
        single-chapter edit — user 不是要 confirm / cancel，也不是要丟全新整體結構，
        而是要動 pending 提案中某一章的描述 / 約束。過去 _handle_pending_reframe
        的 adjust fall-through 對這類訴求只回「我沒判斷該怎麼處理」→ user 卡 loop。

        Args:
            user_message: user reply
            new_chapters: 目前 pending reframe op 的 new_chapters（[{name, description, ...}]）

        Returns:
            {"chapter_index": int (0-based), "new_description": str} — 成功解析出
              針對單一章節的微調訴求。
            None — 不是 single-chapter edit（解析不出明確單章 / LLM 失敗）→ caller
              fallback 回既有 adjust narration（safe，不誤改）。
        """
        msg = (user_message or "").strip()
        if not msg or not new_chapters:
            return None

        # 章節清單（給 LLM 對位用，1-based 顯示，回傳 0-based index）
        chapter_lines = []
        for i, spec in enumerate(new_chapters):
            if not isinstance(spec, dict):
                continue
            name = spec.get("name", "?")
            desc = spec.get("description", "")
            chapter_lines.append(f"  第 {i + 1} 章（index={i}）：{name} — {desc}")
        chapters_str = "\n".join(chapter_lines)

        # dry_run：不打 LLM（unit test 由 _parse override 或直接測 branch）
        if self.dry_run:
            return None

        prompt = (
            "你是一個意圖分析器。系統剛向使用者提出一份「整體重組為 N 章」的研究結構提案，"
            "使用者回了一句訊息。請判斷這句話是否為**針對其中單一章節的微調訴求**"
            "（例如改某一章的描述、為某一章加上取捨 / 約束，如「第 3 章改成只談丹麥德國」、"
            "「國外案例那章把智利拿掉」、「結論那章要寫具體建議」）。\n\n"
            f"目前提案的章節清單：\n{chapters_str}\n\n"
            f"使用者訊息：\n\"\"\"{msg}\"\"\"\n\n"
            "請回傳 JSON：\n"
            "- is_single_chapter_edit: true 僅當這句話明確只動**一個**章節；"
            "若是要整體重組（換掉全部章節 / 改章數）、純確認、純取消、或無法對應到"
            "清單中某一章 → false。\n"
            "- chapter_index: 該章的 0-based index（對照上方清單的 index=）。"
            "user 可能用序號（「第 3 章」→ index=2）或章名（「國外案例那章」→ 比對 name）。"
            "is_single_chapter_edit=false 時填 -1。\n"
            "- new_description: 套用 user 訴求後該章的**新描述**。"
            "**必須逐字保留 user 原話的精確約束詞**（地名取捨如「拿掉智利」、相似性如"
            "「與我國相似」、選材如「分屬不同能源」、具體性如「要寫地名 / 回饋金 / 法規名」），"
            "在原描述基礎上併入，不要用通用詞抹掉。is_single_chapter_edit=false 時填空字串。\n\n"
            "只回傳 JSON，不要其他文字。"
        )
        schema = {
            "type": "object",
            "properties": {
                "is_single_chapter_edit": {"type": "boolean"},
                "chapter_index": {"type": "integer"},
                "new_description": {"type": "string"},
            },
            "required": ["is_single_chapter_edit", "chapter_index", "new_description"],
        }
        try:
            response = await ask_llm(
                prompt,
                schema,
                level="low",
                query_params=getattr(self.handler, "query_params", {}),
                max_length=1024,
            )
            if not response:
                logger.warning(
                    "[LIVE RESEARCH] _parse_per_chapter_reframe_edit: "
                    "ask_llm empty → None (fallback adjust)"
                )
                return None
            if "is_single_chapter_edit" not in response and "properties" in response \
                    and isinstance(response["properties"], dict):
                response = response["properties"]
            if not response.get("is_single_chapter_edit"):
                return None
            idx = response.get("chapter_index", -1)
            new_desc = (response.get("new_description") or "").strip()
            if not isinstance(idx, int) or idx < 0 or idx >= len(new_chapters):
                logger.warning(
                    f"[LIVE RESEARCH] _parse_per_chapter_reframe_edit: "
                    f"index {idx!r} out of range (chapters={len(new_chapters)}) → None"
                )
                return None
            if not new_desc:
                logger.warning(
                    "[LIVE RESEARCH] _parse_per_chapter_reframe_edit: "
                    "empty new_description → None (不靜默清空章描述)"
                )
                return None
            logger.info(
                f"[LIVE RESEARCH] _parse_per_chapter_reframe_edit: "
                f"chapter_index={idx}, new_description={new_desc[:60]!r}"
            )
            return {"chapter_index": idx, "new_description": new_desc}
        except Exception as e:
            logger.warning(
                f"[LIVE RESEARCH] _parse_per_chapter_reframe_edit failed: {e} "
                f"→ None (fallback adjust，不誤改)"
            )
            return None

    # NOTE: `_parse_stage_4_intent` 已於 2026-05-19 TypeAgent refactor 完全移除。
    # 取代者：`_classify_stage_4_response`（產出 typed Stage4Response action），
    # caller 改走 typed dispatcher 路由。CEO 拍板 OQ-1：沒 backward compat tax。

    # ──── Stage 4: 格式確認 ────────────────────────────────────

    async def _run_stage_4(self, state: LiveResearchStageState) -> LiveResearchStageState:
        """Stage 4: 詢問格式需求。"""
        self._maybe_reset_offline_counters(state)  # online substantive advance → reset（plan 3d）
        state.advance_to_stage(4)
        await self._emit_stage_change(4)

        proposal = (
            "寫作風格確認完畢。關於格式：\n\n"
            "1. 每個段落需要什麼特殊格式嗎？（表格、列表、粗體重點等）\n"
            "2. 引用格式偏好？（APA、Chicago、或簡單的 URL 列表）\n\n"
            "告訴我你的偏好，或選擇「你決定就好」使用 Markdown + APA 引用。"
        )
        state.set_checkpoint(proposal)
        await self._emit_checkpoint(stage=4, proposal=proposal)

        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    async def _handle_stage_4_response(self, state, user_message, auto_continue):
        """處理 Stage 4 checkpoint 的使用者回覆 — TypeAgent typed action dispatcher。

        Plan: lr-typeagent-refactor (2026-05-19) — CEO 拍板 OQ-1：
        **完全取代**舊 `_parse_stage_4_intent + free-text dispatch`，改走
        `_classify_stage_4_response` 一次性解 typed Stage4Response、按 action enum
        嚴格路由。沒 backward compat tax。

        Routing table（per spec §4.13）：
        - auto_continue=True / 空訊息 → merge default + complete_stage
        - pending_reframe_json 非空 → 不打 classifier，dispatch 給
          `_handle_pending_reframe`（既有 confirm/cancel/adjust 三分支）
        - 其他訊息 → `_classify_stage_4_response` 解 typed action，按 action 路由
        """
        # === R7: invariant recovery + special_element pending 短路（全在 auto/blank 之前）===
        # (0) 互斥 invariant fail-loud recovery（兩個 pending 不得同時非空）。
        recovered = await self._enforce_pending_exclusivity(state)  # 清 special + SF3b narration，回是否 recover
        # (1) special_element pending 短路（放在 auto/blank 之前，blank/auto 不會誤 complete_stage）。B-order。
        if state.pending_special_element_json:
            return await self._handle_pending_special_element(state, user_message, auto_continue)
        # (2) SF-order（explicit 版）：雙 pending violation recover 清 special 後，若 reframe 仍
        #     pending 且本輪是 blank/auto，必須在此強制 route reframe，不落 auto/blank complete_stage
        #     （否則會把 reframe 也 complete 掉）。一般 reframe 短路仍保留在既有 :3252 原位（零漂移）。
        if recovered and state.pending_reframe_json:
            return await self._handle_pending_reframe(
                state, user_message, target_stage=4
            )

        # === auto_continue / 空訊息 short-circuit（不打 LLM）===
        if auto_continue or not user_message.strip():
            state.format_specs = self._merge_format_specs_default(state.format_specs)
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        # === UX-9: Reframe confirm round short-circuit ===
        # 既有 reframe 提案 pending → 走 _handle_pending_reframe（confirm/cancel/adjust）
        # 此 path 在 typed dispatcher 之前，保留 reframe 既有 LLM 分類器（不重複造輪）
        if state.pending_reframe_json:
            return await self._handle_pending_reframe(
                state, user_message, target_stage=4
            )

        # === TypeAgent typed action dispatcher ===
        from reasoning.schemas_live import Stage4ResponseAction

        response = await self._classify_stage_4_response(state, user_message)
        action = response.action

        if action == Stage4ResponseAction.confirm_format:
            # v8 Bug 2 partial root fix — typed confirm 不 re-emit checkpoint
            state.pending_format_confirmation = False
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        if action == Stage4ResponseAction.confirm_both:
            # 兩個 pending 都 confirm — 但 reframe path 已在上方 short-circuit；
            # 走到這代表 pending_reframe_json 為空、只剩 format。退化為 confirm_format。
            state.pending_format_confirmation = False
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        if action == Stage4ResponseAction.confirm_reframe:
            # pending_reframe_json 空時 user 卻 reply confirm — fallback narration
            logger.warning(
                "[LIVE RESEARCH] confirm_reframe without pending_reframe_json — "
                "narrate ambiguous"
            )
            await self._emit_narration(
                "目前沒有等待確認的結構提案。請具體說明你的訴求。"
            )
            # 同源病 root fix：停在 checkpoint → 重 emit 讓前端 reply UI 恢復
            await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
            return state

        if action == Stage4ResponseAction.cancel_reframe:
            # 同上 — pending_reframe_json 為空已過濾，這條也是 fallback
            logger.warning(
                "[LIVE RESEARCH] cancel_reframe without pending_reframe_json"
            )
            await self._emit_narration("目前沒有可取消的提案。")
            # 同源病 root fix：停在 checkpoint → 重 emit 讓前端 reply UI 恢復
            await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
            return state

        if action == Stage4ResponseAction.adjust_format:
            # 改格式偏好 → 寫 format_specs + advance
            fc = response.format_content
            assert fc is not None  # validator 保證
            state.format_specs = self._merge_format_specs_user(
                state.format_specs,
                fc.format_spec_extracted or user_message,
                special_elements=[e.model_dump() for e in fc.special_elements],
            )
            if fc.citation_style_extracted is not None:
                state.user_voice.citation_style = fc.citation_style_extracted
            # Blocker A (2026-05-19) root fix：寫進 user_voice typed channel +
            # mirror 到 format_specs 供 outline planner prompt 讀取
            if fc.target_word_count is not None:
                state.user_voice.target_word_count = fc.target_word_count
                state.format_specs = dict(state.format_specs or {})
                state.format_specs["target_word_count"] = fc.target_word_count
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        if action == Stage4ResponseAction.add_special_element:
            # 純補 element（不改章節結構、不 advance）— 寫進 format_specs，
            # 保持 checkpoint 狀態等 user confirm format 全套
            fc = response.format_content
            if fc is None or not fc.special_elements:
                # add_special_element 但 payload 空 — fallback narration + 重 emit checkpoint
                await self._emit_narration(
                    "我沒看清楚你想加什麼 element，可以具體說明 type 與位置嗎？"
                )
                await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
                return state
            import json
            state.format_specs = dict(state.format_specs or {})
            # 暫定章名清單（與 classifier 同源 _resolve_chapter_source）
            try:
                _cm = ContextMap.model_validate_json(state.context_map_json)
                _src, _ = self._resolve_chapter_source(_cm, state.format_specs)
                _chapter_names = [
                    (c.get("name", "") if isinstance(c, dict) else getattr(c, "name", ""))
                    for c in _src
                ]
            except Exception:
                _chapter_names = []

            # SF2b（Codex）：既有 list 拷貝時也逐項 serialize（不把舊 session 髒 transient 寫回）
            resolved_elements = [
                _serialize_special_element_for_state(e)
                for e in (state.format_specs.get("special_elements") or [])
                if isinstance(e, dict) or hasattr(e, "model_dump")
            ]
            confirm_pending = []   # LLM clear、待 user 確認（完整 element context）
            clarify_pending = []   # uncertain/對不到，待完整 clarification（完整 element context）
            for e in fc.special_elements:
                d = e.model_dump()
                raw_target = d.get("target_chapter", "")
                layer1 = _resolve_target_chapter_layer1(raw_target, _chapter_names)
                if layer1 is not None:
                    # 第一層命中（exact/唯一/明說全章「」）→ 直接定位、走 serializer 寫入（B3）
                    d["target_chapter"] = layer1
                    resolved_elements.append(_serialize_special_element_for_state(d))
                    continue
                # 第二層：讀 LLM 語意判斷（classifier 已順帶判、在 d 內）
                llm_title = (d.get("resolved_chapter_title") or "").strip()
                conf = d.get("resolution_confidence")
                pend = {"type": d.get("type", ""), "description": d.get("description", ""),
                        "raw_target": raw_target, "resolved_title": ""}
                if conf == "clear" and llm_title and llm_title in _chapter_names:
                    pend["resolved_title"] = llm_title
                    confirm_pending.append(pend)      # 待 user 確認（完整存，B1）
                else:
                    clarify_pending.append(pend)      # uncertain/對不到（完整存，B4）
            # resolved 直接寫入（serializer 已 strip transient，B3）
            state.format_specs["special_elements"] = resolved_elements
            # 兩個 pending 旗標語意正交（Codex should-fix，明確化）：
            #  - pending_format_confirmation（既有布林）＝「Stage 4 有格式偏好待 user 於 checkpoint 一併確認」
            #    的**寬鬆總旗標**，收到 add_special_element 就 True，走既有 stage-4 confirm round。
            #  - pending_special_element_json（新，本 plan）＝「有 special_element target **落灰區**、
            #    正等 user 回答一個**具體澄清問句**」的**精確狀態機**。只在 confirm/clarify 分支才寫。
            # 兩者可並存不衝突：pending_special_element_json 非空時，入口短路**優先**接管
            # user 下一句（進 _handle_pending_special_element），不會落到泛用 pending_format_confirmation
            # round；special pending 清空後，才回到既有 format confirmation 流程。
            state.pending_format_confirmation = True

            from reasoning.schemas_live import ClarificationRequest
            # confirm 優先（clear 單選確認）；有 confirm 則 clarify 併入 pending 下一輪處理
            if confirm_pending:
                state.pending_special_element_json = json.dumps({
                    "kind": "confirm",
                    "elements": confirm_pending,
                    "clarify_backlog": clarify_pending,   # confirm 完再處理 clarify
                    "chapter_names": _chapter_names,
                }, ensure_ascii=False)
                await self._persist_checkpoint_boundary(state)   # B4：pending 已完整存好才 persist
                return await self._emit_clarification(
                    ClarificationRequest(
                        question=lr_copy.special_element_confirm_question(
                            [p["resolved_title"] for p in confirm_pending]
                        ),
                        stage=4,
                    ),
                    state,
                )
            if clarify_pending:
                # 章名空時仍**無條件存 pending**（不丟 element，no silent fail 新洞，Codex）。
                # 問句在無章名時降級為「請直接告訴我章名」（不列枚舉）。
                state.pending_special_element_json = json.dumps({
                    "kind": "clarify",
                    "elements": clarify_pending,
                    "chapter_names": _chapter_names,   # 可能空 list
                }, ensure_ascii=False)
                await self._persist_checkpoint_boundary(state)
                _joined = "、".join(
                    f"「{p['raw_target'] or '（未指定章節）'}」" for p in clarify_pending)
                if _chapter_names:
                    _q = lr_copy.special_element_clarification_question(
                        [p["raw_target"] or "（未指定章節）" for p in clarify_pending], _chapter_names)
                else:
                    _q = (f"你想把特殊格式（如表格）放在哪一章呢？你剛提到的{_joined}我一時對應不準，"
                          f"目前章節還沒定案。請直接告訴我要放哪一章（章名）。")
                return await self._emit_clarification(
                    ClarificationRequest(question=_q, stage=4), state)
            await self._emit_narration(
                f"已記下 {len(resolved_elements)} 個格式 element。確認其他格式偏好？"
            )
            await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
            # plan: lr-special-element-persist — add_special_element 全命中 happy path
            # 寫 special_elements + pending_format_confirmation 後補 persist
            # （斷線/reload 遺失根解；confirm/clarify 分支已各自 persist，此為唯一漏的 terminal）。
            await self._persist_checkpoint_boundary(state)
            return state

        if action in (
            Stage4ResponseAction.adjust_chapters,
            Stage4ResponseAction.new_structure_request,
        ):
            # 改章節 outline / 全新結構訴求 → trigger reframe entry typed。
            # 在進 entry 前先寫 citation_style + special_elements，確保 mock entry
            # 的 unit test 也能驗證 propagation（不依賴 entry 內部 mutation）。
            sc = response.structural_content
            assert sc is not None
            fc = response.format_content
            if fc is not None:
                if fc.citation_style_extracted is not None:
                    state.user_voice.citation_style = fc.citation_style_extracted
                if fc.special_elements:
                    state.format_specs = dict(state.format_specs or {})
                    # reframe 帶表格：章名正在被 reframe 改動、暫定章名未穩定，不在此即時判
                    # target 語意（會白問）；交 Stage 5 後衛兜底。B3：走 serializer strip transient。
                    state.format_specs["special_elements"] = [
                        _serialize_special_element_for_state(e) for e in fc.special_elements
                    ]
                # Blocker A (2026-05-19)：mixed structure + format spec 路徑也要
                # propagate word count（reframe entry 不會自己 propagate）
                if fc.target_word_count is not None:
                    state.user_voice.target_word_count = fc.target_word_count
                    state.format_specs = dict(state.format_specs or {})
                    state.format_specs["target_word_count"] = fc.target_word_count
            return await self._try_stage_4_reframe_entry_typed(
                state, user_message, structural=sc, format_content=fc,
            )

        if action == Stage4ResponseAction.auto_continue:
            state.format_specs = self._merge_format_specs_default(state.format_specs)
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        if action == Stage4ResponseAction.unclear:
            from reasoning.schemas_live import ClarificationRequest
            # 同源病 root fix（對齊 Stage 1/3/5）：re-emit checkpoint 恢復前端 reply UI
            # （continueLiveResearch 已把 _lrAwaitingCheckpointReply 設 false、隱藏 reply UI；
            # checkpoint_prompt 未被 reframe 污染，見 _emit_reframe_proposal 2026-05-18 root-fix）。
            # 收編到共用 _emit_clarification helper（設計文件 §3）。
            return await self._emit_clarification(
                ClarificationRequest(question=response.clarifying_question, stage=4), state
            )

        # Unhandled action — 不可 silent fail
        raise ValueError(f"Unhandled Stage4ResponseAction: {action}")

    async def _enforce_pending_exclusivity(self, state) -> bool:
        """R7 invariant：兩個 pending 不得同時非空。異常時保留 reframe（結構優先）、
        清 special + user-facing 誠實提示（SF3b：已清資料、不承諾「不漏」）。
        Returns True 若發生 recovery（caller 應改 route reframe、不再走 auto/blank）。"""
        if not (state.pending_reframe_json and state.pending_special_element_json):
            return False
        logger.error("[LIVE RESEARCH] INVARIANT VIOLATION: reframe + special_element pending both set")
        state.pending_special_element_json = ""   # 清 special（reframe 優先吃）
        await self._persist_checkpoint_boundary(state)
        # SF3b：誠實文案 —— 資料已清（不能承諾「不漏」），請結構確認完後再說一次
        await self._emit_narration(
            "為了避免放錯，我先暫停這個表格／圖表設定；等結構這步確認完，"
            "請再告訴我一次要放哪一章。"
        )
        return True

    async def _classify_pending_special_element_reply(self, user_message, chapter_names, pending_summary):
        """R7：LLM 判 user 對 special_element 澄清問句的回答意圖（不用 substring/regex）。
        比照 _classify_confirmation_intent 的 LLM classifier pattern（level="low"，LLM fail → 安全 default）。

        Returns dict: {"intent": one of confirm|change_chapter|cancel|reframe|unclear,
                       "target_chapter": <LLM 判 user 想改到的章名原文，僅 change_chapter 有>}
        - confirm：user 接受 pending 提案（「對/是的/好/就放那章」）→ caller 用 pending 的 resolved_title 定位
        - change_chapter：user 改章（「不是，我要國外案例」「放結論那章」「第二個」「最後一章」）
            → LLM 順帶把新 target 語意對應到 chapter_names 裡的某章名（比照 R2 主流程語意判斷）；判不出留空
        - cancel：user 放棄（「算了/不用了/不要表格了」）→ caller 清 pending 不注入
        - reframe：user 沒在回答「放哪一章」，而是提出整體章節結構重組訴求
            （「把 11 個主題重組成三章」「整個架構重新整理」）→ caller 走 B2 逃生口：
            清 pending + 交 Stage 4 typed dispatcher 解結構 payload（2026-07-15 Cayenne 死迴圈修法）
        - unclear：判不準 → caller 重問
        """
        msg = (user_message or "").strip()
        if not msg:
            return {"intent": "unclear", "target_chapter": ""}
        if self.dry_run:
            # unit test 加速：只對明顯短句 keyword fallback（production 永遠走 LLM）。
            # 用 substring 命中（複合短句如「算了不用了」也中），confirm 先判、cancel 後判。
            low = msg.strip(" .,!?！？。，、~～").lower()
            if low in {"對", "是", "好", "沒問題", "就這樣", "ok", "確認", "可以"}:
                return {"intent": "confirm", "target_chapter": ""}
            if any(kw in low for kw in ("算了", "不用了", "不用", "取消", "不要表格", "cancel")):
                return {"intent": "cancel", "target_chapter": ""}
            return {"intent": "unclear", "target_chapter": ""}
        _opts = "、".join(f"{i+1}）{n}" for i, n in enumerate(chapter_names)) or "（章節尚未定案）"
        prompt = (
            "你是意圖分類器。系統剛問使用者一個「表格／圖表要放哪一章」的澄清問句，"
            f"現在使用者回了一句。目前規劃的章節：{_opts}。\n"
            f"待確認的內容：{pending_summary}\n\n"
            f"使用者訊息：\n\"\"\"{msg}\"\"\"\n\n"
            "判斷意圖並回 JSON：\n"
            "- intent=\"confirm\"：接受系統提案（「對」「是的」「好就放那章」「沒錯」）。\n"
            "- intent=\"change_chapter\"：要改放別章（「不是，我要國外案例」「放結論那章」「第二個」「最後一章」）。\n"
            "  * **章節清單非空時**：target_chapter 填 user 想改到的**上方章節清單裡的章名原文**"
            "（用你的語意理解對應，「國外案例」→對到清單裡的「國外案例」、「第二個」→第2章名、「最後一章」→末章名）。\n"
            "  * **章節清單為空（尚未定案）時**：沒有清單可對照 —— 只要 user 給了**具體章名或位置短語**"
            "（如「結論」「政策建議那章」「最後一章」），就 target_chapter 填 **user 的原話 raw target**"
            "（照抄「結論」「政策建議那章」，不要留空）；系統之後會用定案章名去對。\n"
            "  * 只有 user 仍**沒給任何具體 target**（純模糊「隨便」「你決定」）才留空字串。\n"
            "- intent=\"reframe\"：user 沒在回答「放哪一章」，而是提出**整體章節結構重組**訴求\n"
            "  （「把這 11 個主題重組成三章」「改成三章：前言、國際案例分析、結論」「整個架構重來」）。\n"
            "  這不是 change_chapter（change_chapter 是幫這個表格換一章；reframe 是要改掉整份章節結構）。\n"
            "  reframe 時 target_chapter 留空字串。\n"
            "- intent=\"cancel\"：放棄放這個特殊格式（「算了」「不用了」「不要表格了」）。\n"
            "- intent=\"unclear\"：判不準 / 模糊 / user 沒給具體 target。\n\n"
            "只回 JSON。"
        )
        schema = {"type": "object", "properties": {
            "intent": {"type": "string", "enum": ["confirm", "change_chapter", "cancel", "reframe", "unclear"]},
            "target_chapter": {"type": "string"},
        }, "required": ["intent"]}
        try:
            resp = await ask_llm(prompt, schema, level="low",
                                 query_params=getattr(self.handler, "query_params", {}),
                                 max_length=512)   # SF1：比照 _classify_confirmation_intent
        except Exception as e:
            logger.warning(f"[LIVE RESEARCH] pending special_element reply classify fail: {e}")
            return {"intent": "unclear", "target_chapter": ""}   # 安全 default → 重問，不誤動作
        if not resp:
            return {"intent": "unclear", "target_chapter": ""}
        # SF1：unwrap schema-wrapped response（{type,properties,required} → properties），
        # 照 _classify_confirmation_intent 的 pattern（wrapped success 誤當 unclear
        # 會 production 反覆重問）。
        if "intent" not in resp and "properties" in resp and isinstance(resp["properties"], dict):
            resp = resp["properties"]
        intent = resp.get("intent", "unclear")
        if intent not in ("confirm", "change_chapter", "cancel", "reframe", "unclear"):
            logger.warning(f"[LIVE RESEARCH] pending reply unknown intent {intent!r} → unclear")
            intent = "unclear"   # SF1：unknown intent guard → 安全 default
        return {"intent": intent,
                "target_chapter": (resp.get("target_chapter") or "").strip()}

    async def _handle_pending_special_element(self, state, user_message, auto_continue=False):
        """R7：處理 special_element 澄清 round 的 user reply。意圖分流**全交 LLM**
        （_classify_pending_special_element_reply），不用 substring/regex。
        confirm → 用 pending 的 resolved_title 定位；change_chapter → 用 LLM 判的新 target
        定位；cancel → 清 pending 不注入；reframe → B2 逃生口（清 pending + 交 Stage 4
        typed dispatcher 走既有 reframe entry，2026-07-15 Cayenne 死迴圈修法）；unclear → 重問。
        blank/auto_continue（B-order）→ re-emit clarification，不 finalize、不清 pending。
        """
        import json
        from reasoning.schemas_live import ClarificationRequest
        # B-order：blank/auto 時保 pending、re-emit 問句，不 advance（不能誤 complete_stage）
        if auto_continue or not (user_message or "").strip():
            return await self._reemit_pending_special_clarification(state)
        try:
            pend = json.loads(state.pending_special_element_json)
        except Exception:
            # malformed → 清空 + persist（防重連反覆進壞 pending）+ 重問（no silent fail）
            state.pending_special_element_json = ""
            await self._persist_checkpoint_boundary(state)
            await self._emit_narration("剛才的表格設定我沒接好，麻煩再說一次要放哪一章？")
            await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
            return state

        chapter_names = pend.get("chapter_names") or []
        elements = pend.get("elements") or []
        state.format_specs = dict(state.format_specs or {})
        # B3：讀既有 list 也先 sanitize（舊 session 可能已有髒 transient）
        cur = [_serialize_special_element_for_state(e) for e in
               (state.format_specs.get("special_elements") or []) if isinstance(e, dict)]

        _summary = "、".join(
            f"{el.get('type','')}放「{el.get('resolved_title') or el.get('raw_target') or '未指定'}」"
            for el in elements)
        cls = await self._classify_pending_special_element_reply(
            user_message, chapter_names, _summary)
        intent = cls["intent"]

        def _finalize(target_for_all=None):
            """把 pending elements 寫進 special_elements（走 serializer，B3）。
            target_for_all=None → 各 element 用自己的 resolved_title（confirm path）；
            否則全用 target_for_all（change_chapter path，多 element 共用新章 —— 見多-element 註）。"""
            for el in elements:
                tgt = target_for_all if target_for_all is not None else (el.get("resolved_title") or "")
                cur.append(_serialize_special_element_for_state({
                    "type": el.get("type", ""), "target_chapter": tgt,
                    "description": el.get("description", ""),
                }))
            state.format_specs["special_elements"] = cur

        async def _emit_clarify_again():
            state.pending_special_element_json = json.dumps(
                {"kind": "clarify", "elements": elements, "chapter_names": chapter_names},
                ensure_ascii=False)
            await self._persist_checkpoint_boundary(state)
            return await self._emit_clarification(ClarificationRequest(
                question=lr_copy.special_element_clarification_question(
                    [e.get("raw_target") or "（未指定章節）" for e in elements], chapter_names),
                stage=4), state)

        if intent == "cancel":
            state.pending_special_element_json = ""
            await self._persist_checkpoint_boundary(state)
            await self._emit_narration("好的，那就不放這個表格／圖表了。確認其他格式偏好？")
            await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
            return state

        if intent == "confirm":
            # confirm 只在 confirm kind（有 resolved_title）合法；clarify kind 沒有 title → 當 unclear 重問
            if pend.get("kind") == "confirm" and all(el.get("resolved_title") for el in elements):
                _finalize(target_for_all=None)  # 各用自己的 resolved_title
                backlog = pend.get("clarify_backlog") or []
                if backlog:
                    state.pending_special_element_json = json.dumps(
                        {"kind": "clarify", "elements": backlog, "chapter_names": chapter_names},
                        ensure_ascii=False)
                    await self._persist_checkpoint_boundary(state)
                    return await self._emit_clarification(ClarificationRequest(
                        question=lr_copy.special_element_clarification_question(
                            [b["raw_target"] or "（未指定章節）" for b in backlog], chapter_names),
                        stage=4), state)
                state.pending_special_element_json = ""
                await self._persist_checkpoint_boundary(state)
                await self._emit_narration("好的，已放進去。確認其他格式偏好？")
                await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
                return state
            return await _emit_clarify_again()

        if intent == "change_chapter":
            _new_target = cls.get("target_chapter", "")
            # B-loop 根解（Codex）：章名空（chapter_names=[]）時 layer1 必回 None →
            # 不能無限重問。此時 user 已給了明確 target（如「結論」）→ 直接寫入 raw target
            # （走 serializer），交 Stage 5 exact filter / unmatched 後衛兜底（不 silent、不卡死）。
            if not chapter_names and _new_target:
                _finalize(target_for_all=_new_target)
                state.pending_special_element_json = ""
                await self._persist_checkpoint_boundary(state)
                await self._emit_narration(
                    f"好，我先記下放到「{_new_target}」，章節定案後我會對上，"
                    f"如果對不上會再提醒你。確認其他格式偏好？")
                await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
                return state
            # 有章名 → LLM 判的新 target 過第一層 code 短路確認 exact/唯一命中 chapter_names
            new_hit = _resolve_target_chapter_layer1(_new_target, chapter_names)
            if new_hit:  # "" 或 None 都不算命中（"" 是空章名，此處不接受空）
                _finalize(target_for_all=new_hit)
                state.pending_special_element_json = ""
                await self._persist_checkpoint_boundary(state)
                await self._emit_narration(f"好，改放到「{new_hit}」那章。確認其他格式偏好？")
                await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
                return state
            return await _emit_clarify_again()   # 有章名但 LLM 給的 target 對不到 → 重問

        if intent == "reframe":
            # B2 逃生口（2026-07-15 Cayenne 死迴圈，findings B2 / lessons 2026-07-15 條）：
            # user 在澄清 round 提出整體章節結構重組 → 不可重播澄清問句
            # （重播同句 = 實質吞掉訴求 = silent fail，違 CLAUDE.md）。
            # Port Stage 5 pending_recollect_confirmation 段4 substantive fall-through 設計：
            # 訴求交回 Stage 4 正規 typed dispatch（_classify_stage_4_response）解出
            # typed payload，再走既有 _try_stage_4_reframe_entry_typed（不重造 reframe 輪）。
            from reasoning.schemas_live import Stage4ResponseAction
            response = await self._classify_stage_4_response(state, user_message)
            # 分流紀律（lessons「intent parser 回 None ≠ 回 dict 無 action」）：
            # _classify_stage_4_response 在 LLM API 失敗/空回應/validation fail 時回
            # unclear + LLM_UNAVAILABLE_NARRATION sentinel —— 系統端故障 fail-loud 用
            # 系統文案 + pending 原樣保留等重試；不可誤說「沒看懂」、不可丟 pending、
            # 不可進 reframe entry。此判定必在下方結構 payload 路由之前。
            if (response.action == Stage4ResponseAction.unclear
                    and response.clarifying_question == lr_copy.LLM_UNAVAILABLE_NARRATION):
                logger.warning(
                    "[LIVE RESEARCH] pending special_element reframe escape: stage-4 "
                    "classifier LLM fail — keep pending, emit system narration"
                )
                await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
                return await self._reemit_pending_special_clarification(state)
            if (response.action in (Stage4ResponseAction.adjust_chapters,
                                    Stage4ResponseAction.new_structure_request)
                    and response.structural_content is not None):
                # R7 互斥 invariant：_try_stage_4_reframe_entry_typed 內的
                # _emit_reframe_proposal 會設 pending_reframe_json —— special pending
                # 必須在此**先清**（沿用 _enforce_pending_exclusivity 既定策略：
                # reframe 優先、special 清 + 誠實告知，不 silent 丟需求）。
                state.pending_special_element_json = ""
                await self._persist_checkpoint_boundary(state)
                await self._emit_narration(
                    "好，先處理你的結構調整；原本的表格／圖表位置設定我先暫停，"
                    "等新結構確認後，請再告訴我一次要放哪一章。"
                )
                logger.info(
                    "[LIVE RESEARCH] pending special_element reframe escape → "
                    "stage-4 reframe entry (chapters=%d)",
                    len(response.structural_content.new_chapters),
                )
                try:
                    return await self._try_stage_4_reframe_entry_typed(
                        state, user_message,
                        structural=response.structural_content,
                        format_content=response.format_content,
                    )
                except Exception:
                    # SF-A（AR R1 Codex）：此時 pending 已清、persist 已落，proposal 可能
                    # 未發出 —— 不可 silent。誠實告知 user 重述（pending 已空，重述會走
                    # 正規 Stage 4 dispatch 仍可達 reframe，嚴格優於死迴圈現狀）。
                    logger.exception(
                        "[LIVE RESEARCH] pending reframe escape: reframe entry failed "
                        "after pending cleared — fail-loud, ask user to restate"
                    )
                    await self._emit_narration(
                        "抱歉，結構調整處理失敗。請重新告訴我一次你想要的章節結構；"
                        "原本的表格／圖表位置設定已先暫停，結構確認後請再說一次。"
                    )
                    return state
            # 低階分類器判 reframe、typed classifier 沒解出結構 payload（真模糊）→
            # 不硬猜、不丟 pending，重問（user 可換更明確措辭；安全方向）。
            logger.info(
                "[LIVE RESEARCH] pending special_element reframe escape: no structural "
                f"payload (action={response.action}) — re-clarify, pending kept"
            )
            return await _emit_clarify_again()

        # unclear
        return await _emit_clarify_again()

    async def _reemit_pending_special_clarification(self, state):
        """B-order：pending 非空但 user 送 blank/auto → 從 pending 重建問句 re-emit，
        不 finalize、不清 pending、不 advance。malformed → 走 handler 的 malformed recovery。"""
        import json
        from reasoning.schemas_live import ClarificationRequest
        try:
            pend = json.loads(state.pending_special_element_json)
        except Exception:
            state.pending_special_element_json = ""
            await self._persist_checkpoint_boundary(state)
            await self._emit_narration("剛才的表格設定我沒接好，麻煩再說一次要放哪一章？")
            await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
            return state
        elements = pend.get("elements") or []
        chapter_names = pend.get("chapter_names") or []
        if pend.get("kind") == "confirm":
            _q = lr_copy.special_element_confirm_question(
                [e.get("resolved_title") for e in elements if e.get("resolved_title")])
        elif chapter_names:
            _q = lr_copy.special_element_clarification_question(
                [e.get("raw_target") or "（未指定章節）" for e in elements], chapter_names)
        else:
            _q = ("你想把特殊格式（如表格）放在哪一章呢？目前章節還沒定案，"
                  "請直接告訴我要放哪一章（章名）。")
        return await self._emit_clarification(ClarificationRequest(question=_q, stage=4), state)

    async def _classify_stage_4_response(
        self,
        state: "LiveResearchStageState",
        user_message: str,
    ) -> "Stage4Response":
        """TypeAgent dispatcher 入口 — 把 user reply 分類為 typed Stage4Response action。

        Plan: lr-typeagent-refactor (2026-05-19) — 取代舊 `_parse_stage_4_intent`。
        dry_run / LLM 失敗 → safe default 'unclear'（不 silent fail，由 caller narration）。
        """
        from reasoning.schemas_live import Stage4Response, Stage4ResponseAction

        if self.dry_run:
            return Stage4Response(
                action=Stage4ResponseAction.unclear,
                clarifying_question="（dry_run 模式 — 請具體說明訴求）",
            )

        from reasoning.prompts.stage4_intent import Stage4IntentPromptBuilder

        builder = Stage4IntentPromptBuilder()
        # R2：取暫定章名清單餵 classifier，讓 LLM 順帶判 special_element target 語意（不另起 call）
        try:
            _cm = ContextMap.model_validate_json(state.context_map_json)
            _src, _ = self._resolve_chapter_source(_cm, state.format_specs)
            _chapter_names = [
                (c.get("name", "") if isinstance(c, dict) else getattr(c, "name", ""))
                for c in _src
            ]
        except Exception:
            _chapter_names = []
        prompt = builder.build_response_classifier_prompt(
            user_message=user_message,
            pending_reframe=bool(state.pending_reframe_json),
            pending_format_confirmation=state.pending_format_confirmation,
            chapter_names=_chapter_names,
        )

        # JSON schema for typed action — instructor backend reject + retry 不合規 output
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "confirm_reframe", "confirm_format", "confirm_both",
                        "cancel_reframe", "adjust_chapters", "adjust_format",
                        "add_special_element", "new_structure_request",
                        "auto_continue", "unclear",
                    ],
                },
                "confirm_target": {
                    "type": ["string", "null"],
                    "enum": ["reframe", "format", "both", None],
                },
                "structural_content": {
                    "type": ["object", "null"],
                    "properties": {
                        "new_chapters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["narrative_chapter"],
                                    },
                                    "name": {"type": "string", "minLength": 1},
                                    "description": {"type": "string"},
                                    "relevance": {
                                        "type": "string",
                                        "enum": ["core", "supporting", "peripheral"],
                                    },
                                },
                                "required": ["name"],
                            },
                        },
                        "summary": {"type": "string"},
                    },
                },
                "format_content": {
                    "type": ["object", "null"],
                    "properties": {
                        "format_spec_extracted": {"type": "string"},
                        "citation_style_extracted": {
                            "type": ["string", "null"],
                            "enum": [
                                "author_year", "numeric", "footnote", "none", None,
                            ],
                        },
                        # Blocker A (2026-05-19) root fix：中文字數 typed int channel
                        "target_word_count": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                        },
                        "special_elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "table", "list", "chart",
                                            "diagram", "code_block",
                                        ],
                                    },
                                    "target_chapter": {"type": "string"},
                                    "description": {"type": "string"},
                                    # R2 transient 語意判斷（用完即丟，不持久化）
                                    "resolved_chapter_title": {"type": ["string", "null"]},
                                    "resolution_confidence": {
                                        "type": ["string", "null"],
                                        "enum": ["clear", "uncertain", None],  # None = 允許 null
                                    },
                                },
                                "required": ["type"],
                            },
                        },
                    },
                },
                # Blocker C (2026-05-19)：欄位語意 None == "" == 無需澄清。
                # JSON schema 容忍 null（Pydantic field_validator coerce 為 ""），
                # 避免 instructor backend pre-reject null clarifying_question
                # 整個 Stage4Response，導致 dispatcher 拿不到 typed action。
                "clarifying_question": {"type": ["string", "null"]},
            },
            "required": ["action"],
        }

        try:
            response = await ask_llm(
                prompt,
                schema,
                level="low",
                query_params=getattr(self.handler, "query_params", {}),
                max_length=4096,
            )
        except Exception as e:
            logger.warning(
                f"[LIVE RESEARCH] _classify_stage_4_response ask_llm failed: {e}"
            )
            return Stage4Response(
                action=Stage4ResponseAction.unclear,
                clarifying_question=lr_copy.LLM_UNAVAILABLE_NARRATION,
            )

        if not response:
            logger.warning(
                "[LIVE RESEARCH] _classify_stage_4_response: ask_llm returned empty"
            )
            return Stage4Response(
                action=Stage4ResponseAction.unclear,
                clarifying_question=lr_copy.LLM_UNAVAILABLE_NARRATION,
            )

        # Unwrap schema-wrapped response (instructor occasionally returns `{properties: {...}}`)
        if "action" not in response and "properties" in response \
                and isinstance(response["properties"], dict):
            response = response["properties"]

        try:
            return Stage4Response.model_validate(response)
        except Exception as e:
            logger.warning(
                f"[LIVE RESEARCH] Stage4Response validation fail: {e}"
            )
            return Stage4Response(
                action=Stage4ResponseAction.unclear,
                clarifying_question=lr_copy.LLM_UNAVAILABLE_NARRATION,
            )

    async def _try_stage_4_reframe_entry_typed(
        self,
        state: LiveResearchStageState,
        user_message: str,
        structural: "Stage4StructuralPayload",
        format_content: Optional["Stage4FormatPayload"],
    ) -> LiveResearchStageState:
        """TypeAgent typed reframe entry — 取代舊 `_try_stage_4_reframe_entry` legacy path。

        structural.new_chapters 已是 typed List[ChapterSpec]，直接構造 reframe op。
        format_content（若非 None）propagate format_specs / citation_style。
        """
        try:
            context_map = ContextMap.model_validate_json(state.context_map_json)
        except Exception as e:
            logger.warning(
                f"[LIVE RESEARCH] Stage 4 reframe entry (typed): context_map parse fail: {e}"
            )
            await self._emit_narration("目前結構讀取失敗，先繼續格式確認。")
            await self._emit_checkpoint(stage=4, proposal=state.checkpoint_prompt)
            # plan: lr-special-element-persist — B-j caller 委派前寫入的格式偏好
            # （citation_style/special_elements/word_count）在 parse-fail 路徑落盤
            # （偏好與 reframe 成敗正交；對已自行 persist 的 caller 冪等無害）。
            await self._persist_checkpoint_boundary(state)
            return state

        # Propagate format_content（含 special_elements / citation_style）
        if format_content is not None:
            if format_content.special_elements:
                state.format_specs = dict(state.format_specs or {})
                state.format_specs["special_elements"] = [
                    _serialize_special_element_for_state(e) for e in format_content.special_elements
                ]
            if format_content.format_spec_extracted:
                state.format_specs = self._merge_format_specs_user(
                    state.format_specs,
                    format_content.format_spec_extracted,
                    special_elements=[
                        e.model_dump() for e in format_content.special_elements
                    ],
                )
            if format_content.citation_style_extracted is not None:
                state.user_voice.citation_style = format_content.citation_style_extracted
            # Blocker A (2026-05-19) root fix：word count typed propagate
            if format_content.target_word_count is not None:
                state.user_voice.target_word_count = format_content.target_word_count
                state.format_specs = dict(state.format_specs or {})
                state.format_specs["target_word_count"] = format_content.target_word_count
            state.pending_format_confirmation = True

        new_chapter_dicts = [
            ch.model_dump(exclude={"type"}) for ch in structural.new_chapters
        ]
        reframe_op = ContextMapRevisionOperation(
            op_type="reframe_structure",
            new_chapters=new_chapter_dicts,
        )
        summary = structural.summary or f"整體重組為 {len(new_chapter_dicts)} 章"
        logger.info(
            f"[LIVE RESEARCH] Stage 4 reframe entry (typed): "
            f"chapters={len(new_chapter_dicts)} summary={summary!r}"
        )
        return await self._emit_reframe_proposal(
            state, reframe_op, context_map, summary, target_stage=4,
        )

    # APA title fallback (2026-06-18)：author 缺時用文章標題取代，截前 N 字。
    # N=10 是繁中標題「足以辨識 + 不破壞 inline 緊湊」的折衷；中文無「詞」邊界，
    # 故以字元數 adapt APA 7th 英文「前 1-4 words」規則。超長加全形省略號「…」。
    _TITLE_FALLBACK_MAXLEN = 10

    @staticmethod
    def _render_section_citations(
        section: "LiveWriterSectionOutput",
        evidence_lookup: Dict[int, "EvidencePoolEntry"],
        citation_format: Literal["author_year", "numeric", "footnote", "none"],
    ) -> "LiveWriterSectionOutput":
        """Render section_content 中的 {cite:N} placeholder 為 user 拍板的 citation style。

        Plan: lr-typeagent-refactor Target 3（2026-05-19，CEO 拍板 OQ-5 立刻 strict）。
        Writer LLM 只 output `{cite:N}` placeholder + citations list；code 端
        從 evidence_lookup 取真實 author/year metadata，依 citation_format 統一 render，
        消除 long writer call 漂移（v8 regression：author=topic_title / year=n.d.）。

        - author_year (OQ-3 CEO 拍板)：中文 author 整名 render「(王立人, 2022)」
        - numeric: {cite:1} → [1]
        - footnote: {cite:1} → ¹ (unicode superscript)
        - none: {cite:1} → '' (移除)

        Fallback rule（LR VP-3 RCA 2026-05-16 + APA title fallback 2026-06-18）：
        author 與 year 各自獨立 fallback，互不連坐：
        - year 空 → 先試 published_at derive 年份（YYYY-MM-DD 前 4 字）；
          仍空 → 標準「n.d.」。
        - author 空 → 取 entry.title 前 N 字（_TITLE_FALLBACK_MAXLEN）加全形引號
          「」（超長加全形省略號），即 APA 7th「無作者時以標題取代作者位置」標準做法。
          **與 2026-05-16 RCA 禁止的 source_domain 偽裝本質不同**：domain（cna.com.tw）
          放 author 位會被讀成人名而誤導；title 是文章本體資訊 + 加引號已明示「這是標題
          不是作者」，不誤導，且為 APA 標準。**絕不**回退到 source_domain。
        - author 空且 title 也空（極端，理論上不該發生）→ 維持「(來源不明, n.d.)」兜底
          （CLAUDE.md no silent fail —— 明示 metadata 全缺）。
        - author 缺（含極端來源不明）筆數計入 missing_metadata_count，
          觸發時 methodology_note append 記錄並說明已依 APA 慣例改用標題（audit trail）。
        """
        import re

        content = section.section_content
        missing_metadata_count = 0
        SUPERSCRIPT = {
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        }

        def _superscript(n: int) -> str:
            return "".join(SUPERSCRIPT[c] for c in str(n))

        def _replace(match):
            nonlocal missing_metadata_count
            eid = int(match.group(1))
            entry = evidence_lookup.get(eid)
            if entry is None:
                # missing → 移除 placeholder（guard 後續會標 Low confidence）
                return ""

            if citation_format == "numeric":
                return f"[{eid}]"
            if citation_format == "footnote":
                return _superscript(eid)
            if citation_format == "none":
                return ""

            # author_year (OQ-3: 中文 author 整名 render)。
            # author 與 year 各自獨立 fallback，互不連坐（2026-06-18 APA title fallback）。
            author = (entry.author or "").strip()

            # year fallback（維持現狀）。FIX-3 (Cayenne #10): year 缺時從 published_at
            # 取年份（YYYY-MM-DD 前 4 字）。real-retrieval evidence 只填 published_at
            # （Track E），year 欄常空 → 不 derive 則永遠 n.d.。
            year = (entry.year or "").strip()
            if not year and getattr(entry, "published_at", None):
                year = (entry.published_at or "")[:4].strip()
            if not year:
                year = "n.d."

            # author fallback（APA 7th「無作者時以標題取代作者」）。
            # RCA 2026-05-16: 禁止用 source_domain 偽裝 author（cna.com.tw 被讀成人名）。
            # title 取代與 domain 偽裝本質不同：title 是文章本體 + 全形引號明示「這是標題」。
            if not author:
                title = (getattr(entry, "title", "") or "").strip()
                if title:
                    maxlen = LiveResearchOrchestrator._TITLE_FALLBACK_MAXLEN
                    if len(title) > maxlen:
                        title_short = title[:maxlen] + "…"  # 全形省略號（引號內）
                    else:
                        title_short = title
                    author = f"「{title_short}」"
                    missing_metadata_count += 1  # 缺 author 已用 title 取代，計入 audit
                else:
                    # 連 title 都沒有（極端，理論上不該發生）→ no silent fail 兜底
                    missing_metadata_count += 1
                    return "(來源不明, n.d.)"

            return f"({author}, {year})"

        new_content = re.sub(r"\{cite:(\d+)\}", _replace, content)

        new_note = section.methodology_note
        if missing_metadata_count > 0:
            note_addition = (
                f"[citation metadata：{missing_metadata_count} 筆 citation 缺作者，"
                f"已依 APA 慣例改用文章標題標示（標題亦缺者標示為來源不明）]"
            )
            new_note = (
                f"{new_note} {note_addition}".strip()
                if new_note
                else note_addition
            )

        return section.model_copy(update={
            "section_content": new_content,
            "methodology_note": new_note,
        })

    @staticmethod
    def _merge_format_specs_default(existing: dict) -> dict:
        """R3 (2026-05-16)：set default markdown_apa，但保留既有 keys。

        既有 chapters（Plan 2 Phase 4 fallback 寫入的 chapter override）
        或 user_specified 不可被 wipe — auto_continue 只該補 default key 缺的部分。
        """
        merged = dict(existing or {})
        merged.setdefault("default", "markdown_apa")
        return merged

    @staticmethod
    def _merge_format_specs_user(
        existing: dict,
        user_message: str,
        special_elements: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        """R3 (2026-05-16)：set user_specified，但保留既有 chapters / default。

        spec §4.10 (2026-05-16)：special_elements 非 None / 非空時，**覆寫**
        既有 special_elements（user 在 Stage 4 一次定案；空 list 視為「沒新訴求」，
        保留既有；None = caller 沒 propagate parser output，保持既有不動）。
        """
        merged = dict(existing or {})
        merged["user_specified"] = user_message
        if special_elements:
            # 非空 list → user 此輪明確提了 element 訴求，覆寫。B3：統一走 serializer strip transient。
            # （caller 傳進來的 [e.model_dump()...] 含 transient，由此處 serializer 統一 strip。）
            merged["special_elements"] = [
                _serialize_special_element_for_state(e) for e in special_elements
            ]
        return merged

    # ──── Stage 5: 分段輸出 ────────────────────────────────────

    def _resolve_chapter_source(
        self,
        context_map: "ContextMap",
        format_specs: dict,
    ) -> tuple:
        """Plan 2 Phase 2: Resolve Stage 5 writer section source.

        Option B (CEO 拍板): cm.topics default + format_specs.chapters override 補位。
        - 有 format_specs.chapters (非空 list) → 使用 user-specified chapter override
        - 否則 → fallback ContextMap.topics where relevance == "core"

        Returns:
            Tuple[List, bool]:
                writer_sections: 章節迭代來源。chapter override 模式下為
                    List[Dict[str, str]] (each dict: {"name", "outline"});
                    fallback 模式下為 List[ContextMapTopic]。
                using_override: True 表示走 chapter override 路徑 (writer 無對應
                    topic.evidence_ids，第一章拿 union evidence_ids 由 Phase 3 處理);
                    False 表示走既有 core_topics 路徑（行為不變）。

        Note (Plan 4 reuse): Plan 4 的 intent parser fallback 寫入 chapters 進
            format_specs 後，本 helper 直接 honor，無須 Plan 4 改 helper 介面。
        """
        chapters = format_specs.get("chapters") if format_specs else None
        if chapters and isinstance(chapters, list) and len(chapters) > 0:
            return chapters, True
        core_topics = [t for t in context_map.topics if t.relevance == "core"]
        return core_topics, False

    def _recollect_cap(self) -> int:
        """同一 session recollect 次數上限（S5，default 2）。可由 features override。"""
        return int(self.features.get("lr_recollect_cap", 2)) if getattr(self, "features", None) else 2

    async def _dispatch_recollect(self, state: LiveResearchStageState) -> LiveResearchStageState:
        """Stage 5 退回 analyst 補搜 → 重進 analyst→critic→writer→critic loop。

        補搜引擎復用 BABLoopEngine.run_loop（Task2 SEARCH_REQUIRED cap 3 internal +
        gap routing max_external 6），不新增無上限補搜路徑。保留的 evidence_pool 當 seed
        傳進 _run_stage_1（pool + counter 同傳，防 ID 衝突），疊加新 evidence（S2）。

        H（cap 並發 race）：count += 1 後、await _run_stage_1（30-60s）前先強制持久化，
        防雙擊/重送/SSE reconnect 兩 request 都過 cap 檢查 → 雙倍燒錢。
        I（半重置）：reset + count 在 try 內，_run_stage_1 失敗 → rollback 到入口 snapshot
        + emit 明確 error checkpoint（非半重置 broken state）。
        """
        # cap 二次防護（confirm 路徑也檢查，防 pending 期間其他輪推進計數）
        if state.recollect_count >= self._recollect_cap():
            logger.info("[LIVE RESEARCH] _dispatch_recollect: capped, blocked")
            await self._emit_narration(lr_copy.RECOLLECT_CAPPED_NARRATION)
            state.set_checkpoint(lr_copy.RECOLLECT_CAPPED_NARRATION)
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)
            return state

        # deserialize 研究問題 + 保留的 pool（當 seed）— 先取再 reset（reset 不動這兩個）
        context_map = ContextMap.model_validate_json(state.context_map_json)
        query = context_map.research_question
        seed_pool = (
            deserialize_evidence_pool(state.evidence_pool_json)
            if state.evidence_pool_json else None
        )
        seed_counter = max(seed_pool.keys()) if seed_pool else 0

        # I（Codex #7）：commit reset/count 前先 snapshot 入口 state（淺序列化）。
        # _run_stage_1 失敗 → 用此 snapshot 還原，避免「章節已清 + count 已耗 + 補搜沒跑完」
        # 的半重置破碎 state。最小版 rollback：用既有 to_dict/from_dict 對稱，不引入新
        # state-snapshot 架構（見「待 CEO」段 I 的取捨說明 → 已採最小版）。
        snapshot = state.to_dict()

        state.recollect_count += 1
        logger.info(
            f"[LIVE RESEARCH] Stage 5 recollect dispatch "
            f"(count={state.recollect_count}/{self._recollect_cap()})"
        )
        # 清過期下游 + 幽靈 guard + 推理產物，退回 Stage 1（保留 pool / context / 設定 / audit）
        state.reset_for_recollect()

        # H（Gemini #4 + Codex #4）：count+1 + reset 後、await 長跑 _run_stage_1 前
        # 先強制持久化 checkpoint boundary。並發第二 request 重入時讀到已遞增的
        # recollect_count → cap 檢查擋下 → 不雙倍燒錢。
        await self._persist_checkpoint_boundary(state)

        try:
            # B1：seed pool + counter 同傳，engine 從 counter+1 起分配新 ID 疊加（不覆蓋既有）
            return await self._run_stage_1(
                state, query, [],
                seed_evidence_pool=seed_pool, seed_counter=seed_counter,
            )
        except Exception as e:
            # I：rollback 到入口 snapshot（還原章節 + count + 所有清掉的欄位），
            # emit 明確 error checkpoint（不可 silent fail，不留半重置 broken state）。
            logger.error(
                f"[LIVE RESEARCH] _dispatch_recollect: _run_stage_1 failed, "
                f"rolling back recollect reset: {type(e).__name__}: {e}"
            )
            restored = LiveResearchStageState.from_dict(snapshot)
            # 就地覆寫 state 的所有欄位（caller 持有同一 state ref）
            state.__dict__.update(restored.__dict__)
            await self._emit_narration(
                "重新蒐集資料時發生問題，已保留你原本的報告內容，沒有變動。"
                "可以稍後再試，或繼續編輯目前的章節。"
            )
            state.set_checkpoint("目前所有段落已寫完。要修改哪個段落，或進入匯出？")
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)
            return state

    async def _run_stage_5(self, state: LiveResearchStageState) -> LiveResearchStageState:
        """Stage 5: Per-section 寫作（VP-7 反轉 — single-step per call）。

        VP-7 (writer-per-section-checkpoint-plan.md §3.2):
        - 每次只寫**一段**（next_i = last_completed_section_index + 1）。
        - 完成後 emit per-section checkpoint「第 K/N 段完成。要繼續寫、修改某段、
          還是直接匯出？」，set state.checkpoint，state.stage5_waiting_for_user=True，
          return state（dialog loop 等下一輪 user reply）。
        - 寫到最後一段：emit all_done + final checkpoint「進入匯出？」
        - 首次進場（last_completed == -1）：跑 outline planner 一次（idempotent guard）。
        - Idempotent：若 last_completed == total-1 直接 emit final checkpoint，
          不重複寫 section。
        - connection_alive=False → mark offline + cap 檢查：未達 cap 照常寫這一段
          （bounded burn，到 per-section checkpoint 停存檔）；達 cap → 標 capped、
          persist、停（lr-sse-reconnect-resume 2026-06-15 改語意，見下方離線檢查段）。
        - CancelledError 必須 re-raise（task wrap 才收得到 cancel 完成訊號）。
        - finally 清 `stage_5_writer_running` 確保不論何種退出都歸位。

        VP-7 per-section checkpoint 是唯一中斷安全網：每段完成後 writer 自動 pause，
        user 透過 checkpoint 選擇繼續/修改/匯出。停止按鈕與 stop flag 機制已移除
        （2026-06-04，placebo — writer_status="stopped" 從未真正 emit）。
        """
        # online substantive advance → 重置離線計數（plan 3d）。
        # 注意：Stage 5 離線 auto-advance（offline 寫下一段）時 online=False → 不 reset，
        # cap 才能累積；只有重連後 online + 進來寫 = reset。
        self._maybe_reset_offline_counters(state)

        # 只有第一次進入 Stage 5 才 advance；resume 路徑保留既有 stage_status
        if state.current_stage != 5:
            state.advance_to_stage(5)
            await self._emit_stage_change(5)

        context_map = ContextMap.model_validate_json(state.context_map_json)
        # Plan 2 Phase 2: writer section source 走 helper（chapter override or core_topics fallback）
        writer_sections, using_chapter_override = self._resolve_chapter_source(
            context_map, state.format_specs
        )
        total_sections = len(writer_sections)

        # ────────────────────────────────────────────────────────────────────
        # Plan 4 Phase 2: Outline Planner — Stage 5 開頭一次性產出 BookOutline
        # ────────────────────────────────────────────────────────────────────
        # 進場 idempotent guard：state.book_outline_json 已存在 → skip 重 plan（resume
        # 路徑不重複 LLM call）。空 → 呼叫 planner、寫入 state、persist。
        # dry_run 路徑：plan §10 意外發現 #5 — 直接走 skeleton fallback（省 LLM cost、
        # 加速 fixture 跑），仍寫入 state.book_outline_json 確保 Phase 3 writer 看到。
        #
        # R4 staleness fix (RCA v3 ROOT 4)：原本 `if not state.book_outline_json:`
        # 只看「有沒有 cache」，沒驗證「cache 與當前 writer_sections 對齊」。如果
        # Stage 4 reframe 之後 cm.topics / format_specs.chapters 變了，但殘留的
        # outline 沒清 → 整段 planner code 被 skip → no narration（silent skip）。
        # 改成 staleness check：比對 cached chapter 數量 + name 是否與 writer_sections
        # 對齊，不對齊就 invalidate cache、emit 明示 narration、重新規劃。
        outline_is_stale = False
        if state.book_outline_json:
            try:
                cached_outline = BookOutline.model_validate_json(state.book_outline_json)
                expected_titles = [
                    (s.get("name", "") if isinstance(s, dict) else getattr(s, "name", ""))
                    for s in writer_sections
                ]
                cached_titles = [c.title for c in cached_outline.chapters]
                if len(cached_titles) != len(expected_titles):
                    outline_is_stale = True
                    logger.info(
                        f"[LIVE RESEARCH] Outline stale (count mismatch): "
                        f"cached={len(cached_titles)} vs expected={len(expected_titles)} "
                        f"→ invalidate + re-plan"
                    )
                elif cached_titles != expected_titles:
                    # Title-by-title mismatch（順序+內容）
                    outline_is_stale = True
                    logger.info(
                        f"[LIVE RESEARCH] Outline stale (title mismatch): "
                        f"cached={cached_titles} vs expected={expected_titles} "
                        f"→ invalidate + re-plan"
                    )
            except Exception as e:
                # Cache 無法 parse → 視同 stale，重 plan
                outline_is_stale = True
                logger.warning(
                    f"[LIVE RESEARCH] Outline cache unparseable, treat as stale: "
                    f"{type(e).__name__}: {e}"
                )

            if outline_is_stale:
                # CLAUDE.md 紀律：invalidate 必須 user-visible，不可 silent
                await self._emit_narration(
                    "原本的章節規劃已過期（章節數或順序與目前結構不對齊），重新規劃中..."
                )
                state.book_outline_json = ""

        if not state.book_outline_json:
            from reasoning.agents.outline_planner import (
                OutlinePlannerAgent,
                build_skeleton_outline,
            )

            # Parse style features (need it for planner input + downstream writer)
            _style_features_for_plan = None
            if state.style_features_json:
                _style_features_for_plan = StyleAnalysisOutput.model_validate_json(
                    state.style_features_json
                )

            # Track A (sprint 2026-05-28): 載入 evidence_pool 供 outline planner
            # 做 per-chapter evidence allocation (LLM prompt 注入 + skeleton fallback
            # keyword match)。注意：下方 line 2308 也會 load 給 writer 用，但 plan_outline
            # 在前所以這裡先 load。Backward compat：state.evidence_pool_json 為空 →
            # 空 dict → plan_outline 走 backward compat (不注入 listing + validator skip context)。
            _evidence_pool_for_plan = deserialize_evidence_pool(state.evidence_pool_json)
            # 若空 dict → 傳 None 給 plan_outline (讓 validator 走 skip 路徑;
            # 區分「pool 真的空」與「caller 沒提供 pool」)
            _evidence_pool_arg = _evidence_pool_for_plan if _evidence_pool_for_plan else None

            if self.dry_run:
                # dry_run：skeleton fallback，不打 LLM
                outline = build_skeleton_outline(
                    chapter_source=writer_sections,
                    context_map=context_map,
                    format_specs=state.format_specs,
                    evidence_pool=_evidence_pool_arg,
                )
                logger.info(
                    f"[LIVE RESEARCH] Outline planner: dry_run skeleton outline "
                    f"(n={len(outline.chapters)})"
                )
            else:
                # 真實 LLM call；失敗 → Phase 4 skeleton fallback + 明示 narration
                # #8: outline planner 最長可達 90s 全靜窗口（call 前原本無 emit）,
                #     先推進度 narration 避免黑屏 + 助長 SSE idle 斷。
                await self._emit_narration("正在規劃整份報告的章節提綱...")
                try:
                    outline = await OutlinePlannerAgent(
                        self.handler,
                        timeout=CONFIG.reasoning_params.get("outline_planner_timeout", 90),
                    ).plan_outline(
                        chapter_source=writer_sections,
                        context_map=context_map,
                        format_specs=state.format_specs,
                        style_features=_style_features_for_plan,
                        evidence_pool=_evidence_pool_arg,
                    )
                    await self._emit_narration(
                        f"已規劃 {len(outline.chapters)} 章總提綱，開始逐章撰寫。"
                    )
                except Exception as e:
                    logger.warning(
                        f"[LIVE RESEARCH] Outline planner failed, using skeleton fallback: "
                        f"{type(e).__name__}: {e}"
                    )
                    outline = build_skeleton_outline(
                        chapter_source=writer_sections,
                        context_map=context_map,
                        format_specs=state.format_specs,
                        evidence_pool=_evidence_pool_arg,
                    )
                    # CLAUDE.md 紀律：降級必須有明確 user-visible narration，不可 silent fail
                    await self._emit_narration(
                        "總提綱規劃失敗，已降級為預設骨架（章節銜接 hint 可能較弱）。"
                        "後續仍會逐章撰寫。"
                    )

            state.book_outline_json = outline.model_dump_json()
            # R2 no silent fail 後衛：outline 定案後檢查有無 special_element target
            # 對不到任何章（reframe 改章名 / Stage 4 未解的殘留）。report-level 跑一次。
            if not getattr(self, "_special_element_unmatched_narrated", False):
                _all_se = (state.format_specs or {}).get("special_elements") or []
                _titles = [getattr(c, "title", "") for c in outline.chapters]
                _unmatched = _diagnose_unmatched_special_element_targets(_all_se, _titles)
                if _unmatched:
                    self._special_element_unmatched_narrated = True
                    await self._emit_narration(
                        lr_copy.special_element_target_unmatched_narration(_unmatched)
                    )
                    logger.warning(
                        f"[LIVE RESEARCH] special_element targets 對不到任何章: {_unmatched} "
                        f"(chapters={_titles})"
                    )
            # P2 W1（§0 #21，C2/C3）：evidence→章正向回填，涵蓋 LLM plan_outline +
            # skeleton fallback 兩路匯流（此處 outline 是兩路同一變數）。不放在
            # build_skeleton_outline（漏 LLM 主線 → prod 非 dry_run suggested_chapters 恆空）。
            from reasoning.agents.outline_planner import (
                invert_allocation_to_suggested_chapters,
            )
            _per_chapter = {
                i: ch.planned_evidence_ids for i, ch in enumerate(outline.chapters)
            }
            _suggested_map = invert_allocation_to_suggested_chapters(_per_chapter)
            if _evidence_pool_for_plan:
                for _eid, _chapters in _suggested_map.items():
                    _entry = _evidence_pool_for_plan.get(_eid)
                    if _entry is not None:
                        _entry.suggested_chapters = sorted(_chapters)
                # SF4/R2-4：mutate 後 serialize 回 json，否則 hint 只在 in-memory，
                # writer / revise / continue reload 時遺失。serialize 點 = 此處
                # （outline stage 兩路匯流點），非 Stage 1/2 的 evidence serialize。
                state.evidence_pool_json = serialize_evidence_pool(
                    _evidence_pool_for_plan
                )
            await self._persist_progress(state)
        # ────────────────────────────────────────────────────────────────────
        # Plan 2 Phase 3 (Option B-a): chapter override 模式下預先計算所有 evidence_ids
        # 聯集（sorted），交給 _write_section 在 chapter_index=0 分配給第一章；
        # 其餘 chapter analyst_citations=[]（writer prompt 提示「不要強行加 [N]」）。
        all_evidence_ids: list = []
        if using_chapter_override:
            union_ids: set = set()
            for t in context_map.topics:
                union_ids.update(t.evidence_ids)
            all_evidence_ids = sorted(union_ids)
            logger.info(
                f"[LIVE RESEARCH] Stage 5 using format_specs.chapters override "
                f"(n={total_sections})；cm.topics core count "
                f"={sum(1 for t in context_map.topics if t.relevance == 'core')} (ignored)；"
                f"all_evidence_ids={all_evidence_ids} (allocate to chapter[0])"
            )

        # Parse style features if available
        style_features = None
        if state.style_features_json:
            style_features = StyleAnalysisOutput.model_validate_json(state.style_features_json)

        # 載入 evidence_pool（Stage 1 / mock_bab 已持久化），供 Writer 看到真實 [N] 對應
        evidence_pool = deserialize_evidence_pool(state.evidence_pool_json)

        # ────────────────────────────────────────────────────────────────────
        # VP-7 single-step：只寫一段，然後 emit per-section checkpoint return。
        # ────────────────────────────────────────────────────────────────────
        next_i = state.last_completed_section_index + 1

        # ── Idempotent guard：若已全部寫完，直接 emit final checkpoint return ──
        if next_i >= total_sections:
            logger.info(
                f"[LIVE RESEARCH] Stage 5 already complete "
                f"(last_completed={state.last_completed_section_index}, total={total_sections})；"
                f"emit final checkpoint only"
            )
            await self._emit_writer_status({
                "status": "all_done",
                "completed": total_sections,
                "total_sections": total_sections,
            })
            proposal = "所有段落都完成了。需要修改哪個部分嗎？或者可以進入匯出階段？"
            state.set_checkpoint(proposal)
            state.stage5_waiting_for_user = True
            await self._emit_checkpoint(stage=5, proposal=proposal)
            await self._persist_checkpoint_boundary(state)  # plan: persist + offline-count
            return state

        # ── 離線檢查（plan: lr-sse-reconnect-resume, 2026-06-15 改語意）──
        # 舊行為「斷線就 abort return」與「斷線不取消、跑到 checkpoint 才停」矛盾 → 移除。
        # 新行為：離線時 mark offline_since + 檢查防呆上限；未達上限 → 繼續寫這一段
        #（寫完到 per-section checkpoint 才停存檔）；已達上限 → 標 capped、persist、停。
        alive = getattr(self.handler, 'connection_alive_event', None)
        offline = alive is not None and not alive.is_set()
        if offline:
            self._mark_offline_since(state)
            if self._offline_cap_reached(state):
                logger.warning(
                    f"[LIVE RESEARCH] Offline cap reached at Stage 5 section i={next_i}; "
                    f"stopping LR (reason={state.offline_cap_reason})"
                )
                state.offline_capped = True
                await self._persist_progress(state)
                return state
        # 未達上限（或仍在線）→ 照常寫這一段

        # Parse book_outline + 初始化 prev_summary（Plan 4 Phase 3）
        # Resume 路徑：從 written_sections[-1] 復原 prev_summary
        loop_book_outline = None
        if state.book_outline_json:
            try:
                loop_book_outline = BookOutline.model_validate_json(state.book_outline_json)
            except Exception as e:
                logger.warning(
                    f"[LIVE RESEARCH] Stage 5 failed to parse book_outline_json: "
                    f"{type(e).__name__}: {e} — writer falls back to no outline injection"
                )
                loop_book_outline = None

        previous_chapter_summary = ""
        if state.written_sections:
            previous_chapter_summary = state.written_sections[-1].get("chapter_summary", "")

        # ── Emit started（resume-aware completed 數）──
        state.stage_5_writer_running = True
        state.stage5_waiting_for_user = False
        already_done = max(state.last_completed_section_index + 1, 0)
        await self._emit_writer_status({
            "status": "started",
            "total_sections": total_sections,
            "completed": already_done,
        })

        # ── 寫第 next_i 段 ──
        section_spec = writer_sections[next_i]
        section_name = (
            section_spec["name"] if isinstance(section_spec, dict)
            else section_spec.name
        )

        try:
            # dry_run yield point：讓 cancel 有機會 race in
            if self.dry_run:
                await asyncio.sleep(0.05)

            await self._emit_narration(f"正在撰寫「{section_name}」段落...")

            # Track A Task 7 (sprint 2026-05-28): 蒐集前文已出現的實體傳給 writer
            # (綜合 / 結論章 prompt 紀律: 不可引入前文未提及的新實體)。
            # nested key 訪問必用 .get (backward compat: 舊 row 無 'entities' key)。
            _prior_entities_acc: List[str] = []
            for prior in state.written_sections[:next_i]:
                _prior_entities_acc.extend(prior.get("entities", []))
            # dedupe 保 order
            _seen_ent: set = set()
            prior_used_entities_for_chapter: List[str] = []
            for e in _prior_entities_acc:
                if e and e not in _seen_ent:
                    _seen_ent.add(e)
                    prior_used_entities_for_chapter.append(e)

            # B (Cayenne cross-section): 蒐集所有前章摘要（不只最後一章），
            # 供 synthesis 章 prompt 注入「前面各章實際寫了什麼」。
            all_prior_chapter_summaries: List[str] = []
            for prior in state.written_sections[:next_i]:
                _summ = prior.get("chapter_summary", "")
                if _summ:
                    all_prior_chapter_summaries.append(_summ)

            section_output, was_corrected = await self._write_section(
                context_map=context_map,
                topic=section_spec,
                style_features=style_features,
                format_specs=state.format_specs,
                evidence_pool=evidence_pool,
                chapter_index=next_i if using_chapter_override else None,
                all_evidence_ids=all_evidence_ids if using_chapter_override else None,
                book_outline=loop_book_outline,
                current_chapter_index=next_i,
                previous_chapter_summary=previous_chapter_summary,
                user_voice=state.user_voice,
                state=state,  # Track A Task 3: render_grounded_narrative needs state.evidence_usage
                prior_used_entities=prior_used_entities_for_chapter,  # Track A Task 7
                all_prior_chapter_summaries=all_prior_chapter_summaries,  # B
            )
            # Propagate Hallucination Guard 觸發狀態到 state（Stage 6 narration 用）
            state.hallucination_corrected = state.hallucination_corrected or was_corrected

            # Track A Task 7: 寫完抽 entities → 存進 written_sections[i]["entities"]
            # 後續章節 prior_used_entities 用 (跨章 coherence)。
            # blocked_no_evidence / guard_failed 章節 content 已是 blocked 文字, 抽不到
            # 有意義 entity → 仍跑但回 [] 不阻塞 (entity extractor LLM-failure 已容錯)。
            _section_entities: List[str] = []
            if getattr(section_output, "status", "drafted") == "drafted":
                # A: _write_section 內 specificity / (a) gate 已抽過 → 重用，避免重複 LLM call
                _section_entities = getattr(section_output, "_composed_entities", None)
                if _section_entities is None:
                    _section_entities = await self._extract_section_entities(
                        section_output.section_content, self.handler,
                    )

            # addendum I-1: 抽 _section_dict helper 統一構造 (3 mutation 點)
            state.written_sections.append(
                _section_dict(section_output, next_i, entities=_section_entities)
            )
            state.last_completed_section_index = next_i

            # 推送 section 到前端
            await self._emit_section(next_i, section_output, state)

            await self._emit_writer_status({
                "status": "section_done",
                "total_sections": total_sections,
                "completed": next_i + 1,
                "section_title": section_output.section_title,
            })

        except (ValueError, asyncio.TimeoutError) as e:
            # Stage 5 writer LLM-fail 降級（Sentry #7537040772）：
            # base.py:398 raise ValueError("LLM returned empty response...")
            # 或 asyncio.TimeoutError — 屬系統端問題，不可 raw 噴給 user。
            # 對齊 Stage 3 LLM-fail 降級 pattern（commit 4fbe9fc8）：
            # emit lr_copy.LLM_UNAVAILABLE_NARRATION + re-emit 該段 checkpoint + return state。
            # 不可 silent fail — 明確 narration；不可 silent 跳過 — 不推進 last_completed。
            logger.error(
                f"[LIVE RESEARCH] Stage 5 writer LLM-fail (degrading): "
                f"section_name={section_name!r} section_i={next_i} "
                f"{type(e).__name__}: {e}"
            )
            # 友善 narration（系統端，不怪 user）
            await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
            # Re-emit per-section checkpoint：讓 user 可以重試本段
            _completed_so_far = next_i  # next_i 尚未完成，completed count = next_i（前幾段）
            _total = total_sections
            _retry_proposal = (
                f"第 {_completed_so_far}/{_total} 段「{section_name}」寫作時 AI 服務暫時沒有回應。"
                f"你可以：(1) 再試一次（回覆「繼續」）、或 (2) 修改前面已寫好的某一段。"
            )
            state.set_checkpoint(_retry_proposal)
            state.stage5_waiting_for_user = True
            await self._emit_checkpoint(stage=5, proposal=_retry_proposal)
            await self._persist_checkpoint_boundary(state)  # plan: persist + offline-count
            return state

        except asyncio.CancelledError:
            logger.info(
                f"[LIVE RESEARCH] Stage 5 cancelled at section i={next_i} "
                f"(last_completed={state.last_completed_section_index})"
            )
            state.stage_5_writer_running = False
            # 連線已死，保留 state 給 resume；必須 re-raise，否則 task wrap 收不到訊號
            raise

        finally:
            state.stage_5_writer_running = False

        # ── 完成第 next_i 段：emit per-section checkpoint ──
        completed = next_i + 1
        remaining = total_sections - completed

        if remaining == 0:
            # 寫完最後一段 → final checkpoint
            await self._emit_writer_status({
                "status": "all_done",
                "completed": total_sections,
                "total_sections": total_sections,
            })
            proposal = (
                f"全部 {total_sections} 段都完成了。需要修改哪個部分嗎？"
                f"或者可以進入匯出階段？"
            )
        else:
            # 中段 checkpoint：兩選一（#11，2026-06-03）— 移除「直接匯出」選項。
            # 寫到一半（如 2/5 段）就提供直接匯出 = 鼓勵交殘缺報告；匯出僅在全部寫完的
            # final checkpoint（上方 remaining==0 分支）才出現。
            proposal = (
                f"第 {completed}/{total_sections} 段完成（剛寫完「{section_name}」）。"
                f"接下來要：(1) 繼續寫第 {completed + 1} 段、還是 (2) 修改已寫好的某一段？"
            )

        state.set_checkpoint(proposal)
        state.stage5_waiting_for_user = True
        await self._emit_checkpoint(stage=5, proposal=proposal)
        await self._persist_checkpoint_boundary(state)  # plan: persist + offline-count
        return state

    async def _persist_progress(self, state: LiveResearchStageState) -> None:
        """Persist mid-loop state so resume can see it.

        Wraps `handler._save_state` with explicit error handling — save fail
        is logged + raised (no silent fail per CLAUDE.md). Writer caller
        bubbles up; the asyncio task's `_on_lr_research_complete` callback
        logs the exception.
        """
        try:
            await self.handler._save_state(state)
        except Exception as e:
            logger.error(
                f"[LIVE RESEARCH] _persist_progress: save failed: {e}",
                exc_info=True,
            )
            raise

    # ──── 離線防呆燒錢上限 helpers（plan: lr-sse-reconnect-resume, 2026-06-15）────

    def _mark_offline_since(self, state: LiveResearchStageState) -> None:
        """首次偵測離線時，把 offline 起點寫進 state（跨 instance 持久化，防重連歸零）。

        state.offline_since 已有值就不覆寫（重連仍離線時保留原始起點）。
        """
        if getattr(state, "offline_since", None) is None:
            since = getattr(self.handler, "_client_offline_since", None)
            state.offline_since = since if since is not None else time.time()

    def _offline_cap_reached(self, state: LiveResearchStageState) -> bool:
        """離線後是否已達任一防呆上限。只在 connection_alive_event 已 clear 時被呼叫。

        上限狀態讀自 state（DB 持久化），不讀 orchestrator instance counter
        （CEO 拍板：instance counter 重連歸零防不住「斷→連→斷→連」燒錢）。
        config 用扁平 key（對齊既有 analyst_timeout 慣例），非巢狀 live_research dict。
        """
        # wall-clock：自 state.offline_since 起算
        since = getattr(state, "offline_since", None)
        if since is not None:
            elapsed = time.time() - since
            if elapsed >= CONFIG.reasoning_params.get("offline_max_wall_seconds", 900):
                state.offline_cap_reason = "wall_seconds"
                return True
        # 跨 checkpoint 上限：離線後已前進的 checkpoint 數（進 state）
        advances = getattr(state, "offline_checkpoint_advances", 0)
        if advances >= CONFIG.reasoning_params.get("offline_max_checkpoint_advances", 1):
            state.offline_cap_reason = "next_checkpoint"
            return True
        return False

    def _maybe_reset_offline_counters(self, state: LiveResearchStageState) -> None:
        """重置離線計數 — 只在「已通過 intent validation、確定推進 workflow 的 reply」之後。

        正確條件（plan 3d + R3-verify nit）= 兩者皆成立：
        1. **substantive advance 確認**：呼叫此 helper 的位置是 `_run_stage_N` 真正往前跑
           stage 的進入點（vague / invalid / abort reply 不會到 `_run_stage_N`，只 set
           checkpoint 後從 `_handle_stage_N_response` return）→ 到這裡 = 確定推進。
        2. **client 在線**：離線中的 auto-advance（如 offline 跑到 _run_stage_5 寫下一段）
           **絕不** reset，否則 cap 永遠歸零、防呆失效。只有重連後送 substantive reply
           （online）才 reset。

        read-only reconnect（無 POST /continue）不會進 orchestrator → 自然不 reset。
        """
        alive = getattr(self.handler, "connection_alive_event", None)
        online = alive is None or alive.is_set()
        if not online:
            return  # 離線 auto-advance 不 reset（保住 cap）
        if (state.offline_since is not None
                or state.offline_capped
                or state.offline_checkpoint_advances):
            logger.info(
                "[LIVE RESEARCH] Online substantive advance — resetting offline counters "
                f"(was advances={state.offline_checkpoint_advances}, "
                f"capped={state.offline_capped})"
            )
        state.offline_since = None
        state.offline_capped = False
        state.offline_cap_reason = ""
        state.offline_checkpoint_advances = 0

    async def _persist_checkpoint_boundary(self, state: LiveResearchStageState) -> None:
        """每個 durable boundary（set_checkpoint / complete_stage）return 前統一呼叫。

        職責（plan Task 2 + 3d）：
        1. 離線時：mark offline_since + 跨 checkpoint 計數（off-by-one 順序寫死：
           increment → 立刻判 capped → 才 persist），per-call guard 確保一次 continue
           只 +1（即使同一 call 穿越多個 boundary）。
        2. persist state（idempotent：重複呼叫覆寫同 row；連線正常時也存，等同既有行為）。
        """
        alive = getattr(self.handler, "connection_alive_event", None)
        offline = alive is not None and not alive.is_set()
        if offline:
            self._mark_offline_since(state)
            # 順序寫死（Codex off-by-one）：先 increment、再立刻判上限並標 capped、最後才 persist。
            # per-call guard：一次 continue call 只計一次（同 call 多 boundary 不重複加）。
            if not self._offline_advance_counted_this_call:
                state.offline_checkpoint_advances = (
                    getattr(state, "offline_checkpoint_advances", 0) + 1
                )
                self._offline_advance_counted_this_call = True
                _max = CONFIG.reasoning_params.get("offline_max_checkpoint_advances", 1)
                if state.offline_checkpoint_advances >= _max:
                    state.offline_capped = True
                    state.offline_cap_reason = "next_checkpoint"
                    logger.warning(
                        f"[LIVE RESEARCH] Offline cap reached (next_checkpoint): "
                        f"advances={state.offline_checkpoint_advances} >= max={_max}; "
                        f"LR paused for offline protection (stage={state.current_stage})"
                    )
        await self._persist_progress(state)

    async def _emit_writer_status(self, payload: dict) -> None:
        """Emit `live_research_writer_status` SSE event.

        Schema:
        - status: "started" / "section_done" / "all_done"
        - total_sections: int
        - completed: int (0-indexed count of done sections)
        - section_title: str (only on "section_done")
        """
        msg = {"message_type": "live_research_writer_status"}
        msg.update(payload)
        await emit_sse(self.handler, msg)

    async def _apply_degraded_grounding_unavailable(
        self,
        section_output: Any,
        analyst_citations: List[int],
        current_chapter_index: int,
        reason: str,
        evidence_pool: Optional[Dict[int, Any]] = None,   # P2 W7 I1（§0 #23）：全 pool 合法集
    ) -> Any:
        """R1 fail-closed：grounding 判讀 LLM 不可用（exception / 爆窗 / 無法解析）時的
        DR 式退化。**正文一字不改**（不知哪句有問題，不可亂刪）+ confidence 降 Low +
        methodology note 明確標「grounding 系統驗證失敗，本章未經完整查證」。**絕不當作
        全 grounded 放行**（fail-open），錯誤必須浮現。

        D-2026-06-11 決策1（o5a F3 解凍）：退化時附帶一次即時 SSE 旁白 ——
        report 內標註要等匯出才看得到，user 當下必須知道防禦層降級。
        本 method 是單一落點，蓋 _write_section 三個 except GroundingCheckUnavailable
        呼叫點；async 化即為了能 await _emit_narration（呼叫端皆在 async 流程內）。
        旁白 per-run dedup（_grounding_unavailable_narrated）；log 每章照記。
        """
        logger.error(
            f"[LIVE RESEARCH][Chapter:{current_chapter_index}] grounding 判讀系統失敗 "
            f"→ fail-CLOSED 退化（保留正文 + 降 Low + 標未查證）。reason={reason}"
        )
        if not self._grounding_unavailable_narrated:
            self._grounding_unavailable_narrated = True
            await self._emit_narration(lr_copy.GROUNDING_UNAVAILABLE_NARRATION)
        # P2 W7 I1（§0 #23）：退化時保留全 pool 合法引用（非只 analyst_citations 交集），
        # 否則砍掉 pool 內 analyst_citations 外的合法引用 → 線上偶發引用流失。
        # 0（{cite:0} placeholder）一律視為合法（與 W8 allowed ∪ {0} 對齊）。
        _allowed_sources = set((evidence_pool or {}).keys()) | set(analyst_citations or []) | {0}
        _kept_sources = [
            s for s in (section_output.sources_used or [])
            if s in _allowed_sources
        ]
        _existing = section_output.methodology_note or ""
        _note = lr_copy.GROUNDING_UNAVAILABLE_NOTE
        return section_output.model_copy(update={
            # section_content 不動（不知哪句幻覺，不可亂刪）
            "sources_used": _kept_sources,
            "confidence_level": "Low",
            # status 維持 drafted —— 退化非整章失敗；下游 citation render 照常跑
            "methodology_note": (f"{_existing} {_note}".strip() if _existing else _note),
        })

    def _narrate_grounding_extraction_failed(self) -> None:
        """Task 3 callback：抽取層 LLM 故障時由 guard 同步呼叫。只 set pending flag
        （sync 安全），實際 SSE 旁白於 _write_section async 流程內由
        _emit_grounding_extraction_failed_if_pending dedup-gated 補播（emit 是 async）。"""
        self._grounding_extraction_failed_pending = True

    async def _emit_grounding_extraction_failed_if_pending(self) -> None:
        """Task 3：三個 entity_grounding_check callsite 共用的 dedup emit helper。
        每個 await 成功返回後呼叫；pending（本 run 曾有抽取故障）且尚未播報時播一次。
        per-run dedup（narrated flag）；三 callsite 共用同一 helper 避免複製漏改。"""
        if (self._grounding_extraction_failed_pending
                and not self._grounding_extraction_failed_narrated):
            self._grounding_extraction_failed_narrated = True
            await self._emit_narration(
                lr_copy.GROUNDING_EXTRACTION_FAILED_NARRATION
            )

    def _apply_partial_or_degraded_block(
        self,
        section_output: Any,
        ungrounded: List[str],
        analyst_citations: List[int],
        current_chapter_index: int,
        label: str,
        grounded_entities: Optional[List[str]] = None,
        evidence_pool: Optional[Dict[int, Any]] = None,   # P2 W7 I1（§0 #24）：全 pool 合法集
    ) -> Tuple[Any, bool]:
        """Fix 2 (CEO 決策④): 主路徑 (b) sentence-level partial block（只刪「純未驗證句」，
        保留其餘有據 prose）。當刪句會刪掉過多／不安全時，退化路徑 (a) 採 DR 做法——
        **正文一字不改** + sources_used 取與 analyst_citations 交集移除 invalid 引用 +
        confidence 降 Low + methodology note 標註哪幾個 entity 未驗證。
        **絕不整章替換成 [本章內容無法驗證]（丟掉的 (c)）。**

        R3：`grounded_entities` 傳給 split helper 做句子分類——混合句（含已驗證 entity）/
            含 citation / 上下文依賴句**不硬刪**，由 split 回報 `unsafe`；unsafe>0 即代表
            有「不可安全硬刪」的句子殘留 → 走退化 (a)（保留＋標註，不 LLM 改寫）。
        R5：退化（partial→degraded）條件**不只看字數**，加語意維度：
            `removed_sentence_ratio`（刪句比例過高）和 `citation_loss_ratio`（citation
            流失比例過高）也觸發退化。

        Returns:
            (new_section_output, was_degraded)  # was_degraded=True 代表走退化路徑 (a)
        """
        from reasoning.live_research.hallucination_guard import (
            split_and_filter_ungrounded_sentences,
            _CITATION_RE,
        )
        _content = section_output.section_content or ""
        _kept, _removed, _unsafe = split_and_filter_ungrounded_sentences(
            _content, ungrounded_entities=ungrounded,
            grounded_entities=grounded_entities or [],
        )
        _kept = _kept.strip()

        # R5 語意維度：刪句比例 / citation 流失比例
        _total_sentences = max(1, _removed + _content.count("。") + _content.count("\n"))
        _removed_sentence_ratio = _removed / _total_sentences
        _cites_before = len(_CITATION_RE.findall(_content))
        _cites_after = len(_CITATION_RE.findall(_kept))
        _citation_loss_ratio = (
            (_cites_before - _cites_after) / _cites_before if _cites_before else 0.0
        )

        _degenerate = (
            len(_kept) < 150                                  # 字數絕對下限
            or len(_kept) < int(len(_content) * 0.30)         # 字數相對下限
            or _unsafe > 0                                    # R3：有不可安全硬刪的句子
            or _removed_sentence_ratio > 0.50                 # R5：刪句比例過高
            or _citation_loss_ratio > 0.50                    # R5：citation 流失過高
        )
        if _degenerate:
            # 退化路徑 (a) — DR-style：正文不動，只降 confidence + methodology note 標註。
            # （範本：純 DR orchestrator.py:1095-1131）
            logger.warning(
                f"[LIVE RESEARCH][Chapter:{current_chapter_index}][{label}] "
                f"刪句不安全/過多（kept={len(_kept)} chars, removed={_removed}, "
                f"unsafe={_unsafe}, removed_ratio={_removed_sentence_ratio:.2f}, "
                f"citation_loss={_citation_loss_ratio:.2f}, ungrounded={ungrounded}）"
                f"→ 退化路徑 (a)：保留正文、降 Low、methodology 標註。"
            )
            # P2 W7 I1（§0 #24）：sources_used 取與「全 pool 合法集」交集（非只
            # analyst_citations），移除 invalid 引用但保留 pool 內合法引用。0 placeholder 視為合法。
            _allowed_sources = (
                set((evidence_pool or {}).keys()) | set(analyst_citations or []) | {0}
            )
            _kept_sources = [
                s for s in (section_output.sources_used or [])
                if s in _allowed_sources
            ]
            _existing = section_output.methodology_note or ""
            _degrade_note = lr_copy.degraded_low_confidence_note(ungrounded)
            new_out = section_output.model_copy(update={
                # section_content 維持原值（正文一字不改）—— DR 做法核心
                "sources_used": _kept_sources,
                "confidence_level": "Low",
                # status 維持 drafted —— 退化非整章失敗；下游 citation render 照常跑
                "methodology_note": (
                    f"{_existing} {_degrade_note}".strip() if _existing else _degrade_note
                ),
            })
            return new_out, True
        # 主路徑 (b) — sentence-level partial：只刪純未驗證句，保留其餘（含混合/citation/依賴句）。
        logger.warning(
            f"[LIVE RESEARCH][Chapter:{current_chapter_index}][{label}] "
            f"sentence-level partial block: 刪 {_removed} 句純未驗證句含 ungrounded "
            f"entity {ungrounded}，保留其餘 prose（kept={len(_kept)} chars）。"
        )
        _note_prefix = lr_copy.partial_removed_note(_removed, ungrounded)
        _existing = section_output.methodology_note or ""
        new_out = section_output.model_copy(update={
            "section_content": _kept,
            "confidence_level": "Low",
            # status 維持 drafted —— partial 非整章失敗；下游 citation render 照常跑保留段
            "methodology_note": (
                f"{_existing} {_note_prefix}".strip() if _existing else _note_prefix
            ),
        })
        return new_out, False

    async def _extract_section_entities(
        self, content: str, handler: Any
    ) -> List[str]:
        """Track A Task 7 wrapper: 讓 test 可 monkeypatch class method
        (module-level _extract_entities_from_section 是實作底層)。"""
        return await _extract_entities_from_section(content, handler)

    async def _run_publish_gate(
        self,
        section_output: "Any",  # LiveWriterSectionOutput
        current_chapter_index: int,
        chapter_evidence_text: str,
        state: Optional[Any] = None,
    ) -> Tuple[Any, bool]:
        """Track F (sprint 2026-05-28): F1 critic publish gate + F3 CoV-lite (I-4 helper)。

        三層防禦第三層。I-5 紀律：先合併 F1 / F3 verdict 升級，最後一次性 mutate content。

        Path:
          1. status != "drafted" → short-circuit pass-through（F-AMB-7）
          2. F1 critic call → f1_review_initial（含 fail-loud WARN fallback per C-3）
          3. F3 CoV-lite call（如 f1_review_initial.verdict != REJECT）→ cov_summary
          4. F3 auto-escalate → f1_review_final
          5. 根據 f1_review_final.verdict 統一 mutate（REJECT/WARN/PASS branch）
          6. 寫進 state.critic_section_reviews（含 cov_verification_summary）

        Returns:
            (section_output, was_corrected)
        """
        _f1_enabled = self.features.get("live_research_critic_publish_gate", False)
        _f3_enabled = self.features.get("cov_lite_enabled", False)
        # S-2: 可選 LR-only 子 flag 覆寫（acceptance 量測若 LR cov 比 DR 貴 → 立即關 F3）
        _f3_lr_enabled = self.features.get(
            "live_research_cov_lite_enabled", _f3_enabled
        )
        _current_status = getattr(section_output, "status", "drafted")

        if not _f1_enabled or _current_status != "drafted":
            return section_output, False  # F-AMB-7 short-circuit

        # Task 2: 空 / 純空白 evidence 短路。safe-init "" 在 entity-guard try 早期
        # raise 時會流到這裡（orchestrator chapter_evidence_text="" safe-init）。
        # 零 evidence 進 F1/F3 = 沒東西可比對卻要 LLM 判，輸出不可信且純燒 high-tier
        # call → deterministic 短路：不打 LLM，明確標「查無可審來源」+ 降 Low。
        # 不 silent PASS（無據章不可偽裝成已審）。
        if not (chapter_evidence_text or "").strip():
            logger.warning(
                f"[LIVE RESEARCH F1] section {current_chapter_index} "
                f"chapter_evidence_text 為空 → 短路發布審查（標查無來源 + 降 Low）"
            )
            _existing = section_output.methodology_note or ""
            _note = lr_copy.PUBLISH_GATE_NO_EVIDENCE_NOTE
            section_output = section_output.model_copy(update={
                "confidence_level": "Low",
                "methodology_note": (
                    f"{_existing} {_note}".strip() if _existing else _note
                ),
            })
            return section_output, True

        was_corrected = False

        try:
            # C-2 (NF-2 R2 fix 2026-05-29): 從 state.evidence_usage flatten 取
            # BAB Critic 已 WARN entries
            # **重要**: state.evidence_usage 實際 storage type 是 Dict[int, List[Dict]]
            # （GroundedClaim.model_dump() 過的 dict），**不是** List[GroundedClaim]
            # object — 必須用 dict access (`_c.get(...)`) 而非 attr access
            # (`getattr(_c, ...)`)。證據：stage_state.py from_dict 內
            # `evidence_usage: Dict[int, List[Dict]] = {}`。
            warned_critic_claims: List[Dict] = []
            if state is not None and hasattr(state, "evidence_usage"):
                for _eid, _claims in state.evidence_usage.items():
                    for _c in _claims:
                        # dict access (not getattr — getattr on dict 永遠回 default)
                        if isinstance(_c, dict) and _c.get(
                            "from_warned_critic_review", False
                        ) is True:
                            warned_critic_claims.append(_c)

            # I-7: 從 state.time_constraint 撈（Track E land 後生效）
            time_constraint = getattr(state, "time_constraint", None) if state else None

            # === Step 2: F1 critic call ===
            f1_review_initial = await self.critic_agent.review_section_publish_gate(
                section=section_output,
                section_index=current_chapter_index,
                chapter_evidence_text=chapter_evidence_text,
                warned_critic_claims=warned_critic_claims or None,
                time_constraint=time_constraint,
            )

            # === Step 3: F3 CoV-lite (skip if F1 already REJECT) ===
            # I-5: 對 raw section_content 跑 CoV，不對 mutated blocked 文字
            cov_summary = None
            if _f3_lr_enabled and f1_review_initial.verdict != "REJECT":
                try:
                    cov_summary = await self.critic_agent.run_cov_for_lr_section(
                        section_content=section_output.section_content,
                        chapter_evidence_text=chapter_evidence_text,
                    )
                except Exception as e:
                    # I-2: F3 fail 不可 silent fail → degraded result
                    logger.warning(
                        f"[LIVE RESEARCH F3] CoV-lite verification failed (non-fatal): "
                        f"{type(e).__name__}: {e}"
                    )
                    cov_summary = {
                        "verification_status": "unverified",
                        "verification_message": (
                            f"F3 CoV-lite failed ({type(e).__name__}: {e}); "
                            f"unverified per fail-loud discipline"
                        ),
                        "verified_count": 0,
                        "unverified_count": 0,
                        "contradicted_count": 0,
                        "results": [],
                    }

            # === Step 4: F3 auto-escalate verdict (沿 DR critic.py pattern) ===
            f1_review_final = f1_review_initial.model_copy(update={
                "cov_verification_summary": cov_summary,
            })
            if cov_summary:
                contradicted = cov_summary.get("contradicted_count", 0)
                unverified = cov_summary.get("unverified_count", 0)
                if contradicted > 0 and f1_review_final.verdict != "REJECT":
                    f1_review_final = f1_review_final.model_copy(update={
                        "verdict": "REJECT",
                        "overall_explanation": (
                            f1_review_final.overall_explanation +
                            f" [F3 CoV contradicted {contradicted} claims → "
                            f"auto-escalate REJECT]"
                        ),
                    })
                    logger.warning(
                        f"[LIVE RESEARCH F3] section {current_chapter_index} "
                        f"auto-escalate to REJECT due to {contradicted} contradicted claims"
                    )
                elif unverified >= 3 and f1_review_final.verdict == "PASS":
                    f1_review_final = f1_review_final.model_copy(update={
                        "verdict": "WARN",
                        "overall_explanation": (
                            f1_review_final.overall_explanation +
                            f" [F3 CoV {unverified} unverified claims → escalate WARN]"
                        ),
                    })
                    logger.info(
                        f"[LIVE RESEARCH F3] section {current_chapter_index} "
                        f"escalate to WARN due to {unverified} unverified claims"
                    )

            # === Step 5: 統一 mutate based on f1_review_final.verdict (I-5) ===
            if f1_review_final.verdict == "REJECT":
                blocked_content_f1 = lr_copy.critic_rejected_content(
                    len(f1_review_final.claim_issues),
                    [i.claim_text for i in f1_review_final.claim_issues],
                )
                section_output = section_output.model_copy(update={
                    "section_content": blocked_content_f1,
                    "sources_used": [],
                    "confidence_level": "Low",
                    "status": "critic_rejected",
                })
                was_corrected = True
                logger.warning(
                    f"[LIVE RESEARCH F1] section {current_chapter_index} "
                    f"verdict=REJECT, replaced content with blocked text "
                    f"(claim_issues={len(f1_review_final.claim_issues)})"
                )
            elif f1_review_final.verdict == "WARN":
                # I-1: WARN marker dedup — append 前先檢查既有 marker
                existing_note = section_output.methodology_note or ""
                warn_marker = lr_copy.warn_marker(
                    len(f1_review_final.claim_issues),
                    f1_review_final.overall_explanation,
                )
                _had_marker = (
                    lr_copy.WARN_MARKER_PREFIX in existing_note
                    or lr_copy.LEGACY_WARN_MARKER_PREFIX in existing_note
                )
                if _had_marker:
                    # I-1: 已存在 marker（user revise 重跑 case，含舊 session 舊
                    # marker）→ replace 不 append
                    import re as _re
                    new_note = _re.sub(
                        lr_copy.WARN_MARKER_DEDUP_RE,
                        warn_marker,
                        existing_note,
                    ).strip()
                else:
                    new_note = (
                        f"{existing_note} {warn_marker}".strip()
                        if existing_note else warn_marker
                    )
                section_output = section_output.model_copy(update={
                    "methodology_note": new_note,
                })
                logger.info(
                    f"[LIVE RESEARCH F1] section {current_chapter_index} "
                    f"verdict=WARN, marker "
                    f"{'replaced' if _had_marker else 'added'} "
                    f"(claim_issues={len(f1_review_final.claim_issues)})"
                )
            else:
                # PASS — status / content 不變
                logger.info(
                    f"[LIVE RESEARCH F1] section {current_chapter_index} verdict=PASS"
                )

            # === Step 6: 寫進 state.critic_section_reviews ===
            if state is not None:
                state.critic_section_reviews[current_chapter_index] = (
                    f1_review_final.model_dump()
                )

        except Exception as e:
            # Task 1 (CEO 拍板，default=degrade-and-narrate)：拆除原 fail-open 原樣放行。
            # 此 except 接的是 gate body 非預期例外（F1/F3 各自 LLM 失敗已由內層
            # except 處理成 fail-loud WARN/degraded，到不了這裡；這裡多是 state
            # mutation / model_copy / lr_copy 等基建層意外）。
            # 保守方向（與 R1 grounding 退化同構）：保留正文（不知哪句有問題，不亂刪）
            # + 降 Low + methodology 標「本章未經發布審查」+ 一次即時旁白。
            # 絕不 silent 原樣放行（違鐵律），也不整章 block（gate 自身故障不該砍內容）。
            logger.error(
                f"[LIVE RESEARCH F1] _run_publish_gate failed unexpectedly "
                f"→ degrade-and-narrate（保留正文 + 降 Low + 標未審）: "
                f"{type(e).__name__}: {e}"
            )
            _existing = section_output.methodology_note or ""
            _note = lr_copy.PUBLISH_GATE_UNAVAILABLE_NOTE
            section_output = section_output.model_copy(update={
                "confidence_level": "Low",
                "methodology_note": (
                    f"{_existing} {_note}".strip() if _existing else _note
                ),
            })
            was_corrected = True
            if not self._publish_gate_unavailable_narrated:
                self._publish_gate_unavailable_narrated = True
                await self._emit_narration(lr_copy.PUBLISH_GATE_UNAVAILABLE_NARRATION)

        return section_output, was_corrected

    @staticmethod
    def _common_writer_kwargs(state, chapter_sufficiency):
        """組裝 _write_section 四個 compose_section callsite 共用的「背景透傳」參數。

        FIX-5（Architect I-3）：`time_constraint` / `evidence_sufficiency` /
        `knowledge_graph` 是 normal + 三條 rewrite path 都必須帶的背景參數——漏帶任一
        會 silent 退化（writer 簽名 default None → 那一章悄悄掉 KG / calibration /
        時間約束，跨章一致性被破壞且無 test/type 報錯）。集中於此單一來源，新增 rewrite
        callsite 時 `**self._common_writer_kwargs(...)` 或顯式取值即可，不再四處複製
        value-derivation 表達式。state=None 時各參數 fallback None（沿既有語意）。
        """
        return {
            "time_constraint": (state.time_constraint if state is not None else None),
            "evidence_sufficiency": chapter_sufficiency,
            "knowledge_graph": (state.knowledge_graph if state is not None else None),
        }

    async def _write_section(
        self,
        context_map,
        topic,
        style_features,
        format_specs,
        evidence_pool=None,
        chapter_index=None,
        all_evidence_ids=None,
        book_outline=None,
        current_chapter_index=0,
        previous_chapter_summary="",
        user_voice=None,
        revise_instruction=None,
        prior_section_content=None,
        prior_sources_used=None,
        state=None,
        prior_used_entities=None,
        all_prior_chapter_summaries=None,
    ):
        """呼叫 WriterAgent.compose_section() 寫單一 section。

        透過 compose_section() 獲得 boundary token 隔離、citation whitelist、
        結構化 style injection 等安全特性。

        Args:
            topic: 可為 ContextMapTopic（既有 core_topics 路徑）或
                Dict[str, str]（Plan 2 Phase 2: format_specs.chapters override，
                {"name": ..., "outline": ...}）。chapter override 時沒對應
                topic_id / evidence_ids — Phase 3 經 chapter_index + all_evidence_ids
                allocate Option B-a union-to-first 白名單。
            chapter_index: chapter override 模式下，本章在 chapters list 中的 0-based index。
                index=0 → analyst_citations 取 all_evidence_ids（union）；
                index>0 → analyst_citations=[]（writer prompt 提示「不要強行加 [N]」）。
                None 時不啟用 Phase 3 allocation（既有 core_topics 路徑）。
            all_evidence_ids: chapter override 模式下，cm.topics 所有 evidence_ids 的
                聯集（sorted list）。由 _run_stage_5 預先計算後傳入。
            evidence_pool: 全局 evidence_pool dict（從 state.evidence_pool_json 反序列化）。
                None 時保持舊行為（Writer 不看真實 evidence 對照表）。
            prior_sources_used: Track B2 (sprint 2026-05-28) revise path 專用。
                caller（_handle_stage_5_response）傳入 prior section 的 sources_used。
                chapter_index>0 時（原本 analyst_citations=[]），若 prior_sources_used
                非空，fallback 到 prior_sources_used（白名單過濾），避免 revision 引用全掉。
                None = 非 revise path，行為不變。
                [] = revise path 但 prior section 本來就無引用，行為不變。

        Returns:
            Tuple[LiveWriterSectionOutput, bool]: (section_output, was_corrected)
            was_corrected=True 時表示 Hallucination Guard 觸發過自動修正，
            caller 須 propagate 進 state.hallucination_corrected。
        """
        from reasoning.live_research.hallucination_guard import apply_hallucination_guard

        # 白名單：Track B3 (sprint 2026-05-28) — 改用 evidence_pool.keys() 取代
        # ContextMap.topics.evidence_ids 聯集。
        # 原因：evidence_pool 是 BAB engine 的全集（119 筆），topics 聯集可能只有 30 筆；
        # writer 引用合法 pool ID 但不在 topics 聯集時，guard 誤 strip → 改用 pool 全集。
        # evidence_pool=None（舊 caller / legacy path）→ fallback 到 topics 聯集（向後兼容）。
        if evidence_pool:
            valid_ids: set = set(evidence_pool.keys())
        else:
            valid_ids = set()
            for t in context_map.topics:
                valid_ids.update(t.evidence_ids)

        # Plan 2 Phase 2/3: 統一 section_spec 介面（ContextMapTopic vs chapter dict）
        is_chapter_override = isinstance(topic, dict)
        if is_chapter_override:
            section_title = topic.get("name", "")
            section_outline = topic.get("outline", "")
            section_topic_id = None
            # Phase 3 (Option B-a union-to-first): index 0 拿所有 evidence_ids，其餘空。
            if chapter_index == 0 and all_evidence_ids:
                analyst_citations: list = list(all_evidence_ids)
            else:
                analyst_citations = []
            # Track B2 (sprint 2026-05-28): revise path fallback —
            # chapter_index>0 原本設 analyst_citations=[]，導致 revision 引用全掉。
            # 若 caller 傳入 prior_sources_used（非 None 且非空），用 prior sources
            # 作為 fallback（白名單過濾），保留上一版引用給 writer 參考。
            if not analyst_citations and prior_sources_used:
                analyst_citations = [
                    eid for eid in prior_sources_used if eid in valid_ids
                ]
                logger.debug(
                    f"[LIVE RESEARCH] B2 revise fallback: chapter_index={chapter_index} "
                    f"prior_sources_used={prior_sources_used} → "
                    f"analyst_citations={analyst_citations}"
                )
        else:
            section_title = topic.name
            section_outline = topic.description
            section_topic_id = topic.topic_id
            # P2 W2：topic.evidence_ids 只是初值；下游 W3 全 pool evidence_lookup 蓋過
            # writer 可見集，此 list 退居優先 tier 排序提示，非白名單邊界。
            analyst_citations = list(topic.evidence_ids) if topic.evidence_ids else []

        # Plan 4 Phase 3: 升級 analyst_citations — 若有 book_outline，改用
        # ChapterPlan.planned_evidence_ids（LLM-assisted allocation），通過 valid_ids
        # 白名單過濾。Skeleton fallback 的 build_skeleton_outline 已把 union-to-first
        # 邏輯 encode 進 planned_evidence_ids，故兩種模式統一行為（CEO 拍板項 #7）。
        #
        # P2 全局 evidence 模型（W2）：此 list **不再是 writer 可見集的「白名單邊界」**
        # （evidence_lookup 已改全 pool，見 W3）。改作「writer 視圖的優先 tier 提示」餵
        # render_grounding_evidence_view / writer prompt（W5/W6）決定排序與 budget 內誰先進。
        # strip/cap 演算法不變（仍產正確 priority tier），只是消費語意改。
        if book_outline is not None and 0 <= current_chapter_index < len(book_outline.chapters):
            planned = book_outline.chapters[current_chapter_index].planned_evidence_ids
            if planned:
                # 過濾掉 outline planner 可能誤產出的白名單外 ID（hallucination guard）
                analyst_citations = [eid for eid in planned if eid in valid_ids]

        # ──────────────────────────────────────────────────────────────
        # Evidence aggregate cap (chapter-0 context bomb fix, 綁 keystone f172d9b)
        # analyst_citations 已定型（union / planned / topic / B2 fallback 各路徑匯流）。
        # 此處一次 cap → writer 的 evidence_lookup（下方從 analyst_citations 建）受限。
        # 注意（模塊1 A.2 / 43bd5c61, 2026-06-09 起）：critic 的 chapter_evidence_text
        # 已改為「全 evidence pool 視圖」（非 analyst_citations subset），改受
        # schemas_live.GROUNDING_VIEW_CHAR_BUDGET=12000（R2）管轄，不再受此 cap 管轄。
        # 兩條鏈各有 cap，F1/F3 仍不爆窗，但機制已不同源——勿再敘述為「同步受限」。
        # 選擇策略（防 topic starvation）：planned_evidence_ids 全保留 + remaining
        # stratified 均勻抽樣補位 + char budget 主 cap（見 _cap_evidence_citations docstring）。
        # ──────────────────────────────────────────────────────────────
        if evidence_pool:
            # 取該章 planned（book_outline 覆寫區塊同源，避免依賴 local 變數命名）
            _planned_for_cap = None
            if (book_outline is not None
                    and 0 <= current_chapter_index < len(book_outline.chapters)):
                _planned_for_cap = (
                    book_outline.chapters[current_chapter_index].planned_evidence_ids
                )
            _pre_cap = len(analyst_citations)
            analyst_citations = _cap_evidence_citations(
                analyst_citations, evidence_pool,
                planned_evidence_ids=_planned_for_cap,
            )
            if len(analyst_citations) < _pre_cap:
                logger.info(
                    f"[LIVE RESEARCH] chapter {current_chapter_index} evidence "
                    f"capped {_pre_cap} → {len(analyst_citations)} 筆 "
                    f"(planned 全保留 + stratified 補位; "
                    f"MAX_CHARS={MAX_EVIDENCE_CHARS}, MAX_ITEMS={MAX_EVIDENCE_ITEMS})"
                )

        # ────────────────────────────────────────────────────────────────────
        # 模塊5 Task 5 / P2 W9（SF1）: per-chapter evidence 充分度（calibration 通道 B）。
        # 改用「全 pool 有料量」判（_compute_chapter_sufficiency），非 analyst_citations 量
        # —— 全局模型下 writer 讀全 pool，analyst_citations 空 ≠ 沒 evidence。
        # 與 specificity_check 互斥分工；intro/conclusion 章不施加。
        # ────────────────────────────────────────────────────────────────────
        chapter_sufficiency = _compute_chapter_sufficiency(analyst_citations, evidence_pool)
        # intro/conclusion 章不做 calibration（這些章本就偏綜述，不該被叫保守也不被逼具體）
        if book_outline is not None and _is_intro_or_conclusion(book_outline, current_chapter_index):
            chapter_sufficiency = "ok"
        logger.info(
            f"[LIVE RESEARCH] chapter {current_chapter_index} sufficiency="
            f"{chapter_sufficiency} (pool={len(evidence_pool or {})}, "
            f"citations={len(analyst_citations)})"
        )
        # FIX-5: 四個 compose_section callsite 共用的背景透傳參數（time_constraint /
        # evidence_sufficiency / knowledge_graph），單一來源避免漏帶即靜默退化。
        _writer_kw = self._common_writer_kwargs(state, chapter_sufficiency)

        # ────────────────────────────────────────────────────────────────────
        # Track A Task 3 / addendum C-1 deterministic gate (sprint 2026-05-28):
        # body chapter 偵測 analyst_citations 為空 → 不呼叫 writer LLM, 直接
        # 產 LiveWriterSectionOutput(status="blocked_no_evidence")。
        #
        # Gate 條件 (組合):
        # 1. is_chapter_override = True (chapter dict 模式, 對應 Track A 主修場景);
        #    ContextMapTopic 模式有 topic.evidence_ids fallback, 不走此 gate
        # 2. _is_intro_or_conclusion(book_outline, idx) = False
        #    (body chapter; codex C-1 v2: outline=None 預設 False)
        # 3. analyst_citations 為空 (沒可用 evidence)
        #
        # 紅隊 #2 (LLM 把 body 標 intro 想繞 gate): _is_intro_or_conclusion runtime
        # double-check role + idx 雙重一致才回 True。
        # ────────────────────────────────────────────────────────────────────
        # P2 W10：入口 gate 改判。
        # - book_outline 有（真正的全局 evidence 模型 path）：只擋「pool 完全空」（render 前能判的）。
        #   全局模型下 analyst_citations 空 ≠ 沒料（writer 讀全 pool），故不能再用它當 gate 條件；
        #   「有 source 但 render 後實質空」交 post-render gate（下方）。
        # - book_outline=None 純 legacy path（union-to-first，無 evidence_pool dict 但有
        #   all_evidence_ids）：保留既有 `not analyst_citations` 語意（idx>0 空 union → 擋；
        #   idx=0 union 非空 → 不擋），對齊 test_..._uses_union_evidence_ids。
        # R2-3：保留 _is_intro_or_conclusion guard — intro/conclusion 本就可無 evidence，不誤擋。
        _entry_gate_no_evidence = (
            (not evidence_pool) if book_outline is not None else (not analyst_citations)
        )
        if (
            is_chapter_override
            and not _is_intro_or_conclusion(book_outline, current_chapter_index)
            and _entry_gate_no_evidence
        ):
            chapter_title_gate = (
                book_outline.chapters[current_chapter_index].title
                if (book_outline is not None
                    and 0 <= current_chapter_index < len(book_outline.chapters))
                else (topic.get("name", "") if isinstance(topic, dict) else "")
            )
            blocked = LiveWriterSectionOutput(
                section_title=chapter_title_gate,
                section_content=lr_copy.BLOCKED_NO_EVIDENCE_ENTRY,
                sources_used=[],
                confidence_level="Low",
                status="blocked_no_evidence",
            )
            logger.warning(
                f"[LIVE RESEARCH] C-1 入口 gate: evidence_pool 全空 — chapter "
                f"{current_chapter_index} ({chapter_title_gate!r}) BlockedSection "
                "（真零 evidence，明確擋，不呼叫 writer LLM）"
            )
            return blocked, False

        if self.dry_run:
            dummy = LiveWriterSectionOutput(
                section_title=section_title,
                section_content=(
                    f"## {section_title}\n\n"
                    f"{section_outline}\n\n"
                    "台灣近年大力推動再生能源，光電裝置容量大幅成長 [1]。"
                    f"然而，{section_title}議題涉及多方利害關係人，需要審慎評估 [1]。\n\n"
                    "### 主要挑戰\n\n"
                    f"- 政策面：相關法規尚待完善\n"
                    f"- 技術面：基礎設施整合困難\n"
                    f"- 社會面：社區接受度有待提升\n\n"
                    "**參考資料**\n[1] 台灣光電發展現況 https://example.com/1\n"
                ),
                sources_used=[1],
                confidence_level="Medium",
                narration=f"dry-run fixture for section: {section_title}",
                chapter_summary=f"dry-run 摘要：{section_title}",
            )
            # chapter override 模式下 valid_ids 仍為 cm.topics 聯集（沿用 D-2 紀律）；
            # 但 dummy 仍用 [1]，Phase 3 才設計真實白名單 — Phase 2 dry_run 兼容用既有 valid_ids
            return apply_hallucination_guard(dummy, valid_ids)

        from reasoning.agents.writer import WriterAgent
        from reasoning.schemas_live import context_map_extract_for_section

        writer_timeout = CONFIG.reasoning_params.get("writer_timeout", 45)
        _section_start = time.perf_counter()
        logger.info(
            f"[LIVE RESEARCH] Writer section start: topic_id={section_topic_id} "
            f"name={section_title[:50]!r} chapter_override={is_chapter_override} "
            f"timeout_budget={writer_timeout}s"
        )

        writer = WriterAgent(
            self.handler,
            timeout=writer_timeout,
        )

        # Track A Task 3 (sprint 2026-05-28): 替換 hardcoded "" — 從
        # state.evidence_usage render per-chapter findings (Gemini Critical REJECT
        # 入庫 forensic + render 層 filter; WARN 標 low confidence marker)。
        #
        # codex C-1 v2: state 或 book_outline 缺 → log ERROR + 空 findings (legacy
        # path; 入口 gate 已用 analyst_citations 攔過 body chapter empty, 此處
        # render 後 gate 只在「state+book_outline 都有, 但 render 仍空」時觸發)。
        rendered_via_state = False
        writer_evidence_view = None  # P2 W5：全 pool grounding 視圖（chapter_override 分支內組）
        if is_chapter_override:
            if state is not None and book_outline is not None:
                from reasoning.schemas_live import (
                    render_grounded_narrative, GROUNDING_VIEW_CHAR_BUDGET,
                )
                # P2 W4：narrative 走全 pool（對齊 W3/W5），不再逐章 planned。
                # 本章 planned 當排序提示（priority_eids），不當過濾邊界。
                _planned_here = (
                    book_outline.chapters[current_chapter_index].planned_evidence_ids
                    if 0 <= current_chapter_index < len(book_outline.chapters)
                    else []
                )
                relevant_findings = render_grounded_narrative(
                    chapter_eids=list((evidence_pool or {}).keys()),  # 全 pool
                    evidence_usage=state.evidence_usage,
                    evidence_pool=evidence_pool or {},
                    priority_eids=_planned_here,                       # 軟排序提示
                    char_budget=GROUNDING_VIEW_CHAR_BUDGET,
                )
                rendered_via_state = True
            else:
                # codex C-1 v2: state / book_outline 缺 → 紀律破口 → log ERROR + 空 findings;
                # 入口 gate 已用 analyst_citations 攔過 body chapter empty
                # (不會 silent 編造)。但仍 ERROR 以追蹤 caller 修補 (legacy callers)。
                logger.error(
                    "[LIVE RESEARCH] _write_section called without state/book_outline — "
                    "this is a discipline violation (codex C-1 v2): revise/continue caller "
                    "MUST propagate state+book_outline or reject at endpoint with 4xx "
                    "legacy_revise_no_state. Falling through to empty findings (entry "
                    "C-1 gate already blocks body chapters with empty analyst_citations)."
                )
                relevant_findings = ""

            # P2 W5（I2）：組 writer 全 pool grounding 視圖（對齊 Critic 範本）。
            # 組裝點在 narrative render（W4）之後、post-render gate（W10）之前 →
            # gate 讀得到同一變數、無 NameError。view 只需 evidence_pool（此處作用域已有），
            # 不需 evidence_lookup（下方才產生）。analyst_citations 當優先 tier、
            # suggested_chapters 含本章升 tier（current_chapter_index）。
            from reasoning.schemas_live import (
                render_grounding_evidence_view as _render_writer_view,
                GROUNDING_VIEW_CHAR_BUDGET as _WRITER_VIEW_BUDGET,
            )
            # §3：12000 起點，可獨立 tune（現先別名同值）
            WRITER_GROUNDING_VIEW_CHAR_BUDGET = _WRITER_VIEW_BUDGET
            writer_evidence_view = _render_writer_view(
                chapter_eids=list((evidence_pool or {}).keys()),          # 全 pool
                evidence_usage=(
                    getattr(state, "evidence_usage", {}) if state is not None else {}
                ),
                evidence_pool=evidence_pool or {},
                prior_grounded_entities=prior_used_entities or [],         # _write_section 參數
                analyst_citations=analyst_citations,                       # 優先 tier
                char_budget=WRITER_GROUNDING_VIEW_CHAR_BUDGET,
                current_chapter_index=current_chapter_index,               # suggested_chapters 軟排序
            )

            # 第二層 gate (R1 reviewer I-1 fix, sprint 2026-05-28):
            # 「is_chapter_override + book_outline 有 + body chapter + relevant_findings 空」
            # → BlockedSection。獨立於 rendered_via_state — 讓 state=None + book_outline 有
            # 的破口場景也 fire。
            #
            # 原 fix 前: 條件含 `rendered_via_state` → state=None + analyst_citations 非空
            # + book_outline 有 (body chapter) → 入口 gate 不 fire (citations truthy);
            # post-render path 走 else (state/book_outline 缺其一即 else) → relevant_findings='';
            # rendered_via_state=False → 跳過 gate → writer 被 invoke with empty findings +
            # non-empty cite whitelist → C-1 設計目標破口。
            #
            # Fix 後: 條件改為「book_outline 有 + body chapter + findings 空」即攔下,
            # 不問 state 是否有 — 涵蓋 reviewer 描述的破口場景 (state=None 但有 book_outline)。
            #
            # 紀律邊界: book_outline=None 的純 legacy path (union-to-first) **不**走此 gate,
            # 仍由入口 gate (line 2851) 負責 — 兩者語意對齊既有 test
            # test_write_section_chapter_override_first_index_uses_union_evidence_ids:
            # state=None + book_outline=None + idx=0 → analyst_citations=union → 入口 gate 不 fire
            # → writer 被叫 (legacy 行為保留)。
            # P2 W10（R1）：relevant_findings 空 ≠ pool 空。render_grounded_narrative 只渲
            # 有 grounded claim 的 entry；raw pool 可能有 snippet 但還沒 claim → narrative 空但
            # writer 仍可用 grounding view（snippet）寫。故只在「全 pool grounding view + narrative
            # 都實質空」才擋（全 REJECT / 真零料）。移除綜合章特例（全局視圖綜合章本就讀得到
            # 前文 evidence）；R2-3 保留 intro/conclusion guard。明確 log（不可 silent fail）。
            writer_view_empty = not (writer_evidence_view or "").strip()   # W5 全 pool snippet 視圖
            narrative_empty = not (relevant_findings or "").strip()
            if (
                book_outline is not None
                and not _is_intro_or_conclusion(book_outline, current_chapter_index)
                and writer_view_empty
                and narrative_empty
            ):
                chapter_title_post = (
                    book_outline.chapters[current_chapter_index].title
                    if book_outline is not None else section_title
                )
                blocked = LiveWriterSectionOutput(
                    section_title=chapter_title_post,
                    section_content=lr_copy.BLOCKED_NO_EVIDENCE_POST_RENDER,
                    sources_used=[],
                    confidence_level="Low",
                    status="blocked_no_evidence",
                )
                logger.warning(
                    f"[LIVE RESEARCH] C-1 post-render gate: 全 pool grounding view + "
                    f"narrative 都實質空 — chapter {current_chapter_index} "
                    f"({chapter_title_post!r}) BlockedSection（真沒料，明確擋）"
                )
                return blocked, False
            # narrative 空但 grounding view 非空 → 不擋（writer 用 snippet 寫，raw pool 有料只是還沒 claim）
        else:
            relevant_findings = context_map_extract_for_section(context_map, [section_topic_id])

        # P2 全局 evidence 模型（W3）：writer 讀全 pool（與 Critic 對齊）。
        # analyst_citations / suggested_chapters 僅當排序提示（W5/W6），非白名單。
        # phantom 不可能存在（直接從 pool 取）；evidence_pool None/空 → None，
        # 交 W10 gate 明確擋並 log（不在此 silent 放行）。
        evidence_lookup = dict(evidence_pool) if evidence_pool else None

        # format_spec：從 format_specs dict 組合為字串
        format_spec = None
        if format_specs:
            user_spec = format_specs.get("user_specified", format_specs.get("default", "markdown_apa"))
            format_spec = f"格式要求：{user_spec}"

        # spec §4.6.3：special_elements per-chapter filter（R2 澄清機制，2026-07）。
        # target_chapter 在 Stage 4 已（code 短路 / LLM 判 clear→user 確認 / user 澄清）定位成章名原文
        # → 此處 exact 命中 section_title 即注入。空 target → 全章注入。
        # 對不到 → 不注入（Stage 5 後衛 report-level 診斷負責 no silent fail）。
        special_elements_for_chapter: List[Dict[str, str]] = []
        all_special_elements = (
            format_specs.get("special_elements") if format_specs else None
        )
        if all_special_elements:
            for elem in all_special_elements:
                if not isinstance(elem, dict):
                    continue
                # SF2：讀既有 list 先 sanitize（舊 session 髒 transient 不傳給 writer）——
                # 與 Step 6 B3「讀既有 list 也 sanitize」宣稱對齊。
                elem = _serialize_special_element_for_state(elem)
                target = (elem.get("target_chapter") or "").strip()
                if not target:
                    special_elements_for_chapter.append(elem)  # 全章注入
                elif target == section_title:
                    special_elements_for_chapter.append(elem)  # exact 命中章名

        # context_map_summary：整份 ContextMap 摘要
        summary = context_map_to_summary(context_map)

        # citation_format resolution（Plan: lr-user-voice-container-and-4-fixes Fix B）
        # Precedence: user_voice.citation_style > style_features.citation_format > "numeric"
        if user_voice is not None and getattr(user_voice, "citation_style", None) is not None:
            citation_format = user_voice.citation_style
        elif style_features is not None:
            citation_format = style_features.citation_format
        else:
            citation_format = "numeric"

        try:
            section_output = await writer.compose_section(
                section_title=section_title,
                section_outline=section_outline,
                relevant_findings=relevant_findings,
                analyst_citations=analyst_citations,
                style_features=style_features,
                format_spec=format_spec,
                context_map_summary=summary,
                citation_format=citation_format,
                evidence_lookup=evidence_lookup,
                writer_evidence_view=writer_evidence_view,  # P2 W7：全 pool 視圖
                is_chapter_override=is_chapter_override,
                book_outline=book_outline,
                current_chapter_index=current_chapter_index,
                previous_chapter_summary=previous_chapter_summary,
                special_elements_for_chapter=special_elements_for_chapter,
                revise_instruction=revise_instruction,
                prior_section_content=prior_section_content,
                # Track A (sprint 2026-05-28) Task 7: prior_used_entities
                prior_used_entities=prior_used_entities,
                # B (Cayenne cross-section): synthesis 章前章摘要注入
                all_prior_chapter_summaries=all_prior_chapter_summaries,
                # FIX-5: 背景透傳參數（time_constraint / evidence_sufficiency /
                # knowledge_graph）集中於 _common_writer_kwargs，四 callsite 共用單一來源。
                # 顯式列出 knowledge_graph 等鍵以保留 AST 覆蓋（test_prompt_writer_kg）。
                time_constraint=_writer_kw["time_constraint"],
                evidence_sufficiency=_writer_kw["evidence_sufficiency"],
                knowledge_graph=_writer_kw["knowledge_graph"],
            )
        except Exception as e:
            _section_elapsed = time.perf_counter() - _section_start
            logger.error(
                f"[LIVE RESEARCH] Writer section failed: topic_id={section_topic_id} "
                f"elapsed={_section_elapsed:.2f}s error={type(e).__name__}: {e}"
            )
            raise  # 不 silent swallow — bubble up to _run_stage_5 / continueResearch

        _section_elapsed = time.perf_counter() - _section_start
        logger.info(
            f"[LIVE RESEARCH] Writer section done: topic_id={section_topic_id} "
            f"elapsed={_section_elapsed:.2f}s content_len={len(section_output.section_content)} "
            f"sources_used={len(section_output.sources_used)}"
        )

        # Hallucination Guard per-section（port from DR orchestrator.py:1095-1131）
        section_output, was_corrected = apply_hallucination_guard(section_output, valid_ids)

        # Track A (sprint 2026-05-28) Task 5: per-section content-aware entity guard
        # + auto-rewrite 1 次 + C-2 guard_failed status (rewrite 仍 fail → blocked 文字
        # 替換 + status="guard_failed"; **不**留 LLM 原 prose, **不**留 [未經證據驗證]
        # methodology_note fallback)。
        # F-I-1 safe-init: 在 try 之前初始化，確保 try block 若在賦值前 raise，
        # except 之後的 _run_publish_gate 呼叫讀到 "" 而非 unbound NameError。
        chapter_evidence_text = ""
        try:
            from reasoning.live_research.hallucination_guard import (
                entity_grounding_check, GroundingCheckUnavailable,
            )
            # CEO 方向：給 grounding 判讀「良好、全 pool、不截斷、跨章可見」的 evidence 資料源。
            # 用既有持久化結構（evidence_usage / prior_used_entities）建，不重造。
            # CEO 拍板（決策②）：餵全 evidence pool（不只本章 analyst_citations subset）。
            from reasoning.schemas_live import render_grounding_evidence_view
            _all_pool_eids = list((evidence_pool or {}).keys())  # 全 pool，非 subset
            chapter_evidence_text = render_grounding_evidence_view(
                chapter_eids=_all_pool_eids,
                evidence_usage=(
                    getattr(state, "evidence_usage", {}) if state is not None else {}
                ),
                evidence_pool=evidence_pool or {},
                prior_grounded_entities=prior_used_entities or [],
                analyst_citations=analyst_citations,   # R2：本章引用 = budget 最高優先 tier
            )

            try:
                ungrounded = await entity_grounding_check(
                    section=section_output,
                    chapter_evidence_text=chapter_evidence_text,
                    handler=self.handler,
                    on_extraction_failed=self._narrate_grounding_extraction_failed,
                )
                await self._emit_grounding_extraction_failed_if_pending()
            except GroundingCheckUnavailable as _gce:
                # R1 fail-closed：grounding 系統驗證失敗 → 不放行、不炸 pipeline → DR 式退化。
                section_output = await self._apply_degraded_grounding_unavailable(
                    section_output=section_output,
                    analyst_citations=analyst_citations,
                    current_chapter_index=current_chapter_index,
                    reason=str(_gce),
                    evidence_pool=evidence_pool,   # P2 W7 I1：全 pool 合法集
                )
                ungrounded = []      # 已退化處理；不再進 rewrite / partial block
                was_corrected = True
            if ungrounded:
                # N-2 + Gemini Mn-1 拍板: log 加 traceability prefix
                # (session_id + chapter idx + original content len/snippet)
                _session_id = (
                    getattr(self, "session_id", None)
                    or getattr(state, "session_id", None)
                    or "<unknown>"
                )
                _orig_content = section_output.section_content or ""
                logger.warning(
                    f"[LIVE RESEARCH][Session:{_session_id}]"
                    f"[Chapter:{current_chapter_index}] "
                    f"Ungrounded entities detected (1st check) in section "
                    f"{section_output.section_title!r}: {ungrounded}; "
                    f"original section_content len={len(_orig_content)} chars, "
                    f"snippet={_orig_content[:200]!r}... "
                    f"Triggering auto-rewrite."
                )
                # 自動重寫一次, 把 ungrounded 清單當 revision instruction 傳回 writer
                section_output = await writer.compose_section(
                    section_title=section_title,
                    section_outline=section_outline,
                    relevant_findings=relevant_findings,
                    analyst_citations=analyst_citations,
                    style_features=style_features,
                    format_spec=format_spec,
                    context_map_summary=summary,
                    citation_format=citation_format,
                    evidence_lookup=evidence_lookup,
                    writer_evidence_view=writer_evidence_view,  # P2 W7：全 pool 視圖
                    is_chapter_override=is_chapter_override,
                    book_outline=book_outline,
                    current_chapter_index=current_chapter_index,
                    previous_chapter_summary=previous_chapter_summary,
                    special_elements_for_chapter=special_elements_for_chapter,
                    revise_instruction=revise_instruction,
                    prior_section_content=prior_section_content,
                    ungrounded_entities_revision=ungrounded,
                    # Track A Task 7: 第二次重寫也須帶 prior_used_entities (避免 LLM
                    # 在 rewrite 時引入新前文未提及實體)
                    prior_used_entities=prior_used_entities,
                    # B (Cayenne cross-section): rewrite path 也帶前章摘要
                    all_prior_chapter_summaries=all_prior_chapter_summaries,
                    # FIX-5: entity-rewrite path 共用背景透傳參數（單一來源）。
                    time_constraint=_writer_kw["time_constraint"],
                    evidence_sufficiency=_writer_kw["evidence_sufficiency"],
                    knowledge_graph=_writer_kw["knowledge_graph"],
                )
                # 第二次寫完再跑 hallucination guard subset check (citation 白名單)
                section_output, was_corrected_2 = apply_hallucination_guard(
                    section_output, valid_ids,
                )
                was_corrected = was_corrected or was_corrected_2

                # addendum C-2: 第二次重寫後再 entity check 一次, 仍有 ungrounded
                # → Fix2 partial block（CEO 決策④：主路徑刪句 / 退化 DR-style，
                #   丟掉整章替換）。R1：語意層失敗 → DR 式退化。
                try:
                    remaining = await entity_grounding_check(
                        section=section_output,
                        chapter_evidence_text=chapter_evidence_text,
                        handler=self.handler,
                        on_extraction_failed=self._narrate_grounding_extraction_failed,
                    )
                    await self._emit_grounding_extraction_failed_if_pending()
                except GroundingCheckUnavailable as _gce2:
                    # R1 fail-closed（第二呼叫點）：不放行、不炸 pipeline → DR 式退化。
                    section_output = await self._apply_degraded_grounding_unavailable(
                        section_output=section_output,
                        analyst_citations=analyst_citations,
                        current_chapter_index=current_chapter_index,
                        reason=str(_gce2),
                        evidence_pool=evidence_pool,   # P2 W7 I1：全 pool 合法集
                    )
                    remaining = []   # 已退化處理；不再進 partial block
                    was_corrected = True
                if remaining:
                    _rewritten_content = section_output.section_content or ""
                    logger.warning(
                        f"[LIVE RESEARCH][Chapter:{current_chapter_index}] "
                        f"Auto-rewrite failed (2nd check still has ungrounded): "
                        f"section={section_output.section_title!r}, "
                        f"ungrounded_entities_after_rewrite={remaining}; "
                        f"rewritten content len={len(_rewritten_content)} chars, "
                        f"snippet={_rewritten_content[:200]!r}... "
                        f"→ Fix2 partial block（主路徑刪句 / 退化 DR-style）。"
                    )
                    # Fix2 (CEO 決策④): partial block — 主路徑 (b) 只刪純未驗證句 /
                    # 退化 (a) DR-style 保留正文；丟掉 (c) 整章替換。
                    # R3：傳本章已驗證 entity（prior_used_entities ∪ 本章 grounded 候選）供
                    # split helper 做句子分類，避免硬刪含已驗證 entity 的混合句。
                    _verified_for_section = [
                        e for e in (prior_used_entities or [])
                    ]  # known limitation：漏「本章剛驗證」entity（V1 接受，見 plan Task 2.2）
                    section_output, _ = self._apply_partial_or_degraded_block(
                        section_output=section_output,
                        ungrounded=remaining,
                        analyst_citations=analyst_citations,
                        current_chapter_index=current_chapter_index,
                        label="entity-guard rewrite",
                        grounded_entities=_verified_for_section,
                        evidence_pool=evidence_pool,   # P2 W7 I1：全 pool 合法集
                    )
                    was_corrected = True

            # ── A (Cayenne specificity): entity guard（fabrication 方向）通過後，
            #    在同一 try block 內跑對稱的 specificity_check（under-specification 方向）。
            #    body chapter + evidence 有具體資訊 + prose 全抽象 → auto-rewrite 一次。
            #    重用既有 _extract_section_entities（抽到的 entity 也回給 caller 存
            #    written_sections[i]["entities"]，省 loop 重複 call）。
            _composed_entities: List[str] = []
            # 模塊5 Task 5 協調：specificity guard 只對充足章（ok）生效。thin/critical 章
            # 已在 prompt 端放保守 calibration 指示，不再用 specificity guard 逼具體，
            # 否則 prompt（叫保守）與 guard（逼具體）對打，calibration 會被事後 rewrite 推翻。
            if (
                is_chapter_override
                and getattr(section_output, "status", "drafted") == "drafted"
                and not _is_intro_or_conclusion(book_outline, current_chapter_index)
                and chapter_sufficiency == "ok"
            ):
                from reasoning.live_research.hallucination_guard import (
                    specificity_check,
                )
                _composed_entities = await self._extract_section_entities(
                    section_output.section_content, self.handler,
                )
                # evidence_has_concrete：輕量啟發式 — evidence text 是否含數字 / 書名號 /
                # 引號（具體資訊的弱信號）。保守：判不出來時當作有（讓 prompt 第 0 點主導）。
                import re as _re
                _evidence_has_concrete = bool(
                    _re.search(r"\d", chapter_evidence_text)
                    or "《" in chapter_evidence_text
                    or "「" in chapter_evidence_text
                )
                if specificity_check(
                    section=section_output,
                    chapter_evidence_text=chapter_evidence_text,
                    section_entities=_composed_entities,
                    evidence_has_concrete=_evidence_has_concrete,
                ):
                    logger.warning(
                        f"[LIVE RESEARCH] specificity_check flagged chapter "
                        f"{current_chapter_index} ({section_output.section_title!r}) — "
                        f"evidence 有具體資訊但 prose 全抽象，觸發 specificity auto-rewrite。"
                    )
                    section_output = await writer.compose_section(
                        section_title=section_title,
                        section_outline=section_outline,
                        relevant_findings=relevant_findings,
                        analyst_citations=analyst_citations,
                        style_features=style_features,
                        format_spec=format_spec,
                        context_map_summary=summary,
                        citation_format=citation_format,
                        evidence_lookup=evidence_lookup,
                        writer_evidence_view=writer_evidence_view,  # P2 W7：全 pool 視圖
                        is_chapter_override=is_chapter_override,
                        book_outline=book_outline,
                        current_chapter_index=current_chapter_index,
                        previous_chapter_summary=previous_chapter_summary,
                        special_elements_for_chapter=special_elements_for_chapter,
                        prior_used_entities=prior_used_entities,
                        all_prior_chapter_summaries=all_prior_chapter_summaries,
                        revise_instruction=(
                            "上一版內容過於抽象 — 你**沒有**把 evidence 裡的具體案例 / 地名 / "
                            "數字 / 法規寫出來。請重寫，主動把『相關發現』與來源中**已存在**的具體 "
                            "entity 落進 prose（不可編造 evidence 沒有的）。"
                        ),
                        prior_section_content=section_output.section_content,
                        # FIX-5: specificity-rewrite path 共用背景透傳參數（單一來源）。
                        time_constraint=_writer_kw["time_constraint"],
                        evidence_sufficiency=_writer_kw["evidence_sufficiency"],
                        knowledge_graph=_writer_kw["knowledge_graph"],
                    )
                    was_corrected = True
                    # ── Important 1: specificity rewrite 後**重跑三層守門**，確保
                    #    「寫得更具體」沒有反向開出編造 entity / phantom citation 的洞。
                    # (1) citation 白名單 guard
                    section_output, _was_corr_spec = apply_hallucination_guard(
                        section_output, valid_ids,
                    )
                    was_corrected = was_corrected or _was_corr_spec
                    # (2) entity grounding guard（fabrication 方向）— rewrite 出來的具體
                    #     entity 必須仍 grounded；若引入 ungrounded entity → 走既有
                    #     blocked 文字替換 + status=guard_failed 紀律（與第一次 rewrite 對稱）。
                    # R1 fail-closed（specificity 呼叫點）：語意層失敗 → DR 式退化，不放行。
                    try:
                        _spec_ungrounded = await entity_grounding_check(
                            section=section_output,
                            chapter_evidence_text=chapter_evidence_text,
                            handler=self.handler,
                            on_extraction_failed=self._narrate_grounding_extraction_failed,
                        )
                        await self._emit_grounding_extraction_failed_if_pending()
                    except GroundingCheckUnavailable as _gce3:
                        section_output = await self._apply_degraded_grounding_unavailable(
                            section_output=section_output,
                            analyst_citations=analyst_citations,
                            current_chapter_index=current_chapter_index,
                            reason=str(_gce3),
                            evidence_pool=evidence_pool,   # P2 W7 I1：全 pool 合法集
                        )
                        _spec_ungrounded = []   # 已退化處理；不再進 partial block
                        was_corrected = True
                        _composed_entities = []
                    if _spec_ungrounded:
                        logger.warning(
                            f"[LIVE RESEARCH][Chapter:{current_chapter_index}] "
                            f"specificity rewrite 引入 ungrounded entity "
                            f"{_spec_ungrounded} → Fix2 partial block（主路徑刪句/退化 DR-style）。"
                        )
                        # Fix2 (CEO 決策④): partial block 取代整章替換（丟掉 (c)）。
                        section_output, _ = self._apply_partial_or_degraded_block(
                            section_output=section_output,
                            ungrounded=_spec_ungrounded,
                            analyst_citations=analyst_citations,
                            current_chapter_index=current_chapter_index,
                            label="specificity rewrite",
                            grounded_entities=(prior_used_entities or []),  # R3 句子分類
                            evidence_pool=evidence_pool,   # P2 W7 I1：全 pool 合法集
                        )
                        was_corrected = True
                        _composed_entities = []
                    else:
                        # rewrite 後重抽 entity 供 caller 用（content 已具體）
                        _composed_entities = await self._extract_section_entities(
                            section_output.section_content, self.handler,
                        )

            # ── B(a) (Cayenne cross-section deterministic 兜底): synthesis 章寫完後，
            #    抽 prose entity，凡「不在前章 entity 聯集」的新具體 entity → 沿 T5
            #    ungrounded_entities_revision path auto-rewrite 一次（reuse 既有 block，
            #    不新造）。與 A specificity gate 對稱；(b) prompt 注入的 deterministic 收口。
            _is_synth = False
            if (
                book_outline is not None
                and 0 <= current_chapter_index < len(book_outline.chapters)
            ):
                _ch = book_outline.chapters[current_chapter_index]
                _role = getattr(_ch, "role", "")
                _brief = getattr(_ch, "brief", "") or ""
                _is_synth = (_role == "conclusion") or any(
                    k in _brief for k in ("綜合", "結論", "討論")
                )
            if (
                _is_synth
                and getattr(section_output, "status", "drafted") == "drafted"
                and prior_used_entities
            ):
                _synth_entities = await self._extract_section_entities(
                    section_output.section_content, self.handler,
                )
                _prior_set = set(prior_used_entities)
                _new_entities = [e for e in _synth_entities if e not in _prior_set]
                if _new_entities:
                    logger.warning(
                        f"[LIVE RESEARCH][Chapter:{current_chapter_index}] synthesis 章"
                        f"冒出前文未提及的新具體 entity {_new_entities} — "
                        f"觸發 (a) 兜底 auto-rewrite。"
                    )
                    section_output = await writer.compose_section(
                        section_title=section_title,
                        section_outline=section_outline,
                        relevant_findings=relevant_findings,
                        analyst_citations=analyst_citations,
                        style_features=style_features,
                        format_spec=format_spec,
                        context_map_summary=summary,
                        citation_format=citation_format,
                        evidence_lookup=evidence_lookup,
                        writer_evidence_view=writer_evidence_view,  # P2 W7：全 pool 視圖
                        is_chapter_override=is_chapter_override,
                        book_outline=book_outline,
                        current_chapter_index=current_chapter_index,
                        previous_chapter_summary=previous_chapter_summary,
                        special_elements_for_chapter=special_elements_for_chapter,
                        prior_used_entities=prior_used_entities,
                        all_prior_chapter_summaries=all_prior_chapter_summaries,
                        ungrounded_entities_revision=_new_entities,  # reuse T5 block
                        # FIX-5: synthesis-rewrite path 共用背景透傳參數（單一來源）。
                        time_constraint=_writer_kw["time_constraint"],
                        evidence_sufficiency=_writer_kw["evidence_sufficiency"],
                        knowledge_graph=_writer_kw["knowledge_graph"],
                    )
                    was_corrected = True
                    # rewrite 後重跑 citation guard（與 entity-guard rewrite 對稱）
                    section_output, _was_corr_a = apply_hallucination_guard(
                        section_output, valid_ids,
                    )
                    was_corrected = was_corrected or _was_corr_a
                    # rewrite 後重抽，覆寫 _composed_entities 供 caller 用
                    _composed_entities = await self._extract_section_entities(
                        section_output.section_content, self.handler,
                    )

            # 把抽好的 entity 掛到 section_output（caller 取用，省 loop 重複抽）
            try:
                setattr(section_output, "_composed_entities", _composed_entities)
            except Exception:
                pass
        except Exception as e:
            # secondary defense: 不阻塞 pipeline; 但不可 silent fail → log warning
            logger.warning(
                f"[LIVE RESEARCH] Entity grounding guard failed (non-fatal): "
                f"{type(e).__name__}: {e}"
            )
            # D-2026-06-11 決策1（o5c Task2 解凍）：此 except 接到的是非-GCU 例外
            # （render/抽取/rewrite/specificity 等環節；GCU 已由上方三個內層 except
            # 局部接走，到不了這裡）。被吞時補一次即時旁白。
            # 事實對齊（2026-06-10 根因修正）：_run_publish_gate 在本 try 之外照常
            # 執行 → 文案不可稱「未經把關」。per-run dedup 防多章連續失敗轟炸；
            # log 每章照記，不受 flag 影響。
            if not self._guard_error_narrated:
                self._guard_error_narrated = True
                await self._emit_narration(lr_copy.SECTION_GUARD_ERROR_NARRATION)

        # === Track F (sprint 2026-05-28): F1 + F3 publish gate (I-4 抽 helper) ===
        # 三層防禦第三層 — T5 entity guard 跑完後跑 F1 claim-level critic
        # publish gate + F3 CoV-lite verification。content / status mutation 統一
        # 在 helper 內依 verdict 一次完成（I-5 紀律：F1 review + F3 CoV → 升級 →
        # 統一 mutate）。helper 對 status != "drafted" section short-circuit
        # pass-through（F-AMB-7）。
        section_output, _f_was_corrected = await self._run_publish_gate(
            section_output=section_output,
            current_chapter_index=current_chapter_index,
            chapter_evidence_text=chapter_evidence_text,
            state=state,
        )
        if _f_was_corrected:
            was_corrected = True

        # 字數 post-process：章節定稿後發「比預期長」透明化旁白（content 不動）。硬切會
        # 截斷正文使用者要不回，2026-07 改回軟約束。僅 user 明確要求字數才發旁白。
        _uv = getattr(state, "user_voice", None) if state is not None else None
        _user_specified_wc = bool(
            _uv is not None and getattr(_uv, "target_word_count", None)
        ) or self._chapter_has_user_word_target(state, current_chapter_index)
        await self._maybe_narrate_word_overshoot(
            section_output=section_output,
            target=self._resolve_chapter_target_words(
                book_outline, state, current_chapter_index,
                user_specified=_user_specified_wc,
            ),
            user_specified_word_count=_user_specified_wc,
        )

        # TypeAgent Target 3 (2026-05-19, CEO 拍板 OQ-5): typed citations render
        # 在 guard 之後跑（guard 已過濾 phantom citations / sources_used），
        # 用 evidence_lookup 真實 author/year metadata 統一 render `{cite:N}` placeholder。
        # Track A: 若 section status == "guard_failed" (content 已替換為 blocked 文字),
        # 跳過 citation render (沒 citation 可 render)。
        # Track F F1 擴張: critic_rejected 同 guard_failed — content 已是 blocked
        # 文字無 citation 可 render。
        if evidence_lookup and getattr(section_output, "status", "drafted") not in (
            "guard_failed", "critic_rejected",
        ):
            section_output = self._render_section_citations(
                section_output, evidence_lookup, citation_format,
            )
        return section_output, was_corrected

    # 字數超標閾值（lr-chapter-word-budget plan 設計細節 2）：
    # 實際 > target * 1.3（超標 30%）才發提示【初值，可調】。原 prompt 勸告 ±15%；
    # a 是「明顯超標才透明化」的通道，用 1.15 會在正常波動就洗訊息。
    _WORD_OVERSHOOT_RATIO = 1.3

    @staticmethod
    def _chapter_target_words(book_outline, current_chapter_index: int) -> int:
        """從 book_outline 取本章規劃字數（0 = 未指定）。"""
        if book_outline is None:
            return 0
        chapters = getattr(book_outline, "chapters", None) or []
        if not (0 <= current_chapter_index < len(chapters)):
            return 0
        return getattr(chapters[current_chapter_index], "target_word_count", 0) or 0

    @staticmethod
    def _chapter_has_user_word_target(state, current_chapter_index: int) -> bool:
        """user 是否對「本章」明確指定字數（format_specs.chapters[i].word_target）。"""
        if state is None:
            return False
        chapters = (getattr(state, "format_specs", {}) or {}).get("chapters") or []
        if not (0 <= current_chapter_index < len(chapters)):
            return False
        ch = chapters[current_chapter_index]
        wt = ch.get("word_target") if isinstance(ch, dict) else None
        return isinstance(wt, int) and wt > 0

    def _resolve_chapter_target_words(
        self, book_outline, state, current_chapter_index: int, *, user_specified: bool
    ) -> int:
        """SF1：優先 user 真實 surface form（format_specs.chapters[i].word_target），
        退回 outline planner 的 book_outline target。避免 outline LLM 漏抄 → no-op。
        user_specified=True 但仍解出 <=0 → logger.warning（不 silent）。
        """
        chapters = (getattr(state, "format_specs", {}) or {}).get("chapters") or [] if state else []
        if 0 <= current_chapter_index < len(chapters):
            ch = chapters[current_chapter_index]
            wt = ch.get("word_target") if isinstance(ch, dict) else None
            if isinstance(wt, int) and wt > 0:
                return wt
        outline_target = self._chapter_target_words(book_outline, current_chapter_index)
        if user_specified and outline_target <= 0:
            logger.warning(
                f"[LIVE RESEARCH] user 指定字數但本章 target 解不出（outline planner 漏抄?）: "
                f"idx={current_chapter_index}"
            )
        return outline_target

    async def _maybe_narrate_word_overshoot(
        self, *, section_output, target: int, user_specified_word_count: bool,
    ) -> bool:
        """章節字數明顯超標時發「本章比預期長、內容照常保留」旁白（不切 content）。

        僅 user 明確要求字數才發；content 一字不動（硬切會截斷正文）。契約見 spec §4.7.9。

        Returns: True 若發了旁白；False 若未發（未指定 / 未過閾值 / 非 drafted）。
        """
        if not user_specified_word_count or target <= 0:
            return False
        status = getattr(section_output, "status", "drafted")
        if status != "drafted":
            return False
        actual = _count_chapter_words(section_output.section_content)
        if actual <= target * self._WORD_OVERSHOOT_RATIO:
            return False

        chapter_title = getattr(section_output, "section_title", "") or "本章"
        logger.info(
            f"[LIVE RESEARCH] Chapter word overshoot (content kept): {chapter_title!r} "
            f"target={target} actual={actual} (ratio={actual / max(target, 1):.2f})"
        )
        await self._emit_narration(
            lr_copy.chapter_word_overshoot_narration(chapter_title, target, actual)
        )
        return True

    def _stage5_remaining_count(self, state) -> int:
        """回傳還有幾段沒寫完（0 = 全寫完）。

        LR #11 Part B：用於判斷是否允許匯出。mirror auto_continue gate 的 completeness 計算。
        """
        _cm = ContextMap.model_validate_json(state.context_map_json)
        _writer_sections, _ = self._resolve_chapter_source(_cm, state.format_specs)
        _total = len(_writer_sections)
        return max(0, _total - (state.last_completed_section_index + 1))

    async def _handle_stage_5_response(self, state, user_message, auto_continue):
        """處理 Stage 5 回覆 — VP-7 per-section checkpoint dialog loop。

        Dispatch 順序：
        1. auto_continue / empty msg：
           - 還有未寫章節 → 走 _run_stage_5 寫下一段（與 continue keyword 同 path，
             completeness gate，禁止只寫 1 段就匯出 — Cayenne RCA #3）
           - 全部寫完 → complete_stage（進 Stage 6 匯出）
        2. export keyword shortcut → 直接 complete_stage（CEO D-E：不問確認）
        3. continue keyword shortcut → 直接 _run_stage_5（不打 LLM，省成本+省延遲）
        4. LLM intent parse：revise_section / continue_writing / done / structure_change
           （done 含 completeness gate：未寫完不匯出、停 checkpoint 問釐清 — #11B 對齊）
        """
        # S1 四段式 confirm（A/B/K + K Round 4，3方共識 + in-house R3 終驗）：上一輪已 emit
        # recollect consent prompt，這輪 user 回答。分四段路由，避免 v1「非確認詞一律當取消」
        # 吞掉 substantive 訴求，並修 K Round 4「無 token 自然肯定句漏接 → 二次 consent loop」。
        # M-1（已知低風險邊界）：此攔截在 auto_continue 分支**之前**，但條件含
        # `user_message.strip()` —— auto_continue / 離線 auto-advance 通常無 user 文字
        # （空訊息）→ 不會誤觸此攔截、pending flag 保留到下次真 user 回覆。極端情況
        # （pending=True 期間恰有非空 auto 訊息）才可能脫節，機率低且最壞結果是多問一次
        # consent（非刪錯章節）→ 標為已知低風險，不額外加 guard（避免過度工程）。
        if getattr(state, "pending_recollect_confirmation", False) and user_message.strip():
            msg_norm = user_message.strip()
            # 一律先清旗標：無論走哪段，這輪都已消費此 consent（避免殘留下輪誤攔）。
            state.pending_recollect_confirmation = False
            if _looks_like_recollect_confirm(msg_norm):
                # 段1：含確認 token 的 bounded affirmative（「確認」「OK。」「好，開始吧」）
                # → 直接執行補搜（不打 LLM，省成本）。快路徑，明確確認詞即命中。
                logger.info("[LIVE RESEARCH] Stage 5: recollect confirmed by user (token)")
                return await self._dispatch_recollect(state)
            # 段2：未含確認 token 的訊息 → 先打既有 abort 分類器。abort 必須**先於**
            # 段3 的「無 token 短肯定兜底」判定 —— 否則「算了」（短、無修改 marker、無 token）
            # 會被段3 誤當 confirm 觸發不可逆刪章。abort 優先級最高（誤判代價最高）。
            meta = await _classify_meta_intent(user_message, self.handler)
            # B fail-loud（抄 export gate :5844-5854 / nav restart gate :1088-1102）：
            # meta is None = LLM 故障。必在 abort / bounded-affirmative 判定之前攔截，
            # 絕不放行不可逆補搜（清章節 + 重蒐 + 重寫 = 燒錢 + 資料流失）。
            # 不可 silent fail（#21）：停原地、emit 系統端旁白、重設 pending flag 讓 user
            # 恢復後可再確認一次（不誤觸刪章）。
            if meta is None:
                logger.warning(
                    "[LIVE RESEARCH] Stage 5 recollect-confirm meta-intent classify failed "
                    "(None) — stay at checkpoint, NOT dispatching recollect"
                )
                await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
                state.pending_recollect_confirmation = True   # 重設：本輪已於 5738 清，故障後復原讓 user 再確認
                state.set_checkpoint(lr_copy.RECOLLECT_CONSENT_PROMPT)
                await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
                await self._persist_checkpoint_boundary(state)
                return state
            if meta == META_INTENT_ABORT:
                # 明確取消（「算了/取消/不要了」）→ 回常規 Stage 5 checkpoint，不刪章節。
                logger.info("[LIVE RESEARCH] Stage 5: recollect cancelled by user (abort)")
                await self._emit_narration(lr_copy.RECOLLECT_CANCELLED_NARRATION)
                state.set_checkpoint("目前所有段落已寫完。要修改哪個段落，或進入匯出？")
                await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
                await self._persist_checkpoint_boundary(state)
                return state
            # 段3（K Round 4，in-house R3 終驗修）：非 abort、且**無修改 marker 的短肯定句**
            # → 視為確認，執行補搜。這一段**不依賴確認 token 白名單** —— 解決「好，那就重新
            # 蒐集吧」「是的」「行」「成」這類**無 token 自然肯定句**漏接落 substantive →
            # _parse_revision_intent 因含「重新蒐集」重 parse 成 recollect → recollect 分支
            # 再設 pending + 再 emit consent = **二次 consent loop**（user 已確認卻被再問）。
            #
            # 為何不靠 _classify_meta_intent 判「肯定」：親驗 _classify_meta_intent（orchestrator.py
            # :320，2026-06-16）只有 3 個 category（SKIP / ABORT / SUBSTANTIVE），**無 affirmative
            # 類**。「好，那就重新蒐集吧」會被它判 substantive（非 abort、非 skip）→ 無法用它
            # 區分「確認」vs「實質訴求」。故改用語意上界：**在 consent gate 內**（剛被問
            # 「確認要重新蒐集嗎？」），非 abort 的「無修改 marker 短句」語意明確就是確認。
            #
            # B 原罪防護仍在：含修改名詞 marker（段/章/改/加/經濟…）→ 不走此兜底，落段4
            # substantive fall through（「改第3段」「資料還是不夠，連經濟面也查」不會誤觸刪章）。
            if _looks_like_bounded_affirmative_shape(msg_norm):
                logger.info(
                    "[LIVE RESEARCH] Stage 5: recollect confirmed by user "
                    f"(bounded affirmative shape, no token, meta={meta})"
                )
                return await self._dispatch_recollect(state)
            # 段4：其餘 substantive（如「改第3段」「再多查經濟面」「資料不夠連政治面也查」）
            # → 不 return，fall through 到下方既有 dispatch（_parse_revision_intent 正常路由）。
            # 「不漏使用者任何一句話」鐵律：consent round 的 substantive 回覆不可被吞。
            logger.info(
                "[LIVE RESEARCH] Stage 5: pending-confirm got substantive reply "
                f"(meta={meta}) — fall through to normal dispatch"
            )

        if auto_continue or not user_message.strip():
            # mock_bab E2E fix (2026-05-29): 「讀豹決定」/auto_continue 不可在未寫完時
            # 匯出。total 來源 = _resolve_chapter_source (與 _run_stage_5 同源)；
            # 未寫完 → 繼續寫下一段（_run_stage_5 idempotent guard 處理寫完 case）。
            _cm_for_total = ContextMap.model_validate_json(state.context_map_json)
            _writer_sections_for_total, _ = self._resolve_chapter_source(
                _cm_for_total, state.format_specs
            )
            _total_sections = len(_writer_sections_for_total)
            if state.last_completed_section_index + 1 < _total_sections:
                logger.info(
                    f"[LIVE RESEARCH] Stage 5: auto_continue with sections remaining "
                    f"(written={state.last_completed_section_index + 1}/{_total_sections}), "
                    f"writing next section instead of exporting"
                )
                return await self._run_stage_5(state)
            logger.info("[LIVE RESEARCH] Stage 5: all sections written, proceed to export")
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        # Bug #14 root fix：shortcut 改「正規化後整句完全匹配白名單」取代 substring + veto
        # 枚舉。只有整句純確認/純匯出詞才走 shortcut（省 LLM 成本）；任何帶內容句
        # （「好像哪裡怪」「不錯,繼續」「第2段還沒完成」）一律 fall through 到 LLM intent
        # parse（永遠安全，LLM 正確分類）。結構上不再有 substring 漏洞，也不需 veto 詞。
        msg_stripped = user_message.strip()

        # 明確匯出意圖：整句完全等於匯出詞 → completeness-aware
        # LR #11B：沒寫完 → block（不給匯出路徑）；全寫完 → 直接進 Stage 6。
        if _looks_like_export_shortcut(msg_stripped):
            logger.info(f"[LIVE RESEARCH] Stage 5: export keyword shortcut hit ('{msg_stripped}')")
            remaining = self._stage5_remaining_count(state)
            if remaining > 0:
                # 寫到一半：完全不給匯出路徑，只能繼續寫 / 修改已寫的
                logger.info(
                    f"[LIVE RESEARCH] Stage 5: export blocked — {remaining} section(s) remaining"
                )
                narration = f"報告還有 {remaining} 段沒寫完，要先寫完才能匯出。要繼續寫嗎？"
                await self._emit_narration(narration)
                state.set_checkpoint(narration)
                await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state
            # 全寫完 → 維持現狀，直接進 Stage 6
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        # VP-7 Phase 3：continue keyword shortcut — 整句完全等於 continue 動詞 → 直接寫下一段
        if _looks_like_continue_shortcut(msg_stripped):
            logger.info(
                f"[LIVE RESEARCH] Stage 5: continue keyword shortcut hit ('{msg_stripped}')"
            )
            return await self._run_stage_5(state)

        # 窄版 meta-intent abort guardrail（防「一句話誤觸不可逆匯出」，最痛 #）。
        # 「算了/取消/不要了/放棄」不在 export/continue frozenset → 會 fall through 到
        # _parse_revision_intent，可能被 LLM 歸 done → complete_stage → Stage 6 匯出半成品
        # （資料流失級災難）。此處先攔：abort → 絕不匯出，停原地問確認；err toward NOT export。
        meta = await _classify_meta_intent(user_message, self.handler)
        if meta is None:
            # 不可 silent fail（#21）：helper 失敗 → 不可放行匯出，系統端文案 + 停 checkpoint。
            # LR #11B：不再寫死「已寫完」（mid-way 時說謊）→ 改中性文案。
            logger.warning(
                "[LIVE RESEARCH] Stage 5 meta-intent classify failed (None), stay at checkpoint"
            )
            await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
            state.set_checkpoint("系統暫時無法處理，請告訴我要繼續寫、還是修改某段。")
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state
        if meta == META_INTENT_ABORT:
            logger.info("[LIVE RESEARCH] Stage 5: abort/done-ish intent — NOT silent-exporting, ask confirm")
            # LR #11B：completeness-aware abort prompt。
            # 未寫完 → 不給「接受」/「匯出」，只給「繼續寫」/「修改」；
            # 全寫完 → 維持現有「接受 / 繼續編輯」（原來行為）。
            remaining = self._stage5_remaining_count(state)
            if remaining > 0:
                abort_prompt = (
                    f"報告還有 {remaining} 段沒寫完。"
                    "要繼續寫完，還是修改已寫好的某段？"
                )
            else:
                # CEO reframe 2026-06-02：不給「放棄」（phantom need），給「接受 / 繼續編輯」。
                abort_prompt = (
                    "要先這樣完成、把目前進度匯出嗎？\n"
                    "回覆「接受」我就把目前進度整理匯出；"
                    "回覆「繼續編輯」可以接著修改或補充內容。"
                )
            await self._emit_narration(abort_prompt)
            state.set_checkpoint(abort_prompt)
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state  # 停原地，絕不靜默匯出；接受/繼續編輯由下一輪 reply 決定
        if meta == META_INTENT_SKIP:
            # Stage 5 沒有 Stage 3 的「用預設」下游動作（Stage 3 見 META_INTENT_SKIP→use default）。
            # 「跳過/不用了/不提供」在 Stage 5 語意不明：可能想匯出、可能想繼續、可能想停。
            # 過去靜默 fall-through 到 _parse_revision_intent → 可能被誤判 done → 匯出半成品，
            # 且全程不告知 user 系統如何解讀（UX 不透明）。改為停原地、emit completeness-aware
            # 釐清 narration，讓 user 用明確意圖回覆（與 abort/vague 分支同設計：不替 user 猜）。
            logger.info("[LIVE RESEARCH] Stage 5: meta-intent=SKIP — ask user to clarify, not routing to revision intent")
            remaining = self._stage5_remaining_count(state)
            if remaining > 0:
                # 文案必含子字串「繼續寫」（Step 1 測試斷言鍵詞；review round 1 blocker fix，矩陣 #1）。
                # 「繼續寫完」含「繼續寫」；句式與 abort 分支同源（「要繼續寫完，還是修改…」）但
                # 措辭可區分 SKIP 與 ABORT 語境。
                skip_prompt = (
                    f"報告還有 {remaining} 段沒寫完。"
                    "你是想要我繼續寫完剩下的，還是要修改已寫好的某段？"
                )
            else:
                skip_prompt = (
                    "目前所有段落已寫完。"
                    "回覆「接受」我就把目前進度整理匯出；"
                    "回覆「繼續編輯」可以接著修改或補充內容。"
                )
            await self._emit_narration(skip_prompt)
            state.set_checkpoint(skip_prompt)
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state  # 停原地，等 user 下一輪明確意圖
        # 註：「接受」加入 _EXPORT_SHORTCUT_KEYWORDS（見下方）→ 下一輪「接受」
        #     走 export frozenset shortcut 直接匯出；「繼續編輯」fall through 到 revise。
        # META_INTENT_SUBSTANTIVE → fall through 到既有 _parse_revision_intent（零行為改變）。
        # （META_INTENT_SKIP 已在上方顯式攔下並 emit 釐清 narration，不再混入 revision intent。）

        # 用 LLM intent parsing 辨識使用者要修改什麼（CEO 決策：自然語言回饋不可用 title matching）
        revision_intent = await self._parse_revision_intent(user_message, state.written_sections)

        # parse fail 改 B（保持 checkpoint + narration），不再 silent advance。
        # 區分兩種失敗（#20 鋪出的改善）：
        #   - revision_intent is None：_parse_revision_intent 因 LLM API 失敗
        #     （429 quota / timeout / 空回應）回 None → 系統端問題，怪 user「沒看懂」
        #     會誤導（user 重講也沒用）→ 明說系統暫時無法處理。
        #   - dict 但 action 空：LLM 成功但真的判不出意圖 → user 表達模糊，重講有用。
        if revision_intent is None:
            logger.warning("[LIVE RESEARCH] Stage 5 revision intent LLM call failed (None), stay at checkpoint")
            # #20 改善：共用 module 級 lr_copy.LLM_UNAVAILABLE_NARRATION，三處 None 分支文案一致。
            await self._emit_narration(lr_copy.LLM_UNAVAILABLE_NARRATION)
            state.set_checkpoint(
                "目前所有段落已寫完。要修改哪個段落，或進入匯出？"
            )
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state
        if not revision_intent.get("action"):
            logger.warning("[LIVE RESEARCH] Stage 5 intent parse: no action (vague), stay at checkpoint")
            await self._emit_narration(
                "我沒看懂你的意思，可以再說一次嗎？"
                "例如「第 3 段太短，請補充」或「進入匯出」。"
            )
            state.set_checkpoint(
                "目前所有段落已寫完。要修改哪個段落，或進入匯出？"
            )
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        action = revision_intent.get("action", "done")

        if action == "recollect":
            logger.info("[LIVE RESEARCH] Stage 5: recollect intent — emit consent checkpoint")
            # cap 預檢：已達上限直接 block（不進 consent），明確告知（非 silent）。
            if state.recollect_count >= self._recollect_cap():
                logger.info(
                    f"[LIVE RESEARCH] Stage 5: recollect capped "
                    f"(count={state.recollect_count}), blocked"
                )
                await self._emit_narration(lr_copy.RECOLLECT_CAPPED_NARRATION)
                state.set_checkpoint(lr_copy.RECOLLECT_CAPPED_NARRATION)
                await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
                await self._persist_checkpoint_boundary(state)
                return state
            # 未達 cap：S1 informed consent — emit consent prompt，設旗標等下一輪確認。
            state.pending_recollect_confirmation = True
            await self._emit_narration(lr_copy.RECOLLECT_CONSENT_PROMPT)
            state.set_checkpoint(lr_copy.RECOLLECT_CONSENT_PROMPT)
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)
            return state

        if action == "structure_change":
            logger.info("[LIVE RESEARCH] Stage 5: structure_change redirect")
            await self._emit_narration(
                "章節結構在第一階段（研究結構提案）確認。"
                "這個階段只能修改單一段落的內容（例如「第 2 段太短」「第 3 段補充 X」）。"
                "如果需要大幅調整章節結構，建議啟動新查詢。"
            )
            state.set_checkpoint(
                "目前所有段落已寫完。要修改哪個段落，或進入匯出？"
            )
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        if action == "done":
            # D-2026-06-11 決策 4：LLM-done completeness gate（#11 Part B 對齊）。
            # 整句「完成」走 export keyword shortcut 未寫完會被 block（見上方
            # _looks_like_export_shortcut 分支），但語意等價的自然語句（「好了就這樣」）
            # 走 LLM → done 原本直接 complete_stage → Stage 6 gate 是 warn-only →
            # 匯出半成品 = #11「中途完全不給匯出」的漏網路徑。此處補同款 gate：
            # 未寫完 → 停 checkpoint 問釐清（不硬轉 continue，違逆 user 結束意圖）；
            # 全寫完 → 原行為不變。
            remaining = self._stage5_remaining_count(state)
            if remaining > 0:
                logger.info(
                    f"[LIVE RESEARCH] Stage 5: LLM-done blocked — "
                    f"{remaining} section(s) remaining"
                )
                done_gate_prompt = lr_copy.stage5_done_unfinished_gate_prompt(remaining)
                await self._emit_narration(done_gate_prompt)
                state.set_checkpoint(done_gate_prompt)
                await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
                await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
                return state
            logger.info("[LIVE RESEARCH] Stage 5: user confirmed done")
            state.complete_stage()
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state

        if action == "continue_writing":
            # user 選「繼續寫剩下的」→ 重跑 _run_stage_5
            # （resume 邏輯靠 last_completed_section_index skip 已寫段落）
            logger.info(
                f"[LIVE RESEARCH] Stage 5: resume writing from section "
                f"{state.last_completed_section_index + 1}"
            )
            return await self._run_stage_5(state)

        # Bug 1 fix (off-by-one)：_parse_revision_intent 的 prompt 契約已改為回傳
        # **1-based 使用者口語段號**（「第 N 段」就填 N）。此處一次轉成 0-based 餵下游
        # （writer_sections[]、user_voice key、range check、clamp 全對齊 0-based）。
        # 只在 integer 時轉；None（target 不明 → clarifying question 分支）不轉、也不會
        # 被下游 index 取用。轉換只在這唯一一處發生，避免重複減 1。
        revision_target = revision_intent.get("target_index")
        if isinstance(revision_target, int):
            revision_target -= 1
        context_map = ContextMap.model_validate_json(state.context_map_json)
        # writer_sections 用同樣 helper 解析（chapter override 對齊）
        writer_sections, using_chapter_override = self._resolve_chapter_source(
            context_map, state.format_specs
        )

        # FIX-6 (Cayenne #14, 2026-05-29)：反轉舊 D-D 決策。
        # 舊 D-D（已廢）：target_index 缺失時靜默 fallback 到 last_completed_section_index，
        #   理由是「mini-checkpoint 剛寫完第 K 段，模糊回覆幾乎一定指 K」「比 clarifying
        #   question 少一輪互動」。
        # 反轉理由：Cayenne 實測在 Stage 5 說「再修一下」沒指明章節時，系統靜默挑一段
        #   改錯地方 —— 靜默改錯段比多問一句嚴重。user 表達 revise 意圖但沒指明第幾章
        #   時，改成 emit clarifying question 列出已寫章節、停在 checkpoint 等 user 回，
        #   不 mutate 任何 section、不推進。
        # 注意：只攔「意圖 revise 但 target 不明（None）」；target 有給但超出範圍（out of
        #   range）仍走 clamp（user 確實指了一段，只是 index 算錯，不需再問）。
        fallback_target = max(state.last_completed_section_index, 0)
        if revision_target is None:
            # 列出目前已寫章節（1-based 顯示），讓 user 明確指定要改哪一章
            written = state.written_sections or []
            if written:
                chapter_lines = "、".join(
                    f"第 {s.get('section_index', i) + 1} 章「{s.get('title', '')}」"
                    for i, s in enumerate(written)
                )
                chapter_hint = f"目前已寫 {len(written)} 章：{chapter_lines}"
            else:
                chapter_hint = "目前尚未寫完任何章節"
            logger.info(
                "[LIVE RESEARCH] Stage 5 revise_section target missing — "
                "emit clarifying question (FIX-6, no silent fallback)"
            )
            await self._emit_narration(
                f"請指明要修改哪一段（第幾章）？{chapter_hint}。"
            )
            state.set_checkpoint(
                f"請指明要修改哪一段（第幾章）？{chapter_hint}。"
            )
            await self._emit_checkpoint(stage=5, proposal=state.checkpoint_prompt)
            await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
            return state
        if not (0 <= revision_target < len(writer_sections)):
            logger.warning(
                f"[LIVE RESEARCH] Stage 5 revise_section target={revision_target} "
                f"out of range [0, {len(writer_sections)}); clamp to {fallback_target}"
            )
            revision_target = fallback_target

        # 取 topic（可能是 dict 或 ContextMapTopic）— 下游 _write_section 用 writer 端 spec
        topic_spec = writer_sections[revision_target]
        # narration 章名須與 revision_target 的 index 來源對齊：revision_target 是
        # written_sections-based（_parse_revision_intent 用 enumerate(written_sections)
        # 餵段號給 LLM），故從 state.written_sections 取章名、key 為 "title"。
        # （舊 bug：narration 誤用 writer_sections[revision_target]，兩 list 順序不同步 → 顯示錯章名）
        topic_name = (
            state.written_sections[revision_target].get("title", "該段落")
            if 0 <= revision_target < len(state.written_sections)
            else "該段落"
        )
        await self._emit_narration(f"正在修改「{topic_name}」段落...")

        style_features = None
        if state.style_features_json:
            style_features = StyleAnalysisOutput.model_validate_json(state.style_features_json)

        # 載入 evidence_pool（revision 路徑也要傳）
        evidence_pool = deserialize_evidence_pool(state.evidence_pool_json)

        # Plan: lr-user-voice-container-and-4-fixes (Fix I-1)
        # CEO OQ 2 拍板：accumulate list — 同段多次 revise 全保留 ordered list。
        # state 寫入：append 當輪 instruction 到對應 section index list。
        # Writer prompt 拿到：當輪 instruction + prior content（writer 需要看「最新訴求」
        # 配合 prior 版內容）。完整歷史 list 在 prompt builder 端拼接（讓 LLM 看完整
        # 修訂軌跡），但 _write_section interface 只接 str — 由 orchestrator 串接後傳入。
        current_instruction = (revision_intent.get("instruction") or "").strip()
        accumulated_instructions: List[str] = []
        if current_instruction:
            state.user_voice.revise_instructions.setdefault(
                revision_target, []
            ).append(current_instruction)
            accumulated_instructions = list(
                state.user_voice.revise_instructions[revision_target]
            )

        # 串接 instruction 給 writer：最新 instruction 是當輪訴求，前面是歷史 context。
        # 多輪情境下，writer prompt 看到「之前已要求 X，本次再要求 Y」整段，
        # LLM 能合理推斷增量訴求 vs 取消前次訴求。單輪時 list 長度 = 1，行為與 single
        # instruction 等價。空 instruction（user 沒給）→ revise_instruction_to_writer = None。
        revise_instruction_to_writer: Optional[str] = None
        if accumulated_instructions:
            if len(accumulated_instructions) == 1:
                revise_instruction_to_writer = accumulated_instructions[0]
            else:
                # Multi-round：用 numbered list 呈現歷史軌跡 + 標示最新訴求
                numbered = "\n".join(
                    f"{i+1}. {ins}" for i, ins in enumerate(accumulated_instructions)
                )
                revise_instruction_to_writer = (
                    f"使用者對本段提出 {len(accumulated_instructions)} 輪修改訴求（"
                    "最新的為當前要求，先前輪是 context）：\n"
                    f"{numbered}"
                )

        # Prior content：從 written_sections 取對應 section 的上一版內容
        prior_section_content: Optional[str] = None
        if 0 <= revision_target < len(state.written_sections):
            prior_section_content = (
                state.written_sections[revision_target].get("content") or None
            )

        # Track A Task 3 + Mn-1 (sprint 2026-05-28) + codex C-1 v2:
        # revise path 必須 propagate state + book_outline + current_chapter_index
        # 進 _write_section (writer 需要看到 grounding context, 不可 silent fallback
        # 進 writer)。從 state.book_outline_json 還原 outline。
        revise_book_outline = None
        if state.book_outline_json:
            try:
                revise_book_outline = BookOutline.model_validate_json(
                    state.book_outline_json
                )
            except Exception as e:
                logger.warning(
                    f"[LIVE RESEARCH] revise: book_outline_json unparseable "
                    f"({type(e).__name__}: {e}); falling through to None — "
                    "C-1 gate will treat body chapter as blocked"
                )

        # using_chapter_override / all_evidence_ids 對齊 _run_stage_5
        # 推導: _resolve_chapter_source 已在前面 (line 2840) 給出 using_chapter_override
        # 此處只需 all_evidence_ids 給 Phase 3 union-to-first (chapter_index=0 用)
        revise_all_evidence_ids: list = []
        if using_chapter_override:
            _union_ids: set = set()
            for t in context_map.topics:
                _union_ids.update(t.evidence_ids)
            revise_all_evidence_ids = sorted(_union_ids)

        # Track A Task 7: revise path 同樣計算 prior_used_entities (前面 N 章已寫的
        # entities 全集; 不含本章 — revision_target 章節是被重寫的, 用 [:revision_target] slice)
        _revise_prior_acc: List[str] = []
        for prior in state.written_sections[:revision_target]:
            _revise_prior_acc.extend(prior.get("entities", []))
        _revise_seen: set = set()
        revise_prior_used_entities: List[str] = []
        for e in _revise_prior_acc:
            if e and e not in _revise_seen:
                _revise_seen.add(e)
                revise_prior_used_entities.append(e)

        # Track B2 (sprint 2026-05-28): revise path citation preservation —
        # 取 prior section 的 sources_used 作為 fallback，傳入 _write_section。
        # chapter_index>0 原本設 analyst_citations=[] 導致 revision 引用全掉；
        # 有 prior_sources_used 時 _write_section 會 fallback 保留上一版引用。
        revise_prior_sources_used: Optional[list] = None
        if 0 <= revision_target < len(state.written_sections):
            revise_prior_sources_used = list(
                state.written_sections[revision_target].get("sources_used") or []
            )
            logger.debug(
                f"[LIVE RESEARCH] B2: revision_target={revision_target} "
                f"prior_sources_used={revise_prior_sources_used}"
            )

        section_output, was_corrected = await self._write_section(
            context_map=context_map,
            topic=topic_spec,
            style_features=style_features,
            format_specs=state.format_specs,
            evidence_pool=evidence_pool,
            chapter_index=revision_target if using_chapter_override else None,
            all_evidence_ids=revise_all_evidence_ids if using_chapter_override else None,
            book_outline=revise_book_outline,
            current_chapter_index=revision_target,
            user_voice=state.user_voice,
            revise_instruction=revise_instruction_to_writer,
            prior_section_content=prior_section_content,
            state=state,  # Track A Mn-1 / codex C-1 v2: propagate state for render
            prior_used_entities=revise_prior_used_entities,  # Track A Task 7
            prior_sources_used=revise_prior_sources_used,  # Track B2: citation preservation
        )
        state.hallucination_corrected = state.hallucination_corrected or was_corrected

        # Track A Task 7: 重寫後重抽 entities (內容變了, entities 也可能變;
        # blocked / guard_failed 章節跳過抽取)
        _revise_section_entities: List[str] = []
        if getattr(section_output, "status", "drafted") == "drafted":
            _revise_section_entities = await self._extract_section_entities(
                section_output.section_content, self.handler,
            )

        # addendum I-1: 抽 _section_dict helper 統一構造 (revise in-place + else-branch
        # 同走 helper; 避免漏 entities/status key)
        if revision_target < len(state.written_sections):
            state.written_sections[revision_target] = _section_dict(
                section_output, revision_target, entities=_revise_section_entities,
            )
        else:
            state.written_sections.append(
                _section_dict(
                    section_output, revision_target,
                    entities=_revise_section_entities,
                )
            )

        await self._emit_section(revision_target, section_output, state)

        # 再次 checkpoint — 保持在 Stage 5 dialogue loop
        checkpoint_text = "修改完成。還需要調整其他段落嗎？或者可以進入匯出？"
        state.set_checkpoint(checkpoint_text)
        await self._emit_checkpoint(stage=5, proposal=checkpoint_text)
        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state  # 保持在 Stage 5 checkpoint

    async def _parse_revision_intent(self, user_message: str, written_sections: list) -> dict:
        """用 LLM 判斷使用者要修改哪個段落。

        CEO 決策：段落修改辨識同理，不可用 title matching。
        """
        if self.dry_run:
            return {"action": "done", "reason": "dry-run: always done"}

        from core.llm import ask_llm

        # Build section summary for context（1-based 顯示，對齊使用者口語「第 N 段」）
        sections_summary = ""
        for i, section in enumerate(written_sections):
            title = section.get("title", f"段落 {i+1}")
            sections_summary += f"第 {i+1} 段：{title}\n"

        prompt = f"""你是一個意圖分析器。使用者正在閱讀一份研究報告，並提出修改意見。

已完成的段落（以使用者口語段號排列，從第 1 段開始）：
{sections_summary}
使用者訊息：
{user_message}

判斷使用者的意圖，回傳 JSON：

- action:
  * "revise_section"（user 要求修改某個特定段落，**包含對「一段之內」的任何編輯**：
      重寫整段、加強、補資料、精簡、刪掉段內某些句子，以及**段內順序操作**
      （把一段裡的論點/句子對調、重排、調順序、換順序）。例如：
      「第 2 段太弱，多加引用」/「把離岸風電那段重寫」/「改第三段」/
      「第一段論點不清楚，重組一下」/「核能那段加上 IAEA 數據」/
      「把第1段重新排列，先講結論再講背景」/「把這段的論點順序對調」/
      「最後一段順序調一下」/「這部分的順序調一下」。
      ⚠ 只要順序/排列動詞（對調/重排/調順序/換順序）作用在**一段之內**，
      無論錨點是段號、章節標題、近指代「這段/這部分/這裡」還是位置序數
      「最後一段」，都是 revise_section，**不是** structure_change）
  * "revise_all"（user 要求全部重寫或整體大規模重做，例如：
      「全部重寫」/「整篇重來」/「都不滿意，重做」/「整份報告重新寫過」）
  * "done"（user 明確表達完成接受、要進入匯出，例如：
      「先這樣」/「可以了」/「夠了」/「完成」/「進入匯出」/「OK 匯出吧」/「就這樣」）
  * "structure_change"（**僅限「章 / 章與章之間 / 整章」層級**的結構操作：
      合併整章、拆分整章、刪整章、改章數、章與章之間重排。
      例如：「合併第 1+3 章」/「拆分第 2 章」/「改成 5 章」/「刪掉第 3 章」。
      ⚠ 關鍵區隔：**一段之內**的重排/對調/調順序/刪句/改寫**不屬於**
      structure_change（那是 revise_section）。structure_change 的操作對象
      一定是「整章」或「章與章之間」，不會是某一段內部。
      這類訴求現階段無法處理，分類即可，後續會 friendly redirect 給 user）
  * "continue_writing"（user 剛 stop 後選擇繼續寫剩下的段落，例如：
      「繼續」/「繼續寫」/「寫完剩下的」/「continue」/「剩下的」/「把剩下的寫完」/
      「往下寫」/「接著寫」。只在 user 之前按過停按鈕後出現此意圖，
      表示 user 想 resume writer loop 從上次中斷處繼續）
  * "recollect"（user 要求**去找更多/新的資料**來補強，而非用現有資料重寫。例如：
      「這部分資料不夠，去多查一些」/「證據太薄，需要更多來源」/「再去找一些相關報導」/
      「資料量不足，請補充蒐集」。與 revise_section 的關鍵區別：revise_section 是用
      現有資料重寫某段；recollect 是要求重新蒐集**新資料**再整體重做。）

- target_index: **只有當 user 用「段號」或「章節標題」明確指出是哪一段時**才填。填**使用者口語的段號（第 N 段就填整數 N，從 1 開始算）**，對齊上方「已完成的段落」清單的段號。例如使用者說「第 2 段」就填 2；用標題指定（如「離岸風電那段」「核能那段」）時，對照清單找出該標題對應的段號（第幾段）填入。
  **位置序數**（如「第一段」「最後一段」「倒數第二段」）可對照上方清單算出是第幾段 → 視同明確指定，填該段整數（最後一段 = 清單最後一筆的段號；倒數第二段 = 倒數第二筆）。
  以下情況 target_index **一律必須為 null**（**不要猜、不要預設、不要挑任何一段**，系統會反問 user 要改哪一段，避免改錯段）：
    1. 完全沒指明哪一段（如「再修一下」「改一下」「重寫」未帶任何段號/標題/位置線索）。
    2. **近指代名詞（proximal deixis）「這…」**：用「這段」「這裡」「這部分」「這邊」「這一段」等**「這／此」開頭、指稱使用者『眼前正在看』的某段**，而**沒有**附帶段號、章節標題或位置序數（如「這段怪怪的」「這裡卡卡的」「語氣太硬」「這部分重寫」）。你**看不到 user 正在看哪一段**，「這段」無法對應到清單的任何段號，**必須回 null**，絕對不可猜成第 2 段或任何一段。（注意：「最後一段」「第三段」「離岸風電那段」這類有位置/段號/標題的不屬於此類，要正常填整數。）
    3. **相對指代（relative / anaphoric deixis）「前面那段／上一段／下一段」**：用「前面那段」「上一段」「前一段」「後面那段」「下一段」「上面那段」「下面那段」等**相對於使用者『目前視線焦點』的方向詞**，而**沒有**附帶段號、章節標題或可對照清單算出的位置序數。這類「前／後／上／下」是相對 user 眼前在看的位置，你**看不到 user 正在看哪一段**，無法換算成清單上的任何段號，**必須回 null**，絕對不可猜成 last_completed 前一段或任何一段。（關鍵區隔：「最後一段」「倒數第二段」是**絕對位置序數**，能對照清單算出第幾段，要正常填整數；「前面那段」「上一段」是**相對方向**，算不出絕對段號，回 null。）
  只有當訊息本身帶有可對照上方清單的**段號**（第 N 段）、**絕對位置序數**（最後一段／倒數第 N 段）或**章節標題**時才填整數。近指代「這…」、相對指代「前面那段／上一段」這類無法對應到清單具體段號的，一律 null，不准猜。其他 action 一律 null。
- instruction: 使用者的修改指示（繁體中文原文摘要）
- reason: 簡述判斷原因（繁體中文）

紀律：
- 任何包含「繼續」「寫完」「剩下的」「接著寫」等 resume 動詞 → continue_writing
- 任何「資料不夠/不足/太薄/去多查/找更多來源/補充蒐集」等要求蒐集新資料的訊號 → recollect
  （注意：「第 N 段重寫/加強」用現有資料 → revise_section；「去找更多資料」→ recollect）
- 順序/排列動詞（對調/重排/調順序/換順序）**作用在「一段之內」**→ revise_section
  （錨點是段號、標題、「這段/這部分/這裡」或「最後一段」都一樣，是段內操作）
- 任何包含「特定段落 index/標題」+「修改動詞」（重寫/加強/補/改/精簡/換/重組/刪句）→ revise_section
- **章 / 章與章之間 / 整章**層級的結構操作（合併整章/拆分整章/改章數/刪整章/章與章重排）→ structure_change
- ⚠ 動詞本身不決定 action：「重排/對調/調順序」要看作用層級 —— 一段之內 = revise_section，章與章之間 = structure_change
- 任何明確的接受/完成/匯出訊號 → done
- done 僅適用於 user 完全沒提任何修改、只表達接受或要求匯出
- 如果無法明確分類，傾向 revise_section（保守）而非 done（會吃掉 user 訴求）
"""
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": [
                    "revise_section", "revise_all", "done",
                    "structure_change", "continue_writing", "recollect",
                ]},
                "target_index": {"type": ["integer", "null"]},
                "instruction": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["action", "reason"],
        }
        try:
            response = await ask_llm(
                prompt,
                schema,
                level="low",
                query_params=getattr(self.handler, 'query_params', {}),
                max_length=4096,
            )
            if not response:
                # 改動 2：parse fail 改 B（caller 偵測 None 後 narration + 保持 checkpoint）
                logger.warning("[LIVE RESEARCH] _parse_revision_intent: ask_llm returned empty")
                return None
            # Unwrap schema-wrapped response: {type, properties, required} → properties dict
            if "action" not in response and "properties" in response and isinstance(response["properties"], dict):
                response = response["properties"]
            return response
        except Exception as e:
            logger.warning(f"[LIVE RESEARCH] _parse_revision_intent failed: {e}")
            return None

    # ──── Stage 6: 匯出 ───────────────────────────────────────

    def _sanitize_report_title(self, raw: str) -> str:
        """AR P2：把 LLM 生成標題正規化成單行 markdown-safe 文字。
        換行摺空格、剝前導 markdown 標記（# / > / * / -）、collapse 空白、超長截斷。
        """
        import re
        t = re.sub(r"\s+", " ", (raw or "")).strip()
        # 剝前導 markdown 區塊「語法」標記（標記+空白，如 '# ' '> ' '- '），
        # 避免生成標題本身帶 '# ' 破壞 H1 結構。
        # AR R2 nit：只剝「標記+空白」的真前導語法，不誤刪以 '#'/'-' 字面開頭的
        # 合法標題（如 '#MeToo運動' / '-40%的降幅'）。
        t = re.sub(r"^(#{1,6}\s+|[>*\-]\s+)+", "", t).strip()
        if len(t) > _REPORT_TITLE_MAX_LEN:
            t = t[:_REPORT_TITLE_MAX_LEN].rstrip()
        return t

    async def _generate_report_title(
        self, context_map: "ContextMap", written_sections: List[Dict]
    ) -> tuple:
        """Stage 6 組 H1 前生成報告標題（CEO 拍板：low-tier LLM）。

        Returns:
            (title: str, was_generated: bool)
            - 成功 → (後處理過的生成標題, True)
            - 失敗 / timeout / 空回應 → (research_question, False) + logger.warning
              （讀豹鐵律：不可 silent fail；AR R1：降級用顯式 boolean 而非字串相等傳遞）

        input = research_question + 各章標題/摘要。鏡像 loop_engine title backfill 降級 pattern。
        """
        research_question = (context_map.research_question or "").strip()

        # 組各章「標題：摘要」清單餵 LLM（有 chapter_summary 用摘要，無則只給標題）
        chapter_lines = []
        for s in written_sections:
            title = str(s.get("title", "") or "").strip()
            if not title:
                continue
            summary = str(s.get("chapter_summary", "") or "").strip()
            chapter_lines.append(f"- {title}：{summary}" if summary else f"- {title}")
        chapters_block = "\n".join(chapter_lines) if chapter_lines else "（無章節摘要）"

        prompt = (
            "以下是一份研究報告的原始研究問題與各章節標題／摘要。\n"
            "請為整份報告生成一個有質感、具體、有資訊量的繁體中文標題（不超過 40 字），"
            "概括報告的核心主旨與張力。\n"
            "禁止用「研究報告」「分析」「探討」「淺談」等泛化空詞當開頭套語，"
            "直接點出主題。\n\n"
            f"原始研究問題：{research_question}\n\n"
            f"各章節：\n{chapters_block}"
        )

        try:
            response = await ask_llm(
                prompt,
                GeneratedReportTitle.model_json_schema(),
                level="low",
                query_params=getattr(self.handler, "query_params", {}),
                timeout=_REPORT_TITLE_TIMEOUT,
            )
            raw = GeneratedReportTitle.model_validate(response).title or ""
            generated = self._sanitize_report_title(raw)
            if generated:
                return (generated, True)
            # 空回應（含後處理後變空）→ 降級（不 silent fail）
            logger.warning(
                "[LIVE RESEARCH] report title LLM returned empty; "
                "degrading to research_question=%r",
                research_question,
            )
            return (research_question, False)
        except Exception as e:
            # 失敗 / timeout → 降級 research_question（不 silent fail）
            logger.warning(
                "[LIVE RESEARCH] report title LLM generation failed "
                "(%s: %s); degrading to research_question=%r",
                type(e).__name__, e, research_question,
            )
            return (research_question, False)

    async def _run_stage_6(self, state: LiveResearchStageState) -> LiveResearchStageState:
        """Stage 6: 組合並匯出。"""
        self._maybe_reset_offline_counters(state)  # online substantive advance → reset（plan 3d）
        state.advance_to_stage(6)
        await self._emit_stage_change(6)

        # mock_bab E2E fix (2026-05-29): 防禦性 completeness gate（Cayenne lesson —
        # 不可逆動作匯出前需 completeness gate）。正常 path（auto_continue / continue /
        # done）已確保寫完才到這；此為最後一道防線，若仍有未寫章節即匯出 → 明確 narration
        # 警告（CLAUDE.md：可降級但必須明確訊息，不可 silent）。
        try:
            _cm_gate = ContextMap.model_validate_json(state.context_map_json)
            _writer_sections_gate, _ = self._resolve_chapter_source(
                _cm_gate, state.format_specs
            )
            _total_expected = len(_writer_sections_gate)
            _written_count = len(state.written_sections)
            if _written_count < _total_expected:
                await self._emit_narration(
                    f"⚠ 報告僅完成 {_written_count}/{_total_expected} 章節即匯出，"
                    "尚有章節未撰寫。建議返回繼續撰寫剩餘章節後再匯出。"
                )
                logger.warning(
                    f"[LIVE RESEARCH] Stage 6 export with incomplete sections: "
                    f"written={_written_count} < expected={_total_expected}"
                )
        except Exception as e:
            # gate 本身故障不可阻斷匯出，但必須留 log（不 silent fail）
            logger.warning(
                f"[LIVE RESEARCH] Stage 6 completeness gate check failed "
                f"({type(e).__name__}: {e}); proceeding with export"
            )

        # Hallucination Guard 觸發過 → narration 提示使用者檢視 confidence=Low 段落
        if state.hallucination_corrected:
            await self._emit_narration(lr_copy.HALLUCINATION_CORRECTED_NARRATION)

        # Track A (sprint 2026-05-28) addendum C-2 + codex Imp-2:
        # 偵測 blocked / guard_failed chapter, 發 SSE narration + final report
        # header user-visible 警告 (雙重提醒 — SSE 可能被 user dismiss, header
        # 持久 user export 後仍看得到)。
        problematic = [
            s for s in state.written_sections
            if s.get("status") in _PROBLEMATIC_STATUSES
        ]
        if problematic:
            n = len(problematic)
            titles = ", ".join(
                f"「{s.get('title', '?')}」"
                f"（{_PROBLEMATIC_REASON_ZH.get(s.get('status'), '未完成')}）"
                for s in problematic
            )
            await self._emit_narration(
                lr_copy.problematic_chapters_narration(n, titles)
            )
            logger.warning(
                f"[LIVE RESEARCH] Stage 6 export with {n} problematic chapters: "
                f"{titles}"
            )

        # 組合所有 sections 為 Markdown
        parts = []
        context_map = ContextMap.model_validate_json(state.context_map_json)

        # Track A codex Imp-2: final report header user-visible 警告 (持久)
        # SSE narration 是 runtime 提醒 user 可能 dismiss, header 是持久警告 user
        # export 後仍看得到。**兩處都做**, 不可只做其中之一。
        if problematic:
            n = len(problematic)
            # Bug G：章號 1-based 組裝抽到 lr_copy.build_problematic_chapters_md
            # （reason_map 由此處單一來源 _PROBLEMATIC_REASON_ZH 傳入，不重複定義）。
            problems_md = lr_copy.build_problematic_chapters_md(
                problematic, _PROBLEMATIC_REASON_ZH
            )
            parts.append(
                lr_copy.problematic_chapters_header(n, problems_md)
            )
            logger.warning(
                f"[LIVE RESEARCH] codex Imp-2: prepended incomplete-banner header "
                f"to final_report ({n} problematic chapters)"
            )

        # LR 報告標題生成（plan: lr-report-title-generation）：Stage 6 組 H1 前
        # low-tier LLM 生成有質感標題（CEO 拍板）。失敗已在 helper 內降級退回
        # research_question 且 logger.warning（不 silent fail）。
        # AR R1：helper 回 (title, was_generated)，用顯式 boolean 驅動下游（非字串相等）。
        _research_question = context_map.research_question
        title_for_h1, _title_was_generated = await self._generate_report_title(
            context_map, state.written_sections
        )
        # state 持久化：只在「真生成」時存純標題值（降級存空字串，符合 Task 2「空=降級」定義，
        # 前端 fallback 據此正確分流，不會把降級的 raw query 誤當生成標題）。
        state.generated_report_title = title_for_h1 if _title_was_generated else ""
        parts.append(f"# {title_for_h1}\n")
        # 原始查詢降為副標保留給 user 看他問了什麼（CEO 拍板呈現）。
        # 只在「真生成」時加副標——降級時 H1 已是原查詢，加副標會冗餘重複（用 boolean 判非字串相等）。
        if _title_was_generated:
            parts.append(f"> 原始查詢：{_research_question}\n")

        for section in state.written_sections:
            parts.append(f"## {section['title']}\n")
            parts.append(section['content'])
            parts.append("")

        # References master list（Task 10 — 把 BAB 抓到的 evidence URL 列出來）
        references_block = self._build_references_block(state)
        if references_block:
            parts.append(references_block)

        full_report = "\n".join(parts)

        # Track D D1 (sprint 2026-05-28): KG SSE event metadata（餵前端 D3 視覺化）。
        # kg_payload 仍需構築供下方 SSE "knowledge_graph" 欄位使用。
        # ⚠️ 報告末段 KG JSON section 已於 2026-07-21 暫移除（匯出只支援純文字、
        # KG 的 raw JSON 使檔案體積雙倍且不可讀），待 KG overhaul 後恢復。
        # 原本此處把 kg_payload 以 ```json fence 拼進 full_report 的 markdown section
        # 已刪除；SSE knowledge_graph payload（下方 :emit）保留不變。
        kg_payload = self._build_kg_export_payload(state.knowledge_graph)

        # 路 3 (P-回顧): 把組好的完整報告（含 H1 + sections + references）
        # 存進 state，隨下方 _persist_checkpoint_boundary 落 live_research_state JSONB。
        # 前端回顧主路徑直接讀此字串丟 showLRExport，與本次 export 逐字一致。
        # 雙重組裝根源消除：前端不再自己重組報告（fallback 僅供欄位上線前舊 session）。
        state.final_report_markdown = full_report

        # 推送完整報告（O5+O5b: 走 emit_sse，sender None/例外時 fallback + log，
        # 不靜默吞整份報告）
        await emit_sse(self.handler, {
            "message_type": "live_research_export",
            "format": "markdown",
            "content": full_report,
            # O2 / O2-TF: eid -> {url,title,domain,quote}（與 section event 同 schema）
            "citation_sources": self._build_citation_sources(state),
            # Track D D1: KG metadata 隨 export event 一起送前端
            # (前端 displayKnowledgeGraph 消費；N-9: 擴張 Optional 欄位不破壞
            # 既有 consumer — 沿 Track E E-AMB-3 邊界 lemma)
            "knowledge_graph": kg_payload,  # None / dict — helper 已處理
        })

        state.complete_stage()
        await self._emit_narration("報告匯出完成！")
        await self._persist_checkpoint_boundary(state)  # plan: durable boundary persist + offline-count
        return state

    def _build_kg_export_payload(self, state_kg) -> Optional[Dict[str, Any]]:
        """Track D D1 (sprint 2026-05-28): KG export payload helper.

        fix-up round 1 S-5 / N-12: markdown section + SSE event 共用同一 dict
        構築，避免重複計算 + DRY。

        Args:
            state_kg: state.knowledge_graph (Optional[KnowledgeGraph])

        Returns:
            None 表示無 KG (state_kg=None 或空 entities/relationships) → Stage 6
            export markdown 跳過 KG section, SSE knowledge_graph 設 None。
            dict 表示 KG payload (entities / relationships / metadata)。

        D-AMB-3 LOCKED Option (d) 雙路 payload 格式（沿 DR
        orchestrator.py:1711 pattern）。
        """
        if not state_kg or not (state_kg.entities or state_kg.relationships):
            return None
        return {
            "entities": [e.model_dump() for e in state_kg.entities],
            "relationships": [r.model_dump() for r in state_kg.relationships],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "entity_count": len(state_kg.entities),
                "relationship_count": len(state_kg.relationships),
            },
        }

    # 採納 Decision 2'：只有 internal source 的 snippet 是 articleBody 逐字
    # （spike 2026-06-15 雙錨點 29/32=90% 命中）；web 是 Google snippet 含省略號
    # （必 miss）、wiki/llm_knowledge 無對應站外逐字原文 → 一律不交 quote。
    _TEXTFRAG_OK_SOURCES = frozenset({"internal"})

    @staticmethod
    def _extract_quote(snippet: str) -> str:
        """從 EvidencePoolEntry.snippet 取 verbatim 子句供前端組 text fragment。

        紀律（命中率風險專章 Decision 3，採納 normalize 矛盾修正）：
        - **只 trim 頭尾空白、不動內部空白**（不 collapse、不轉全半形、不去標點）。
          「collapse 連續空白」會讓 fragment 偏離瀏覽器 rendered text → 反而 miss；
          spike 的成功比對是「去所有空白後比」，但錨點不能去空白塞 URL（瀏覽器拿
          錨點去比帶空白的 rendered text）。短錨點（前端 12–16 字）本身已大幅降低
          內部空白差異的命中影響面。
        - 不在此截 START/END 短錨點（那是前端 buildTextFragmentUrl 的職責）。後端
          只負責交出乾淨的 verbatim quote。
        - snippet 空 → 回 ""（前端據此降級裸 URL）。
        """
        if not snippet:
            return ""
        return snippet.strip()  # 只 trim 頭尾，不動內部空白

    @staticmethod
    def _build_citation_sources(state: "LiveResearchStageState") -> Dict[str, Dict[str, str]]:
        """攤平 evidence_pool 為 eid(str) -> {url,title,domain,quote}，供前端 inline
        citation 點擊回溯 + text fragment highlight（O2 / O2-TF）。

        - key 用 str(eid)：跨 SSE/JSON 後前端用 String(eid) 查，避免 int/str 比對陷阱。
        - quote = verbatim snippet 子句（text fragment 來源；空 → 前端降級裸 URL）。
          **Decision 2' 分流**：只有 source ∈ _TEXTFRAG_OK_SOURCES（internal）才交
          quote；web / wiki / llm_knowledge 一律交 quote=""（spike 證 web 含省略號
          必 miss），讓前端降級判據維持單一（quote 空 → 裸 URL），不需感知 source。
        - 帶原始 url（含 urn:llm:knowledge: / private:// 等非 http scheme），由前端
          決定渲染（外部連結 vs 標籤），與後端 references master list 一致。
        - pool 空 → 回 {}（caller emit 時帶空 dict，前端 graceful no-op）。
        - 不可 silent fail：deserialize 失敗讓例外自然浮現（與 _build_references_block 同層）。
        """
        evidence_pool = deserialize_evidence_pool(state.evidence_pool_json)
        if not evidence_pool:
            return {}
        out: Dict[str, Dict[str, str]] = {}
        for eid, entry in evidence_pool.items():
            src = (getattr(entry, "source", "internal") or "internal").strip()
            raw_snippet = getattr(entry, "snippet", "") or ""
            # Decision 2'：非 internal source 不交 quote（避免組必 miss 的 fragment）
            quote = (
                LiveResearchOrchestrator._extract_quote(raw_snippet)
                if src in LiveResearchOrchestrator._TEXTFRAG_OK_SOURCES
                else ""
            )
            out[str(eid)] = {
                "url": (getattr(entry, "url", "") or "").strip(),
                "title": (getattr(entry, "title", "") or "").strip(),
                "domain": (getattr(entry, "source_domain", "") or "").strip(),
                "quote": quote,
            }
        return out

    def _build_references_block(self, state: LiveResearchStageState) -> str:
        """組合 references master list — 列 evidence_pool 全部條目（DR parity B1）。

        設計（Track B1 DR parity sprint 2026-05-28）：
        - 主段「## 參考文獻」：列 sections 真正引用過的 evidence（按首次出現順序）。
        - 附後段「## 研究時搜尋到的相關資料」：列 evidence_pool 中未被任何 section
          引用的剩餘條目（Contains ALL items，對齊 DR _format_result 行為）。
        - Phantom citation（Writer 填了 pool 沒有的 ID）顯示「來源遺失」警示行而
          非靜默跳過，遵守 CLAUDE.md「絕對不可以讓錯誤被無視」。
        - pool 為空 → 回傳空字串，caller 跳過附加 references。
        """
        evidence_pool = deserialize_evidence_pool(state.evidence_pool_json)
        if not evidence_pool:
            return ""

        # 收集所有 section 引用過的 evidence_ids（去重，保持出現順序）
        cited_ids: list = []
        seen: set = set()
        for section in state.written_sections:
            for eid in section.get("sources_used", []):
                if eid not in seen:
                    seen.add(eid)
                    cited_ids.append(eid)

        # citation_style 驅動 references 條目格式（與內文 inline citation 對齊）。
        # user 拍板 author_year（APA）→ APA 條目；否則維持既有數字格式（預設，不破壞既有行為）。
        citation_style = getattr(
            getattr(state, "user_voice", None), "citation_style", None
        )

        def _format_entry(eid: int, entry) -> str:
            """格式化單一 evidence 條目（數字或 APA）。"""
            if citation_style == "author_year":
                return self._format_apa_reference(entry)
            url_part = f" {entry.url}" if entry.url else ""
            domain_part = f" — {entry.source_domain}" if entry.source_domain else ""
            return f"[{eid}] {entry.title}{domain_part}{url_part}"

        lines: list = []

        # 主段：被引用的條目
        if cited_ids:
            lines += ["", "---", "", "## 參考文獻", ""]
            for eid in cited_ids:
                entry = evidence_pool.get(eid)
                if entry is None:
                    # Phantom citation — Writer 填了不在 pool 內的 ID
                    # 不 silent skip：標記出來方便 debug（CLAUDE.md no silent fail）
                    lines.append(lr_copy.reference_missing_entry(eid))
                else:
                    lines.append(_format_entry(eid, entry))

        # 附後段：pool 中未被引用的剩餘條目（B1 DR parity：Contains ALL items）
        uncited_ids = [eid for eid in sorted(evidence_pool.keys()) if eid not in seen]
        if uncited_ids:
            lines += ["", "---", "", "## 研究時搜尋到的相關資料", ""]
            for eid in uncited_ids:
                entry = evidence_pool[eid]
                lines.append(_format_entry(eid, entry))

        return "\n".join(lines) if lines else ""

    @staticmethod
    def _format_apa_reference(entry) -> str:
        """組一條 APA 風格的 references 條目：`作者. (年份). 標題. 網域. URL`。

        Graceful degradation（CLAUDE.md no silent fail — 不可輸出 `, ).` 滿天）：
        真實 BAB path 的 evidence author/year 常為空（fixture 是 backfill 的）。
        - author 空 → 用 source_domain 作為來源機構名（references 條目用機構名是
          APA 對團體作者的合理慣例，與內文 inline「(來源不明, n.d.)」不同層級：
          references 是給 reader 查證用，列出能找到的最具體資訊優於標「來源不明」）。
        - year 空 → 標 `(n.d.)`（APA 對無日期來源的標準寫法）。
        - title 空 → 用 source_domain 或「未知標題」。
        """
        author = (getattr(entry, "author", "") or "").strip()
        year = (getattr(entry, "year", "") or "").strip()
        # FIX-3 (Cayenne #10): year 缺時從 published_at 取年份，與內文 inline
        # citation 一致（real-retrieval evidence 只填 published_at）。
        if not year and getattr(entry, "published_at", None):
            year = (getattr(entry, "published_at", "") or "")[:4].strip()
        title = (getattr(entry, "title", "") or "").strip()
        domain = (getattr(entry, "source_domain", "") or "").strip()
        url = (getattr(entry, "url", "") or "").strip()

        author_part = author or domain or "佚名"
        year_part = year or "n.d."
        title_part = title or domain or "未知標題"

        parts = [f"{author_part}. ({year_part}). {title_part}."]
        if domain and domain != author_part:
            parts.append(f" {domain}.")
        if url:
            parts.append(f" {url}")
        return "".join(parts)

    # ──── 工具方法 ─────────────────────────────────────────────

    def _format_initial_items(self, items: list) -> Optional[str]:
        """格式化初始 retrieval items 為 context string。"""
        if not items:
            return None
        lines = []
        for i, item in enumerate(items[:10], 1):
            # Normalize: retrieval returns list/tuple, not dict
            if isinstance(item, (list, tuple)):
                item = BABLoopEngine._normalize_item(item)
            title = item.get('name', '')
            desc = item.get('description', '')[:300]
            lines.append(f"[{i}] {title}\n{desc}\n")
        return "\n".join(lines) if lines else None

    def _load_mock_bab_fixture(self) -> ContextMap:
        """載入真實 LLM 產出的 ContextMap fixture（mock_bab 模式用）。

        載入來源：tests/fixtures/{_MOCK_BAB_FIXTURE_DIRNAME}/context_map.json
        （現行：Cayenne 綠能命題 prod session 8e1db658，567 筆真語料 + 20 topics v25）。
        """
        import json
        from pathlib import Path

        # 從 repo root 解析 fixture 路徑（相對 orchestrator.py 往上 4 層到 repo root）
        repo_root = Path(__file__).parents[4]
        fixture_path = repo_root / "code" / "python" / "tests" / "fixtures" / _MOCK_BAB_FIXTURE_DIRNAME / "context_map.json"
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"[LIVE RESEARCH] mock_bab context_map fixture not found: {fixture_path}. "
                f"Expected fixture dir: tests/fixtures/{_MOCK_BAB_FIXTURE_DIRNAME}/（撈法見 docs/specs/mock-bab-playbook.md）"
            )

        with open(fixture_path, "r", encoding="utf-8") as f:
            context_map_data = json.loads(f.read().strip())

        context_map = ContextMap.model_validate(context_map_data)
        logger.info(
            f"[LIVE RESEARCH] mock_bab: loaded real fixture ContextMap "
            f"({len(context_map.topics)} topics, {len(context_map.relations)} relations, v{context_map.version}) "
            f"from {fixture_path.name}"
        )
        return context_map

    def _load_mock_evidence_pool_fixture(self) -> str:
        """載入 mock_bab fixture 的 evidence_pool，回傳 serialized JSON 字串。

        載入來源：tests/fixtures/{_MOCK_BAB_FIXTURE_DIRNAME}/evidence_pool.json
        （現行：Cayenne prod session 8e1db658，567 筆真實 evidence）。

        fixture 內 evidence_pool 是 dict[str(id), entry_dict]，本方法直接序列化成
        state.evidence_pool_json 所需的字串格式（與 serialize_evidence_pool 相容）。

        若 fixture 檔不存在或為空：log warning + 回傳空字串（不 crash），
        Stage 6 references block 會跳過（符合 CLAUDE.md no silent fail —— 有 log）。
        """
        import json
        from pathlib import Path

        repo_root = Path(__file__).parents[4]
        fixture_path = repo_root / "code" / "python" / "tests" / "fixtures" / _MOCK_BAB_FIXTURE_DIRNAME / "evidence_pool.json"
        if not fixture_path.exists():
            logger.warning(
                f"[LIVE RESEARCH] mock_bab evidence_pool fixture not found: {fixture_path} — "
                "Stage 6 references block will be empty"
            )
            return ""

        with open(fixture_path, "r", encoding="utf-8") as f:
            raw_pool = json.loads(f.read().strip())

        if not raw_pool:
            logger.warning(
                "[LIVE RESEARCH] mock_bab evidence_pool fixture is empty — "
                "Stage 6 references block will be empty"
            )
            return ""

        # raw_pool keys 已是 str（JSON 規範），entries 已是 EvidencePoolEntry-shaped dict
        pool_json = json.dumps(raw_pool, ensure_ascii=False)
        logger.info(
            f"[LIVE RESEARCH] mock_bab: loaded real evidence_pool fixture: "
            f"{len(raw_pool)} entries from {fixture_path.name}"
        )
        return pool_json

    def _load_mock_evidence_usage_fixture(self) -> "Dict[int, List[Dict]]":
        """載入 mock_bab fixture 的 evidence_usage，回傳 Dict[int, List[Dict]]。

        載入來源：tests/fixtures/{_MOCK_BAB_FIXTURE_DIRNAME}/evidence_usage.json
        （現行：Cayenne prod session 8e1db658，40 evidence ids / 172 grounded claims）。

        JSON key 一律是 str，載入時轉回 int（state.evidence_usage 型別契約 Dict[int, ...]）。
        value 保持 List[Dict]（不還原成 GroundedClaim model），與 loop_engine
        gc.model_dump() 寫入 pattern 以及 render_grounding_evidence_view dict 消費對齊。

        若 fixture 檔不存在或 JSON parse 失敗：明確 raise FileNotFoundError / ValueError，
        不可 silent fail（CLAUDE.md 紀律）。
        """
        import json
        from pathlib import Path
        from typing import Dict, List

        repo_root = Path(__file__).parents[4]
        fixture_path = (
            repo_root / "code" / "python" / "tests" / "fixtures"
            / _MOCK_BAB_FIXTURE_DIRNAME / "evidence_usage.json"
        )
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"[LIVE RESEARCH] mock_bab evidence_usage fixture not found: {fixture_path}. "
                f"Expected fixture dir: tests/fixtures/{_MOCK_BAB_FIXTURE_DIRNAME}/"
                "（撈法見 docs/specs/mock-bab-playbook.md）"
            )

        with open(fixture_path, "r", encoding="utf-8") as f:
            raw_content = f.read().strip()

        if not raw_content:
            raise ValueError(
                f"[LIVE RESEARCH] mock_bab evidence_usage fixture is empty: {fixture_path}. "
                "Cannot proceed — chapter-override writer requires evidence_usage."
            )

        try:
            raw_usage = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[LIVE RESEARCH] mock_bab evidence_usage fixture JSON parse failed: "
                f"{fixture_path} — {e}"
            ) from e

        # JSON key 一律是 str，轉回 int（state.evidence_usage 型別契約 Dict[int, ...]）。
        # 非 int key 或非 list value → 明確 log error + skip（不吞掉，仍可繼續）。
        evidence_usage: Dict[int, List[Dict]] = {}
        for k, v in raw_usage.items():
            try:
                eid = int(k)
            except (TypeError, ValueError):
                logger.error(
                    "[LIVE RESEARCH] mock_bab evidence_usage: non-int key %r skipped",
                    k,
                )
                continue
            if not isinstance(v, list):
                logger.error(
                    "[LIVE RESEARCH] mock_bab evidence_usage[%d]: expected list, got %s — skipped",
                    eid, type(v).__name__,
                )
                continue
            evidence_usage[eid] = list(v)

        logger.info(
            f"[LIVE RESEARCH] mock_bab: loaded real evidence_usage fixture: "
            f"{len(evidence_usage)} evidence ids, "
            f"{sum(len(c) for c in evidence_usage.values())} claims "
            f"from {fixture_path.name}"
        )
        return evidence_usage

    def _load_mock_book_outline_fixture(self) -> "BookOutline":
        """載入 mock_bab fixture 的 BookOutline，回傳 BookOutline model。

        載入來源：tests/fixtures/{_MOCK_BAB_FIXTURE_DIRNAME}/book_outline.json
        （現行：Cayenne prod session 8e1db658，3 章：前言/國際案例分析/結論）。

        fixture JSON 結構與 BookOutline schema 對齊（chapter_index / title / brief /
        target_word_count / planned_evidence_ids / transition_hint / role）。
        model_validate 失敗（fixture 缺/壞）→ 明確 raise，不可 silent fail
        （CLAUDE.md 紀律，照 _load_mock_evidence_usage_fixture 的錯誤處理 pattern）。
        """
        import json
        from pathlib import Path

        repo_root = Path(__file__).parents[4]
        fixture_path = (
            repo_root / "code" / "python" / "tests" / "fixtures"
            / _MOCK_BAB_FIXTURE_DIRNAME / "book_outline.json"
        )
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"[LIVE RESEARCH] mock_bab book_outline fixture not found: {fixture_path}. "
                f"Expected fixture dir: tests/fixtures/{_MOCK_BAB_FIXTURE_DIRNAME}/"
                "（撈法見 docs/specs/mock-bab-playbook.md）"
            )

        with open(fixture_path, "r", encoding="utf-8") as f:
            raw_content = f.read().strip()

        if not raw_content:
            raise ValueError(
                f"[LIVE RESEARCH] mock_bab book_outline fixture is empty: {fixture_path}. "
                "Cannot proceed — Stage 5 outline planner requires book_outline."
            )

        try:
            raw_data = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[LIVE RESEARCH] mock_bab book_outline fixture JSON parse failed: "
                f"{fixture_path} — {e}"
            ) from e

        try:
            book_outline = BookOutline.model_validate(raw_data)
        except Exception as e:
            raise ValueError(
                f"[LIVE RESEARCH] mock_bab book_outline fixture failed BookOutline validation: "
                f"{fixture_path} — {e}"
            ) from e

        logger.info(
            f"[LIVE RESEARCH] mock_bab: loaded real book_outline fixture: "
            f"{len(book_outline.chapters)} chapters "
            f"from {fixture_path.name}"
        )
        return book_outline

    def _context_map_to_outline(self, context_map: ContextMap) -> str:
        """將 ContextMap 轉為使用者可讀的 outline。"""
        lines = []
        lines.append(f"**研究問題**：{context_map.research_question}")
        if context_map.working_hypothesis:
            lines.append(f"**工作假設**：{context_map.working_hypothesis}")
        lines.append("")

        for i, topic in enumerate(context_map.topics, 1):
            relevance_label = {"core": "核心", "supporting": "輔助", "peripheral": "周邊"}.get(
                topic.relevance, ""
            )
            lines.append(f"{i}. **{topic.name}**（{relevance_label}）— {topic.description}")

        if context_map.followup_questions:
            lines.append("\n**待探索的問題**：")
            for q in context_map.followup_questions:
                lines.append(f"- {q}")

        return "\n".join(lines)

    async def _emit_narration(self, text: str):
        if not text:
            return
        await emit_sse(self.handler, {
            "message_type": "live_research_narration",
            "text": text,
        })

    async def _emit_stage_change(self, stage: int):
        await emit_sse(self.handler, {
            "message_type": "live_research_stage_change",
            "stage": stage,
        })

    def _build_topic_evidence_list(
        self,
        topic: "ContextMapTopic",
        evidence_pool: "Dict[int, EvidencePoolEntry]",
    ) -> list:
        """依 topic.evidence_ids 從 evidence_pool 取 metadata，回傳 frontend-ready dict list。

        - 跳過 pool 沒有的 phantom ID（不插 None）。
        - 回傳欄位：id / title / url / source_domain / published_at / source。
        - 不排序；保留 topic.evidence_ids 原始順序。
        """
        result = []
        for eid in topic.evidence_ids:
            entry = evidence_pool.get(eid)
            if entry is None:
                logger.debug(
                    f"[LIVE RESEARCH] evidence_id {eid} in topic '{topic.name}' "
                    f"not found in pool — skipped"
                )
                continue
            result.append({
                "id": entry.evidence_id,
                "title": entry.title,
                "url": entry.url,
                "source_domain": entry.source_domain,
                "published_at": entry.published_at,  # str YYYY-MM-DD or None
                "source": entry.source,
            })
        return result

    def _build_drift_pause_proposal(self, review) -> str:
        """把 Consistency Monitor 的 drift banner + 具體漂移描述組成 checkpoint proposal。

        review: ConsistencyReview（engine.consistency_review）。drift_description 可能為空
        （LLM 沒填）→ 只出 banner 主體，不拼空描述（不對 user 顯示「（無描述）」這種噪音）。
        """
        from reasoning.live_research import lr_copy
        banner = lr_copy.DRIFT_PAUSE_BANNER
        desc = (getattr(review, "drift_description", "") or "").strip()
        if desc:
            return f"{banner}\n\n目前觀察到的偏移：{desc}"
        return banner

    async def _emit_checkpoint(
        self,
        stage: int,
        proposal: str,
        context_map_summary: str = "",
        evidence_list: Optional[List[dict]] = None,
        show_new_sample_button: bool = False,
        evidence_total: Optional[int] = None,
    ):
        # evidence_total：evidence_pool 完整筆數（與 Stage 2 consolidation narration
        # 「共蒐集到 N 筆」同源 len(pool)）。前端「N 筆資料」標題只拿得到 evidence_list
        # 子集長度，會讓 user 誤以為只蒐到那幾筆；帶上總量讓前端標示「節選 vs 總量」。
        # 不傳時 fallback 為 evidence_list 長度（向後兼容：舊 caller / Stage 3 等無 pool 的 checkpoint）。
        _evidence_list = evidence_list or []
        _evidence_total = evidence_total if evidence_total is not None else len(_evidence_list)
        await emit_sse(self.handler, {
            "message_type": "live_research_checkpoint",
            "stage": stage,
            "proposal": proposal,
            "context_map_summary": context_map_summary,
            "auto_continue_option": True,
            "evidence_list": _evidence_list,
            "evidence_total": _evidence_total,
            # Stage 3 風格 checkpoint 才設 True：前端據此顯示「重新提供範本」按鈕。
            "show_new_sample_button": show_new_sample_button,
        })

    async def _emit_clarification(self, req, state):
        """通用澄清 dispatcher（設計文件 §3）：emit 問句 narration + re-emit 該 stage
        的 checkpoint（恢復前端 reply UI）+ return state（不 advance）。

        收斂三條同型 spine。前端 continueLiveResearch 已把 _lrAwaitingCheckpointReply
        設 false、隱藏 reply UI，故必須 re-emit checkpoint 恢復（不能只 narration）。
        """
        await self._emit_narration(req.question)
        await self._emit_checkpoint(stage=req.stage, proposal=state.checkpoint_prompt)
        return state

    async def _emit_section(self, index: int, section: LiveWriterSectionOutput,
                            state: "LiveResearchStageState"):
        await emit_sse(self.handler, {
            "message_type": "live_research_section",
            "section_index": index,
            "title": section.section_title,
            "content": section.section_content,
            "sources": section.sources_used,
            # O2 / O2-TF: eid -> {url,title,domain,quote}，供前端 inline citation
            # 點擊回溯 + text fragment highlight
            "citation_sources": self._build_citation_sources(state),
            # #4 fix (2026-05-29): L3 WARN marker 存在 methodology_note，
            # 即時 SSE 也要帶（不只 _section_dict 持久化），否則 live 渲染收不到
            "methodology_note": getattr(section, "methodology_note", "") or "",
        })


# ============================================================================
# Stage 1 Dialog Loop: ContextMap mutation engine (pure functions, module-level)
# ============================================================================


def _op_merge_topics(cm, op, delta, warnings):
    """合併多個 topic 成一個新 topic；relations 涉及 source 的 endpoint 重 map 到新 topic。"""
    sources = [t for t in cm.topics if t.topic_id in op.source_topic_ids]
    if len(sources) < 2:
        warnings.append(f"merge_topics: 來源不足（找到 {len(sources)} 個）")
        return
    merged_name = op.merged_name or " + ".join(s.name for s in sources)
    merged_evidence = sorted(set(eid for s in sources for eid in s.evidence_ids))
    merged_desc = "\n".join(s.description for s in sources if s.description)
    # 最高優先度 relevance 保留：core > supporting > peripheral
    relevances = [s.relevance for s in sources]
    new_relevance = "core" if "core" in relevances else (
        "supporting" if "supporting" in relevances else "peripheral"
    )
    new_topic = ContextMapTopic(
        name=merged_name,
        domain=sources[0].domain,
        description=merged_desc,
        relevance=new_relevance,
        evidence_ids=merged_evidence,
    )
    source_ids = {s.topic_id for s in sources}
    cm.topics = [t for t in cm.topics if t.topic_id not in source_ids]
    cm.topics.append(new_topic)
    # Relations: 把 source endpoint 重 map 到新 topic_id；source-to-source 變 self-loop 則移除
    new_relations = []
    for r in cm.relations:
        src = new_topic.topic_id if r.source_topic_id in source_ids else r.source_topic_id
        tgt = new_topic.topic_id if r.target_topic_id in source_ids else r.target_topic_id
        if src == tgt:
            continue  # self-loop 移除
        r.source_topic_id = src
        r.target_topic_id = tgt
        new_relations.append(r)
    cm.relations = new_relations


def _op_split_topic(cm, op, delta, warnings):
    """把一個 topic 拆成多個新 topic；涉及原 topic 的 relations 全砍（v1 簡化）。"""
    src = next((t for t in cm.topics if t.topic_id == op.split_from_topic_id), None)
    if src is None:
        warnings.append(f"split_topic: 找不到 {op.split_from_topic_id}")
        return
    if not op.split_into:
        warnings.append("split_topic: split_into 為空")
        return
    new_topics = []
    for spec in op.split_into:
        new_topics.append(ContextMapTopic(
            name=spec.get("name", ""),
            domain=src.domain,
            description=spec.get("description", ""),
            relevance=src.relevance,
            evidence_ids=spec.get("evidence_ids", []),
        ))
    # 落單 evidence_ids（src 有但 split_into 沒分到的）放第一個新 topic
    used = set(eid for spec in op.split_into for eid in spec.get("evidence_ids", []))
    leftover = [eid for eid in src.evidence_ids if eid not in used]
    if leftover and new_topics:
        new_topics[0].evidence_ids.extend(leftover)
    cm.topics = [t for t in cm.topics if t.topic_id != src.topic_id]
    cm.topics.extend(new_topics)
    # 移除涉及 src 的 relations（split 後 relation 無法 auto-map）
    pre_count = len(cm.relations)
    cm.relations = [
        r for r in cm.relations
        if r.source_topic_id != src.topic_id and r.target_topic_id != src.topic_id
    ]
    if len(cm.relations) < pre_count:
        # 改動 1：backend log only，不 emit narration 給 user
        logger.warning(
            f"[LIVE RESEARCH] split_topic: {pre_count - len(cm.relations)} 條關係已失效"
            f"（split 後無法保留），split_from_topic_id={src.topic_id}"
        )


def _op_add_topic(cm, op, delta, warnings):
    """新增一個 topic。domain 繼承同一 context_map 裡既有 topic 的 domain
    （同一研究 session 的 topic 同屬一個領域，對齊 _op_split_topic 的 domain=src.domain 慣例）；
    跳過殘留的「(待補)」/空值，無可繼承時退到誠實的「未分類」（禁止佔位符外洩給 user）。"""
    if not op.new_topic_name:
        warnings.append("add_topic: new_topic_name 為空")
        return
    inherited_domain = next(
        (t.domain for t in cm.topics if t.domain and t.domain != "(待補)"),
        "未分類",
    )
    cm.topics.append(ContextMapTopic(
        name=op.new_topic_name,
        domain=inherited_domain,
        description=op.new_topic_description,
        relevance=op.new_topic_relevance,
        evidence_ids=op.new_topic_evidence_ids,
    ))


def _op_remove_topic(cm, op, delta, warnings):
    """刪除一個 topic 並移除涉及它的 relations。"""
    target = next((t for t in cm.topics if t.topic_id == op.target_topic_id), None)
    if target is None:
        warnings.append(f"remove_topic: 找不到 {op.target_topic_id}")
        return
    cm.topics = [t for t in cm.topics if t.topic_id != op.target_topic_id]
    cm.relations = [
        r for r in cm.relations
        if r.source_topic_id != op.target_topic_id and r.target_topic_id != op.target_topic_id
    ]


def _op_rename_topic(cm, op, delta, warnings):
    """修改 topic.name；記錄 modified_topics。"""
    target = next((t for t in cm.topics if t.topic_id == op.target_topic_id), None)
    if target is None:
        warnings.append(f"rename_topic: 找不到 {op.target_topic_id}")
        return
    target.name = op.new_name or target.name
    if op.target_topic_id not in delta.modified_topics:
        delta.modified_topics.append(op.target_topic_id)


def _op_change_relevance(cm, op, delta, warnings):
    """修改 topic.relevance；記錄 modified_topics。"""
    target = next((t for t in cm.topics if t.topic_id == op.target_topic_id), None)
    if target is None:
        warnings.append(f"change_relevance: 找不到 {op.target_topic_id}")
        return
    target.relevance = op.new_relevance
    if op.target_topic_id not in delta.modified_topics:
        delta.modified_topics.append(op.target_topic_id)


def _op_change_description(cm, op, delta, warnings):
    """修改 topic.description；記錄 modified_topics。"""
    target = next((t for t in cm.topics if t.topic_id == op.target_topic_id), None)
    if target is None:
        warnings.append(f"change_description: 找不到 {op.target_topic_id}")
        return
    target.description = op.new_description or target.description
    if op.target_topic_id not in delta.modified_topics:
        delta.modified_topics.append(op.target_topic_id)


# UX-9: D-3 relevance heuristic — chapter name 含關鍵字 → 推斷 relevance
# default "core"（Stage 2 BAB 只跑 core，全 core 確保 user 列的章節都會被寫到）
_REFRAME_RELEVANCE_SUPPORTING_KEYWORDS = (
    "背景", "文獻", "延伸", "附錄", "回顧", "歷史",
)


def _infer_chapter_relevance(name: str, explicit: str = "") -> str:
    """D-3 heuristic：依 chapter name 推斷 relevance。

    explicit 非空且為合法值時直接採用（user / LLM 明確指定優先）；
    否則用 keyword match，default "core"。
    """
    if explicit in ("core", "supporting", "peripheral"):
        return explicit
    for kw in _REFRAME_RELEVANCE_SUPPORTING_KEYWORDS:
        if kw in name:
            return "supporting"
    return "core"


def _op_reframe_structure(cm, op, delta, warnings):
    """UX-9: 整體重組 — Replace All semantics（D-1）。

    步驟（plan Task 2.1）：
    1. Validate op.new_chapters 非空
    2. 收集 all_evidence_ids = sorted(set(all topic.evidence_ids))
    3. 清空 cm.topics / cm.relations
    4. 對每個 new_chapter 建 ContextMapTopic
       - domain 填「（reframe）」標記由 reframe 產生（後續 Stage 2 可再 enrich）
       - relevance 用 _infer_chapter_relevance（D-3 heuristic）
    5. 第一個 chapter 接收所有 leftover evidence_ids（D-2：evidence pool intact，
       evidence_ids 重塞至前言章）
    6. op.new_research_question 非空 → 覆寫 cm.research_question
    7. delta.modified_topics 不填（reframe 本質是「全砍重建」，
       added/removed 由 _apply_context_map_revisions outer 算）

    Evidence preservation（D-2）：
    - state.evidence_pool_json 在 caller (orchestrator) 不動，pool 完整保留
    - 新 topics 的 evidence_ids 全塞至第一個 chapter，避免 leak
    - Writer 依然能透過 evidence_lookup 看到所有 [N] 對應
    """
    if not op.new_chapters:
        warnings.append("reframe_structure: new_chapters 為空")
        return

    # 收集舊 topics 的所有 evidence_ids（dedup + sorted，可重複套用穩定）
    all_evidence_ids = sorted(set(
        eid for t in cm.topics for eid in t.evidence_ids
    ))

    # Clear topics + relations（Replace All）
    cm.topics = []
    cm.relations = []

    # 建立新 chapters
    new_topics_built = []
    for i, spec in enumerate(op.new_chapters):
        if not isinstance(spec, dict):
            warnings.append(f"reframe_structure: new_chapters[{i}] 不是 dict，已略過")
            continue
        name = spec.get("name", "").strip()
        if not name:
            warnings.append(f"reframe_structure: new_chapters[{i}].name 為空，已略過")
            continue
        description = spec.get("description", "")
        explicit_relevance = spec.get("relevance", "")
        relevance = _infer_chapter_relevance(name, explicit_relevance)

        new_topic = ContextMapTopic(
            name=name,
            domain="（reframe）",
            description=description,
            relevance=relevance,
            evidence_ids=[],
        )
        new_topics_built.append(new_topic)

    if not new_topics_built:
        # 全部 spec 都被略過 → 視同 rejection（caller empty guard 也會擋）
        warnings.append("reframe_structure: 所有 new_chapters 都無效")
        return

    # 第一個 chapter 接收 leftover evidence_ids（D-2）
    if all_evidence_ids and new_topics_built:
        new_topics_built[0].evidence_ids = all_evidence_ids

    cm.topics = new_topics_built

    # 覆寫 research_question（optional, D-4）
    new_rq = (op.new_research_question or "").strip()
    if new_rq:
        cm.research_question = new_rq


_REVISION_HANDLERS = {
    "merge_topics": _op_merge_topics,
    "split_topic": _op_split_topic,
    "add_topic": _op_add_topic,
    "remove_topic": _op_remove_topic,
    "rename_topic": _op_rename_topic,
    "change_relevance": _op_change_relevance,
    "change_description": _op_change_description,
    "reframe_structure": _op_reframe_structure,
}


def _apply_context_map_revisions(context_map, operations, parse_summary):
    """Apply LLM-generated mutation operations to a ContextMap (pure function).

    改動 3：transactional safety — 在 deep copy `cm_working` 上 mutate，
    中途 exception 或 empty guard 整體 abort，return (None, None, warnings)，
    caller 看 None 走 narration + 保持 checkpoint path。

    UX-9 evidence preservation 紀律（D-2）：
    - evidence_pool（references master list）存在 state.evidence_pool_json，
      不在 ContextMap 內。此函式只操作 ContextMap，**不會動 evidence_pool**。
    - reframe_structure 把 cm.topics 全砍重建後，所有 evidence_ids 重塞至
      第一個新章節（前言）；evidence_pool 完整保留，後續 Writer 透過
      evidence_lookup 仍能看到所有 [N] 對應。
    - integration test 應驗 reframe 前後 state.evidence_pool_json byte-equal。

    Returns:
        (mutated_context_map, delta, warnings) on success;
        (None, None, warnings) if mutation would leave ContextMap empty
        or any op handler raised exception.
    """
    cm_working = context_map.model_copy(deep=True)

    pre_topic_ids = {t.topic_id for t in cm_working.topics}
    pre_relation_ids = {r.relation_id for r in cm_working.relations}

    delta = ContextMapDelta(
        from_version=cm_working.version,
        to_version=cm_working.version + 1,
        reason=parse_summary or "使用者建議調整",
    )
    warnings: list = []

    try:
        for op in operations:
            handler = _REVISION_HANDLERS.get(op.op_type)
            if handler is None:
                warnings.append(f"未知 op_type: {op.op_type}")
                continue
            handler(cm_working, op, delta, warnings)
    except Exception as e:
        # 改動 3：op handler 拋 exception → 整體 abort，不留下 half-mutated state
        warnings.append(f"mutation 中途失敗：{e}")
        logger.warning(f"[LIVE RESEARCH] _apply_context_map_revisions abort: {e}")
        return None, None, warnings

    if len(cm_working.topics) == 0:
        warnings.append("拒絕：至少要保留一個研究主題")
        return None, None, warnings

    # Bug 4a (2026-05-18) root-fix：chapter order 是 semantic info，**禁用**
    # `set(post) - set(pre)` 算 added_topics — Python set iteration 是 hash-based、
    # 跟插入順序無關，導致 reframe narration「整體重組為 N 章：A/B/C/...」順序亂跳。
    # 改用 `cm_working.topics` 自身的順序（reframe handler 已按 `op.new_chapters`
    # 順序 append），這順序就是 user 拍板的章節順序。
    post_topic_ids_ordered = [t.topic_id for t in cm_working.topics]
    post_topic_ids = set(post_topic_ids_ordered)
    delta.added_topics = [
        tid for tid in post_topic_ids_ordered if tid not in pre_topic_ids
    ]
    delta.removed_topics = list(pre_topic_ids - post_topic_ids)
    post_relation_ids = {r.relation_id for r in cm_working.relations}
    delta.removed_relations = list(pre_relation_ids - post_relation_ids)

    cm_working.version += 1
    cm_working.revision_history.append(delta)
    cm_working.last_refined_at = datetime.now().isoformat()
    return cm_working, delta, warnings
