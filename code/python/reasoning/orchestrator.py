"""
Deep Research Orchestrator - Coordinates the Actor-Critic reasoning loop.
"""

import asyncio
import json
import re
import sentry_sdk
import time
import types
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import quote
from misc.logger.logging_config_helper import get_configured_logger
from core.retriever import search as retriever_search
from core.config import CONFIG
from core.temporal_anchor import annotate_relative_years, anchor_note, TEMPORAL_ANCHOR_RULE
from reasoning.agents.analyst import AnalystAgent
from reasoning.agents.critic import CriticAgent
from reasoning.agents.writer import WriterAgent
from reasoning.filters.source_tier import SourceTierFilter, NoValidSourcesError
from reasoning.schemas import WriterComposeOutput
from reasoning.research_state import ResearchState
from reasoning.orchestrator_base import OrchestratorBase, ResearchCancelledError, ProgressConfig  # noqa: F401
from reasoning.relevance_gate_core import RELEVANCE_GATE_DIGEST_CHAR_BUDGET


logger = get_configured_logger("reasoning.orchestrator")


def _normalize_web_query(q: str) -> str:
    """dedup key normalize：strip + 內部 whitespace collapse + casefold。集中一處避免各點不一致。"""
    return re.sub(r"\s+", " ", (q or "").strip()).casefold()


# === KG Editing: ResearchState cache for selective re-run ===
_research_state_cache: Dict[str, dict] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour
_MAX_CACHE_SIZE = 50


def _cache_research_state(query_id: str, state: 'ResearchState'):
    """Cache data fields from ResearchState after phase 1 for potential selective re-run.

    Only caches serializable data fields. Session-specific objects
    (iteration_logger, tracer) are NOT cached — they are recreated per session.
    """
    _research_state_cache[query_id] = {
        'formatted_context': state.formatted_context,
        'source_map': state.source_map,
        'current_context': state.current_context,
        'query': state.query,
        'mode': state.mode,
        'temporal_context': state.temporal_context,
        'enable_kg': state.enable_kg,
        'enable_web_search': state.enable_web_search,
        'items': state.items,
        'cached_at': time.time(),
    }
    # Cleanup expired entries
    now = time.time()
    expired = [k for k, v in _research_state_cache.items() if now - v['cached_at'] > _CACHE_TTL_SECONDS]
    for k in expired:
        del _research_state_cache[k]
    # Evict oldest entries if cache exceeds max size
    if len(_research_state_cache) >= _MAX_CACHE_SIZE:
        sorted_keys = sorted(_research_state_cache.keys(), key=lambda k: _research_state_cache[k]['cached_at'])
        evict_count = len(_research_state_cache) - _MAX_CACHE_SIZE + 1  # bring below limit
        for k in sorted_keys[:evict_count]:
            del _research_state_cache[k]
        logger.info(f"[RERUN CACHE] Evicted {evict_count} oldest entries (cache size={len(_research_state_cache)})")
    logger.info(f"[RERUN CACHE] Cached state for query_id={query_id} (cache size={len(_research_state_cache)})")


def get_cached_research_state(query_id: str) -> Optional[dict]:
    """Retrieve cached ResearchState data for selective re-run.

    Returns None if not found or expired.
    """
    cached = _research_state_cache.get(query_id)
    if cached is None:
        return None
    if time.time() - cached['cached_at'] > _CACHE_TTL_SECONDS:
        del _research_state_cache[query_id]
        return None
    return cached


# === Bug 1: rerunState 持久化——抽精簡子集 + 從 research_report 重建（純函式，write/read path 共用） ===

# [R5 剝欄位 — CEO 拍板選項一] rerun path 對 source_map item 讀取的欄位窮舉集合（親讀 orchestrator.py
# _format_context_shared + citation 解析下游驗）：url/link、title/name、description/articleBody、site、
# datePublished——**只有這 5 類**。schema_json 全文其餘、vector（list-row item[4]）、scores（item[5]）
# 從不被 rerun 讀。Task 5 Step 1 實測：temporal 87 筆 source_map 達 238KB（超 200KB JSONB warn），
# 大頭正是每 item ~2.6KB 的 schema_json 全文 blob → 剝欄位後 payload 預估 ~15-30KB。
# 若未來 rerun 下游讀了這 5 類以外欄位 → 剝欄位會使該欄失真 → 必須重審此窮舉集合。
_RERUN_ITEM_FIELDS = ("url", "title", "description", "site", "datePublished")


def _slim_item(item) -> dict:
    """把單個檢索 item 剝成 rerun 真正用到的精簡 dict（只留 5 欄），丟掉肥大的 schema_json 全文 + vector + scores。

    [verified] 真實 retriever 回的 item 是 **6-element list-row**
    `[url, schema_json_str, title, source, vector, scores_dict]`（Task 5 Step 1 實測確認）——
    **description/datePublished 藏在 item[1] 的 schema_json 字串裡**（不是頂層），故不能直接丟 schema_json，
    要先 json.loads(item[1]) 抽出。統一剝成 dict，讓 restore 後下游 `_format_context_shared` dict 分支
    正確讀取（不再走 list-row 分支去找已不存在的 item[1]）。dict 格式 item 直接從頂層抽
    （`description or articleBody` / `title or name` / `url or link`，對齊 `_format_context_shared`
    dict 分支的 fallback 語義）。
    """
    if isinstance(item, dict):
        return {
            'url': item.get('url') or item.get('link', ''),
            'title': item.get('title') or item.get('name', ''),
            'description': item.get('description') or item.get('articleBody', ''),
            'site': item.get('site', ''),
            'datePublished': item.get('datePublished', ''),
        }
    if isinstance(item, (list, tuple)):
        try:
            schema = json.loads(item[1]) if len(item) > 1 and isinstance(item[1], str) else (item[1] if len(item) > 1 else {})
        except (json.JSONDecodeError, TypeError) as e:
            # [AR R1 should-fix] 降級必留訊息（不可 silent fail）：description/datePublished 會變空
            logger.warning(f"[RERUN SLIM] item schema_json 解析失敗（url={item[0] if len(item) > 0 else '?'!r}）"
                           f"→ description/datePublished 降空: {e}")
            schema = {}
        _schema_ok = isinstance(schema, dict)
        if not _schema_ok and schema:
            logger.warning(f"[RERUN SLIM] item schema_json 非 dict（type={type(schema).__name__}, "
                           f"url={item[0] if len(item) > 0 else '?'!r}）→ description/datePublished 降空")
        return {
            'url': item[0] if len(item) > 0 else '',
            'title': item[2] if len(item) > 2 else '',
            'description': (schema.get('description') or schema.get('articleBody', '')) if _schema_ok else '',
            'site': item[3] if len(item) > 3 else '',
            'datePublished': schema.get('datePublished', '') if _schema_ok else '',
        }
    # 非 dict/list（不預期）→ 回空 5 欄殼（不 crash，下游 dict 分支讀到空值優雅降級）
    # [AR R1 should-fix] 降級必留訊息（不可 silent fail）
    logger.warning(f"[RERUN SLIM] 未知 item 型別 {type(item).__name__}（非 dict/list）→ 剝成空 5 欄殼")
    return {'url': '', 'title': '', 'description': '', 'site': '', 'datePublished': ''}


def build_rerun_state_subset(state: 'ResearchState') -> dict:
    """從 phase 1 完成的 ResearchState 抽出 rerun 真正需要的精簡子集，供併進 research_report JSONB。

    [verified] phase 2-4 對 items 的唯一使用是 len(state.items)（sources_filtered stat，
    orchestrator.py:1554）——故**不存全量 items**，只留 items_count。source_map/current_context
    才是結構性需要（Analyst/Critic 輸入、citation 解析、web search extend）。

    [R5 剝欄位 — CEO 拍板選項一] source_map 的 value 用 `_slim_item` 剝成精簡 5 欄 dict（見 _slim_item
    docstring + `_RERUN_ITEM_FIELDS`）：丟掉每 item ~2.6KB 的 schema_json 全文 blob + vector + scores，
    只留 rerun 真正讀的 url/title/description/site/datePublished。Task 5 Step 1 實測 temporal 87 筆
    source_map 達 238KB（超 200KB JSONB warn）→ 剝後預估 ~15-30KB。這比 plan 原選項 A/B（都存整個肥
    item、只糾結去不去重）更根本——真問題是 item 本身太肥。

    source_map key 是 int（citation id），JSON object key 只能是 str → 這裡轉 str 存；
    重建（restore_rerun_state_from_report）時轉回 int（G17 round-trip 陷阱）。

    [R2 修訂 S1] 走選項 A（預設）：**不單存 current_context**——restore 從 source_map 依 cid
    排序重建（省一份重複複製、payload 減半）。此一致性只在 phase-1 快照點成立（cc_eq_sm
    invariant scope，見 items 大小評估段），而 build 拿的就是 :1811 phase-1 快照 state，故成立。
    Task 5 Step 1 量測時若 cc_eq_sm==False（不預期）→ stop-and-report、退選項 B（見 ⚠️ 註）。

    [R3 修訂 S2-new] 加存 `query_id`：DB fallback 用前端帶的 session_id 讀 research_report.rerunState，
    但一個 session 可跑多次 DR（不同 query_id），research_report 只存最後一次的產出。若前端 stale
    （同 session 跑了第二次 DR，但 KG 編輯用的還是第一次的 query_id）→ DB fallback 會拿「最後一次 DR
    的 rerunState」去 rerun「另一次 query 的 KG 編輯」→ 用錯 formatted_context/source_map → 張冠李戴。
    memory cache 路徑用 query_id 當 key 天然對齊、無此縫；DB fallback 引入此縫，故 rerunState 必須綁定
    產生它的 query_id，restore 後由 pre-check/execute_rerun 驗 `query_id == original_query_id`。
    [verified] `state.query_id` 可讀（research_state.py:33 `query_id: str = ""`；:1811 `_cache_research_state(state.query_id, state)` 已用它當 cache key，證明此 scope 可讀）。
    """
    return {
        'query': state.query,
        'mode': state.mode,
        'temporal_context': state.temporal_context,
        'enable_kg': state.enable_kg,
        'enable_web_search': state.enable_web_search,
        'formatted_context': state.formatted_context,
        # [R5 剝欄位] int key → str key（JSON 相容）；value 用 _slim_item 剝成精簡 5 欄 dict
        # （丟 schema_json 全文 / vector / scores，payload 238KB → ~15-30KB）。重建時 key 轉回 int。
        'source_map': {str(k): _slim_item(v) for k, v in state.source_map.items()},
        # 選項 A：不存 current_context——restore 從 source_map 依 cid 排序重建
        'items_count': len(state.items),
        # [R3 修訂 S2-new] 綁定產生此 rerunState 的 query_id，供 DB fallback 驗 session↔query_id 對齊
        'query_id': state.query_id,
    }


def restore_rerun_state_from_report(research_report: dict) -> Optional[dict]:
    """從 DB research_report 的內層 rerunState 重建 rerun 所需 state dict。

    回 None 若 research_report 無有效 rerunState（session 14 舊 row / 空內層）——判「有意義內容」
    而非純 truthiness（模組陷阱 #3：{} / {source_map:{}} 都 truthy 但無用 → 判無效）。

    重建做：(1) source_map str key → int；(2) current_context 依 cid 排序從 source_map 重建
    （去重，rerunState 未單存 current_context，見 build 的 payload 優化）；(3) items 重建成
    長度 items_count 的 placeholder list（phase 2-4 只用 len）。
    """
    if not isinstance(research_report, dict):
        return None
    rs = research_report.get('rerunState')
    # 有意義內容 predicate（非 truthiness）：必須有 formatted_context 有值 且 source_map 是非空 dict
    if not isinstance(rs, dict):
        return None
    # [R3 修訂 S4-補] source_map 型別檢查納入無效判定：build 恆存 dict，但 DB corrupt/舊資料可能存成
    # 非 dict（如 list / str）→ 下面 .items() 會拋 AttributeError（原 try/except 只包 int(k) 未涵蓋
    # 此類）。既然原則是「DB corrupt/舊資料不能把 400 變 500」，該一次涵蓋整類（根解非補丁）。
    if not (rs.get('formatted_context') and isinstance(rs.get('source_map'), dict) and rs['source_map']):
        return None
    # source_map str key → int（G17）。JSON 讀回的 key 恆為 str，這裡權威轉回 int。
    # [R2 修訂 S4 + R3 修訂 S4-補] DB 可能存 corrupt/舊資料——一次涵蓋整類轉換失敗：
    #   (a) source_map 非數字 key → int(k) 拋 ValueError/TypeError（原 R2 已涵蓋）；
    #   (b) source_map 值不是 dict 或結構異常（AttributeError 於 .items() 已由上面 isinstance 攔）；
    #   (c) items_count 非數字（如 "abc"）→ int(...) 拋 ValueError（R3 新增，原本裸露在 return 段外會炸 500）。
    # 全包一個 try/except（ValueError/TypeError/AttributeError）→ log warning（明確訊息、不 silent
    # fail）、return None → 上層走「無有效 rerunState → 400」正常降級。
    # 注意：這 try/except 是針對「DB corrupt 資料」的邊界降級，**不是**掩蓋「build 忘轉 str→int」的
    # 容錯（build 端恆存合法數字字串 key + 數字 items_count，round-trip test 以強斷言 int key 鎖死正確性）。
    try:
        source_map = {int(k): v for k, v in rs['source_map'].items()}
        # current_context 依 cid 排序從 source_map 重建（build 未單存 current_context 以省 payload）
        current_context = [source_map[cid] for cid in sorted(source_map.keys())]
        # [R3 修訂 S4-補] items_count parse 一併納入 try（非數字 → ValueError，不炸 500）
        items_count = int(rs.get('items_count', len(current_context)))
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"[RERUN] research_report.rerunState corrupt（非數字 source_map key / items_count "
                       f"非數字 / source_map 結構異常，舊資料或損毀）→ 判無效 rerunState、回落 400: {e}")
        return None
    return {
        'query': rs.get('query', ''),
        'mode': rs.get('mode', 'discovery'),
        'temporal_context': rs.get('temporal_context'),
        'enable_kg': rs.get('enable_kg', False),
        'enable_web_search': rs.get('enable_web_search', False),
        'formatted_context': rs['formatted_context'],
        'source_map': source_map,
        'current_context': current_context,
        # phase 2-4 只用 len(items)（stat）→ placeholder，長度 = 原 items_count
        'items': [None] * items_count,
        # [R3 修訂 S2-new] 帶回 query_id 供 DB fallback 驗 session↔query_id 對齊（舊資料缺 → None，
        # 上層 degrade 成不匹配 → 400，見向後相容決策）
        'query_id': rs.get('query_id'),
        'cached_at': time.time(),
    }


# === Graph fallback：actor-critic 多輪迭代防 graph 產出蒸發（2026-07-15 rerun E2E 撞出）===
# 根因：gap-enrichment / revise 輪的 LLM 可能省略 graph（analyst.py C5 已有 observability log），
# 而 pipeline 一律拿「最終輪」analyst output 去 serialize → 前輪已產好的 KG/argument/推論鏈被
# 空值覆蓋、整包蒸發（rerun E2E 實證：research 輪 KG 10+10、enriched 輪 0+0 → 前端空殼）。
# 修法：loop 內追蹤最新「非空」graph → post-loop 最終輪欄位空殼時補回。主 run 與 rerun 同受益。

_ANALYST_GRAPH_FIELDS = ("knowledge_graph", "argument_graph", "reasoning_chain_analysis")


def _graph_field_has_content(field: str, value) -> bool:
    """判 graph 欄位是否有「有意義內容」。

    陷阱：KnowledgeGraph 空物件（entities=[] relationships=[]）是 truthy（pydantic BaseModel
    instance 恆 truthy），`if analyst_output.knowledge_graph:` 擋不住 → KG 需看 entities/
    relationships 是否非空。argument_graph 是 list、reasoning_chain_analysis 是 Optional 物件，
    truthy 判準即足。
    """
    if not value:
        return False
    if field == "knowledge_graph":
        return bool(getattr(value, "entities", None) or getattr(value, "relationships", None))
    return True


def track_nonempty_graphs(response, tracker: dict) -> None:
    """actor-critic loop 每輪 analyst 產出後呼叫：記下最新「非空」的 graph 欄位值。

    兩輪都非空 → 記最新（最終輪語意優先，fallback 只在最終輪空殼時啟動）；
    空殼輪不覆蓋已記錄的非空版本。
    """
    for f in _ANALYST_GRAPH_FIELDS:
        v = getattr(response, f, None)
        if _graph_field_has_content(f, v):
            tracker[f] = v


def apply_graph_fallback(response, tracker: dict):
    """post-loop：最終輪 response 的 graph 欄位空殼/None、而本 run 前輪有非空版本 → 補回。

    fallback 來源是**同一 run 內的前一輪**（rerun 情境即「已含使用者 KG 編輯前提」產的那輪），
    非舊報告 stale graph，不張冠李戴。回 (response, restored_fields)——restored_fields 供
    caller log（可觀測，不 silent）；無補回時原物件原樣返回（零開銷）。
    """
    updates = {}
    for f, v in tracker.items():
        if hasattr(response, f) and not _graph_field_has_content(f, getattr(response, f, None)):
            updates[f] = v
    if not updates:
        return response, []
    return response.model_copy(update=updates), sorted(updates.keys())


class DeepResearchOrchestrator(OrchestratorBase):
    """
    Orchestrator for the Actor-Critic reasoning system.

    Coordinates the iterative loop between Analyst (Actor) and Critic,
    then uses Writer to format the final report.
    """

    # §v5: user-facing 文案，繁中、無內部用詞（mode 名 / discovery / strict）、誠實。
    # 此為深層防線（dead catch）文案——0 筆正常情境走 β-path，補不到落
    # _create_no_results_response（那才是使用者實際會看到的主文案）；此 catch
    # 只在非預期地拋 NoValidSourcesError 才顯示。
    # 措辭刻意不假設「網路已搜過」——此 catch 可能在 web search 未實際嘗試就觸發
    # （Codex AR Round 1）。故只說「沒有可用來源能支撐研究報告」，不宣稱已試過網路。
    _NO_VALID_SOURCES_MESSAGE = (
        "很抱歉，這次沒有可用來源能支撐研究報告，因此無法產出結果。\n\n"
        "建議您：\n"
        "1. 換個關鍵詞或更具體的描述再試一次\n"
        "2. 確認問題中的人名、機構名或事件名稱是否正確\n"
        "3. 稍後再試（部分即時資料可能尚未收錄）"
    )

    def __init__(self, handler: Any):
        """
        Initialize orchestrator with reasoning agents.

        Args:
            handler: Request handler with LLM configuration
        """
        super().__init__(handler)
        self.logger = get_configured_logger("reasoning.orchestrator")

        # Initialize agents
        analyst_timeout = CONFIG.reasoning_params.get("analyst_timeout", 60)
        critic_timeout = CONFIG.reasoning_params.get("critic_timeout", 30)
        writer_timeout = CONFIG.reasoning_params.get("writer_timeout", 45)

        self.analyst = AnalystAgent(handler, timeout=analyst_timeout)
        self.critic = CriticAgent(handler, timeout=critic_timeout)
        self.writer = WriterAgent(handler, timeout=writer_timeout)

        # Initialize source tier filter
        self.source_filter = SourceTierFilter(CONFIG.reasoning_source_tiers)

        # Unified context storage (Single Source of Truth)
        self.formatted_context = ""
        self.source_map = {}

    def _format_context_shared(self, items: List[Dict[str, Any]], start_id: int = 1) -> tuple[str, Dict[int, Dict]]:
        """
        Format context with citation markers - SINGLE SOURCE OF TRUTH.

        This ensures all agents (Analyst, Critic, Writer) use the same
        citation numbering system, preventing citation mismatch issues.

        Args:
            items: List of source items (already filtered and enriched by SourceTierFilter)

        Returns:
            Tuple of (formatted_string, source_map)
                - formatted_string: Context with [1], [2], [3] markers (token-budgeted for AI)
                - source_map: Dict mapping citation ID to source item (complete, for frontend)

        Design:
            - source_map: Contains ALL items (no limit) - managed by code, not LLM
            - formatted_context: Token-budgeted for LLM consumption
            - Citation numbers are consistent between both
        """
        MAX_TOTAL_CHARS = 20000  # ~10k tokens budget for formatted_context
        MAX_SNIPPET_LENGTH = 500
        OVERHEAD_PER_ITEM = 100  # Citation marker + source + title + newlines

        source_map = {}
        formatted_parts = []

        # ===== Step 1: Build COMPLETE source_map (no limit) =====
        # This is the ground truth for citation -> item mapping
        # Frontend will use this to display all citation references
        for idx, item in enumerate(items, start_id):
            source_map[idx] = item

        # ===== Step 2: Calculate how many items fit in token budget =====
        # Dynamically determine items_in_budget based on actual content size
        cumulative_chars = 0
        items_in_budget = 0

        for item in items:
            if isinstance(item, dict):
                desc_len = len(item.get("description") or item.get("articleBody", ""))
            else:
                desc_len = 0
            item_chars = min(desc_len, MAX_SNIPPET_LENGTH) + OVERHEAD_PER_ITEM

            if cumulative_chars + item_chars > MAX_TOTAL_CHARS:
                break
            cumulative_chars += item_chars
            items_in_budget += 1

        # Ensure at least some items are included
        items_in_budget = max(items_in_budget, min(10, len(items)))

        # ===== Step 3: Calculate snippet length for budgeted items =====
        total_estimated = sum(
            min(len((item.get("description") or item.get("articleBody", "")) if isinstance(item, dict) else ""), MAX_SNIPPET_LENGTH) + OVERHEAD_PER_ITEM
            for item in items[:items_in_budget]
        )

        if total_estimated > MAX_TOTAL_CHARS:
            reduction_ratio = MAX_TOTAL_CHARS / total_estimated
            snippet_length = max(int(MAX_SNIPPET_LENGTH * reduction_ratio), 100)
            self.logger.warning(
                f"Context too large ({total_estimated} chars for {items_in_budget} items), "
                f"reducing snippet length to {snippet_length} chars (ratio: {reduction_ratio:.2f})"
            )
        else:
            snippet_length = MAX_SNIPPET_LENGTH

        # ===== Step 4: Format context for AI (only budgeted items) =====
        for idx, item in enumerate(items[:items_in_budget], start_id):
            # Handle both dict and tuple/list formats
            if isinstance(item, dict):
                title = item.get("title") or item.get("name", "No title")
                description = item.get("description") or item.get("articleBody", "")
                source = item.get("site", "Unknown")
                date_published = item.get("datePublished", "")
            elif isinstance(item, (list, tuple)):
                title = item[2] if len(item) > 2 else "No title"
                try:
                    schema_json = item[1] if len(item) > 1 else "{}"
                    schema_obj = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
                    description = schema_obj.get("description") or schema_obj.get("articleBody", "")
                    date_published = schema_obj.get("datePublished", "")
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.debug(f"Failed to parse schema_json: {e}")
                    description = ""
                    date_published = ""
                source = item[3] if len(item) > 3 else "Unknown"
            else:
                title = "No title"
                description = ""
                source = "Unknown"
                date_published = ""

            # Format date for display (extract YYYY-MM-DD from ISO format)
            date_str = ""
            if date_published:
                date_str = str(date_published).split("T")[0]

            snippet = description[:snippet_length] + (
                "..." if len(description) > snippet_length else ""
            )
            # 發布日期錨定（core.temporal_anchor）：來源內文的「今年/去年」以該報導
            # 發布年換算，年份由 code 算好標在詞後，LLM 不必拿全域「今天」去推
            # → 不再把 2025 年報導的「今年」寫成 2026。標註在截斷後做，不動上方 char budget。
            snippet = annotate_relative_years(snippet, date_published)
            header = f"[{idx}] {source} - {title}"
            if date_str:
                header += f" ({date_str})"
            note = anchor_note(date_published)
            if note:
                header += note
            formatted_parts.append(f"{header}\n{snippet}\n")

        formatted_string = "\n".join(formatted_parts)

        # Add current datetime header for temporal query accuracy
        current_time_header = self._get_current_time_header()
        if current_time_header:
            formatted_string = current_time_header + formatted_string

        self.logger.info(
            f"Formatted context: {items_in_budget}/{len(items)} sources in AI context, "
            f"{len(source_map)} total in source_map, {len(formatted_string)} chars"
        )

        # Check if context is empty
        if not formatted_string or len(source_map) == 0:
            self.logger.warning(
                f"Empty context generated! items count: {len(items)}, "
                f"formatted_parts count: {len(formatted_parts)}"
            )

        return formatted_string, source_map

    def _extract_item_fields(self, item) -> tuple:
        """Shape-aware 抽 (title, source, description, url)——source_map item 有兩種形狀：
        dict（tier6 web/wiki doc、私有檔）或 list/tuple（站內 6-element row，
        description 藏在 item[1] schema_json 內）。原邏輯自 _build_critic_reference_sheet
        平移集中（票 2026-07-28-f：相關性 gate 與 writer evidence_lookup 共用）。
        """
        if isinstance(item, dict):
            title = item.get("title") or item.get("name", "No title")
            source = item.get("site", "Unknown")
            description = item.get("description") or item.get("articleBody", "")
            url = item.get("url") or item.get("link", "") or ""
        elif isinstance(item, (list, tuple)):
            title = item[2] if len(item) > 2 else "No title"
            source = item[3] if len(item) > 3 else "Unknown"
            url = item[0] if len(item) > 0 else ""
            try:
                schema_json = item[1] if len(item) > 1 else "{}"
                schema_obj = (
                    json.loads(schema_json) if isinstance(schema_json, str) else schema_json
                )
                description = schema_obj.get("description") or schema_obj.get("articleBody", "")
            except Exception as e:
                self.logger.warning(
                    f"_extract_item_fields: failed to parse description: {e}"
                )
                description = ""
        else:
            title, source, description, url = "No title", "Unknown", "", ""
        return title, source, description, url

    def _extract_item_date(self, item) -> str:
        """Shape-aware 抽 datePublished（發布日期錨定用）；抽不到回空字串（不猜年份）。

        形狀同 _extract_item_fields：dict（tier6 web/wiki doc、私有檔）或 list/tuple
        （站內 row，datePublished 藏在 item[1] schema_json 內）。
        """
        if isinstance(item, dict):
            return str(item.get("datePublished", "") or "")
        if isinstance(item, (list, tuple)):
            try:
                schema_json = item[1] if len(item) > 1 else "{}"
                schema_obj = (
                    json.loads(schema_json) if isinstance(schema_json, str) else schema_json
                )
                return str(schema_obj.get("datePublished", "") or "")
            except Exception as e:
                # 不可 silent fail：抽不到日期 → 該來源不做年份錨定，明示降級
                self.logger.warning(
                    f"_extract_item_date: failed to parse datePublished: {e}"
                )
                return ""
        return ""

    def _build_critic_reference_sheet(self, citations_used: List[int]) -> str:
        """
        SEC-6: Build a compact reference sheet for Critic containing only cited sources.

        Instead of passing full formatted_context to Critic, extract only the
        sources actually cited by Analyst, reducing token usage significantly.

        Args:
            citations_used: List of citation IDs from Analyst's response

        Returns:
            Formatted reference sheet string
        """
        snippet_length = CONFIG.reasoning_params.get("agent_isolation", {}).get(
            "reference_sheet_snippet_length", 500
        )
        parts = []
        for cid in sorted(set(citations_used)):
            item = self.source_map.get(cid)
            if not item:
                self.logger.warning(f"SEC-6: citation [{cid}] not found in source_map")
                continue

            title, source, description, _url = self._extract_item_fields(item)
            date_published = self._extract_item_date(item)

            snippet = description[:snippet_length] + ("..." if len(description) > snippet_length else "")
            # 與 _format_context_shared 同口徑做發布日期錨定，否則 critic 讀到的
            # 「今年」與 analyst/writer 讀到的年份不同步。
            snippet = annotate_relative_years(snippet, date_published)
            parts.append(f"[{cid}] {source} - {title}{anchor_note(date_published, include_date=True)}\n{snippet}\n")

        return "\n".join(parts)

    def _create_no_results_response(self, query: str) -> List[Dict[str, Any]]:
        """
        Create a response indicating no relevant documents were found.

        Used when the formatted context is empty, preventing the Analyst
        from hallucinating content without any source material.

        Args:
            query: The user's original query

        Returns:
            List with single NLWeb Item dict with no-results message
        """
        return [{
            "@type": "Item",
            "url": "internal://no-results",
            "name": f"查無相關資料：{query}",
            "site": "系統訊息",
            "siteUrl": "internal",
            "score": 0,
            "description": (
                f"# 查無相關資料\n\n"
                f"針對「{query}」的搜尋未找到任何相關文件。\n\n"
                f"**建議**：\n"
                f"1. 嘗試使用不同的關鍵詞\n"
                f"2. 在搜尋介面調整來源篩選，包含更多新聞來源\n"
                f"3. 確認資料庫中有相關內容"
            )
        }]

    def _get_current_time_header(self) -> str:
        """
        Generate current datetime header for temporal query accuracy.

        Returns:
            Formatted datetime header string or empty string if disabled.
        """
        try:
            # Get timezone from config (default: Asia/Taipei)
            timezone_str = CONFIG.reasoning_params.get("timezone", "Asia/Taipei")

            try:
                import pytz
                tz = pytz.timezone(timezone_str)
                current_time = datetime.now(tz)
            except ImportError:
                # Fallback if pytz not available
                current_time = datetime.now()
                self.logger.debug("pytz not available, using local time")

            # Format: 2026-01-13 14:30:00 星期一 (台北時間)
            weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            weekday = weekday_names[current_time.weekday()]

            header = f"""## 當前時間
{current_time.strftime('%Y-%m-%d %H:%M:%S')} {weekday} ({timezone_str})

當用戶詢問「今天」、「最近」、「現在」等時間相關詞彙時，請參考上述當前時間。
**注意：上述當前時間只用於理解「使用者問句」的相對時間，不可用來換算「來源內文」的相對時間。**
{TEMPORAL_ANCHOR_RULE}

## 可用資料來源
"""
            return header

        except Exception as e:
            self.logger.warning(f"Failed to generate current time header: {e}")
            return ""

    async def _extract_search_subject(self, state) -> str:
        """β-path 補搜查詢構造（票 2026-07-28-f Fix 1）：抽「查詢主體」取代整句 user query。

        整句問句（「找出台大邱啟新副教授的公開發言」）直接餵 Wikipedia/CSE 會
        fuzzy match 出沾邊條目（李登輝/彭明敏/動漫頁，prod ccdb5ab6 實錘）。三層策略：
        1. 現成產物零成本復用：QU 已抽出的人名（handler.author_search.author_name）。
           （耦合註記：票 2026-07-28-e 正在重議 author 語意；此處以 getattr 防禦性讀取，
           該票改 shape 時只需同步這一個讀取點。）
        2. low-tier LLM 抽一次（僅 β-path 觸發且無現成產物時；成本 +1 次 low call）。
        3. fail-open：LLM 失敗 / sentinel / 回空 → 退回整句 state.query（不比現狀差），
           warning log（不可 silent fail）。
        """
        # 1. 現成產物（QU regex/LLM 已抽的人名——author 誤判情境下這正是查詢主體）
        author = getattr(self.handler, "author_search", None) or {}
        if author.get("is_author_search") and str(author.get("author_name") or "").strip():
            subject = str(author["author_name"]).strip()
            self.logger.info(
                f"[ZERO-RESULTS] search subject from QU author_search: {subject!r}"
            )
            return subject

        # 2. low-tier LLM 抽取（局部 import 對齊 hallucination_guard 慣例；
        #    測試 patch core.llm.ask_llm）
        from core.llm import ask_llm, LLMError

        prompt = (
            "你是搜尋查詢精煉助手。使用者的原始問句將被用於 Wikipedia 與 Google 搜尋；"
            "整句問句會讓 Wikipedia fuzzy match 出無關條目。\n"
            "請從問句抽出「查詢主體」——問句真正要查的具名實體"
            "（人名 / 組織 / 公司 / 地名 / 事件名）；若無具名實體則抽核心主題詞（2-8 個字）。\n"
            "規則：\n"
            "- 只回主體本身，不含「找出」「請幫我」「的公開發言」等請求語與修飾\n"
            "- 有具名實體優先實體（例：「找出台大邱啟新副教授的公開發言」→「邱啟新」）\n"
            "- 多個實體時取最核心的一個，可帶必要的消歧義詞（例：「台大 邱啟新」）\n\n"
            f"## 問句\n{state.query}\n\n"
            '回傳 JSON：{"search_subject": "..."}'
        )
        schema = {
            "type": "object",
            "properties": {"search_subject": {"type": "string"}},
            "required": ["search_subject"],
        }
        try:
            resp = await ask_llm(
                prompt,
                schema,
                level="low",
                query_params=getattr(self.handler, "query_params", {}),
                max_length=256,
                timeout=15,
            )
        except Exception as e:
            self.logger.warning(
                f"[ZERO-RESULTS] subject extraction LLM failed "
                f"({type(e).__name__}: {e}); fallback to full query"
            )
            return state.query
        # LLMError 是 falsy dict 子類——顯式偵測，不可用 truthiness 吞掉 provider 故障
        if isinstance(resp, LLMError) or not isinstance(resp, dict):
            self.logger.warning(
                f"[ZERO-RESULTS] subject extraction unusable ({resp!r}); "
                f"fallback to full query"
            )
            return state.query
        subject = str(resp.get("search_subject") or "").strip()
        if not subject:
            self.logger.warning(
                "[ZERO-RESULTS] subject extraction empty; fallback to full query"
            )
            return state.query
        self.logger.info(
            f"[ZERO-RESULTS] search subject extracted: {subject!r} "
            f"(from query {state.query!r})"
        )
        return subject

    async def _attempt_zero_results_web_search(self, state) -> bool:
        """
        β path: when internal retrieval returned zero sources, run ONE Google web
        search (synthetic WEB_SEARCH gap = state.query) to backfill grounding before
        giving up. Returns True if web search added at least one source (caller then
        proceeds to Actor-Critic), False otherwise (caller keeps no-results response).

        Anti-hallucination: this is the ONLY place the Analyst is allowed to run on an
        initially-empty source_map, and ONLY because real external sources were fetched.
        """
        import types
        from reasoning.schemas_enhanced import GapResolution, GapResolutionType

        # 1. Guard: web search disabled -> visible (non-silent) degradation.
        if not state.enable_web_search:
            self.logger.info(
                "[ZERO-RESULTS] web search disabled (enable_web_search=False); "
                "returning no-results"
            )
            return False

        # 2. Build synthetic single-WEB_SEARCH-gap response (only .gap_resolutions is read).
        #    Fix 1（票 2026-07-28-f）：search_query 不用整句 user query——整句餵 wiki 會
        #    fuzzy match 沾邊條目進池；改抽「查詢主體」（QU 現成產物 → low-tier LLM → 整句 fail-open）。
        search_subject = await self._extract_search_subject(state)
        gap = GapResolution(
            gap_type="zero_retrieved",
            resolution=GapResolutionType.WEB_SEARCH,
            search_query=search_subject,
            reason="zero internal sources retrieved",
            requires_web_search=True,
        )
        response = types.SimpleNamespace(gap_resolutions=[gap])

        # 3. Record current context size before backfill.
        before = len(state.current_context)

        # 4. Reuse the existing gap-resolution machinery (Google + optional Wikipedia
        #    fallback, tracing, ImportError/Exception log-only degradation all reused).
        await self._process_gap_resolutions(
            response=response,
            mode=state.mode,
            current_context=state.current_context,
            enable_web_search=state.enable_web_search,
            tracer=state.tracer,
            query_id=state.query_id,
        )

        # 5. Measure how many sources were appended.
        added = len(state.current_context) - before
        self.logger.info(f"[ZERO-RESULTS] web search added {added} sources")

        # 6. Nothing added -> caller keeps no-results (anti-hallucination preserved).
        if added == 0:
            return False

        # 7. Re-format so formatted_context / source_map are contiguous + consistent
        #    for the Analyst/Critic. _execute_web_searches only mutated self.source_map;
        #    rebuild the whole map from the full enriched current_context, then re-establish
        #    the G2 reference-identity invariant (mirror the sync in _phase_filter_and_prepare).
        state.formatted_context, state.source_map = self._format_context_shared(state.current_context)
        self.formatted_context = state.formatted_context
        self.source_map = state.source_map

        # 8. Web grounding fetched -> caller proceeds to Actor-Critic.
        return True

    async def _resolve_web_search_gaps_in_loop(
        self,
        response: Any,
        mode: str,
        state: 'ResearchState',
        enable_web_search: bool,
        web_searched_queries: set,
        tracer: Any = None,
        query_id: str = None,
    ) -> bool:
        """防線三：在 actor-critic loop 內，把 response 的 web_search 類 gap_resolutions
        兌現成真正的 Google 呼叫，不論 response.status 是 SEARCH_REQUIRED 還是 DRAFT_READY。

        起因：Analyst 可能同一次輸出 status=SEARCH_REQUIRED（驅動站內 secondary search）
        + gap_resolutions[web_search]（該上網查的具名實體）。orchestrator 的
        SEARCH_REQUIRED 分支會在抵達主路徑 :846 _process_gap_resolutions 之前 continue 掉，
        導致 web_search gap 永遠不被兌現（高端訓 zero-Google bug）。此 helper 在 SEARCH_REQUIRED
        分支入口先把 web_search gap 兌現，讓 web source 與站內 secondary source 同迭代合流。

        復用既有 _process_gap_resolutions 引擎（同 v1 β-path 的 reuse-not-fork）。
        **隱性契約**：本 helper 依賴 _process_gap_resolutions 只讀 response.gap_resolutions
        （:2034 迴圈），故餵一個只帶 gap_resolutions 的 SimpleNamespace 即可；若未來
        _process_gap_resolutions 開始讀 response 的其他欄位（status/draft/...），此處會漏傳、需同步。

        R3 記帳集中（Gemini nit：dedup 互動精確描述）：normalize / response 內部去重 / 跨路徑 dedup /
        cap / mark 五件事**全部由引擎 _process_gap_resolutions 的 web gap 收集迴圈（:2034-2058）做**
        （Step 4a），**helper 完全不讀也不寫 web_searched_queries**（純委派，只把 set 傳穿透給引擎）。
        引擎收集迴圈的順序（每個 web gap 依序）：(1) _normalize_web_query 產 key → (2) 命中 response 內部
        seen set 則 skip → (3) 命中跨路徑 web_searched_queries 則 skip → (4) 達 cap 則 skip →
        (5) 通過才 append 進 web_search_gaps 並同時把 key add 進兩個 set（response 內部 + 跨路徑）。
        - dedup：web_searched_queries 跨迭代持久 + 跨路徑共享（DRAFT_READY :857 主路徑用同一 set），
          同一 normalize 後 key 全 session 只打一次 Google（跨路徑防「helper 打一次、DRAFT_READY 再打一次」）。
        - cap：run 級 hard cap（tier_6.gap_routing.max_external_calls_per_run，沿用 LR 鍵），
          `len(web_searched_queries)` >= cap 時不再收新 distinct web query、visible log（cap_skipped）
          （每 accept 同步 mark，故 len(set) 即時含 prior + 本輪已 accept，不 double-count）。
        - mark：引擎在收集決定送出 web gap 的當下 add（步驟 5），**不在被 mock 的 _execute_web_searches**
          （Codex B2：故 mock _execute_web_searches 的測試依然穩定，set 仍被正確寫）。
          trade-off（in-house nit）：_execute_web_searches try/except 降級（:2303-2306 log-only）
          時 query 已被 mark、本 run 不重試——換取「測試穩定（mark 在未被 mock 的收集迴圈）」
          與「防額度迴圈（異常不重打同 query）」；下游降級有 visible warning log，不 silent。

        re-format 分工（isolation-aware，B-ISO snapshot-before）：
        - non-isolation：全量重建 formatted_context / source_map（mirror v1 β-path :414）。
          全量重建會覆寫引擎剛登記的 web 區間、重新分配連續 id，故無「引擎 append + helper 再 append」疊加。
        - isolation（SEC-6「只看本迭代新 docs」）：**不全量重建、不自己 append source_map**——
          引擎（_execute_web_searches:2282-2288）已把 web docs append 進 source_map（start_id=before_max+1）。
          helper 用 snapshot 的 before_max 取「引擎已登記的 web 區間」（keys > before_max 的 items），
          用**引擎給的實際 id** format 成 pending_web_formatted（start_id 對齊 before_max+1，id 一致），
          暫存回 state（供 :761/:785 與站內新 docs 串接）。**helper 對 source_map 只讀不寫**（避免雙重 ID）。

        SF1 契約：pending_web_formatted 同輪必被 SEARCH_REQUIRED 站內分支消費（:761 / :785 串接），
        消費即清（串接後設 None）。isolation 路徑才會設此欄位；non-isolation 全量重建不用它。

        Returns True 若至少補進一個 web source（呼叫端需 re-format 已在此完成 / 暫存）。
        """
        from reasoning.schemas_enhanced import GapResolutionType

        # 1. Guard：web search 關 → 可見（非 silent）no-op。
        if not enable_web_search:
            self.logger.info("[WEB-GAP-RESOLVE] web search disabled; skip web gap resolution")
            return False

        # 2. Harvest：判斷本 response 有無 WEB_SEARCH gap 可兌現（僅判斷有無，不 dedup/不 cap
        #    ——normalize/response 內部去重/跨路徑 dedup/cap/mark 全由引擎收集迴圈做，Step 4a）。
        #    無 web gap → early-return no-op（絕大多數 SEARCH_REQUIRED 查詢只有 internal new_queries）。
        _all_web = [
            g for g in (getattr(response, 'gap_resolutions', None) or [])
            if g.resolution == GapResolutionType.WEB_SEARCH and g.search_query
        ]
        _harvested = len(_all_web)
        if _harvested == 0:
            return False

        # 3. 兌現前 snapshot：context size + source_map 最大 id（B-ISO snapshot-before）。
        #    self.source_map is state.source_map（同一 dict reference，:456-458）；引擎兌現時會 append
        #    web docs 進去（start_id=before_max+1）。snapshot before_max 供事後取「引擎已登記的 web 區間」。
        before = len(state.current_context)
        before_max = max(state.source_map.keys(), default=0)
        self.logger.info(
            f"[WEB-GAP-RESOLVE] 送 {_harvested} 個 web_search gap 進引擎收集迴圈（引擎做 dedup/cap/mark）："
            f"{[g.search_query for g in _all_web]}（before_max_id={before_max}）"
        )

        # 4. 復用既有引擎（餵整批 web gap；normalize/去重/dedup/cap/mark 全在引擎收集迴圈做）。
        #    傳共享 web_searched_queries set —— 引擎在收集決定送出 web gap 的當下 add 進 set。
        #    引擎內 _execute_web_searches:2282-2288 會 current_context.extend + source_map append。
        filtered = types.SimpleNamespace(gap_resolutions=_all_web)
        await self._process_gap_resolutions(
            response=filtered,
            mode=mode,
            current_context=state.current_context,
            enable_web_search=enable_web_search,
            tracer=tracer,
            query_id=query_id,
            web_searched_queries=web_searched_queries,  # R3：引擎收集迴圈做 normalize/去重/dedup/cap/mark
        )

        # 5. 量補進幾筆；0 → 不 re-format、回 False（全被 dedup/cap 擋掉 或 Google 回空）。
        added = len(state.current_context) - before
        # 標準化 summary log（Codex SF-3，固定 ASCII key 便於 E2E grep）：
        # harvested= dedup_skipped= cap_skipped= executed= sources_added=（skip 細節由引擎收集迴圈逐條 log；
        # 此處 dedup_skipped/cap_skipped/executed 由引擎回填或此處近似，統一 underscore key）。
        self.logger.info(
            f"[WEB-GAP-RESOLVE] harvested={_harvested} sources_added={added}"
        )
        if added == 0:
            return False

        # 6. re-format（isolation-aware，B-ISO snapshot-before）。
        if getattr(state, "enable_isolation", False):
            # SEC-6：引擎已把 web docs append 進 source_map（id > before_max）。helper 不再 append，
            # 直接取引擎已登記的 web 區間、用引擎給的實際 id format 成 pending_web_formatted。
            new_web_ids = sorted(k for k in state.source_map.keys() if k > before_max)
            new_web_items = [state.source_map[k] for k in new_web_ids]
            # start_id 對齊引擎登記區間起點 → _format_context_shared 產出的 [id] marker 與引擎 source_map id 完全一致。
            web_formatted, _ = self._format_context_shared(
                new_web_items, start_id=(before_max + 1)
            )
            state.pending_web_formatted = web_formatted  # SF1：供站內分支 :761/:785 串接，消費即清
            # ★ 不對 state.source_map 做任何 append/update —— 引擎已登記，helper 只讀（避免雙重 ID）。
        else:
            # non-isolation：全量重建（mirror v1 β-path :414，行為 byte-identical）。
            # 全量重建覆寫引擎剛登記的區間、重新分配連續 id，無疊加。
            state.formatted_context, state.source_map = self._format_context_shared(state.current_context)
            self.formatted_context = state.formatted_context
            self.source_map = state.source_map
        return True

    # 相關性 gate 的池 digest 字數上限：超出部分不送判、一律保留（fail-open）。
    # 設計餘裕：每筆 title+150 字摘要 ~200 字 → 24000 涵蓋 ~117 筆（AR R1 in-house
    # 獨立驗算），高於一般 DR 池量級；即使量級估錯，超出方向是 fail-open 不誤殺。
    # 票 2026-07-28-m：權威值移至 relevance_gate_core（DR/LR 共用）；保 class attr 名
    # 不變 → 既有測試 mock `orch._RELEVANCE_GATE_DIGEST_CHAR_BUDGET` 不破。
    _RELEVANCE_GATE_DIGEST_CHAR_BUDGET = RELEVANCE_GATE_DIGEST_CHAR_BUDGET

    async def _relevance_gate_source_pool(self, state) -> None:
        """Phase 1 證據池相關性 gate（票 2026-07-28-f Fix 2：根治 empty ≠ irrelevant）。

        進 Actor-Critic 前對整池（站內 + β-path 補進的 tier6）做一次批次語意判定：
        - 全池不相關 → state.early_return = 既有 _create_no_results_response（誠實查無）
        - 部分不相關 → 濾掉後全量重建 formatted_context / source_map（mirror β step 7）
        - 判定失敗（exception / LLMError / 回傳無法解析）→ fail-open 全池保留 + warning。
          與 LR hallucination_guard R1 fail-closed 反向：gate 誤殺真證據 = 使用者拿不到
          報告，代價高於留噪音（噪音下游還有 Analyst 判讀 + Critic + SEC-5 兜底）。
        成本：每 DR run +1 次 low-tier call（批次判全池，不逐 source call）。
        """
        if not state.source_map:
            return  # 空池由既有 β-path / no-results 分支處理

        # 1. 池 digest（shape-aware；字數上限內的 id 才送判，超出者保留 = fail-open）
        digest_lines = []
        judged_ids = []
        used_chars = 0
        for cid in sorted(state.source_map):
            title, source, description, _url = self._extract_item_fields(
                state.source_map[cid]
            )
            line = f"[{cid}] {source} - {title}：{(description or '')[:150]}"
            if used_chars + len(line) > self._RELEVANCE_GATE_DIGEST_CHAR_BUDGET:
                self.logger.warning(
                    f"[RELEVANCE-GATE] digest 達字數上限，[{cid}] 起未送判"
                    f"（fail-open 保留）"
                )
                break
            digest_lines.append(line)
            judged_ids.append(cid)
            used_chars += len(line) + 1  # +1 = join 換行符（in-house nit：budget 誤差歸零）
        digest = "\n".join(digest_lines)

        # 2+3. 批次判 + clamp（委派共用核心；票 2026-07-28-m 抽出，DR/LR 同一份 prompt）
        from reasoning.relevance_gate_core import judge_irrelevant_source_ids

        irrelevant = await judge_irrelevant_source_ids(
            query=state.query,
            digest=digest,
            judged_ids=judged_ids,
            query_params=getattr(self.handler, "query_params", {}),
            logger=self.logger,
            log_prefix="RELEVANCE-GATE",
        )
        # 🔧 R1：fail-open（core 回 None，已在 core 印過 warning）→ 全池保留、直接 return。
        # 對齊 DR 原 :948-962 語義（warning 後 return，不動池、不印「全相關」info）。
        if irrelevant is None:
            return

        if not irrelevant:
            self.logger.info(
                f"[RELEVANCE-GATE] 全池 {len(state.source_map)} 筆判定相關，池不動"
            )
            return

        kept_ids = [cid for cid in sorted(state.source_map) if cid not in irrelevant]
        self.logger.warning(
            f"[RELEVANCE-GATE] 剔除 {len(irrelevant)}/{len(state.source_map)} 筆"
            f"不相關來源：{sorted(irrelevant)}"
        )

        # 4a. 零相關 → 誠實查無（復用既有 builder，不重寫）
        if not kept_ids:
            self.logger.warning(
                "[RELEVANCE-GATE] 全池不相關——走誠實查無（防方法論自述長文）"
            )
            state.early_return = self._create_no_results_response(state.query)
            return

        # 4b. 部分相關：濾池 + 全量重建 + G2 sync（mirror β-path step 7 / :646-648）
        state.current_context = [state.source_map[cid] for cid in kept_ids]
        state.formatted_context, state.source_map = self._format_context_shared(
            state.current_context
        )
        self.formatted_context = state.formatted_context
        self.source_map = state.source_map

    async def _phase_filter_and_prepare(self, state: 'ResearchState') -> 'ResearchState':
        """
        Phase 1: Filter sources by tier + format context with citations.

        Reads: state.items, state.mode, state.tracer
        Writes: state.current_context, state.formatted_context, state.source_map
        May set: state.early_return (if no sources)
        """
        await self._emit_phase_event("filter_and_prepare", "started")

        # Phase 1a: Filter and prepare sources.
        # §v5: no try/except here — _filter_and_prepare_sources no longer raises on
        # empty (guardrail removed). Empty current_context flows to Phase 1b → empty
        # source_map → :598 β-path.
        state.current_context = await self._filter_and_prepare_sources(
            items=state.items,
            mode=state.mode,
            tracer=state.tracer,
        )

        # Phase 1b: Format context with citations
        state.formatted_context, state.source_map = await self._format_research_context(
            items=state.current_context,
            tracer=state.tracer,
        )

        # Sync to instance attributes (backward compat until all phases migrated)
        # SYNC: self.source_map and state.source_map must be the SAME dict reference,
        # because _process_gap_resolutions() and _build_critic_reference_sheet()
        # read/mutate self.source_map directly. Remove when all helpers read from state.
        self.formatted_context = state.formatted_context
        self.source_map = state.source_map

        # RSN-11 + β: empty source_map → try one web search before giving up
        if not state.source_map:
            self.logger.warning(
                "RSN-11: source_map empty -- no internal sources. "
                "β path: attempting zero-results web search backfill."
            )
            web_recovered = await self._attempt_zero_results_web_search(state)
            if not web_recovered:
                self.logger.warning(
                    "RSN-11/β: web search yielded no sources (or disabled) -- "
                    "returning no-results response to prevent hallucination."
                )
                state.early_return = self._create_no_results_response(state.query)
                await self._emit_phase_event("filter_and_prepare", "completed")
                return state
            # web_recovered: fall through to normal flow with enriched source_map
            self.logger.info(
                f"RSN-11/β: web search recovered {len(state.source_map)} sources; "
                "proceeding to Actor-Critic with web-augmented context."
            )

        # 票 2026-07-28-f Fix 2：empty ≠ irrelevant——非空池（含 β 補回的池）也要過
        # 相關性 gate。零相關 → 誠實查無 early return；部分相關 → 池已濾好續跑。
        await self._relevance_gate_source_pool(state)
        if state.early_return is not None:
            await self._emit_phase_event("filter_and_prepare", "completed")
            return state

        await self._emit_phase_event("filter_and_prepare", "completed")
        return state

    def _build_retrieval_evidence_summary(self) -> Optional[str]:
        """組主查詢站內檢索的「證據強弱」摘要，供 Analyst gap 分類時參考。

        復用 handler 上已算好的 Signal A/B flag（正確 0.0 排除邏輯已在 postgres_client
        跑過，不在此重算）。命中筆數從 items 拿；最高真實向量分數用共用純函式（同排除邏輯）。
        回傳 None = 無 items（0 結果由防線一 short-circuit，這裡不注入）。

        items 形狀（2026-07-05 親讀 postgres_client.py:791-825 + baseHandler.py:597-606 驗）：
        `self.handler.final_retrieved_items` = search() 的 return，元素是 **list row 非 dict**：
          - AGGREGATOR_KEEP_SCORES='1'（預設）→ 6-item list
            [url, schema_str, title, source, vector_or_None, scores_dict]，
            scores_dict['vector_score'] 是真實向量分數。
          - flag='0' → legacy 4/5-tuple（無 scores）。
          - baseHandler 在 include_private_sources 時 prepend 私有檔 4-element list
            [url, json_str, name, site]（無 scores）。
        max_real_vector_score 已 shape-aware：6-item 讀 index 5、legacy/私有檔略過。
        命中筆數 n=len(items) 含私有檔（它們也是使用者看到的檢索結果）。
        """
        from retrieval_providers.postgres_client import max_real_vector_score

        items = getattr(self.handler, "final_retrieved_items", None) or []
        n = len(items)
        if n == 0:
            return None

        low_rel = bool(getattr(self.handler, "low_relevance_warning", False))
        low_kw = bool(getattr(self.handler, "low_keyword_match_warning", False))
        top = max_real_vector_score(items)
        top_str = f"{top:.2f}" if top is not None else "無真實向量證據（純關鍵字命中）"

        lines = [
            "## 站內檢索證據強弱（供 gap 分類參考，非 binding）",
            f"- 站內命中筆數：{n}",
            f"- 最高真實向量相關分數：{top_str}",
        ]
        if low_rel:
            lines.append("- ⚠️ Signal A：站內結果關聯性較弱（最高向量分數低於品質門檻）")
        if low_kw:
            lines.append("- ⚠️ Signal B：站內結果關鍵字字面吻合度低")
        if not low_rel and not low_kw:
            lines.append("- 站內證據關聯性正常")

        summary = "\n".join(lines)
        # no-silent：分類判錯時的驗偽依據
        self.logger.info(
            f"[GAP-EVIDENCE] n={n} top_vec={top_str} "
            f"signalA(low_rel)={low_rel} signalB(low_kw)={low_kw}"
        )
        return summary

    async def _phase_actor_critic_loop(self, state: 'ResearchState') -> 'ResearchState':
        """
        Phase 2: Actor-Critic iterative loop.

        Contains: Analyst research/revise, Gap Detection, Gap Resolution (Tier 6),
                  Critic review, convergence check.

        Reads: state.formatted_context, state.source_map, state.current_context,
               state.query, state.mode, state.temporal_context, state.enable_kg,
               state.enable_web_search, state.query_id, state.tracer,
               state.iteration_logger, state.enable_isolation, state.max_iterations
        Writes: state.draft, state.review, state.response, state.iteration,
                state.reject_count, state.seen_citation_ids, state.analyst_citations,
                state.formatted_context, state.source_map, state.current_context
        May set: state.early_return (if max iterations exhausted with no data)

        Raises:
            ResearchCancelledError: If client disconnects (7 checkpoints)
        """
        await self._emit_phase_event("actor_critic_loop", "started")

        max_iterations = state.max_iterations
        iteration = 0
        final_pass = False  # 防線四：迴圈前初始化，防 max_iterations=0 邊界 post-loop 讀 NameError
        draft = None
        review = None
        response = None  # RSN-7: Initialize to avoid unbound variable if first analyst call fails
        reject_count = 0

        # SEC-6: Agent isolation state
        enable_isolation = state.enable_isolation
        if enable_isolation:
            self.logger.info("SEC-6: Agent isolation ENABLED")
        seen_citation_ids: set = set()
        # 防線三：跨迭代 web query 去重（同一 search_query 全 session 只打一次 Google，防額度燒爆）
        web_searched_queries: set = set()
        # Graph fallback：追蹤本 run 最新非空 graph 產出（KG/argument/推論鏈），post-loop
        # 最終輪空殼時補回——防 gap-enrichment/revise 輪 LLM 省略 graph 導致產出蒸發
        last_nonempty_graphs: dict = {}

        # Aliases for readability (state is the source of truth)
        query = state.query
        mode = state.mode
        temporal_context = state.temporal_context
        enable_kg = state.enable_kg
        enable_web_search = state.enable_web_search
        query_id = state.query_id
        tracer = state.tracer
        iteration_logger = state.iteration_logger

        while iteration < max_iterations:
            web_added = False  # 防線三 B-LAST：每輪顯式初始化（Codex R4 blocker）
            # ★ 必須在 :679 SEARCH_REQUIRED 分支之前初始化。
            # 理由：:794 讀點在 SEARCH_REQUIRED 分支內（:779 else 路徑），
            #       :829 讀點在 SEARCH_REQUIRED 分支外、之後（:819 draft= 落下處）。
            # 非 SEARCH_REQUIRED 路徑（如 DRAFT_READY）從不進 4b 賦值，
            # 若無顯式初始化則 :829 guard 讀到上輪殘值或 NameError（空 draft 路徑）。
            # 防線四：本輪是否為最後一輪（iteration 0-indexed，+1 才是「第幾輪」）。
            # v3 extra-pass 退一格（:980 iteration -= 1）後，下一輪迴圈頂用當下 iteration 重算 →
            # 退一格自然把「下一輪才是 final」蓋到，不需特殊處理。
            final_pass = (iteration + 1 >= max_iterations)
            self._check_connection()  # Checkpoint 1: loop start
            self.logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")

            # Tracing: Iteration start
            if tracer:
                tracer.start_iteration(iteration + 1, max_iterations)

            # Send progress: Analyst analyzing
            await self._send_progress({
                "message_type": "intermediate_result",
                "stage": "analyst_analyzing",
                "iteration": iteration + 1,
                "total_iterations": max_iterations
            })

            # Analyst: Research or revise
            if review and review.status == "REJECT":
                # Revise based on critique
                reject_count += 1
                self.logger.info("Analyst revising draft based on critique")
                analyst_input = {
                    "original_draft": draft,
                    "review": review,
                    "formatted_context": state.formatted_context
                }

                self._check_connection()  # Checkpoint 2: before analyst.revise()
                if tracer:
                    with tracer.agent_span("analyst", "revise", analyst_input) as span:
                        response = await self.analyst.revise(
                            original_draft=draft,
                            review=review,
                            formatted_context=state.formatted_context,
                            query=query,
                            enable_kg=enable_kg,  # B9: match research() schema selection
                            final_pass=final_pass,  # 防線四（修訂 3）：最後一輪 revise 也強制寫稿
                        )
                        span.set_result(response)
                else:
                    response = await self.analyst.revise(
                        original_draft=draft,
                        review=review,
                        formatted_context=state.formatted_context,
                        query=query,
                        enable_kg=enable_kg,  # B9: match research() schema selection
                        final_pass=final_pass,  # 防線四（修訂 3）：最後一輪 revise 也強制寫稿
                    )

                iteration_logger.log_agent_output(
                    iteration=iteration + 1,
                    agent_name="analyst_revise",
                    input_prompt=f"Draft: {draft[:100]}...\nReview: {review}",
                    output_response=response
                )
            else:
                # Initial research
                self.logger.info("Analyst conducting research")
                analyst_input = {
                    "query": query,
                    "formatted_context": state.formatted_context,
                    "mode": mode,
                    "temporal_context": temporal_context
                }

                # 防線二：組站內檢索證據強弱摘要，注入 gap 分類輸入（只在初次 research；
                # research_with_enriched_data 的 re-run 刻意不注入，gap 已分類完）
                retrieval_evidence = self._build_retrieval_evidence_summary()

                self._check_connection()  # Checkpoint 3: before analyst.research()
                if tracer:
                    with tracer.agent_span("analyst", "research", analyst_input) as span:
                        response = await self.analyst.research(
                            query=query,
                            formatted_context=state.formatted_context,
                            mode=mode,
                            temporal_context=temporal_context,
                            enable_kg=enable_kg,  # Phase KG: Pass per-request flag
                            enable_web_search=enable_web_search,  # Stage 5: Pass web search flag
                            retrieval_evidence=retrieval_evidence,  # 防線二：gap 分類證據注入
                            final_pass=final_pass,  # 防線四：最後一輪強制 best-effort 寫稿
                        )
                        span.set_result(response)
                else:
                    response = await self.analyst.research(
                        query=query,
                        formatted_context=state.formatted_context,
                        mode=mode,
                        temporal_context=temporal_context,
                        enable_kg=enable_kg,  # Phase KG: Pass per-request flag
                        enable_web_search=enable_web_search,  # Stage 5: Pass web search flag
                        retrieval_evidence=retrieval_evidence,  # 防線二：gap 分類證據注入
                        final_pass=final_pass,  # 防線四：最後一輪強制 best-effort 寫稿
                    )

                iteration_logger.log_agent_output(
                    iteration=iteration + 1,
                    agent_name="analyst_research",
                    input_prompt=f"Query: {query}\nMode: {mode}",
                    output_response=response
                )

            # Graph fallback：記下本輪（research/revise）非空 graph 產出
            track_nonempty_graphs(response, last_nonempty_graphs)

            # Gap detection: Handle SEARCH_REQUIRED
            if response.status == "SEARCH_REQUIRED":
                self.logger.warning(
                    f"Analyst requested additional search (iteration {iteration + 1}): "
                    f"{response.new_queries}"
                )

                # 防線三 N-2 defensive：進 SEARCH_REQUIRED 分支先清 pending_web_formatted（消費即清雙保險）。
                # 正常流程 :761/:785 消費後已設 None；此處 reset 防「上輪異常路徑漏清」殘留串進本輪視野。
                state.pending_web_formatted = None
                # 防線三：先兌現本 response 的 web_search 類 gap（不論 status）。
                # Analyst 可能同時回 SEARCH_REQUIRED + web_search gap；若只走站內 secondary
                # search（下方），web gap 會在 iteration+=1;continue 時被丟棄（高端訓 zero-Google bug）。
                # 在此把 web source 與站內 secondary source 同迭代合流補進 context，下輪 Analyst 一起整合。
                web_added = await self._resolve_web_search_gaps_in_loop(
                    response=response,
                    mode=mode,
                    state=state,
                    enable_web_search=enable_web_search,
                    web_searched_queries=web_searched_queries,
                    tracer=tracer,
                    query_id=query_id,
                )
                if web_added:
                    self.logger.info(
                        "[WEB-GAP-RESOLVE] web source 已合流進 context，"
                        "與站內 secondary search 結果一起交下輪 Analyst 整合"
                    )

                # Tracing: Gap detection
                if tracer:
                    tracer.condition_branch(
                        "GAP_DETECTION",
                        "SEARCH_REQUIRED",
                        {
                            "missing_information": response.missing_information,
                            "new_queries": response.new_queries
                        }
                    )

                # Send progress message to frontend
                await self._send_progress({
                    "message_type": "intermediate_result",
                    "stage": "gap_search_started",
                    "gap_reason": ", ".join(response.missing_information) if response.missing_information else "資料缺口",
                    "new_queries": response.new_queries,
                    "iteration": iteration + 1
                })

                # Execute secondary search for each new query
                secondary_results = []

                self._check_connection()  # Checkpoint 4: before secondary searches
                for new_query in response.new_queries:
                    try:
                        # Call retriever with same parameters as original search.
                        # NOTE: deliberately NO handler= here. The low-relevance/low-keyword
                        # warnings are bound to the initial research-start retrieval only
                        # (the research-start baseline). Gap search is a web-augment
                        # reinforcement step that re-enters research() via the actor-critic loop;
                        # passing handler= would re-set the monotonic (set-only, never-reset)
                        # warning flags from gap-search results and spuriously re-emit the
                        # SSE warning mid-research. Mirrors the LR loop_engine per-loop guard.
                        results = await retriever_search(
                            query=new_query,
                            site=self.handler.site,
                            num_results=20,  # Smaller batch for gap search
                            query_params=self.handler.query_params,
                        )
                        secondary_results.extend(results)
                        self.logger.info(f"Gap search for '{new_query}': {len(results)} results")
                    except Exception as e:
                        self.logger.error(f"Secondary search failed for '{new_query}': {e}")

                # Handle search results
                if secondary_results:
                    # Prepare new results (source tier enrichment removed; pass-through)
                    new_context = self.source_filter.filter_and_enrich(secondary_results, mode)

                    # Merge with existing context
                    state.current_context.extend(new_context)
                    self.logger.info(f"Added {len(new_context)} sources from secondary search (total: {len(state.current_context)})")

                    # Tracing: Secondary search context update
                    if tracer:
                        tracer.context_update(
                            "SECONDARY_SEARCH",
                            {
                                "queries_executed": response.new_queries,
                                "results_found": len(secondary_results),
                                "new_sources_added": len(new_context)
                            }
                        )

                    if enable_isolation:
                        # SEC-6: Only format NEW documents for LLM context
                        new_start_id = max(state.source_map.keys(), default=0) + 1
                        new_formatted, new_source_map = self._format_context_shared(
                            new_context, start_id=new_start_id
                        )
                        # SEC-6: Invariant check - new IDs must not collide with existing
                        overlap = set(new_source_map.keys()) & set(state.source_map.keys())
                        if overlap:
                            self.logger.error(f"SEC-6: source_map ID collision detected: {overlap}")
                        state.source_map.update(new_source_map)
                        pending_web = getattr(state, "pending_web_formatted", None)
                        if pending_web:
                            # 防線三 B-ISO：web 新 docs 也是本迭代新 docs，與站內新 docs 串接（SEC-6 語意擴充）
                            state.formatted_context = pending_web + "\n\n" + new_formatted
                            state.pending_web_formatted = None  # 消費後清空，避免下輪重複串接
                        else:
                            state.formatted_context = new_formatted  # Only new docs for Analyst
                        # SYNC: Keep self.* in sync for helper methods that read self.source_map
                        self.formatted_context = state.formatted_context
                        self.source_map = state.source_map
                        self.logger.info(
                            f"SEC-6: gap_search new_docs={len(new_context)}, "
                            f"new_start_id={new_start_id}, total_source_map={len(state.source_map)}"
                        )
                    else:
                        # Re-format unified context with updated citations
                        state.formatted_context, state.source_map = self._format_context_shared(state.current_context)
                        # SYNC: Keep self.* in sync for helper methods that read self.source_map
                        self.formatted_context = state.formatted_context
                        self.source_map = state.source_map

                    # Continue to next iteration (Analyst will retry with expanded context)
                    iteration += 1
                    # 防線三 B-LAST 孿生洞（land-review blocker）：本迭代 helper 補了 web docs
                    # （站內 secondary 有結果、web 也有料）時，若這是最後一輪，:963 iteration+=1 + :964
                    # continue 會在 :724 while 邊界退出 → 空 draft → :1307 SEARCH_REQUIRED → no-results
                    # 頁，把剛補的 web+站內源整個丟棄。與 :995 「站內查無」case 完全同構，退回一格保證
                    # 下一輪 Analyst pass 消費補源。
                    # 有界性同 :995：dedup（同 web query 下輪已在 web_searched_queries → helper no-op
                    #        → web_added 回 False → 不再退）+ run 級 cap → 額外 pass ≤ min(distinct web
                    #        query 數, cap)，天然有限。
                    if web_added:
                        iteration -= 1  # 退回一格：本輪 web pass 不計入 budget
                        self.logger.info(
                            "[WEB-GAP-RESOLVE][extra-pass] 站內 secondary 有結果且本輪補了 web docs，"
                            "退回一格保證一次 Analyst pass 消費（不在 :724 while 邊界丟棄補源）；"
                            "此 iteration 編號因 web extra-pass 退一格、與下一次正常 pass 共用編號"
                        )
                    continue
                else:
                    # No results found - force Analyst to work with existing data
                    self.logger.warning("Secondary search returned no results")

                    # RSN-5: Rebuild formatted_context fresh instead of accumulating hints
                    # Re-format from source of truth to avoid stale hints from previous iterations
                    # 防線三 B-ISO / SF1（修訂 2，Codex B3）：isolation 下若本輪 helper 已補 web docs，
                    # 不做全量重建（全量重建重新分配 id 會與引擎登記的 web marker id 脫節=組合重複）。
                    # 直接用 web 新 docs 當本迭代 formatted（SEC-6「只看本迭代新 docs」語意維持），
                    # source_map 保持引擎登記不再動；web 不重複、無雙套 id。
                    pending_web = getattr(state, "pending_web_formatted", None)
                    if getattr(state, "enable_isolation", False) and pending_web:
                        state.formatted_context = pending_web
                        state.pending_web_formatted = None  # SF1：消費即清
                        # source_map 不動（引擎已登記 web 區間；站內無新結果，無站內 docs 要重建）
                    else:
                        # non-isolation（或 isolation 但本輪無 web docs）：維持原全量重建，行為 byte-identical。
                        state.formatted_context, state.source_map = self._format_context_shared(state.current_context)
                    system_hint = "\n\n[系統提示] 針對缺口的補充搜尋未發現有效結果，請基於現有資訊推論。"
                    state.formatted_context += system_hint
                    # SYNC: Keep self.* in sync for helper methods
                    self.formatted_context = state.formatted_context
                    self.source_map = state.source_map

                    # Increment iteration and let it proceed to Critic evaluation
                    iteration += 1
                    # 防線三 B-LAST：本迭代 helper 補了 web docs（站內 secondary 查無但 web 有料）時，
                    # 不能在此邊界 error return 丟棄 web docs——退回一格保證下一輪 Analyst pass 消費。
                    # 有界性：dedup（同 web query 下輪已在 web_searched_queries → helper no-op → web_added 回 False）
                    #        + run 級 cap → 額外 pass ≤ min(distinct web query 數, cap)，天然有限。
                    if web_added:
                        iteration -= 1  # 退回一格：本輪 web pass 不計入 budget
                        self.logger.info(
                            "[WEB-GAP-RESOLVE][extra-pass] 站內 secondary 查無但本輪補了 web docs，"
                            "退回一格保證一次 Analyst pass 消費（不在 :794 邊界丟棄 web docs）；"
                            "此 iteration 編號因 web extra-pass 退一格、與下一次正常 pass 共用編號"
                        )
                        # 落下去（不 continue）：接著跑 empty-draft guard；draft 仍空 → :829 也有 web_added 保命 continue。
                    if iteration >= max_iterations:
                        self.logger.error("Max iterations reached after failed gap search")
                        # 防線四：final-pass 指示下仍拒絕產文 → 可觀測（未來調 prompt 措辭的驗偽依據，no silent）。
                        if final_pass:
                            self.logger.warning("[FINAL-PASS] analyst still refused (exit=stalled_secondary_max_iter)")
                        # Return error result
                        state.early_return = [{
                            "@type": "Item",
                            "url": "internal://error",
                            "name": "Deep Research 資料不足",
                            "site": "系統訊息",
                            "siteUrl": "internal",
                            "score": 0,
                            "description": (
                                f"**無法完成研究**\n\n"
                                f"原因：經過 {max_iterations} 次迭代後，仍然缺少關鍵資訊。\n\n"
                                f"**缺失的資訊**：\n" +
                                "\n".join(f"- {info}" for info in response.missing_information) +
                                f"\n\n**建議的補充搜尋**：\n" +
                                "\n".join(f"- {q}" for q in response.new_queries) +
                                "\n\n補充搜尋已執行但未找到相關結果。"
                            )
                        }]
                        await self._emit_phase_event("actor_critic_loop", "completed")
                        return state
                    # Do NOT continue - let it fall through to force Analyst to produce something
                    # (will be caught in next iteration with system hint)

            draft = response.draft

            # SEC-6: Track and validate citations
            if enable_isolation and response and hasattr(response, 'citations_used'):
                seen_citation_ids.update(response.citations_used)
                invalid_citations = [c for c in response.citations_used if c not in state.source_map]
                if invalid_citations:
                    self.logger.error(f"SEC-6: citations not in source_map: {invalid_citations}")

            # RSN-1: Guard against empty draft before proceeding to Critic
            if not draft or not draft.strip():
                # 防線三 B-LAST：本迭代補了 web docs 但 draft 空時，continue 讓下輪 Analyst 拿 web docs 重寫，
                # 不落原 empty-draft error 分支丟棄。有界性同 :794（dedup + cap）。
                if web_added:
                    self.logger.info(
                        "[WEB-GAP-RESOLVE][extra-pass] empty draft 但本輪補了 web docs，"
                        "continue 保證一次 Analyst pass 消費（不丟棄 web docs）"
                    )
                    continue
                self.logger.warning("Empty draft detected after gap resolution, cannot proceed to review")
                # If we still have iterations left, increment and retry
                if iteration + 1 < max_iterations:
                    self.logger.info("Retrying with next iteration due to empty draft")
                    iteration += 1
                    continue
                else:
                    # Max iterations exhausted with empty draft
                    self.logger.error("Max iterations reached with empty draft")
                    # 防線四：final-pass 指示下仍拒絕產文 → 可觀測（未來調 prompt 措辭的驗偽依據，no silent）。
                    if final_pass:
                        self.logger.warning("[FINAL-PASS] analyst still refused (exit=empty_draft_max_iter)")
                    state.early_return = self._format_error_result(
                        query,
                        "分析階段無法產生有效內容，請嘗試調整搜尋條件或使用不同的查詢。"
                    )
                    await self._emit_phase_event("actor_critic_loop", "completed")
                    return state

            # Stage 5: Process gap_resolutions for web search
            gap_resolution_added_data = False
            if hasattr(response, 'gap_resolutions') and response.gap_resolutions:
                self.logger.info("="*80)
                self.logger.info(f"[STAGE 5] GAP DETECTION TRIGGERED - Found {len(response.gap_resolutions)} gap resolutions")
                for i, gap in enumerate(response.gap_resolutions, 1):
                    self.logger.info(f"  Gap {i}: type={gap.gap_type}, resolution={gap.resolution}, reason={gap.reason}")
                self.logger.info("="*80)

                self._check_connection()  # Checkpoint 5: before gap resolutions
                context_before = len(state.current_context)
                await self._process_gap_resolutions(
                    response=response,
                    mode=mode,
                    current_context=state.current_context,
                    enable_web_search=enable_web_search,
                    tracer=tracer,
                    query_id=query_id,
                    web_searched_queries=web_searched_queries,  # 防線三 B-DEDUP-SCOPE：跨路徑共享 dedup
                )
                context_after = len(state.current_context)
                gap_resolution_added_data = context_after > context_before
            else:
                self.logger.warning("[STAGE 5] No gap_resolutions found (gap_resolutions is empty or missing)")

            # If new data was added, re-run Analyst to integrate it
            if gap_resolution_added_data:
                self.logger.info(f"Gap resolution added {context_after - context_before} items. Re-running Analyst to integrate new data.")

                await self._send_progress({
                    "message_type": "intermediate_result",
                    "stage": "analyst_integrating_new_data"
                })

                # Re-run Analyst with enriched context
                # Stage 5: Simplified tracer input (avoid logging full context)
                analyst_input = {
                    "query": query,
                    "context_count": len(state.current_context),
                    "mode": mode,
                    "enable_web_search": False  # Don't trigger another round of web search
                }

                self._check_connection()  # Checkpoint 6: before analyst re-run with enriched data

                if enable_isolation:
                    # SEC-6: Format only NEW context items for Analyst
                    new_items = state.current_context[context_before:]
                    new_start_id = max(state.source_map.keys(), default=0) + 1
                    formatted_context_enriched, new_source_map = self._format_context_shared(
                        new_items, start_id=new_start_id
                    )
                    # SEC-6: Invariant check before merge
                    overlap = set(new_source_map.keys()) & set(state.source_map.keys())
                    if overlap:
                        self.logger.error(f"SEC-6: enriched source_map ID collision: {overlap}")
                    state.source_map.update(new_source_map)
                    # SYNC: Keep self.* in sync for helper methods
                    self.source_map = state.source_map
                    previous_draft_for_analyst = draft  # Pass previous draft so Analyst knows prior analysis
                    self.logger.info(
                        f"SEC-6: enriched re-run new_docs={len(new_items)}, "
                        f"previous_draft_len={len(draft) if draft else 0}, "
                        f"total_source_map={len(state.source_map)}"
                    )
                else:
                    # Format context for re-analysis with enriched data.
                    # FIX (source tier Phase B 140ffb3a regression): 不可裸 doc.get()——
                    # secondary-search 經 no-op filter_and_enrich 後可能把 list/tuple
                    # 格式的 retriever item extend 進 current_context，裸 .get() 會
                    # AttributeError。改走 tuple-safe 的 _format_context_shared（與
                    # isolation 路徑、初次 analyst run、:709 既有非 isolation 先例一致）。
                    # 四值一起設：state.formatted_context 必須同步更新，否則下游
                    # 非 isolation Critic（:936-937 critic_context = state.formatted_context）
                    # 會 review 到 stale 的 enrich 前 context，與重建的 source_map 錯位。
                    state.formatted_context, state.source_map = self._format_context_shared(
                        state.current_context
                    )
                    self.formatted_context = state.formatted_context
                    self.source_map = state.source_map
                    formatted_context_enriched = state.formatted_context
                    previous_draft_for_analyst = None

                if tracer:
                    with tracer.agent_span("analyst", "research_with_enriched_data", analyst_input) as span:
                        response = await self.analyst.research(
                            query=query,
                            formatted_context=formatted_context_enriched,
                            mode=mode,
                            temporal_context=temporal_context,
                            enable_kg=enable_kg,
                            enable_web_search=False,  # Disable for re-analysis (already got data)
                            previous_draft=previous_draft_for_analyst,  # SEC-6
                            final_pass=final_pass  # 防線四：enriched re-run 最後一輪也強制寫稿
                        )
                        span.set_result(response)
                else:
                    response = await self.analyst.research(
                        query=query,
                        formatted_context=formatted_context_enriched,
                        mode=mode,
                        temporal_context=temporal_context,
                        enable_kg=enable_kg,
                        enable_web_search=False,  # Disable for re-analysis (already got data)
                        previous_draft=previous_draft_for_analyst,  # SEC-6
                        final_pass=final_pass  # 防線四：enriched re-run 最後一輪也強制寫稿
                    )

                draft = response.draft

                iteration_logger.log_agent_output(
                    iteration=iteration + 1,
                    agent_name="analyst_enriched",
                    input_prompt=f"Query: {query} (with {len(state.current_context)} enriched sources)",
                    output_response=response
                )

                # Graph fallback：記下 enriched 輪非空 graph 產出（此輪 LLM 可能省略 graph，
                # 空殼不會覆蓋 research 輪已記錄的版本——2026-07-15 rerun E2E 實證的蒸發點）
                track_nonempty_graphs(response, last_nonempty_graphs)

            # Send progress: Analyst complete
            await self._send_progress({
                "message_type": "intermediate_result",
                "stage": "analyst_complete",
                "citations_count": len(response.citations_used)
            })

            # Critic: Review draft
            # Send progress: Critic reviewing
            await self._send_progress({
                "message_type": "intermediate_result",
                "stage": "critic_reviewing"
            })

            self._check_connection()  # Checkpoint 7: before critic.review()
            self.logger.info("Critic reviewing draft")
            critic_input = {
                "draft": draft,
                "query": query,
                "mode": mode
            }

            # SEC-6: Build critic context (reference sheet or full context)
            if enable_isolation and hasattr(response, 'citations_used') and response.citations_used:
                ref_sheet = self._build_critic_reference_sheet(response.citations_used)
                iso_config = CONFIG.reasoning_params.get("agent_isolation", {})
                min_chars = iso_config.get("critic_reference_sheet_min_chars", 1000)
                min_citations = iso_config.get("critic_reference_sheet_min_citations", 2)
                if (len(ref_sheet) < min_chars
                        or len(response.citations_used) < min_citations):
                    # Rebuild full context from source of truth (self.formatted_context
                    # may only contain latest batch in isolation mode)
                    full_context, _ = self._format_context_shared(state.current_context)
                    self.logger.info(
                        f"SEC-6: Reference sheet too small "
                        f"(chars={len(ref_sheet)}<{min_chars} or "
                        f"citations={len(response.citations_used)}<{min_citations}), "
                        f"falling back to full context ({len(full_context)} chars)"
                    )
                    critic_context = full_context
                else:
                    full_len = len(state.formatted_context)
                    reduction = 1.0 - (len(ref_sheet) / full_len) if full_len > 0 else 0
                    self.logger.info(
                        f"SEC-6: critic ref_sheet={len(ref_sheet)} chars "
                        f"(full={full_len}, reduction={reduction:.0%})"
                    )
                    critic_context = ref_sheet
            else:
                critic_context = state.formatted_context

            if tracer:
                with tracer.agent_span("critic", "review", critic_input) as span:
                    review = await self.critic.review(
                        draft, query, mode,
                        analyst_output=response,
                        formatted_context=critic_context  # Phase 2 CoV / SEC-6
                    )
                    span.set_result(review)
            else:
                review = await self.critic.review(
                    draft, query, mode,
                    analyst_output=response,
                    formatted_context=critic_context  # Phase 2 CoV / SEC-6
                )

            iteration_logger.log_agent_output(
                iteration=iteration + 1,
                agent_name="critic",
                input_prompt=f"Draft: {draft[:100]}...",
                output_response=review
            )

            # Send progress: Critic complete
            await self._send_progress({
                "message_type": "intermediate_result",
                "stage": "critic_complete",
                "status": review.status
            })

            # Check convergence
            # Tracing: Convergence check
            if tracer:
                tracer.condition_branch(
                    "CONVERGENCE",
                    review.status,
                    {
                        "critique": review.critique[:200] + "..." if len(review.critique) > 200 else review.critique,
                        "suggestions": review.suggestions,
                        "mode_compliance": review.mode_compliance
                    }
                )

            if review.status in ["PASS", "WARN"]:
                self.logger.info(f"Convergence achieved: {review.status}")
                # Tracing: Iteration end
                if tracer:
                    tracer.end_iteration()
                break

            iteration += 1

        # === Post-loop: Write results back to state ===
        # Graph fallback：最終輪 analyst 輸出的 graph 欄位空殼/None、而本 run 前輪有非空版本
        # → 補回前輪版本（防 gap-enrichment/revise 輪 LLM 省略 graph 導致 KG/推論鏈整包蒸發）
        if response is not None and last_nonempty_graphs:
            response, _restored_graph_fields = apply_graph_fallback(response, last_nonempty_graphs)
            if _restored_graph_fields:
                self.logger.info(
                    f"[GRAPH-FALLBACK] 最終輪 analyst 輸出缺 {_restored_graph_fields}，"
                    f"已用本 run 前輪非空版本補回（enriched/revise 輪 LLM 省略 graph 的既有縫）")
        state.draft = draft
        state.review = review
        state.response = response
        state.iteration = iteration
        state.reject_count = reject_count
        state.seen_citation_ids = seen_citation_ids

        # Check if we have a valid draft
        if not draft:
            self.logger.error("No draft generated after iterations")
            # 防線四：final-pass 指示下仍拒絕產文 → 可觀測（未來調 prompt 措辭的驗偽依據，no silent）。
            # final_pass 迴圈內 local，迴圈結束後仍持有最後一輪值（配合迴圈前初始化，零 NameError）。
            if final_pass:
                self.logger.warning(
                    "[FINAL-PASS] analyst still refused (exit=post_loop_no_draft, status=%s)",
                    getattr(response, "status", "?"),
                )
            # Check if this was due to continuous SEARCH_REQUIRED without results
            if response and response.status == "SEARCH_REQUIRED":
                state.early_return = self._format_friendly_no_data_result(
                    query=query,
                    mode=mode,
                    missing_info=response.missing_information,
                    attempted_queries=response.new_queries,
                    reasoning_chain=response.reasoning_chain
                )
                await self._emit_phase_event("actor_critic_loop", "completed")
                return state
            # Otherwise, generic error (include reasoning if available)
            error_details = ""
            if response and hasattr(response, 'reasoning_chain') and response.reasoning_chain:
                error_details = f"\n\n**分析過程：**\n{response.reasoning_chain}"
            state.early_return = self._format_error_result(
                query,
                f"分析階段未能產生有效內容，請嘗試調整搜尋條件或使用不同的查詢。{error_details}"
            )
            await self._emit_phase_event("actor_critic_loop", "completed")
            return state

        # Graceful degradation check
        if reject_count >= max_iterations and review.status == "REJECT":
            self.logger.warning(
                f"Max iterations with continuous REJECTs ({reject_count}). "
                f"Degrading gracefully."
            )
            # Add warning to critique (Pydantic models are immutable by default)
            # We'll pass original review to Writer, which will handle REJECT status

        # SEC-6: Writer draft length monitoring
        if draft and enable_isolation:
            threshold = CONFIG.reasoning_params.get("agent_isolation", {}).get(
                "draft_length_warning_threshold", 20000
            )
            if len(draft) > threshold:
                self.logger.warning(
                    f"SEC-6: draft length {len(draft)} exceeds threshold {threshold}"
                )

        # SEC-5: Validate analyst citations against source_map
        raw_citations = response.citations_used
        valid_citations = [c for c in raw_citations if c in state.source_map]
        if len(valid_citations) < len(raw_citations):
            removed = set(raw_citations) - set(valid_citations)
            logger.warning(f"Removed phantom citations not in source_map: {removed}")
        state.analyst_citations = valid_citations

        await self._emit_phase_event("actor_critic_loop", "completed")
        return state

    def _build_writer_evidence_lookup(self, state) -> Dict[int, Dict[str, str]]:
        """SEC-5 白名單 ID → 真實來源明細，餵 Writer compose prompt（票 2026-07-28-f Fix 3）。

        範圍限 analyst_citations（Writer 白名單；給全池會誘導引用白名單外 id →
        被 :1724 guard 剝掉）。shape-aware 抽取復用 _extract_item_fields。
        """
        lookup: Dict[int, Dict[str, str]] = {}
        for cid in sorted(set(state.analyst_citations)):
            item = state.source_map.get(cid)
            if item is None:
                continue  # SEC-5 已濾 phantom；防禦性跳過
            title, source, description, url = self._extract_item_fields(item)
            date_published = self._extract_item_date(item)
            lookup[cid] = {
                "title": title,
                "site": source,
                "url": url,
                # 發布日期錨定（core.temporal_anchor）：writer 讀這份摘要寫報告，
                # 沒有發布日期就會拿全域「今天」去換算內文的「今年」。
                "snippet": annotate_relative_years((description or "")[:200], date_published),
                "date": str(date_published or ""),
            }
        return lookup

    async def _phase_writer(self, state: 'ResearchState') -> 'ResearchState':
        """
        Phase 3: Writer compose + Hallucination Guard.

        Contains: plan_and_write feature flag check, writer.compose(),
                  hallucination guard (set operation), progress messages.

        Reads: state.draft, state.review, state.response, state.analyst_citations,
               state.source_map, state.query, state.mode, state.iteration,
               state.max_iterations, state.tracer, state.iteration_logger
        Writes: state.final_report, state.plan

        Raises:
            ResearchCancelledError: If client disconnects (checkpoint 8)
        """
        await self._emit_phase_event("writer", "started")

        # Check if plan-and-write is enabled
        enable_plan_and_write = CONFIG.reasoning_params.get("features", {}).get(
            "plan_and_write", False
        )

        self._check_connection()  # Checkpoint 8: before writer phase

        plan = None
        if enable_plan_and_write:
            # Step 1: Plan
            await self._send_progress({
                "message_type": "intermediate_result",
                "stage": "writer_planning",
                "iteration": state.iteration + 1,
                "total_iterations": state.max_iterations
            })

            self.logger.info("Writer planning report structure")
            plan = await self.writer.plan(
                analyst_draft=state.draft,
                critic_review=state.review,
                user_query=state.query,
                target_length=2000
            )

            # Step 2: Compose
            await self._send_progress({
                "message_type": "intermediate_result",
                "stage": "writer_composing",
                "iteration": state.iteration + 1,
                "total_iterations": state.max_iterations
            })

            self.logger.info("Writer composing long-form report based on plan")
        else:
            # Standard single-step compose
            await self._send_progress({
                "message_type": "intermediate_result",
                "stage": "writer_composing"
            })

            self.logger.info("Writer composing final report")

        # Build citation lookup once: 供 compose prompt（Fix 3）+ logging 共用
        evidence_lookup = self._build_writer_evidence_lookup(state)
        citation_details = {
            cid: (
                f"{e['title'][:60]}... ({e['url'][:40]}...)" if e["url"] else e["title"][:60]
            )
            for cid, e in evidence_lookup.items()
        }

        writer_input = {
            "analyst_draft": state.draft[:200] + "...",  # Show preview
            "critic_review": state.review,
            "analyst_citations": state.analyst_citations,
            "citation_details": citation_details,  # Show actual source info
            "mode": state.mode,
            "user_query": state.query
        }

        if state.tracer:
            with state.tracer.agent_span("writer", "compose", writer_input) as span:
                final_report = await self.writer.compose(
                    analyst_draft=state.draft,
                    critic_review=state.review,
                    analyst_citations=state.analyst_citations,
                    mode=state.mode,
                    user_query=state.query,
                    plan=plan,  # Pass plan (None if not enabled)
                    evidence_lookup=evidence_lookup,
                )
                span.set_result(final_report)
        else:
            final_report = await self.writer.compose(
                analyst_draft=state.draft,
                critic_review=state.review,
                analyst_citations=state.analyst_citations,
                mode=state.mode,
                user_query=state.query,
                plan=plan,  # Pass plan (None if not enabled)
                evidence_lookup=evidence_lookup,
            )

        state.iteration_logger.log_agent_output(
            iteration=state.iteration + 1,
            agent_name="writer",
            input_prompt=f"Draft: {state.draft[:100]}...",
            output_response=final_report
        )

        # Send progress: Writer complete
        await self._send_progress({
            "message_type": "intermediate_result",
            "stage": "writer_complete"
        })

        # Hallucination Guard: Verify Writer sources subset of Analyst citations
        invalid_sources = []
        needs_correction = False
        if not set(final_report.sources_used).issubset(set(state.analyst_citations)):
            self.logger.error(
                f"Writer hallucination detected: {final_report.sources_used} "
                f"not subset of {state.analyst_citations}"
            )
            # Auto-correct: Only keep intersection (Pydantic models are immutable)
            corrected_sources = list(set(final_report.sources_used) & set(state.analyst_citations))
            invalid_sources = list(set(final_report.sources_used) - set(state.analyst_citations))
            needs_correction = True
            self.logger.warning(f"Corrected sources from {final_report.sources_used} to: {corrected_sources}")

            # Create corrected version (rebuild model with corrected data)
            final_report = WriterComposeOutput(
                final_report=final_report.final_report,
                sources_used=corrected_sources,
                confidence_level="Low",
                methodology_note=final_report.methodology_note + " [自動修正：移除未驗證來源]"
            )

        # Tracing: Hallucination guard
        if state.tracer:
            state.tracer.condition_branch(
                "HALLUCINATION_GUARD",
                "PASSED" if not needs_correction else "CORRECTED",
                {
                    "writer_sources": final_report.sources_used,
                    "analyst_sources": list(state.source_map.keys()),
                    "invalid_sources": invalid_sources if needs_correction else []
                }
            )

        state.final_report = final_report
        state.plan = plan

        await self._emit_phase_event("writer", "completed")
        return state

    async def _phase_format_result(self, state: 'ResearchState') -> 'ResearchState':
        """
        Phase 4: Session logging + Chain Analysis + Format NLWeb result.

        Contains: iteration_logger.log_summary(), reasoning chain analysis
                  (if argument_graph exists), RSN-4 verification_status transfer,
                  _format_result() call, tracing end.

        Reads: state.response, state.review, state.final_report, state.iteration,
               state.current_context, state.query, state.mode, state.tracer,
               state.iteration_logger, state.items
        Writes: state.chain_analysis, state.result
        """
        await self._emit_phase_event("format_result", "started")

        # Log session summary
        state.iteration_logger.log_summary(
            total_iterations=state.iteration + 1,
            final_status=state.review.status,
            mode=state.mode,
            metadata={
                "sources_analyzed": len(state.current_context),
                "sources_filtered": len(state.items) - len(state.current_context)
            }
        )

        # Phase 3.5: Analyze reasoning chain if argument_graph exists
        if hasattr(state.response, 'argument_graph') and state.response.argument_graph:
            from reasoning.utils.chain_analyzer import ReasoningChainAnalyzer

            self.logger.info("Analyzing reasoning chain for impact and critical nodes")

            # Get weaknesses from critic
            weaknesses = getattr(state.review, 'structured_weaknesses', None)

            # Analyze chain
            try:
                analyzer = ReasoningChainAnalyzer(state.response.argument_graph, weaknesses)
                chain_analysis = analyzer.analyze()

                # B9/C3: preserve runtime type (incl. a future Live response)
                # instead of narrowing to a fixed Enhanced/KG type. model_copy
                # keeps the subclass + all fields; only reasoning_chain_analysis
                # is updated. (Does NOT re-run validation — value is deterministic.)
                state.response = state.response.model_copy(
                    update={"reasoning_chain_analysis": chain_analysis}
                )

                state.chain_analysis = chain_analysis

                self.logger.info(
                    f"Chain analysis: {len(chain_analysis.critical_nodes)} critical nodes, "
                    f"max_depth={chain_analysis.max_depth}, "
                    f"logic_inconsistencies={chain_analysis.logic_inconsistencies}"
                )

                # Display in console tracer (Developer Mode in Terminal)
                if state.tracer:
                    state.tracer.reasoning_chain_analysis(state.response.argument_graph, chain_analysis)

            except Exception as e:
                self.logger.error(f"Failed to analyze reasoning chain: {e}", exc_info=True)

        # RSN-4: Transfer verification_status from critic review to final_report
        # critic.py sets these fields dynamically on review.__dict__ when CoV fails.
        # _format_result reads them from final_report.__dict__ to include in schema_obj.
        if state.review and state.review.__dict__.get("verification_status"):
            state.final_report.__dict__["verification_status"] = state.review.__dict__["verification_status"]
            state.final_report.__dict__["verification_message"] = state.review.__dict__.get(
                "verification_message", "本報告未經完整事實驗證"
            )
            self.logger.info(
                f"RSN-4: Transferred verification_status='{state.review.__dict__['verification_status']}' "
                "from critic review to final_report"
            )

        # Phase 4: Format as NLWeb result (pass context for source extraction)
        state.result = self._format_result(
            state.query, state.mode, state.final_report,
            state.iteration + 1, state.current_context,
            analyst_output=state.response
        )
        self.logger.info(f"Research completed: {state.iteration + 1} iterations")

        # Tracing: Research end
        # RSN-10: Safe access to tracer.start_time
        if state.tracer:
            start_time = getattr(state.tracer, 'start_time', None)
            if start_time is None:
                start_time = time.time()  # fallback to now
            total_time = time.time() - start_time
        else:
            total_time = 0
        if state.tracer:
            state.tracer.end_research(
                final_status=state.review.status,
                iterations=state.iteration + 1,
                total_time=total_time
            )

        await self._emit_phase_event("format_result", "completed")
        return state

    async def _filter_and_prepare_sources(
        self,
        items: List[Dict[str, Any]],
        mode: str,
        tracer,
    ) -> List[Dict[str, Any]]:
        """
        Prepare context sources for research.

        Source tier enrichment was removed (2026-06); this is now a pass-through
        that retains the empty-source guardrail.

        Returns:
            Prepared items list
        """
        # Phase 1: Prepare context (source tier enrichment removed; pass-through)
        current_context = self.source_filter.filter_and_enrich(items, mode)
        self.logger.info(f"Prepared context: {len(current_context)} sources (from {len(items)})")

        # Tracing: Context preparation
        if tracer:
            tracer.context_preparation(
                original_items=items,
                filtered_items=current_context,
                mode=mode
            )

        # §v5: empty current_context is NOT an error — it means upstream retrieval
        # returned 0 sources. Return empty so Phase 1b formats an empty source_map,
        # which the β-path (:598) picks up for zero-results web backfill. The old
        # `raise ValueError` here (plus the filter-layer raise removed in V5-1) was
        # a leftover guardrail from the removed source-tier mechanism that shadowed
        # the β-path, making §v1 dead code in the real 0-source path.
        if not current_context:
            self.logger.info(
                f"No prepared sources (original items: {len(items)}); "
                "flowing to β-path for zero-results web backfill."
            )

        return current_context

    async def _format_research_context(
        self,
        items: List[Dict[str, Any]],
        tracer,
    ) -> tuple[str, Dict[int, Dict[str, Any]]]:
        """
        Format items into citation context.

        Returns:
            Tuple of (formatted_context_string, source_id_map)
        """
        # Unified context formatting (Single Source of Truth)
        formatted_context, source_map = self._format_context_shared(items)

        # Tracing: Context formatted
        if tracer:
            tracer.context_formatted(
                source_map=source_map,
                formatted_context=formatted_context
            )

        return formatted_context, source_map

    async def run_research(
        self,
        query: str,
        mode: str,
        items: List[Dict[str, Any]],
        temporal_context: Optional[Dict[str, Any]] = None,
        enable_kg: bool = False,
        enable_web_search: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute deep research. Dispatches to composable or legacy pipeline based on feature flag.

        Args:
            query: User's research question
            mode: Research mode (strict, discovery, monitor)
            items: Retrieved items from search (pre-filtered by temporal range)
            temporal_context: Optional temporal information
            enable_kg: Enable knowledge graph generation (Phase KG, per-request override)
            enable_web_search: Enable web search for dynamic data (Stage 5)

        Returns:
            List of NLWeb Item dicts compatible with create_assistant_result().
        """
        # composable_pipeline flag 曾用於 gate Tasks 0-4 refactor；legacy 與 composable
        # 已 zero-behavior-change 收斂為同一實作（見原 _run_research_legacy docstring）。
        # config 顯式設 composable_pipeline: true → legacy 在 prod 不會被走到（dead-in-prod）。
        # 直呼 composable，flag 異常時發 warning 觀測 log（不再分支）。
        # I-1：下面 default 從現行 .get(..., False) 翻成 True 是 intentional 且等價 ——
        # config 已有此 key 並設 true，default 翻 True 只是「config 萬一遺失時也走 composable」，
        # 不改變現行（config=true）行為。
        # F1（不可 silent fail）：log 條件必須涵蓋「key 完全缺失」與「明確 false」兩種異常。
        # 若只寫 `if not ...get(..., True)`，key 缺失時 get 回 True → not True → False → 靜默，
        # 反而比舊 code（default=False、缺 key 走 legacy 發 warning）更安靜 → 違反不可 silent fail。
        # 故顯式檢查 key 是否在 features 中，缺失或 false 都發 warning。
        features = CONFIG.reasoning_params.get("features", {})
        if "composable_pipeline" not in features or not features.get("composable_pipeline", True):
            logger.warning(
                "run_research: composable_pipeline 未明確啟用（key 缺失或 false）— "
                "legacy 與 composable 已合一，直呼 composable"
            )
        return await self._run_research_composable(
            query, mode, items, temporal_context, enable_kg, enable_web_search
        )

    async def _run_research_composable(
        self,
        query: str,
        mode: str,
        items: List[Dict[str, Any]],
        temporal_context: Optional[Dict[str, Any]] = None,
        enable_kg: bool = False,
        enable_web_search: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Composable phase pipeline for deep research.

        Each phase reads/writes ResearchState, enabling:
        - Non-blocking execution (asyncio.create_task, Task 6)
        - Phase-level cache / timeout / monitoring
        - Selective re-run
        - Cancel at phase boundary

        Args:
            query: User's research question
            mode: Research mode (strict, discovery, monitor)
            items: Retrieved items from search (pre-filtered by temporal range)
            temporal_context: Optional temporal information
            enable_kg: Enable knowledge graph generation (Phase KG, per-request override)
            enable_web_search: Enable web search for dynamic data (Stage 5)

        Returns:
            List of NLWeb Item dicts compatible with create_assistant_result().
            Each dict contains: @type, url, name, site, siteUrl, score, description
        """
        # Setup: Initialize logging and tracing
        query_id = getattr(self.handler, 'query_id', f'reasoning_{hash(query)}')

        # Note: log_query_start() is now called in api.py before handler.runQuery()
        # to avoid FK violations. No need to call it again here.

        iteration_logger, tracer = self._setup_research_session(
            query_id=query_id,
            query=query,
            mode=mode,
            items=items,
            enable_web_search=enable_web_search,
        )

        # Initialize explicit research state
        state = ResearchState(
            query=query,
            mode=mode,
            items=items,
            temporal_context=temporal_context,
            enable_kg=enable_kg,
            enable_web_search=enable_web_search,
            query_id=query_id,
            iteration_logger=iteration_logger,
            tracer=tracer,
            max_iterations=CONFIG.reasoning_params.get("max_iterations", 3),
            enable_isolation=CONFIG.reasoning_params.get("features", {}).get("agent_isolation", False),
        )

        logger.info(f"[Orchestrator] ResearchState initialized: enable_kg={enable_kg}, enable_web_search={enable_web_search}, query_id={query_id}")

        try:
            # Phase 1: Filter and prepare sources
            state = await self._phase_filter_and_prepare(state)
            if state.early_return is not None:
                return state.early_return

            # Cache state for potential KG editing selective re-run
            _cache_research_state(state.query_id, state)
            # Bug 1：把精簡子集掛到 handler，供 runQuery 組 research_report 時併入持久化（DB fallback）
            if getattr(self, "handler", None) is not None:
                self.handler._rerun_state_subset = build_rerun_state_subset(state)

            # Phase 2: Actor-Critic Loop
            state = await self._phase_actor_critic_loop(state)
            if state.early_return is not None:
                return state.early_return

            # Phase 3: Writer + Hallucination Guard
            state = await self._phase_writer(state)

            # Phase 4: Session logging + Chain Analysis + Format Result
            state = await self._phase_format_result(state)
            return state.result

        except ResearchCancelledError:
            self.logger.info("Deep Research cancelled: client disconnected")
            print("[CANCEL] Deep Research cancelled - client disconnected, no further LLM calls")
            return []

        except NoValidSourcesError as e:
            self.logger.error(f"No valid sources after filtering: {e}")
            if tracer:
                tracer.error(f"No valid sources after filtering: {e}")
            return self._format_error_result(
                query,
                self._NO_VALID_SOURCES_MESSAGE
            )

        except (asyncio.TimeoutError, TimeoutError) as e:
            self.logger.error(f"Timeout error in orchestrator: {e}")
            if tracer:
                tracer.error(f"Research timeout: {str(e)}")
            return self._format_error_result(
                query,
                "研究請求超時，請稍後再試或縮小搜尋範圍。"
            )

        except (ConnectionError, OSError) as e:
            self.logger.error(f"Network error in orchestrator: {e}")
            if tracer:
                tracer.error(f"Network error: {str(e)}")
            return self._format_error_result(
                query,
                "網路連線發生問題，請檢查連線後再試。"
            )

        except Exception as e:
            self.logger.critical(f"Unexpected error in orchestrator: {e}", exc_info=True)
            if tracer:
                tracer.error(f"Research failed: {str(e)}", exception=e)
            # Re-raise in development/testing mode
            if CONFIG.should_raise_exceptions():
                raise
            sentry_sdk.capture_exception(e)
            return self._format_error_result(query, "系統發生未預期的錯誤，我們已記錄此問題，請稍後再試。")

    async def run_research_rerun(
        self,
        original_query_id: str,
        modified_query: str,
        restored_state: Optional[dict] = None,   # Bug 1: DB fallback 重建的 rerunState（cache miss 時）
    ) -> List[Dict[str, Any]]:
        """
        Selective re-run: skip search phase, reuse cached formatted_context.
        Used when user edits KG and wants re-analysis with same articles.

        Args:
            original_query_id: query_id of the original research whose cached state to reuse
            modified_query: the modified query with KG edit instructions appended
            restored_state: rerunState reconstructed from DB research_report
                (restore_rerun_state_from_report)。非 None 時作為記憶體 cache miss 的
                fallback（cache hit 仍優先走記憶體）。

        Returns:
            List of NLWeb Item dicts compatible with create_assistant_result().

        Raises:
            ValueError: if no cached state exists for the given query_id
        """
        # [R4 修訂 RC3] 深度防禦（縱深，defense in depth）：restored_state 是 DB fallback 來的，
        # 其 query_id 理應已由上游兩道（execute_rerun / api pre-check）驗過 == original_query_id。
        # 但即使上游驗證因故失效／被繞過，最內層再守一道——restored_state.query_id 與請求不符 → log +
        # 當無效（不用它、退回記憶體 cache，此時 miss → 落 raise ValueError → SSE error）。不張冠李戴。
        # 只驗 restored_state；記憶體 cache 用 query_id 當 key 天然對齊（G2 verified）、無此縫、不驗。
        if restored_state is not None and restored_state.get('query_id') != original_query_id:
            self.logger.warning(
                f"[RERUN] 深度防禦攔截：restored_state.query_id={restored_state.get('query_id')!r} "
                f"與請求 original_query_id={original_query_id!r} 不符（上游對齊驗證應已擋下，"
                f"此為縱深防線）→ 判無效、棄用 restored_state")
            restored_state = None
        # 記憶體 cache 優先（hit 走記憶體、快）；miss 才用 DB fallback 傳入的 restored_state
        cached = get_cached_research_state(original_query_id) or restored_state
        if not cached:
            raise ValueError(f"No cached research state for query_id={original_query_id}")

        # Create a new query_id for the rerun
        new_query_id = f"rerun_{original_query_id}_{abs(hash(modified_query)) % 10**8}"

        # Ensure new_query_id exists in queries table for FK references (analytics, tier_6, etc.)
        try:
            from core.query_logger import get_query_logger
            ql = get_query_logger()
            if ql:
                ql.log_query_start(
                    query_id=new_query_id,
                    user_id=getattr(self.handler, 'user_id', '') or '',
                    query_text=modified_query,
                    site=getattr(self.handler, 'site', 'all'),
                    mode='deep_research_rerun',
                )
        except Exception as e:
            self.logger.warning(f"Failed to log rerun query start (non-fatal): {e}")

        # Setup logging/tracing for the new session
        iteration_logger, tracer = self._setup_research_session(
            query_id=new_query_id,
            query=modified_query,
            mode=cached['mode'],
            items=cached['items'],
            enable_web_search=cached['enable_web_search'],
        )

        # Build new state with modified query, reusing cached context
        state = ResearchState(
            query=modified_query,
            mode=cached['mode'],
            items=cached['items'],
            temporal_context=cached['temporal_context'],
            enable_kg=cached['enable_kg'],
            enable_web_search=cached['enable_web_search'],
            query_id=new_query_id,
            iteration_logger=iteration_logger,
            tracer=tracer,
            max_iterations=CONFIG.reasoning_params.get("max_iterations", 3),
            enable_isolation=CONFIG.reasoning_params.get("features", {}).get("agent_isolation", False),
            is_rerun=True,
        )

        # Restore cached context from phase 1 (skip search)
        state.formatted_context = cached['formatted_context']
        state.source_map = cached['source_map']
        state.current_context = cached['current_context']

        self.logger.info(
            f"[RERUN] Starting selective re-run: original_query_id={original_query_id}, "
            f"new_query_id={new_query_id}, sources={len(state.source_map)}"
        )

        try:
            # Emit phase event for rerun start
            await self._emit_phase_event("rerun", "started")

            # Run phases 2-4 only (skip phase 1 search)
            state = await self._phase_actor_critic_loop(state)
            if state.early_return is not None:
                return state.early_return

            state = await self._phase_writer(state)

            state = await self._phase_format_result(state)
            return state.result

        except ResearchCancelledError:
            self.logger.info("Selective re-run cancelled: client disconnected")
            return []

        except (asyncio.TimeoutError, TimeoutError) as e:
            self.logger.error(f"Timeout error in rerun: {e}")
            if tracer:
                tracer.error(f"Rerun timeout: {str(e)}")
            return self._format_error_result(
                modified_query,
                "重新分析請求超時，請稍後再試。"
            )

        except (ConnectionError, OSError) as e:
            self.logger.error(f"Network error in rerun: {e}")
            if tracer:
                tracer.error(f"Network error: {str(e)}")
            return self._format_error_result(
                modified_query,
                "網路連線發生問題，請檢查連線後再試。"
            )

        except Exception as e:
            self.logger.critical(f"Unexpected error in rerun: {e}", exc_info=True)
            if tracer:
                tracer.error(f"Rerun failed: {str(e)}", exception=e)
            if CONFIG.should_raise_exceptions():
                raise
            sentry_sdk.capture_exception(e)
            return self._format_error_result(modified_query, "重新分析時發生未預期的錯誤，請稍後再試。")

    def _format_result(
        self,
        query: str,
        mode: str,
        final_report: Dict[str, Any],
        iterations: int,
        context: List[Any],
        analyst_output=None
    ) -> List[Dict[str, Any]]:
        """
        Format final report as NLWeb Item.

        Args:
            query: User's query
            mode: Research mode
            final_report: Final report from writer
            iterations: Number of iterations completed
            context: Source items used
            analyst_output: Optional analyst output with knowledge graph (Phase KG)

        Returns:
            List with single NLWeb Item dict

        ⚠️ CRITICAL: Must match schema expected by create_assistant_result()
        """
        # Convert source_map to URL array for frontend citation linking
        # Frontend expects: sources[0] = URL for [1], sources[1] = URL for [2], etc.
        # We build a complete array from citation ID 1 to max ID used
        source_urls = []
        writer_citations = final_report.sources_used  # List of citation IDs like [1, 4, 10, 18...]
        # Bug #25 Plan A: Extend max_cid to cover Writer's actual citations (even if out of source_map range)
        max_cid = max(
            max(self.source_map.keys(), default=0),
            max(writer_citations, default=0)
        )

        self.logger.info(f"Writer cited {len(writer_citations)} sources: {writer_citations}")
        self.logger.info(f"Building complete source URL array from 1 to {max_cid}")

        for cid in range(1, max_cid + 1):
            if cid in self.source_map:
                item = self.source_map[cid]
                # Handle both dict and tuple formats
                if isinstance(item, dict):
                    url = item.get("url") or item.get("link", "")
                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    url = item[0]  # First element is URL in tuple format
                else:
                    url = ""
                    self.logger.warning(f"Citation ID {cid} has invalid format: {type(item)}")

                # Add Chrome Text Fragment for paragraph-level deep linking
                if url and not url.startswith(("urn:", "private://")):
                    description = ""
                    if isinstance(item, dict):
                        description = item.get("description") or item.get("articleBody", "")
                    if description:
                        snippet = description.strip()[:80].strip()
                        if snippet:
                            url = f"{url}#:~:text={quote(snippet)}"

                source_urls.append(url if url else "")  # Keep empty string to maintain index alignment
            else:
                # Missing citation ID - maintain index alignment with empty string
                source_urls.append("")
                self.logger.warning(f"Citation ID {cid} missing in source_map")

        self.logger.info(f"Converted source_map ({len(self.source_map)} items) to {len(source_urls)} URLs for frontend")

        # Serialize knowledge graph if present (Phase KG)
        kg_json = None
        if analyst_output and hasattr(analyst_output, 'knowledge_graph') and analyst_output.knowledge_graph:
            kg = analyst_output.knowledge_graph
            kg_json = {
                "entities": [e.model_dump() for e in kg.entities],
                "relationships": [r.model_dump() for r in kg.relationships],
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "entity_count": len(kg.entities),
                    "relationship_count": len(kg.relationships)
                }
            }
            self.logger.info(f"Serialized knowledge graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships")

        # Build schema_object
        schema_obj = {
            "@type": "ResearchReport",
            "mode": mode,
            "iterations": iterations,
            "sources_used": source_urls,  # Now contains actual URLs instead of citation IDs
            "confidence": final_report.confidence_level,
            "methodology": final_report.methodology_note,
            "total_sources_analyzed": len(context)
        }

        # Add knowledge graph if available (Phase KG)
        if kg_json:
            schema_obj["knowledge_graph"] = kg_json

        # Add reasoning chain if available (Phase 4)
        if analyst_output and hasattr(analyst_output, 'argument_graph') and analyst_output.argument_graph:
            schema_obj["argument_graph"] = [node.model_dump() for node in analyst_output.argument_graph]

            if hasattr(analyst_output, 'reasoning_chain_analysis') and analyst_output.reasoning_chain_analysis:
                schema_obj["reasoning_chain_analysis"] = analyst_output.reasoning_chain_analysis.model_dump()

        # RSN-4: Add verification status if CoV failed (set dynamically on final_report by orchestrator)
        verification_status = final_report.__dict__.get("verification_status")
        if verification_status:
            schema_obj["verification_status"] = verification_status
            verification_message = final_report.__dict__.get("verification_message")
            if verification_message:
                schema_obj["verification_message"] = verification_message

        return [{
            "@type": "Item",
            "url": f"internal://system/{mode}/{query[:50]}",
            "name": f"深度研究報告：{query}",
            "site": "讀豹系統",
            "siteUrl": "internal://system",
            "score": 95,
            "description": final_report.final_report,
            "schema_object": schema_obj
        }]

    def _format_friendly_no_data_result(
        self,
        query: str,
        mode: str,
        missing_info: List[str],
        attempted_queries: List[str],
        reasoning_chain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Format a user-friendly response when no relevant data is found.

        Args:
            query: User's query
            mode: Research mode
            missing_info: List of missing information identified by Analyst
            attempted_queries: List of supplementary queries that were attempted
            reasoning_chain: Optional detailed reasoning from Analyst

        Returns:
            List with single NLWeb Item dict with friendly no-data message
        """
        # Build friendly description
        description_parts = [
            f"# 抱歉，目前找不到關於「{query}」的相關資料\n",
            f"## 搜尋說明\n",
            "我們已進行了深度搜尋，但資料庫中沒有找到符合條件的新聞或資料。\n"
        ]

        if missing_info:
            description_parts.append("\n### 缺少的關鍵資訊：\n")
            for info in missing_info:
                description_parts.append(f"- {info}\n")

        if attempted_queries:
            description_parts.append("\n### 已嘗試的補充搜尋：\n")
            for q in attempted_queries:
                description_parts.append(f"- `{q}`\n")

        # Add detailed reasoning if available (optional, for transparency)
        if reasoning_chain:
            description_parts.append("\n---\n")
            description_parts.append("\n<details>\n<summary>📊 詳細分析過程（點擊展開）</summary>\n\n")
            description_parts.append(reasoning_chain)
            description_parts.append("\n</details>\n")

        description_parts.extend([
            "\n---\n",
            "\n## 建議您可以：\n",
            "1. **調整關鍵字**：嘗試使用不同的詞彙或更廣泛的搜尋詞\n",
            "2. **擴大時間範圍**：如果您指定了特定日期，可以嘗試更寬的時間範圍\n",
            "3. **調整來源範圍**：在搜尋介面開啟更多新聞來源（或包含全部來源）\n",
            "4. **確認資料可用性**：有些資訊可能尚未被收錄到資料庫中\n",
            "\n有其他想了解的內容嗎？"
        ])

        return [{
            "@type": "Item",
            "url": "internal://system/no-data",
            "name": f"找不到相關資料：{query}",
            "site": "讀豹系統",
            "siteUrl": "internal://system",
            "score": 0,
            "description": "".join(description_parts),
            "schema_object": {
                "@type": "NoDataReport",
                "query": query,
                "mode": mode,
                "missing_information": missing_info,
                "attempted_queries": attempted_queries
            }
        }]

    def _format_error_result(
        self,
        query: str,
        error_message: str
    ) -> List[Dict[str, Any]]:
        """
        Format error as NLWeb Item.

        Args:
            query: User's query
            error_message: Error description

        Returns:
            List with single NLWeb Item dict containing error message
        """
        return [{
            "@type": "Item",
            "url": "internal://system/error",
            "name": f"研究錯誤：{query}",
            "site": "讀豹系統",
            "siteUrl": "internal://system",
            "score": 0,
            "description": f"## 錯誤\n\n{error_message}",
            "schema_object": {
                "@type": "ErrorReport",
                "error": error_message
            }
        }]

    async def _process_gap_resolutions(
        self,
        response: Any,
        mode: str,
        current_context: List[Dict[str, Any]],
        enable_web_search: bool,
        tracer: Any = None,
        query_id: str = None,
        web_searched_queries: Optional[set] = None,
    ) -> None:
        """
        Process gap_resolutions from Analyst output (Stage 5).

        Handles multiple types of gap resolution:
        1. LLM Knowledge: Creates virtual documents with URN
        2. Web Search: Executes Google search if enabled
        3. Internal Search: Uses existing vector DB (handled by main loop)
        4. Stock APIs: STOCK_TW (TWSE/TPEX), STOCK_GLOBAL (yfinance)
        5. Wikipedia: Direct Wikipedia API call

        Args:
            response: Analyst output with gap_resolutions
            mode: Research mode
            current_context: Current context list (modified in place)
            enable_web_search: Whether web search is enabled
            tracer: Optional console tracer
            query_id: Query ID for analytics logging
        """
        from reasoning.schemas_enhanced import GapResolutionType

        web_search_gaps = []
        llm_knowledge_items = []
        stock_tw_gaps = []
        stock_global_gaps = []
        wikipedia_gaps = []
        weather_tw_gaps = []
        weather_global_gaps = []
        company_tw_gaps = []
        company_global_gaps = []

        # R3 記帳集中：normalize / response 內部去重 / 跨路徑 dedup / cap / mark 全在下方收集迴圈。
        _seen_this_response: set = set()           # (2) response 內部去重（區域）
        # (4) run 級 cap（沿用 LR 鍵，source-agnostic）。
        # ★ 本 cap 僅限 distinct web-search queries；Wikipedia/stock/weather 等其他外部 API
        #   呼叫刻意不計入此 cap（parallel 補料策略下這些源另有各自的觸發條件，不共用 Google 額度）。
        _web_cap = CONFIG.reasoning_params.get("tier_6", {}).get("gap_routing", {}).get(
            "max_external_calls_per_run", 6
        )

        for gap in response.gap_resolutions:
            if gap.resolution == GapResolutionType.LLM_KNOWLEDGE:
                # Create virtual document for LLM knowledge
                topic = gap.topic or gap.gap_type.replace(" ", "_")
                urn = f"urn:llm:knowledge:{topic}"

                virtual_doc = {
                    "url": urn,
                    "title": f"AI 背景知識：{gap.gap_type}",
                    "site": "LLM Knowledge",
                    "description": f"[Tier 6 | llm_knowledge] {gap.llm_answer or ''}",
                    "_reasoning_metadata": {
                        "tier": 6,
                        "type": "llm_knowledge",
                        "original_source": "LLM Knowledge",
                        "gap_type": gap.gap_type,
                        "confidence": gap.confidence
                    }
                }
                llm_knowledge_items.append(virtual_doc)
                self.logger.info(f"Created LLM knowledge document: {urn}")

            elif gap.resolution == GapResolutionType.WEB_SEARCH:
                if enable_web_search and gap.search_query:
                    if web_searched_queries is None:
                        # v1 相容：不傳 set → 不 dedup/不 cap，byte-identical（v1 β-path 呼叫點）。
                        web_search_gaps.append(gap)
                    else:
                        key = _normalize_web_query(gap.search_query)
                        # (0) SF2（land-review should-fix，Codex）：whitespace-only search_query
                        #     normalize 成空 key → 在 dedup/cap/mark **之前** skip，避免空 query 佔 cap
                        #     slot 並被 mark（空 key 對 Google 無意義、mark '' 會白佔 distinct query 額度）。
                        if not key:
                            self.logger.info(
                                f"[WEB-GAP-RESOLVE] empty-after-normalize skip query={gap.search_query!r}"
                            )
                        # (2) response 內部去重：同一 response 內重複 query 只送一次（SF-3 ASCII key）。
                        elif key in _seen_this_response:
                            self.logger.info(f"[WEB-GAP-RESOLVE] intra_response_skip query={gap.search_query!r}")
                        # (3) 跨路徑 dedup：全 session 同 query 只打一次 Google（SF-3 ASCII key）。
                        elif key in web_searched_queries:
                            self.logger.info(f"[WEB-GAP-RESOLVE] dedup_skipped query={gap.search_query!r}（跨迭代/跨路徑已搜過）")
                        # (4) run 級 hard cap：達上限不再收新 distinct web query（bound Google 燒錢源，SF-3 ASCII key）。
                        #     len(web_searched_queries) 已含「prior + 本輪已 accept」（每 accept 同步 mark，見下）。
                        elif len(web_searched_queries) >= _web_cap:
                            self.logger.warning(
                                f"[WEB-GAP-RESOLVE] cap_skipped query={gap.search_query!r} "
                                f"distinct_web_query_cap={_web_cap}（cap 計 distinct web query 數，非外部 API 呼叫總數；"
                                f"parallel 策略下 Wikipedia 不另計）"
                            )
                        else:
                            web_search_gaps.append(gap)
                            _seen_this_response.add(key)
                            # (5) mark 時點 = 收集決定送出的當下（Codex B2：mark 在收集迴圈，不在被 mock 的 _execute_web_searches）。
                            #     accept 同步 mark → len(web_searched_queries) 即時反映本輪已 accept 數，供 (4) cap 判定。
                            web_searched_queries.add(key)
                elif gap.requires_web_search:
                    # Mark as needing web search but not enabled
                    self.logger.info(f"Web search required but not enabled for: {gap.search_query}")

            elif gap.resolution == GapResolutionType.STOCK_TW:
                stock_tw_gaps.append(gap)

            elif gap.resolution == GapResolutionType.STOCK_GLOBAL:
                stock_global_gaps.append(gap)

            elif gap.resolution == GapResolutionType.WIKIPEDIA:
                wikipedia_gaps.append(gap)

            elif gap.resolution == GapResolutionType.WEATHER_TW:
                weather_tw_gaps.append(gap)

            elif gap.resolution == GapResolutionType.WEATHER_GLOBAL:
                weather_global_gaps.append(gap)

            elif gap.resolution == GapResolutionType.COMPANY_TW:
                company_tw_gaps.append(gap)

            elif gap.resolution == GapResolutionType.COMPANY_GLOBAL:
                company_global_gaps.append(gap)

        # Add LLM knowledge items to context
        if llm_knowledge_items:
            current_context.extend(llm_knowledge_items)
            # Update source_map with new items
            start_idx = max(self.source_map.keys(), default=0) + 1
            for i, item in enumerate(llm_knowledge_items):
                self.source_map[start_idx + i] = item
            self.logger.info(f"Added {len(llm_knowledge_items)} LLM knowledge items to context")

        # Execute stock API calls
        if stock_tw_gaps:
            await self._execute_stock_tw_searches(stock_tw_gaps, current_context, tracer, query_id)

        if stock_global_gaps:
            await self._execute_stock_global_searches(stock_global_gaps, current_context, tracer, query_id)

        # Execute weather API calls
        if weather_tw_gaps:
            await self._execute_weather_tw_searches(weather_tw_gaps, current_context, tracer, query_id)

        if weather_global_gaps:
            await self._execute_weather_global_searches(weather_global_gaps, current_context, tracer, query_id)

        # Execute company API calls
        if company_tw_gaps:
            await self._execute_company_tw_searches(company_tw_gaps, current_context, tracer, query_id)

        if company_global_gaps:
            await self._execute_company_global_searches(company_global_gaps, current_context, tracer, query_id)

        # Execute Wikipedia searches
        if wikipedia_gaps:
            await self._execute_wikipedia_searches(wikipedia_gaps, current_context, tracer, query_id)

        # Execute web searches in parallel if enabled
        if web_search_gaps and enable_web_search:
            await self._execute_web_searches(web_search_gaps, mode, current_context, tracer, query_id)

    async def _execute_web_searches(
        self,
        gaps: List[Any],
        mode: str,
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """
        Execute web searches for gap resolutions using multiple Tier 6 sources.

        Args:
            gaps: List of GapResolution objects requiring web search
            mode: Research mode
            current_context: Current context list (modified in place)
            tracer: Optional console tracer
            query_id: Query ID for analytics logging
        """
        import asyncio

        # Get configuration
        tier_6_config = CONFIG.reasoning_params.get("tier_6", {})
        web_config = tier_6_config.get("web_search", {})
        wiki_config = tier_6_config.get("wikipedia", {})
        max_results = web_config.get("max_results", 5)
        enrichment_strategy = tier_6_config.get("enrichment_strategy", "parallel")

        self.logger.info(f"Executing {len(gaps)} web searches (strategy={enrichment_strategy})")

        # Send progress
        await self._send_progress({
            "message_type": "intermediate_result",
            "stage": "web_search_started",
            "queries": [g.search_query for g in gaps]
        })

        try:
            # Initialize Google Search client
            from retrieval_providers.google_search_client import GoogleSearchClient
            google_client = GoogleSearchClient()

            # Initialize Wikipedia client if enabled
            wiki_client = None
            if wiki_config.get("enabled", False):
                try:
                    from retrieval_providers.wikipedia_client import WikipediaClient
                    wiki_client = WikipediaClient()
                    if not wiki_client.is_available():
                        wiki_client = None
                        self.logger.debug("Wikipedia client disabled or library not installed")
                except ImportError:
                    self.logger.debug("Wikipedia library not installed")

            # Build search tasks
            search_tasks = []
            for gap in gaps:
                if gap.search_query:
                    # Google Search task
                    google_task = google_client.search_all_sites(
                        query=gap.search_query,
                        num_results=max_results,
                        query_id=query_id
                    )
                    search_tasks.append(("google", gap, google_task))

                    # Wikipedia task (parallel strategy)
                    if wiki_client and enrichment_strategy == "parallel":
                        wiki_task = wiki_client.search(
                            query=gap.search_query,
                            query_id=query_id
                        )
                        search_tasks.append(("wikipedia", gap, wiki_task))

            # Gather results
            all_results = []
            google_count = 0
            wiki_count = 0

            for source_type, gap, task in search_tasks:
                try:
                    results = await task

                    if source_type == "google":
                        # Process Google results (tuple format)
                        for result in results:
                            if isinstance(result, (list, tuple)) and len(result) >= 4:
                                schema_json = result[1] if len(result) > 1 else "{}"
                                try:
                                    schema_obj = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
                                except json.JSONDecodeError:
                                    schema_obj = {}

                                web_doc = {
                                    "url": result[0],
                                    "title": result[2] if len(result) > 2 else "Web Result",
                                    "site": result[3] if len(result) > 3 else "Web",
                                    "description": f"[Tier 6 | web_reference] {schema_obj.get('description', '')}",
                                    "_reasoning_metadata": {
                                        "tier": 6,
                                        "type": "web_reference",
                                        "original_source": result[3] if len(result) > 3 else "Web",
                                        "gap_query": gap.search_query
                                    }
                                }
                                all_results.append(web_doc)
                                google_count += 1

                    elif source_type == "wikipedia":
                        # Process Wikipedia results (dict format)
                        for result in results:
                            if isinstance(result, dict):
                                wiki_doc = {
                                    "url": result.get("link", ""),
                                    "title": result.get("title", "Wikipedia"),
                                    "site": "Wikipedia",
                                    "description": f"[Tier 6 | encyclopedia] {result.get('snippet', '')}",
                                    "_reasoning_metadata": {
                                        "tier": 6,
                                        "type": "encyclopedia",
                                        "original_source": "Wikipedia",
                                        "gap_query": gap.search_query
                                    }
                                }
                                all_results.append(wiki_doc)
                                wiki_count += 1

                    self.logger.info(f"{source_type} search for '{gap.search_query}': {len(results)} results")

                except Exception as e:
                    self.logger.error(f"{source_type} search failed for '{gap.search_query}': {e}")

            # Sequential fallback: Try Wikipedia if Google returned few results
            if wiki_client and enrichment_strategy == "sequential" and google_count < 3:
                self.logger.info("Sequential fallback: trying Wikipedia for additional context")
                for gap in gaps:
                    if gap.search_query:
                        try:
                            wiki_results = await wiki_client.search(
                                query=gap.search_query,
                                query_id=query_id
                            )
                            for result in wiki_results:
                                if isinstance(result, dict):
                                    wiki_doc = {
                                        "url": result.get("link", ""),
                                        "title": result.get("title", "Wikipedia"),
                                        "site": "Wikipedia",
                                        "description": f"[Tier 6 | encyclopedia] {result.get('snippet', '')}",
                                        "_reasoning_metadata": {
                                            "tier": 6,
                                            "type": "encyclopedia",
                                            "original_source": "Wikipedia",
                                            "gap_query": gap.search_query
                                        }
                                    }
                                    all_results.append(wiki_doc)
                                    wiki_count += 1
                        except Exception as e:
                            self.logger.error(f"Wikipedia fallback failed: {e}")

            # Add to context
            if all_results:
                current_context.extend(all_results)
                # Update source_map
                start_idx = max(self.source_map.keys(), default=0) + 1
                for i, item in enumerate(all_results):
                    self.source_map[start_idx + i] = item
                self.logger.info(f"Added {len(all_results)} Tier 6 results (Google: {google_count}, Wikipedia: {wiki_count})")

                # Tracing
                if tracer:
                    tracer.context_update(
                        "WEB_SEARCH",
                        {
                            "queries_executed": [g.search_query for g in gaps],
                            "results_found": len(all_results),
                            "google_count": google_count,
                            "wikipedia_count": wiki_count
                        }
                    )

        except ImportError:
            self.logger.warning("Google Search client not available, skipping web search")
        except Exception as e:
            self.logger.error(f"Web search execution failed: {e}")

    async def _execute_api_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        client_module: str,
        client_class: str,
        tracer_label: str,
        log_name: str,
        extract_param,
        search_fn,
        tracer_detail_fn,
        tracer: Any = None,
        query_id: str = None,
        transform_fn=None,
    ) -> None:
        """
        Generic API search executor for gap resolutions.

        Shared logic for all Tier 6 API searches: import client, check availability,
        loop gaps to collect results, update source_map and context, trace.

        Args:
            gaps: List of GapResolution objects
            current_context: Current context list (modified in place)
            client_module: Module path for dynamic import (e.g., "retrieval_providers.twse_client")
            client_class: Class name to instantiate (e.g., "TwseClient")
            tracer_label: Label for tracer.context_update (e.g., "STOCK_TW")
            log_name: Human-readable name for log messages (e.g., "TWSE")
            extract_param: Callable(gap) -> param value or None
            search_fn: Callable(client, param, query_id) -> awaitable results
            tracer_detail_fn: Callable(gaps) -> dict of tracer detail fields
            tracer: Optional console tracer
            query_id: Query ID for analytics logging
            transform_fn: Optional callable(results, gap) -> transformed results list.
                          If None, results are used as-is.
        """
        try:
            import importlib
            mod = importlib.import_module(client_module)
            cls = getattr(mod, client_class)
            client = cls()

            if not client.is_available():
                self.logger.debug(f"{log_name} client not enabled or not available")
                return

            all_results = []
            for gap in gaps:
                param = extract_param(gap)
                if param:
                    try:
                        results = await search_fn(client, param, query_id)
                        if transform_fn:
                            all_results.extend(transform_fn(results, gap))
                        else:
                            all_results.extend(results)
                        self.logger.info(f"{log_name} search for '{param}': {len(results)} results")
                    except Exception as e:
                        self.logger.error(f"{log_name} search failed for '{param}': {e}")

            # Add to context and update source_map
            if all_results:
                current_context.extend(all_results)
                start_idx = max(self.source_map.keys(), default=0) + 1
                for i, item in enumerate(all_results):
                    self.source_map[start_idx + i] = item
                self.logger.info(f"Added {len(all_results)} {log_name} results")

                if tracer:
                    details = tracer_detail_fn(gaps)
                    details["results_found"] = len(all_results)
                    tracer.context_update(tracer_label, details)

        except ImportError:
            self.logger.debug(f"{log_name} client not available")
        except Exception as e:
            self.logger.error(f"{log_name} search failed: {e}")

    async def _execute_stock_tw_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """Execute Taiwan stock API calls for gap resolutions."""
        import re

        def extract_param(gap):
            symbol = gap.api_params.get("symbol") if gap.api_params else None
            if not symbol and gap.search_query:
                match = re.search(r'\b(\d{4,5})\b', gap.search_query)
                if match:
                    symbol = match.group(1)
            return symbol

        await self._execute_api_searches(
            gaps=gaps,
            current_context=current_context,
            client_module="retrieval_providers.twse_client",
            client_class="TwseClient",
            tracer_label="STOCK_TW",
            log_name="TWSE",
            extract_param=extract_param,
            search_fn=lambda client, param, qid: client.search(param, query_id=qid),
            tracer_detail_fn=lambda gs: {"symbols_queried": [g.api_params.get("symbol") if g.api_params else None for g in gs]},
            tracer=tracer,
            query_id=query_id,
        )

    async def _execute_stock_global_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """Execute global stock API calls via yfinance for gap resolutions."""
        import re

        def extract_param(gap):
            symbol = gap.api_params.get("symbol") if gap.api_params else None
            if not symbol and gap.search_query:
                match = re.search(r'\b([A-Z]{1,5})\b', gap.search_query.upper())
                if match:
                    symbol = match.group(1)
            return symbol

        await self._execute_api_searches(
            gaps=gaps,
            current_context=current_context,
            client_module="retrieval_providers.yfinance_client",
            client_class="YfinanceClient",
            tracer_label="STOCK_GLOBAL",
            log_name="yFinance",
            extract_param=extract_param,
            search_fn=lambda client, param, qid: client.search(param, query_id=qid),
            tracer_detail_fn=lambda gs: {"symbols_queried": [g.api_params.get("symbol") if g.api_params else None for g in gs]},
            tracer=tracer,
            query_id=query_id,
        )

    async def _execute_wikipedia_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """Execute Wikipedia API calls for gap resolutions."""

        def extract_param(gap):
            return gap.search_query or (gap.api_params.get("query") if gap.api_params else None)

        def transform_fn(results, gap):
            """Convert Wikipedia results to standard Tier 6 document format."""
            docs = []
            for result in results:
                if isinstance(result, dict):
                    docs.append({
                        "url": result.get("link", ""),
                        "title": result.get("title", "Wikipedia"),
                        "site": "Wikipedia",
                        "description": f"[Tier 6 | encyclopedia] {result.get('snippet', '')}",
                        "_reasoning_metadata": {
                            "tier": 6,
                            "type": "encyclopedia",
                            "original_source": "Wikipedia",
                            "gap_query": extract_param(gap)
                        }
                    })
            return docs

        await self._execute_api_searches(
            gaps=gaps,
            current_context=current_context,
            client_module="retrieval_providers.wikipedia_client",
            client_class="WikipediaClient",
            tracer_label="WIKIPEDIA",
            log_name="Wikipedia",
            extract_param=extract_param,
            search_fn=lambda client, param, qid: client.search(param, query_id=qid),
            tracer_detail_fn=lambda gs: {"queries_executed": [g.search_query for g in gs if g.search_query]},
            tracer=tracer,
            query_id=query_id,
            transform_fn=transform_fn,
        )

    async def _execute_weather_tw_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """Execute Taiwan weather API calls for gap resolutions."""

        def extract_param(gap):
            location = gap.api_params.get("location") if gap.api_params else None
            if not location and gap.search_query:
                location = gap.search_query
            return location

        await self._execute_api_searches(
            gaps=gaps,
            current_context=current_context,
            client_module="retrieval_providers.cwb_weather_client",
            client_class="CwbWeatherClient",
            tracer_label="WEATHER_TW",
            log_name="CWB Weather",
            extract_param=extract_param,
            search_fn=lambda client, param, qid: client.search(param, query_id=qid),
            tracer_detail_fn=lambda gs: {"locations_queried": [g.api_params.get("location") if g.api_params else g.search_query for g in gs]},
            tracer=tracer,
            query_id=query_id,
        )

    async def _execute_weather_global_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """Execute global weather API calls via OpenWeatherMap for gap resolutions."""

        def extract_param(gap):
            city = gap.api_params.get("city") if gap.api_params else None
            if not city and gap.search_query:
                city = gap.search_query
            return city

        await self._execute_api_searches(
            gaps=gaps,
            current_context=current_context,
            client_module="retrieval_providers.global_weather_client",
            client_class="GlobalWeatherClient",
            tracer_label="WEATHER_GLOBAL",
            log_name="Global Weather",
            extract_param=extract_param,
            search_fn=lambda client, param, qid: client.search(param, query_id=qid),
            tracer_detail_fn=lambda gs: {"cities_queried": [g.api_params.get("city") if g.api_params else g.search_query for g in gs]},
            tracer=tracer,
            query_id=query_id,
        )

    async def _execute_company_tw_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """Execute Taiwan company registration API calls for gap resolutions."""

        def extract_param(gap):
            query = None
            if gap.api_params:
                query = gap.api_params.get("name") or gap.api_params.get("ubn")
            if not query and gap.search_query:
                query = gap.search_query
            return query

        await self._execute_api_searches(
            gaps=gaps,
            current_context=current_context,
            client_module="retrieval_providers.tw_company_client",
            client_class="TwCompanyClient",
            tracer_label="COMPANY_TW",
            log_name="TW Company",
            extract_param=extract_param,
            search_fn=lambda client, param, qid: client.search(param, query_id=qid),
            tracer_detail_fn=lambda gs: {"queries_executed": [g.api_params.get("name") if g.api_params else g.search_query for g in gs]},
            tracer=tracer,
            query_id=query_id,
        )

    async def _execute_company_global_searches(
        self,
        gaps: List[Any],
        current_context: List[Dict[str, Any]],
        tracer: Any = None,
        query_id: str = None
    ) -> None:
        """Execute global company/entity API calls via Wikidata for gap resolutions."""

        # entity_type varies per gap (gap.api_params.get("type", "company")), so we
        # carry it alongside the name inside the extracted param tuple. _execute_api_searches
        # calls extract_param then search_fn once per gap in order, so each gap's entity_type
        # travels with its own name — no cross-gap state, no name-keyed collision. Returns
        # None when there is no name so the executor's `if param` skip is preserved.
        def extract_param(gap):
            name = gap.api_params.get("name") if gap.api_params else None
            if not name and gap.search_query:
                name = gap.search_query
            if not name:
                return None
            entity_type = gap.api_params.get("type", "company") if gap.api_params else "company"
            return (name, entity_type)

        async def search_with_entity_type(client, param, qid):
            name, entity_type = param
            return await client.search(name, entity_type=entity_type, query_id=qid)

        await self._execute_api_searches(
            gaps=gaps,
            current_context=current_context,
            client_module="retrieval_providers.wikidata_client",
            client_class="WikidataClient",
            tracer_label="COMPANY_GLOBAL",
            log_name="Wikidata",
            extract_param=extract_param,
            search_fn=search_with_entity_type,
            tracer_detail_fn=lambda gs: {"queries_executed": [g.api_params.get("name") if g.api_params else g.search_query for g in gs]},
            tracer=tracer,
            query_id=query_id,
        )
