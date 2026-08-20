# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file contains the code for the ranking stage.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

from core.utils.json_utils import trim_json
from core.temporal_anchor import annotate_relative_years
from core.llm import ask_llm
from core.prompts import find_prompt, fill_prompt
from core.schemas import create_assistant_result, Message, SenderType
from core.config import CONFIG
from misc.logger.logging_config_helper import get_configured_logger
import asyncio
import json
import sentry_sdk
from typing import Dict

# Analytics logging
from core.query_logger import get_query_logger

logger = get_configured_logger("ranking_engine")


def _item_date_published(json_str) -> str:
    """從 schema_json 抽 datePublished（發布日期錨定用）；抽不到回空字串（不猜年份）。"""
    try:
        schema_obj = json.loads(json_str) if isinstance(json_str, str) else (json_str or {})
        return str(schema_obj.get("datePublished", "") or "")
    except Exception as e:
        # 不可 silent fail：抽不到 → 該筆不做年份錨定，明示降級
        logger.warning(f"_item_date_published: failed to parse datePublished: {e}")
        return ""


def dedup_by_title_and_source(results: list) -> list:
    """
    Remove duplicate results that share the same (title, source) pair.

    Chinatimes (and similar publishers) store the same article under multiple
    URLs with different category codes (e.g. -260402 vs -260405).  The
    retrieval-layer URL dedup cannot catch this because the URL strings differ,
    so the same article can appear 2-3 times in ranked output.

    Strategy:
    - Key: (result['name'], result['site'])
    - Keep only the result with the highest ranking score for each key.
    - Articles from *different* sources with the same title are NOT merged
      (they may represent different editorial perspectives).

    Should be called after LLM scoring + sort, before MMR.
    """
    seen: dict = {}
    for result in results:
        key = (result.get('name', ''), result.get('site', ''))
        # R2：比較過 _safe_score——helper 是模組級公開面，輸入不保證已過 do() filter
        # （殘值已濾）的順序不變式；殘值（'70分'）裸比較會 TypeError。降位 0 參與比較。
        if key not in seen or _safe_score(result) > _safe_score(seen[key]):
            seen[key] = result

    deduplicated = list(seen.values())
    removed = len(results) - len(deduplicated)
    if removed > 0:
        logger.info(f"Title dedup: removed {removed} duplicates")
    return deduplicated


def _safe_score(result: dict):
    """安全提取 ranking score 為數值，供所有比較/排序/輸出點使用（defense-in-depth）。

    根解在 core/llm.ask_llm 的 coerce 收斂點——字串分數在流入 ranking 前已 int/float 化，
    正常情況此處恆拿到數值。但 coerce 對『真正轉不動』的字串（'70分'）會保留原值 + warning；
    這些殘值一旦流到裸讀比較點會 TypeError：rankItem :195（try 內 → 單件丟 = AF-1 症狀）、
    do() filter/sort（try 外 → **整批 query 的 ranking 全滅**）。此 helper 把非數值 score
    視為 0（排到底/不 early-send）+ log，讓單件退位而非崩——不是替代根解，是最後一道網。

    R1 一致化：rankItem/shouldSend/sendAnswers/do() 的所有 score 讀取點統一過此函式，
    殘值語義全鏈一致（降位 0、item 保留、description 還在）。回傳**原樣數值**（int 保持
    int，不強制 float）——sendAnswers 輸出 JSON 的 score 型別不因防線漂移。
    full-scan CORE-2 + R1 #1。
    """
    score = result.get('ranking', {}).get('score', 0)
    if isinstance(score, bool):  # bool 是 int 子類但非有效分數
        return 0
    if isinstance(score, (int, float)):
        return score
    # 走到這裡代表 coerce 收斂點已保留了非數值字串（罕見）——不讓它崩/丟件。
    logger.warning(
        "Non-numeric ranking score %r for '%s' survived coercion; "
        "treating as 0 in compare/sort (single item demoted, batch protected).",
        score, result.get('name', 'unknown'),
    )
    return 0


class Ranking:
     
    EARLY_SEND_THRESHOLD = 59
    NUM_RESULTS_TO_SEND = 10

    REGULAR_TRACK = 2
    CONVERSATION_SEARCH = 3

    # This is the default ranking prompt, in case, for some reason, we can't find the site_type.xml file.
    RANKING_PROMPT = ["""針對以下 {site.itemType}，評估與使用者提問的相關程度，給予 0-100 分。
若分數高於 50，撰寫一段與使用者提問相關的簡短描述，不提及使用者問題本身。
使用者提問：{request.query}
項目描述：{item.description}""",
    {"score" : "0-100 整數",
     "description" : "項目簡短描述"}]

    RANKING_PROMPT_NAME = "RankingPrompt"
     
    def get_ranking_prompt(self):
        site = self.handler.site
        item_type = self.handler.item_type

        # Check for custom prompts in prompts.xml
        prompt_str, ans_struc = find_prompt(site, item_type, self.RANKING_PROMPT_NAME)
        if prompt_str is None:
            logger.debug("Using default ranking prompt")
            return self.RANKING_PROMPT[0], self.RANKING_PROMPT[1]
        else:
            logger.debug(f"Using custom ranking prompt for site: {site}, item_type: {item_type}")
            return prompt_str, ans_struc
        
    def __init__(self, handler, items, ranking_type=REGULAR_TRACK, level="low"):
        ll = len(items)
        if ranking_type == self.REGULAR_TRACK:
            self.ranking_type_str = "REGULAR_TRACK"
        elif ranking_type == self.CONVERSATION_SEARCH:
            self.ranking_type_str = "CONVERSATION_SEARCH"
        else:
            self.ranking_type_str = "UNKNOWN"
        logger.info(f"Initializing Ranking with {ll} items, type: {self.ranking_type_str}")
        self.handler = handler
        self.level = level
        self.items = items
        self.num_results_sent = 0
        self.rankedAnswers = []
        self.ranking_type = ranking_type
        self._sent_title_keys = set()  # Track (name, site) to prevent sending duplicates

    async def rankItem(self, item):
        name = "unknown"
        try:
            # Handle Dict format (new) or Tuple format (legacy)
            if isinstance(item, dict):
                url = item.get('url', '')
                json_str = item.get('schema_json', '')
                name = item.get('title', '')
                site = item.get('site', '')
                retrieval_scores = item.get('retrieval_scores', {})
                vector = item.get('vector')
            elif len(item) >= 6:
                # 6-tuple: [url, json_str, name, site, vector_or_None, retrieval_scores]
                # This is the core read point that feeds non-zero retrieval
                # features (index 14-18) to the XGBoost shadow ranker.
                url, json_str, name, site, vector, retrieval_scores = (
                    item[0], item[1], item[2], item[3], item[4], item[5]
                )
            elif len(item) == 5:
                url, json_str, name, site, vector = item
                retrieval_scores = {}  # Legacy format doesn't have retrieval scores
            else:
                url, json_str, name, site = item
                retrieval_scores = {}
                vector = None

            prompt_str, ans_struc = self.get_ranking_prompt()
            description = trim_json(json_str)
            # 發布日期錨定（core.temporal_anchor）：ranking 產出的 description 就是卡片摘要，
            # 也是 summarize 的素材。內文「今年」以該報導發布年換算，年份由 code 標好，
            # 免得 LLM 拿 prompt 另一端的「今天」推成當前年。
            # str(): trim_json 回 dict，而 fill_prompt 最後也是 str(value) → 內容等價
            description = annotate_relative_years(str(description), _item_date_published(json_str))
            prompt = fill_prompt(prompt_str, self.handler, {"item.description": description})
            ranking = await ask_llm(prompt, ans_struc, level=self.level, query_params=self.handler.query_params)

            if not ranking or not isinstance(ranking, dict):
                logger.error(f"LLM returned empty response for ranking: {name}")
                sentry_sdk.capture_message(f"LLM returned empty in ranking.rankItem for: {name}")
                ranking = {"score": 0, "description": "LLM ranking failed", "final_score": 0}

            # Handle both string and dictionary inputs for json_str
            schema_object = json_str if isinstance(json_str, dict) else json.loads(json_str)

            # If schema_object is an array, set it to the first item
            if isinstance(schema_object, list) and len(schema_object) > 0:
                schema_object = schema_object[0]

            ansr = {
                'url': url,
                'site': site,
                'name': name,
                'ranking': ranking,
                'schema_object': schema_object,
                'sent': False,
                'retrieval_scores': retrieval_scores,  # Preserve retrieval scores for XGBoost
            }

            # Add vector if available (for MMR)
            if vector is not None:
                ansr['vector'] = vector
            
            # Check if required_item_type is specified and filter based on @type
            if self.handler.required_item_type is not None:
                item_type = schema_object.get('@type', None)
                if item_type != self.handler.required_item_type:
                    logger.debug(f"Item type mismatch: expected {self.handler.required_item_type}, got {item_type} - setting score to 0")
                    ranking["score"] = 0
            
            # R1 #1：比較過 _safe_score——coerce 殘值（'70分'）在此裸比較會 TypeError
            # → 被 :225 except 吞 → 單件丟件（AF-1 症狀回歸）。降位 0 保留 item。
            if (_safe_score(ansr) > self.EARLY_SEND_THRESHOLD):
                # Skip early send in unified mode — articles sent as batch after ranking
                if self.handler.generate_mode != 'unified':
                    logger.info(f"High score item: {name} (score: {ranking['score']}) - sending early {self.ranking_type_str}")
                    try:
                        await self.sendAnswers([ansr])
                    except (BrokenPipeError, ConnectionResetError):
                        logger.warning(f"Client disconnected while sending early answer for {name}")
                        self.handler.connection_alive_event.clear()
                        return

            logger.debug(f"Item {name} ranked successfully")

            # Analytics: Log ranking score (position=-1 = pending; updated in batch after sort)
            if hasattr(self.handler, 'query_id'):
                query_logger = get_query_logger()
                try:
                    query_logger.log_ranking_score(
                        query_id=self.handler.query_id,
                        doc_url=url,
                        ranking_position=-1,  # Placeholder; updated by update_ranking_positions()
                        llm_final_score=float(_safe_score(ansr)),  # 殘值記 0.0 而非拋 ValueError 進 log_err
                        llm_snippet=ranking.get("description", ""),
                        ranking_method='llm'
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to log ranking score: {log_err}")

            return ansr

        except Exception as e:
            logger.error(f"Error in rankItem for {name}: {str(e)}")
            logger.debug(f"Full error trace: ", exc_info=True)
            if CONFIG.should_raise_exceptions():
                raise  # Re-raise in testing/development mode
            sentry_sdk.capture_exception(e)

    def shouldSend(self, result):
        # Don't send if we've already reached the limit
        if self.num_results_sent >= self.NUM_RESULTS_TO_SEND:
            logger.debug(f"Not sending {result['name']} - already at limit ({self.num_results_sent}/{self.NUM_RESULTS_TO_SEND})")
            return False
            
        should_send = False
        # Allow sending if we're still well below the limit
        if (self.num_results_sent < self.NUM_RESULTS_TO_SEND - 3):
            should_send = True
        else:
            # Near the limit - only send if this result is better than something we already sent
            for r in self.rankedAnswers:
                if r["sent"] == True and _safe_score(r) < _safe_score(result):
                    should_send = True
                    break
        
        logger.debug(f"Should send result {result['name']}? {should_send} (sent: {self.num_results_sent}/{self.NUM_RESULTS_TO_SEND})")
        return should_send
    
    async def sendAnswers(self, answers, force=False):
        if not self.handler.connection_alive_event.is_set():
            logger.warning("Connection lost during ranking, skipping sending results")
            return
        
        json_results = []
        logger.debug(f"Considering sending {len(answers)} answers (force: {force})")
        
        for result in answers:
            # Additional safety check - never exceed the limit even when forced
            if self.num_results_sent + len(json_results) >= self.NUM_RESULTS_TO_SEND:
                logger.info(f"Stopping at {len(json_results)} results to avoid exceeding limit of {self.NUM_RESULTS_TO_SEND}")
                break

            # Title dedup gate: skip if same (name, site) already sent
            title_key = (result.get("name", ""), result.get("site", ""))
            if title_key in self._sent_title_keys:
                logger.info(f"Send dedup: skipping already-sent '{title_key[0]}' from {title_key[1]}")
                result['sent'] = True  # Mark as sent to prevent re-send
                continue

            if self.shouldSend(result) or force:
                result_item = {
                    "@type": "Item",
                    "url": result["url"],
                    "name": result["name"],
                    "site": result["site"],
                    "siteUrl": result["site"],
                    "score": _safe_score(result),  # 殘值輸出 0 而非字串；正常 int 原樣（不 float 化）
                    "description": result["ranking"]["description"],
                    "schema_object": result["schema_object"]
                }

                json_results.append(result_item)

                result["sent"] = True
                self._sent_title_keys.add(title_key)
            
        if (json_results):  # Only attempt to send if there are results
            # Wait for pre checks to be done using event
            await self.handler.pre_checks_done_event.wait()

            try:
                # Final safety check before sending
                if self.num_results_sent + len(json_results) > self.NUM_RESULTS_TO_SEND:
                    # Trim the results to not exceed the limit
                    allowed_count = self.NUM_RESULTS_TO_SEND - self.num_results_sent
                    json_results = json_results[:allowed_count]
                    logger.warning(f"Trimmed results to {len(json_results)} to stay within limit of {self.NUM_RESULTS_TO_SEND}")

                # Use the new schema to create and auto-send the message.
                # generate_mode discriminator stays here (caller-side); the send
                # goes through send_sse(path="full") -> message_sender.send_message.
                if self.handler.generate_mode == 'unified':
                    # Unified mode: send as 'articles' with await for ordering guarantee
                    from core.sse.send import send_sse  # local import: avoid load cycle
                    articles_message = {
                        "message_type": "articles",
                        "content": json_results
                    }
                    await send_sse(self.handler, articles_message, path="full")
                else:
                    create_assistant_result(json_results, handler=self.handler)
                self.num_results_sent += len(json_results)
                logger.info(f"Sent {len(json_results)} results, total sent: {self.num_results_sent}/{self.NUM_RESULTS_TO_SEND}")
            except (ConnectionError, ConnectionResetError, BrokenPipeError, OSError) as e:
                logger.error(f"Client disconnected while sending answers: {str(e)}")
                self.handler.connection_alive_event.clear()
            except Exception as e:
                logger.error(f"Error sending answers (non-connection): {str(e)}")
                sentry_sdk.capture_exception(e)
                # Do NOT clear connection_alive_event — connection may still be alive
  
    async def sendMessageOnSitesBeingAsked(self, top_embeddings):
        if (self.handler.site == "all" or self.handler.site == "nlws"):
            sites_in_embeddings = {}
            for item in top_embeddings:
                # Handle Dict (new format) and Tuple (legacy 4/5/6) formats.
                # Only `site` (index 3) is needed; use index access so a 6-tuple
                # (retrieval_scores at index 5) does not hit a hard 4-unpack.
                if isinstance(item, dict):
                    site = item.get('site', '')
                elif len(item) >= 4:
                    site = item[3]
                else:
                    url, json_str, name, site = item
                sites_in_embeddings[site] = sites_in_embeddings.get(site, 0) + 1

            top_sites = sorted(sites_in_embeddings.items(), key=lambda x: x[1], reverse=True)[:3]
            top_sites_str = ", ".join([self.prettyPrintSite(x[0]) for x in top_sites])
            logger.info(f"Sending sites message: {top_sites_str}")
            
            try:
                # Create a custom message with asking_sites type
                message = Message(
                    sender_type=SenderType.SYSTEM,
                    message_type="asking_sites",  # Custom message type
                    content="Asking " + top_sites_str,
                    conversation_id=self.handler.conversation_id if hasattr(self.handler, 'conversation_id') else None
                )
                asyncio.create_task(self.handler.send_message(message.to_dict()))
                self.handler.sites_in_embeddings_sent = True
            except (BrokenPipeError, ConnectionResetError):
                logger.warning("Client disconnected when sending sites message")
                self.handler.connection_alive_event.clear()
    
    async def do(self):
        logger.info(f"Starting ranking process with {len(self.items)} items")

        # Create a mapping from URL to vector (if vectors are included)
        self.url_to_vector = {}
        for item in self.items:
            # Handle Dict format (new)
            if isinstance(item, dict):
                url = item.get('url', '')
                vector = item.get('vector')
                if vector is not None:
                    self.url_to_vector[url] = vector
            # Handle Tuple format (legacy 5-tuple or 6-tuple with scores).
            # Use len >= 5 so the 6-tuple's vector (index 4) is not silently
            # dropped (which would starve MMR of vectors).
            elif len(item) >= 5:  # [url, json_str, name, site, vector(, retrieval_scores)]
                vector = item[4]
                if vector is not None:
                    self.url_to_vector[item[0]] = vector

        if self.url_to_vector:
            logger.info(f"Vectors available for {len(self.url_to_vector)} items (MMR-ready)")
            # Store vectors on handler for PostRanking to use
            self.handler.url_to_vector = self.url_to_vector

        tasks = []
        for item in self.items:
            # Pass the full item (Dict or Tuple) to rankItem for better data preservation
            if self.handler.connection_alive_event.is_set():  # Only add new tasks if connection is still alive
                tasks.append(asyncio.create_task(self.rankItem(item)))
            else:
                logger.warning("Connection lost, not creating new ranking tasks")

        await self.sendMessageOnSitesBeingAsked(self.items)

        try:
            logger.debug(f"Running {len(tasks)} ranking tasks concurrently")
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Collect results from tasks instead of shared list mutation
            for result in task_results:
                if isinstance(result, Exception):
                    logger.warning(f"Ranking task failed: {result}")
                elif result is not None:
                    self.rankedAnswers.append(result)
        except Exception as e:
            logger.error(f"Error during ranking tasks: {str(e)}")

        if not self.handler.connection_alive_event.is_set():
            logger.warning("Connection lost during ranking, skipping sending results")
            return

        # Wait for pre checks using event
        await self.handler.pre_checks_done_event.wait()

        filtered = [r for r in self.rankedAnswers if _safe_score(r) > 51]
        ranked = sorted(filtered, key=_safe_score, reverse=True)

        # Analytics: Batch-update ranking_position now that final order is known
        if hasattr(self.handler, 'query_id') and ranked:
            try:
                query_logger = get_query_logger()
                positions = [(r.get('url', ''), idx) for idx, r in enumerate(ranked)]
                query_logger.update_ranking_positions(self.handler.query_id, positions)
            except Exception as log_err:
                logger.warning(f"Failed to update ranking positions: {log_err}")

        # Title+source dedup: remove same-article duplicates from publishers
        # that index the same content under multiple category-URL paths.
        # Must run after LLM scoring (needs scores to pick the winner) and
        # before MMR (MMR needs a clean deduplicated candidate set).
        ranked = dedup_by_title_and_source(ranked)

        # Phase A: Apply XGBoost ML re-ranking (shadow mode - logs predictions without changing rankings)
        xgboost_enabled = CONFIG.xgboost_params.get('enabled', False)

        if xgboost_enabled and len(ranked) > 0:
            try:
                from core.xgboost_ranker import XGBoostRanker

                logger.info(f"[XGBoost] Starting shadow mode prediction for {len(ranked)} results")

                # Initialize XGBoost ranker
                xgb_ranker = XGBoostRanker(CONFIG.xgboost_params)

                # Prepare ranking results for XGBoost (extract features from ranked results)
                # Note: ranked is a list of dicts with structure: {'url', 'name', 'schema', 'ranking': {'score', 'snippet'}}
                # XGBoost needs: url, llm_score, and query_id for logging

                # Attach query_id to results for logging
                for result in ranked:
                    result['query_id'] = self.handler.query_id

                # Call XGBoost rerank (in shadow mode, this logs but doesn't change order)
                reranked_by_xgb, xgb_metadata = xgb_ranker.rerank(ranked, self.handler.query)

                logger.info(f"[XGBoost Shadow] Avg score: {xgb_metadata.get('avg_xgboost_score', 0):.3f}, "
                           f"Avg confidence: {xgb_metadata.get('avg_confidence', 0):.3f}")

                # In shadow mode, reranked_by_xgb == ranked (unchanged)
                # Continue with original ranking

            except Exception as e:
                logger.error(f"[XGBoost] Shadow mode failed: {e}")
                logger.exception("XGBoost traceback:")
                # Continue with original ranking if XGBoost fails
        else:
            if not xgboost_enabled:
                logger.debug("[XGBoost] Disabled in config")

        # Apply MMR diversity re-ranking if enabled and vectors available
        mmr_enabled = CONFIG.mmr_params.get('enabled', True)
        mmr_threshold = CONFIG.mmr_params.get('threshold', 3)

        if mmr_enabled and len(ranked) > mmr_threshold and self.url_to_vector:
            logger.info(f"[MMR] Applying diversity re-ranking to {len(ranked)} results")

            # Attach vectors to ranked results
            for result in ranked:
                url = result.get('url', '')
                if url in self.url_to_vector:
                    result['vector'] = self.url_to_vector[url]

            # Apply MMR
            from core.mmr import MMRReranker
            mmr_lambda = CONFIG.mmr_params.get('lambda', 0.7)
            mmr_reranker = MMRReranker(lambda_param=mmr_lambda, query=self.handler.query)
            reranked_results, mmr_scores = mmr_reranker.rerank(
                ranked_results=ranked,
                top_k=self.NUM_RESULTS_TO_SEND
            )

            # Log MMR scores to analytics
            query_logger = get_query_logger()
            if hasattr(self.handler, 'query_id'):
                for idx, (result, mmr_score) in enumerate(zip(reranked_results, mmr_scores)):
                    url = result.get('url', '')
                    query_logger.log_mmr_score(
                        query_id=self.handler.query_id,
                        doc_url=url,
                        mmr_score=mmr_score,
                        ranking_position=idx
                    )

            self.handler.final_ranked_answers = reranked_results
            logger.info(f"[MMR] Re-ranking complete: {len(reranked_results)} diverse results")

            # Clean up: Remove vectors from results before passing to LLM prompts
            # Vectors are 1536 floats and will pollute the console output
            for result in self.handler.final_ranked_answers:
                result.pop('vector', None)
        else:
            # No MMR: use original ranking
            self.handler.final_ranked_answers = ranked[:self.NUM_RESULTS_TO_SEND]
            if not mmr_enabled:
                logger.info("MMR disabled in config, using standard ranking")
            elif len(ranked) <= mmr_threshold:
                logger.info(f"MMR skipped: only {len(ranked)} results (threshold: {mmr_threshold})")
            elif not self.url_to_vector:
                logger.info("MMR skipped: no vectors available")

        logger.info(f"Filtered to {len(filtered)} results with score > 51")
        logger.debug(f"Top 3 results: {[(r['name'], _safe_score(r)) for r in self.handler.final_ranked_answers[:3]]}")

        results = [r for r in self.rankedAnswers if r['sent'] == False]
        if (self.num_results_sent >= self.NUM_RESULTS_TO_SEND):
            logger.info(f"Already sent {self.num_results_sent} results, returning without sending more")
            return
       
        # Sort by score in descending order
        sorted_results = sorted(results, key=_safe_score, reverse=True)
        good_results = [x for x in sorted_results if _safe_score(x) > 51]

        # Calculate how many more results we can send
        remaining_slots = self.NUM_RESULTS_TO_SEND - self.num_results_sent
        if remaining_slots <= 0:
            logger.info(f"Already sent {self.num_results_sent} results, at or above limit of {self.NUM_RESULTS_TO_SEND}")
            return
            
        if len(good_results) >= remaining_slots:
            tosend = good_results[:remaining_slots]
        else:
            tosend = good_results

        try:
            logger.info(f"Sending final batch of {len(tosend)} results")
            await self.sendAnswers(tosend, force=True)
        except (BrokenPipeError, ConnectionResetError):
            logger.error("Client disconnected during final answer sending")
            self.handler.connection_alive_event.clear()

    def prettyPrintSite(self, site):
        ans = site.replace("_", " ")
        words = ans.split()
        return ' '.join(word.capitalize() for word in words)
