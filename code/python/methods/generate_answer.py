# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
This file contains code for the 'generate answer' path, which provides
a flow that is more similar to RAG.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import asyncio
import json
import sentry_sdk
import traceback
from datetime import datetime, timezone

from core.baseHandler import NLWebHandler
from core.llm import ask_llm
from core.prompts import PromptRunner, find_prompt, fill_prompt
from core.retriever import search
from core.utils.json_utils import trim_json, trim_json_hard
from core.utils.utils import log, get_param
from core.temporal_anchor import annotate_relative_years, anchor_note, TEMPORAL_ANCHOR_RULE
from misc.logger.logging_config_helper import get_configured_logger
import core.query_analysis.analyze_query as analyze_query
import core.query_analysis.relevance_detection as relevance_detection
import core.query_analysis.memory as memory
import core.query_analysis.required_info as required_info


logger = get_configured_logger("generate_answer")


class GenerateAnswer(NLWebHandler):

    GATHER_ITEMS_THRESHOLD = 55

    RANKING_PROMPT_NAME = "RankingPromptForGenerate"
    SYNTHESIZE_PROMPT_NAME = "SynthesizePromptForGenerate"
    DESCRIPTION_PROMPT_NAME = "DescriptionPromptForGenerate"

    def __init__(self, query_params, handler):
        super().__init__(query_params, handler)
        self.items = []
        self._results_lock = asyncio.Lock()  # Add lock for thread-safe operations
        logger.info(f"GenerateAnswer initialized with query_params: {query_params}")
        log(f"GenerateAnswer query_params: {query_params}")

    async def runQuery(self):
        try:
            logger.info(f"Starting query execution for conversation_id: {self.conversation_id}")

            await self.prepare()
            if (self.query_done):
                logger.info("Query done prematurely")
                return self.return_value
            await self.get_ranked_answers()
            self.return_value["conversation_id"] = self.conversation_id
            logger.info(f"Query execution completed for conversation_id: {self.conversation_id}")
            return self.return_value
        except Exception as e:
            logger.exception(f"Error in runQuery: {e}")
            traceback.print_exc()
            raise
    
    async def prepare(self):
        # runs the tasks that need to be done before retrieval, ranking, etc.
        logger.info("Starting preparation phase")
        tasks = []
        
        # Adding all necessary preparation tasks
        tasks.append(asyncio.create_task(analyze_query.DetectItemType(self).do()))
        tasks.append(asyncio.create_task(self.decontextualizeQuery().do()))
        tasks.append(asyncio.create_task(relevance_detection.RelevanceDetection(self).do()))
        tasks.append(asyncio.create_task(memory.Memory(self).do()))
        tasks.append(asyncio.create_task(required_info.RequiredInfo(self).do()))
         
        try:
            logger.debug(f"Running {len(tasks)} preparation tasks concurrently")
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.exception(f"Error during preparation tasks: {e}")
        finally:
            self.pre_checks_done_event.set()  # Signal completion regardless of errors
            self.state.set_pre_checks_done()
            
        logger.info("Preparation phase completed")
   
    async def rankItem(self, url, json_str, name, site):
        if not self.connection_alive_event.is_set():
            logger.warning("Connection lost, skipping item ranking")
            return

        try:
            logger.debug(f"Ranking item: {name} from {site}")

            # Parse schema to extract metadata
            schema_obj = json.loads(json_str)
            date_published = schema_obj.get('datePublished', 'Unknown')

            # Calculate age in days
            age_days = 'Unknown'
            if date_published != 'Unknown':
                try:
                    # Handle different date formats
                    if isinstance(date_published, str):
                        # Remove timezone info if present, parse as naive datetime
                        date_str = date_published.split('T')[0] if 'T' in date_published else date_published
                        pub_date = datetime.strptime(date_str, '%Y-%m-%d')
                        # Make it timezone-aware UTC
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    else:
                        pub_date = date_published

                    now = datetime.now(timezone.utc)
                    age_days = (now - pub_date).days
                    print(f"[DEBUG-DATE] {name[:40]}: pub={date_published[:10]}, age={age_days} days")
                except Exception as e:
                    logger.debug(f"Could not parse date {date_published}: {e}")
                    age_days = 'Unknown'
                    print(f"[DEBUG-DATE] {name[:40]}: PARSE ERROR for date: {date_published}, error: {e}")

            # Extract source/publisher
            source = schema_obj.get('publisher', {})
            if isinstance(source, dict):
                source = source.get('name', site)
            else:
                source = site

            # Get prompt and fill with metadata
            prompt_str, ans_struc = find_prompt(site, self.item_type, self.RANKING_PROMPT_NAME)

            # DEBUG: Check what prompt we got

            description = trim_json_hard(json_str)
            # 發布日期錨定（core.temporal_anchor）：ranking 產出的 description 會直接
            # 變成卡片摘要、並被 request.answers 帶進 synthesize/summarize。若不錨定，
            # 2025 年報導的「今年」會被 LLM 依「今天」改寫成當前年。
            # str(): trim_json_hard 回 dict，而 fill_prompt 最後也是 str(value) → 內容等價
            description = annotate_relative_years(str(description), date_published)

            prompt = fill_prompt(prompt_str, self, {
                "item.description": description,
                "item.datePublished": str(date_published),
                "item.age_days": str(age_days),
                "item.source": str(source)
            })

            logger.debug(f"Sending multi-signal ranking request to LLM for item: {name}")
            ranking = await ask_llm(prompt, ans_struc, level="low", query_params=self.query_params)

            if not ranking or not isinstance(ranking, dict):
                logger.error(f"LLM returned empty response for ranking item")
                sentry_sdk.capture_message("LLM returned empty in generate_answer.rankItem")
                ranking = {"score": 0, "final_score": 0, "description": "LLM ranking failed"}

            # DEBUG: Print raw LLM response

            # Log the multi-signal breakdown
            final_score = ranking.get('final_score', ranking.get('score', 0))
            print(f"[RANKING] {name[:50]}: final={final_score}, semantic={ranking.get('semantic_score', 'N/A')}, keyword={ranking.get('keyword_score', 'N/A')}, freshness={ranking.get('freshness_score', 'N/A')}, authority={ranking.get('authority_score', 'N/A')}")
            logger.info(f"Ranked {name}: final={final_score}, semantic={ranking.get('semantic_score', 'N/A')}, keyword={ranking.get('keyword_score', 'N/A')}, freshness={ranking.get('freshness_score', 'N/A')}, authority={ranking.get('authority_score', 'N/A')}")

            # Store with final_score for threshold comparison
            ranking['score'] = final_score  # Ensure 'score' key exists for backward compatibility

            ansr = {
                'url': url,
                'site': site,
                'name': name,
                'ranking': ranking,
                'schema_object': schema_obj,
                'sent': False,
            }

            if (final_score > self.GATHER_ITEMS_THRESHOLD):
                logger.info(f"High score item: {name} (score: {final_score})")
                async with self._results_lock:  # Thread-safe append
                    self.final_ranked_answers.append(ansr)

        except Exception as e:
            logger.error(f"Error in rankItem: {e}")
            logger.debug("Full error trace: ", exc_info=True)
            sentry_sdk.capture_exception(e)

    async def get_ranked_answers(self):
        logger.info("Starting retrieval and ranking process")

        # Analytics: Generate unique query ID and log query start BEFORE any cache checks
        # This ensures analytics logging happens even when using cached results
        import time
        from core.query_logger import get_query_logger

        self.query_id = f"query_{int(time.time() * 1000)}"
        query_logger = get_query_logger()

        try:
            query_logger.log_query_start(
                query_id=self.query_id,
                user_id=self.user_id or "anonymous",
                query_text=self.query,
                site=str(self.site) if isinstance(self.site, list) else self.site,
                mode=self.generate_mode or "generate",
                decontextualized_query=self.decontextualized_query,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                model=self.model,
                parent_query_id=self.parent_query_id,
                org_id=self.org_id
            )
            # Allow parent commit to propagate to avoid foreign key race conditions
            await asyncio.sleep(0.15)
        except Exception as e:
            logger.warning(f"Failed to log query start: {e}")

        # Send begin-nlweb-response message for analytics
        try:
            await self.message_sender.send_begin_response()
        except Exception as e:
            logger.warning(f"Failed to send begin response: {e}")

        # Check if we should reuse cached results from list mode
        # DEFAULT TO TRUE - only skip if explicitly set to false
        reuse_results_param = self.query_params.get('reuse_results', ['true'])
        reuse_results = reuse_results_param[0].lower() != 'false' if isinstance(reuse_results_param, list) else str(reuse_results_param).lower() != 'false'

        if reuse_results:
            try:
                from core.results_cache import get_results_cache
                import json
                cache = get_results_cache()
                # CORE-1: 按 trusted user_id 隔離 retrieve。空 conversation_id 或
                # 無 user_id → cache 層必 miss（與 baseHandler store 對稱，不再用
                # 可碰撞的 query+site fallback key）。
                cached_results = cache.retrieve(
                    self.conversation_id, user_id=self.user_id
                )

                if cached_results:
                    print(f"[CACHE] OK: Reusing {len(cached_results)} cached results for conversation {self.conversation_id}")
                    logger.info(f"Reusing {len(cached_results)} cached results - skipping retrieval and ranking")

                    # Use cached results directly
                    self.final_ranked_answers = cached_results

                    # Reconstruct self.items format from cached results
                    self.items = []
                    for r in cached_results:
                        schema_json = json.dumps(r['schema_object']) if 'schema_object' in r else '{}'
                        self.items.append([r['url'], schema_json, r['name'], r['site']])

                    print(f"[GENERATE] Using {len(self.final_ranked_answers)} cached items for answer synthesis")

                    # Skip directly to synthesis
                    logger.info("Cached results loaded, proceeding to answer synthesis")
                    await self.synthesizeAnswer()
                    return
                else:
                    print(f"[CACHE] MISS: No cached results found for conversation {self.conversation_id}")
                    logger.warning(f"No cached results found for conversation {self.conversation_id}, falling back to fresh retrieval")
            except Exception as e:
                logger.error(f"Error retrieving cached results: {e}, falling back to fresh retrieval")
                print(f"[CACHE] Error: {e}, doing fresh retrieval")

        # Free conversation mode - try to use cached results first, but don't do new retrieval
        if self.free_conversation:
            logger.info("[FREE_CONVERSATION] Free conversation mode - checking for cached results")
            print("[FREE_CONVERSATION] Free conversation mode - checking for cached results")

            # Try to get cached results from the initial search
            cached_results = None
            if reuse_results:
                try:
                    from core.results_cache import get_results_cache
                    import json
                    cache = get_results_cache()

                    # CORE-1: free conversation 也按 trusted user_id 隔離 retrieve。
                    # 結果只在 store 時用 f"{user_id}:{conversation_id}" 記入，故這裡
                    # 一律以 conversation_id + user_id 查（舊的 first_query/query+site
                    # 弱 fallback key 已不再是有效 store key，移除以免跨 user 碰撞）。
                    cached_results = cache.retrieve(
                        self.conversation_id, user_id=self.user_id
                    )

                    if cached_results:
                        print(f"[FREE_CONVERSATION] OK: Found {len(cached_results)} cached results for free conversation")
                        logger.info(f"Free conversation using {len(cached_results)} cached results as context")

                        # Use cached results
                        self.final_ranked_answers = cached_results
                        self.items = []
                        for r in cached_results:
                            schema_json = json.dumps(r['schema_object']) if 'schema_object' in r else '{}'
                            self.items.append([r['url'], schema_json, r['name'], r['site']])
                    else:
                        print(f"[FREE_CONVERSATION] MISS: No cached results found, proceeding without articles")
                except Exception as e:
                    logger.exception(f"Error retrieving cached results for free conversation: {e}")

            # Check for private sources even in free conversation mode
            logger.info(f"[FREE_CONVERSATION] Checking private sources: include_private_sources={getattr(self, 'include_private_sources', 'NOT SET')}, user_id={getattr(self, 'user_id', 'NOT SET')}")
            print(f"[FREE_CONVERSATION] include_private_sources={getattr(self, 'include_private_sources', 'NOT SET')}, user_id={getattr(self, 'user_id', 'NOT SET')}")
            if self.include_private_sources and self.user_id:
                try:
                    from core.user_data_retriever import search_user_documents, format_private_result_for_display
                    import json

                    logger.info(f"[FREE_CONVERSATION] Searching user's private documents for user_id={self.user_id}")
                    print(f"[FREE_CONVERSATION] Searching private documents")

                    # Search private documents (already in async context, just await)
                    private_results = await search_user_documents(
                        query=self.query,
                        user_id=self.user_id,
                        top_k=10,
                        query_params=self.query_params,
                        org_id=self.org_id
                    )

                    if private_results:
                        logger.info(f"[FREE_CONVERSATION] Found {len(private_results)} private documents")
                        print(f"[FREE_CONVERSATION] OK: Found {len(private_results)} private documents")

                        # Format private results and add to items
                        private_items = []
                        for result in private_results:
                            formatted = format_private_result_for_display(result)
                            private_item = [
                                formatted['url'],
                                json.dumps({'text': formatted['text'], 'metadata': formatted.get('metadata', {})}),
                                formatted['title'],
                                formatted['site']
                            ]
                            private_items.append(private_item)

                        # Prepend private items to existing items
                        if not hasattr(self, 'items') or self.items is None:
                            self.items = []
                        self.items = private_items + self.items
                        logger.info(f"[FREE_CONVERSATION] Added {len(private_items)} private items to context")
                    else:
                        logger.info("[FREE_CONVERSATION] No private documents found")

                except Exception as e:
                    logger.exception(f"[FREE_CONVERSATION] Error searching private documents: {e}")
                    print(f"[FREE_CONVERSATION] Private search error: {e}")

            # Use conversation-only synthesis (with or without cached articles)
            logger.info("Free conversation mode: proceeding to answer synthesis using conversation context")
            await self.synthesize_free_conversation()
            return

        try:
            # Original retrieval logic - runs if not reusing or cache miss
            logger.info("Retrieving items for query")

            # Detect if query has temporal keywords
            temporal_keywords = ['最新', '最近', '近期', 'latest', 'recent', '新', '現在', '目前', '當前']
            is_temporal_query = any(keyword in self.query for keyword in temporal_keywords)

            if is_temporal_query:
                print(f"[RETRIEVAL] Temporal query detected: retrieving 150 items, will pre-filter by date")
                num_to_retrieve = 150
            else:
                print(f"[RETRIEVAL] Non-temporal query: retrieving 50 items")
                num_to_retrieve = 50

            top_embeddings = await search(
                self.decontextualized_query,
                self.site,
                num_results=num_to_retrieve,
                query_params=self.query_params
            )

            # Pre-filter by date for temporal queries
            if is_temporal_query:
                from datetime import datetime, timezone, timedelta
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=365)  # Only articles from last year

                filtered_embeddings = []
                for item in top_embeddings:
                    # Handle 4-tuple, 5-tuple (with vector), and 6-tuple
                    # (with retrieval_scores) formats. Only url/json_str are
                    # needed here; the full item is preserved below so a
                    # 6-tuple's retrieval_scores (index 5) is NOT dropped.
                    if len(item) >= 6:
                        url, json_str = item[0], item[1]
                    elif len(item) == 5:
                        url, json_str, name, site, vector = item
                    else:
                        url, json_str, name, site = item
                        vector = None
                    try:
                        schema_obj = json.loads(json_str)
                        date_published = schema_obj.get('datePublished', 'Unknown')

                        if date_published != 'Unknown':
                            pub_date = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                            # Append the ORIGINAL item (do not rebuild) so a
                            # 6-tuple's retrieval_scores survive the temporal filter.
                            if pub_date >= cutoff_date:
                                filtered_embeddings.append(item)
                        # If no date, skip it for temporal queries
                    except Exception as e:
                        logger.warning(f"Failed to parse date for article {url}: {e}")
                        pass  # Skip articles with bad dates

                # If we filtered too aggressively, take top 50 by semantic similarity anyway
                if len(filtered_embeddings) < 50:
                    print(f"[FILTER] Only {len(filtered_embeddings)} recent articles found, using all {len(top_embeddings)} retrieved")
                    top_embeddings = top_embeddings[:80]  # Compromise: use more but not all
                else:
                    print(f"[FILTER] Filtered {len(top_embeddings)} → {len(filtered_embeddings)} recent articles (last 365 days)")
                    top_embeddings = filtered_embeddings[:80]  # Top 80 recent articles

            self.items = top_embeddings  # Store all retrieved items
            print(f"[RETRIEVAL] Will rank {len(top_embeddings)} items")
            logger.debug(f"Retrieved {len(top_embeddings)} items from database")

            # Rank each item
            tasks = []
            for item in top_embeddings:
                # Handle 4-tuple, 5-tuple (with vector), and 6-tuple
                # (with retrieval_scores). rankItem here only needs the first
                # four fields; index 4/5 are ignored on this code path.
                if len(item) >= 5:
                    url, json_str, name, site = item[0], item[1], item[2], item[3]
                else:
                    url, json_str, name, site = item
                tasks.append(asyncio.create_task(self.rankItem(url, json_str, name, site)))


            logger.debug(f"Running {len(tasks)} ranking tasks concurrently")
            await asyncio.gather(*tasks, return_exceptions=True)

            print(f"[RANKING] ===== Initial ranking completed: {len(self.final_ranked_answers)} items above threshold {self.GATHER_ITEMS_THRESHOLD} =====")
            logger.info(f"Initial ranking completed: {len(self.final_ranked_answers)} items above threshold")

            # Log initial ranked order
            if self.final_ranked_answers:
                # Sort by score to show initial ranking
                sorted_answers = sorted(self.final_ranked_answers, key=lambda x: x['ranking'].get('score', 0), reverse=True)
                print("[RANKING] === INITIAL RANKING (by score) ===")
                logger.info("=== INITIAL RANKING (by score) ===")
                for idx, item in enumerate(sorted_answers[:10], 1):  # Show top 10
                    score = item['ranking'].get('score', 0)
                    print(f"[RANKING] {idx}. [{score}] {item['name'][:60]}")
                    logger.info(f"{idx}. [{score}] {item['name'][:60]}")

            # OLD DIVERSITY CODE - COMMENTED OUT FOR A/B TESTING
            # MMR diversity is now handled in ranking.py during the ranking stage
            # This ensures both list/summarize AND generate modes get diverse results
            #
            # print(f"[DIVERSITY CHECK] Have {len(self.final_ranked_answers)} items, threshold is 3")
            # if len(self.final_ranked_answers) > 3:
            #     print(f"[DIVERSITY] Applying diversity re-ranking to {len(self.final_ranked_answers)} items")
            #     logger.info(f"Applying diversity re-ranking to {len(self.final_ranked_answers)} items")
            #     await self.apply_diversity_reranking()
            #     print(f"[DIVERSITY] Diversity re-ranking completed")
            #
            #     # Log final ranked order after diversity
            #     logger.info("=== FINAL RANKING (after diversity) ===")
            #     for idx, item in enumerate(self.final_ranked_answers[:10], 1):  # Show top 10
            #         score = item['ranking'].get('score', 0)
            #         logger.info(f"{idx}. [{score}] {item['name'][:60]}")
            # else:
            #     logger.info(f"Skipping diversity re-ranking (only {len(self.final_ranked_answers)} items)")
            #
            # To re-enable old LLM-based diversity: uncomment above and disable MMR in config_retrieval.yaml

            # CRITICAL: Limit to top 10 items to match list mode behavior
            # This ensures generate mode and list/summary mode reference the same articles
            if len(self.final_ranked_answers) > 10:
                original_count = len(self.final_ranked_answers)
                self.final_ranked_answers = self.final_ranked_answers[:10]
                print(f"[CONSISTENCY] Limited {original_count} items → 10 to match list mode")
                logger.info(f"Limited {original_count} items to top 10 for consistency with list mode")

            # Synthesize the answer from ranked items
            logger.info("Ranking completed, synthesizing answer")
            await self.synthesizeAnswer()

        except Exception as e:
            logger.exception(f"Error in get_ranked_answers: {e}")
            raise

    async def apply_diversity_reranking(self):
        """Apply MMR-style diversity re-ranking to final_ranked_answers"""
        if not self.connection_alive_event.is_set():
            logger.warning("Connection lost, skipping diversity re-ranking")
            return

        try:
            # Sort by score (descending) first
            self.final_ranked_answers.sort(key=lambda x: x['ranking'].get('score', 0), reverse=True)

            # Prepare ranked items for the prompt
            ranked_items_str = ""
            for idx, item in enumerate(self.final_ranked_answers):
                title = item['name']
                score = item['ranking'].get('score', 0)
                desc = item['ranking'].get('description', '')
                source = item['schema_object'].get('publisher', {})
                if isinstance(source, dict):
                    source = source.get('name', item['site'])
                date_pub = item['schema_object'].get('datePublished', 'Unknown')

                ranked_items_str += f"{idx}. [{score}] {title}\n   Source: {source} | Date: {date_pub}\n   {desc}\n\n"

            # Get diversity prompt
            prompt_str, ans_struc = find_prompt(self.site, self.item_type, "DiversityReranking")
            prompt = fill_prompt(prompt_str, self, {"request.ranked_items": ranked_items_str})

            logger.debug("Sending diversity re-ranking request to LLM")
            diversity_result = await ask_llm(prompt, ans_struc, level="low", query_params=self.query_params)

            if not diversity_result or not isinstance(diversity_result, dict):
                logger.error("LLM returned empty response for diversity reranking")
                return ranked_items  # 保留原順序

            # Apply reordering
            reordered_indices = diversity_result.get('reordered_indices', [])
            diversity_notes = diversity_result.get('diversity_notes', '')

            if reordered_indices and len(reordered_indices) == len(self.final_ranked_answers):
                logger.info(f"Diversity re-ranking applied: {diversity_notes}")
                # Reorder the list
                original_list = self.final_ranked_answers.copy()
                self.final_ranked_answers = [original_list[i] for i in reordered_indices]
            else:
                logger.warning(f"Invalid reordering indices from diversity prompt, keeping original order")

        except Exception as e:
            logger.error(f"Error in apply_diversity_reranking: {e}")
            logger.debug("Full error trace: ", exc_info=True)
            sentry_sdk.capture_exception(e)
            # Continue with original ranking if diversity fails

    async def getDescription(self, url, json_str, query, answer, name, site):
        try:
            logger.debug(f"Getting description for item: {name}")
            description = await PromptRunner(self).run_prompt(self.DESCRIPTION_PROMPT_NAME)
            logger.debug(f"Got description for item: {name}")
            if description is None:
                logger.warning(f"PromptRunner returned None for description of '{name}' — using empty string fallback")
                return (url, name, site, "", json_str)
            desc_text = description.get("description")
            if desc_text is None:
                logger.warning(f"Description key missing or None for '{name}' — using empty string fallback. keys={list(description.keys())}")
                desc_text = ""
            return (url, name, site, desc_text, json_str)
        except Exception as e:
            logger.error(f"Error getting description for {name}: {str(e)}")
            logger.debug("Full error trace: ", exc_info=True)
            raise

    async def synthesize_free_conversation(self):
        """
        Synthesize answer for free conversation mode - uses conversation context
        and cached articles (if available), but no new retrieval.
        """
        if not self.connection_alive_event.is_set():
            logger.warning("Connection lost, skipping free conversation synthesis")
            print("[CANCEL] Skipping free conversation synthesis - client disconnected")
            return

        try:
            # Check for research report passed from frontend (new simplified approach)
            research_report = get_param(self.query_params, "research_report", str, "")
            if research_report:
                report_len = len(research_report)
                logger.info(f"[FREE_CONVERSATION] Received research report from frontend ({report_len} chars)")
                print(f"[FREE_CONVERSATION] Received research report from frontend ({report_len} chars)")

                # Warn if report is very large (may exceed LLM context window)
                if report_len > 50000:  # ~12k tokens
                    logger.warning(f"[FREE_CONVERSATION] Research report is very large ({report_len} chars), may exceed context window")
                    print(f"[FREE_CONVERSATION] WARNING: Research report is very large ({report_len} chars)")

                self.injected_research_report = research_report
                self.injected_source_urls = []  # Sources are embedded in the report as [n] citations
            else:
                logger.info(f"[FREE_CONVERSATION] No research report in request")
                print(f"[FREE_CONVERSATION] No research report in request")
                self.injected_research_report = None
                self.injected_source_urls = None

            has_cached_articles = len(self.final_ranked_answers) > 0
            logger.info(f"Starting free conversation synthesis (cached articles: {len(self.final_ranked_answers)})")
            print(f"[FREE_CONVERSATION] Synthesizing answer using conversation context and {len(self.final_ranked_answers)} cached articles")

            # Build conversation context from previous queries
            conversation_context = ""
            if self.prev_queries:
                conversation_context = "Previous questions in this conversation:\n"
                for idx, prev_q in enumerate(self.prev_queries, 1):
                    conversation_context += f"{idx}. {prev_q}\n"
                conversation_context += "\n"

            # Build article context if we have cached articles OR private documents
            article_context = ""

            # Add private documents first (from self.items)
            if self.items and len(self.items) > 0:
                article_context += "\n===您的私人文件===\n"
                for idx, item in enumerate(self.items, 1):
                    url, json_str, title, site = item
                    import json
                    content_obj = json.loads(json_str) if json_str else {}
                    text = content_obj.get('text', '')

                    article_context += f"文件 {idx}: {title}\n"
                    article_context += f"   來源: {site}\n"
                    article_context += f"   內容: {text[:500]}...\n"  # First 500 chars
                    article_context += "\n"
                article_context += "===\n\n"

            if has_cached_articles:
                article_context += "\n===相關新聞文章（來自之前的搜尋）===\n"
                for idx, item in enumerate(self.final_ranked_answers[:8], 1):  # Top 8 articles for more context
                    title = item['name']
                    desc = item['ranking'].get('description', '')
                    url = item.get('url', '')
                    source = item.get('schema_object', {}).get('publisher', {})
                    if isinstance(source, dict):
                        source = source.get('name', item.get('site', ''))
                    date = item.get('schema_object', {}).get('datePublished', '')
                    if date:
                        date = date.split('T')[0]  # Just the date part

                    # 發布日期錨定（core.temporal_anchor）：內文「今年」以該報導發布年換算
                    desc = annotate_relative_years(desc, date)
                    article_context += f"文章 {idx}: {title}\n"
                    if source or date:
                        article_context += f"   來源: {source} | 日期: {date}{anchor_note(date)}\n"
                    article_context += f"   摘要: {desc}\n"
                    if url:
                        article_context += f"   網址: {url}\n"
                    article_context += "\n"
                article_context += "===\n\n"

            # Build pinned articles context if user pinned specific news cards
            pinned_articles_context = ""
            pinned_articles = get_param(self.query_params, "pinned_articles", list, [])
            if pinned_articles:
                pinned_articles_context = "\n===用戶釘選的重點文章===\n"
                for idx, article in enumerate(pinned_articles, 1):
                    title = article.get('title', '') if isinstance(article, dict) else ''
                    desc = article.get('description', '') if isinstance(article, dict) else ''
                    url = article.get('url', '') if isinstance(article, dict) else ''
                    pinned_articles_context += f"釘選 {idx}: {title}\n"
                    if desc:
                        pinned_articles_context += f"   摘要: {desc}\n"
                    if url:
                        pinned_articles_context += f"   網址: {url}\n"
                    pinned_articles_context += "\n"
                pinned_articles_context += "===\n\n"
                logger.info(f"[FREE_CONVERSATION] {len(pinned_articles)} pinned articles injected into prompt")

            # Build Deep Research report context if available
            research_report_context = ""
            has_research_report = getattr(self, 'injected_research_report', None) is not None
            if has_research_report:
                research_report_context = f"""
===前次深度研究報告===
{self.injected_research_report}
===

注意：
- 報告中的 [數字] 為引用標記，對應的來源 URL 已記錄
- 請基於此報告回答用戶的追問
- 如用戶詢問特定引用的詳細內容，可告知該來源的資訊

"""
                logger.info(f"[FREE_CONVERSATION] Injected research report into prompt ({len(self.injected_research_report)} chars)")
                print(f"[FREE_CONVERSATION] Injected research report into prompt ({len(self.injected_research_report)} chars)")

            # Create a detailed prompt for high-quality conversational responses
            # Priority: Deep Research report > cached articles > general conversation
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Shared system context for all prompt variants (Bug #1, #3, #9, #24)
            system_context = f"""**今天的日期是：{current_date}**
如果用戶詢問日期相關問題，請使用此日期，不要從搜尋結果中推測日期。
{TEMPORAL_ANCHOR_RULE}

**重要系統限制**：
- 你只能存取資料庫中已收錄的新聞，不代表所有新聞。
- **如果用戶問「為什麼只有某日期/某主題的新聞」，最可能的原因是資料庫收錄範圍有限。**
- **絕對不要猜測或合理化新聞數量/日期分布的原因。**
- 誠實回答：「這可能是因為資料庫收錄範圍的限制，建議調整搜尋條件或時間範圍重新搜尋。」

**你的能力範圍**：
- 你可以分析和討論已搜尋到的新聞文章
- 你可以回答基於搜尋結果的問題
- 如果用戶的問題超出目前搜尋結果的範圍，建議他們：修改搜尋關鍵字重新搜尋、調整時間範圍、或使用深度研究模式
- 不要說「我無法存取即時新聞」，而是說「目前搜尋結果中沒有相關資訊，建議您重新搜尋 [具體建議]」

請使用 Markdown 格式回答。段落之間用空行分隔，列表使用 - 或 1. 2. 3. 格式，重要概念可用 **粗體** 強調。
"""

            if has_research_report:
                # Has previous Deep Research report - use it as primary context
                prompt = f"""你是專業台灣新聞分析助手。用戶先前進行深度研究，現針對該研究結果追問。

{system_context}
{research_report_context}
{pinned_articles_context}{conversation_context}當前問題: {self.query}

回答要求：
1. **基於報告回答** - 優先使用深度研究報告中的內容回答，同時參考用戶釘選的重點文章
2. **保持引用格式** - 如果引用報告中的資訊，保留原有的 [數字] 引用標記
3. **直接且具體** - 使用報告中的具體數據、公司名稱、技術名稱
4. **結構清晰** - 用 1-2 段組織回答，直接回應用戶的追問
5. **承認不確定性** - 如果報告中沒有相關資訊，明確說明
6. **長度限制** - 回答必須控制在 800 字以內，簡潔有力，不要過度展開

請用繁體中文回答，保持專業且資訊豐富。"""
            elif has_cached_articles:
                prompt = f"""你是專業台灣新聞分析助手，可參考用戶先前搜尋的新聞報導。

{system_context}
{conversation_context}{article_context}{pinned_articles_context}當前問題: {self.query}

回答要求：
1. **直接且具體回答** - 使用文章中的具體公司名稱、產品名稱、技術名稱
2. **引用具體數據** - 包含百分比、數字、金額、日期等具體資訊
3. **提供實例** - 從多篇文章中引用 2-3 個具體案例
4. **結構清晰** - 用 2-3 段組織回答：
   - 第1段：直接回答問題（包含具體名稱）
   - 第2段：詳細證據與實例（公司+技術+數據）
   - 第3段：趨勢分析或影響（基於文章內容）

5. **引用來源** - 在提及具體資訊時，用 [來源](url) 格式標註

範例（好的回答）：
「Momo 使用 GenAI 檢測違規商品，準確率達 99%，並透過 Gemini 優化個人化推薦 [來源](https://...)。」

範例（避免的回答）：
「許多零售商使用 AI 來提升效率。」（太籠統）

6. **長度限制** - 回答必須控制在 800 字以內，簡潔有力，不要過度展開

請用繁體中文回答，保持專業且資訊豐富。"""
            else:
                # No cached articles - provide general conversational response
                prompt = f"""你是專業台灣新聞分析助手，協助回答台灣新聞與時事相關問題。

{system_context}
{conversation_context}{article_context}{pinned_articles_context}當前問題: {self.query}

請根據對話脈絡和提供的文件內容提供有幫助的回答。

回答要求：
1. 如果問題參考了之前的對話或提供的文件，請確認理解用戶的意圖
2. 提供有建設性的回應或建議
3. 如果需要更多資訊，可以建議用戶進行新的搜尋
4. **長度限制** - 回答必須控制在 800 字以內，簡潔有力，不要過度展開

請用繁體中文回答，保持專業且有幫助。"""

            # Call LLM directly for conversational response
            # Use larger max_length for detailed conversational responses (2048 tokens for Chinese text)
            response = await ask_llm(
                prompt,
                {"answer": "string"},
                level="high",
                query_params=self.query_params,
                max_length=2048
            )

            if not response or not isinstance(response, dict):
                logger.error("LLM returned empty response for free conversation")
                response = {"answer": "抱歉，系統暫時無法生成回應，請稍後再試。"}

            answer = response.get("answer", "抱歉，我無法生成回應。請重新表述您的問題。")

            # Send the answer
            message = {
                "message_type": "nlws",
                "@type": "GeneratedAnswer",
                "answer": answer,
                "items": []
            }

            logger.info("Sending free conversation response")
            print(f"[FREE_CONVERSATION] Sending answer: {answer[:100]}...")
            await self.send_message(message)

        except Exception as e:
            logger.exception(f"Error in synthesize_free_conversation: {e}")
            if self.connection_alive_event.is_set():
                try:
                    error_msg = {
                        "message_type": "nlws",
                        "@type": "GeneratedAnswer",
                        "answer": "抱歉，處理您的問題時發生錯誤。請再試一次。",
                        "items": []
                    }
                    await self.send_message(error_msg)
                except Exception as e:
                    logger.warning(f"Failed to send error message to client: {e}")
                    pass
            raise

    async def synthesizeAnswer(self):
        if not self.connection_alive_event.is_set():
            logger.warning("Connection lost, skipping answer synthesis")
            print("[CANCEL] Skipping answer synthesis - client disconnected")
            return

        # G4(ii): msg_type moved ABOVE the try so the except-fallback (:931) can
        # read it even if an exception fires before this line — unifying the
        # exception fallback to msg_type (was hardcoded "nlws", invisible in
        # unified mode). Task 12: sends route through send_sse(path="full").
        from core.sse.send import send_sse  # local import: avoid load cycle
        msg_type = "answer" if self.generate_mode == 'unified' else "nlws"

        try:
            logger.info("Starting answer synthesis")

            # Check if we have any ranked answers to work with
            if not self.final_ranked_answers:
                logger.warning("No ranked answers found, sending empty response")
                message = {
                    "message_type": msg_type,
                    "@type": "GeneratedAnswer",
                    "answer": "抱歉，找不到與您問題相關的資訊。",
                    "items": []
                }
                await send_sse(self, message, path="full")
                return

            response = await PromptRunner(self).run_prompt(self.SYNTHESIZE_PROMPT_NAME, timeout=100, verbose=False, max_length=2048)
            logger.debug(f"Synthesis response received")

            # Check if response is None or empty (prompt not found or LLM error)
            if not response:
                logger.error("Synthesis prompt returned None - prompt may not be found or LLM error occurred")
                message = {
                    "message_type": msg_type,
                    "@type": "GeneratedAnswer",
                    "answer": "抱歉，無法生成回答。系統配置可能有問題，請聯繫管理員。",
                    "items": []
                }
                await send_sse(self, message, path="full")
                return

            json_results = []
            description_tasks = []

            # Join paragraphs array with HTML breaks for proper frontend rendering
            # Using <br><br> ensures visible paragraph spacing in both textContent and innerHTML contexts
            if "paragraphs" in response and isinstance(response["paragraphs"], list):
                answer = "<br><br>".join(response["paragraphs"])
                logger.info(f"Joined {len(response['paragraphs'])} paragraphs into answer with HTML breaks")
            else:
                # Fallback for old format
                answer = response.get("answer", "")
                logger.warning("Response did not contain 'paragraphs' array, using fallback")

            # Clean up any raw URLs that the LLM may have included despite instructions
            # Convert (https://example.com) to markdown hyperlink format [來源](https://example.com)
            import re
            def convert_url_to_link(match):
                url = match.group(0)
                if url.startswith('('):
                    # Remove parentheses and create markdown link
                    clean_url = url[1:-1]  # Remove ( and )
                    return f'[來源]({clean_url})'
                else:
                    # Bare URL - wrap in markdown link
                    return f'[來源]({url})'

            # Match URLs in parentheses like (https://...)
            answer = re.sub(r'\(https?://[^\)]+\)', convert_url_to_link, answer)
            # Match bare URLs not already in markdown format
            answer = re.sub(r'(?<!\]\()https?://\S+', convert_url_to_link, answer)

            # Create initial message with just the answer (msg_type set at function start)
            message = {"message_type": msg_type, "@type": "GeneratedAnswer", "answer": answer, "items": json_results}
            logger.info("Sending initial answer")
            await send_sse(self, message, path="full")
            
            # Process each URL mentioned in the response
            if "urls" in response and response["urls"]:
                for url in response["urls"]:
                    # Find the matching item in our items list
                    matching_items = [item for item in self.items if item[0] == url]
                    if not matching_items:
                        logger.warning(f"URL {url} referenced in response not found in items")
                        continue
                        
                    item = matching_items[0]
                    # item[:4] so a 5-/6-tuple (vector / retrieval_scores at
                    # index 4-5) does not raise "too many values to unpack".
                    (url, json_str, name, site) = item[:4]
                    logger.debug(f"Creating description task for item: {name}")
                    t = asyncio.create_task(self.getDescription(url, json_str, self.decontextualized_query, answer, name, site))
                    description_tasks.append(t)
                    
                if description_tasks:
                    logger.info(f"Waiting for {len(description_tasks)} description tasks to complete")
                    desc_answers = await asyncio.gather(*description_tasks, return_exceptions=True)
                    
                    for result in desc_answers:
                        if isinstance(result, Exception):
                            logger.error(f"Error getting description: {result}")
                            continue
                            
                        url, name, site, description, json_str = result
                        logger.debug(f"Adding result for {name} to final message")
                        json_results.append({
                            "@type": "Item",
                            "url": url,
                            "name": name,
                            "description": description,
                            "site": site,
                            "schema_object": json.loads(json_str),
                        })
                        
                    # Update message with descriptions
                    message = {"message_type": msg_type, "@type": "GeneratedAnswer", "answer": answer, "items": json_results}
                    logger.info(f"Sending final answer with {len(json_results)} item descriptions")
                    await send_sse(self, message, path="full")
            else:
                logger.warning("No URLs found in synthesis response")

            # UNIFICATION: Also send the ranked list items (like summarize mode does)
            # This allows the frontend to show both the AI answer AND the source list
            # Skip in unified mode — articles already sent by ranking.py
            if self.generate_mode != 'unified':
                logger.info("Sending ranked list items for display alongside generated answer")
                await self._send_ranked_list()

        except Exception as e:
            logger.exception(f"Error in synthesizeAnswer: {e}")
            if self.connection_alive_event.is_set():
                try:
                    # G4(ii): use msg_type (was hardcoded "nlws"). In unified mode
                    # the frontend has no 'nlws' case, so the fallback text was
                    # never rendered; msg_type="answer" makes it visible. msg_type
                    # is defined above the try so this always reads it.
                    error_msg = {"message_type": msg_type, "@type": "GeneratedAnswer", "answer": "抱歉，生成回答時發生錯誤，請重新嘗試。", "items": []}
                    await send_sse(self, error_msg, path="full")
                except Exception as e:
                    logger.warning(f"Failed to send error message to client: {e}")
                    pass
            raise

    async def _send_ranked_list(self):
        """
        Send the ranked list of items (like standard ranking does).
        This unifies generate mode with list mode - same items shown in both.
        """
        try:
            from core.schemas import create_assistant_result

            # Take top 10 ranked answers for the list view
            list_items = self.final_ranked_answers[:10]

            # Format for sending
            json_results = []
            for result in list_items:
                result_item = {
                    "@type": "Item",
                    "url": result["url"],
                    "name": result["name"],
                    "site": result["site"],
                    "siteUrl": result["site"],
                    "score": result["ranking"]["score"],
                    "description": result["ranking"]["description"],
                    "schema_object": result["schema_object"]
                }
                json_results.append(result_item)

            # Send using the standard result schema
            if json_results:
                create_assistant_result(json_results, handler=self)
                logger.info(f"Sent {len(json_results)} ranked list items")
            else:
                logger.warning("No ranked items to send in list")

        except Exception as e:
            logger.error(f"Error sending ranked list: {e}")
            sentry_sdk.capture_exception(e)
            # Don't fail the whole synthesis if list sending fails
            pass