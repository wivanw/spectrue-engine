from spectrue_core.verification.search_mgr import SearchManager, SEARCH_COSTS
from spectrue_core.verification.evidence import build_evidence_pack
from spectrue_core.verification.source_utils import canonicalize_source
from spectrue_core.utils.text_processing import clean_article_text, normalize_search_query
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trust_utils import enrich_sources_with_trust
from spectrue_core.utils.trace import Trace
from spectrue_core.config import SpectrueConfig
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.graph import ClaimGraphBuilder
from spectrue_core.verification.pipeline_claim_graph import run_claim_graph_flow
from spectrue_core.verification.pipeline_evidence import EvidenceFlowInput, run_evidence_flow
from spectrue_core.verification.pipeline_oracle import OracleFlowInput, run_oracle_flow
from spectrue_core.verification.pipeline_search import (
    SearchFlowInput,
    SearchFlowState,
    run_search_flow,
)
from spectrue_core.verification.pipeline_input import (
    extract_url_anchors,
    is_url_input,
    resolve_url_content,
    restore_urls_from_anchors,
)
from spectrue_core.verification.pipeline_queries import (
    build_assertion_query,
    get_claim_units_for_evidence_mapping,
    get_query_by_role,
    is_fuzzy_duplicate,
    normalize_and_sanitize,
    select_diverse_queries,
    select_queries_from_claim_units,
)
import logging
import asyncio
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# M63: Intents that should trigger Oracle check
ORACLE_CHECK_INTENTS = {"news", "evergreen", "official"}
# M63: Intents that should skip Oracle (opinion, prediction)
ORACLE_SKIP_INTENTS = {"opinion", "prediction"}


@dataclass(slots=True)
class _PreparedInput:
    fact: str
    original_fact: str
    final_context: str
    final_sources: list
    inline_sources: list[dict]


class ValidationPipeline:
    """
    Orchestrates the fact-checking waterfall process.
    """
    def __init__(self, config: SpectrueConfig, agent: FactCheckerAgent, translation_service=None):
        self.config = config
        self.agent = agent
        # M63: Pass oracle_skill to SearchManager for hybrid mode
        self.search_mgr = SearchManager(config, oracle_validator=agent.oracle_skill)
        # M67: Optional translation service for Oracle result localization
        self.translation_service = translation_service
        
        # M72: ClaimGraph for key claim identification
        self._claim_graph: ClaimGraphBuilder | None = None
        claim_graph_enabled = (
            getattr(getattr(getattr(config, "runtime", None), "claim_graph", None), "enabled", False)
            is True
        )
        if config and claim_graph_enabled:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(api_key=config.openai_api_key)
            self._claim_graph = ClaimGraphBuilder(
                config=config.runtime.claim_graph,
                openai_client=openai_client,
                edge_typing_skill=agent.edge_typing_skill,
            )

    async def execute(
        self,
        fact: str,
        search_type: str,
        gpt_model: str,
        lang: str,
        content_lang: str | None = None,
        max_cost: int | None = None,
        progress_callback=None,
        preloaded_context: str | None = None,
        preloaded_sources: list | None = None,
        needs_cleaning: bool = False,  # M60: Text from extension needs LLM cleaning
        source_url: str | None = None,  # M60: Original URL for inline source exclusion
    ) -> dict:
        Trace.event(
            "pipeline.run.start",
            {
                "search_type": search_type,
                "lang": lang,
                "needs_cleaning": needs_cleaning,
                "has_preloaded_context": bool(preloaded_context),
                "has_preloaded_sources": bool(preloaded_sources),
            },
        )

        if progress_callback:
            await progress_callback("analyzing_input")

        self.search_mgr.reset_metrics()

        prepared = await self._prepare_input(
            fact=fact,
            preloaded_context=preloaded_context,
            preloaded_sources=preloaded_sources,
            needs_cleaning=needs_cleaning,
            source_url=source_url,
            progress_callback=progress_callback,
        )
        fact = prepared.fact
        original_fact = prepared.original_fact
        final_context = prepared.final_context
        final_sources = prepared.final_sources
        inline_sources = prepared.inline_sources

        claims, should_check_oracle, article_intent, fast_query = await self._extract_claims(
            fact=fact,
            lang=lang,
            progress_callback=progress_callback,
        )

        final_sources = await self._verify_inline_sources(
            inline_sources=inline_sources,
            claims=claims,
            fact=fact,
            final_sources=final_sources,
            progress_callback=progress_callback,
        )

        oracle_flow = await run_oracle_flow(
            self.search_mgr,
            inp=OracleFlowInput(
                original_fact=original_fact,
                fast_query=fast_query,
                lang=lang,
                article_intent=article_intent,
                should_check_oracle=should_check_oracle,
                claims=claims,
                oracle_check_intents=ORACLE_CHECK_INTENTS,
                oracle_skip_intents=ORACLE_SKIP_INTENTS,
                progress_callback=progress_callback,
            ),
            finalize_jackpot=lambda oracle_result: self._finalize_oracle_hybrid(
                oracle_result,
                original_fact,
                lang=lang,
                progress_callback=progress_callback,
            ),
            create_evidence_source=self._create_oracle_source,
        )

        if oracle_flow.early_result:
            Trace.event("pipeline.run.completed", {"outcome": "oracle_early"})
            return oracle_flow.early_result

        if oracle_flow.evidence_source:
            canonical = canonicalize_source(oracle_flow.evidence_source)
            final_sources.append(canonical or oracle_flow.evidence_source)

        await run_claim_graph_flow(
            self._claim_graph,
            claims=claims,
            runtime_config=self.config.runtime,
            progress_callback=progress_callback,
        )

        search_queries = self._select_diverse_queries(claims, max_queries=3, fact_fallback=fact)

        search_state = await run_search_flow(
            config=self.config,
            search_mgr=self.search_mgr,
            agent=self.agent,
            can_add_search=self._can_add_search,
            inp=SearchFlowInput(
                fact=fact,
                lang=lang,
                gpt_model=gpt_model,
                search_type=search_type,
                max_cost=max_cost,
                article_intent=article_intent,
                search_queries=search_queries,
                claims=claims,
                preloaded_context=preloaded_context,
                progress_callback=progress_callback,
            ),
            state=SearchFlowState(
                final_context=final_context,
                final_sources=final_sources,
                preloaded_context=preloaded_context,
                used_orchestration=False,
            ),
        )
        final_context = search_state.final_context
        final_sources = search_state.final_sources

        if getattr(search_state, "hard_reject", False):
            reason = getattr(search_state, "reject_reason", "irrelevant")
            Trace.event("pipeline.run.completed", {"outcome": "hard_reject", "reason": reason})
            return {
                "verified_score": 0.0,
                "analysis": f"Search results are irrelevant to the claim: {reason}",
                "rationale": reason,
                "cost": self.search_mgr.calculate_cost(gpt_model, search_type),
                "text": fact,
                "search_meta": self.search_mgr.get_search_meta(),
                "sources": [],
            }

        result = await run_evidence_flow(
            agent=self.agent,
            search_mgr=self.search_mgr,
            build_evidence_pack=build_evidence_pack,
            enrich_sources_with_trust=enrich_sources_with_trust,
            inp=EvidenceFlowInput(
                fact=fact,
                original_fact=original_fact,
                lang=lang,
                content_lang=content_lang,
                gpt_model=gpt_model,
                search_type=search_type,
                progress_callback=progress_callback,
            ),
            claims=claims,
            sources=final_sources,
        )
        Trace.event("pipeline.run.completed", {"outcome": "scored"})
        return result

    async def _prepare_input(
        self,
        *,
        fact: str,
        preloaded_context: str | None,
        preloaded_sources: list | None,
        needs_cleaning: bool,
        source_url: str | None,
        progress_callback,
    ) -> _PreparedInput:
        final_context = preloaded_context or ""
        final_sources = preloaded_sources or []

        original_fact = fact
        inline_sources: list[dict] = []
        exclude_url = source_url

        if self._is_url(fact) and not preloaded_context:
            exclude_url = fact
            fetched_text = await self._resolve_url_content(fact)
            if fetched_text:
                url_anchors = self._extract_url_anchors(fetched_text, exclude_url=exclude_url)
                if url_anchors:
                    logger.info("[Pipeline] Found %d URL-anchor pairs in raw text", len(url_anchors))

                if len(fetched_text) > 10000 and progress_callback:
                    await progress_callback("processing_large_text")
                    logger.info(
                        "[Pipeline] Large text detected: %d chars, extended timeout",
                        len(fetched_text),
                    )

                cleaned_article = await self.agent.clean_article(fetched_text)
                fact = cleaned_article or fetched_text
                final_context = fact

                if url_anchors and cleaned_article:
                    inline_sources = self._restore_urls_from_anchors(cleaned_article, url_anchors)
                    if inline_sources:
                        logger.info(
                            "[Pipeline] Restored %d inline source candidates after cleaning",
                            len(inline_sources),
                        )
                        Trace.event(
                            "pipeline.inline_sources",
                            {
                                "count": len(inline_sources),
                                "urls": [s["url"][:80] for s in inline_sources[:5]],
                            },
                        )
                        for src in inline_sources:
                            src["is_primary_candidate"] = True

        elif needs_cleaning and not self._is_url(fact):
            logger.info("[Pipeline] Extension page mode: cleaning %d chars", len(fact))

            url_anchors = self._extract_url_anchors(fact, exclude_url=exclude_url)
            if url_anchors:
                logger.info("[Pipeline] Found %d URL-anchor pairs in extension text", len(url_anchors))

            if len(fact) > 10000 and progress_callback:
                await progress_callback("processing_large_text")
                logger.info("[Pipeline] Large text detected: %d chars, extended timeout", len(fact))

            cleaned_article = await self.agent.clean_article(fact)
            if cleaned_article:
                if url_anchors:
                    inline_sources = self._restore_urls_from_anchors(cleaned_article, url_anchors)
                    if inline_sources:
                        logger.info(
                            "[Pipeline] Restored %d inline source candidates after cleaning",
                            len(inline_sources),
                        )
                        Trace.event(
                            "pipeline.inline_sources",
                            {
                                "count": len(inline_sources),
                                "urls": [s["url"][:80] for s in inline_sources[:5]],
                            },
                        )
                        for src in inline_sources:
                            src["is_primary_candidate"] = True

                fact = cleaned_article
                final_context = fact

        elif not self._is_url(fact) and not needs_cleaning:
            url_anchors = self._extract_url_anchors(fact, exclude_url=exclude_url)
            if url_anchors:
                logger.info("[Pipeline] Found %d URL-anchor pairs in plain text", len(url_anchors))
                for item in url_anchors:
                    inline_sources.append(
                        {
                            "url": item["url"],
                            "title": item["anchor"],
                            "domain": item["domain"],
                            "source_type": "inline",
                            "is_trusted": False,
                            "is_primary_candidate": True,
                        }
                    )
                if inline_sources:
                    Trace.event(
                        "pipeline.inline_sources",
                        {
                            "count": len(inline_sources),
                            "urls": [s["url"][:80] for s in inline_sources[:5]],
                        },
                    )

        return _PreparedInput(
            fact=fact,
            original_fact=original_fact,
            final_context=final_context,
            final_sources=final_sources,
            inline_sources=inline_sources,
        )

    async def _extract_claims(
        self,
        *,
        fact: str,
        lang: str,
        progress_callback,
    ) -> tuple[list, bool, str, str]:
        if progress_callback:
            await progress_callback("extracting_claims")

        fact_first_line = fact.strip().split("\n")[0]
        blob = fact_first_line if len(fact_first_line) > 20 else fact[:200]

        cleaned_fact = clean_article_text(fact)
        task_claims = asyncio.create_task(self.agent.extract_claims(cleaned_fact[:4000], lang=lang))

        if len(blob) > 150:
            blob = blob[:150].rsplit(" ", 1)[0]
        fast_query = normalize_search_query(blob)

        if progress_callback:
            await progress_callback("extracting_claims")

        try:
            claims_result = await task_claims
            if isinstance(claims_result, tuple) and len(claims_result) >= 3:
                claims, should_check_oracle, article_intent = claims_result
            elif isinstance(claims_result, tuple) and len(claims_result) == 2:
                claims, should_check_oracle = claims_result
                article_intent = "news"
            else:
                claims, should_check_oracle, article_intent = claims_result, False, "news"
        except asyncio.CancelledError:
            claims = []
            should_check_oracle = False
            article_intent = "news"

        if claims:
            for i, c in enumerate(claims):
                c["id"] = f"c{i+1}"

        return claims, should_check_oracle, article_intent, fast_query

    async def _verify_inline_sources(
        self,
        *,
        inline_sources: list[dict],
        claims: list,
        fact: str,
        final_sources: list,
        progress_callback,
    ) -> list:
        if inline_sources and claims:
            if progress_callback:
                await progress_callback("verifying_sources")

            article_excerpt = fact[:500] if fact else ""
            verified_inline_sources = []

            verification_tasks = [
                self.agent.verify_inline_source_relevance(claims, src, article_excerpt)
                for src in inline_sources
            ]
            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

            for src, result in zip(inline_sources, verification_results):
                if isinstance(result, Exception):
                    logger.warning("[Pipeline] Inline source verification failed: %s", result)
                    src["is_primary"] = False
                    src["is_relevant"] = True
                    verified_inline_sources.append(src)
                    continue

                is_relevant = result.get("is_relevant", True)
                is_primary = result.get("is_primary", False)

                if not is_relevant:
                    logger.debug("[Pipeline] Inline source rejected: %s", src.get("domain"))
                    continue

                src["is_primary"] = is_primary
                src["is_relevant"] = True
                if is_primary:
                    src["is_trusted"] = True
                verified_inline_sources.append(src)
                logger.info(
                    "[Pipeline] Inline source %s: relevant=%s, primary=%s",
                    src.get("domain"),
                    is_relevant,
                    is_primary,
                )

            if verified_inline_sources:
                Trace.event(
                    "pipeline.inline_sources_verified",
                    {
                        "total": len(inline_sources),
                        "passed": len(verified_inline_sources),
                        "primary": len([s for s in verified_inline_sources if s.get("is_primary")]),
                    },
                )
                final_sources.extend(verified_inline_sources)
            return final_sources

        if inline_sources:
            logger.info(
                "[Pipeline] No claims extracted, adding %d inline sources as secondary",
                len(inline_sources),
            )
            for src in inline_sources:
                src["is_primary"] = False
                src["is_relevant"] = True
            final_sources.extend(inline_sources)

        return final_sources

    def _extract_url_anchors(self, text: str, exclude_url: str | None = None) -> list[dict]:
        """Extract URL-anchor pairs from article text.
        
        Finds URLs with their surrounding context (anchor text) that can be
        used to identify if the URL reference survives LLM cleaning.
        
        Args:
            text: Raw article text with URLs
            exclude_url: URL to exclude (e.g., the article's own URL)
            
        Returns:
            List of dicts with 'url', 'anchor', and 'domain' keys
        """
        return extract_url_anchors(text, exclude_url=exclude_url)
    
    def _restore_urls_from_anchors(self, cleaned_text: str, url_anchors: list[dict]) -> list[dict]:
        """Find which URL anchors survived cleaning and return them as sources.
        
        Args:
            cleaned_text: LLM-cleaned article text
            url_anchors: List of URL-anchor pairs from _extract_url_anchors
            
        Returns:
            List of source dicts for anchors that survived in cleaned text
        """
        return restore_urls_from_anchors(cleaned_text, url_anchors)


    def _is_url(self, text: str) -> bool:
        return is_url_input(text)

    async def _resolve_url_content(self, url: str) -> str | None:
        """Fetch URL content via Tavily Extract. Cleaning happens in claim extraction."""
        return await resolve_url_content(self.search_mgr, url, log=logger)

    # ─────────────────────────────────────────────────────────────────────────
    # M64: Topic-Aware Round-Robin Query Selection ("Coverage Engine")
    # ─────────────────────────────────────────────────────────────────────────
    
    def _select_diverse_queries(
        self,
        claims: list,
        max_queries: int = 3,
        fact_fallback: str = ""
    ) -> list[str]:
        return select_diverse_queries(
            claims,
            max_queries=max_queries,
            fact_fallback=fact_fallback,
            log=logger,
        )
    
    def _get_query_by_role(self, claim: dict, role: str) -> str | None:
        """
        M64: Extract query with specific role from claim's query_candidates.
        
        Args:
            claim: Claim dict with query_candidates and/or search_queries
            role: Query role ("CORE", "NUMERIC", "ATTRIBUTION", "LOCAL")
            
        Returns:
            Query text or None if not found
        """
        return get_query_by_role(claim, role)
    
    def _normalize_and_sanitize(self, query: str) -> str | None:
        """
        M64: Normalize query.
        
        Note: Strict gambling keywords removal is deprecated (M64).
        We rely on LLM constraints and Tavily 'topic="news"' mode 
        to prevent gambling/spam results instead of hardcoded stoplists.
        
        Args:
            query: Raw query text
            
        Returns:
            Normalized query or None if invalid
        """
        return normalize_and_sanitize(query)
    
    def _is_fuzzy_duplicate(self, query: str, existing: list[str], threshold: float = 0.9) -> bool:
        """
        M64: Check if query is >threshold similar to any existing query.
        
        Uses Jaccard similarity on word sets.
        
        Args:
            query: Query to check
            existing: List of already-selected queries
            threshold: Similarity threshold (0.9 = 90% word overlap)
            
        Returns:
            True if query is a duplicate
        """
        return is_fuzzy_duplicate(query, existing, threshold=threshold, log=logger)


    def _can_add_search(self, model, search_type, max_cost):
        # Speculative cost check
        current = self.search_mgr.calculate_cost(model, search_type)
        # Add cost of 1 search
        step_cost = int(SEARCH_COSTS.get(search_type, 80))
        return self.search_mgr.can_afford(current + step_cost, max_cost)

    def _finalize_oracle(self, oracle_res: dict, fact: str) -> dict:
        """Format oracle result for return (legacy)."""
        oracle_res["text"] = fact
        oracle_res["search_meta"] = self.search_mgr.get_search_meta()
        return oracle_res

    async def _finalize_oracle_hybrid(
        self, 
        oracle_result: dict, 
        fact: str, 
        lang: str = "en", 
        progress_callback=None
    ) -> dict:
        """
        M63: Format Oracle JACKPOT result for immediate return.
        M67: Added lang parameter for localization support.
        M69: Added granular progress updates (localizing_content).
        
        Converts OracleCheckResult to FactCheckResponse format.
        """
        oracle_result.get("status", "MIXED")
        rating = oracle_result.get("rating", "")
        publisher = oracle_result.get("publisher", "Fact Check")
        url = oracle_result.get("url", "")
        claim_reviewed = oracle_result.get("claim_reviewed", "")
        summary = oracle_result.get("summary", "")
        
        # M67: Use LLM-determined scores from OracleValidationSkill (no heuristics!)
        verified_score = float(oracle_result.get("verified_score", -1.0))
        danger_score = float(oracle_result.get("danger_score", -1.0))
        
        # Fallback for tests / legacy Oracle results without LLM-derived scores.
        if verified_score < 0:
            status_norm = str(oracle_result.get("status", "") or "").upper()
            rating_norm = str(oracle_result.get("rating", "") or "").upper()
            marker = status_norm or rating_norm
            if any(x in marker for x in ("REFUTED", "FALSE", "INCORRECT", "PANTS_ON_FIRE")):
                verified_score = 0.1
            elif any(x in marker for x in ("TRUE", "SUPPORTED", "CORRECT")):
                verified_score = 0.9
            else:
                verified_score = 0.5
        if danger_score < 0:
            danger_score = 0.0
        
        # Build response (English first)
        analysis = f"According to {publisher}, this claim is rated as '{rating}'. {summary}"
        rationale = f"Fact check by {publisher}: Rated as '{rating}'. {claim_reviewed}"
        
        # M67: Translate if non-English and translation_service available
        if lang and lang.lower() not in ("en", "en-us") and self.translation_service:
            if progress_callback:
                await progress_callback("localizing_content")
            
            try:
                analysis = await self.translation_service.translate(analysis, target_lang=lang)
                rationale = await self.translation_service.translate(rationale, target_lang=lang)
            except Exception as e:
                logger.warning("[Pipeline] Translation failed for Oracle result: %s", e)
                # Keep English if translation fails
        
        sources = [{
            "title": f"Fact Check by {publisher}",
            "link": url,
            "url": url,
            "domain": get_registrable_domain(url) if url else publisher.lower().replace(" ", ""),
            "snippet": f"Rating: {rating}. {summary}",
            "origin": "GOOGLE_FACT_CHECK",
            "source_type": "fact_check",
            "is_trusted": True,
        }]
        
        return {
            "verified_score": verified_score,
            "confidence_score": 1.0,
            "danger_score": danger_score,
            "context_score": 1.0,
            "style_score": 1.0,
            "analysis": analysis,
            "rationale": rationale,
            "sources": sources,
            "cost": 0,  # Oracle is free!
            "rgba": [danger_score, verified_score, 1.0, 1.0],
            "text": fact,
            "search_meta": self.search_mgr.get_search_meta(),
            "oracle_jackpot": True,  # M63: Flag for frontend
        }

    def _create_oracle_source(self, oracle_result: dict) -> dict:
        """
        M63: Create source dict from Oracle result for EVIDENCE scenario.
        
        This source is added to the evidence pack as a Tier A (high trust) source.
        """
        url = oracle_result.get("url", "")
        publisher = oracle_result.get("publisher", "Fact Check")
        rating = oracle_result.get("rating", "")
        claim_reviewed = oracle_result.get("claim_reviewed", "")
        summary = oracle_result.get("summary", "")
        relevance = oracle_result.get("relevance_score", 0.0)
        status = oracle_result.get("status", "MIXED")
        
        return {
            "url": url,
            "domain": get_registrable_domain(url) if url else publisher.lower().replace(" ", ""),
            "title": f"Fact Check: {claim_reviewed[:50]}..." if len(claim_reviewed) > 50 else f"Fact Check: {claim_reviewed}",
            "content": f"{publisher} rated this claim as '{rating}': {summary}",
            "snippet": f"Rating: {rating}. {summary[:200]}",
            "source_type": "fact_check",
            "is_trusted": True,
            "origin": "GOOGLE_FACT_CHECK",
            # M63: Oracle metadata for transparency in scoring
            "oracle_metadata": {
                "relevance_score": relevance,
                "status": status,
                "publisher": publisher,
                "rating": rating,
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # M70: Schema-First Query Generation (Assertion-Based)
    # ─────────────────────────────────────────────────────────────────────────

    def _select_queries_from_claim_units(
        self,
        claim_units: list,
        max_queries: int = 3,
        fact_fallback: str = "",
    ) -> list[str]:
        """
        M70: Generate search queries from structured ClaimUnits.
        
        Key difference from legacy:
        - Only FACT assertions generate verification queries
        - CONTEXT assertions are informational (no refutation search)
        - Queries are built from assertion_key + value
        
        Args:
            claim_units: List of ClaimUnit objects (from schema)
            max_queries: Maximum queries to return
            fact_fallback: Fallback text if no queries generated
            
        Returns:
            List of search queries
        """
        return select_queries_from_claim_units(
            claim_units,
            max_queries=max_queries,
            fact_fallback=fact_fallback,
            log=logger,
        )

    def _build_assertion_query(self, unit, assertion) -> str | None:
        """
        M70: Build search query for a specific assertion.
        
        Query structure: "{subject} {assertion.value} {context}"
        
        Examples:
        - event.location.city: "Joshua Paul fight Miami official location"
        - numeric.value: "Bitcoin price $42000 official"
        - event.time: "Joshua Paul fight March 2025 date confirmed"
        
        Args:
            unit: ClaimUnit containing the assertion
            assertion: Assertion to build query for
            
        Returns:
            Search query string or None
        """
        return build_assertion_query(unit, assertion)

    def _get_claim_units_for_evidence_mapping(
        self,
        claim_units: list,
        sources: list[dict],
    ) -> dict[str, list[str]]:
        """
        M70: Map sources to assertion_keys for targeted verification.
        
        This is used by clustering to understand which assertion
        each piece of evidence relates to.
        
        Returns:
            Dict of claim_id -> list of assertion_keys that need evidence
        """
        return get_claim_units_for_evidence_mapping(claim_units, sources)
