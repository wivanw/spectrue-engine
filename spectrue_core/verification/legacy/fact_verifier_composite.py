from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.tools.search_tool import WebSearchTool
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.verification.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.verification.evidence_pack import (
    ArticleContext, Claim, ClaimMetrics, ConfidenceConstraints,
    EvidenceMetrics, EvidencePack, SearchResult,
)
from spectrue_core.config import SpectrueConfig
import asyncio
import re
import logging
from datetime import datetime
from spectrue_core.utils.runtime import is_local_run
from spectrue_core.utils.trace import Trace
from uuid import uuid4
from spectrue_core import __version__ as ENGINE_VERSION, PROMPT_VERSION, SEARCH_STRATEGY_VERSION

logger = logging.getLogger(__name__)

RAG_ONLY_COST = 0
SEARCH_COSTS = {"basic": 80, "advanced": 160}
MODEL_COSTS = {"gpt-5-nano": 5, "gpt-5-mini": 20, "gpt-5.2": 100}


class FactVerifierComposite:
    def __init__(self, config: SpectrueConfig):
        self.config = config
        self.search_tool = WebSearchTool(config)
        self.google_tool = GoogleFactCheckTool(config)
        self.google_cse_tool = GoogleCSESearchTool(config)
        self.agent = FactCheckerAgent(config)
        self.time_sensitive_ttl = 300

    def _normalize_search_query(self, query: str) -> str:
        q = (query or "").strip()
        q = re.sub(r"\s+", " ", q).strip()
        q = q.strip("“”„«»\"'`")
        q = q.replace("…", " ").strip()
        q = re.sub(r"\s+", " ", q).strip()
        # FIX 3.2: Remove trailing "date" or "time" if query is long enough
        if len(q) > 20:
             q = re.sub(r"\s+(date|time|when)$", "", q, flags=re.IGNORECASE).strip()
        if len(q) > 256:
            q = q[:256].strip()
        return q

    def _is_mixed_script(self, text: str) -> bool:
        s = text or ""
        has_latin = re.search(r"[A-Za-z]", s) is not None
        has_cyr = re.search(r"[А-Яа-яІіЇїЄєҐґ]", s) is not None
        return has_latin and has_cyr


    def _clamp(self, value) -> float:
        """Clamp value between 0.0 and 1.0."""
        try:
            if value is None:
                return 0.5
            val = float(value)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5

    def _enrich_sources_with_trust(self, sources_list: list) -> list:
        """Enrich sources with trust indicators."""
        from spectrue_core.verification.trusted_sources import TRUSTED_SOURCES
        from urllib.parse import urlparse
        
        domain_to_category: dict[str, str] = {}
        for category, domains in TRUSTED_SOURCES.items():
            for domain in domains:
                domain_to_category[domain.lower()] = category
        
        enriched = []
        for source in sources_list:
            source_copy = dict(source)
            
            url = source.get("link") or source.get("url") or ""
            try:
                parsed = urlparse(url)
                host = (parsed.netloc or "").lower()
                if host.startswith("www."):
                    host = host[4:]
                
                category = domain_to_category.get(host)
                if not category:
                    parts = host.split(".")
                    for i in range(len(parts) - 1):
                        parent = ".".join(parts[i:])
                        if parent in domain_to_category:
                            category = domain_to_category[parent]
                            break
                
                source_copy["is_trusted"] = category is not None
                source_copy["trust_category"] = category
                
            except Exception:
                source_copy["is_trusted"] = False
                source_copy["trust_category"] = None
            
            enriched.append(source_copy)
        
        return enriched

    def _registrable_domain(self, url: str) -> str | None:
        """
        Best-effort registrable domain extraction without external deps.

        This is intentionally approximate and conservative (good enough for "independent_sources" counting).
        """
        from urllib.parse import urlparse

        if not url or not isinstance(url, str):
            return None
        try:
            host = (urlparse(url).netloc or "").lower().strip()
        except Exception:
            return None
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        # Drop port if present.
        host = host.split(":")[0].strip()
        if not host or "." not in host:
            return None

        parts = [p for p in host.split(".") if p]
        if len(parts) < 2:
            return None

        # Minimal set of common 2-level public suffixes seen in our traffic.
        # Heuristic for 2-level TLDs (e.g. co.uk, com.ua):
        # If last part is 2 chars (ccTLD) AND 2nd-last is <= 3 chars (com, net, org, co, gov, ac, edu),
        # take 3 parts. Otherwise take 2.
        if len(parts) >= 3:
            last = parts[-1]
            second_last = parts[-2]
            if len(last) == 2 and len(second_last) <= 3:
                return ".".join(parts[-3:])
        return ".".join(parts[-2:])

    def _extract_source_text(self, source: dict) -> str:
        # Deterministic: use only fields we already have.
        pieces = [
            source.get("title"),
            source.get("snippet"),
            source.get("content"),
        ]
        s = " ".join([p for p in pieces if isinstance(p, str) and p.strip()])
        return re.sub(r"\s+", " ", s).strip()

    def _normalize_subject_for_match(self, subject: str) -> str:
        """
        Strict subject normalization for account-handle matching.
        Lowercase, remove whitespace/punctuation; keep only a-z0-9.
        """
        s = (subject or "").lower()
        return re.sub(r"[^a-z0-9]+", "", s)


    async def fetch_url_content(self, url: str) -> str | None:
        """Securely fetch URL content via Search Provider (Tavily)."""
        if not url:
            return None
        return await self.search_tool._fetch_extract_text(url)

    def _clean_article_text(self, text: str) -> str:
        """
        Remove common boilerplate from article text before Nano processing.
        Helps extract_claims focus on actual content, not navigation/ads.
        """
        if not text:
            return ""
        
        # Common boilerplate patterns to remove
        boilerplate_patterns = [
            r"Read more\s*\.{0,3}",
            r"Share\s+(this|on|via)\s+\w+",
            r"Subscribe\s+(to|for|now)",
            r"Sign up\s+(for|to)",
            r"Follow us\s+on",
            r"Related\s+(articles?|stories|posts)",
            r"Advertisement",
            r"Sponsored\s+content",
            r"Cookie\s+(policy|notice|consent)",
            r"Privacy\s+policy",
            r"Terms\s+(of\s+)?(use|service)",
            r"All\s+rights\s+reserved",
            r"©\s*\d{4}",
            r"\[.*?\]",  # Remove markdown links
            r"#{1,6}\s*$",  # Empty headings
        ]
        
        result = text
        for pattern in boilerplate_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        
        # Collapse multiple newlines/spaces
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = re.sub(r" {2,}", " ", result)
        
        return result.strip()


    def _build_evidence_pack(
        self,
        *,
        fact: str,
        claims: list[Claim] | None,
        sources: list[dict],
        search_results_clustered: list[SearchResult] | None = None,
        article_context: ArticleContext | None = None,
        content_lang: str | None = None,
    ) -> EvidencePack:
        """
        Build structured Evidence Pack for LLM scorer.
        
        Code computes all metrics and caps; LLM produces verdict within caps.
        This is the contract between code (evidence collector) and LLM (judge).
        """
        
        # 1. Article context (if URL was provided)
        if article_context is None:
            article_context = ArticleContext(
                text_excerpt=fact[:500] if fact else "",
                content_lang=content_lang,
            )
        
        # 2. Claims (if not provided, create single "core" claim)
        if not claims:
            claims = [
                Claim(
                    id="c1",
                    text=fact,
                    type="core",
                    importance=1.0,
                    evidence_requirement={
                        "needs_primary_source": False,
                        "needs_independent_2x": True,
                    },
                    search_queries=[],
                )
            ]
        
        # 3. Convert sources to SearchResult format with enrichments
        search_results: list[SearchResult] = []
            
        if search_results_clustered:
            search_results = search_results_clustered
            seen_domains = set()
            for r in search_results:
                d = r.get("domain") or self._registrable_domain(r.get("url") or "")
                if d:
                    r["domain"] = d
                    seen_domains.add(d)
        else:
            seen_domains = set()
            for s in (sources or []):
                url = (s.get("link") or s.get("url") or "").strip()
                domain = self._registrable_domain(url)
                
                # Detect duplicates by domain
                is_dup = domain in seen_domains if domain else False
                if domain:
                    seen_domains.add(domain)
                
                # Map source_type based on is_trusted and evidence_tier
                tier = (s.get("evidence_tier") or "").strip().upper()
                is_trusted = bool(s.get("is_trusted"))
                
                if tier == "A":
                    source_type = "primary"
                elif tier == "A'":
                    source_type = "official"
                elif tier == "B" or is_trusted:
                    # Trusted domains (Reuters, NASA, ESA, etc.) are independent media
                    source_type = "independent_media"
                elif tier == "D":
                    source_type = "social"
                else:
                    # Default to aggregator for unknown/C tier
                    source_type = "aggregator"
                
                # Determine stance (placeholder - will be computed by LLM in T167)
                stance = s.get("stance", "unclear")
                
                # Build SearchResult
                search_result = SearchResult(
                    claim_id="c1",  # Default to first claim (multi-claim in T163)
                    url=url,
                    domain=domain,
                    title=s.get("title", ""),
                    snippet=s.get("snippet", ""),
                    # Truncate content to prevent context explosion (44k→15k target)
                    content_excerpt=(s.get("content") or s.get("extracted_content") or "")[:1500],
                    published_at=s.get("published_date"),
                    source_type=source_type,  # type: ignore
                    stance=stance,  # type: ignore
                    relevance_score=float(s.get("relevance_score", 0.0) or 0.0),
                    key_snippet=None,
                    quote_matches=[],
                    is_trusted=bool(s.get("is_trusted")),
                    is_duplicate=is_dup,
                    duplicate_of=None,
                )
                search_results.append(search_result)
        
        # 4. Compute metrics
        unique_domains = len(seen_domains)
        total_sources = len(search_results)
        
        duplicate_count = sum(1 for r in search_results if r.get("is_duplicate"))
        duplicate_ratio = duplicate_count / total_sources if total_sources > 0 else 0.0
        
        # Count source types
        type_dist: dict[str, int] = {}
        for r in search_results:
            st = r.get("source_type", "unknown")
            type_dist[st] = type_dist.get(st, 0) + 1
        
        # Per-claim metrics (for now, all sources belong to c1)
        claim_metrics: dict[str, ClaimMetrics] = {}
        for claim in claims:
            cid = claim.get("id", "c1")
            claim_sources = [r for r in search_results if r.get("claim_id") == cid]
            
            # Count independent domains for this claim
            claim_domains = {r.get("domain") for r in claim_sources if r.get("domain")}
            independent = len(claim_domains)
            
            # Check for primary/official sources
            primary = any(r.get("source_type") == "primary" for r in claim_sources)
            official = any(r.get("source_type") == "official" for r in claim_sources)
            
            # Stance distribution
            stance_dist: dict[str, int] = {}
            for r in claim_sources:
                st = r.get("stance", "unclear")
                stance_dist[st] = stance_dist.get(st, 0) + 1
            
            # Coverage: ratio of sources with relevance > 0.5
            relevant = sum(1 for r in claim_sources if (r.get("relevance_score") or 0) > 0.5)
            coverage = relevant / len(claim_sources) if claim_sources else 0.0
            
            claim_metrics[cid] = ClaimMetrics(
                independent_domains=independent,
                primary_present=primary,
                official_present=official,
                stance_distribution=stance_dist,
                coverage=coverage,
                freshness_days_median=None,  # TODO: compute from published_at
                source_type_distribution=type_dist,  # type: ignore
            )
        
        # Overall coverage
        coverages = [m.get("coverage", 0) for m in claim_metrics.values()]
        overall_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        
        evidence_metrics = EvidenceMetrics(
            total_sources=total_sources,
            unique_domains=unique_domains,
            duplicate_ratio=duplicate_ratio,
            per_claim=claim_metrics,
            overall_coverage=overall_coverage,
            freshness_days_median=None,  # TODO: aggregate
            source_type_distribution=type_dist,
        )
        
        # 5. Initialize confidence constraints (now handled by LLM)
        # We set global_cap to 1.0 and provide no reasons for capping,
        # trusting the LLM to make nuanced judgments on evidence sufficiency.
        constraints = ConfidenceConstraints(
            cap_per_claim={},
            global_cap=1.0,
            cap_reasons=["Confidence capping is now determined by LLM discretion."],
        )
        
        # 6. Build final Evidence Pack
        pack = EvidencePack(
            article=article_context,
            original_fact=fact,
            claims=claims,
            search_results=search_results,
            metrics=evidence_metrics,
            constraints=constraints,
        )
        
        Trace.event("evidence_pack.built", {
            "total_sources": total_sources,
            "unique_domains": unique_domains,
            "domains": list(seen_domains),
            "duplicate_ratio": round(duplicate_ratio, 2),
            "global_cap": round(global_cap, 2),
            "cap_reasons": cap_reasons,
        })
        
        return pack


    async def _get_final_analysis(
        self,
        fact: str,
        context: str,
        sources_list: list,
        gpt_model: str,
        cost: int,
        lang: str,
        analysis_mode: str = "general",
        claim_decomposition: dict | None = None,
        content_lang: str | None = None,
        search_meta: dict | None = None,
        claims: list[Claim] | None = None,
        progress_callback=None,
    ) -> dict:
        # T168: Nano Stance Clustering
        search_results_clustered = None
        if claims and sources_list:
            if progress_callback:
                await progress_callback("clustering_evidence")
            try:
                # Cluster results to update claim_id and stance
                search_results_clustered = await self.agent.cluster_evidence(
                    claims, sources_list, lang=lang
                )
            except Exception as e:
                logger.warning("[Verify] Stance clustering error: %s", e)

        # Build Evidence Pack for LLM Scoring
        pack = self._build_evidence_pack(
            fact=fact,
            claims=claims,
            sources=sources_list,
            search_results_clustered=search_results_clustered,
            article_context=None,
            content_lang=content_lang or lang,
        )

        Trace.event("score.llm_input", {"claims": len(pack.get("claims") or []), "sources": len(pack.get("search_results") or [])})

        # LLM Score (T164)
        response_lang = content_lang or lang
        score_result = await self.agent.score_evidence(pack, model=gpt_model, lang=response_lang)
        
        # T169: Evidence Gap Detection
        evidence_gaps = self.agent.detect_evidence_gaps(pack)
        score_result["evidence_gaps"] = evidence_gaps
        
        # Add metadata & enrich sources
        score_result["cost"] = cost
        score_result["text"] = fact
        score_result["search_meta"] = search_meta
        score_result["sources"] = self._enrich_sources_with_trust(sources_list)

        # CRITICAL: Cap enforcement REMOVED (User request).
        # We trust the LLM to judge evidence sufficiency.
        # The 'constraints' in pack are now just advisory guidelines.
        
        global_cap = pack.get("constraints", {}).get("global_cap", 1.0)
        raw_verified = score_result.get("verified_score", 0.5)
        
        if raw_verified > global_cap:
            logger.info("[Cap] LLM score %.2f exceeds suggested cap %.2f, but ALLOWING it (LLM discretion).", raw_verified, global_cap)
        
        score_result["verified_score_raw"] = raw_verified
        score_result["cap_applied"] = False

        # RGBA Mapping
        r = self._clamp(score_result.get("danger_score", 0.0))
        g = self._clamp(score_result.get("verified_score", 0.5))
        b = self._clamp(score_result.get("style_score", 0.5))
        a = self._clamp(score_result.get("explainability_score", 0.5))
        
        score_result["rgba"] = [r, g, b, a]
        
        # API compatibility: confidence_score is alias for verified_score (G channel)
        score_result["confidence_score"] = score_result.get("verified_score", 0.5)
        
        return score_result

    async def verify_fact(
        self, 
        fact: str, 
        search_type: str, 
        gpt_model: str, 
        lang: str, 
        analysis_mode: str = "general", 
        progress_callback=None, 
        context_text: str = "", 
        preloaded_context: str = None, 
        preloaded_sources: list = None,
        content_lang: str = None,
        include_internal: bool = False,
        search_provider: str = "auto",
        max_cost: int | None = None,
    ):
        """
        Web-only verification (no RAG).
        Strategy: Oracle -> Tier 1 -> Deep Dive
        """
        trace_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{str(uuid4())[:6]}" # Trace ID is now created at a higher level (engine.py)
        # Trace.start(trace_id, runtime=self.config.runtime) # Removed: tracing initiated at SpectrueEngine.analyze_text
        # Use current trace ID if available, otherwise fallback to generated one (should not happen for top-level call)
        current_tid = Trace.current_trace_id() or trace_id
        Trace.event(
            "verify.start",
            {
                "trace_id": current_tid, # Add current trace_id to event
                "fact": fact,
                "search_type": search_type,
                "gpt_model": gpt_model,
                "lang": lang,
                "content_lang": content_lang,
                "analysis_mode": analysis_mode,
                "search_provider": search_provider,
                "include_internal": include_internal,
                "max_cost": max_cost,
                "context_text": context_text,
                "preloaded_context": preloaded_context,
                "preloaded_sources": preloaded_sources,
            },
        )
        if is_local_run():
            logger.info("[Trace] verify_fact trace_id=%s (file: data/trace/%s.jsonl)", current_tid, current_tid)
        
        search_provider = (search_provider or "auto").lower()
        if search_provider not in ("auto", "tavily", "google_cse"):
            search_provider = "auto"

        # Budget-aware gating: keep total cost for this fact within `max_cost` (credits).
        model_cost = int(MODEL_COSTS.get(gpt_model, 20) or 0)
        
        # M47: URL-mode handling.
        # If the input 'fact' is a URL, fetch it first to extract the ACTUAL claim/title.
        # This prevents the LLM from hallucinating claims based on the URL string itself.
        if fact and ("http://" in fact or "https://" in fact) and len(fact) < 500:
            from urllib.parse import urlparse
            try:
                u = urlparse(fact)
                if u.scheme in ("http", "https") and u.netloc and "." in u.netloc:
                    logger.info("[Verify] Input detected as URL: %s. Fetching content...", fact)
                    
                    if not self.search_tool:
                         logger.info("[M47] Creating WebSearchTool instance...")
                         self.search_tool = WebSearchTool(self.config)
                    
                    logger.info("[M47] Calling _fetch_extract_text for URL: %s", fact)
                    
                    # Try to fetch content (title + text)
                    extracted = await self.search_tool._fetch_extract_text(fact)
                    logger.info("[M47] Tavily Extract call done. extracted=%s, len=%d", 
                                 "None" if extracted is None else "string", 
                                 len(extracted) if extracted else 0)
                    
                    if extracted and len(extracted) > 50:
                        logger.info("[M47] Tavily Extract returned %d chars", len(extracted))
                        
                        # Clean markdown using regex (works for single-line dumps too)
                        # Note: sidebar/footer filtering removed — too brittle. 
                        # Future: use LLM Nano for content extraction if needed.

                        # 2. Find start of content using heuristics
                        # Priority: "# Title"
                        title_match = re.search(r'(#\s+.*)', extracted)
                        if title_match:
                            extracted = extracted[title_match.start():]
                        else:
                            # Fallback: remove leading images and known junk
                            extracted = re.sub(r'^\s*!\[.*?\]\(.*?\)\s*', '', extracted)
                            extracted = re.sub(r'^\s*!\[.*?\]\s*', '', extracted)

                        # 3. Clean Markdown syntax
                        # Remove all images ![alt](url)
                        extracted = re.sub(r'!\[.*?\]\(.*?\)', '', extracted)
                        extracted = re.sub(r'!\[.*?\]', '', extracted)
                        
                        # Convert links [text](url) -> text (url)
                        extracted = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', extracted)
                        
                        # Remove header markers "# " but keep text
                        extracted = re.sub(r'#+\s+', '', extracted)
                        
                        # Normalize whitespace (replace newlines/tabs with space if mixed) but try to keep paragraphs if possible
                        # For now, simplistic normalization used in test is safe
                        cleaned = re.sub(r'\s+', ' ', extracted).strip()
                        
                        if not cleaned or len(cleaned) < 100:
                            cleaned = extracted  # Fallback
                        
                        logger.info("[M47] Cleaned article: %d chars (from orig)", len(cleaned))
                        Trace.event("verify.url_resolved", {"original": fact, "cleaned_chars": len(cleaned)})
                        
                        fact = cleaned
                        context_text = cleaned
                        

            except Exception as e:
                logger.warning("[Verify] Failed to resolve URL input: %s", e)

        budget_limited = False
        per_search_cost = int(SEARCH_COSTS.get(search_type, 80) or 0)
        google_cse_cost = int(getattr(self.config.runtime.search, "google_cse_cost", 0) or 0)
        # Default: do not charge extra for Google CSE fallback to keep costs predictable.
        google_cse_cost = max(0, min(int(google_cse_cost), int(per_search_cost)))
        try:
            max_cost_int = int(max_cost) if max_cost is not None else None
        except Exception:
            max_cost_int = None

        budget_limited = False
        skipped_steps: list[str] = []

        def _billed_total_cost(*, tavily: int, google_cse: int) -> int:
            return int(model_cost) + int(per_search_cost) * int(tavily) + int(google_cse_cost) * int(google_cse)

        def _can_afford_total(total: int) -> bool:
            if max_cost_int is None:
                return True
            return int(total) <= max_cost_int

        def _can_add_tavily_calls(n: int = 1) -> bool:
            return _can_afford_total(
                _billed_total_cost(
                    tavily=int(tavily_calls) + int(n),
                    google_cse=int(google_cse_calls),
                )
            )

        def _can_add_google_cse_calls(n: int = 1) -> bool:
            return _can_afford_total(
                _billed_total_cost(
                    tavily=int(tavily_calls),
                    google_cse=int(google_cse_calls) + int(n),
                )
            )

        tavily_calls = 0
        google_cse_calls = 0
        page_fetches = 0
        cache_hit_any = False
        search_meta: dict = {}

        # Global context optimization
        if preloaded_context:
            if max_cost_int is not None and max_cost_int < model_cost:
                Trace.event("verify.result", {"path": "preloaded", "error_key": "app.error_insufficient_credits"})
                Trace.stop()
                return {
                    "error_key": "app.error_insufficient_credits",
                    "required": model_cost,
                    "remaining": max_cost_int,
                    "cost": 0,
                    "text": fact,
                    "sources": [],
                    "rgba": [0.5, 0.5, 0.5, 0.5],
                    "search": {
                        "provider": "preloaded",
                        "quality": "unknown",
                        "fallback_used": False,
                        "tavily_calls": 0,
                        "google_cse_calls": 0,
                        "page_fetches": 0,
                        "budget_limited": True,
                        "skipped_steps": ["ai_analysis"],
                    },
                }
            if progress_callback:
                await progress_callback("using_global_context")
            
            logger.debug("[Waterfall] Using preloaded global context (%d chars).", len(preloaded_context))
            context_to_use = preloaded_context[:100000]
            sources_to_use = preloaded_sources or []
            
            if progress_callback:
                await progress_callback("ai_analysis")
            
            result = await self._get_final_analysis(
                fact, context_to_use, sources_to_use[:10], gpt_model,
                model_cost, lang, analysis_mode, progress_callback=progress_callback
            )
            result["search"] = {
                "provider": "preloaded",
                "quality": "unknown",
                "fallback_used": False,
                "tavily_calls": 0,
                "google_cse_calls": 0,
                "page_fetches": 0,
            }
            if include_internal:
                result["_internal"] = {"context": context_to_use, "sources": sources_to_use[:10]}
            Trace.event("verify.result", {"path": "preloaded", "verified_score": result.get("verified_score"), "cost": result.get("cost")})
            Trace.stop()
            return result

        # Extract Claims & Generate Queries (T163 + T164)
        if progress_callback:
            await progress_callback("extracting_claims")
        
        search_queries = [fact]
        is_short_fact = len(fact) < 300
        claims = None
        
        try:
            # Clean article text before Nano processing (remove boilerplate)
            clean_fact = self._clean_article_text(fact)
            claims = await self.agent.extract_claims(clean_fact or fact, lang=lang)
            Trace.event("claim.extracted", {"count": len(claims)})
            if is_local_run():
                logger.info("[Claim] Extracted %d claims: %s", len(claims), [c.get("text") for c in claims])
            
            qs = []
            for c in claims:
                sq = c.get("search_queries")
                if sq:
                    qs.extend(sq)
            
            # Dedupe preserving order
            seen = set()
            search_queries = []
            for q in qs:
                if q and q not in seen:
                    seen.add(q)
                    search_queries.append(q)
            
            # Limit for initial search - 1 primary + 1 fallback (cost optimization)
            search_queries = search_queries[:2]
            if not search_queries:
                search_queries = [fact]
                
        except Exception as e:
            logger.warning("[Waterfall] Failed to extract claims: %s. Using fallback.", e)
            search_queries = [fact]
        
        # Use first claim as decomposition source for strict query building
        claim_decomposition = claims[0] if claims else None

        # Normalize queries (used by Oracle and search providers) to keep payloads valid and deterministic.
        search_queries = [self._normalize_search_query(q) for q in (search_queries or [])]
        if not search_queries:
            search_queries = [self._normalize_search_query(fact), self._normalize_search_query(fact)]
        if len(search_queries) == 1:
            search_queries.append(search_queries[0])
        if not search_queries[0]:
            search_queries[0] = self._normalize_search_query(fact)
        if not search_queries[1]:
            search_queries[1] = self._normalize_search_query(fact)
        
        # Oracle (Google Fact Check)
        if progress_callback:
            await progress_callback("checking_oracle")
            
        oracle_query = search_queries[0]
        logger.debug("[Waterfall] Oracle query: '%s...'", oracle_query[:100])
        
        oracle_result = await self.google_tool.search(oracle_query, content_lang or lang)
        
        if oracle_result:
            oracle_result["text"] = fact
            logger.debug("[Waterfall] ✓ Oracle hit (Google Fact Check). Stopping.")
            oracle_result["search"] = {
                "provider": "google_fact_check",
                "quality": "good",
                "fallback_used": False,
                "tavily_calls": tavily_calls,
                "google_cse_calls": google_cse_calls,
                "page_fetches": page_fetches,
            }
            oracle_result["search_cache_hit"] = False
            if include_internal:
                oracle_result["_internal"] = {
                    "context": oracle_result.get("analysis") or oracle_result.get("rationale") or "",
                    "sources": oracle_result.get("sources") or [],
                }
            Trace.event("verify.result", {"path": "oracle", "verified_score": oracle_result.get("verified_score"), "cost": oracle_result.get("cost")})
            Trace.stop()
          # TTL strategy: Default to None (standard cache). 
        # Time sensitivity heuristic removed in favor of LLM analysis.
        ttl = None
        
        if progress_callback:
            await progress_callback("searching_tier1")
        
        search_lang = content_lang if content_lang else lang
        tier1_domains = get_trusted_domains_by_lang(search_lang)
        logger.debug("[Waterfall] Tier 1 domains for lang='%s': %d domains", search_lang, len(tier1_domains))
        
        # Tier 1: Use content_lang query first (search_queries[1] if available).
        # If poor results, refine pass will use English query (search_queries[0]).
        tier1_query = search_queries[1] if len(search_queries) > 1 else search_queries[0]
        logger.debug("[Waterfall] Tier 1 query (%s): '%s...'", content_lang or lang, tier1_query[:80])
        tier1_context = ""
        tier1_sources: list[dict] = []
        tier1_tool_meta: dict = {}
        passes_detail: list[dict] = []

        if max_cost_int is not None and max_cost_int < model_cost:
            # Even the analysis step can't fit in budget; return a clean error.
            Trace.event("verify.result", {"path": "budget_gate", "error_key": "app.error_insufficient_credits"})
            Trace.stop()
            return {
                "error_key": "app.error_insufficient_credits",
                "required": model_cost,
                "remaining": max_cost_int,
                "cost": 0,
                "text": fact,
                "sources": [],
                "rgba": [0.5, 0.5, 0.5, 0.5],
                "search": {
                    "provider": "none",
                    "quality": "unknown",
                    "fallback_used": False,
                    "tavily_calls": 0,
                    "google_cse_calls": 0,
                    "page_fetches": 0,
                },
            }

        # If user doesn't have budget for even 1 web-search call, skip web search entirely.
        tier1_provider = "google_cse" if search_provider == "google_cse" else "tavily"
        can_run_tier1 = _can_add_google_cse_calls(1) if tier1_provider == "google_cse" else _can_add_tavily_calls(1)
        if not can_run_tier1:
            budget_limited = True
            skipped_steps.append("tier1_search")
        else:
            if search_provider == "google_cse":
                tier1_context, tier1_sources = await self.google_cse_tool.search(
                    tier1_query, lang=search_lang, max_results=7, ttl=ttl
                )
                google_cse_calls += 1
                cache_hit_any = cache_hit_any or bool(self.google_cse_tool.last_cache_hit)
                tier1_tool_meta = {
                    "provider": "google_cse",
                    "quality": "good" if len(tier1_sources) >= 3 else "poor",
                    "fallback_used": False,
                }
                passes_detail.append(
                    {
                        "pass": "tier1",
                        "provider": "google_cse",
                        "queries": [tier1_query],
                        "domains_count": len(tier1_domains or []),
                        "lang": search_lang,
                        "meta": {"sources_count": len(tier1_sources or [])},
                    }
                )
            else:
                # FIX 3.1: Always start with basic depth
                tier1_depth = "basic"
                Trace.event(
                    "search.tavily.start",
                    {
                        "query": tier1_query,
                        "lang": search_lang,
                        "depth": tier1_depth,
                        "ttl": ttl,
                        "domains_count": len(tier1_domains or []),
                    },
                )
                tier1_context, tier1_sources = await self.search_tool.search(
                    tier1_query,
                    search_depth=tier1_depth,
                    ttl=ttl,
                    domains=tier1_domains,
                    lang=search_lang,
                )
                Trace.event(
                    "search.tavily.done",
                    {
                        "context_chars": len(tier1_context or ""),
                        "sources_count": len(tier1_sources or []),
                        "meta": self.search_tool.last_search_meta,
                        "cache_hit": bool(self.search_tool.last_cache_hit),
                    },
                )
                tavily_calls += 1
                cache_hit_any = cache_hit_any or bool(self.search_tool.last_cache_hit)
                page_fetches += int((self.search_tool.last_search_meta or {}).get("page_fetches") or 0)
                tier1_tool_meta = dict(self.search_tool.last_search_meta or {})
                passes_detail.append(
                    {
                        "pass": "tier1",
                        "provider": "tavily",
                        "queries": [tier1_query],
                        "domains_count": len(tier1_domains or []),
                        "lang": search_lang,
                        "meta": {
                            "best_relevance": tier1_tool_meta.get("best_relevance"),
                            "avg_relevance_top5": tier1_tool_meta.get("avg_relevance_top5"),
                            "sources_count": tier1_tool_meta.get("sources_count"),
                        },
                    }
                )

        def _dedupe_sources(primary: list[dict], extra: list[dict], *, limit: int = 10) -> list[dict]:
            all_sources = list(primary or []) + list(extra or [])
            seen = set()
            merged: list[dict] = []
            for s in all_sources:
                link = (s.get("link") or "").strip()
                if not link or link in seen:
                    continue
                seen.add(link)
                merged.append(s)
                if len(merged) >= limit:
                    break
            return merged

        quality = tier1_tool_meta.get("quality") or ("good" if len(tier1_sources) >= 3 else "poor")
        poor = (quality == "poor") or (len(tier1_sources) < 3)
        fallback_used = False
        fallback_provider: str | None = None
        passes: list[str] = ["tier1"]

        # Multi-pass search escalation (per SPEC KIT):
        # If relevance is low, trigger a refined Tier 1 pass with stricter anchoring.
        # `fallback_used` MUST mean a real strategy change (query regeneration), not a provider retry.
        try:
            best_rel = float(tier1_tool_meta.get("best_relevance"))
        except Exception:
            best_rel = None
        try:
            avg_rel = float(tier1_tool_meta.get("avg_relevance_top5"))
        except Exception:
            avg_rel = None

        needs_anchor_refine = (
            (best_rel is not None and best_rel < 0.35)
            or (avg_rel is not None and avg_rel < 0.30)
        )
        if needs_anchor_refine and tier1_provider == "tavily":
            # Default: deterministic anchored refinement.
            refine_query = ""
            refine_pass = "anchored_refine"
            refine_event_key = "anchored_refine"

            # For short claims, avoid an extra pre-search LLM call by default.
            # Only attempt LLM query regeneration when it's likely to pay off (deep/advanced),
            # and only after we already observed low relevance from Tier-1 search.
            try:
                allow_llm_refine = is_short_fact and (analysis_mode == "deep" or search_type == "advanced")
            except Exception:
                allow_llm_refine = False

            if allow_llm_refine:
                try:
                    llm_queries = await self.agent.generate_search_queries(
                        fact,
                        context=context_text,
                        lang=lang,
                        content_lang=content_lang,
                        allow_short_llm=True,
                    )
                    if llm_queries:
                        refine_query = llm_queries[1] if (content_lang and len(llm_queries) > 1) else llm_queries[0]
                        refine_query = self._normalize_search_query(refine_query)
                        if refine_query:
                            refine_pass = "llm_query_regen"
                            refine_event_key = "llm_query_regen"
                except Exception as e:
                    logger.debug("[Waterfall] LLM query regeneration failed; falling back to deterministic refine. %s", e)

            if not refine_query:
                # Tier1 used content_lang query, so refine uses English query (index 0)
                if claims and claims[0].get("search_queries"):
                    claim_queries = claims[0].get("search_queries") or []
                    # Use first query (English) for refine
                    refine_query = claim_queries[0] if claim_queries else ""
                    refine_query = self._normalize_search_query(refine_query)
                
                # Fallback to search_queries[0] if still empty
                if not refine_query:
                    refine_query = search_queries[0] if search_queries else tier1_query

            if _can_add_tavily_calls(1):
                if progress_callback:
                    await progress_callback("searching_deep")
                if is_local_run():
                    logger.info(
                        "[Waterfall] Low relevance (best=%s avg_top5=%s). %s: %s",
                        best_rel,
                        avg_rel,
                        refine_pass,
                        refine_query[:100],
                    )
                else:
                    logger.debug(
                        "[Waterfall] Low relevance (best=%s avg_top5=%s). %s: %s",
                        best_rel,
                        avg_rel,
                        refine_pass,
                        refine_query[:100],
                    )
                Trace.event(
                    "search.tavily.start",
                    {
                        "query": refine_query,
                        "lang": search_lang,
                        "depth": search_type,
                        "ttl": ttl,
                        "domains_count": len(tier1_domains or []),
                        refine_event_key: True,
                    },
                )
                refine_context, refine_sources = await self.search_tool.search(
                    refine_query,
                    search_depth=search_type,
                    ttl=ttl,
                    domains=tier1_domains,
                    lang=search_lang,
                )
                Trace.event(
                    "search.tavily.done",
                    {
                        "context_chars": len(refine_context or ""),
                        "sources_count": len(refine_sources or []),
                        "meta": self.search_tool.last_search_meta,
                        "cache_hit": bool(self.search_tool.last_cache_hit),
                        refine_event_key: True,
                    },
                )
                tavily_calls += 1
                cache_hit_any = cache_hit_any or bool(self.search_tool.last_cache_hit)
                page_fetches += int((self.search_tool.last_search_meta or {}).get("page_fetches") or 0)

                fallback_used = True
                fallback_provider = refine_pass
                passes.append(refine_pass)
                tier1_sources = _dedupe_sources(tier1_sources, refine_sources, limit=10)
                tier1_context = f"{tier1_context}\n{refine_context}".strip()
                tier1_tool_meta = dict(self.search_tool.last_search_meta or {})
                passes_detail.append(
                    {
                        "pass": refine_pass,
                        "provider": "tavily",
                        "queries": [refine_query],
                        "domains_count": len(tier1_domains or []),
                        "lang": search_lang,
                        "meta": {
                            "best_relevance": tier1_tool_meta.get("best_relevance"),
                            "avg_relevance_top5": tier1_tool_meta.get("avg_relevance_top5"),
                            "sources_count": tier1_tool_meta.get("sources_count"),
                        },
                    }
                )

                # Re-evaluate quality after anchored refine.
                quality = tier1_tool_meta.get("quality") or ("good" if len(tier1_sources) >= 3 else "poor")
                poor = (quality == "poor") or (len(tier1_sources) < 3)
            else:
                budget_limited = True
                if "anchored_refine" not in skipped_steps:
                    skipped_steps.append("anchored_refine")

        # Smart Basic: at most 1 extra cheap fallback search, no deep dive.
        if (
            search_type == "basic"
            and poor
            and search_provider in ("auto", "tavily")
        ):
            if progress_callback:
                await progress_callback("searching_deep")

            if search_provider == "auto" and self.google_cse_tool.enabled():
                if not _can_add_google_cse_calls(1):
                    budget_limited = True
                    skipped_steps.append("fallback_google_cse")
                else:
                    cse_context, cse_sources = await self.google_cse_tool.search(
                        tier1_query, lang=search_lang, max_results=6, ttl=ttl
                    )
                    google_cse_calls += 1
                    cache_hit_any = cache_hit_any or bool(self.google_cse_tool.last_cache_hit)
                    fallback_used = True
                    fallback_provider = "google_cse"
                    passes.append("google_cse")
                    passes_detail.append(
                        {
                            "pass": "google_cse",
                            "provider": "google_cse",
                            "queries": [tier1_query],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": {"sources_count": len(cse_sources or [])},
                        }
                    )
                    tier1_sources = _dedupe_sources(tier1_sources, cse_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{cse_context}".strip()
            else:
                if not _can_add_tavily_calls(1):
                    budget_limited = True
                    skipped_steps.append("fallback_tavily")
                else:
                    # Tavily fallback (still basic): drop domain filter, keep it cheap and predictable.
                    fb_query = search_queries[1] if (content_lang and content_lang != "en") else search_queries[0]
                    Trace.event(
                        "search.tavily.start",
                        {
                            "query": fb_query,
                            "lang": search_lang,
                            "depth": "basic",
                            "ttl": ttl,
                            "domains_count": 0,
                            "fallback": True,
                        },
                    )
                    fb_context, fb_sources = await self.search_tool.search(
                        fb_query,
                        search_depth="basic",
                        ttl=ttl,
                        domains=None,
                        lang=search_lang,
                    )
                    Trace.event(
                        "search.tavily.done",
                        {
                            "context_chars": len(fb_context or ""),
                            "sources_count": len(fb_sources or []),
                            "meta": self.search_tool.last_search_meta,
                            "cache_hit": bool(self.search_tool.last_cache_hit),
                            "fallback": True,
                        },
                    )
                    tavily_calls += 1
                    cache_hit_any = cache_hit_any or bool(self.search_tool.last_cache_hit)
                    page_fetches += int((self.search_tool.last_search_meta or {}).get("page_fetches") or 0)
                    # Update tool meta so `best_relevance` / `avg_relevance_top5` reflect this real strategy change.
                    tier1_tool_meta = dict(self.search_tool.last_search_meta or {})
                    fallback_used = True
                    fallback_provider = "tavily_refine"
                    passes.append("tavily_refine")
                    passes_detail.append(
                        {
                            "pass": "tavily_refine",
                            "provider": "tavily",
                            "queries": [fb_query],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": dict(self.search_tool.last_search_meta or {}),
                        }
                    )
                    tier1_sources = _dedupe_sources(tier1_sources, fb_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{fb_context}".strip()

            # Re-evaluate quality after fallback.
            quality = "good" if len(tier1_sources) >= 3 else "poor"
            poor = (quality == "poor") or (len(tier1_sources) < 3)

        # Advanced: allow quality-gated fallback + deep dive, but only if budget allows it.
        if search_type == "advanced" and poor and search_provider in ("auto", "tavily"):
            # Optional: try Google CSE once (auto only) before deep dive, if enabled and budget allows.
            if (
                search_provider == "auto"
                and self.google_cse_tool.enabled()
            ):
                if not _can_add_google_cse_calls(1):
                    budget_limited = True
                    skipped_steps.append("fallback_google_cse")
                else:
                    cse_context, cse_sources = await self.google_cse_tool.search(
                        tier1_query, lang=search_lang, max_results=6, ttl=ttl
                    )
                    google_cse_calls += 1
                    cache_hit_any = cache_hit_any or bool(self.google_cse_tool.last_cache_hit)
                    fallback_used = True
                    fallback_provider = "google_cse"
                    passes.append("google_cse")
                    passes_detail.append(
                        {
                            "pass": "google_cse",
                            "provider": "google_cse",
                            "queries": [tier1_query],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": {"sources_count": len(cse_sources or [])},
                        }
                    )
                    tier1_sources = _dedupe_sources(tier1_sources, cse_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{cse_context}".strip()

                    quality = "good" if len(tier1_sources) >= 3 else "poor"
                    poor = (quality == "poor") or (len(tier1_sources) < 3)

            # Deep dive (EN + Native): only in Advanced and only if we can afford 2 more search calls.
            if poor and search_provider != "google_cse":
                if not _can_add_tavily_calls(2):
                    budget_limited = True
                    skipped_steps.append("deep_dive")
                else:
                    logger.debug("[Waterfall] Tier 1 weak. Running deep dive (EN + Native)...")
                    if progress_callback:
                        await progress_callback("searching_deep")

                    en_task = self.search_tool.search(
                        search_queries[0],
                        search_depth=search_type,
                        ttl=ttl,
                        lang="en",
                    )
                    native_task = self.search_tool.search(
                        search_queries[1],
                        search_depth=search_type,
                        ttl=ttl,
                        lang=search_lang,
                    )
                    (en_context, en_sources), (native_context, native_sources) = await asyncio.gather(
                        en_task, native_task
                    )
                    tavily_calls += 2
                    cache_hit_any = cache_hit_any or bool(self.search_tool.last_cache_hit)
                    page_fetches += int((self.search_tool.last_search_meta or {}).get("page_fetches") or 0)
                    # Keep meta updated (even though deep dive runs 2 searches, last_search_meta is best-effort).
                    tier1_tool_meta = dict(self.search_tool.last_search_meta or {})
                    fallback_used = True
                    fallback_provider = fallback_provider or "deep_dive"
                    passes.append("deep_dive")
                    passes_detail.append(
                        {
                            "pass": "deep_dive",
                            "provider": "tavily",
                            "queries": [search_queries[0], search_queries[1]],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": dict(self.search_tool.last_search_meta or {}),
                        }
                    )

                    # Aggregate
                    tier1_sources = _dedupe_sources(tier1_sources, en_sources + native_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{en_context}\n{native_context}".strip()

                    if len(tier1_context) > 100000:
                        tier1_context = tier1_context[:100000]

                    quality = "good" if len(tier1_sources) >= 3 else "poor"
                    poor = (quality == "poor") or (len(tier1_sources) < 3)

        # Final search metadata (for UI + billing transparency)
        def _final_relevance_metrics(sources: list[dict]) -> dict:
            rels = [
                float(s.get("relevance_score") or 0.0)
                for s in (sources or [])
                if isinstance(s.get("relevance_score"), (int, float))
            ]
            rels.sort(reverse=True)
            top = rels[:5]
            avg = (sum(top) / len(top)) if top else 0.0
            best = rels[0] if rels else 0.0
            return {"best_relevance": best, "avg_relevance_top5": avg}

        final_rel = _final_relevance_metrics(tier1_sources)
        search_meta = {
            "provider": (tier1_tool_meta.get("provider") or ("none" if not (tavily_calls or google_cse_calls) else "tavily")),
            "quality": quality if (tavily_calls or google_cse_calls) else "unknown",
            "avg_relevance_top5": tier1_tool_meta.get("avg_relevance_top5"),
            "best_relevance": tier1_tool_meta.get("best_relevance"),
            # Final relevance metrics computed from the final merged source list (useful after multi-pass).
            "avg_relevance_top5_final": final_rel.get("avg_relevance_top5"),
            "best_relevance_final": final_rel.get("best_relevance"),
            "fallback_used": bool(fallback_used),
            "fallback_provider": fallback_provider,
            "passes": passes,
            "passes_detail": passes_detail,
            "multi_pass_used": len(passes) > 1,
            "tavily_calls": tavily_calls,
            "google_cse_calls": google_cse_calls,
            "page_fetches": page_fetches,
            "budget_limited": bool(budget_limited),
            "skipped_steps": skipped_steps,
        }
        Trace.event(
            "search.passes",
            {
                "passes": passes,
                "passes_detail": passes_detail,
                "best_relevance": search_meta.get("best_relevance"),
                "avg_relevance_top5": search_meta.get("avg_relevance_top5"),
                "fallback_used": search_meta.get("fallback_used"),
            },
        )

        if progress_callback:
            await progress_callback("analyzing_evidence")

        # Cost is per Tavily call; Google CSE fallback is cheap by default (configurable).
        total_cost = _billed_total_cost(tavily=int(tavily_calls), google_cse=int(google_cse_calls))
        if max_cost_int is not None and total_cost > max_cost_int:
            # Shouldn't happen due to gating, but keep it safe.
            total_cost = max_cost_int
            budget_limited = True
            if "budget_cap" not in skipped_steps:
                skipped_steps.append("budget_cap")

        result = await self._get_final_analysis(
            fact,
            tier1_context,
            tier1_sources[:10],
            gpt_model,
            total_cost,
            lang,
            analysis_mode,
            claim_decomposition=claim_decomposition,
            content_lang=content_lang or lang,
            search_meta=search_meta,
            claims=claims,
            progress_callback=progress_callback,
        )
        result["search"] = search_meta
        result["search_cache_hit"] = bool(cache_hit_any)
        result["checks"] = {
            "engine_version": ENGINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "search_strategy": SEARCH_STRATEGY_VERSION,
        }
        if include_internal:
            result["_internal"] = {"context": tier1_context, "sources": tier1_sources[:10]}
        Trace.event(
            "verify.result",
            {
                "path": "analysis",
                "verified_score": result.get("verified_score"),
                "confidence_score": result.get("confidence_score"),
                "cost": result.get("cost"),
                "search": result.get("search"),
            },
        )
        Trace.stop()
        return result
