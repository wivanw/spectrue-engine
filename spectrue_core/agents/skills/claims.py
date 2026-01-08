# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.verification.evidence.evidence_pack import Claim, ClaimAnchor, EvidenceRequirement, ArticleIntent
from .base_skill import BaseSkill, logger
from spectrue_core.utils.text_chunking import CoverageSampler, TextChunk
from spectrue_core.utils.trace import Trace
from spectrue_core.constants import SUPPORTED_LANGUAGES
import re
from spectrue_core.agents.llm_schemas import (
    CLAIM_RETRIEVAL_SCHEMA,
    VERIFIABLE_CORE_CLAIM_SCHEMA,  # M125
)
from .claims_prompts import (
    build_core_extraction_prompt,
    build_retrieval_planning_prompt,
)
import asyncio
from .claims_parsing import (
    clamp_float,
    clamp_int,
    normalize_article_intent,
    normalize_claim_category,
    normalize_topic_group,
)
from .schema_logger import (
    validate_claim_response,
    log_claim_field_defaults,
)

# Import schema module for structured claims
from spectrue_core.schema import (
    ClaimStructureType,
)
from spectrue_core.models import DEFAULT_MODEL_OPENAI_NANO

# Import claim metadata types
from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    EvidenceChannel,
)

from spectrue_core.agents.skills.claim_metadata_parser import (
    parse_claim_metadata as parse_claim_metadata_v1,
    default_channels as default_channels_v1,
)
from spectrue_core.runtime_config import ContentBudgetConfig

# Whitelist of allowed retrieval-planning fields (everything else is dropped)
RETRIEVAL_ALLOWED_FIELDS = {
    "claim_category", "harm_potential", "verification_target", "claim_role",
    "satire_likelihood", "importance", "check_worthiness",
    "structure", "search_locale_plan", "retrieval_policy", "metadata_confidence",
    "query_candidates", "search_method", "search_queries",
    "evidence_req", "evidence_need", "check_oracle",
}

POST_EVIDENCE_ALLOWED_FIELDS: set[str] = set()


# Predicate types that don't require explicit time anchors
# - quote: verifiable via source attribution (who said it)
# - policy: verifiable via official records
# - ranking: verifiable via current datasets
# - existence: verifiable via document/entity lookup
TIME_ANCHOR_EXEMPT_PREDICATES = {"quote", "policy", "ranking", "existence"}


# Default system instructions for claim extraction (MANDATORY)
# These MUST always be present in LLM calls to prevent extraction failures.
DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS = """
You are extracting factual claims for verification.
Do NOT summarize. Do NOT judge truth.
Your task is to enumerate verifiable factual units.

Rules:
- Prefer over-extraction to under-extraction.
- Extract events, measurements, quantities, dates, and quoted statements.
- Each claim must be atomic and independently verifiable.
- Output must strictly follow the provided JSON schema.
"""


def validate_core_claim(claim: dict) -> tuple[bool, list[str]]:
    """
    Deterministic validation for verifiable claims.
    
    Validates structural requirements:
    - claim_text non-empty, <= 240 chars (or use claim_text/text field)
    - subject_entities with minItems >= 1
    - retrieval_seed_terms length between 3 and 10
    - falsifiability.is_falsifiable == True
    - time_anchor.type != "unknown" OR predicate_type in {policy, ranking, existence}
    
    Returns:
        (ok: bool, reason_codes: list[str])
    """
    reason_codes: list[str] = []
    
    # Get claim text (support both claim_text and text field names)
    claim_text = claim.get("claim_text") or claim.get("text") or ""
    
    # 1. Claim text validation
    if not claim_text:
        reason_codes.append("empty_claim_text")
    elif len(claim_text) > 500:
        reason_codes.append("claim_text_too_long")
    
    # 2. Subject entities validation
    entities = claim.get("subject_entities")
    if not entities or not isinstance(entities, list) or len(entities) < 1:
        reason_codes.append("missing_subject_entities")
    else:
        # Check for non-empty entities
        valid_entities = [e for e in entities if isinstance(e, str) and e.strip()]
        if len(valid_entities) < 1:
            reason_codes.append("invalid_subject_entities")
    
    # 3. Retrieval seed terms validation
    seed_terms = claim.get("retrieval_seed_terms")
    if not seed_terms or not isinstance(seed_terms, list):
        reason_codes.append("missing_retrieval_seed_terms")
    else:
        valid_terms = [t for t in seed_terms if isinstance(t, str) and len(t) >= 2]
        if len(valid_terms) < 3:
            reason_codes.append("insufficient_retrieval_seed_terms")
        elif len(valid_terms) > 10:
            reason_codes.append("too_many_retrieval_seed_terms")
    
    # 4. Falsifiability validation
    falsifiability = claim.get("falsifiability")
    if not falsifiability or not isinstance(falsifiability, dict):
        reason_codes.append("missing_falsifiability")
    else:
        is_falsifiable = falsifiability.get("is_falsifiable")
        if is_falsifiable is not True:
            reason_codes.append("not_falsifiable")
    
    # 5. Time anchor validation (with predicate exemption)
    predicate_type = claim.get("predicate_type", "other")
    time_anchor = claim.get("time_anchor")
    
    if predicate_type not in TIME_ANCHOR_EXEMPT_PREDICATES:
        if not time_anchor or not isinstance(time_anchor, dict):
            reason_codes.append("missing_time_anchor")
        else:
            anchor_type = time_anchor.get("type", "unknown")
            if anchor_type == "unknown":
                reason_codes.append("unknown_time_anchor")
    
    ok = len(reason_codes) == 0
    return ok, reason_codes


class ExtractionStats:
    """
    Track extraction statistics for observability.
    """
    def __init__(self):
        self.claims_extracted_total = 0
        self.claims_dropped_nonverifiable = 0
        self.claims_emitted_targets = 0
        self.drop_reason_counts: dict[str, int] = {}
    
    def record_drop(self, reason_codes: list[str]) -> None:
        self.claims_dropped_nonverifiable += 1
        for reason in reason_codes:
            self.drop_reason_counts[reason] = self.drop_reason_counts.get(reason, 0) + 1
    
    def record_emit(self) -> None:
        self.claims_emitted_targets += 1
    
    def to_trace_dict(self) -> dict:
        return {
            "claims_extracted_total": self.claims_extracted_total,
            "claims_dropped_nonverifiable": self.claims_dropped_nonverifiable,
            "claims_emitted_targets": self.claims_emitted_targets,
            "drop_reason_counts": self.drop_reason_counts,
        }


def extract_keywords_deterministic(text: str, max_tokens: int = 6) -> str:
    """
    Extract keyword query from claim text (deterministic, language-agnostic).
    
    Used as fallback when LLM returns empty search_queries.
    This is structural repair, not semantic heuristics.
    
    Algorithm:
    1. Lowercase
    2. Drop punctuation
    3. Drop tokens with length < 3
    4. Keep first N unique tokens
    5. Join with spaces
    6. If empty, return truncated original text
    """
    if not text:
        return ""
    
    # Lowercase
    lowered = text.lower()
    
    # Drop punctuation (keep Unicode word chars and spaces)
    cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    
    # Tokenize
    tokens = cleaned.split()
    
    # Filter short tokens and keep unique
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if len(t) >= 3 and t not in seen:
            unique_tokens.append(t)
            seen.add(t)
        if len(unique_tokens) >= max_tokens:
            break
    
    if unique_tokens:
        return " ".join(unique_tokens)
    
    # Ultimate fallback: truncate original text
    return text[:80].rstrip(".,!?;")


def sanitize_retrieval_response(data: dict, claim_text: str = "") -> dict:
    """
    Sanitize LLM retrieval-planning response before schema validation.
    
    This is a deterministic normalization, not a heuristic:
    1. Whitelist fields (drop unknown like premises_2, type_order)
    2. Add defaults for optional nested fields (dependencies, evidence_gaps)
    3. Merge numbered variants (premises_2 → premises)
    4. Repair empty search_queries with keyword extraction from claim_text
    """
    if not isinstance(data, dict):
        return {}
    
    # Step 1: Whitelist top-level fields
    sanitized = {k: v for k, v in data.items() if k in RETRIEVAL_ALLOWED_FIELDS}
    
    # Step 2: Sanitize nested 'structure' object
    if "structure" in sanitized and isinstance(sanitized["structure"], dict):
        struct = sanitized["structure"]
        
        # Merge premises_N → premises
        premises = struct.get("premises", [])
        if not isinstance(premises, list):
            premises = []
        for key in list(struct.keys()):
            if key.startswith("premises_") and isinstance(struct[key], list):
                premises.extend(struct[key])
                del struct[key]
        struct["premises"] = premises
        
        # Default empty arrays for optional fields
        if "dependencies" not in struct:
            struct["dependencies"] = []
        if "conclusion" not in struct and not struct.get("conclusion"):
            struct["conclusion"] = ""
        
        # Drop unknown structure fields
        allowed_struct_fields = {"type", "premises", "conclusion", "dependencies"}
        struct_sanitized = {k: v for k, v in struct.items() if k in allowed_struct_fields}
        sanitized["structure"] = struct_sanitized
    
    # Step 3: Sanitize evidence_req - add defaults
    if "evidence_req" in sanitized and isinstance(sanitized["evidence_req"], dict):
        ev_req = sanitized["evidence_req"]
        if "evidence_gaps" not in ev_req:
            ev_req["evidence_gaps"] = []
        # Drop unknown fields in evidence_req
        allowed_ev_fields = {"needs_primary", "needs_2_independent", "evidence_gaps"}
        sanitized["evidence_req"] = {k: v for k, v in ev_req.items() if k in allowed_ev_fields}
    
    # Step 4: Sanitize retrieval_policy
    if "retrieval_policy" in sanitized and isinstance(sanitized["retrieval_policy"], dict):
        rp = sanitized["retrieval_policy"]
        allowed_rp_fields = {"channels_allowed"}
        sanitized["retrieval_policy"] = {k: v for k, v in rp.items() if k in allowed_rp_fields}
    
    # Step 5: Repair empty search_queries using keyword extraction
    queries = sanitized.get("search_queries", [])
    if not queries or not isinstance(queries, list):
        # Generate fallback query from claim text
        if claim_text:
            fallback_query = extract_keywords_deterministic(claim_text)
            if fallback_query:
                sanitized["search_queries"] = [fallback_query]
                Trace.event("retrieval.search_queries.repaired", {
                    "original_count": 0,
                    "repaired_query": fallback_query[:80],
                })
    
    return sanitized


def sanitize_post_evidence_response(data: dict) -> dict:
    """
    Sanitize post-evidence enrichment response before schema validation.
    """
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in POST_EVIDENCE_ALLOWED_FIELDS}


class ClaimExtractionSkill(BaseSkill):

    # M73.5: Dynamic timeout constants
    BASE_TIMEOUT_SEC = 60.0     # Minimum timeout (Increased for DeepSeek)
    TIMEOUT_PER_1K_CHARS = 3.0  # Additional seconds per 1000 chars
    MAX_TIMEOUT_SEC = 120.0     # Maximum timeout cap

    def __init__(self, config, llm_client):
        super().__init__(config, llm_client)
        cfg = getattr(self.runtime, "content_budget", None)
        self._budget_config = cfg if isinstance(cfg, ContentBudgetConfig) else ContentBudgetConfig()
        self._claim_excerpt_budget = min(int(self._budget_config.max_clean_text_chars_default), 12_000)
        self._coverage_sampler = CoverageSampler()

    def _calculate_timeout(self, text_len: int, *, base_offset: float = 0.0) -> float:
        """
        Calculate dynamic timeout based on text length.
        
        Args:
            text_len: Length of input text in characters
            base_offset: Additional base time for complex operations (e.g., structured extraction)
        """
        extra = (text_len / 1000) * self.TIMEOUT_PER_1K_CHARS
        timeout = self.BASE_TIMEOUT_SEC + base_offset + extra
        return min(timeout, self.MAX_TIMEOUT_SEC + base_offset)

    def _chunk_for_claims(self, text: str) -> tuple[list[TextChunk], str]:
        """
        Split text into manageable chunks for claim extraction without hard-capping total length.

        Returns (chunks, stitched_text) where stitched_text preserves all chunks with separators.
        """
        stripped = (text or "").strip()
        if not stripped:
            return [], ""

        chunks = self._coverage_sampler.chunk(stripped, max_chunk_chars=self._claim_excerpt_budget)
        stitched = self._coverage_sampler.merge([c.text for c in chunks])
        return chunks, stitched


    async def extract_claims(
        self,
        text: str,
        *,
        chunks: list[TextChunk] | None = None,  # M74
        lang: str = "en",
        max_claims: int = 5,
    ) -> tuple[list[Claim], bool, ArticleIntent, str]:
        """
        Extract atomic verifiable claims from article text with chunked, deterministic coverage.
        
        Refactored to 2-stage pipeline (M120):
        1. Core Extraction: Fetch bare claims (text + normalized) from chunks.
        2. Metadata Enrichment: Fan-out parallel processing for classification & strategy.
        """
        text = (text or "").strip()
        if not text:
            return [], False, "news", ""

        chunks_res, stitched_text = self._chunk_for_claims(text)
        chunks = chunks or chunks_res
        if not chunks:
            return [], False, "news", stitched_text


        try:
            # --- Stage 1: Core Extraction ---
            core_tasks = []
            for chunk in chunks:
                core_tasks.append(self._extract_core_from_chunk(chunk))
            
            core_results = await asyncio.gather(*core_tasks, return_exceptions=True)
            
            # Aggregate extraction statistics
            aggregated_stats = ExtractionStats()
            
            # Flatten and filter results
            valid_core_claims: list[tuple[dict, str]] = []  # (claim_dict, chunk_text_context)
            overall_intent: ArticleIntent = "news"
            
            for i, res in enumerate(core_results):
                if isinstance(res, Exception):
                    logger.error("[Claims] Core extraction failed for chunk %d: %s", i, res)
                    continue
                
                # Unpack 3-tuple (claims, intent, stats)
                chunk_claims, chunk_intent, chunk_stats = res
                
                if i == 0:
                    overall_intent = chunk_intent
                
                # Aggregate stats
                aggregated_stats.claims_extracted_total += chunk_stats.claims_extracted_total
                aggregated_stats.claims_dropped_nonverifiable += chunk_stats.claims_dropped_nonverifiable
                aggregated_stats.claims_emitted_targets += chunk_stats.claims_emitted_targets
                for reason, count in chunk_stats.drop_reason_counts.items():
                    aggregated_stats.drop_reason_counts[reason] = (
                        aggregated_stats.drop_reason_counts.get(reason, 0) + count
                    )
                
                chunk_text = chunks[i].text
                for c in chunk_claims:
                    valid_core_claims.append((c, chunk_text))
            
            # Emit aggregated extraction stats
            Trace.event("claim_extraction.stats", aggregated_stats.to_trace_dict())
            
            if not valid_core_claims:
                logger.warning("[Claims] No core claims extracted. Using fallback.")
                fallback_text = text[:300] + "..." if len(text) > 300 else text
                return [
                    Claim(
                        id="c1",
                        text=fallback_text,
                        normalized_text=fallback_text,
                        type="core",
                        topic_group="Other",
                        topic_key="General",
                        importance=1.0,
                        check_worthiness=0.5,
                        evidence_requirement=EvidenceRequirement(
                            needs_primary_source=False,
                            needs_independent_2x=True
                        ),
                        search_queries=[],
                        query_candidates=[],
                        metadata=None,
                    )
                ], False, "news", stitched_text

            # --- Stage 2: Enrichment (with rate limiting) ---
            MAX_CONCURRENT_ENRICHMENTS = 5
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_ENRICHMENTS)
            
            async def rate_limited_enrich(claim_id: str, core_data: dict, context_text: str, lang: str):
                async with semaphore:
                    return await self._enrich_claim(
                        claim_id=claim_id,
                        core_data=core_data,
                        context_text=context_text,
                        lang=lang
                    )
            
            enrich_tasks = []
            
            for idx, (core_claim, context_text) in enumerate(valid_core_claims):
                claim_id = f"c{idx+1}"
                enrich_tasks.append(
                    rate_limited_enrich(
                        claim_id=claim_id,
                        core_data=core_claim,
                        context_text=context_text,
                        lang=lang
                    )
                )

            enrich_results = await asyncio.gather(*enrich_tasks, return_exceptions=True)
            
            enriched_claims: list[Claim] = []
            should_check_oracle_agg = False
            
            for res in enrich_results:
                if isinstance(res, Exception):
                    logger.error("[Claims] Enrichment failed: %s", res)
                    continue
                if res:
                    enriched_claims.append(res)
                    if res.get("check_oracle"):
                        should_check_oracle_agg = True

            # Dedupe and Sort
            final_claims = self._dedupe_claims(enriched_claims)
            final_claims.sort(key=lambda x: x.get("harm_potential", 1), reverse=True)
            
            # Tracing
            self._trace_extracted_claims(final_claims)
            self._trace_metadata_distribution(final_claims)
            
            return final_claims, should_check_oracle_agg, overall_intent, stitched_text
        
        except Exception as e:
            logger.exception("Critical failure in split claim extraction: %s", e)
            raise

    async def _extract_core_from_chunk(self, chunk: TextChunk) -> tuple[list[dict], ArticleIntent, ExtractionStats]:
        """
        Stage 1: Core claim extraction using DeepSeek (temp=0).
        
        Uses VERIFIABLE_CORE_CLAIM_SCHEMA and validates each claim.
        Drops non-verifiable claims with trace events.
        
        Returns:
            (valid_claims, article_intent, extraction_stats)
        """
        stats = ExtractionStats()
        prompt = build_core_extraction_prompt(text_excerpt=chunk.text)
        
        # Hard guard - instructions MUST always be present
        # The prompt returned by build_core_extraction_prompt is the user/input message.
        # We inject system instructions separately to ensure they are never empty.
        instructions = DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS
        instructions_injected_fallback = False
        
        # Trace guard: log if we're using fallback (this should always be the case now)
        Trace.event("claim_extraction.guard.instructions_injected", {
            "instructions_len": len(instructions),
            "fallback_used": True,  # We always inject now as a safety measure
        })
        
        # We assume 1-2 attempts is enough for this simpler task
        result = await self.llm_client.call_json(
            model=self.runtime.llm.model_claim_extraction,  # DeepSeek
            input=prompt,
            instructions=instructions,  # MANDATORY instructions
            response_schema=VERIFIABLE_CORE_CLAIM_SCHEMA,  # verifiable schema
            reasoning_effort="low",
            cache_key=f"core_extract_v3_{hash(chunk.text)}",  # v3 for mandatory instructions
            timeout=self._calculate_timeout(len(chunk.text)),
            trace_kind="claim_extraction_core",
            temperature=0.0,  # Strict determinism
        )
        
        raw_claims = result.get("claims", [])
        stats.claims_extracted_total = len(raw_claims)
        
        valid_claims: list[dict] = []
        
        for idx, claim in enumerate(raw_claims):
            if not isinstance(claim, dict):
                continue
            
            # Deterministic validation
            ok, reason_codes = validate_core_claim(claim)
            
            if ok:
                # Normalize field names for compatibility
                # New schema uses "claim_text", old code expects "text"
                if "claim_text" in claim and "text" not in claim:
                    claim["text"] = claim["claim_text"]
                
                valid_claims.append(claim)
                stats.record_emit()
            else:
                stats.record_drop(reason_codes)
                
                # Emit trace event for dropped claim
                claim_preview = (claim.get("claim_text") or claim.get("text") or "")[:100]
                Trace.event("claim.dropped", {
                    "temp_id": f"drop_{idx}",
                    "reason_codes": reason_codes,
                    "claim_preview": claim_preview,
                    "predicate_type": claim.get("predicate_type", "unknown"),
                })
                logger.debug(
                    "Dropped claim: reasons=%s, preview=%s",
                    reason_codes, claim_preview[:50]
                )
        
        intent = normalize_article_intent(result.get("article_intent", "news"))
        return valid_claims, intent, stats

    async def _enrich_claim(self, claim_id: str, core_data: dict, context_text: str, lang: str) -> Claim | None:
        """
        Stage 2: Retrieval planning using Nano.
        """
        claim_text = core_data.get("text", "")
        if not claim_text:
            return None
            
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")
        
        prompt = build_retrieval_planning_prompt(
            claim_text=claim_text,
            article_context_sm=context_text,
            lang_name=lang_name
        )
        
        # NOTE: Using simple instructions for enrichment, not the full claim_strategist ones
        # The full prompt already contains all necessary context
        instructions = (
            "You are producing retrieval-planning metadata for a single claim. "
            "Output only the JSON object with the required fields. Do NOT wrap in claims[] array."
        )

        try:
            # Using Nano for enrichment to save cost
            result = await self.llm_client.call_json(
                model=DEFAULT_MODEL_OPENAI_NANO,
                input=prompt,
                instructions=instructions,
                response_schema=CLAIM_RETRIEVAL_SCHEMA,
                reasoning_effort="low",
                cache_key=f"retrieval_plan_v1_{hash(claim_text)}",
                timeout=45.0, # Nano is faster usually
                trace_kind="claim_retrieval_plan",
                temperature=0.0, # Keep steady
            )
        except Exception as e:
            logger.warning("Enrichment failed for %s: %s", claim_id, e)
            # Return minimal claim if meaningful
            result = {}

        # Fallback: unwrap if LLM returned {"claims": [...]} or {"article_intent": ...}
        if "claims" in result and isinstance(result.get("claims"), list):
            claims_list = result["claims"]
            if claims_list:
                result = claims_list[0]  # Take first claim's enrichment
                logger.debug("[Claims] Unwrapped claims[] wrapper for %s", claim_id)
            else:
                result = {}
        # Also strip unexpected wrapper fields
        result.pop("article_intent", None)
        result.pop("text", None)
        result.pop("normalized_text", None)

        # Sanitize enrichment response (whitelist, defaults, merge, repair empty queries)
        result = sanitize_retrieval_response(result, claim_text=claim_text)

        # Merge core data + enriched data
        merged = {**core_data, **result}
        
        # Validate defaults
        defaults_used, invalid_fields = validate_claim_response(merged, claim_id)
        if defaults_used or invalid_fields:
             log_claim_field_defaults(
                 claim_id=claim_id,
                 defaults_used=defaults_used,
                 claim_text=claim_text
             )
        
        req_raw = merged.get("evidence_req", {})
        req = EvidenceRequirement(
            needs_primary_source=bool(req_raw.get("needs_primary")),
            needs_independent_2x=bool(req_raw.get("needs_2_independent")),
            needs_quote_verification=bool(req_raw.get("needs_quote")),
            is_time_sensitive=bool(req_raw.get("needs_recent")),
        )

        normalized = merged.get("normalized_text") or claim_text
        topic = normalize_topic_group(merged.get("topic_group", "Other") or "Other")
        topic_key = merged.get("topic_key") or claim_id or topic

        worthiness = merged.get("check_worthiness")
        if worthiness is None:
            worthiness = merged.get("importance", 0.5)
        worthiness = clamp_float(worthiness, default=0.5, lo=0.0, hi=1.0)

        harm_potential = clamp_int(merged.get("harm_potential", 1), default=1, lo=1, hi=5)
        claim_category = normalize_claim_category(merged.get("claim_category", "FACTUAL"))
        satire_likelihood = clamp_float(merged.get("satire_likelihood", 0.0), default=0.0, lo=0.0, hi=1.0)

        metadata = self._parse_claim_metadata(merged, lang, harm_potential, claim_category, satire_likelihood)
        
        structure = self._parse_claim_structure(
            merged,
            fallback_conclusion=normalized,
        )
        
        claim_role = (
            metadata.claim_role.value
            if metadata
            else str(merged.get("claim_role", "core")).lower()
        )

        search_queries = merged.get("search_queries", [])
        query_candidates = merged.get("query_candidates", [])
        should_skip_search = (
            satire_likelihood >= 0.8
            or claim_category == "SATIRE"
            or metadata.should_skip_search
        )
        if should_skip_search:
            search_queries = []
            query_candidates = []
        
        # Anchor fallback
        anchor = self._locate_anchor_in_text(context_text, [claim_text, normalized])
        if not anchor:
             anchor = {
                "chunk_id": "unknown", 
                "char_start": 0, 
                "char_end": len(claim_text), 
                "section_path": []
            }

        c = Claim(
            id=claim_id,
            text=claim_text,
            language=lang,
            type=merged.get("type", "core"),
            importance=float(merged.get("importance", 0.5)),
            evidence_requirement=req,
            search_queries=search_queries,
            check_oracle=bool(merged.get("check_oracle", False)),
            normalized_text=normalized,
            topic_group=topic,
            check_worthiness=worthiness,
            topic_key=topic_key,
            query_candidates=query_candidates,
            search_method=merged.get("search_method", "general_search"),
            evidence_need=merged.get("evidence_need", "unknown"),
            anchor=anchor,
            harm_potential=harm_potential,
            claim_category=claim_category,
            satire_likelihood=satire_likelihood,
            metadata=metadata,
            claim_role=claim_role,
            structure=structure,
            verification_target=(
                metadata.verification_target.value
                if metadata
                else str(merged.get("verification_target", "reality")).lower()
            ),
            role=claim_role,
            temporality=merged.get("temporality"),
            locale_plan=merged.get("locale_plan") or ({
                    "ui_locale": lang,
                    "content_lang": lang,
                    "context_lang": lang,
                    "primary": metadata.search_locale_plan.primary,
                    "fallbacks": metadata.search_locale_plan.fallback,
                    "justification": "derived"
                } if metadata else None),
            metadata_confidence=metadata.metadata_confidence.value if metadata else "medium",
        )
        return c

    def _trace_extracted_claims(self, claims: list[Claim]) -> None:
        """
        M81/T4: Emit trace event with individual claim details for debugging.
        
        Logs first 100 chars of each claim text with metadata.
        Critical for diagnosing attribution/reality misclassification.
        """
        if not claims:
            return

        claims_data = []
        for c in claims[:7]:  # Max 7 claims
            metadata = c.get("metadata")
            claims_data.append({
                "id": c.get("id", "?"),
                "text": c.get("text", "")[:100],  # First 100 chars
                "verification_target": metadata.verification_target.value if metadata else "?",
                "claim_role": metadata.claim_role.value if metadata else "?",
                "check_worthiness": c.get("check_worthiness", 0),
                "metadata_confidence": metadata.metadata_confidence.value if metadata else "?",
            })

        Trace.event("claim_extraction.claims_extracted", {
            "count": len(claims),
            "claims": claims_data,
        })

    def _trace_metadata_distribution(self, claims: list[Claim]) -> None:
        """
        Trace: Emit trace event with metadata distribution for debugging.
        
        Logs distribution counts for:
        - verification_target: {reality: N, attribution: M, existence: K, none: L}
        - claim_role: {core: N, support: M, context: K, ...}
        - metadata_confidence: {low: N, medium: M, high: K}
        """
        if not claims:
            return

        # Count verification_target distribution
        target_dist: dict[str, int] = {"reality": 0, "attribution": 0, "existence": 0, "none": 0}
        role_dist: dict[str, int] = {}
        confidence_dist: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        skip_search_count = 0

        for claim in claims:
            metadata = claim.get("metadata")
            if metadata:
                # Target distribution
                target = metadata.verification_target.value
                target_dist[target] = target_dist.get(target, 0) + 1

                # Role distribution
                role = metadata.claim_role.value
                role_dist[role] = role_dist.get(role, 0) + 1

                # Confidence distribution
                confidence = metadata.metadata_confidence.value
                confidence_dist[confidence] = confidence_dist.get(confidence, 0) + 1

                # Skip search count
                if metadata.should_skip_search:
                    skip_search_count += 1

        # Emit trace event
        Trace.event("claim_extraction.metadata_distribution", {
            "total_claims": len(claims),
            "verification_target": target_dist,
            "claim_role": role_dist,
            "metadata_confidence": confidence_dist,
            "skip_search_count": skip_search_count,
        })

        # Also log summary
        logger.debug(
            "[Orchestration] Metadata: targets=%s, roles=%s, confidence=%s, skip_search=%d",
            target_dist, role_dist, confidence_dist, skip_search_count
        )

    def _default_channels(
        self,
        harm_potential: int,
        verification_target: VerificationTarget,
    ) -> list[EvidenceChannel]:
        return default_channels_v1(
            harm_potential=harm_potential,
            verification_target=verification_target,
        )

    def _parse_claim_metadata(
        self,
        rc: dict,
        lang: str,
        harm_potential: int,
        claim_category: str,
        satire_likelihood: float,
    ) -> ClaimMetadata:
        """
        Parse claim metadata from LLM response with safe fallback defaults.
        """
        return parse_claim_metadata_v1(
            rc,
            lang=lang,
            harm_potential=harm_potential,
            claim_category=claim_category,
            satire_likelihood=satire_likelihood,
        )

    async def enrich_claims_post_evidence(
        self,
        claims: list[Claim],
        *,
        lang: str = "en",
        evidence_by_claim: dict[str, list[dict]] | None = None,
    ) -> list[Claim]:
        """
        Optional post-evidence enrichment stage.
        """
        if not POST_EVIDENCE_ALLOWED_FIELDS:
            Trace.event(
                "claim_enrichment.skipped",
                {
                    "reason": "no_post_evidence_fields",
                    "claims": len(claims or []),
                },
            )
            return claims

        return claims

    def _parse_claim_structure(
        self,
        rc: dict,
        *,
        fallback_conclusion: str,
    ) -> dict | None:
        """
        Parse claim structure fields with safe fallback (atomic if missing).
        """
        raw = rc.get("structure")
        if not isinstance(raw, dict):
            return None

        allowed_types = {
            t.value for t in ClaimStructureType if t != ClaimStructureType.OTHER
        }

        try:
            raw_type = str(raw.get("type", "")).lower().strip()
            if raw_type not in allowed_types:
                return None

            premises_raw = raw.get("premises", [])
            premises = [
                p.strip()
                for p in premises_raw
                if isinstance(p, str) and p.strip()
            ]

            conclusion = raw.get("conclusion")
            if not isinstance(conclusion, str) or not conclusion.strip():
                conclusion = fallback_conclusion

            deps_raw = raw.get("dependencies", [])
            dependencies = [
                d.strip()
                for d in deps_raw
                if isinstance(d, str) and d.strip()
            ]

            return {
                "type": raw_type,
                "premises": premises,
                "conclusion": conclusion,
                "dependencies": dependencies,
            }
        except Exception:
            return None

    def _locate_anchor(
        self,
        text: str,
        *,
        normalized_text: str | None,
        chunks: list[TextChunk] | None,
        full_text: str | None = None,
    ) -> ClaimAnchor | None:
        """
        Locate claim anchor within original chunks (deterministic substring match).
        """
        if not text or not chunks:
            chunks = []

        candidates = []
        for candidate in (text, normalized_text):
            if not isinstance(candidate, str):
                continue
            cleaned = candidate.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        def _normalize(val: str, *, strip_punct: bool = False) -> str:
            out = val.lower()
            if strip_punct:
                out = re.sub(r"[^a-z0-9]+", " ", out)
            return " ".join(out.split())

        for candidate in candidates:
            target = candidate.lower()
            if target:
                for ch in chunks:
                    hay_raw = ch.text.lower()
                    if target in hay_raw:
                        start = hay_raw.find(target)
                        return {
                            "chunk_id": ch.chunk_id,
                            "char_start": ch.char_start + start,
                            "char_end": ch.char_start + start + len(candidate),
                            "section_path": ch.section_path,
                        }

            target = _normalize(candidate)[:160]
            if target:
                for ch in chunks:
                    hay = _normalize(ch.text)
                    if target in hay:
                        start = hay.find(target)
                        return {
                            "chunk_id": ch.chunk_id,
                            "char_start": ch.char_start + start,
                            "char_end": ch.char_start + start + len(candidate),
                            "section_path": ch.section_path,
                        }

            target = _normalize(candidate, strip_punct=True)[:160]
            if target:
                for ch in chunks:
                    hay = _normalize(ch.text, strip_punct=True)
                    if target in hay:
                        start = hay.find(target)
                        return {
                            "chunk_id": ch.chunk_id,
                            "char_start": ch.char_start + start,
                            "char_end": ch.char_start + start + len(candidate),
                            "section_path": ch.section_path,
                        }

        if full_text:
            anchor = self._locate_anchor_in_text(full_text, candidates)
            if anchor:
                Trace.event(
                    "anchor_position.fallback",
                    {
                        "method": anchor["method"],
                        "match_quality": anchor["match_quality"],
                        "candidate_len": anchor["candidate_len"],
                        "candidate_sample": (candidates[0][:80] if candidates else ""),
                    },
                )
                return {
                    "chunk_id": "full_text",
                    "char_start": anchor["char_start"],
                    "char_end": anchor["char_end"],
                    "section_path": [],
                }

        return None

    def _locate_anchor_in_text(
        self,
        full_text: str,
        candidates: list[str],
    ) -> dict[str, object] | None:
        if not full_text or not candidates:
            return None

        hay = full_text.lower()
        for candidate in candidates:
            cand = candidate.strip().lower()
            if not cand:
                continue
            idx = hay.find(cand)
            if idx >= 0:
                return {
                    "char_start": idx,
                    "char_end": idx + len(cand),
                    "method": "exact",
                    "match_quality": 1.0,
                    "candidate_len": len(cand),
                }

            tokens = re.findall(r"[\w]+", cand, flags=re.UNICODE)
            if len(tokens) < 2:
                continue
            pattern = r"\b" + r"\W+".join(re.escape(t) for t in tokens) + r"\b"
            match = re.search(pattern, hay, flags=re.UNICODE)
            if match:
                return {
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "method": "token_span",
                    "match_quality": 0.6,
                    "candidate_len": len(cand),
                }

        return None

    def _should_check_oracle(
        self,
        claims: list[Claim],
        article_intent: ArticleIntent,
    ) -> tuple[bool, ArticleIntent]:
        """
        Determine whether Oracle check is needed.
        """
        intent = normalize_article_intent(article_intent or "news")
        has_flag = any(c.get("check_oracle") for c in claims)
        if has_flag:
            return True, intent
        if intent in ("opinion", "prediction"):
            return False, intent
        return True, intent

    def _dedupe_claims(self, claims: list[Claim]) -> list[Claim]:
        """
        Deduplicate claims by normalized_text + location bucket.

        Keeps max importance/worthiness, merges query candidates, and remaps IDs.
        """
        if not claims:
            return []

        seen: dict[str, Claim] = {}
        key_to_ids: dict[str, list[str]] = {}

        for c in claims:
            base_key = " ".join((c.get("normalized_text") or c.get("text") or "").lower().split())
            if not base_key:
                continue

            key = base_key
            anchor = c.get("anchor") or {}
            if isinstance(anchor, dict) and "char_start" in anchor:
                try:
                    bucket = int(anchor.get("char_start", 0)) // 200
                    key = f"{base_key}|loc:{bucket}"
                except Exception:
                    key = base_key

            if c.get("id"):
                key_to_ids.setdefault(key, []).append(c["id"])

            if key in seen:
                existing = seen[key]
                existing["importance"] = max(
                    float(existing.get("importance", 0.5)),
                    float(c.get("importance", 0.5)),
                )
                existing["check_worthiness"] = max(
                    float(existing.get("check_worthiness", 0.5)),
                    float(c.get("check_worthiness", 0.5)),
                )
                existing_queries = existing.get("query_candidates", []) or []
                new_queries = c.get("query_candidates", []) or []
                seen_texts = {q.get("text") for q in existing_queries if q}
                for q in new_queries:
                    if q and q.get("text") not in seen_texts:
                        existing_queries.append(q)
                        seen_texts.add(q.get("text"))
                existing["query_candidates"] = existing_queries
                if c.get("check_oracle"):
                    existing["check_oracle"] = True
            else:
                seen[key] = c

        deduped_items = list(seen.items())
        deduped: list[Claim] = []
        id_map: dict[str, str] = {}
        for idx, (key, c) in enumerate(deduped_items):
            new_id = f"c{idx + 1}"
            for old_id in key_to_ids.get(key, []):
                id_map[old_id] = new_id
            c["id"] = new_id
            deduped.append(c)

        known_ids = {c.get("id") for c in deduped if c.get("id")}
        for c in deduped:
            structure = c.get("structure")
            if not isinstance(structure, dict):
                continue
            deps = structure.get("dependencies", [])
            if not isinstance(deps, list):
                structure["dependencies"] = []
                continue
            remapped: list[str] = []
            for dep in deps:
                if not isinstance(dep, str):
                    continue
                new_dep = id_map.get(dep, dep)
                if new_dep in known_ids and new_dep != c.get("id"):
                    remapped.append(new_dep)
            structure["dependencies"] = remapped

        return deduped
