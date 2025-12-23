from spectrue_core.verification.evidence_pack import Claim, ClaimAnchor, EvidenceRequirement, ArticleIntent
from .base_skill import BaseSkill, logger
from spectrue_core.utils.text_chunking import TextChunk
from spectrue_core.utils.trace import Trace
from spectrue_core.constants import SUPPORTED_LANGUAGES
from .claims_prompts import (
    build_claim_schema_instructions,
    build_claim_schema_prompt,
    build_claim_strategist_instructions,
    build_claim_strategist_prompt,
)
from .claims_parsing import (
    CLAIM_TYPE_MAPPING,
    DOMAIN_MAPPING,
    SEARCH_INTENTS,
    TOPIC_GROUPS,
    clamp_float,
    clamp_int,
    normalize_article_intent,
    normalize_claim_category,
    normalize_topic_group,
)

# M70: Import schema module for structured claims
from spectrue_core.schema import (
    ClaimUnit,
    Assertion,
    Dimension,
    ClaimType,
    ClaimDomain,
    EvidenceRequirementSpec,
    EventQualifiers,
    LocationQualifier,
    ClaimStructure,
    ClaimStructureType,
    ClaimRole,
)

# M80: Import claim metadata types
from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    EvidenceChannel,
)

from spectrue_core.agents.skills.claim_metadata_parser import (
    parse_claim_metadata as parse_claim_metadata_v1,
    default_channels as default_channels_v1,
)

class ClaimExtractionSkill(BaseSkill):
    
    # M73.5: Dynamic timeout constants
    BASE_TIMEOUT_SEC = 35.0     # Minimum timeout
    TIMEOUT_PER_1K_CHARS = 2.0  # Additional seconds per 1000 chars
    MAX_TIMEOUT_SEC = 75.0      # Maximum timeout cap
    
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
    
    async def extract_claims(
        self,
        text: str,
        *,
        chunks: list[TextChunk] | None = None,  # M74
        lang: str = "en",
        max_claims: int = 5,
    ) -> tuple[list[Claim], bool, ArticleIntent]:
        """
        Extract atomic verifiable claims from article text.
        
        M62: Context-aware claims with normalized_text, topic_group, check_worthiness
        M63: Returns article_intent for Oracle triggering
        M64: Topic-Aware Round-Robin support (topic_key, query_candidates)
             + Negative Constraints (Gambling Guardrail via LLM)
        
        Returns:
            tuple of (claims, should_check_oracle, article_intent)
        """
        text = (text or "").strip()
        if not text:
            return [], False, "news"  # Default intent

        # Limit input to prevent token overflow
        text_excerpt = text[:8000] if len(text) > 8000 else text
        
        # M57: Resolve language name for bilingual query generation
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")
        
        topics_str = ", ".join(TOPIC_GROUPS)
        intents_str = ", ".join(SEARCH_INTENTS)
        
        # Updated Strategist Prompt with Salience & Harm Potential (M77) + Satire (M78)
        instructions = build_claim_strategist_instructions(
            intents_str=intents_str,
            topics_str=topics_str,
            lang_name=lang_name,
        )
        prompt = build_claim_strategist_prompt(text_excerpt=text_excerpt, max_claims=max_claims)
        try:
            # M81: Updated cache key to force prompt refresh with calibration rules
            cache_key = f"claim_strategist_v6_{lang}"

            # M73.5: Dynamic timeout based on input size
            dynamic_timeout = self._calculate_timeout(len(text_excerpt))
            logger.debug("[Claims] Input: %d chars, timeout: %.1f sec", len(text_excerpt), dynamic_timeout)
            
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="low",
                cache_key=cache_key,
                timeout=dynamic_timeout,
                trace_kind="claim_extraction",
            )
            
            raw_claims = result.get("claims", [])
            claims: list[Claim] = []
            
            # M63: Extract article intent
            raw_intent = normalize_article_intent(result.get("article_intent", "news"))
            article_intent: ArticleIntent = raw_intent  # type: ignore
            
            for idx, rc in enumerate(raw_claims):
                if not isinstance(rc, dict) or not rc.get("text"):
                    continue
                
                req_raw = rc.get("evidence_req", {})
                req = EvidenceRequirement(
                    needs_primary_source=bool(req_raw.get("needs_primary")),
                    needs_independent_2x=bool(req_raw.get("needs_2_independent")),
                    needs_quote_verification=bool(req_raw.get("needs_quote")),
                    is_time_sensitive=bool(req_raw.get("needs_recent")),
                )
                
                # M62: Extract new fields with safe defaults
                normalized = rc.get("normalized_text", "") or rc.get("text", "")
                topic = normalize_topic_group(rc.get("topic_group", "Other") or "Other")
                
                # M64: topic_key extraction
                topic_key = rc.get("topic_key") or topic  # Fallback to group if key missing
                
                worthiness = rc.get("check_worthiness")
                if worthiness is None:
                    worthiness = rc.get("importance", 0.5)
                worthiness = clamp_float(worthiness, default=0.5, lo=0.0, hi=1.0)
                
                # M77: Harm Potential
                harm_potential = clamp_int(rc.get("harm_potential", 1), default=1, lo=1, hi=5)

                # M78: Claim Category & Satire Likelihood
                claim_category = normalize_claim_category(rc.get("claim_category", "FACTUAL"))
                satire_likelihood = clamp_float(rc.get("satire_likelihood", 0.0), default=0.0, lo=0.0, hi=1.0)

                # M80: Parse ClaimMetadata
                metadata = self._parse_claim_metadata(rc, lang, harm_potential, claim_category, satire_likelihood)

                # M62+: Extract search strategy if present
                strategy = rc.get("search_strategy", {})

                # M93: Parse claim structure (premises/conclusion/dependencies)
                structure = self._parse_claim_structure(
                    rc,
                    fallback_conclusion=normalized or rc.get("text", ""),
                )
                claim_role = (
                    metadata.claim_role.value
                    if metadata
                    else str(rc.get("claim_role", "core")).lower()
                )
                
                # M78+M80: Satire/Non-verifiable Routing - clear search data
                search_queries = rc.get("search_queries", [])
                query_candidates = rc.get("query_candidates", [])
                should_skip_search = (
                    satire_likelihood >= 0.8 or 
                    claim_category == "SATIRE" or
                    metadata.should_skip_search
                )
                if should_skip_search:
                    search_queries = []  # Skip search
                    query_candidates = []
                    reason = "satire" if satire_likelihood >= 0.8 else f"target={metadata.verification_target.value}"
                    logger.info("[M80] Skip search (%s): %s", reason, normalized[:50])
                
                c = Claim(
                    id=f"c{idx+1}",
                    text=rc.get("text", ""),
                    type=rc.get("type", "core"),  # type: ignore
                    importance=float(rc.get("importance", 0.5)),
                    evidence_requirement=req,
                    search_queries=search_queries,
                    check_oracle=bool(rc.get("check_oracle", False)),
                    # M62: New fields
                    normalized_text=normalized,
                    topic_group=topic,
                    check_worthiness=worthiness,
                    # M64: New fields
                    topic_key=topic_key,
                    query_candidates=query_candidates,
                    # M66: Smart Routing
                    search_method=rc.get("search_method", "general_search"),
                    # M73 Layer 4: Evidence-Need Routing
                    evidence_need=rc.get("evidence_need", "unknown"),
                    # M74: Anchor
                    anchor=self._locate_anchor(rc.get("text", ""), chunks),
                    # M77: Salience
                    harm_potential=harm_potential,
                    # M78: Satire
                    claim_category=claim_category,
                    satire_likelihood=satire_likelihood,
                    # M80: Orchestration Metadata
                    metadata=metadata,
                    # M93: Claim structure + role
                    claim_role=claim_role,
                    structure=structure,
                )
                
                # Log strategy for debugging
                if strategy:
                    intent = strategy.get("intent", "?")
                    reasoning = strategy.get("reasoning", "")[:50]
                    logger.debug(
                        "[Strategist] Claim %d: intent=%s, harm=%d, category=%s | %s",
                        idx+1, intent, harm_potential, claim_category, reasoning
                    )
                
                claims.append(c)
            
            # ─────────────────────────────────────────────────────────────────
            # M73.5: DEDUPLICATION - Merge claims with identical normalized_text
            # ─────────────────────────────────────────────────────────────────
            claims = self._dedupe_claims(claims)

            # M77: Sort by harm_potential DESC
            claims.sort(key=lambda x: x.get("harm_potential", 1), reverse=True)
            
            # M78: Count satire claims for telemetry
            satire_count = sum(1 for c in claims if c.get("satire_likelihood", 0) >= 0.8 or c.get("claim_category") == "SATIRE")
            if satire_count > 0:
                logger.info("[M78] Detected %d satire/hyperbolic claims", satire_count)
            
            # Log topic and strategy distribution
            topics_found = [c.get("topic_key", "?") for c in claims]
            logger.info("[Claims] Extracted %d claims (after dedup/sort). Topics keys: %s", len(claims), topics_found)
                
            # M60 Oracle Optimization: Check if ANY claim needs oracle
            check_oracle = any(c.get("check_oracle", False) for c in claims)
            
            # M63: Log intent for debugging
            logger.info("[Claims] Article intent: %s (check_oracle=%s)", article_intent, check_oracle)
            
            # M81/T4: Trace extracted claims for debugging
            self._trace_extracted_claims(claims)
            
            # M80/T8: Trace metadata distribution for debugging
            self._trace_metadata_distribution(claims)
            
            return claims, check_oracle, article_intent
            
        except Exception as e:
            logger.warning("[M48] Claim extraction failed: %s. Using fallback.", e)
            # Fallback: Treat entire text as one core claim
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
                )
            ], False, "news"  # Default intent on fallback

    # ─────────────────────────────────────────────────────────────────────────
    # M70: Schema-First Extraction (Spec Producer)
    # ─────────────────────────────────────────────────────────────────────────

    async def extract_claims_structured(
        self,
        text: str,
        *,
        chunks: list[TextChunk] | None = None,  # M74
        lang: str = "en",
        max_claims: int = 5,
    ) -> tuple[list[ClaimUnit], bool, ArticleIntent]:
        """
        M70: Schema-constrained claim extraction.
        
        This is the SPEC PRODUCER. LLM acts as parser + interpreter:
        - Fills structured ClaimUnit fields
        - Classifies each field as FACT / CONTEXT / INTERPRETATION
        - Does NOT decide truth (that's for scoring)
        
        Key Design:
        - time_reference ≠ location (the core bug fix)
        - Each assertion has explicit dimension
        - Schema is stable contract for downstream consumers
        
        Returns:
            tuple of (claim_units, should_check_oracle, article_intent)
        """
        text = (text or "").strip()
        if not text:
            return [], False, "news"

        text_excerpt = text[:8000] if len(text) > 8000 else text
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")
        topics_str = ", ".join(TOPIC_GROUPS)

        # M70: Schema-Constrained Generation Prompt
        instructions = build_claim_schema_instructions(topics_str=topics_str, lang_name=lang_name)
        prompt = build_claim_schema_prompt(text_excerpt=text_excerpt)

        try:
            cache_key = f"claim_schema_v1_{lang}"

            # M73.5: Dynamic timeout with +10s offset for structured extraction complexity
            dynamic_timeout = self._calculate_timeout(len(text_excerpt), base_offset=10.0)
            logger.debug("[Claims Structured] Input: %d chars, timeout: %.1f sec", len(text_excerpt), dynamic_timeout)
            
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="medium",  # Higher effort for schema parsing
                cache_key=cache_key,
                timeout=dynamic_timeout,
                trace_kind="claim_extraction_structured",
            )

            raw_claims = result.get("claims", [])
            claim_units: list[ClaimUnit] = []

            # M63: Extract article intent
            raw_intent = normalize_article_intent(result.get("article_intent", "news"))
            article_intent: ArticleIntent = raw_intent  # type: ignore

            for idx, rc in enumerate(raw_claims):
                if not isinstance(rc, dict):
                    continue

                claim_id = rc.get("id") or f"c{idx + 1}"

                # Parse assertions
                assertions: list[Assertion] = []
                for ra in rc.get("assertions", []):
                    if not isinstance(ra, dict):
                        continue
                    
                    # Parse dimension (LLM decides this, we just validate)
                    dim_str = ra.get("dimension", "FACT").upper()
                    if dim_str == "CONTEXT":
                        dimension = Dimension.CONTEXT
                    elif dim_str == "INTERPRETATION":
                        dimension = Dimension.INTERPRETATION
                    else:
                        dimension = Dimension.FACT

                    assertions.append(Assertion(
                        key=ra.get("key", "claim.text"),
                        value=ra.get("value"),
                        value_raw=ra.get("value_raw"),
                        dimension=dimension,
                        importance=float(ra.get("importance", 1.0)),
                        is_inferred=bool(ra.get("is_inferred", False)),
                        evidence_requirement=EvidenceRequirementSpec(
                            needs_primary=bool(ra.get("needs_primary", False)),
                            needs_2_independent=bool(ra.get("needs_2_independent", False)),
                        ),
                    ))

                # Parse qualifiers
                qualifiers = None
                raw_qual = rc.get("qualifiers")
                if isinstance(raw_qual, dict):
                    location = None
                    raw_loc = raw_qual.get("location")
                    if isinstance(raw_loc, dict):
                        location = LocationQualifier(
                            venue=raw_loc.get("venue"),
                            city=raw_loc.get("city"),
                            region=raw_loc.get("region"),
                            country=raw_loc.get("country"),
                            is_inferred=bool(raw_loc.get("is_inferred", False)),
                        )

                    qualifiers = EventQualifiers(
                        time_reference=raw_qual.get("time_reference"),
                        location=location,
                        participants=raw_qual.get("participants", []),
                    )

                # Map claim type
                raw_type = rc.get("claim_type", "other").lower()
                claim_type = CLAIM_TYPE_MAPPING.get(raw_type, ClaimType.OTHER)
                # Also try direct mapping for M70 types
                for ct in ClaimType:
                    if ct.value == raw_type:
                        claim_type = ct
                        break

                # Map domain
                topic_group = rc.get("topic_group", "Other")
                if topic_group not in TOPIC_GROUPS:
                    topic_group = "Other"
                domain = DOMAIN_MAPPING.get(topic_group, ClaimDomain.OTHER)
                # Also try direct mapping
                raw_domain = rc.get("domain", "").lower()
                for cd in ClaimDomain:
                    if cd.value == raw_domain:
                        domain = cd
                        break

                # Parse claim role
                raw_role = rc.get("claim_role", "")
                try:
                    claim_role = ClaimRole(str(raw_role).lower())
                except ValueError:
                    claim_role = ClaimRole.CORE

                # Parse structure
                structure_data = self._parse_claim_structure(
                    rc,
                    fallback_conclusion=rc.get("normalized_text", rc.get("text", "")),
                )
                structure = None
                if structure_data:
                    try:
                        structure = ClaimStructure(
                            type=ClaimStructureType(structure_data["type"]),
                            premises=structure_data.get("premises", []),
                            conclusion=structure_data.get("conclusion"),
                            dependencies=structure_data.get("dependencies", []),
                        )
                    except ValueError:
                        structure = None

                # Create ClaimUnit
                claim_unit = ClaimUnit(
                    id=claim_id,
                    domain=domain,
                    claim_type=claim_type,
                    claim_role=claim_role,
                    structure=structure,
                    subject=rc.get("subject"),
                    predicate=rc.get("predicate", ""),
                    object=rc.get("object"),
                    qualifiers=qualifiers,
                    assertions=assertions,
                    importance=float(rc.get("importance", 0.5)),
                    check_worthiness=float(rc.get("check_worthiness", 0.5)),
                    extraction_confidence=float(rc.get("extraction_confidence", 1.0)),
                    text=rc.get("text", ""),
                    normalized_text=rc.get("normalized_text", rc.get("text", "")),
                    topic_group=topic_group,
                    topic_key=rc.get("topic_key", topic_group),
                    language=lang,
                )

                claim_units.append(claim_unit)

                # Debug logging
                fact_count = len(claim_unit.get_fact_assertions())
                context_count = len(claim_unit.get_context_assertions())
                logger.debug(
                    "[M70] Claim %s: %d FACT, %d CONTEXT assertions | type=%s",
                    claim_id, fact_count, context_count, claim_type.value
                )

            # Log summary
            total_facts = sum(len(c.get_fact_assertions()) for c in claim_units)
            total_context = sum(len(c.get_context_assertions()) for c in claim_units)
            logger.info(
                "[M70] Extracted %d claims: %d FACT assertions, %d CONTEXT assertions",
                len(claim_units), total_facts, total_context
            )

            # M93: Filter dependencies to known claim IDs
            known_ids = {c.id for c in claim_units}
            for c in claim_units:
                if not c.structure:
                    continue
                deps = [
                    d for d in c.structure.dependencies
                    if d in known_ids and d != c.id
                ]
                c.structure.dependencies = deps

            # Check oracle from query_candidates
            check_oracle = any(
                bool(rc.get("check_oracle", False))
                for rc in raw_claims
                if isinstance(rc, dict)
            )

            return claim_units, check_oracle, article_intent

        except Exception as e:
            logger.warning("[M70] Structured extraction failed: %s. Using fallback.", e)
            # Fallback: Create minimal ClaimUnit
            fallback_text = text[:300] + "..." if len(text) > 300 else text
            return [
                ClaimUnit(
                    id="c1",
                    claim_type=ClaimType.OTHER,
                    text=fallback_text,
                    normalized_text=fallback_text,
                    topic_group="Other",
                    topic_key="General",
                    importance=1.0,
                    check_worthiness=0.5,
                    assertions=[
                        Assertion(
                            key="claim.text",
                            value=fallback_text,
                            dimension=Dimension.FACT,
                        )
                    ],
                    language=lang,
                )
            ], False, "news"

    # ─────────────────────────────────────────────────────────────────────────
    # M73.5: Claim Deduplication
    # ─────────────────────────────────────────────────────────────────────────

    def _locate_anchor(self, text: str, chunks: list[TextChunk] | None) -> ClaimAnchor | None:
        """M74: Locate claim anchor in text chunks."""
        if not text or not chunks:
            return None
        
        # Simple exact substring match of first 100 chars (case-insensitive)
        # Sufficient for anchoring
        target = " ".join(text.lower().split())[:100]
        
        for ch in chunks:
            if target in ch.text.lower():
                 start = ch.text.lower().find(target)
                 return {
                     "chunk_id": ch.chunk_id,
                     "char_start": ch.char_start + start,
                     "char_end": ch.char_start + start + len(text),
                     "section_path": ch.section_path
                 }
        return None

    def _dedupe_claims(self, claims: list[Claim]) -> list[Claim]:
        """
        Deduplicate claims by normalized_text hash.
        
        Merges duplicates:
        - Takes MAX importance (most important wins)
        - Takes MAX check_worthiness
        - Keeps first occurrence (for type, topic_group, etc.)
        - Merges search_queries and query_candidates
        
        M74: Spatial Deduplication
        - Uses simple bucketing (200 chars) to determine if claims are distinct instances
        """
        if not claims:
            return []
        
        # Group by normalized_text (lowercased, stripped)
        seen: dict[str, Claim] = {}
        key_to_ids: dict[str, list[str]] = {}
        
        for c in claims:
            # Normalize key: lowercase, strip, collapse whitespace
            base_key = " ".join((c.get("normalized_text") or c.get("text") or "").lower().split())
            if not base_key:
                continue
            
            # M74: Spatial bucketing
            key = base_key
            if c.get("anchor"):
                # Claims >200 chars apart are treated as separate
                bucket = c["anchor"]["char_start"] // 200
                key = f"{base_key}|loc:{bucket}"
            
            if c.get("id"):
                key_to_ids.setdefault(key, []).append(c["id"])

            if key in seen:
                # Merge: take max importance
                existing = seen[key]
                existing["importance"] = max(
                    float(existing.get("importance", 0.5)),
                    float(c.get("importance", 0.5))
                )
                existing["check_worthiness"] = max(
                    float(existing.get("check_worthiness", 0.5)),
                    float(c.get("check_worthiness", 0.5))
                )
                # Merge query candidates (dedupe by text)
                existing_queries = existing.get("query_candidates", []) or []
                new_queries = c.get("query_candidates", []) or []
                seen_texts = {q.get("text") for q in existing_queries if q}
                for q in new_queries:
                    if q and q.get("text") not in seen_texts:
                        existing_queries.append(q)
                        seen_texts.add(q.get("text"))
                existing["query_candidates"] = existing_queries
                # Oracle: if ANY duplicate wants oracle, check it
                if c.get("check_oracle"):
                    existing["check_oracle"] = True
                logger.debug("[Dedup] Merged claim: %s", key[:50])
            else:
                seen[key] = c
        
        # Re-assign IDs (c1, c2, ...) and build old->new mapping
        deduped_items = list(seen.items())
        deduped: list[Claim] = []
        id_map: dict[str, str] = {}
        for idx, (key, c) in enumerate(deduped_items):
            new_id = f"c{idx + 1}"
            old_ids = key_to_ids.get(key, [])
            for old_id in old_ids:
                id_map[old_id] = new_id
            c["id"] = new_id
            deduped.append(c)

        # Remap structure dependencies to new IDs
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
        
        # Log dedup stats
        if len(claims) != len(deduped):
            logger.info("[Dedup] Merged %d → %d claims", len(claims), len(deduped))
        
        return deduped

    # ─────────────────────────────────────────────────────────────────────────
    # M80: Metadata Parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_claim_metadata(
        self,
        rc: dict,
        lang: str,
        harm_potential: int,
        claim_category: str,
        satire_likelihood: float,
    ) -> ClaimMetadata:
        """
        M80: Parse Claim metadata from LLM response with safe fallback defaults.

        Delegated to `spectrue_core.agents.skills.claim_metadata_parser.parse_claim_metadata`
        to keep this file readable.
        """
        return parse_claim_metadata_v1(
            rc,
            lang=lang,
            harm_potential=harm_potential,
            claim_category=claim_category,
            satire_likelihood=satire_likelihood,
        )

    def _parse_claim_structure(
        self,
        rc: dict,
        *,
        fallback_conclusion: str,
    ) -> dict | None:
        """
        M93: Parse claim structure fields with safe fallback.

        Returns None when structure is missing/invalid (atomic fallback).
        """
        raw = rc.get("structure")
        if not isinstance(raw, dict):
            return None

        allowed_types = {
            t.value for t in ClaimStructureType
            if t != ClaimStructureType.OTHER
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
        M80/T8: Emit trace event with metadata distribution for debugging.
        
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
        logger.info(
            "[M80] Metadata: targets=%s, roles=%s, confidence=%s, skip_search=%d",
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
