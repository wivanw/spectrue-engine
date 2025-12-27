from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import logging

from spectrue_core.agents.skills.oracle_validation import EVIDENCE_THRESHOLD
from spectrue_core.utils.text_processing import normalize_search_query
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


FinalizeJackpot = Callable[[dict], Awaitable[dict]]
CreateEvidenceSource = Callable[[dict], dict]
ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class OracleFlowInput:
    original_fact: str
    fast_query: str
    lang: str
    article_intent: str
    should_check_oracle: bool
    claims: list[dict]
    oracle_check_intents: set[str]
    oracle_skip_intents: set[str]
    progress_callback: ProgressCallback | None


@dataclass(slots=True)
class OracleFlowResult:
    early_result: dict | None = None
    evidence_source: dict | None = None
    ran_oracle: bool = False
    skip_reason: str | None = None


async def run_oracle_flow(
    search_mgr,
    *,
    inp: OracleFlowInput,
    finalize_jackpot: FinalizeJackpot,
    create_evidence_source: CreateEvidenceSource,
) -> OracleFlowResult:
    """
    M63/M81: Hybrid Oracle flow (Jackpot/Evidence/Miss) with quota gating.

    Returns:
    - early_result: final pipeline result for JACKPOT (caller should return immediately)
    - evidence_source: a source dict to be appended to evidence pack (EVIDENCE case)
    """
    # Check Oracle based on article_intent (news/evergreen/official) OR should_check_oracle flag
    run_oracle = (
        (inp.article_intent in inp.oracle_check_intents) or inp.should_check_oracle
    ) and (inp.article_intent not in inp.oracle_skip_intents)

    # M81/T5: Gate Oracle by verification_target
    # Skip Oracle for interview/attribution-heavy articles (waste of quota)
    if run_oracle and inp.claims:
        reality_count = 0
        attribution_count = 0
        for c in inp.claims:
            metadata = c.get("metadata")
            if metadata:
                target = metadata.verification_target.value
                if target == "reality":
                    reality_count += 1
                elif target in ("attribution", "existence"):
                    attribution_count += 1

        # Skip if NO reality claims (all attribution/existence/none)
        if reality_count == 0 and attribution_count > 0:
            oracle_skip_reason = "all_attribution_claims"
            run_oracle = False
            logger.debug("[M81] Skipping Oracle: all claims are attribution/existence (quota save)")
            Trace.event(
                "oracle.skipped",
                {
                    "reason": oracle_skip_reason,
                    "reality_count": reality_count,
                    "attribution_count": attribution_count,
                    "total_claims": len(inp.claims),
                },
            )

    if not run_oracle:
        skip_reason = (
            "opinion/prediction intent"
            if inp.article_intent in inp.oracle_skip_intents
            else "no markers"
        )
        logger.debug(
            "[Pipeline] Skipping Oracle (%s, intent=%s)", skip_reason, inp.article_intent
        )
        return OracleFlowResult(ran_oracle=False, skip_reason=skip_reason)

    if inp.progress_callback:
        await inp.progress_callback("checking_oracle")

    # M63: Identify claims to check - use normalized_text if available
    candidates = [c for c in inp.claims if c.get("check_oracle")]

    # Fallback: Use first core claim with high importance
    if not candidates:
        core_claims = [c for c in inp.claims if c.get("type") == "core"]
        if core_claims:
            candidates = [core_claims[0]]
        else:
            candidates = [{"text": inp.fast_query, "normalized_text": inp.fast_query}]

    # Limit to 1 candidate to preserve quota (more targeted now)
    candidates = candidates[:1]

    oracle_evidence_source = None

    for cand in candidates:
        # M71: Try English query first (Google Fact Check has better EN coverage)
        en_query = None
        for qc in cand.get("query_candidates", []):
            qc_text = qc.get("text", "")
            if qc_text and sum(1 for c in qc_text if ord(c) < 128) / len(qc_text) > 0.9:
                en_query = qc_text
                break

        if en_query:
            query_text = en_query.strip(" ,.-:;")
            logger.debug("[Pipeline] Oracle: Using English query candidate")
        else:
            query_text = (cand.get("normalized_text") or cand.get("text", "")).strip(
                " ,.-:;"
            )

        q = normalize_search_query(query_text)
        logger.debug(
            "[Pipeline] Oracle Hybrid Check: intent=%s, query=%s",
            inp.article_intent,
            q[:300],
        )

        try:
            oracle_result = await search_mgr.check_oracle_hybrid(q, intent=inp.article_intent)
        except Exception as e:
            Trace.event(
                "pipeline.oracle.exception",
                {
                    "intent": inp.article_intent,
                    "query_used": q[:300],
                    "error": str(e)[:200],
                },
            )
            logger.warning("[Pipeline] Oracle exception: %s", e)
            continue
        if not oracle_result:
            continue

        status = oracle_result.get("status", "EMPTY")
        relevance = oracle_result.get("relevance_score", 0.0)
        is_jackpot = oracle_result.get("is_jackpot", False)

        trace_payload = {
            "intent": inp.article_intent,
            "query_used": q[:300],
            "relevance_score": relevance,
            "status": status,
            "is_jackpot": is_jackpot,
        }

        # M73.4: Handle ERROR status (API failure)
        if status == "ERROR":
            code = oracle_result.get("error_status_code")
            detail = oracle_result.get("error_detail", "")
            trace_payload.update({"error_code": code, "error_detail": detail})
            Trace.event("pipeline.oracle_hybrid_error", trace_payload)

            logger.warning("[Pipeline] Oracle API ERROR (%s): %s", code, detail)

            # Break loop on auth/quota errors, continue on timeouts
            if code in (403, 429):
                logger.error(
                    "[Pipeline] Oracle Quota Exceeded/Auth Error. Stopping Oracle loop."
                )
                break
            continue

        # M73.4: Handle DISABLED status (no validator configured)
        if status == "DISABLED":
            Trace.event("pipeline.oracle_hybrid", {**trace_payload, "disabled": True})
            logger.warning("[Pipeline] Oracle DISABLED (no LLM validator). Skipping.")
            break

        Trace.event("pipeline.oracle_hybrid", trace_payload)

        if status == "EMPTY":
            logger.debug("[Pipeline] Oracle: No results found. Continuing to search.")
            continue

        # SCENARIO A: JACKPOT (relevance > 0.9)
        if is_jackpot:
            logger.debug(
                "[Pipeline] 🎰 JACKPOT! Oracle hit (score=%.2f). Stopping pipeline.",
                relevance,
            )
            oracle_final = await finalize_jackpot(oracle_result)
            Trace.event(
                "pipeline.result",
                {
                    "type": "oracle_jackpot",
                    "verified_score": oracle_final.get("verified_score"),
                    "status": status,
                    "source": oracle_result.get("publisher"),
                },
            )
            return OracleFlowResult(early_result=oracle_final, ran_oracle=True)

        # SCENARIO B: EVIDENCE (0.5 < relevance <= 0.9)
        if relevance > EVIDENCE_THRESHOLD:
            logger.debug(
                "[Pipeline] 📚 EVIDENCE: Oracle related (score=%.2f). Adding to pack.",
                relevance,
            )
            oracle_evidence_source = create_evidence_source(oracle_result)
            oracle_evidence_source["claim_id"] = cand.get("id")
            continue

        # SCENARIO C: MISS (relevance <= 0.5)
        logger.debug("[Pipeline] Oracle MISS (score=%.2f). Proceeding to search.", relevance)

    if not oracle_evidence_source:
        logger.debug("[Pipeline] Oracle checks finished. No relevant hits.")

    return OracleFlowResult(
        evidence_source=oracle_evidence_source,
        ran_oracle=True,
    )
