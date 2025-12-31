# Spectrue Engine - main entry point

import logging
import json
from typing import Dict, Any, Optional, List
from langdetect import detect_langs, DetectorFactory, LangDetectException
import re
from uuid import uuid4
from datetime import datetime

from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.verifier import FactVerifier
from spectrue_core.utils.trace import Trace
from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core.billing.metering import TavilyMeter
from spectrue_core.billing.config_loader import load_pricing_policy


# Make language detection deterministic
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

def detect_content_language(text: str, fallback: str = "en") -> tuple[str, float]:
    """Detect language of input text using langdetect library, returning (lang, confidence)."""
    clean_text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    clean_text = clean_text.strip()
    
    if len(clean_text) < 20:
        return fallback, 0.0
    
    try:
        langs = detect_langs(clean_text)
        if not langs:
            return fallback, 0.0
        detected_lang = langs[0].lang
        confidence = float(getattr(langs[0], "prob", 0.0) or 0.0)
        
        supported = ["en", "uk", "ru", "de", "es", "fr", "ja", "zh"]
        if detected_lang in supported:
            return detected_lang, confidence
        
        # Unsupported language → fallback to English
        return fallback, confidence
        
    except LangDetectException:
        return fallback, 0.0


class SpectrueEngine:
    """The main entry point for the Spectrue Fact-Checking Engine."""
    
    def __init__(self, config: SpectrueConfig, translation_service=None):
        self.config = config
        # Optional translation_service for Oracle result localization
        self.verifier = FactVerifier(config, translation_service=translation_service)
        try:
            logger.debug("Effective config: %s", json.dumps(self.config.runtime.to_safe_log_dict(), ensure_ascii=False))
        except Exception:
            pass

    async def fetch_url_content(self, url: str) -> str | None:
        """Fetch URL content securely via configured search provider (no local requests)."""
        return await self.verifier.fetch_url_content(url)

    async def analyze_text(
        self, 
        text: str,
        lang: str = "en",
        analysis_mode: str = "general",
        gpt_model: str = None,
        search_type: str = "advanced",
        progress_callback = None,
        max_credits: Optional[int] = None,
        sentences: Optional[List[str]] = None,  # Pre-segmented sentences (skip segmentation)
    ) -> Dict[str, Any]:
        """
        Analyze text with content detection and waterfall verification.
        
        Args:
            text: Text to analyze
            lang: UI language ISO code
            analysis_mode: "general" or "lite"
            gpt_model: Model override (optional, uses config default if None)
            search_type: "basic" or "advanced"
            progress_callback: Async callable(stage: str)
            
        Returns:
            Dict with analysis result
        """
        trace_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{str(uuid4())[:6]}"
        Trace.start(trace_id, runtime=self.config.runtime)
        
        try:
            model = gpt_model or self.config.openai_model
            
            # Detect content language
            detected_lang, detected_prob = detect_content_language(text, fallback=lang)
            min_prob = float(getattr(self.config.runtime.tunables, "langdetect_min_prob", 0.80) or 0.80)
            content_lang = detected_lang if detected_prob >= min_prob else lang

            Trace.event("engine.analyze_text.start", {
                "analysis_mode": analysis_mode,
                "text_len": len(text),
                "model": model,
            })

            # Setup metering for pre-pipeline operations (like URL fetch)
            policy = load_pricing_policy()
            engine_ledger = CostLedger(run_id=trace_id)
            tavily_meter = TavilyMeter(ledger=engine_ledger, policy=policy)
            
            # Temporarily inject meter into WebSearchTool for fetch_url_content
            # Note: This affects the singleton web_tool instance, so strictly scoped.
            web_tool = self.verifier.pipeline.search_mgr.web_tool
            prior_meter = getattr(web_tool._tavily, "_meter", None)
            web_tool._tavily._meter = tavily_meter
            
            fetch_cost = 0.0

            if analysis_mode == "deep":
                if progress_callback:
                    await progress_callback("extracting_claims")

                # Deep analysis: if input is a URL, first extract the article content
                working_text = text
                if text.strip().startswith("http://") or text.strip().startswith("https://"):
                    try:
                        # Fetch article content before claim extraction
                        extracted = await self.fetch_url_content(text.strip())
                        if extracted:
                            working_text = extracted
                            Trace.event("engine.deep.url_extracted", {"chars": len(extracted)})
                        else:
                            Trace.event("engine.deep.url_extraction_failed", {"url": text.strip()[:100]})
                    finally:
                        # Capture cost and restore meter
                        fetch_cost = float(engine_ledger.total_credits)
                        web_tool._tavily._meter = prior_meter

                # Single-pipeline deep mode: extract once, verify all claims together
                # Step 1: Extract claims from the ENTIRE text once
                initial_result = await self.verifier.verify_fact(
                    fact=working_text,
                    search_type=search_type,
                    gpt_model=model,
                    lang=lang,
                    progress_callback=progress_callback,
                    content_lang=content_lang,
                    max_cost=max_credits,
                    extract_claims_only=True,  # Just extract claims, don't verify
                )
                
                # Get extracted claims from the initial run
                extracted_claims = initial_result.get("_extracted_claims", [])
                if not extracted_claims:
                    # Fallback: use the full text as single claim
                    extracted_claims = [{"id": "c1", "text": working_text.strip()[:500]}]
                
                Trace.event("engine.deep.claims_extracted", {
                    "count": len(extracted_claims),
                    "claims": [c.get("text", "")[:80] for c in extracted_claims[:5]]
                })

                extraction_cost = float(
                    (initial_result.get("cost_summary") or {}).get("total_credits")
                    or initial_result.get("cost")
                    or 0.0
                )
                
                if progress_callback:
                    await progress_callback("verifying_claims")

                # Step 2: Single pipeline run with all claims (avoids N+1 pipeline runs)
                # This replaces the per-claim loop that caused N+1 pipeline runs
                verification_result = await self.verifier.verify_fact(
                    fact=working_text,  # Full article text for context
                    search_type=search_type,
                    gpt_model=model,
                    lang=lang,
                    progress_callback=progress_callback,
                    content_lang=content_lang,
                    max_cost=max_credits - int(extraction_cost) if max_credits is not None else None,
                    pipeline_profile="deep",
                    preloaded_claims=extracted_claims,  # Skip re-extraction
                )

                verification_cost = float(
                    (verification_result.get("cost_summary") or {}).get("total_credits")
                    or verification_result.get("cost")
                    or 0.0
                )
                total_cost = fetch_cost + extraction_cost + verification_cost
                
                # Build details from claim_verdicts
                details = []
                claim_verdicts = verification_result.get("claim_verdicts") or []
                sources = verification_result.get("sources") or []
                
                for claim_obj in extracted_claims:
                    claim_id = claim_obj.get("id")
                    claim_text = claim_obj.get("text", "").strip()
                    if not claim_text:
                        continue
                    
                    # Find verdict for this claim
                    cv = next(
                        (v for v in claim_verdicts if v.get("claim_id") == claim_id),
                        {}
                    )
                    
                    # Get per-claim sources
                    claim_sources = [
                        s for s in sources 
                        if s.get("claim_id") == claim_id or s.get("claim_id") is None
                    ]
                    
                    details.append({
                        "text": claim_text,
                        "rgba": cv.get("rgba") or verification_result.get("rgba") or [0.0, 0.0, 0.0, 0.5],
                        "rationale": cv.get("reason") or "",
                        "sources": claim_sources,
                        "verified_score": cv.get("verdict_score") or verification_result.get("verified_score"),
                        "danger_score": verification_result.get("danger_score"),
                    })

                # Use verification result's aggregated scores
                final = {
                    "text": text,
                    "verified_score": verification_result.get("verified_score", 0.0),
                    "rgba": verification_result.get("rgba") or [0.0, 0.0, 0.0, 0.5],
                    "average_rgba": verification_result.get("rgba") or [0.0, 0.0, 0.0, 0.5],
                    "details": details,
                    "sources": sources,
                    "cost": total_cost,
                    "cost_summary": verification_result.get("cost_summary") or {
                        "total_credits": total_cost,
                        "total_usd": total_cost * 0.01,
                    },
                    "claims": [c.get("text", "") for c in extracted_claims],
                    "detected_lang": detected_lang,
                    "detected_lang_prob": detected_prob,
                    "search_lang": content_lang,
                    "analysis_mode": "deep",
                }
                if max_credits is not None:
                    final["budget"] = {
                        "max_credits": int(max_credits),
                        "spent": total_cost,
                        "limited": False,  # Single pipeline handles budget internally
                    }
                
                Trace.event("engine.deep.single_pipeline_completed", {
                    "claims_count": len(extracted_claims),
                    "total_cost": total_cost,
                    "verified_score": final["verified_score"],
                })
                return final

            # General mode: single pass
            result = await self.verifier.verify_fact(
                fact=text,
                search_type=search_type,
                gpt_model=model,
                lang=lang,
                progress_callback=progress_callback,
                content_lang=content_lang,
                max_cost=max_credits,
                pipeline_profile="normal",  # Use normal profile for general mode
            )
            result.pop("_internal", None)
            result["detected_lang"] = detected_lang
            result["detected_lang_prob"] = detected_prob
            result["search_lang"] = content_lang
            if max_credits is not None:
                result["budget"] = {
                    "max_credits": int(max_credits),
                    "spent": int(result.get("cost", 0) or 0),
                    "limited": bool((result.get("search") or {}).get("budget_limited")),
                }
            return result
        finally:
            Trace.stop()
