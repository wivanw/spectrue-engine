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
from spectrue_core.verification.costs import MODEL_COSTS
from spectrue_core.utils.trace import Trace
from spectrue_core.analysis.text_analyzer import TextAnalyzer

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
            per_claim_model_cost = int(MODEL_COSTS.get(model, 20) or 0)
            
            # Detect content language
            detected_lang, detected_prob = detect_content_language(text, fallback=lang)
            min_prob = float(getattr(self.config.runtime.tunables, "langdetect_min_prob", 0.80) or 0.80)
            content_lang = detected_lang if detected_prob >= min_prob else lang

            Trace.event("engine.analyze_text.start", {
                "analysis_mode": analysis_mode,
                "text_len": len(text),
                "model": model,
            })

            if analysis_mode == "deep":
                if progress_callback:
                    await progress_callback("extracting_claims")

                # Deep analysis: if input is a URL, first extract the article content
                working_text = text
                if text.strip().startswith("http://") or text.strip().startswith("https://"):
                    # Fetch article content before claim extraction
                    extracted = await self.fetch_url_content(text.strip())
                    if extracted:
                        working_text = extracted
                        Trace.event("engine.deep.url_extracted", {"chars": len(extracted)})
                    else:
                        Trace.event("engine.deep.url_extraction_failed", {"url": text.strip()[:100]})

                # Deep analysis: Extract claims from the ENTIRE text once
                # Then verify each claim with a full pipeline pass
                # First, run one pipeline pass to extract claims
                initial_result = await self.verifier.verify_fact(
                    fact=working_text,
                    search_type=search_type,
                    gpt_model=model,
                    lang=lang,
                    progress_callback=progress_callback,
                    content_lang=content_lang,
                    max_cost=max_credits,
                    extract_claims_only=True,  # New flag: just extract claims, don't verify
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

                details = []
                total_cost = int(initial_result.get("cost", 0) or 0)  # Include initial extraction cost
                accumulated_sources: list[dict] = initial_result.get("sources", []) or []
                all_verified_claims: list[str] = []

                # Verify each extracted claim with full pipeline
                for idx, claim_obj in enumerate(extracted_claims):
                    claim_text = claim_obj.get("text", "").strip()
                    if not claim_text:
                        continue
                        
                    # Budget check: stop if no credits remain
                    if max_credits is not None:
                        remaining = max_credits - total_cost
                        if remaining < per_claim_model_cost:
                            break

                    # Emit per-claim progress status
                    if progress_callback:
                        await progress_callback(f"verifying_claim:{idx + 1}:{len(extracted_claims)}")

                    # Each claim gets full verification (primary treatment)
                    # Pass accumulated sources as preloaded, but pipeline will supplement if not relevant
                    claim_result = await self.verifier.verify_fact(
                        fact=claim_text,
                        search_type=search_type,
                        gpt_model=model,
                        lang=lang,
                        progress_callback=progress_callback,
                        content_lang=content_lang,
                        preloaded_sources=accumulated_sources if accumulated_sources else None,
                        max_cost=max_credits - total_cost if max_credits is not None else None,
                    )

                    claim_cost = int(claim_result.get("cost", 0) or 0)
                    total_cost += claim_cost
                    all_verified_claims.append(claim_text)

                    # Accumulate new sources for subsequent claims
                    new_sources = claim_result.get("sources", []) or []
                    seen_urls = {s.get("url") for s in accumulated_sources if s.get("url")}
                    for src in new_sources:
                        if src.get("url") and src.get("url") not in seen_urls:
                            accumulated_sources.append(src)
                            seen_urls.add(src.get("url"))

                    # Uniform detail entry for all claims
                    details.append({
                        "text": claim_text,
                        "rgba": claim_result.get("rgba") or [0.0, 0.0, 0.0, 0.5],
                        "rationale": claim_result.get("rationale", ""),
                        "sources": new_sources,
                        "verified_score": claim_result.get("verified_score"),
                        "danger_score": claim_result.get("danger_score"),
                    })

                # Aggregate RGBA for overall score
                rgba_list = [
                    d.get("rgba") for d in details 
                    if isinstance(d.get("rgba"), list) and len(d.get("rgba")) == 4
                ]
                if rgba_list:
                    avg_rgba = [
                        sum(v[i] for v in rgba_list) / len(rgba_list)
                        for i in range(4)
                    ]
                else:
                    avg_rgba = [0.0, 0.0, 0.0, 0.5]

                # Aggregate verified_score (average of all claims)
                verified_scores = [
                    d.get("verified_score") for d in details 
                    if isinstance(d.get("verified_score"), (int, float))
                ]
                avg_verified = sum(verified_scores) / len(verified_scores) if verified_scores else 0.0

                final = {
                    "text": text,
                    "verified_score": avg_verified,
                    "rgba": avg_rgba,
                    "average_rgba": avg_rgba,
                    "details": details,
                    "sources": accumulated_sources,
                    "cost": total_cost,
                    "claims": all_verified_claims,
                    "detected_lang": detected_lang,
                    "detected_lang_prob": detected_prob,
                    "search_lang": content_lang,
                    "analysis_mode": "deep",
                }
                if max_credits is not None:
                    final["budget"] = {
                        "max_credits": int(max_credits),
                        "spent": total_cost,
                        "limited": len(all_verified_claims) < len(claims),
                    }
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
