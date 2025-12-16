# Simplified engine.py without TextAnalyzer dependency

import logging
import json
from typing import Dict, Any, Optional
from langdetect import detect_langs, DetectorFactory, LangDetectException
import re
from uuid import uuid4
from datetime import datetime

from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.fact_verifier_composite import FactVerifierComposite, MODEL_COSTS
from spectrue_core.utils.trace import Trace

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
        
        lang_mapping = {
            "pt": "es", "it": "es", "ca": "es",
            "nl": "de", "pl": "uk", "cs": "uk", "sk": "uk", "be": "uk",
            "bg": "ru", "sr": "ru",
            "ko": "ja", "vi": "zh", "th": "zh",
        }
        
        return lang_mapping.get(detected_lang, fallback), confidence
        
    except LangDetectException:
        return fallback, 0.0


class SpectrueEngine:
    """The main entry point for the Spectrue Fact-Checking Engine."""
    
    def __init__(self, config: SpectrueConfig):
        self.config = config
        self.verifier = FactVerifierComposite(config)
        try:
            logger.info("Effective config: %s", json.dumps(self.config.runtime.to_safe_log_dict(), ensure_ascii=False))
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
        search_provider: str = "auto",
        progress_callback = None,
        max_credits: Optional[int] = None,
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

            def extract_claims(raw: str, max_claims: int = 2) -> list[str]:
                s = re.sub(r"\s+", " ", (raw or "")).strip()
                if not s:
                    return []
                if len(s) <= 260:
                    return [s]

                # Lightweight sentence splitting (no extra model calls).
                parts = re.split(r"(?<=[.!?…])\s+", s)
                candidates = []
                for p in parts:
                    p = p.strip()
                    if 30 <= len(p) <= 260:
                        candidates.append(p)

                if not candidates:
                    return [s[:260]]

                def score(sent: str) -> float:
                    t = sent
                    sc = 0.0
                    # Prefer numerals, percentages, dates — usually factual claims
                    if re.search(r"\b\d{2,4}\b", t):
                        sc += 2.0
                    if "%" in t:
                        sc += 1.0
                    if re.search(r"[$€₴₽]|USD|EUR|UAH|RUB", t):
                        sc += 0.5
                    # Prefer longer but not too long
                    sc += min(1.0, len(t) / 180.0)
                    return sc

                candidates.sort(key=score, reverse=True)
                out: list[str] = []
                for c in candidates:
                    if c not in out:
                        out.append(c)
                    if len(out) >= max_claims:
                        break
                return out

            if analysis_mode == "deep":
                if progress_callback:
                    await progress_callback("extracting_claims")

                # Cost-aware: cap number of separate claim analyses.
                max_claims = int(getattr(self.config.runtime.tunables, "max_claims_deep", 2) or 2)
                max_claims = max(1, min(max_claims, 3))

                claims = extract_claims(text, max_claims=max_claims)
                if not claims:
                    claims = [text.strip()]

                primary = claims[0]
                primary_result = await self.verifier.verify_fact(
                    fact=primary,
                    search_type=search_type,
                    gpt_model=model,
                    lang=lang,
                    progress_callback=progress_callback,
                    content_lang=content_lang,
                    include_internal=True,
                    max_cost=max_credits,
                )

                internal = primary_result.pop("_internal", None) or {}
                shared_context = internal.get("context") or ""
                shared_sources = internal.get("sources") or primary_result.get("sources") or []

                if not shared_context:
                    claims = claims[:1]

                details = []
                total_cost = int(primary_result.get("cost", 0) or 0)
                remaining_budget = None
                if max_credits is not None:
                    try:
                        remaining_budget = int(max_credits) - total_cost
                    except Exception:
                        remaining_budget = None

                details.append(
                    {
                        "text": primary,
                        "rgba": primary_result.get("rgba"),
                        "rationale": primary_result.get("rationale", ""),
                        "sources": primary_result.get("sources", []),
                    }
                )

                extra_claims: list[str] = []
                for claim in claims[1:]:
                    if remaining_budget is not None and remaining_budget < per_claim_model_cost:
                        break
                    extra_claims.append(claim)
                    if remaining_budget is not None:
                        remaining_budget -= per_claim_model_cost

                for claim in extra_claims:
                    claim_result = await self.verifier.verify_fact(
                        fact=claim,
                        search_type=search_type,
                        gpt_model=model,
                        lang=lang,
                        progress_callback=progress_callback,
                        preloaded_context=shared_context,
                        preloaded_sources=shared_sources,
                        content_lang=content_lang,
                        include_internal=False,
                        max_cost=per_claim_model_cost if max_credits is not None else None,
                    )
                    total_cost += int(claim_result.get("cost", 0) or 0)
                    details.append(
                        {
                            "text": claim,
                            "rgba": claim_result.get("rgba"),
                            "rationale": claim_result.get("rationale", ""),
                            "sources": claim_result.get("sources", []),
                        }
                    )

                # Aggregate for UI compatibility
                rgba_list = [d.get("rgba") for d in details if isinstance(d.get("rgba"), list) and len(d.get("rgba")) == 4]
                if rgba_list:
                    avg = [
                        sum(v[i] for v in rgba_list) / len(rgba_list)
                        for i in range(4)
                    ]
                else:
                    avg = primary_result.get("rgba")

                final = dict(primary_result)
                final["text"] = text
                final["details"] = details
                final["average_rgba"] = avg
                final["cost"] = total_cost
                final["claims"] = [primary] + extra_claims
                final["detected_lang"] = detected_lang
                final["detected_lang_prob"] = detected_prob
                final["search_lang"] = content_lang
                if max_credits is not None:
                    final["budget"] = {
                        "max_credits": int(max_credits),
                        "spent": total_cost,
                        "limited": len(extra_claims) < max(0, len(claims) - 1),
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
                include_internal=False,
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
