# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

import httpx
import logging
import re
from typing import Optional, Dict, Any
from spectrue_core.config import SpectrueConfig
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.evidence_pack import OracleCheckResult

logger = logging.getLogger(__name__)


class GoogleFactCheckTool:
    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, config: SpectrueConfig = None, oracle_validator=None):
        """
        Initialize GoogleFactCheckTool.
        
        Args:
            config: Spectrue configuration with API keys
            oracle_validator: Optional OracleValidationSkill for semantic validation (M63)
        """
        if config:
            # Prefer dedicated Fact Check key. Fallback to the CSE key if user uses one key for both APIs.
            self.api_key = config.google_fact_check_key or config.google_search_api_key
        else:
            self.api_key = None
        self.client = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._oracle_validator = oracle_validator

    def _normalize_query(self, query: str) -> str:
        q = (query or "").strip()
        q = re.sub(r"http\\S+", "", q)
        q = re.sub(r"@\\w+", "", q)
        q = re.sub(r"#\\w+", "", q)
        q = re.sub(r"\\s+", " ", q).strip()
        if len(q) > 256:
            q = q[:256].strip()
        return q

    async def _fetch_all_candidates(self, query: str) -> list[dict]:
        """
        M63: Fetch ALL fact-check candidates from API for batch LLM validation.
        
        Returns list of candidate dicts (up to 5) for batch validation.
        Returns empty list if no claims found or API error.
        """
        if not self.api_key:
            logger.warning("[Google FC] Key is not set. Skipping Google Fact Check.")
            return []

        q = self._normalize_query(query)
        if len(q) < 3:
            return []

        try:
            params = {
                "query": q,
                "key": self.api_key,
                "pageSize": 5,
            }

            Trace.event("google_fact_check.request", {"url": self.BASE_URL, "query": q[:50]})
            response = await self.client.get(self.BASE_URL, params=params)
            Trace.event(
                "google_fact_check.response",
                {
                    "status_code": response.status_code,
                    "text": response.text[:500] if response.text else "",
                },
            )
            response.raise_for_status()
            data = response.json()
            claims = data.get("claims", [])

            if not claims:
                return []

            # Convert all claims to candidate format for batch validation
            candidates = []
            query_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', q))
            
            for idx, claim in enumerate(claims[:5]):  # Max 5 candidates
                claim_text = claim.get("text", "")
                claim_review = claim.get("claimReview", [])
                
                if not claim_review:
                    continue
                
                review = claim_review[0]
                publisher_name = review.get("publisher", {}).get("name", "Unknown")
                
                # Calculate keyword overlap for transparency
                claim_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', claim_text))
                overlap = len(query_words & claim_words)
                
                candidates.append({
                    "index": idx,
                    "claim_text": claim_text,
                    "claimant": claim.get("claimant", ""),
                    "publisher": publisher_name,
                    "url": review.get("url", ""),
                    "title": review.get("title", claim_text),
                    "rating": review.get("textualRating", "Unknown"),
                    "language_code": review.get("languageCode", ""),
                    "keyword_overlap": overlap,
                    # M63: source_provider for UX
                    "source_provider": f"{publisher_name} via Google Fact Check",
                })
            
            logger.info("[Google FC] Found %d candidates for batch validation", len(candidates))
            return candidates

        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", None)
            body = ""
            try:
                body = (e.response.text or "")[:300]
            except Exception:
                body = ""
            logger.error("[Google FC] ✗ HTTP status error: %s (%s) body=%s", e, status, body)
            return []
        except httpx.HTTPError as e:
            logger.error("[Google FC] ✗ HTTP error: %s", e)
            return []
        except Exception as e:
            logger.exception("[Google FC] ✗ Unexpected error: %s", e)
            return []

    async def search_and_validate(self, user_claim: str, intent: str = "news") -> OracleCheckResult:
        """
        M63: Search fact-checks and validate ALL candidates with LLM in one batch call.
        
        This is the "hybrid mode" entry point that:
        1. Fetches up to 5 candidates from Google Fact Check API
        2. Sends ALL candidates to LLM in a SINGLE prompt for batch evaluation
        3. Returns the BEST match as OracleCheckResult
        
        Three-scenario flow based on relevance_score:
        - JACKPOT (>0.9): Stop pipeline, return Oracle result immediately
        - EVIDENCE (0.5-0.9): Add to evidence pack, continue to Tavily
        - MISS (<0.5): Ignore, proceed to standard search
        
        Args:
            user_claim: The claim text to search for
            intent: Article intent (news/evergreen/official/opinion/prediction)
            
        Returns:
            OracleCheckResult with status, relevance_score, is_jackpot, source_provider
        """
        empty_result: OracleCheckResult = {
            "status": "EMPTY",
            "url": None,
            "claim_reviewed": None,
            "summary": None,
            "relevance_score": 0.0,
            "is_jackpot": False,
            "publisher": None,
            "rating": None,
            "source_provider": None,
        }
        
        # Step 1: Fetch ALL candidates from API
        candidates = await self._fetch_all_candidates(user_claim)
        
        # Technical Note: Skip LLM call entirely if 0 results
        if not candidates:
            logger.debug("[Google FC] No candidates found - skipping LLM validation")
            return empty_result
        
        logger.info("[Google FC] Batch validating %d candidates", len(candidates))
        
        # Step 2: Batch LLM validation (if validator available)
        if self._oracle_validator:
            validation = await self._oracle_validator.validate_batch(
                user_claim=user_claim,
                candidates=candidates,
            )
            
            best_idx = validation.get("best_index", -1)
            relevance_score = validation.get("relevance_score", 0.0)
            status = validation.get("status", "MIXED")
            is_jackpot = validation.get("is_jackpot", False)
            
            # Get the winning candidate
            if 0 <= best_idx < len(candidates):
                winner = candidates[best_idx]
            else:
                # LLM said no good matches
                logger.info("[Google FC] Batch validation: no suitable match found")
                return empty_result
            
            logger.info(
                "[Google FC] Batch validation: best=%d, score=%.2f, status=%s, jackpot=%s",
                best_idx, relevance_score, status, is_jackpot
            )
        else:
            # Fallback: Pick by highest keyword overlap (no LLM)
            winner = max(candidates, key=lambda c: c.get("keyword_overlap", 0))
            overlap = winner.get("keyword_overlap", 0)
            
            # Require minimum overlap
            if overlap < 2:
                logger.info("[Google FC] No LLM validator and weak keyword match")
                return empty_result
            
            relevance_score = min(0.75, 0.2 + overlap * 0.15)
            status = self._derive_status_from_rating(winner["rating"])
            is_jackpot = False
            
            logger.warning("[Google FC] No LLM validator - using keyword heuristic (score=%.2f)", relevance_score)
        
        return {
            "status": status,
            "url": winner["url"],
            "claim_reviewed": winner["claim_text"],
            "summary": winner["title"],
            "relevance_score": relevance_score,
            "is_jackpot": is_jackpot,
            "publisher": winner["publisher"],
            "rating": winner["rating"],
            "source_provider": winner.get("source_provider", f"{winner['publisher']} via Google Fact Check"),
        }

    def _derive_status_from_rating(self, rating: str) -> str:
        """Derive Oracle status from textual rating (fallback without LLM)."""
        rating_lower = (rating or "").lower()
        
        if any(x in rating_lower for x in ["false", "fake", "incorrect", "debunked", "lie", "pants on fire"]):
            return "REFUTED"
        elif any(x in rating_lower for x in ["true", "correct", "accurate", "verified"]):
            return "CONFIRMED"
        elif any(x in rating_lower for x in ["misleading", "mixture", "half-true", "partly", "needs context"]):
            return "MIXED"
        
        return "MIXED"

    async def search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Searches for fact checks using Google Fact Check Tools API.
        Returns a FactVerificationResult-compatible dictionary if a high-confidence match is found.
        
        NOTE: This is the LEGACY method. For M63 hybrid mode, use search_and_validate() instead.
        """
        if not self.api_key:
            logger.warning("[Google FC] Key is not set. Skipping Google Fact Check.")
            return None

        q = self._normalize_query(query)
        if len(q) < 3:
            return None

        try:
            # Don't filter by language - fact-checks are mostly in English regardless of claim language.
            base_params = {
                "query": q,
                "key": self.api_key,
                "pageSize": 5,
            }

            params = dict(base_params)

            Trace.event("google_fact_check.request", {"url": self.BASE_URL, "params": params})
            response = await self.client.get(self.BASE_URL, params=params)
            Trace.event(
                "google_fact_check.response",
                {
                    "status_code": response.status_code,
                    "text": response.text,
                },
            )
            response.raise_for_status()
            data = response.json()
            claims = data.get("claims", [])

            if not claims:
                return None

            # Find most relevant claim (not just first!)
            # Google may return unrelated claims matching just some keywords
            query_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', q))
            
            best_claim = None
            best_overlap = 0
            
            for claim in claims:
                claim_text = claim.get("text", "")
                claim_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', claim_text))
                
                # Count overlapping significant words
                overlap = len(query_words & claim_words)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_claim = claim
            
            # Require at least 2 matching words to consider it relevant
            if best_overlap < 2 or not best_claim:
                logger.info("[Google FC] No relevant claim found (best overlap: %d words)", best_overlap)
                return None
            
            claim = best_claim
            claim_review = claim.get("claimReview", [])
            if not claim_review:
                logger.warning("[Google FC] Claim found but no reviews available")
                return None
            
            review = claim_review[0]
            publisher = review.get("publisher", {}).get("name", "Unknown")
            url = review.get("url", "")
            title = review.get("title", claim.get("text", ""))
            rating = review.get("textualRating", "Unknown")
            
            # Determine scores based on rating (heuristic)
            # This is a simplified mapping. Real-world mapping might need more robust NLP or specific string matching.
            verified_score = 0.5
            danger_score = 0.5
            
            rating_lower = rating.lower()
            if any(x in rating_lower for x in ["false", "fake", "incorrect", "debunked", "lie"]):
                verified_score = 0.1
                danger_score = 0.9
            elif any(x in rating_lower for x in ["true", "correct", "accurate", "verified"]):
                verified_score = 0.9
                danger_score = 0.1
            elif any(x in rating_lower for x in ["misleading", "mixture", "half-true", "partly false"]):
                verified_score = 0.4
                danger_score = 0.6

            # Construct analysis text
            analysis_text = f"According to {publisher}, this claim is rated as '{rating}'. {title}"

            # Construct sources
            sources = [
                {
                    "title": f"Fact Check by {publisher}",
                    "link": url,
                    "snippet": f"Rating: {rating}. {title}",
                    "origin": "GOOGLE_FACT_CHECK"
                }
            ]

            # Construct rationale for user display
            rationale = f"Fact check by {publisher}: Rated as '{rating}'. {title}"
            
            return {
                "verified_score": verified_score,
                "confidence_score": 1.0,  # High confidence in the fact check existence
                "danger_score": danger_score,
                "context_score": 1.0,     # Trusted source
                "style_score": 1.0,       # Professional content
                "analysis": analysis_text,
                "rationale": rationale,   # For frontend display
                "sources": sources,
                "cost": 0,
                "rgba": [danger_score, verified_score, 1.0, 1.0]
            }

        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", None)
            body = ""
            try:
                body = (e.response.text or "")[:300]
            except Exception:
                body = ""
            logger.error("[Google FC] ✗ HTTP status error: %s (%s) body=%s", e, status, body)
            return None
        except httpx.HTTPError as e:
            logger.error("[Google FC] ✗ HTTP error: %s", e)
            return None
        except Exception as e:
            logger.exception(f"[Google FC] ✗ Unexpected error: {e}")
            return None
