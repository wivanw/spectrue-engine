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
from typing import Optional, Dict, Any
from spectrue_core.config import SpectrueConfig

logger = logging.getLogger(__name__)

class GoogleFactCheckTool:
    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, config: SpectrueConfig = None):
        if config:
            self.api_key = config.google_search_api_key
        else:
             # Fallback (optional, better to require config)
             self.api_key = None

    async def search(self, query: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        """
        Searches for fact checks using Google Fact Check Tools API.
        Returns a FactVerificationResult-compatible dictionary if a high-confidence match is found.
        """
        if not self.api_key:
            logger.warning("GOOGLE_FACT_CHECK_KEY is not set. Skipping Google Fact Check.")
            return None

        try:
            logger.info(f"[Google FC] Querying: '{query[:100]}...' (lang={lang})")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.BASE_URL,
                    params={
                        "query": query,
                        "key": self.api_key,
                        "languageCode": lang,
                        "pageSize": 1
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                
            logger.info(f"[Google FC] Response status: {response.status_code}")
            claims = data.get("claims", [])
            logger.info(f"[Google FC] Found {len(claims)} claim(s)")
            
            if not claims:
                return None

            claim = claims[0]
            claim_review = claim.get("claimReview", [])
            if not claim_review:
                logger.warning("[Google FC] Claim found but no reviews available")
                return None
            
            review = claim_review[0]
            publisher = review.get("publisher", {}).get("name", "Unknown")
            url = review.get("url", "")
            title = review.get("title", claim.get("text", ""))
            rating = review.get("textualRating", "Unknown")
            
            logger.info(f"[Google FC] Publisher: {publisher}, Rating: {rating}")
            
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

            logger.info(f"[Google FC] ✓ Matched! Returning verified_score={verified_score}")
            
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

        except httpx.HTTPError as e:
            logger.error(f"[Google FC] ✗ HTTP error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[Google FC] ✗ Unexpected error: {e}")
            return None