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
import os
from typing import Tuple, List, Optional

from spectrue_core.config import SpectrueConfig
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.evidence_pack import OracleCheckResult

logger = logging.getLogger(__name__)


class GoogleFactCheckTool:
    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, config: SpectrueConfig = None, oracle_validator=None):
        # Clean API Key Logic (Standard Only)
        self.api_key = (
            (config.google_fact_check_key if config else None)
            or (config.google_search_api_key if config else None)
            or os.getenv("GOOGLE_FACT_CHECK_KEY")
        )
        
        self.client = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._oracle_validator = oracle_validator

    async def aclose(self):
        """Explicit resource cleanup."""
        await self.client.aclose()

    def _normalize_query(self, query: str) -> str:
        """Correct Regex for Raw Strings."""
        q = (query or "").strip()
        q = re.sub(r"http\S+", "", q)
        q = re.sub(r"@\w+", "", q)
        q = re.sub(r"#\w+", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        if len(q) > 256:
            q = q[:256].strip()
        return q

    async def _fetch_all_candidates(self, query: str) -> Tuple[Optional[List[dict]], Optional[dict]]:
        """Returns (candidates, error_info)."""
        if not self.api_key:
            return None, {"code": 401, "msg": "API Key missing"}

        q = self._normalize_query(query)
        if len(q) < 3:
            return [], None

        try:
            params = {"query": q, "key": self.api_key, "pageSize": 5}
            Trace.event("google_fact_check.request", {"url": self.BASE_URL, "query": q[:300]})
            
            response = await self.client.get(self.BASE_URL, params=params)
            
            # Log Response Trace for Debugging
            Trace.event("google_fact_check.response", {
                "status_code": response.status_code,
                "text": (response.text or "")[:500],
            })
            
            if response.is_error:
                logger.warning(f"[Google FC] HTTP Error {response.status_code}: {response.text[:200]}")
                return None, {"code": response.status_code, "msg": response.text[:200]}

            data = response.json()
            claims = data.get("claims", [])
            if not claims:
                return [], None

            candidates = []
            for idx, claim in enumerate(claims[:5]):
                claim_text = claim.get("text", "")
                review = claim.get("claimReview", [])[0] if claim.get("claimReview") else {}
                publisher_name = review.get("publisher", {}).get("name", "Unknown")
                
                candidates.append({
                    "index": idx,
                    "claim_text": claim_text,
                    "publisher": publisher_name,
                    "url": review.get("url", ""),
                    "title": review.get("title", claim_text),
                    "rating": review.get("textualRating", "Unknown"),
                    "source_provider": f"{publisher_name} via Google Fact Check",
                })
            
            return candidates, None

        except Exception as e:
            logger.exception(f"[Google FC] Unexpected error: {e}")
            return None, {"code": 500, "msg": str(e)}

    async def search_and_validate(self, user_claim: str, intent: str = "news") -> OracleCheckResult:
        """Hybrid check with safe LLM integration."""
        empty_result: OracleCheckResult = {
            "status": "EMPTY",
            "url": None, "claim_reviewed": None, "summary": None,
            "relevance_score": 0.0, "is_jackpot": False,
            "publisher": None, "rating": None, "source_provider": None
        }
        
        candidates, error_info = await self._fetch_all_candidates(user_claim)
        
        if candidates is None:
            err_res = empty_result.copy()
            err_res["status"] = "ERROR"
            err_res["error_status_code"] = error_info.get("code")
            err_res["error_detail"] = error_info.get("msg")
            return err_res
            
        if not candidates:
            return empty_result
        
        # If no validator configured - explicit disabled mode
        if not self._oracle_validator:
            logger.warning("[Google FC] ⚠️ No LLM validator configured. Oracle disabled.")
            disabled_res = empty_result.copy()
            disabled_res["status"] = "DISABLED"
            return disabled_res
        
        # Named arguments for validator call
        validation = await self._oracle_validator.validate_batch(
            user_claim=user_claim, 
            candidates=candidates
        )
        
        # Strict Bounds Check
        best_idx = int(validation.get("best_index", -1))
        if best_idx < 0 or best_idx >= len(candidates):
            # Valid logic: LLM looked but found nothing relevant
            return empty_result
            
        winner = candidates[best_idx]
        
        return {
            "status": validation.get("status", "MIXED"),
            "url": winner["url"],
            "claim_reviewed": winner["claim_text"],
            "summary": winner["title"],
            "relevance_score": validation.get("relevance_score", 0.0),
            "is_jackpot": validation.get("is_jackpot", False),
            "verified_score": validation.get("verified_score"),
            "danger_score": validation.get("danger_score"),
            "publisher": winner["publisher"],
            "rating": winner["rating"],
            "source_provider": winner.get("source_provider"),
        }
