import os
from dataclasses import dataclass

from spectrue_core.verification.evidence_pack import Claim
from spectrue_core.agents.skills.base_skill import BaseSkill, logger


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


@dataclass(frozen=True)
class _SocialDecisionParams:
    """Decision economics for A' promotion.

    We intentionally avoid hard caps like p>0.8.
    Promotion is decided by expected value:
        EV = p_true * benefit_true - (1 - p_true) * cost_false
    """

    benefit_true: float
    cost_false: float

    @staticmethod
    def from_env() -> "_SocialDecisionParams":
        # Conservative defaults: false promotion is costlier than missing a true one.
        # Support both old (M67) and new (SOCIAL) env var names for backwards compatibility.
        benefit_true = float(
            os.getenv("SPECTRUE_SOCIAL_BENEFIT_TRUE") 
            or os.getenv("SPECTRUE_M67_BENEFIT_TRUE", "1.0")
        )
        cost_false = float(
            os.getenv("SPECTRUE_SOCIAL_COST_FALSE")
            or os.getenv("SPECTRUE_M67_COST_FALSE", "2.0")
        )
        # Avoid zero/negative which would break EV semantics.
        benefit_true = benefit_true if benefit_true > 0 else 1.0
        cost_false = cost_false if cost_false > 0 else 2.0
        return _SocialDecisionParams(benefit_true=benefit_true, cost_false=cost_false)

class InlineVerificationSkill(BaseSkill):
    """
    Skill for verifying specific types of evidence inline (e.g., Social Media Identity).
    Part of Tier A' Verification (Social Inline Verification).
    """

    async def verify_social_statement(
        self, 
        claim: Claim, 
        social_snippet: str | None, 
        profile_url: str | None
    ) -> dict:
        """
        Verify if a social media snippet is an official statement from the claim subject.
        
        Performs two independent checks:
        1. Identity Verification: Is this account the official account of the entity?
        2. Content Support: Does the text explicitly support the claim?
        
        Returns:
            {
                "is_identity_verified": float, # 0.0 - 1.0
                "is_statement_supported": float, # 0.0 - 1.0
                "tier": "A'" | "D",
                "decision": {
                    "p_identity": float,
                    "p_support": float,
                    "p_joint": float,
                    "p_joint_low": float,
                    "ev": float,
                    "benefit_true": float,
                    "cost_false": float,
                },
                "llm": {
                    "identity": {"p": float, "ci": [float,float], "reasons": list[str], "rule_violations": list[str]},
                    "support": {"p": float, "ci": [float,float], "reasons": list[str], "rule_violations": list[str]},
                }
            }
        """
        claim_text = None
        if isinstance(claim, dict):
            claim_text = claim.get("text")
        else:
            claim_text = getattr(claim, "text", None)

        if not social_snippet or not claim_text:
            return {"is_identity_verified": 0.0, "is_statement_supported": 0.0, "tier": "D"}

        params = _SocialDecisionParams.from_env()

        # 1) LLM returns probabilistic *features* (soft evidence), not a verdict.
        identity = await self._check_identity(str(claim_text), profile_url, social_snippet)
        support = await self._check_support(str(claim_text), social_snippet)

        p_identity = _clamp01(identity.get("p", 0.0))
        p_support = _clamp01(support.get("p", 0.0))

        # Conservative joint probability via independence assumption + lower-bound of CI.
        ci_i = identity.get("ci") or [p_identity, p_identity]
        ci_s = support.get("ci") or [p_support, p_support]
        p_joint = _clamp01(p_identity * p_support)
        p_joint_low = _clamp01(_clamp01(ci_i[0]) * _clamp01(ci_s[0]))

        # 2) Decision via expected value (no hard p-thresholds).
        # EV = p_true * benefit_true - (1 - p_true) * cost_false
        p_use = p_joint_low if p_joint_low > 0 else p_joint
        ev = (p_use * params.benefit_true) - ((1.0 - p_use) * params.cost_false)
        tier = "A'" if ev > 0 else "D"
            
        logger.debug(
            "[M67] Social Verification: p_id=%.2f p_supp=%.2f p_joint=%.2f (low=%.2f) EV=%.3f -> Tier %s",
            p_identity, p_support, p_joint, p_joint_low, ev, tier
        )
            
        return {
            "is_identity_verified": p_identity,
            "is_statement_supported": p_support,
            "tier": tier,
            "decision": {
                "p_identity": p_identity,
                "p_support": p_support,
                "p_joint": p_joint,
                "p_joint_low": p_joint_low,
                "ev": ev,
                "benefit_true": params.benefit_true,
                "cost_false": params.cost_false,
            },
            "llm": {
                "identity": identity,
                "support": support,
            },
        }

    async def _check_identity(self, claim_text: str, url: str | None, snippet: str) -> dict:
        prompt = f"""You are a strict auditor. Do NOT give a verdict.

We are extracting *probabilistic features* for an engine that will decide later.

INPUT FIELDS (treat as ground truth strings, do not hallucinate):
- claim_text: "{claim_text}"
- profile_url: "{(url or '').strip()}"
- snippet: "{snippet[:650]}"

TASK:
Estimate P(account belongs to the main entity in claim_text).

Rules:
- Use ONLY the given profile_url + snippet. If missing, reflect uncertainty.
- No world knowledge lookups. No guessing beyond the provided strings.
- If you cannot justify a high probability, keep it low.

Return STRICT JSON with:
{{
  "p": float,                  // 0..1
  "ci": [float, float],        // conservative 90% interval, low<=p<=high
  "reasons": [string],         // short, factual; refer to observed cues (handle match, official badge text, etc.)
  "rule_violations": [string]  // e.g., "insufficient_input", "ambiguous_entity", "requires_external_knowledge"
}}
"""
        return await self._run_check(prompt, "m67_identity")

    async def _check_support(self, claim_text: str, snippet: str) -> dict:
        prompt = f"""You are a strict auditor. Do NOT give a verdict.

We are extracting *probabilistic features* for an engine that will decide later.

INPUT FIELDS (treat as ground truth strings, do not hallucinate):
- claim_text: "{claim_text}"
- snippet: "{snippet[:1200]}"

TASK:
Estimate P(snippet explicitly supports claim_text).

Rules:
- Consider only the snippet. No external context.
- Explicit support means the snippet unambiguously confirms the claim (not vague, not just related).
- If snippet contradicts, set p near 0.

Return STRICT JSON with:
{{
  "p": float,                  // 0..1
  "ci": [float, float],        // conservative 90% interval
  "reasons": [string],
  "rule_violations": [string]
}}
"""
        return await self._run_check(prompt, "m67_support")

    async def _run_check(self, prompt: str, kind: str) -> dict:
        try:
            res = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a strict verification auditor.",
                reasoning_effort="low",
                timeout=10.0,
                trace_kind=kind
            )
            p = _clamp01(res.get("p", res.get("score", 0.0)))
            ci = res.get("ci")
            if not (isinstance(ci, list) and len(ci) == 2):
                ci = [p, p]
            ci = [_clamp01(ci[0]), _clamp01(ci[1])]
            lo, hi = (min(ci[0], ci[1]), max(ci[0], ci[1]))
            # Ensure p is inside CI conservatively.
            p = _clamp01(min(max(p, lo), hi))
            reasons = res.get("reasons")
            if not isinstance(reasons, list):
                reason = res.get("reason")
                reasons = [str(reason)] if reason else []
            rule_violations = res.get("rule_violations")
            if not isinstance(rule_violations, list):
                rule_violations = []
            return {
                "p": p,
                "ci": [lo, hi],
                "reasons": [str(r) for r in reasons if str(r).strip()][:6],
                "rule_violations": [str(v) for v in rule_violations if str(v).strip()][:6],
            }
        except Exception as e:
            logger.warning("Check %s failed: %s", kind, e)
            return {"p": 0.0, "ci": [0.0, 1.0], "reasons": [], "rule_violations": ["exception"]}
