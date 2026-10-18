# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Core extraction prompt contract: rules added after benchmark / trace defects (2026-10-18)."""

from spectrue_core.adapters.llm.claims_prompts import build_core_extraction_prompt


def test_core_prompt_keeps_main_assertion_and_drops_glosses_and_ads():
    p = build_core_extraction_prompt(text_excerpt="X")
    assert "THE MAIN ASSERTION IS NEVER DROPPED" in p          # avt_372: thesis dropped, gloss kept
    assert "**Glosses**" in p
    assert "**Ads and sponsor segments**" in p                  # St Petersburg deep trace: "Liberty VPN" ad checked


def test_standard_and_deep_judges_share_the_assertion_rules():
    """Deep said NEI for 'VK reported a hit' (wanted VK's own statement) where standard said 0.88 (2026-10-18)."""
    from spectrue_core.adapters.llm.claim_judge_prompts import build_claim_judge_system_prompt
    from spectrue_core.adapters.llm.scoring_contract import ASSERTION_RULES, SCORING_SCALE
    assert "ATTRIBUTION CLAIMS" in ASSERTION_RULES and "WHAT IS SCORED" in ASSERTION_RULES
    assert ASSERTION_RULES in SCORING_SCALE
    assert ASSERTION_RULES in build_claim_judge_system_prompt(lang="uk")


def test_core_prompt_asks_two_to_three_distinct_queries():
    """Six-fact run 2026-10-18: with call B off only call A's queries remain; 2 were too few to find the 1960 table."""
    p = build_core_extraction_prompt(text_excerpt="X")
    assert "2-3 web search queries" in p and "primary record" in p
