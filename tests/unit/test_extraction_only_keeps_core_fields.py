# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Deep mode's extraction-only first pass must keep core claim fields.

Regression from trace 2026-10-18_15-54-29: `skip_enrichment=True` rebuilt claims
from 7 fields, dropping subject_entities / retrieval_seed_terms / time_anchor.
The stage-2 planning call does not produce them, so every deep claim reached
clustering with no entities and set-cover representatives fell back to top-k.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from spectrue_core.adapters.llm.claims import ClaimExtractionSkill
from spectrue_core.runtime_config import EngineRuntimeConfig

TEXT = ("Бамбук може вирости на 91 см за одну добу — це найшвидше зростання серед усіх рослин. "
        "Ейфелева вежа має висоту 330 метрів. Населення Києва перевищує 2,9 мільйона осіб. "
        "На мою думку, бамбук — найкрасивіша рослина у світі. Завтра в Києві буде сонячно.")
CORE = json.loads((Path(__file__).parent.parent / "fixtures/traces/deep_core_extraction_2026-10-18.json").read_text())


@pytest.mark.asyncio
async def test_extraction_only_claims_keep_entities_and_seed_terms():
    cfg = MagicMock()
    cfg.runtime = EngineRuntimeConfig.load_from_env()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=CORE)
    skill = ClaimExtractionSkill(cfg, llm)
    claims, *_ = await skill.extract_claims(TEXT, lang="uk", skip_enrichment=True)
    by_text = {c["normalized_text"]: c for c in claims}
    eiffel = next(c for t, c in by_text.items() if "Eiffel" in t)
    assert eiffel["subject_entities"] == ["Eiffel Tower"]
    assert eiffel["retrieval_seed_terms"]
    assert all(c.get("subject_entities") for c in claims)
    # no invalid placeholder role; stage-2 enrichment assigns the real one
    assert all(c.get("claim_role") != "target" for c in claims)
