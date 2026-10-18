# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""M146 T010/T013/T016: topics from entities, canonical accessors, CJK length bound."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from spectrue_core.domain.claims import fields
from spectrue_core.domain.claims.extraction import validate_core_claim
from spectrue_core.domain.claims.topics import assign_topics

FIX = Path(__file__).parent.parent / "fixtures/metadata/golden_responses.json"


def test_topics_are_components_of_shared_entities():
    claims = [
        {"id": "c1", "subject_entities": ["Bamboo"]},
        {"id": "c2", "subject_entities": ["bamboo", "plants"]},
        {"id": "c3", "subject_entities": ["Eiffel Tower"]},
        {"id": "c4", "subject_entities": []},
    ]
    assign_topics(claims)
    assert claims[0]["topic_key"] == claims[1]["topic_key"] != claims[2]["topic_key"]
    assert claims[0]["topic_group"] == "bamboo"
    assert claims[3]["topic_key"] == "tp_claim_c4" and claims[3]["topic_group"] == "Other"


def test_topic_key_is_stable():
    a = [{"id": "x", "subject_entities": ["Kyiv"]}]
    b = [{"id": "y", "subject_entities": ["KYIV "]}]
    assign_topics(a), assign_topics(b)
    assert a[0]["topic_key"] == b[0]["topic_key"]


def test_accessors_read_metadata_and_keep_zero():
    c = {"importance": 0.0, "metadata": SimpleNamespace(verification_target=SimpleNamespace(value="NONE"))}
    assert fields.unit(c, "importance") == 0.0  # was coerced to 0.5 by `or 0.5`
    assert fields.or_default(fields.unit(c, "importance"), 0.5) == 0.0
    assert fields.enum_value(c, "verification_target") == "none"
    assert fields.unit({}, "importance") is None


def test_cjk_claims_pass_validation_on_golden_set():
    data = json.loads(FIX.read_text())
    for key in ("ja_0", "zh_0"):
        core = next(x for x in data[key]["calls"] if x["trace_kind"] == "claim_extraction_core")["response"]
        results = [validate_core_claim(c) for c in core["claims"]]
        assert all(ok for ok, _ in results), (key, results)
