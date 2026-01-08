# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

import pytest
from pydantic import ValidationError

from spectrue_core.schema.rgba_audit import ClaimAudit, EvidenceAudit
from tests.fixtures.rgba_audit_fixtures import (
    make_claim_audit,
    make_evidence_audit,
)


def test_claim_audit_validates():
    audit = ClaimAudit(**make_claim_audit())
    assert audit.claim_id == "c1"
    assert audit.assertion_strength == "strong"


def test_claim_audit_rejects_invalid_predicate_type():
    with pytest.raises(ValidationError):
        ClaimAudit(**make_claim_audit(predicate_type="unsupported"))


def test_evidence_audit_validates():
    audit = EvidenceAudit(**make_evidence_audit())
    assert audit.evidence_id == "e1"
    assert audit.stance == "support"


def test_evidence_audit_rejects_invalid_stance():
    with pytest.raises(ValidationError):
        EvidenceAudit(**make_evidence_audit(stance="maybe"))
