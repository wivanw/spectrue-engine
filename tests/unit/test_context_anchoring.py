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

"""
Unit tests for M129 EVIDENCE_MISMATCH status.
"""

from spectrue_core.schema.rgba_audit import RGBAStatus, RGBA_STATUS_LEGACY_CODES


class TestEvidenceMismatchStatus:
    """Tests for new EVIDENCE_MISMATCH status."""

    def test_evidence_mismatch_status_exists(self):
        """RGBAStatus should have EVIDENCE_MISMATCH."""
        assert hasattr(RGBAStatus, "EVIDENCE_MISMATCH")
        assert RGBAStatus.EVIDENCE_MISMATCH.value == "EVIDENCE_MISMATCH"

    def test_evidence_mismatch_legacy_code(self):
        """EVIDENCE_MISMATCH should map to -6."""
        assert RGBA_STATUS_LEGACY_CODES[RGBAStatus.EVIDENCE_MISMATCH] == -6

    def test_all_statuses_have_legacy_codes(self):
        """All RGBAStatus values should have legacy codes."""
        for status in RGBAStatus:
            assert status in RGBA_STATUS_LEGACY_CODES
