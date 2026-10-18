# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.



from spectrue_core.domain.verification.search.search_policy import (
    SearchProfileName,
    default_search_policy,
    resolve_profile_name,
)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_profile_name() Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveProfileName:
    """Tests for resolve_profile_name() function."""

    def test_none_returns_general(self):
        """None input should return 'general' profile."""
        assert resolve_profile_name(None) == SearchProfileName.GENERAL

    def test_empty_string_returns_general(self):
        """Empty string should return 'general' profile."""
        assert resolve_profile_name("") == SearchProfileName.GENERAL

    def test_general_returns_general(self):
        """'general' should return 'general' profile."""
        assert resolve_profile_name("general") == SearchProfileName.GENERAL

    def test_deep_returns_deep(self):
        """'deep' should return 'deep' profile."""
        assert resolve_profile_name("deep") == SearchProfileName.DEEP

    def test_deep_v2_returns_deep(self):
        """'deep_v2' should return 'deep' profile."""
        assert resolve_profile_name("deep_v2") == SearchProfileName.DEEP

    def test_case_insensitive(self):
        """Profile name resolution should be case-insensitive."""
        assert resolve_profile_name("DEEP") == SearchProfileName.DEEP
        assert resolve_profile_name("DEEP_V2") == SearchProfileName.DEEP
        assert resolve_profile_name("GENERAL") == SearchProfileName.GENERAL

    def test_whitespace_trimmed(self):
        """Whitespace should be trimmed from input."""
        assert resolve_profile_name("  deep  ") == SearchProfileName.DEEP
        assert resolve_profile_name("\tdeep_v2\n") == SearchProfileName.DEEP

    def test_unknown_returns_general(self):
        """Unknown profile names should default to 'general'."""
        assert resolve_profile_name("unknown") == SearchProfileName.GENERAL
        assert resolve_profile_name("custom") == SearchProfileName.GENERAL

    def test_analysis_mode_enum_general(self):
        """AnalysisMode.GENERAL enum should return 'general' profile."""
        from spectrue_core.pipeline.mode import AnalysisMode
        assert resolve_profile_name(AnalysisMode.GENERAL) == SearchProfileName.GENERAL

    def test_analysis_mode_enum_deep(self):
        """AnalysisMode.DEEP enum should return 'deep' profile."""
        from spectrue_core.pipeline.mode import AnalysisMode
        assert resolve_profile_name(AnalysisMode.DEEP) == SearchProfileName.DEEP

    def test_analysis_mode_enum_deep_v2(self):
        """AnalysisMode.DEEP_V2 enum should return 'deep' profile."""
        from spectrue_core.pipeline.mode import AnalysisMode
        assert resolve_profile_name(AnalysisMode.DEEP_V2) == SearchProfileName.DEEP


# ─────────────────────────────────────────────────────────────────────────────
# default_search_policy() profiles
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultSearchPolicyProfiles:
    """Profile values the retrieval steps read via resolve_profile_name()."""

    def test_deep_v2_resolves_to_deep_profile(self):
        profile = default_search_policy().get_profile(resolve_profile_name("deep_v2"))
        assert profile.max_hops == 3
        assert profile.max_results == 7

    def test_general_resolves_to_general_profile(self):
        profile = default_search_policy().get_profile(resolve_profile_name("general"))
        assert profile.max_hops == 1
        assert profile.max_results == 3
