# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M70: Schema Claims Tests

Tests for the core bug fix: time_reference vs location distinction.
"""

from spectrue_core.schema import (
    ClaimUnit,
    Assertion,
    Dimension,
    ClaimType,
    EventQualifiers,
    LocationQualifier,
)


class TestClaimUnitSchema:
    """Tests for ClaimUnit schema structure."""

    def test_simple_fact_claim(self):
        """Test simple claim with one FACT assertion."""
        claim = ClaimUnit(
            id="c1",
            claim_type=ClaimType.DEFINITION,
            text="Water boils at 100°C",
            assertions=[
                Assertion(key="physical.boiling_point", value=100, dimension=Dimension.FACT)
            ],
        )
        
        assert claim.id == "c1"
        assert len(claim.assertions) == 1
        assert len(claim.get_fact_assertions()) == 1
        assert len(claim.get_context_assertions()) == 0

    def test_event_claim_with_time_and_location(self):
        """Test event claim with explicit time and location assertions."""
        claim = ClaimUnit(
            id="c2",
            claim_type=ClaimType.EVENT,
            subject="Joshua",
            predicate="fights",
            object="Paul",
            assertions=[
                Assertion(key="event.time", value="03:00", dimension=Dimension.FACT),
                Assertion(key="event.location.city", value="Miami", dimension=Dimension.FACT),
            ],
        )
        
        assert len(claim.get_fact_assertions()) == 2
        assert claim.subject == "Joshua"
        assert claim.object == "Paul"


class TestTimeReferenceDistinction:
    """
    T13: Bug Reproduction Test - Time Reference.
    
    CRITICAL: "(в Україні)" after time is CONTEXT, not location!
    """

    def test_time_reference_is_context_not_location(self):
        """
        The core bug fix: time_reference is CONTEXT, location is FACT.
        
        Input: "03:00 (за Києвом)" and "(в Україні)"
        Expected:
          - time_reference: "Ukraine time" → CONTEXT
          - location: should NOT be set from time_reference
        """
        claim = ClaimUnit(
            id="c1",
            claim_type=ClaimType.EVENT,
            subject="Joshua vs Paul",
            assertions=[
                # Time is FACT
                Assertion(key="event.time", value="03:00", dimension=Dimension.FACT),
                # Time reference is CONTEXT (THE BUG FIX)
                Assertion(
                    key="event.time_reference",
                    value="Kyiv time",
                    value_raw="(за Києвом)",
                    dimension=Dimension.CONTEXT,  # CRITICAL: This is CONTEXT!
                ),
                # Location is FACT (separate from time_reference)
                Assertion(
                    key="event.location.city",
                    value="Miami",
                    dimension=Dimension.FACT,
                ),
            ],
        )
        
        # Assertions
        fact_assertions = claim.get_fact_assertions()
        context_assertions = claim.get_context_assertions()
        
        # Time reference should be CONTEXT
        time_ref = next((a for a in claim.assertions if a.key == "event.time_reference"), None)
        assert time_ref is not None, "time_reference assertion should exist"
        assert time_ref.dimension == Dimension.CONTEXT, "time_reference MUST be CONTEXT"
        
        # Location should be FACT and separate
        location = next((a for a in claim.assertions if a.key == "event.location.city"), None)
        assert location is not None, "location assertion should exist"
        assert location.dimension == Dimension.FACT, "location MUST be FACT"
        assert location.value == "Miami", "location should be Miami, NOT Ukraine"
        
        # FACT assertions should NOT include time_reference
        fact_keys = [a.key for a in fact_assertions]
        assert "event.time_reference" not in fact_keys, "time_reference should NOT be in FACT assertions"
        
        # Counts
        assert len(fact_assertions) == 2, "Should have 2 FACT assertions (time, location)"
        assert len(context_assertions) == 1, "Should have 1 CONTEXT assertion (time_reference)"

    def test_location_not_inferred_from_timezone(self):
        """
        Location should NOT be inferred from timezone/time_reference.
        
        If article says "03:00 (Ukraine time)" but event is in Miami,
        location should be Miami, not Ukraine.
        """
        # Correct: time_reference separate from location
        qualifiers = EventQualifiers(
            time_reference="Ukraine time",  # CONTEXT
            location=LocationQualifier(
                city="Miami",
                country="USA",
                is_inferred=False,
            ),  # FACT
        )
        
        assert qualifiers.time_reference == "Ukraine time"
        assert qualifiers.location is not None
        assert qualifiers.location.city == "Miami"
        assert qualifiers.location.country == "USA"
        
        # These should be completely separate concerns
        assert qualifiers.time_reference != qualifiers.location.country

    def test_claim_has_location_check(self):
        """Test has_location() method."""
        # Claim with location
        with_location = ClaimUnit(
            id="c1",
            claim_type=ClaimType.EVENT,
            qualifiers=EventQualifiers(
                location=LocationQualifier(city="Miami"),
            ),
            assertions=[],
        )
        assert with_location.has_location() is True
        
        # Claim without location (only time_reference)
        only_time_ref = ClaimUnit(
            id="c2",
            claim_type=ClaimType.EVENT,
            qualifiers=EventQualifiers(
                time_reference="Ukraine time",
                location=None,  # No location!
            ),
            assertions=[],
        )
        assert only_time_ref.has_location() is False


class TestDimensionClassification:
    """Tests for dimension (FACT/CONTEXT/INTERPRETATION) handling."""

    def test_dimension_enum_values(self):
        """Test dimension enum has expected values."""
        assert Dimension.FACT.value == "FACT"
        assert Dimension.CONTEXT.value == "CONTEXT"
        assert Dimension.INTERPRETATION.value == "INTERPRETATION"

    def test_assertion_default_dimension_is_fact(self):
        """Default dimension should be FACT."""
        assertion = Assertion(key="test", value="test")
        assert assertion.dimension == Dimension.FACT

    def test_get_fact_assertions_filters_correctly(self):
        """get_fact_assertions() should only return FACT dimension."""
        claim = ClaimUnit(
            id="c1",
            assertions=[
                Assertion(key="a", value=1, dimension=Dimension.FACT),
                Assertion(key="b", value=2, dimension=Dimension.CONTEXT),
                Assertion(key="c", value=3, dimension=Dimension.FACT),
                Assertion(key="d", value=4, dimension=Dimension.INTERPRETATION),
            ],
        )
        
        fact_keys = [a.key for a in claim.get_fact_assertions()]
        assert fact_keys == ["a", "c"]
        
        context_keys = [a.key for a in claim.get_context_assertions()]
        assert context_keys == ["b"]
