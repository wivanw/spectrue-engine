"""
Factory functions for execution plan phases.
"""
from __future__ import annotations

from spectrue_core.domain.evidence.model import EvidenceChannel
from spectrue_core.domain.verification.search.types import SearchDepth
from .models import Phase


def phase_a(locale: str) -> Phase:
    """
    Phase A: Primary search, high quality, basic depth.
    """
    return Phase(
        phase_id="A",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ],
        search_depth=SearchDepth.BASIC.value,
        max_results=3,
        is_expensive=False,
        description=f"Primary search: {locale}, high-quality, k=3",
    )


def phase_a_light(locale: str) -> Phase:
    """
    Phase A-light: Cheap, fast search for low-priority claims.
    """
    return Phase(
        phase_id="A-light",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
        ],
        search_depth=SearchDepth.BASIC.value,
        max_results=2,
        is_expensive=False,
        description=f"Fail-open minimal: {locale}, authoritative only, k=2",
    )


def phase_a_origin(locale: str) -> Phase:
    """
    Phase A-origin: Origin-focused search for attribution claims.
    """
    return Phase(
        phase_id="A-origin",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ],
        search_depth=SearchDepth.BASIC.value,
        max_results=3,
        is_expensive=False,
        description=f"Origin search: {locale}, finding original source",
    )


def phase_b(locale: str) -> Phase:
    """
    Phase B: Primary locale, expanded channels, advanced depth.
    """
    return Phase(
        phase_id="B",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
            EvidenceChannel.LOCAL_MEDIA,
        ],
        search_depth=SearchDepth.ADVANCED.value,
        max_results=5,
        is_expensive=True,
        description=f"Expanded search: {locale}, +local media, k=5",
    )


def phase_c(locale: str) -> Phase:
    """
    Phase C: Fallback locale, high-quality channels.
    """
    return Phase(
        phase_id="C",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
        ],
        search_depth=SearchDepth.BASIC.value,
        max_results=3,
        is_expensive=False,
        description=f"Fallback locale search: {locale}, k=3",
    )


def phase_d(locale: str = "en") -> Phase:
    """
    Phase D: Last resort, all channels, maximum depth.
    """
    return Phase(
        phase_id="D",
        locale=locale,
        channels=[
            EvidenceChannel.AUTHORITATIVE,
            EvidenceChannel.REPUTABLE_NEWS,
            EvidenceChannel.LOCAL_MEDIA,
            EvidenceChannel.SOCIAL,
            EvidenceChannel.LOW_RELIABILITY,
        ],
        search_depth=SearchDepth.ADVANCED.value,
        max_results=7,
        is_expensive=True,
        description=f"Last resort: {locale}, all channels, k=7",
    )
