"""Backward-compatible exports for deduplication helpers."""

from spectrue_core.domain.evidence.deduplication import (  # noqa: F401
    evidence_text_payload,
    normalize_publisher,
    normalize_text_for_hash,
    sha256_hex,
    simhash64,
    simhash_bucket_id,
)

__all__ = [
    "evidence_text_payload",
    "normalize_publisher",
    "normalize_text_for_hash",
    "sha256_hex",
    "simhash64",
    "simhash_bucket_id",
]
