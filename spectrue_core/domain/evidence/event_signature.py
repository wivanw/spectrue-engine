from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventSignature:
    """
    Deterministic, low-cardinality signature for claim/evidence routing.
    Built from structured metadata only (no text heuristics).
    """
    entities: tuple[str, ...]
    time_bucket: str
    locale: str


def claim_event_signature(claim: dict[str, Any]) -> EventSignature:
    md = claim.get("metadata") if isinstance(claim.get("metadata"), dict) else {}

    # Entities: prefer subject_entities, else empty
    ents = claim.get("subject_entities")
    if not isinstance(ents, list):
        ents = []
    ents_norm = tuple(str(x).strip()[:48] for x in ents[:5] if x)

    # Time bucket: from time_signals if present
    time_bucket = ""
    ts = md.get("time_signals") if isinstance(md, dict) else {}
    if isinstance(ts, dict):
        time_bucket = str(ts.get("time_bucket") or ts.get("year") or "").strip()[:32]

    # Locale: from locale_signals if present
    locale = ""
    ls = md.get("locale_signals") if isinstance(md, dict) else {}
    if isinstance(ls, dict):
        locale = str(ls.get("country") or ls.get("locale") or "").strip()[:32]

    return EventSignature(entities=ents_norm, time_bucket=time_bucket, locale=locale)


def evidence_event_signature(src: dict[str, Any]) -> EventSignature | None:
    """
    Evidence signature, if already present on the evidence item.
    We keep it optional: v5 can function without it.
    """
    es = src.get("event_signature")
    if not isinstance(es, dict):
        return None
    ents = es.get("entities") or []
    tb = es.get("time_bucket") or ""
    loc = es.get("locale") or ""
    if not isinstance(ents, list):
        ents = []
    return EventSignature(
        entities=tuple(str(x).strip()[:48] for x in ents[:5] if x),
        time_bucket=str(tb).strip()[:32],
        locale=str(loc).strip()[:32],
    )


def signature_compatible(a: EventSignature, b: EventSignature | None) -> bool:
    """
    Deterministic compatibility:
    - if evidence signature missing -> do not block
    - else require some overlap on at least one axis
    """
    if b is None:
        return True

    ent_overlap = bool(set(a.entities) & set(b.entities)) if a.entities and b.entities else False
    time_ok = (a.time_bucket and b.time_bucket and a.time_bucket == b.time_bucket) or (not a.time_bucket or not b.time_bucket)
    loc_ok = (a.locale and b.locale and a.locale == b.locale) or (not a.locale or not b.locale)

    # Require at least one strong overlap (entities) OR both time+locale consistent.
    if ent_overlap:
        return True
    if time_ok and loc_ok:
        return True
    return False
