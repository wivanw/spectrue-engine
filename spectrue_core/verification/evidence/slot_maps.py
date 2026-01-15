from __future__ import annotations

from typing import Iterable


# Slots we use for compatibility and evidence routing.
SLOTS = {
    "entity",
    "time",
    "location",
    "quantity",
    "attribution",
    "causal",
    "other",
}


def slots_from_assertion_key(assertion_key: str) -> set[str]:
    """
    Deterministic mapping from structured assertion_key namespace to slots.
    This is NOT text heuristics on claim prose; it's mapping on schema keys.
    """
    k = (assertion_key or "").lower()
    if not k:
        return set()

    out: set[str] = set()

    # location
    if ".location" in k or k.endswith(".city") or k.endswith(".country") or k.endswith(".region"):
        out.add("location")

    # time
    if ".time" in k or "date" in k or "year" in k or "month" in k or "timeline" in k:
        out.add("time")

    # quantity / numeric
    if "count" in k or "number" in k or "percent" in k or "rate" in k or "total" in k or "amount" in k:
        out.add("quantity")

    # attribution
    if "attribution" in k or "said" in k or "statement" in k or "quote" in k or "spokesperson" in k:
        out.add("attribution")

    # causal
    if "cause" in k or "because" in k or "led_to" in k or "result" in k or "trigger" in k:
        out.add("causal")

    # entity (default for most event.* keys)
    if k.startswith("event.") or k.startswith("person.") or k.startswith("org.") or k.startswith("entity."):
        out.add("entity")

    if not out:
        out.add("other")
    return out


def required_slots_for_verification_target(verification_target: str) -> set[str]:
    """
    Deterministic requirement mapping from claim VerificationTarget to slots.
    """
    vt = (verification_target or "").lower().strip()
    if not vt:
        return set()

    # Schema enum values in spectrue_core.schema.claim_metadata.VerificationTarget
    if vt == "attribution":
        return {"entity", "attribution"}
    if vt == "existence":
        return {"entity"}
    if vt == "reality":
        # general reality claims may be about anything; don't overconstrain
        return set()
    if vt == "none":
        return set()
    return set()


def merge_covers(existing: Iterable[str] | None, derived: set[str]) -> set[str]:
    out = set()
    if existing:
        for x in existing:
            s = str(x).strip().lower()
            if s in SLOTS:
                out.add(s)
    out |= {x for x in derived if x in SLOTS}
    return out
