# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Inactivity burn manager for free tier credits.

Burn Logic:
- Users with ACTIVE paid plans are NEVER burned (even if inactive)
- Users without plan: burn after `days_threshold` days of inactivity
- Users with EXPIRED plan: burn after `days_threshold` days AFTER plan expiry

This protects purchased credits from unfair burn.
"""
from datetime import datetime, timedelta
from firebase_admin import firestore


def burn_inactive_users(db, days_threshold: int = 365) -> int:
    """Burn credits from inactive users.
    
    Args:
        db: Firestore client
        days_threshold: Days of inactivity before burn (default: 365)
        
    Returns:
        Number of users burned
        
    Note:
        Users with active paid plans are NEVER burned.
        For users with expired plans, the threshold starts from plan_expires_at.
    """
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days_threshold)
    users_ref = db.collection("users")

    # Query: last_seen_at < cutoff AND balance_sc > 0
    # We filter active plans in code (Firestore doesn't support OR well)
    query = users_ref.where("last_seen_at", "<", cutoff).where("balance_sc", ">", 0)

    batch = db.batch()
    count = 0
    burned_count = 0

    for doc in query.stream():
        data = doc.to_dict()

        # --- SKIP: Users with active paid plans ---
        plan_expires_at = data.get("plan_expires_at")
        if plan_expires_at:
            # Convert to datetime if needed
            if hasattr(plan_expires_at, 'timestamp'):
                plan_expires_at = plan_expires_at

            # If plan is still active, NEVER burn
            if plan_expires_at > now:
                continue

            # Plan expired - check if threshold passed since expiry
            burn_cutoff_for_expired = plan_expires_at + timedelta(days=days_threshold)
            if now < burn_cutoff_for_expired:
                continue  # Not enough time passed since plan expiry

        # --- BURN: Free user or plan expired + threshold passed ---
        balance = data.get("balance_sc", 0)

        # Burn
        batch.update(doc.reference, {
            "balance_sc": 0,
            "burn_reason": "inactivity_365d",
            "burned_amount": balance,
            "burned_at": firestore.SERVER_TIMESTAMP
        })

        count += 1
        burned_count += 1

        if count % 400 == 0:
            batch.commit()
            batch = db.batch()

    if count % 400 != 0:
        batch.commit()

    return burned_count
