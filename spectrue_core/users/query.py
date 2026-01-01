# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from datetime import datetime, timedelta
from typing import Iterator
from spectrue_core.auth.models import User

def get_active_users(db, days_threshold: int = 7) -> Iterator[User]:
    cutoff = datetime.utcnow() - timedelta(days=days_threshold)
    users_ref = db.collection("users")
    # Note: Requires composite index on last_seen_at
    query = users_ref.where("last_seen_at", ">=", cutoff)

    for doc in query.stream():
        yield User.from_dict(doc.to_dict())

def count_active_users(db, days_threshold: int = 7) -> int:
    cutoff = datetime.utcnow() - timedelta(days=days_threshold)
    users_ref = db.collection("users")
    query = users_ref.where("last_seen_at", ">=", cutoff).count()
    # aggregate_query returns list of AggregationResult
    results = query.get()
    return int(results[0][0].value)
