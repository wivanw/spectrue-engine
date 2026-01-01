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
