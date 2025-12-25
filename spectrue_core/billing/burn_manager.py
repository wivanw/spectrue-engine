from datetime import datetime, timedelta
from firebase_admin import firestore

def burn_inactive_users(db, days_threshold: int = 365) -> int:
    cutoff = datetime.utcnow() - timedelta(days=days_threshold)
    users_ref = db.collection("users")
    
    # Query: last_seen_at < cutoff AND balance_sc > 0
    # Requires composite index.
    query = users_ref.where("last_seen_at", "<", cutoff).where("balance_sc", ">", 0)
    
    batch = db.batch()
    count = 0
    burned_count = 0
    
    for doc in query.stream():
        data = doc.to_dict()
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
