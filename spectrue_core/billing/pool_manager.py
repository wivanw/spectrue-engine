from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Optional
from spectrue_core.billing.models import FreeSubsidyPool

class FreePoolAllocator:
    LOCK_PERIOD_DAYS = 90

    @staticmethod
    def deposit(
        pool: FreeSubsidyPool, 
        amount: Decimal, 
        lock_ratio: Decimal, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Allocates a deposit into available and locked portions.
        Updates the pool state in-place.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        locked_amount = amount * lock_ratio
        available_amount = amount - locked_amount
        
        # Add to available
        pool.available_balance_sc += available_amount
        
        if locked_amount > 0:
            # Determine release date (e.g. now + 90 days)
            release_date = timestamp + timedelta(days=FreePoolAllocator.LOCK_PERIOD_DAYS)
            release_key = release_date.strftime("%Y-%m-%d")
            
            current_locked = pool.locked_buckets.get(release_key, Decimal("0.0"))
            pool.locked_buckets[release_key] = current_locked + locked_amount
        
        pool.last_updated = timestamp

    @staticmethod
    def release_matured_funds(pool: FreeSubsidyPool, as_of_date: datetime) -> Decimal:
        """
        Moves funds from matured buckets to available balance.
        Returns the total amount released.
        """
        total_released = Decimal("0.0")
        keys_to_remove: List[str] = []
        
        for date_str, amount in pool.locked_buckets.items():
            try:
                bucket_date = datetime.strptime(date_str, "%Y-%m-%d")
                # If bucket date is today or in the past
                if bucket_date.date() <= as_of_date.date():
                    total_released += amount
                    keys_to_remove.append(date_str)
            except ValueError:
                # Handle or log invalid date format if necessary
                continue
        
        for k in keys_to_remove:
            del pool.locked_buckets[k]
            
        pool.available_balance_sc += total_released
        # We don't necessarily update last_updated here as it might be part of a larger transaction
        # but usually consistent with the operation time.
        return total_released

    @staticmethod
    def deduct(pool: FreeSubsidyPool, amount: Decimal) -> bool:
        """
        Deducts amount from available balance if sufficient.
        Returns True if successful, False if insufficient funds.
        """
        if pool.available_balance_sc >= amount:
            pool.available_balance_sc -= amount
            return True
        return False
