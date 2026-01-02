# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.verification.search.trusted_sources import TRUSTED_SOURCES
from urllib.parse import urlparse

def enrich_sources_with_trust(sources_list: list) -> list:
    """Enrich sources with trust indicators based on TRUSTED_SOURCES registry."""

    domain_to_category: dict[str, str] = {}
    for category, domains in TRUSTED_SOURCES.items():
        for domain in domains:
            domain_to_category[domain.lower()] = category

    enriched = []
    for source in sources_list:
        source_copy = dict(source)

        url = source.get("link") or source.get("url") or ""
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]

            category = domain_to_category.get(host)
            if not category:
                # Try parent domain matching
                parts = host.split(".")
                for i in range(len(parts) - 1):
                    parent = ".".join(parts[i:])
                    if parent in domain_to_category:
                        category = domain_to_category[parent]
                        break

            # T7: Preserve is_trusted=True if already set (e.g., primary inline sources)
            # Only apply registry-based trust if not already trusted
            if not source.get("is_trusted"):
                source_copy["is_trusted"] = category is not None
            source_copy["trust_category"] = category

        except Exception:
            # T7: Only override to False if not already trusted
            if not source.get("is_trusted"):
                source_copy["is_trusted"] = False
            source_copy["trust_category"] = None

        enriched.append(source_copy)

    return enriched
