from spectrue_core.verification.trusted_sources import TRUSTED_SOURCES
from spectrue_core.utils.url_utils import get_registrable_domain
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
            
            source_copy["is_trusted"] = category is not None
            source_copy["trust_category"] = category
            
        except Exception:
            source_copy["is_trusted"] = False
            source_copy["trust_category"] = None
        
        enriched.append(source_copy)
    
    return enriched
