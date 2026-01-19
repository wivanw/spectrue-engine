# Copyright (C) 2025 Ivan Bondarenko
#
from spectrue_core.schema.claim_metadata import EvidenceChannel


# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Tier 1 Trusted Sources Registry (v2)
=====================================
Comprehensive registry of high-quality, vetted domains based on:
- IFCN (International Fact-Checking Network) standards
- IMI (Institute of Mass Information, Ukraine) whitelist
- Global reliability ratings and editorial standards

Categories:
- global_news_agencies: Wire services with strict editorial standards
- general_news_western: Major Western broadcasters and newspapers
- ukraine_imi_whitelist: Ukrainian independent media (IMI validated)
- general_news_ukraine_broad: Ukrainian mainstream (broad, audited)
- europe_tier1: Quality European national media
- asia_pacific: Reliable Asia-Pacific sources
- science_and_health: Scientific journals and health authorities
- technology: Tech news with editorial standards
- fact_checking_ifcn: IFCN-certified fact-checkers
- russia_independent_exiled: Independent Russian media (exiled, for counter-propaganda)
- international_public_bodies: International public bodies (primary statements/data)
- finance_tier1: Finance / markets (topic-gated)
- education_tier1: Education / higher-ed (topic-gated)
- law_tier1: Legal / courts / legislation (topic-gated)
"""

TRUSTED_SOURCES: dict[str, list[str]] = {
    "global_news_agencies": [
        "reuters.com", "apnews.com", "afp.com", "bloomberg.com",
        "dpa-international.com", "efe.com"
    ],
    "general_news_western": [
        "bbc.com", "bbc.co.uk", "npr.org", "pbs.org", "dw.com",
        "euronews.com", "france24.com", "rfi.fr", "guardian.co.uk",
        "theguardian.com", "wsj.com", "ft.com", "economist.com",
        "cnbc.com", "usatoday.com", "axios.com", "politico.com",
        "washingtonpost.com", "nytimes.com", "csmonitor.com",
        "cbc.ca", "aljazeera.com", "latimes.com", "time.com", "theatlantic.com"
    ],
    "ukraine_imi_whitelist": [
        "suspilne.media", "radiosvoboda.org", "pravda.com.ua", "babel.ua",
        "hromadske.ua", "texty.org.ua", "zn.ua", "espreso.tv",
        "slovoidilo.ua", "tyzhden.ua", "hromadske.radio", "ukrinform.ua",
        "interfax.com.ua", "gwaramedia.com", "lb.ua", "liga.net"
    ],
    "general_news_ukraine_broad": [
        "nv.ua",
        "unian.ua",
        "rbc.ua",
        "24tv.ua",
        "zaxid.net",
        "glavcom.ua",
    ],
    "europe_tier1": [
        "tagesschau.de", "sueddeutsche.de", "faz.net", "spiegel.de",
        "zeit.de", "lemonde.fr", "lefigaro.fr", "elpais.com",
        "elmundo.es", "corriere.it", "repubblica.it", "ansa.it",
        "svt.se", "nrk.no", "yle.fi", "dr.dk", "swissinfo.ch",
        "nos.nl", "lesoir.be", "derstandard.at", "wyborcza.pl",
        "kathimerini.gr", "lastampa.it", "lavanguardia.com", "ouest-france.fr",
    ],
    "asia_pacific": [
        "channelnewsasia.com", "straitstimes.com", "scmp.com",
        "kyodonews.net", "asahi.com", "japantimes.co.jp", "nikkei.com",
        "mainichi.jp", "abc.net.au", "sbs.com.au", "rnz.co.nz",
        "twreporter.org", "theinitium.com"
    ],
    "science_and_health": [
        "nature.com", "science.org", "scientificamerican.com",
        "sciencenews.org", "newscientist.com", "phys.org",
        "thelancet.com", "nejm.org", "bmj.com", "jama.jamanetwork.com",
        "who.int", "cdc.gov", "nih.gov", "nasa.gov", "esa.int",
        "space.com", "livescience.com", "sciencedaily.com", "sciencealert.com",
        "popsci.com", "discovermagazine.com", "smithsonianmag.com", "nationalgeographic.com",
        "ipcc.ch",
    ],
    # Astronomy & Astrophysics — tiered for evidence scoring
    # Tier A: Professional journals/databases (can raise verified_score)
    "astronomy_tier_a": [
        "arxiv.org",  # Preprints (astro-ph) — ok for "hypothesis exists"
        "ui.adsabs.harvard.edu",  # NASA ADS — gold standard
        "astronomerstelegram.org",  # ATel — transient alerts, critical
        "simbad.u-strasbg.fr",  # SIMBAD database
        "vizier.u-strasbg.fr",  # VizieR catalogs
        "iopscience.iop.org",  # IOP Science (ApJ, MNRAS)
        "aanda.org",  # Astronomy & Astrophysics journal
        "academic.oup.com",  # Oxford (MNRAS)
        # Official space agencies & observatories
        "science.nasa.gov",  # NASA Science
        "hubblesite.org",  # Hubble / STScI
        "stsci.edu",  # Space Telescope Science Institute
        "eso.org",  # European Southern Observatory
        "mpifr-bonn.mpg.de",  # Max Planck Radio Astronomy
        "mpia.de",  # Max Planck Astronomy
        "aas.org",  # American Astronomical Society
        "lbl.gov",  # Lawrence Berkeley National Lab (DESI host)
        "noirlab.edu",  # NSF NOIRLab (Kitt Peak, Gemini)
        "slac.stanford.edu",  # SLAC National Accelerator Lab
        "cern.ch",  # CERN
        "fnal.gov",  # Fermilab
    ],
    # Tier B: Semi-professional/curated (supportive, capped weight)
    "astronomy_tier_b": [
        "aasnova.org",  # AAS Nova digests
        "skyandtelescope.org",  # Respected science journalism
        "earthsky.org",  # Science communication
    ],
    # Tier C: Science-popular (context/plausibility only, not proof)
    "astronomy_tier_c": [
        "universetoday.com",
        "centauri-dreams.org",
        "astronomy.com",
        "astrobites.org",  # Grad student summaries
    ],
    "technology": [
        "arstechnica.com", "wired.com", "theverge.com", "techcrunch.com",
        "venturebeat.com", "recode.net", "engadget.com",
        "bleepingcomputer.com", "theregister.com", "zdnet.com",
        "cnet.com", "technologyreview.com",
    ],
    "fact_checking_ifcn": [
        "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
        "stopfake.org", "voxukraine.org", "maldita.es", "newtral.es",
        "pagellapolitica.it", "correctiv.org", "africacheck.org",
        "chequeado.com", "tfc-taiwan.org.tw", "mygopen.com",
        "factchecklab.org", "sciencefeedback.co",
        "leadstories.com",
    ],
    "russia_independent_exiled": [
        "meduza.io", "theins.ru", "zona.media", "ovd.info",
        "novayagazeta.eu", "thebell.io", "proekt.media",
        "istories.media", "holod.media",
        "tvrain.tv", "agentstvo.media", "verstka.media",
    ],
    "international_public_bodies": [
        "ec.europa.eu",
        "un.org",
        "worldbank.org",
        "oecd.org",
        "imf.org",
    ],
    "finance_tier1": [
        "wsj.com",
        "ft.com",
        "bloomberg.com",
        "economist.com",
        "cnbc.com",
        "fortune.com",
        "marketwatch.com",
        "forbes.com",
    ],
    "education_tier1": [
        "chronicle.com",
        "insidehighered.com",
        "edweek.org",
        "chalkbeat.org",
        "hechingerreport.org",
        "timeshighereducation.com",
        "universityworldnews.com",
        "the74million.org",
    ],
    "law_tier1": [
        "scotusblog.com",
        "jurist.org",
        "lawfaremedia.org",
        "justsecurity.org",
        "themarshallproject.org",
        "abajournal.com",
        "thecrimereport.org",
    ],
    "reference_and_encyclopedic": [
        "wikipedia.org",
    ],
}

# Helper to flatten all domains for broad searches (deduped, order-preserving).
_ALL_TRUSTED_DOMAINS_RAW: list[str] = [
    domain for category in TRUSTED_SOURCES.values() for domain in category
]
_seen = set()
ALL_TRUSTED_DOMAINS: list[str] = []
for _d in _ALL_TRUSTED_DOMAINS_RAW:
    if _d in _seen:
        continue
    _seen.add(_d)
    ALL_TRUSTED_DOMAINS.append(_d)
del _ALL_TRUSTED_DOMAINS_RAW, _seen, _d


def get_trusted_domains_by_lang(lang: str) -> list[str]:
    """
    Returns a prioritized list of domains based on language.
    
    Args:
        lang: ISO language code (e.g., 'uk', 'en', 'de', 'ru')
    
    Returns:
        List of domain strings prioritized for that language
    """
    lang = (lang or "en").lower()
    supported = {
        "en", "uk", "ru", "de", "fr", "es", "it", "pl", "nl", "pt", "sv", "no", "da", "fi",
        "cs", "ro", "hu", "tr", "ar", "he", "ja", "ko", "zh",
    }
    if lang not in supported:
        lang = "en"

    # Removed "international_public_bodies" from base list.
    # IGO sites (un.org, imf.org, oecd.org, worldbank.org, ec.europa.eu) return legal PDFs,
    # historical documents, and archives — NOT breaking news.
    # They remain in TRUSTED_SOURCES for topic-gated searches (e.g., claim mentions UN resolution).
    base = (
        TRUSTED_SOURCES["global_news_agencies"]
        + TRUSTED_SOURCES["fact_checking_ifcn"]
        + TRUSTED_SOURCES["science_and_health"]
        + TRUSTED_SOURCES["technology"]
        + TRUSTED_SOURCES["reference_and_encyclopedic"]
        # + TRUSTED_SOURCES["international_public_bodies"]  # removed for news queries
    )

    # Language-group routing (Spec Kit).
    # TODO: Refactor hardcoded region logic to be dynamic or configuration-based (Technical Debt)
    if lang == "uk":
        return TRUSTED_SOURCES["ukraine_imi_whitelist"] + TRUSTED_SOURCES["general_news_ukraine_broad"] + base
    if lang == "ru":
        return TRUSTED_SOURCES["russia_independent_exiled"] + base

    eu_base_langs = {"de", "nl", "sv", "no", "da", "fi", "pl", "cs", "ro", "hu"}
    if lang in eu_base_langs:
        return TRUSTED_SOURCES["europe_tier1"] + base

    eu_plus_western = {"fr", "es", "it", "pt", "tr"}
    if lang in eu_plus_western:
        return TRUSTED_SOURCES["europe_tier1"] + base + TRUSTED_SOURCES["general_news_western"]

    if lang in {"ja", "zh", "ko"}:
        return TRUSTED_SOURCES["asia_pacific"] + base

    if lang in {"ar", "he"}:
        return base + TRUSTED_SOURCES["general_news_western"]

    # Default / en
    return base + TRUSTED_SOURCES["general_news_western"]


# Backward compatibility alias
def get_domains_for_language(lang: str) -> list[str]:
    """Alias for get_trusted_domains_by_lang (backward compatibility)."""
    return get_trusted_domains_by_lang(lang)


# Academic domains for evidence_need == 'academic'
def get_academic_domains() -> list[str]:
    """
    Returns domains for academic/research claims.
    Used when evidence_need == 'academic' for better source coverage.
    
    Includes:
    - Astronomy Tier A (arxiv, ADS, observatories)
    - Science & Health journals
    - Academic databases (semantic scholar, pubmed, jstor)
    """
    return (
        TRUSTED_SOURCES.get("astronomy_tier_a", []) +
        TRUSTED_SOURCES.get("science_and_health", []) +
        [
            "scholar.google.com",
            "semanticscholar.org",
            "pubmed.ncbi.nlm.nih.gov",
            "jstor.org",
            "researchgate.net",
        ]
    )


# LLM-based topic detection
# Available topics for LLM to choose from during query generation.
# LLM receives this list and selects ALL matching topics for the claim.
AVAILABLE_TOPICS: list[str] = [
    "astronomy",       # Scientific: stars, planets, space, telescopes, cosmic events
    "technology",      # Tech: AI, software, hardware, cyber, companies
    "science_health",  # Health/medicine: vaccines, diseases, research, FDA/CDC/WHO
    "finance",         # Markets, stocks, banking, crypto, IMF/OECD
    "education",       # Schools, universities, students, exams
    "law",             # Courts, lawsuits, legislation, trials
]


# Topic → TRUSTED_SOURCES category mapping
_TOPIC_TO_CATEGORIES: dict[str, list[str]] = {
    "astronomy": ["astronomy_tier_a", "astronomy_tier_b", "astronomy_tier_c"],
    "technology": ["technology"],
    "science_health": ["science_and_health"],
    "finance": ["finance_tier1"],
    "education": ["education_tier1"],
    "law": ["law_tier1"],
}


def get_domains_by_topics(topics: list[str]) -> list[str]:
    """
    Get trusted domains for LLM-selected topics.

    Args:
        topics: List of topic strings selected by LLM (from AVAILABLE_TOPICS)

    Returns:
        Deduplicated list of domain strings for the selected topics
    """
    if not topics:
        return []

    result: list[str] = []
    for topic in topics:
        topic_lower = topic.lower().strip()
        categories = _TOPIC_TO_CATEGORIES.get(topic_lower, [])
        for cat in categories:
            result.extend(TRUSTED_SOURCES.get(cat, []))

    # Deduplicate while preserving order
    seen = set()
    out: list[str] = []
    for d in result:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


# Tier A Definition Constants
TIER_A_TLDS = {".gov", ".mil", ".int", ".edu"}
TIER_A_SUFFIXES = {".europa.eu", ".ac.uk", ".gov.uk", ".gov.ua", ".gov.pl", ".bund.de", ".gc.ca"}

# Social Platforms Registry
SOCIAL_PLATFORMS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", 
    "t.me", "tiktok.com", "youtube.com", "linkedin.com", "threads.net"
}

def get_tier_ceiling_for_domain(domain: str) -> float:
    """
    Returns the maximum probability score (Evidence Strength Cap) allowed for a single source 
    from this domain.
    
    Principles (RGBA):
    - Tier A (0.90): Institutional authority or Primary Scientific Source.
    - Tier B (0.75): Trusted Editorial Media.
    - Tier C (0.55): General Web.
    - Tier D (0.35): Low authority (Social).
    """
    if not domain: 
        return 0.55 # Default to Tier C

    d = domain.lower().strip()

    # 1. Tier A Checks (Institutional)
    # Check TLDs
    for tld in TIER_A_TLDS:
        if d.endswith(tld):
            return 0.90

    # Check Suffixes (Sub-TLDs)
    for suffix in TIER_A_SUFFIXES:
        if d.endswith(suffix):
            return 0.90

    # Check Explicit Lists
    if d in TRUSTED_SOURCES.get("international_public_bodies", []):
        return 0.90
    if d in TRUSTED_SOURCES.get("science_and_health", []):
        return 0.90
    if d in TRUSTED_SOURCES.get("astronomy_tier_a", []):
        return 0.90

    # 2. Tier B Checks (Trusted Media)
    if d in ALL_TRUSTED_DOMAINS:
        return 0.90  # TierPrior B

    # 3. Default (Tier C / D)
    # Check social for Tier D
    if is_social_platform(d):
        return 0.80 # TierPrior D

    # Default Tier C (General Web / Local)
    return 0.85 # TierPrior C


def is_social_platform(domain: str) -> bool:
    """
    Check if a domain is a known social media platform.
    """
    if not domain:
        return False

    d = domain.lower().strip()

    # Direct match or suffix match (e.g. m.facebook.com)
    for p in SOCIAL_PLATFORMS:
        if d == p or d.endswith("." + p):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Tier Detection Logic (Moved from sufficiency.py)
# ─────────────────────────────────────────────────────────────────────────────

def is_authoritative(domain: str) -> bool:
    """
    Check if domain is authoritative (Tier A).
    """
    if not domain:
        return False

    d = domain.lower().strip()

    # Check TLDs
    for tld in TIER_A_TLDS:
        if d.endswith(tld):
            return True

    # Check sub-TLDs
    for suffix in TIER_A_SUFFIXES:
        if d.endswith(suffix):
            return True

    # Check explicit lists
    if d in TRUSTED_SOURCES.get("international_public_bodies", []):
        return True
    if d in TRUSTED_SOURCES.get("science_and_health", []):
        return True
    if d in TRUSTED_SOURCES.get("astronomy_tier_a", []):
        return True
    if d in TRUSTED_SOURCES.get("global_news_agencies", []):
        return True
    if d in TRUSTED_SOURCES.get("fact_checking_ifcn", []):
        return True

    return False


def is_reputable_news(domain: str) -> bool:
    """
    Check if domain is reputable news (Tier B).
    """
    if not domain:
        return False

    d = domain.lower().strip()

    # Check against all trusted domains
    if d in ALL_TRUSTED_DOMAINS:
        return True

    # Check specific categories
    tier_b_categories = [
        "general_news_western",
        "ukraine_imi_whitelist",
        "general_news_ukraine_broad",
        "europe_tier1",
        "asia_pacific",
        "technology",
        "astronomy_tier_b",
        "russia_independent_exiled",
    ]

    for category in tier_b_categories:
        if d in TRUSTED_SOURCES.get(category, []):
            return True

    return False


def is_local_media(domain: str) -> bool:
    """
    Check if domain is local/regional media (Tier C).
    """
    if not domain:
        return False

    d = domain.lower().strip()

    # Not social and not authoritative/reputable
    if is_social_platform(d):
        return False
    if is_authoritative(d) or is_reputable_news(d):
        return False

    # Assume local media if it has news-like TLD patterns
    news_patterns = [".news", ".media", ".tv", ".radio", ".times", ".post", ".herald"]
    for pattern in news_patterns:
        if pattern in d:
            return True

    return False


def get_domain_tier(domain: str) -> EvidenceChannel:
    """
    Determine the evidence channel for a domain.
    """
    if not domain:
        return EvidenceChannel.LOW_RELIABILITY

    if is_authoritative(domain):
        return EvidenceChannel.AUTHORITATIVE

    if is_reputable_news(domain):
        return EvidenceChannel.REPUTABLE_NEWS

    if is_social_platform(domain):
        return EvidenceChannel.SOCIAL

    if is_local_media(domain):
        return EvidenceChannel.LOCAL_MEDIA

    return EvidenceChannel.LOW_RELIABILITY


def get_tier_code(domain: str) -> str:
    """
    Get string tier code (A, B, C, D) for a domain.
    
    Used for RGBA explainability adjustment.
    """
    channel = get_domain_tier(domain)
    match channel:
        case EvidenceChannel.AUTHORITATIVE:
            return "A"
        case EvidenceChannel.REPUTABLE_NEWS:
            return "B"
        case EvidenceChannel.LOCAL_MEDIA:
            return "C"
        case EvidenceChannel.SOCIAL:
            return "D"
        case EvidenceChannel.LOW_RELIABILITY:
            return "D"
        case _:
            return "D"
