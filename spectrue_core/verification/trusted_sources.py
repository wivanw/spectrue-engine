# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# Constants for Google CSE (loaded from env usually, but defined here for legacy compat if they were here)
# Upon review, these were NOT in trusted_sources.py, they were likely imported from there by mistake in my previous turn.
# I will NOT add them here. I will fix verifier.py instead.


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
    # M44: Astronomy & Astrophysics — tiered for evidence scoring
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

    # M61: Removed "international_public_bodies" from base list.
    # IGO sites (un.org, imf.org, oecd.org, worldbank.org, ec.europa.eu) return legal PDFs,
    # historical documents, and archives — NOT breaking news.
    # They remain in TRUSTED_SOURCES for topic-gated searches (e.g., claim mentions UN resolution).
    base = (
        TRUSTED_SOURCES["global_news_agencies"]
        + TRUSTED_SOURCES["fact_checking_ifcn"]
        + TRUSTED_SOURCES["science_and_health"]
        + TRUSTED_SOURCES["technology"]
        # + TRUSTED_SOURCES["international_public_bodies"]  # M61: removed for news queries
    )

    # Language-group routing (Spec Kit).
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


# M45: LLM-based topic detection
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

