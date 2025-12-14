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
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

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

    base = (
        TRUSTED_SOURCES["global_news_agencies"]
        + TRUSTED_SOURCES["fact_checking_ifcn"]
        + TRUSTED_SOURCES["science_and_health"]
        + TRUSTED_SOURCES["technology"]
        + TRUSTED_SOURCES["international_public_bodies"]
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


def get_domains_for_topic(keywords: list[str]) -> list[str]:
    """
    Get relevant trusted domains based on detected topic keywords.
    
    Args:
        keywords: List of keywords from the claim
    
    Returns:
        List of domain strings relevant to the topic
    """
    topic_keywords = {
        "technology": [
            "apple", "google", "microsoft", "ai", "software", "app", "tech",
            "cyber", "hack", "computer", "phone", "iphone", "android", "tesla",
            "openai", "chatgpt", "nvidia", "meta",
        ],
        "science_and_health": [
            "vaccine", "virus", "covid", "health", "doctor", "study",
            "research", "cancer", "drug", "fda", "cdc", "nih", "who", "ipcc",
            "climate", "science", "medical", "hospital", "disease", "space",
        ],
        "finance_tier1": [
            "stock", "stocks", "market", "markets", "shares", "equity", "equities",
            "earnings", "revenue", "profit", "dividend", "bond", "bonds", "yield", "yields",
            "rates", "interest", "inflation", "gdp", "imf", "oecd", "world bank", "worldbank",
            "forex", "fx", "crypto", "bitcoin", "ethereum", "bank", "banking",
        ],
        "education_tier1": [
            "school", "schools", "university", "universities", "college", "student", "students",
            "exam", "exams", "curriculum", "teacher", "teachers", "higher ed", "higher education",
            "campus", "admissions", "scholarship",
        ],
        "law_tier1": [
            "court", "courts", "judge", "judges", "lawsuit", "sued", "legal", "legislation",
            "bill", "statute", "regulation", "attorney", "lawyer", "prosecutor", "supreme court",
            "constitutional", "scotus", "criminal", "trial", "sentenced",
        ],
        "ukraine_imi_whitelist": [
            "ukraine", "kyiv", "zelensky", "war", "crimea",
            "donbas", "kharkiv", "odesa", "ukrainian", "україна",
        ],
        "russia_independent_exiled": [
            "russia", "putin", "moscow", "kremlin", "russian",
            "navalny", "росія", "путін",
        ],
    }
    
    keywords_lower = [k.lower() for k in keywords]
    matched_categories = set()
    
    for category, triggers in topic_keywords.items():
        for trigger in triggers:
            if any(trigger in kw for kw in keywords_lower):
                matched_categories.add(category)
                break
    
    if not matched_categories:
        return []

    result: list[str] = []
    for cat in matched_categories:
        # Topic-gated: only include specialized domains if the topic bucket matched.
        result.extend(TRUSTED_SOURCES.get(cat, []))

    # Deduplicate while preserving order.
    seen = set()
    out: list[str] = []
    for d in result:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out
