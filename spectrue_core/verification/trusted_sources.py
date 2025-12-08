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
- europe_tier1: Quality European national media
- asia_pacific: Reliable Asia-Pacific sources
- science_and_health: Scientific journals and health authorities
- technology: Tech news with editorial standards
- fact_checking_ifcn: IFCN-certified fact-checkers
- russia_independent_exiled: Independent Russian media (exiled, for counter-propaganda)
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
        "washingtonpost.com", "nytimes.com", "csmonitor.com"
    ],
    "ukraine_imi_whitelist": [
        "suspilne.media", "radiosvoboda.org", "pravda.com.ua", "babel.ua",
        "hromadske.ua", "texty.org.ua", "zn.ua", "espreso.tv",
        "slovoidilo.ua", "tyzhden.ua", "hromadske.radio", "ukrinform.ua",
        "interfax.com.ua", "gwaramedia.com", "lb.ua", "liga.net"
    ],
    "europe_tier1": [
        "tagesschau.de", "sueddeutsche.de", "faz.net", "spiegel.de",
        "zeit.de", "lemonde.fr", "lefigaro.fr", "elpais.com",
        "elmundo.es", "corriere.it", "repubblica.it", "ansa.it",
        "svt.se", "nrk.no", "yle.fi", "dr.dk", "swissinfo.ch"
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
        # Popular science & space
        "space.com", "livescience.com", "sciencedaily.com", "sciencealert.com",
        "popsci.com", "discovermagazine.com", "smithsonianmag.com", "nationalgeographic.com"
    ],
    "technology": [
        "arstechnica.com", "wired.com", "theverge.com", "techcrunch.com",
        "venturebeat.com", "recode.net", "engadget.com",
        "bleepingcomputer.com", "theregister.com", "zdnet.com"
    ],
    "fact_checking_ifcn": [
        "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
        "stopfake.org", "voxukraine.org", "maldita.es", "newtral.es",
        "pagellapolitica.it", "correctiv.org", "africacheck.org",
        "chequeado.com", "tfc-taiwan.org.tw", "mygopen.com",
        "factchecklab.org", "sciencefeedback.co"
    ],
    "russia_independent_exiled": [
        "meduza.io", "theins.ru", "zona.media", "ovd.info",
        "novayagazeta.eu", "thebell.io", "proekt.media",
        "istories.media", "holod.media"
    ]
}

# Helper to flatten all domains for broad searches
ALL_TRUSTED_DOMAINS: list[str] = [
    domain 
    for category in TRUSTED_SOURCES.values() 
    for domain in category
]


def get_trusted_domains_by_lang(lang: str) -> list[str]:
    """
    Returns a prioritized list of domains based on language.
    
    Args:
        lang: ISO language code (e.g., 'uk', 'en', 'de', 'ru')
    
    Returns:
        List of domain strings prioritized for that language
    """
    # Universal knowledge (Science & Tech) - relevant for all languages as they are primary sources
    universal = TRUSTED_SOURCES["science_and_health"] + TRUSTED_SOURCES["technology"]
    
    base = TRUSTED_SOURCES["global_news_agencies"] + TRUSTED_SOURCES["fact_checking_ifcn"] + universal
    
    if lang == "uk":
        # Ukrainian users: Ukrainian media first, then base, then Western
        return (
            TRUSTED_SOURCES["ukraine_imi_whitelist"] + 
            base + 
            TRUSTED_SOURCES["general_news_western"]
        )
    elif lang == "ru":
        # Russian users: Independent exiled media first (counter-propaganda)
        return (
            TRUSTED_SOURCES["russia_independent_exiled"] + 
            base + 
            TRUSTED_SOURCES["general_news_western"]
        )
    elif lang == "de":
        # German users: European Tier 1 first
        return TRUSTED_SOURCES["europe_tier1"] + base
    elif lang in ("fr", "es", "it"):
        # Other European languages
        return TRUSTED_SOURCES["europe_tier1"] + base + TRUSTED_SOURCES["general_news_western"]
    elif lang in ("ja", "zh", "ko"):
        # Asian languages
        return TRUSTED_SOURCES["asia_pacific"] + base
    else:
        # Default mix for English/others
        return (
            base + 
            TRUSTED_SOURCES["general_news_western"]
        )


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
            "openai", "chatgpt", "nvidia", "meta", "facebook"
        ],
        "science_and_health": [
            "vaccine", "virus", "covid", "health", "doctor", "study",
            "research", "cancer", "drug", "fda", "cdc", "nasa", "space",
            "climate", "science", "medical", "hospital", "disease"
        ],
        "ukraine_imi_whitelist": [
            "ukraine", "kyiv", "zelensky", "war", "crimea",
            "donbas", "kharkiv", "odesa", "ukrainian", "україна"
        ],
        "russia_independent_exiled": [
            "russia", "putin", "moscow", "kremlin", "russian",
            "navalny", "росія", "путін"
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
    
    result = []
    for cat in matched_categories:
        result.extend(TRUSTED_SOURCES.get(cat, []))
    
    return result
