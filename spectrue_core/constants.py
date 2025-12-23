from typing import Dict

# Map of ISO 639-1 language codes to their full English names.
# This is used for LLM prompting to specify target languages.
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "uk": "Ukrainian",
    "ru": "Russian",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese",
}

# Default language to use if the requested language is not found.
DEFAULT_LANGUAGE: str = "en"

# Temporal defaults for time-window normalization.
DEFAULT_TIME_GRANULARITY_DAYS: Dict[str, int] = {
    "day": 0,
    "week": 7,
    "month": 31,
    "year": 365,
    "range": 0,
    "relative": 30,
}
DEFAULT_RELATIVE_WINDOW_DAYS: int = 30

# Locale defaults for search routing.
DEFAULT_PRIMARY_LOCALE: str = DEFAULT_LANGUAGE
DEFAULT_FALLBACK_LOCALES: list[str] = ["en"]
DEFAULT_LOCALE_MAX_FALLBACKS: int = 2

def get_language_name(code: str) -> str:
    """Return the full English name for a language code, defaulting to English."""
    return SUPPORTED_LANGUAGES.get((code or "").lower(), "English")
