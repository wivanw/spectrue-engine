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

def get_language_name(code: str) -> str:
    """Return the full English name for a language code, defaulting to English."""
    return SUPPORTED_LANGUAGES.get((code or "").lower(), "English")
