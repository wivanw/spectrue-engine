# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Словарь для хранения всех загруженных переводов
_PROMPTS = {}

# Use path relative to this file
AGENTS_ROOT = Path(__file__).resolve().parent
LOCALES_PATH = AGENTS_ROOT / 'locales'


def load_prompts(locales_dir: Path = None):
    """
    Загружает все yml-файлы из папки locales.
    
    Args:
        locales_dir: Optional custom path to locales directory. 
                     Defaults to 'locales' next to this file.
    """
    global _PROMPTS, LOCALES_PATH

    if locales_dir:
        LOCALES_PATH = Path(locales_dir)

    if not LOCALES_PATH.is_dir():
        # Only print error if it's the default path. If custom path fails, user should handle it.
        if locales_dir is None:
             logger.warning("Locales directory not found at %s", LOCALES_PATH)
        return

    for f in LOCALES_PATH.glob("*.yml"):
        lang = f.stem
        with open(f, "r", encoding="utf-8") as yf:
            try:
                data = yaml.safe_load(yf)
                if data and isinstance(data, dict) and lang in data:
                    _PROMPTS[lang] = data[lang]
            except yaml.YAMLError as e:
                logger.warning("YAML parsing failed for %s: %s", f.name, e)

    logger.debug("Loaded prompts for languages: %s", list(_PROMPTS.keys()))


def get_prompt(lang: str, key: str) -> str:
    """Безопасно получает промпт по ключу, с фолбэком на английский."""

    # Auto-load if empty (lazy loading)
    if not _PROMPTS:
        load_prompts()

    key_parts = key.split('.')

    # Сначала пытаемся найти на нужном языке
    lang_data = _PROMPTS.get(lang, _PROMPTS.get("en", {}))

    # Ищем вложенный ключ
    prompt_template = lang_data
    for part in key_parts:
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template.get(part)
        else:
            prompt_template = None
            break

    # Если не нашли, пытаемся найти в английском
    if prompt_template is None and lang != "en":
        en_data = _PROMPTS.get("en", {})
        prompt_template = en_data
        for part in key_parts:
            if isinstance(prompt_template, dict):
                prompt_template = prompt_template.get(part)
            else:
                prompt_template = None
                break

    return prompt_template or f"Prompt key '{key}' not found."