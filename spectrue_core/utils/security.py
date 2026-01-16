# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import hashlib
import re

def hash_pii(text: str) -> str:
    """Hash sensitive data (PII) for logging/storage."""
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent prompt injection and control character issues.
    - Removes control characters.
    - Neutralizes attempts to break XML tagging structure.
    """
    if not text:
        return ""
    # Remove control chars (0-31) except tab (9), newline (10), carriage return (13)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Neutralize XML closing tags if used in prompts
    # Assuming we use <statement>, <context> tags
    text = text.replace("</statement>", "< /statement>")
    text = text.replace("</context>", "< /context>")

    return text.strip()


def redact_log_data(data: dict) -> dict:
    """
    Redact PII from a dictionary for logging.
    """
    if not isinstance(data, dict):
        return data

    safe_data = data.copy()
    sensitive_keys = {"email", "phone", "token", "authorization", "password", "secret", "key"}

    for key in list(safe_data.keys()):
        if str(key).lower() in sensitive_keys:
            safe_data[key] = "[REDACTED]"

    return safe_data