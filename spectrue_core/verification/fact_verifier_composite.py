from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.tools.search_tool import WebSearchTool, TRUSTED_DOMAINS
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.verification.trusted_sources import get_domains_for_language, get_trusted_domains_by_lang
from spectrue_core.config import SpectrueConfig
import asyncio
import os
import re
import math
import logging
from spectrue_core.utils.runtime import is_local_run
from spectrue_core.utils.trace import Trace
from uuid import uuid4
from spectrue_core import __version__ as ENGINE_VERSION, PROMPT_VERSION, SEARCH_STRATEGY_VERSION

logger = logging.getLogger(__name__)

RAG_ONLY_COST = 0
SEARCH_COSTS = {"basic": 80, "advanced": 160}
MODEL_COSTS = {"gpt-5-nano": 5, "gpt-5-mini": 20, "gpt-5.2": 100}


class FactVerifierComposite:
    def __init__(self, config: SpectrueConfig):
        self.config = config
        self.web_search_tool = WebSearchTool(config)
        self.google_tool = GoogleFactCheckTool(config)
        self.google_cse_tool = GoogleCSESearchTool(config)
        self.agent = FactCheckerAgent(config)
        self.time_sensitive_ttl = 300

    def _normalize_search_query(self, query: str) -> str:
        q = (query or "").strip()
        q = re.sub(r"\s+", " ", q).strip()
        q = q.strip("“”„«»\"'`")
        q = q.replace("…", " ").strip()
        q = re.sub(r"\s+", " ", q).strip()
        if len(q) > 256:
            q = q[:256].strip()
        return q

    def _is_mixed_script(self, text: str) -> bool:
        s = text or ""
        has_latin = re.search(r"[A-Za-z]", s) is not None
        has_cyr = re.search(r"[А-Яа-яІіЇїЄєҐґ]", s) is not None
        return has_latin and has_cyr

    def _env_true(self, name: str, default: bool = True) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")

    def _is_time_sensitive(self, fact: str, lang: str) -> bool:
        """Check if fact requires recent data using simple heuristics."""
        time_keywords = [
            "today", "yesterday", "this week", "this month", "breaking",
            "just announced", "latest", "new", "recent", "now",
            "сьогодні", "вчора", "щойно", "новий", "останній",
            "сегодня", "вчера", "только что", "новый", "последний"
        ]
        fact_lower = fact.lower()
        return any(kw in fact_lower for kw in time_keywords)

    def _clamp(self, value) -> float:
        """Clamp value between 0.0 and 1.0."""
        try:
            if value is None:
                return 0.5
            val = float(value)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5

    def _enrich_sources_with_trust(self, sources_list: list) -> list:
        """Enrich sources with trust indicators."""
        from spectrue_core.verification.trusted_sources import TRUSTED_SOURCES
        from urllib.parse import urlparse
        
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

    def _compute_explainability_score(self, analysis_result: dict, sources: list[dict]) -> float:
        """
        Explainability ≠ confidence.
        It reflects how well the result can be explained using sources: quantity, agreement, and transparency.
        """
        from urllib.parse import urlparse

        n = len(sources or [])
        evidence = min(1.0, n / 6.0)

        domains: set[str] = set()
        for s in sources or []:
            url = (s.get("link") or s.get("url") or "").strip()
            if not url:
                continue
            try:
                host = (urlparse(url).netloc or "").lower()
                if host.startswith("www."):
                    host = host[4:]
                if host:
                    domains.add(host)
            except Exception:
                continue
        diversity = min(1.0, len(domains) / 4.0) if domains else 0.0

        # Agreement proxy: relevance distribution (higher + tighter cluster => better topical alignment).
        rels: list[float] = []
        for s in sources or []:
            v = s.get("relevance_score")
            if isinstance(v, (int, float)):
                rels.append(float(v))
        if rels:
            mean = sum(rels) / len(rels)
            var = sum((x - mean) ** 2 for x in rels) / len(rels)
            stdev = math.sqrt(var)
            agreement = max(0.0, min(1.0, (mean * 1.2) * (1.0 - min(0.6, stdev * 1.8))))
        else:
            agreement = 0.55

        rationale = analysis_result.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            transparency = 0.75
            if re.search(r"\b[a-z0-9-]+\.[a-z]{2,}\b", rationale, flags=re.IGNORECASE):
                transparency = 1.0
        else:
            transparency = 0.30

        score = 0.15 + 0.40 * evidence + 0.15 * diversity + 0.20 * agreement + 0.10 * transparency
        return self._clamp(score)

    def _extract_claim_attributes(self, fact: str) -> dict:
        s = (fact or "").strip()
        # Very lightweight heuristics; aim to detect whether the claim is underspecified.
        has_where = False
        has_when = False
        has_by_whom = False

        # Location cues
        if re.search(r"\b(?:in|at|near|from)\s+[A-Z][A-Za-z-]{2,}\b", s):
            has_where = True
        if re.search(r"\b(?:в|у|на)\s+[А-ЯІЇЄҐ][А-Яа-яІіЇїЄєҐґ’'\\-]{2,}", s):
            has_where = True

        # Date/time cues
        if re.search(r"\b20\\d{2}\b", s) or re.search(r"\b\\d{1,2}[./-]\\d{1,2}[./-]\\d{2,4}\b", s):
            has_when = True
        if re.search(r"\b(?:today|yesterday|this week|this month|recently|just)\b", s, flags=re.IGNORECASE):
            has_when = True
        if re.search(r"\b(?:сьогодні|вчора|цього\\s+тижня|цього\\s+місяця|щойно|нещодавно)\b", s, flags=re.IGNORECASE):
            has_when = True
        if re.search(r"\b(?:сегодня|вчера|на\\s+этой\\s+неделе|в\\s+этом\\s+месяце|только\\s+что|недавно)\b", s, flags=re.IGNORECASE):
            has_when = True

        # Actor cues (who imposed it): organizers/venue/authorities/security/etc.
        if re.search(r"\b(?:organizer|organisers|venue|authorities|security|police)\b", s, flags=re.IGNORECASE):
            has_by_whom = True
        if re.search(r"\b(?:організатор|організатори|майданчик|охорона|влада|поліція)\b", s, flags=re.IGNORECASE):
            has_by_whom = True
        if re.search(r"\b(?:организатор|организаторы|площадка|охрана|власти|полиция)\b", s, flags=re.IGNORECASE):
            has_by_whom = True

        missing: list[str] = []
        if not has_where:
            missing.append("where")
        if not has_when:
            missing.append("when")
        if not has_by_whom:
            missing.append("by_whom")

        return {"has_where": has_where, "has_when": has_when, "has_by_whom": has_by_whom, "missing": missing}

    def _is_high_impact_claim(self, fact: str) -> bool:
        s = (fact or "").lower()
        verbs = [
            # EN
            "banned", "prohibited", "forbidden", "arrested", "detained", "expelled", "closed", "shut down",
            # UK
            "заборонили", "заборона", "арештували", "затримали", "вигнали", "закрили",
            # RU
            "запретили", "запрет", "арестовали", "задержали", "выгнали", "закрыли",
        ]
        return any(v in s for v in verbs)

    def _evidence_ladder(self, sources: list[dict]) -> dict:
        """
        Evidence ladder (deterministic).

        Tiers (ceiling):
        - A: Official / primary sources (0.90)
        - A′: Official statement on a social platform (0.75)
        - B: Top-tier media / high-authority sources (0.75)
        - C: Local media / aggregators / unknown authority (0.55)
        - D: Social media / reposts (0.35)
        """
        ceilings = {"A": 0.90, "A'": 0.75, "B": 0.75, "C": 0.55, "D": 0.35}
        counts = {"A": 0, "A'": 0, "B": 0, "C": 0, "D": 0}
        highest = "D"

        def _tier_rank(t: str) -> int:
            return {"D": 1, "C": 2, "B": 3, "A'": 4, "A": 5}.get(t, 1)

        for s in (sources or []):
            t = (s.get("evidence_tier") or "C").strip().upper()
            if t not in counts:
                t = "C"
            counts[t] += 1
            if _tier_rank(t) > _tier_rank(highest):
                highest = t

        return {"highest_tier": highest, "counts": counts, "ceiling": ceilings[highest]}

    def _augment_rationale_with_gaps(self, rationale: str, *, lang: str, missing: list[str], max_tier: int) -> str:
        if not missing or not isinstance(rationale, str):
            return rationale or ""

        # Avoid duplicating if the model already mentions missing specifics.
        if re.search(r"\b(де\\b|коли\\b|ким\\b|where\\b|when\\b|who\\b)\b", rationale, flags=re.IGNORECASE):
            return rationale

        lc = (lang or "en").lower()
        if lc == "uk":
            label = "Прогалини:"
            parts = []
            if "where" in missing:
                parts.append("не вказано місце/країну")
            if "when" in missing:
                parts.append("не вказано дату/період")
            if "by_whom" in missing:
                parts.append("неясно, хто саме встановив заборону (гурт/організатори/майданчик/влада)")
            if max_tier <= 2:
                parts.append("частина джерел може бути переказами/соцмережами без офіційного підтвердження")
            gap = f"{label} " + "; ".join(parts) + "."
        elif lc == "ru":
            label = "Пробелы:"
            parts = []
            if "where" in missing:
                parts.append("не указано место/страну")
            if "when" in missing:
                parts.append("не указана дата/период")
            if "by_whom" in missing:
                parts.append("неясно, кто именно ввёл запрет (группа/организаторы/площадка/власти)")
            if max_tier <= 2:
                parts.append("часть источников может быть пересказами/соцсетями без официального подтверждения")
            gap = f"{label} " + "; ".join(parts) + "."
        else:
            label = "Gaps:"
            parts = []
            if "where" in missing:
                parts.append("no clear location")
            if "when" in missing:
                parts.append("no clear date/timeframe")
            if "by_whom" in missing:
                parts.append("unclear who imposed the restriction (band vs organizers/venue/authorities)")
            if max_tier <= 2:
                parts.append("some sources may be secondary or social without an official statement")
            gap = f"{label} " + "; ".join(parts) + "."

        # Append as a new line; keep it compact.
        if "\n" in rationale:
            return (rationale.strip() + "\n" + gap).strip()
        return (rationale.strip() + "\n" + gap).strip()

    def _registrable_domain(self, url: str) -> str | None:
        """
        Best-effort registrable domain extraction without external deps.

        This is intentionally approximate and conservative (good enough for "independent_sources" counting).
        """
        from urllib.parse import urlparse

        if not url or not isinstance(url, str):
            return None
        try:
            host = (urlparse(url).netloc or "").lower().strip()
        except Exception:
            return None
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        # Drop port if present.
        host = host.split(":")[0].strip()
        if not host or "." not in host:
            return None

        parts = [p for p in host.split(".") if p]
        if len(parts) < 2:
            return None

        # Minimal set of common 2-level public suffixes seen in our traffic.
        two_level_suffixes = {
            "co.uk", "org.uk", "ac.uk",
            "com.au", "net.au", "org.au",
            "com.ua", "org.ua", "gov.ua",
            "co.jp", "or.jp", "ne.jp",
            "com.br", "com.tr",
        }
        last2 = ".".join(parts[-2:])
        if last2 in two_level_suffixes and len(parts) >= 3:
            return ".".join(parts[-3:])
        return last2

    def _extract_source_text(self, source: dict) -> str:
        # Deterministic: use only fields we already have.
        pieces = [
            source.get("title"),
            source.get("snippet"),
            source.get("content"),
        ]
        s = " ".join([p for p in pieces if isinstance(p, str) and p.strip()])
        return re.sub(r"\s+", " ", s).strip()

    def _normalize_subject_for_match(self, subject: str) -> str:
        """
        Strict subject normalization for account-handle matching.
        Lowercase, remove whitespace/punctuation; keep only a-z0-9.
        """
        s = (subject or "").lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    def _extract_social_account_id(self, url: str) -> tuple[str | None, str | None]:
        """
        Extract (platform, account_id) from a social URL path when possible.

        This is intentionally conservative: if the account identifier can't be reliably extracted,
        returns (platform, None) and the caller must not promote the source.
        """
        from urllib.parse import urlparse

        if not url or not isinstance(url, str):
            return (None, None)
        try:
            parsed = urlparse(url)
        except Exception:
            return (None, None)
        host = (parsed.netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        path = (parsed.path or "").strip("/")
        parts = [p for p in path.split("/") if p]
        first = (parts[0] if parts else "").strip()

        if host == "instagram.com":
            platform = "instagram"
            if first in ("reel", "p", "tv", "explore", "accounts", "about", "help", "privacy"):
                return (platform, None)
            if first == "stories" and len(parts) >= 2:
                return (platform, parts[1].strip() or None)
            return (platform, first or None)

        if host in ("facebook.com", "m.facebook.com"):
            platform = "facebook"
            if first in (
                "reel",
                "watch",
                "story.php",
                "photo.php",
                "permalink.php",
                "groups",
                "events",
                "share",
                "login",
                "help",
                "privacy",
                "policies",
                "profile.php",
            ):
                return (platform, None)
            return (platform, first or None)

        if host in ("x.com", "twitter.com"):
            platform = "x"
            if not first or first in ("i", "home", "search", "explore", "share", "intent"):
                return (platform, None)
            return (platform, first or None)

        if host in ("youtube.com", "m.youtube.com"):
            platform = "youtube"
            if first.startswith("@"):
                return (platform, first[1:] or None)
            if first == "user" and len(parts) >= 2:
                return (platform, parts[1].strip() or None)
            return (platform, None)

        if host == "youtu.be":
            return ("youtube", None)

        return (None, None)

    def _is_first_person_official_marker(self, text: str, *, content_lang: str | None) -> bool:
        """
        First-person statement heuristic for official posts (deterministic).
        Must be conservative: used only as one condition for Tier A′ promotion.
        """
        s = (text or "").lower()
        lc = (content_lang or "en").lower()

        if lc == "uk":
            markers = [" ми ", " наш", " на наш", " під час наш", " на нашому", " на нашому концерті"]
        elif lc == "ru":
            markers = [" мы ", " наш", " на наш", " во время наш", " на нашем", " на нашем концерте"]
        else:
            markers = [" we ", " our ", " us ", " on our ", " at our ", " at our concert", " on our show"]

        padded = f" {s} "
        return any(m in padded for m in markers)

    def _has_social_attribution_corroboration(
        self,
        *,
        subject_norm: str,
        non_social_texts: list[str],
        content_lang: str | None,
    ) -> bool:
        """
        Condition D: lightweight independent corroboration.

        At least one non-social source (Tier B/C) must attribute the statement to the official account
        (e.g., "the band wrote on Instagram...").
        """
        if not subject_norm:
            return False

        lc = (content_lang or "en").lower()
        platform_markers = ["instagram", "facebook", "twitter", "x", "youtube"]
        if lc == "uk":
            platform_markers += ["інстаграм", "фейсбук", "твіттер", "ютуб", "ікс"]
        elif lc == "ru":
            platform_markers += ["инстаграм", "фейсбук", "твиттер", "ютуб", "икс"]

        for t in (non_social_texts or []):
            if not isinstance(t, str) or not t.strip():
                continue
            lower = t.lower()
            if not any(p in lower for p in platform_markers):
                continue
            tn = re.sub(r"[^a-z0-9]+", "", lower)
            if subject_norm in tn:
                return True
        return False

    def _is_official_social_A_prime(
        self,
        source: dict,
        *,
        claim_decomposition: dict | None,
        content_lang: str | None,
        non_social_texts: list[str],
    ) -> bool:
        """
        Tier A′ classification (Official Social Statement) — conservative and deterministic.

        Conditions (all must pass):
        A) Platform host is in the allowed set (FB/IG/X/YT).
        B) Account identifier matches normalized subject (strict), with only explicit whitelist affixes.
        C) First-person statement markers exist in the snippet/content.
        D) At least one non-social source (Tier B/C) attributes the statement to the platform + subject.
        """
        url = (source.get("link") or source.get("url") or "").strip()
        platform, account_id = self._extract_social_account_id(url)
        if platform not in ("facebook", "instagram", "x", "youtube"):
            return False
        if not account_id:
            return False

        subject = ""
        if isinstance(claim_decomposition, dict):
            subject = str(claim_decomposition.get("subject") or "")
        subject_norm = self._normalize_subject_for_match(subject)
        if not subject_norm:
            return False

        account_norm = self._normalize_subject_for_match(account_id)
        if not account_norm:
            return False

        allowed_variants = {
            subject_norm,
            f"the{subject_norm}",
            f"official{subject_norm}",
            f"{subject_norm}official",
            f"{subject_norm}band",
        }
        if account_norm not in allowed_variants:
            return False

        src_text = self._extract_source_text(source)
        if not self._is_first_person_official_marker(src_text, content_lang=content_lang):
            return False

        if not self._has_social_attribution_corroboration(
            subject_norm=subject_norm,
            non_social_texts=non_social_texts,
            content_lang=content_lang,
        ):
            return False

        return True

    def _extract_location_tokens(self, text: str) -> list[str]:
        """
        Extract coarse location-like tokens from text using simple patterns.
        Used only for plausibility consistency (not verification).
        """
        s = text or ""
        out: list[str] = []

        # Latin "in/at" + ProperNoun
        for m in re.finditer(r"\b(?:in|at|near|from)\s+([A-Z][A-Za-z-]{2,})(?:\s+([A-Z][A-Za-z-]{2,}))?", s):
            token = (m.group(1) or "").strip()
            token2 = (m.group(2) or "").strip()
            if token2:
                token = f"{token} {token2}"
            out.append(token.lower())

        # Cyrillic "в/у/на" + ProperNoun (best-effort for uk/ru)
        for m in re.finditer(r"\b(?:в|у|на)\s+([А-ЯІЇЄҐ][А-Яа-яІіЇїЄєҐґ’'\\-]{2,})(?:\s+([А-ЯІЇЄҐ][А-Яа-яІіЇїЄєҐґ’'\\-]{2,}))?", s):
            token = (m.group(1) or "").strip()
            token2 = (m.group(2) or "").strip()
            if token2:
                token = f"{token} {token2}"
            out.append(token.lower())

        # Bound and dedupe.
        seen = set()
        uniq: list[str] = []
        for t in out:
            t = re.sub(r"\s+", " ", (t or "").strip())
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq[:5]

    def _extract_date_markers(self, text: str) -> list[str]:
        """
        Extract coarse date markers (years / numeric dates / relative time keywords).
        Used only for plausibility consistency.
        """
        s = text or ""
        markers: list[str] = []

        markers.extend(re.findall(r"\b20\d{2}\b", s))
        markers.extend(re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", s))

        # Relative markers (keep small & deterministic).
        rel = [
            "today", "yesterday", "this week", "this month",
            "сьогодні", "вчора", "цього тижня", "цього місяця",
            "сегодня", "вчера", "на этой неделе", "в этом месяце",
        ]
        sl = s.lower()
        for k in rel:
            if k in sl:
                markers.append(k)

        # Normalize, dedupe, bound.
        seen = set()
        uniq: list[str] = []
        for m in markers:
            m = re.sub(r"\s+", " ", (m or "").strip().lower())
            if not m:
                continue
            if m in seen:
                continue
            seen.add(m)
            uniq.append(m)
        return uniq[:5]

    def _details_consistency(self, *, locations: list[str], dates: list[str]) -> str:
        """
        Heuristic:
        - HIGH: ≥2 sources share same key detail (location or date) and no conflicting strong alternative.
        - MED: partial alignment (some details present, but not repeated).
        - LOW: details absent or conflicting.
        """
        def _freq(items: list[str]) -> dict[str, int]:
            f: dict[str, int] = {}
            for it in items:
                if not it:
                    continue
                f[it] = f.get(it, 0) + 1
            return f

        loc_f = _freq(locations)
        date_f = _freq(dates)

        def _top2(freq: dict[str, int]) -> tuple[int, int]:
            vals = sorted(freq.values(), reverse=True)
            if not vals:
                return (0, 0)
            if len(vals) == 1:
                return (vals[0], 0)
            return (vals[0], vals[1])

        loc_top, loc_second = _top2(loc_f)
        date_top, date_second = _top2(date_f)

        # Conflicts: two different details both repeated.
        if loc_second >= 2 or date_second >= 2:
            return "low"

        if (loc_top >= 2 or date_top >= 2) and (loc_second < 2 and date_second < 2):
            return "high"

        if loc_f or date_f:
            return "medium"

        return "low"

    def _plausibility_explain(self, *, lang: str, signals: dict) -> str:
        """
        One-sentence, template-based explanation (no LLM).
        Must stay neutral and evidence-structure-based.
        """
        lc = (lang or "en").lower()
        tiers = set(signals.get("tiers_present") or [])
        independent = int(signals.get("independent_sources") or 0)
        has_by_whom = bool(signals.get("has_by_whom"))

        only_d = (tiers == {"D"}) or (tiers == set())  # treat "no tiers" like weak evidence
        has_a = "A" in tiers
        has_a_prime = "A'" in tiers
        has_b = "B" in tiers
        has_c = "C" in tiers

        if lc == "uk":
            if only_d:
                return "Джерела переважно соцмережі/репости, тож сценарій можливий, але слабко підтверджений."
            if has_a:
                return "Є посилання на первинні/офіційні джерела, тому сценарій виглядає правдоподібним, але це не дорівнює верифікації факту."
            if has_a_prime:
                return "Є пряма заява від офіційного акаунту у соцмережі, але це не замінює офіційні документи/заяви організаторів."
            if (has_b or has_c) and independent >= 2 and not has_by_whom:
                return "Є згадки в кількох незалежних джерелах, але неясно, хто саме ухвалив/озвучив рішення."
            if has_b or has_c:
                return "Є згадки в кількох джерелах, але без первинного підтвердження в наданих матеріалах."
            return "Є часткові сигнали, але деталей замало для впевненого висновку про правдоподібність."

        if lc == "ru":
            if only_d:
                return "Источники в основном соцсети/репосты, поэтому сценарий возможен, но слабо подтверждён."
            if has_a:
                return "Есть ссылки на первичные/официальные источники, поэтому сценарий выглядит правдоподобным, но это не равно верификации факта."
            if has_a_prime:
                return "Есть прямое заявление с официального аккаунта в соцсети, но это не заменяет официальные документы/заявления организаторов."
            if (has_b or has_c) and independent >= 2 and not has_by_whom:
                return "Есть упоминания в нескольких независимых источниках, но неясно, кто именно принял/озвучил решение."
            if has_b or has_c:
                return "Есть упоминания в нескольких источниках, но без первичного подтверждения в предоставленных материалах."
            return "Есть частичные сигналы, но деталей недостаточно для уверенного вывода о правдоподобности."

        # en default
        if only_d:
            return "Sources are mostly social reposts, so the scenario may be possible but weakly supported."
        if has_a:
            return "There are references to primary/official sources, so the scenario looks plausible, but this is not the same as verification."
        if has_a_prime:
            return "There is a direct statement from an official social account, but this does not replace official documents/organizer statements."
        if (has_b or has_c) and independent >= 2 and not has_by_whom:
            return "Multiple independent sources mention it, but it’s unclear who explicitly made the decision/statement."
        if has_b or has_c:
            return "Multiple sources mention it, but no primary confirmation is visible in the provided materials."
        return "There are partial signals, but not enough consistent detail to judge plausibility confidently."

    def _compute_plausibility_metric(
        self,
        *,
        fact: str,
        lang: str,
        claim_decomposition: dict | None,
        sources: list[dict],
        ladder: dict,
    ) -> dict:
        # Signals derived deterministically from claim decomposition + sources.
        has_where = bool(isinstance(claim_decomposition, dict) and claim_decomposition.get("where"))
        has_when = bool(isinstance(claim_decomposition, dict) and claim_decomposition.get("when"))
        has_by_whom = bool(isinstance(claim_decomposition, dict) and claim_decomposition.get("by_whom"))

        # Domain-level independence.
        domains: list[str] = []
        for s in (sources or []):
            d = self._registrable_domain((s.get("link") or s.get("url") or ""))
            if d:
                domains.append(d)
        independent_sources = min(6, len(set(domains)))

        tiers_present = sorted({(s.get("evidence_tier") or "").strip().upper() for s in (sources or []) if (s.get("evidence_tier"))})
        tiers_present = [t for t in tiers_present if t in ("A", "A'", "B", "C", "D")]

        # Extract details from sources to infer missing fields and consistency.
        location_hits: list[str] = []
        date_hits: list[str] = []
        combined = []
        for s in (sources or []):
            combined.append(self._extract_source_text(s))
            location_hits.extend(self._extract_location_tokens(combined[-1]))
            date_hits.extend(self._extract_date_markers(combined[-1]))
        combined_text = " ".join([c for c in combined if c]).strip()

        # Where/when/by_whom can be inferred from repeated source patterns when claim is missing.
        if not has_where:
            # If any location token appears at least twice across sources, treat as a "where" signal.
            freq: dict[str, int] = {}
            for t in location_hits:
                freq[t] = freq.get(t, 0) + 1
            has_where = any(v >= 2 for v in freq.values())

        if not has_when:
            freq: dict[str, int] = {}
            for t in date_hits:
                freq[t] = freq.get(t, 0) + 1
            has_when = any(v >= 2 for v in freq.values())

        if not has_by_whom:
            # Reuse the existing heuristic for "by_whom" cues.
            has_by_whom = bool((self._extract_claim_attributes(combined_text or fact) or {}).get("has_by_whom"))

        details_consistency = self._details_consistency(locations=location_hits, dates=date_hits)

        signals = {
            "has_where": bool(has_where),
            "has_when": bool(has_when),
            "has_by_whom": bool(has_by_whom),
            "independent_sources": int(independent_sources),
            "tiers_present": tiers_present,
            "details_consistency": details_consistency,
        }

        # Plausibility score: conservative fixed weights.
        p = 0.10
        p += 0.10 if has_where else 0.0
        p += 0.07 if has_when else 0.0
        p += 0.10 if has_by_whom else 0.0

        tiers = set(tiers_present)
        if "C" in tiers:
            p += 0.10
        if "B" in tiers:
            p += 0.15
        if "A'" in tiers:
            # Treat A′ similarly to B for plausibility (primary statement exists, but not full Tier A).
            p += 0.15
        if "A" in tiers:
            p += 0.25

        p += min(0.18, 0.06 * independent_sources)

        if details_consistency == "high":
            p += 0.10
        elif details_consistency == "medium":
            p += 0.05

        p = self._clamp(p)

        # Caps/guards:
        # - Only Tier D sources => hard cap.
        if tiers_present and set(tiers_present) == {"D"}:
            p = min(p, 0.35)
        # - Only C/D and no explicit "by_whom" => conservative cap.
        if tiers_present and set(tiers_present).issubset({"C", "D"}) and not has_by_whom:
            p = min(p, 0.50)
        # - Global cap: keep it conservative.
        p = min(p, 0.70)
        p = self._clamp(p)

        if p < 0.33:
            label = "low"
        elif p < 0.60:
            label = "medium"
        else:
            label = "high"

        explain = self._plausibility_explain(lang=lang, signals=signals)

        return {"score": p, "label": label, "explain": explain, "signals": signals}

    def _classify_source_tier(self, source: dict, *, claim_decomposition: dict | None) -> str:
        """
        Deterministically classify a source into base evidence tiers A/B/C/D.

        Note: This is intentionally conservative and code-based (no LLM).
        """
        from urllib.parse import urlparse

        social_hosts = {
            "facebook.com", "m.facebook.com",
            "instagram.com",
            "tiktok.com",
            "youtube.com", "youtu.be",
            "x.com", "twitter.com",
            "telegram.me", "t.me",
        }

        url = (source.get("link") or source.get("url") or "").strip()
        host = ""
        try:
            host = (urlparse(url).netloc or "").lower()
        except Exception:
            host = ""
        if host.startswith("www."):
            host = host[4:]

        if host in social_hosts:
            return "D"

        # Tier A: official/primary sources.
        # - Government/authority domains
        # - International orgs (.int)
        # - "Official-site" heuristic: host contains subject/by_whom token (band/org/venue).
        if host.endswith(".gov") or host.endswith(".mil") or host.endswith(".int") or host.endswith(".gov.ua"):
            return "A"

        # Treat known official health/space org domains as Tier A (primary statements/data).
        official_hosts = {"who.int", "cdc.gov", "nih.gov", "nasa.gov", "esa.int"}
        if host in official_hosts:
            return "A"

        tokens: list[str] = []
        if isinstance(claim_decomposition, dict):
            for k in ("subject", "by_whom"):
                v = claim_decomposition.get(k)
                if isinstance(v, str) and v.strip():
                    tokens.extend(re.findall(r"[a-z0-9]{4,}", v.lower()))
        tokens = [t for t in tokens if t and len(t) >= 4]
        host_compact = re.sub(r"[^a-z0-9]", "", host)
        if tokens and host_compact:
            if any(t in host_compact for t in tokens[:6]):
                return "A"

        # Tier B: trusted high-authority domains (registry-based).
        if bool(source.get("is_trusted")):
            return "B"

        # Tier C: everything else (local media / aggregators / unknown authority).
        return "C"

    def _deterministic_verified_score(
        self,
        *,
        sources: list[dict],
        search_meta: dict | None,
        claim_decomposition: dict | None,
        fact: str,
    ) -> dict:
        """
        Deterministic verified_score calculation (code-based).

        Scoring intent:
        - Verification strength comes from high-authority sources (A/B/C); social sources (D) add context only.
        - Missing key attributes are penalized explicitly, with stricter penalties for strong-verb claims.
        - The final score is capped by the highest evidence tier ceiling found (Evidence Ladder).
        """
        ceilings = {"A": 0.90, "A'": 0.75, "B": 0.75, "C": 0.55, "D": 0.35}

        # Determine the highest tier present (A > A′ > B > C > D).
        tier_rank = {"D": 1, "C": 2, "B": 3, "A'": 4, "A": 5}
        highest_tier = "D"
        for s in (sources or []):
            t = (s.get("evidence_tier") or "C").strip().upper()
            if t not in tier_rank:
                t = "C"
            if tier_rank[t] > tier_rank[highest_tier]:
                highest_tier = t
        ceiling = ceilings[highest_tier]

        # Evidence strength signal (deterministic):
        # Use the best relevance score among sources at the highest tier (excluding D),
        # otherwise fall back to search-level relevance signals.
        strength = 0.0
        if highest_tier != "D":
            rels = [
                float(s.get("relevance_score") or 0.0)
                for s in (sources or [])
                if (s.get("evidence_tier") == highest_tier and isinstance(s.get("relevance_score"), (int, float)))
            ]
            strength = max(rels) if rels else 0.0
        if strength <= 0.0:
            try:
                best = float((search_meta or {}).get("best_relevance") or 0.0)
            except Exception:
                best = 0.0
            try:
                avg = float((search_meta or {}).get("avg_relevance_top5") or 0.0)
            except Exception:
                avg = 0.0
            strength = max(best, avg)
        strength = self._clamp(strength)

        # Base score by tier (before penalties). Keep it conservative.
        if highest_tier == "A":
            base = 0.45 + 0.55 * strength
        elif highest_tier in ("A'", "B"):
            base = 0.35 + 0.55 * strength
        elif highest_tier == "C":
            base = 0.25 + 0.45 * strength
        else:
            # Social-only evidence: do not treat as verification strength.
            base = 0.20

        # Determine missing attributes (prefer claim decomposition; fall back to heuristic extraction).
        missing: list[str] = []
        action = ""
        if isinstance(claim_decomposition, dict):
            action = str(claim_decomposition.get("action") or "")
            for k in ("where", "when", "by_whom"):
                if claim_decomposition.get(k) in (None, ""):
                    missing.append(k)
        else:
            attrs = self._extract_claim_attributes(fact)
            missing = list(attrs.get("missing") or [])
            action = ""

        # Strong-verb claims require stricter penalties for missing details.
        strong_verbs = {"ban", "prohibit", "forbid", "restrict"}
        strong_claim = any(v in (action or "").lower() for v in strong_verbs) or self._is_high_impact_claim(fact)

        penalties: dict[str, float] = {}
        if "where" in missing:
            penalties["where"] = 0.12 if strong_claim else 0.08
        if "when" in missing:
            penalties["when"] = 0.08 if strong_claim else 0.06
        if "by_whom" in missing:
            penalties["by_whom"] = 0.18 if strong_claim else 0.14

        # Extra penalty when the claim is very underspecified (prevents inflated confidence for vague claims).
        if len(missing) >= 2 and highest_tier in ("C", "D"):
            penalties["underspecified"] = 0.06 if strong_claim else 0.04

        score_before_cap = self._clamp(base - sum(penalties.values()))
        final_score = min(score_before_cap, ceiling)
        final_score = self._clamp(final_score)

        return {
            "highest_tier": highest_tier,
            "tier_ceiling": ceiling,
            "strength": strength,
            "base": self._clamp(base),
            "missing": missing,
            "strong_claim": bool(strong_claim),
            "penalties": penalties,
            "score_before_ceiling": score_before_cap,
            "final_verified_score": final_score,
        }

    def _postprocess_scores_for_evidence(
        self,
        result: dict,
        *,
        fact: str,
        lang: str,
        claim_decomposition: dict | None,
        content_lang: str | None,
        search_meta: dict | None,
    ) -> dict:
        sources = result.get("sources") or []
        # Classify each source into base evidence tiers (deterministic, code-based).
        for s in (sources or []):
            try:
                s["evidence_tier"] = self._classify_source_tier(s, claim_decomposition=claim_decomposition)
            except Exception:
                s["evidence_tier"] = "C"

        # Tier A′ (official social statements): promote only when a conservative deterministic policy passes.
        # This must not rely on follower counts, badges, scraping, or LLM judgments.
        non_social_texts: list[str] = [
            self._extract_source_text(s)
            for s in (sources or [])
            if s.get("evidence_tier") in ("B", "C") and isinstance(self._extract_source_text(s), str)
        ]
        for s in (sources or []):
            if (s.get("evidence_tier") or "").strip().upper() != "D":
                continue
            try:
                if self._is_official_social_A_prime(
                    s,
                    claim_decomposition=claim_decomposition,
                    content_lang=content_lang or lang,
                    non_social_texts=non_social_texts,
                ):
                    s["evidence_tier"] = "A'"
            except Exception:
                # Conservative default: do not promote.
                pass
        ladder = self._evidence_ladder(sources)

        # Missing attributes: prefer claim decomposition when available.
        missing: list[str] = []
        if isinstance(claim_decomposition, dict):
            for k in ("where", "when", "by_whom"):
                if claim_decomposition.get(k) in (None, ""):
                    missing.append(k)
        else:
            missing = list((self._extract_claim_attributes(fact) or {}).get("missing") or [])

        # Attach diagnostic metadata (useful for SaaS reproducibility and UI explanations).
        result.setdefault("evidence", {})
        result["evidence"]["ladder"] = ladder
        result["evidence"]["claim_decomposition"] = claim_decomposition
        # Keep a heuristic "claim_attributes" object for backward compatibility with older diagnostics.
        result["evidence"]["claim_attributes"] = self._extract_claim_attributes(fact)
        Trace.event(
            "evidence.ladder",
            {
                "highest_tier": ladder.get("highest_tier"),
                "counts": ladder.get("counts"),
                "ceiling": ladder.get("ceiling"),
            },
        )

        # Deterministic verified_score derivation (LLM must not compute the final score).
        derivation = self._deterministic_verified_score(
            sources=sources,
            search_meta=search_meta,
            claim_decomposition=claim_decomposition,
            fact=fact,
        )
        result["evidence"]["verified_score_derivation"] = derivation
        result["verified_score"] = derivation["final_verified_score"]
        Trace.event(
            "score.verified",
            {
                "verified_score": result.get("verified_score"),
                "highest_tier": derivation.get("highest_tier"),
                "tier_ceiling": derivation.get("tier_ceiling"),
                "missing": derivation.get("missing"),
                "penalties": derivation.get("penalties"),
                "strength": derivation.get("strength"),
                "strong_claim": derivation.get("strong_claim"),
            },
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[Score] verified_score=%s tier=%s ceiling=%s missing=%s penalties=%s strength=%s",
                derivation.get("final_verified_score"),
                derivation.get("highest_tier"),
                derivation.get("tier_ceiling"),
                derivation.get("missing"),
                derivation.get("penalties"),
                derivation.get("strength"),
            )

        # Keep confidence conservative and bounded relative to deterministic verification strength.
        try:
            c = float(result.get("confidence_score", 0.5))
        except Exception:
            c = 0.5
        c = self._clamp(c)
        c_cap = self._clamp(float(result["verified_score"]) + 0.15)
        if len(derivation.get("missing") or []) >= 2:
            c_cap = min(c_cap, 0.65)
        result["confidence_score"] = min(c, c_cap)

        # Other Metrics: plausibility (must NOT affect RGBA / verdict).
        result.setdefault("other_metrics", {})
        result["other_metrics"]["plausibility"] = self._compute_plausibility_metric(
            fact=fact,
            lang=lang,
            claim_decomposition=claim_decomposition,
            sources=sources,
            ladder=ladder,
        )
        Trace.event(
            "other_metrics.plausibility",
            {
                "score": result["other_metrics"]["plausibility"].get("score"),
                "label": result["other_metrics"]["plausibility"].get("label"),
                "signals": result["other_metrics"]["plausibility"].get("signals"),
            },
        )

        # Explainability should also be conservative for underspecified claims.
        try:
            e = float(result.get("explainability_score", 0.5))
        except Exception:
            e = 0.5
        if len(missing) >= 2:
            e = min(e, 0.75)
        if ladder.get("highest_tier") in ("C", "D"):
            e = min(e, 0.65)
        result["explainability_score"] = self._clamp(e)

        # Augment rationale with explicit missing facts (if needed).
        if isinstance(result.get("rationale"), str):
            tier_map = {"A": 4, "A'": 3, "B": 3, "C": 2, "D": 1}
            result["rationale"] = self._augment_rationale_with_gaps(
                result["rationale"],
                lang=lang,
                missing=missing,
                max_tier=tier_map.get(ladder.get("highest_tier") or "C", 2),
            )

            # Explainability requirement for Tier A′: add one short template sentence (no LLM).
            has_a_prime = any((src.get("evidence_tier") == "A'") for src in (sources or []))
            if has_a_prime:
                lc = (lang or "en").lower()
                if lc == "uk":
                    note = "Є пряма заява від офіційного акаунту у соцмережі, але це не замінює офіційні документи/заяви організаторів."
                elif lc == "ru":
                    note = "Есть прямое заявление с официального аккаунта в соцсети, но это не заменяет официальные документы/заявления организаторов."
                else:
                    note = "There is a direct statement from an official social account, but this does not replace official documents/organizer statements."
                if note.lower() not in result["rationale"].lower():
                    result["rationale"] = (result["rationale"].strip() + "\n" + note).strip()

        return result

    async def _get_final_analysis(self, fact: str, context: str, sources_list: list, gpt_model: str, cost: int,
                                  lang: str, analysis_mode: str = "general",
                                  claim_decomposition: dict | None = None,
                                  content_lang: str | None = None,
                                  search_meta: dict | None = None) -> dict:
        analysis_result = await self.agent.analyze(fact, context, gpt_model, lang, analysis_mode)
        
        enriched_sources = self._enrich_sources_with_trust(sources_list)
        analysis_result["sources"] = enriched_sources
        analysis_result["cost"] = cost
        analysis_result["text"] = fact

        explainability_score = self._compute_explainability_score(analysis_result, enriched_sources)
        analysis_result["explainability_score"] = explainability_score
        
        r = self._clamp(analysis_result.get("danger_score", 0.0))
        context_score = self._clamp(analysis_result.get("context_score", 0.5))
        style_score = self._clamp(analysis_result.get("style_score", 0.5))
        b = (context_score + style_score) / 2.0

        # Postprocess: evidence ladder + deterministic verified_score + penalties.
        analysis_result = self._postprocess_scores_for_evidence(
            analysis_result,
            fact=fact,
            lang=lang,
            claim_decomposition=claim_decomposition,
            content_lang=content_lang,
            search_meta=search_meta,
        )

        # RGBA: keep computed in code and aligned with post-processed scores.
        g = self._clamp(analysis_result.get("verified_score", 0.5))
        a = self._clamp(analysis_result.get("explainability_score", explainability_score))
        analysis_result["rgba"] = [r, g, b, a]
        return analysis_result

    async def verify_fact(
        self, 
        fact: str, 
        search_type: str, 
        gpt_model: str, 
        lang: str, 
        analysis_mode: str = "general", 
        progress_callback=None, 
        context_text: str = "", 
        preloaded_context: str = None, 
        preloaded_sources: list = None,
        content_lang: str = None,
        include_internal: bool = False,
        search_provider: str = "auto",
        max_cost: int | None = None,
    ):
        """
        Web-only verification (no RAG).
        Strategy: Oracle -> Tier 1 -> Deep Dive
        """
        trace_id = str(uuid4())
        Trace.start(trace_id)
        Trace.event(
            "verify.start",
            {
                "fact": fact,
                "search_type": search_type,
                "gpt_model": gpt_model,
                "lang": lang,
                "content_lang": content_lang,
                "analysis_mode": analysis_mode,
                "search_provider": search_provider,
                "include_internal": include_internal,
                "max_cost": max_cost,
                "context_text": context_text,
                "preloaded_context": preloaded_context,
                "preloaded_sources": preloaded_sources,
            },
        )
        if is_local_run():
            logger.info("[Trace] verify_fact trace_id=%s (file: data/trace/%s.jsonl)", trace_id, trace_id)
        
        search_provider = (search_provider or "auto").lower()
        if search_provider not in ("auto", "tavily", "google_cse"):
            search_provider = "auto"

        # Budget-aware gating: keep total cost for this fact within `max_cost` (credits).
        model_cost = int(MODEL_COSTS.get(gpt_model, 20) or 0)
        per_search_cost = int(SEARCH_COSTS.get(search_type, 80) or 0)
        try:
            google_cse_cost = int(os.getenv("SPECTRUE_GOOGLE_CSE_COST", "0") or 0)
        except Exception:
            google_cse_cost = 0
        # Default: do not charge extra for Google CSE fallback to keep costs predictable.
        google_cse_cost = max(0, min(int(google_cse_cost), int(per_search_cost)))
        try:
            max_cost_int = int(max_cost) if max_cost is not None else None
        except Exception:
            max_cost_int = None

        budget_limited = False
        skipped_steps: list[str] = []

        def _billed_total_cost(*, tavily: int, google_cse: int) -> int:
            return int(model_cost) + int(per_search_cost) * int(tavily) + int(google_cse_cost) * int(google_cse)

        def _can_afford_total(total: int) -> bool:
            if max_cost_int is None:
                return True
            return int(total) <= max_cost_int

        def _can_add_tavily_calls(n: int = 1) -> bool:
            return _can_afford_total(
                _billed_total_cost(
                    tavily=int(tavily_calls) + int(n),
                    google_cse=int(google_cse_calls),
                )
            )

        def _can_add_google_cse_calls(n: int = 1) -> bool:
            return _can_afford_total(
                _billed_total_cost(
                    tavily=int(tavily_calls),
                    google_cse=int(google_cse_calls) + int(n),
                )
            )

        tavily_calls = 0
        google_cse_calls = 0
        page_fetches = 0
        cache_hit_any = False
        search_meta: dict = {}

        # Global context optimization
        if preloaded_context:
            if max_cost_int is not None and max_cost_int < model_cost:
                Trace.event("verify.result", {"path": "preloaded", "error_key": "app.error_insufficient_credits"})
                Trace.stop()
                return {
                    "error_key": "app.error_insufficient_credits",
                    "required": model_cost,
                    "remaining": max_cost_int,
                    "cost": 0,
                    "text": fact,
                    "sources": [],
                    "rgba": [0.5, 0.5, 0.5, 0.5],
                    "search": {
                        "provider": "preloaded",
                        "quality": "unknown",
                        "fallback_used": False,
                        "tavily_calls": 0,
                        "google_cse_calls": 0,
                        "page_fetches": 0,
                        "budget_limited": True,
                        "skipped_steps": ["ai_analysis"],
                    },
                }
            if progress_callback:
                await progress_callback("using_global_context")
            
            logger.debug("[Waterfall] Using preloaded global context (%d chars).", len(preloaded_context))
            context_to_use = preloaded_context[:100000]
            sources_to_use = preloaded_sources or []
            
            if progress_callback:
                await progress_callback("ai_analysis")
            
            result = await self._get_final_analysis(
                fact, context_to_use, sources_to_use[:10], gpt_model,
                model_cost, lang, analysis_mode
            )
            result["search"] = {
                "provider": "preloaded",
                "quality": "unknown",
                "fallback_used": False,
                "tavily_calls": 0,
                "google_cse_calls": 0,
                "page_fetches": 0,
            }
            if include_internal:
                result["_internal"] = {"context": context_to_use, "sources": sources_to_use[:10]}
            Trace.event("verify.result", {"path": "preloaded", "verified_score": result.get("verified_score"), "cost": result.get("cost")})
            Trace.stop()
            return result

        # Generate search queries
        if progress_callback:
            await progress_callback("generating_queries")
        
        SHORT_TEXT_THRESHOLD = 300
        search_queries = [fact, fact]
        
        try:
            if len(fact) < SHORT_TEXT_THRESHOLD:
                logger.debug("[Waterfall] Short text (%d chars). Using fast query strategy.", len(fact))
                # Avoid duplicate queries and try to keep an English-ish query in slot 0 for better global coverage.
                # Keep at least 2 queries because deep dive expects EN + Native.
                fallback_queries = self.agent._smart_fallback(fact, lang=lang, content_lang=content_lang or lang)
                if len(fallback_queries) < 2:
                    fallback_queries.append(fallback_queries[0])
                search_queries = fallback_queries[:2]

                # Optional: use LLM to rewrite/clean short queries (helps with typos like "Image Dragons").
                # Keep it opt-out to preserve the previous fast path.
                if self._env_true("SPECTRUE_LLM_QUERY_REWRITE_SHORT", True):
                    try:
                        llm_queries = await self.agent.generate_search_queries(
                            fact, context=context_text, lang=lang, content_lang=content_lang
                        )
                        if llm_queries:
                            logger.debug("[Waterfall] Short text: LLM rewrote queries: %s", llm_queries[:2])
                            # Preserve structure: slot 0 is global/EN, slot 1 is content language when available.
                            search_queries[0] = llm_queries[0]
                            if len(llm_queries) > 1:
                                search_queries[1] = llm_queries[1]
                    except Exception as e:
                        logger.debug("[Waterfall] Short text: LLM rewrite failed, keeping fallback. %s", e)
            else:
                queries_list = await self.agent.generate_search_queries(
                    fact, context=context_text, lang=lang, content_lang=content_lang
                )
                if queries_list and len(queries_list) > 0:
                    search_queries = queries_list[:2]
                    logger.debug("[Waterfall] Generated %d queries (LLM): %s", len(queries_list), search_queries)
                else:
                    logger.debug("[Waterfall] GPT-5 Nano returned empty, using fallback.")
        except Exception as e:
            logger.warning("[Waterfall] Failed to generate queries: %s. Using fallback.", e)

        claim_decomposition = None
        try:
            claim_decomposition = (self.agent.last_query_meta or {}).get("claim_decomposition")
        except Exception:
            claim_decomposition = None
        if isinstance(claim_decomposition, dict):
            Trace.event("claim.decomposition", {"claim": claim_decomposition})
            if is_local_run():
                logger.info("[Claim] Decomposition: %s", claim_decomposition)
            else:
                logger.debug("[Claim] Decomposition: %s", claim_decomposition)

        # Normalize queries (used by Oracle and search providers) to keep payloads valid and deterministic.
        search_queries = [self._normalize_search_query(q) for q in (search_queries or [])]
        if not search_queries:
            search_queries = [self._normalize_search_query(fact), self._normalize_search_query(fact)]
        if len(search_queries) == 1:
            search_queries.append(search_queries[0])
        if not search_queries[0]:
            search_queries[0] = self._normalize_search_query(fact)
        if not search_queries[1]:
            search_queries[1] = self._normalize_search_query(fact)
        
        # Oracle (Google Fact Check)
        if progress_callback:
            await progress_callback("checking_oracle")
            
        oracle_query = search_queries[0]
        logger.debug("[Waterfall] Oracle query: '%s...'", oracle_query[:100])
        
        oracle_result = await self.google_tool.search(oracle_query, content_lang or lang)
        
        if oracle_result:
            oracle_result["text"] = fact
            logger.debug("[Waterfall] ✓ Oracle hit (Google Fact Check). Stopping.")
            oracle_result["search"] = {
                "provider": "google_fact_check",
                "quality": "good",
                "fallback_used": False,
                "tavily_calls": tavily_calls,
                "google_cse_calls": google_cse_calls,
                "page_fetches": page_fetches,
            }
            oracle_result["search_cache_hit"] = False
            if include_internal:
                oracle_result["_internal"] = {
                    "context": oracle_result.get("analysis") or oracle_result.get("rationale") or "",
                    "sources": oracle_result.get("sources") or [],
                }
            Trace.event("verify.result", {"path": "oracle", "verified_score": oracle_result.get("verified_score"), "cost": oracle_result.get("cost")})
            Trace.stop()
            return oracle_result

        # Tier 1 (Trusted Domains)
        ttl = self.time_sensitive_ttl if self._is_time_sensitive(fact, lang) else None
        
        if progress_callback:
            await progress_callback("searching_tier1")
        
        search_lang = content_lang if content_lang else lang
        tier1_domains = get_trusted_domains_by_lang(search_lang)
        logger.debug("[Waterfall] Tier 1 domains for lang='%s': %d domains", search_lang, len(tier1_domains))
        
        tier1_query = search_queries[1] if len(search_queries) > 1 and content_lang else search_queries[0]
        logger.debug("[Waterfall] Tier 1 query: '%s...'", tier1_query[:80])
        tier1_context = ""
        tier1_sources: list[dict] = []
        tier1_tool_meta: dict = {}
        passes_detail: list[dict] = []

        if max_cost_int is not None and max_cost_int < model_cost:
            # Even the analysis step can't fit in budget; return a clean error.
            Trace.event("verify.result", {"path": "budget_gate", "error_key": "app.error_insufficient_credits"})
            Trace.stop()
            return {
                "error_key": "app.error_insufficient_credits",
                "required": model_cost,
                "remaining": max_cost_int,
                "cost": 0,
                "text": fact,
                "sources": [],
                "rgba": [0.5, 0.5, 0.5, 0.5],
                "search": {
                    "provider": "none",
                    "quality": "unknown",
                    "fallback_used": False,
                    "tavily_calls": 0,
                    "google_cse_calls": 0,
                    "page_fetches": 0,
                },
            }

        # If user doesn't have budget for even 1 web-search call, skip web search entirely.
        tier1_provider = "google_cse" if search_provider == "google_cse" else "tavily"
        can_run_tier1 = _can_add_google_cse_calls(1) if tier1_provider == "google_cse" else _can_add_tavily_calls(1)
        if not can_run_tier1:
            budget_limited = True
            skipped_steps.append("tier1_search")
        else:
            if search_provider == "google_cse":
                tier1_context, tier1_sources = await self.google_cse_tool.search(
                    tier1_query, lang=search_lang, max_results=7, ttl=ttl
                )
                google_cse_calls += 1
                cache_hit_any = cache_hit_any or bool(self.google_cse_tool.last_cache_hit)
                tier1_tool_meta = {
                    "provider": "google_cse",
                    "quality": "good" if len(tier1_sources) >= 3 else "poor",
                    "fallback_used": False,
                }
                passes_detail.append(
                    {
                        "pass": "tier1",
                        "provider": "google_cse",
                        "queries": [tier1_query],
                        "domains_count": len(tier1_domains or []),
                        "lang": search_lang,
                        "meta": {"sources_count": len(tier1_sources or [])},
                    }
                )
            else:
                Trace.event(
                    "search.tavily.start",
                    {
                        "query": tier1_query,
                        "lang": search_lang,
                        "depth": search_type,
                        "ttl": ttl,
                        "domains_count": len(tier1_domains or []),
                    },
                )
                tier1_context, tier1_sources = await self.web_search_tool.search(
                    tier1_query,
                    search_depth=search_type,
                    ttl=ttl,
                    domains=tier1_domains,
                    lang=search_lang,
                )
                Trace.event(
                    "search.tavily.done",
                    {
                        "context_chars": len(tier1_context or ""),
                        "sources_count": len(tier1_sources or []),
                        "meta": self.web_search_tool.last_search_meta,
                        "cache_hit": bool(self.web_search_tool.last_cache_hit),
                    },
                )
                tavily_calls += 1
                cache_hit_any = cache_hit_any or bool(self.web_search_tool.last_cache_hit)
                page_fetches += int((self.web_search_tool.last_search_meta or {}).get("page_fetches") or 0)
                tier1_tool_meta = dict(self.web_search_tool.last_search_meta or {})
                passes_detail.append(
                    {
                        "pass": "tier1",
                        "provider": "tavily",
                        "queries": [tier1_query],
                        "domains_count": len(tier1_domains or []),
                        "lang": search_lang,
                        "meta": {
                            "best_relevance": tier1_tool_meta.get("best_relevance"),
                            "avg_relevance_top5": tier1_tool_meta.get("avg_relevance_top5"),
                            "sources_count": tier1_tool_meta.get("sources_count"),
                        },
                    }
                )

        def _dedupe_sources(primary: list[dict], extra: list[dict], *, limit: int = 10) -> list[dict]:
            all_sources = list(primary or []) + list(extra or [])
            seen = set()
            merged: list[dict] = []
            for s in all_sources:
                link = (s.get("link") or "").strip()
                if not link or link in seen:
                    continue
                seen.add(link)
                merged.append(s)
                if len(merged) >= limit:
                    break
            return merged

        quality = tier1_tool_meta.get("quality") or ("good" if len(tier1_sources) >= 3 else "poor")
        poor = (quality == "poor") or (len(tier1_sources) < 3)
        fallback_used = False
        fallback_provider: str | None = None
        passes: list[str] = ["tier1"]

        # Multi-pass search escalation (per SPEC KIT):
        # If relevance is low, trigger a refined Tier 1 pass with stricter anchoring.
        # `fallback_used` MUST mean a real strategy change (query regeneration), not a provider retry.
        try:
            best_rel = float(tier1_tool_meta.get("best_relevance"))
        except Exception:
            best_rel = None
        try:
            avg_rel = float(tier1_tool_meta.get("avg_relevance_top5"))
        except Exception:
            avg_rel = None

        needs_anchor_refine = (
            (best_rel is not None and best_rel < 0.35)
            or (avg_rel is not None and avg_rel < 0.30)
        )
        if needs_anchor_refine and tier1_provider == "tavily":
            strict_queries = self.agent.build_strict_queries(
                claim_decomposition, lang=lang, content_lang=content_lang
            )
            strict_queries = [self._normalize_search_query(q) for q in (strict_queries or [])]
            strict_query = (
                strict_queries[1]
                if (content_lang and len(strict_queries) > 1)
                else (strict_queries[0] if strict_queries else "")
            )
            strict_query = self._normalize_search_query(strict_query) or tier1_query

            if _can_add_tavily_calls(1):
                if progress_callback:
                    await progress_callback("searching_deep")
                if is_local_run():
                    logger.info(
                        "[Waterfall] Low relevance (best=%s avg_top5=%s). Anchored refine: %s",
                        best_rel,
                        avg_rel,
                        strict_query[:100],
                    )
                else:
                    logger.debug(
                        "[Waterfall] Low relevance (best=%s avg_top5=%s). Anchored refine: %s",
                        best_rel,
                        avg_rel,
                        strict_query[:100],
                    )
                Trace.event(
                    "search.tavily.start",
                    {
                        "query": strict_query,
                        "lang": search_lang,
                        "depth": search_type,
                        "ttl": ttl,
                        "domains_count": len(tier1_domains or []),
                        "anchored_refine": True,
                    },
                )
                refine_context, refine_sources = await self.web_search_tool.search(
                    strict_query,
                    search_depth=search_type,
                    ttl=ttl,
                    domains=tier1_domains,
                    lang=search_lang,
                )
                Trace.event(
                    "search.tavily.done",
                    {
                        "context_chars": len(refine_context or ""),
                        "sources_count": len(refine_sources or []),
                        "meta": self.web_search_tool.last_search_meta,
                        "cache_hit": bool(self.web_search_tool.last_cache_hit),
                        "anchored_refine": True,
                    },
                )
                tavily_calls += 1
                cache_hit_any = cache_hit_any or bool(self.web_search_tool.last_cache_hit)
                page_fetches += int((self.web_search_tool.last_search_meta or {}).get("page_fetches") or 0)

                fallback_used = True
                fallback_provider = "anchored_refine"
                passes.append("anchored_refine")
                tier1_sources = _dedupe_sources(tier1_sources, refine_sources, limit=10)
                tier1_context = f"{tier1_context}\n{refine_context}".strip()
                tier1_tool_meta = dict(self.web_search_tool.last_search_meta or {})
                passes_detail.append(
                    {
                        "pass": "anchored_refine",
                        "provider": "tavily",
                        "queries": [strict_query],
                        "domains_count": len(tier1_domains or []),
                        "lang": search_lang,
                        "meta": {
                            "best_relevance": tier1_tool_meta.get("best_relevance"),
                            "avg_relevance_top5": tier1_tool_meta.get("avg_relevance_top5"),
                            "sources_count": tier1_tool_meta.get("sources_count"),
                        },
                    }
                )

                # Re-evaluate quality after anchored refine.
                quality = tier1_tool_meta.get("quality") or ("good" if len(tier1_sources) >= 3 else "poor")
                poor = (quality == "poor") or (len(tier1_sources) < 3)
            else:
                budget_limited = True
                if "anchored_refine" not in skipped_steps:
                    skipped_steps.append("anchored_refine")

        # Smart Basic: at most 1 extra cheap fallback search, no deep dive.
        if (
            search_type == "basic"
            and poor
            and search_provider in ("auto", "tavily")
        ):
            if progress_callback:
                await progress_callback("searching_deep")

            if search_provider == "auto" and self.google_cse_tool.enabled():
                if not _can_add_google_cse_calls(1):
                    budget_limited = True
                    skipped_steps.append("fallback_google_cse")
                else:
                    cse_context, cse_sources = await self.google_cse_tool.search(
                        tier1_query, lang=search_lang, max_results=6, ttl=ttl
                    )
                    google_cse_calls += 1
                    cache_hit_any = cache_hit_any or bool(self.google_cse_tool.last_cache_hit)
                    fallback_used = True
                    fallback_provider = "google_cse"
                    passes.append("google_cse")
                    passes_detail.append(
                        {
                            "pass": "google_cse",
                            "provider": "google_cse",
                            "queries": [tier1_query],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": {"sources_count": len(cse_sources or [])},
                        }
                    )
                    tier1_sources = _dedupe_sources(tier1_sources, cse_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{cse_context}".strip()
            else:
                if not _can_add_tavily_calls(1):
                    budget_limited = True
                    skipped_steps.append("fallback_tavily")
                else:
                    # Tavily fallback (still basic): drop domain filter, keep it cheap and predictable.
                    fb_query = search_queries[1] if (content_lang and content_lang != "en") else search_queries[0]
                    Trace.event(
                        "search.tavily.start",
                        {
                            "query": fb_query,
                            "lang": search_lang,
                            "depth": "basic",
                            "ttl": ttl,
                            "domains_count": 0,
                            "fallback": True,
                        },
                    )
                    fb_context, fb_sources = await self.web_search_tool.search(
                        fb_query,
                        search_depth="basic",
                        ttl=ttl,
                        domains=None,
                        lang=search_lang,
                    )
                    Trace.event(
                        "search.tavily.done",
                        {
                            "context_chars": len(fb_context or ""),
                            "sources_count": len(fb_sources or []),
                            "meta": self.web_search_tool.last_search_meta,
                            "cache_hit": bool(self.web_search_tool.last_cache_hit),
                            "fallback": True,
                        },
                    )
                    tavily_calls += 1
                    cache_hit_any = cache_hit_any or bool(self.web_search_tool.last_cache_hit)
                    page_fetches += int((self.web_search_tool.last_search_meta or {}).get("page_fetches") or 0)
                    # Update tool meta so `best_relevance` / `avg_relevance_top5` reflect this real strategy change.
                    tier1_tool_meta = dict(self.web_search_tool.last_search_meta or {})
                    fallback_used = True
                    fallback_provider = "tavily_refine"
                    passes.append("tavily_refine")
                    passes_detail.append(
                        {
                            "pass": "tavily_refine",
                            "provider": "tavily",
                            "queries": [fb_query],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": dict(self.web_search_tool.last_search_meta or {}),
                        }
                    )
                    tier1_sources = _dedupe_sources(tier1_sources, fb_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{fb_context}".strip()

            # Re-evaluate quality after fallback.
            quality = "good" if len(tier1_sources) >= 3 else "poor"
            poor = (quality == "poor") or (len(tier1_sources) < 3)

        # Advanced: allow quality-gated fallback + deep dive, but only if budget allows it.
        if search_type == "advanced" and poor and search_provider in ("auto", "tavily"):
            # Optional: try Google CSE once (auto only) before deep dive, if enabled and budget allows.
            if (
                search_provider == "auto"
                and self.google_cse_tool.enabled()
            ):
                if not _can_add_google_cse_calls(1):
                    budget_limited = True
                    skipped_steps.append("fallback_google_cse")
                else:
                    cse_context, cse_sources = await self.google_cse_tool.search(
                        tier1_query, lang=search_lang, max_results=6, ttl=ttl
                    )
                    google_cse_calls += 1
                    cache_hit_any = cache_hit_any or bool(self.google_cse_tool.last_cache_hit)
                    fallback_used = True
                    fallback_provider = "google_cse"
                    passes.append("google_cse")
                    passes_detail.append(
                        {
                            "pass": "google_cse",
                            "provider": "google_cse",
                            "queries": [tier1_query],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": {"sources_count": len(cse_sources or [])},
                        }
                    )
                    tier1_sources = _dedupe_sources(tier1_sources, cse_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{cse_context}".strip()

                    quality = "good" if len(tier1_sources) >= 3 else "poor"
                    poor = (quality == "poor") or (len(tier1_sources) < 3)

            # Deep dive (EN + Native): only in Advanced and only if we can afford 2 more search calls.
            if poor and search_provider != "google_cse":
                if not _can_add_tavily_calls(2):
                    budget_limited = True
                    skipped_steps.append("deep_dive")
                else:
                    logger.debug("[Waterfall] Tier 1 weak. Running deep dive (EN + Native)...")
                    if progress_callback:
                        await progress_callback("searching_deep")

                    en_task = self.web_search_tool.search(
                        search_queries[0],
                        search_depth=search_type,
                        ttl=ttl,
                        lang="en",
                    )
                    native_task = self.web_search_tool.search(
                        search_queries[1],
                        search_depth=search_type,
                        ttl=ttl,
                        lang=search_lang,
                    )
                    (en_context, en_sources), (native_context, native_sources) = await asyncio.gather(
                        en_task, native_task
                    )
                    tavily_calls += 2
                    cache_hit_any = cache_hit_any or bool(self.web_search_tool.last_cache_hit)
                    page_fetches += int((self.web_search_tool.last_search_meta or {}).get("page_fetches") or 0)
                    # Keep meta updated (even though deep dive runs 2 searches, last_search_meta is best-effort).
                    tier1_tool_meta = dict(self.web_search_tool.last_search_meta or {})
                    fallback_used = True
                    fallback_provider = fallback_provider or "deep_dive"
                    passes.append("deep_dive")
                    passes_detail.append(
                        {
                            "pass": "deep_dive",
                            "provider": "tavily",
                            "queries": [search_queries[0], search_queries[1]],
                            "domains_count": 0,
                            "lang": search_lang,
                            "meta": dict(self.web_search_tool.last_search_meta or {}),
                        }
                    )

                    # Aggregate
                    tier1_sources = _dedupe_sources(tier1_sources, en_sources + native_sources, limit=10)
                    tier1_context = f"{tier1_context}\n{en_context}\n{native_context}".strip()

                    if len(tier1_context) > 100000:
                        tier1_context = tier1_context[:100000]

                    quality = "good" if len(tier1_sources) >= 3 else "poor"
                    poor = (quality == "poor") or (len(tier1_sources) < 3)

        # Final search metadata (for UI + billing transparency)
        def _final_relevance_metrics(sources: list[dict]) -> dict:
            rels = [
                float(s.get("relevance_score") or 0.0)
                for s in (sources or [])
                if isinstance(s.get("relevance_score"), (int, float))
            ]
            rels.sort(reverse=True)
            top = rels[:5]
            avg = (sum(top) / len(top)) if top else 0.0
            best = rels[0] if rels else 0.0
            return {"best_relevance": best, "avg_relevance_top5": avg}

        final_rel = _final_relevance_metrics(tier1_sources)
        search_meta = {
            "provider": (tier1_tool_meta.get("provider") or ("none" if not (tavily_calls or google_cse_calls) else "tavily")),
            "quality": quality if (tavily_calls or google_cse_calls) else "unknown",
            "avg_relevance_top5": tier1_tool_meta.get("avg_relevance_top5"),
            "best_relevance": tier1_tool_meta.get("best_relevance"),
            # Final relevance metrics computed from the final merged source list (useful after multi-pass).
            "avg_relevance_top5_final": final_rel.get("avg_relevance_top5"),
            "best_relevance_final": final_rel.get("best_relevance"),
            "fallback_used": bool(fallback_used),
            "fallback_provider": fallback_provider,
            "passes": passes,
            "passes_detail": passes_detail,
            "multi_pass_used": len(passes) > 1,
            "tavily_calls": tavily_calls,
            "google_cse_calls": google_cse_calls,
            "page_fetches": page_fetches,
            "budget_limited": bool(budget_limited),
            "skipped_steps": skipped_steps,
        }
        Trace.event(
            "search.passes",
            {
                "passes": passes,
                "passes_detail": passes_detail,
                "best_relevance": search_meta.get("best_relevance"),
                "avg_relevance_top5": search_meta.get("avg_relevance_top5"),
                "fallback_used": search_meta.get("fallback_used"),
            },
        )

        if progress_callback:
            await progress_callback("ai_analysis")

        # Cost is per Tavily call; Google CSE fallback is cheap by default (configurable).
        total_cost = _billed_total_cost(tavily=int(tavily_calls), google_cse=int(google_cse_calls))
        if max_cost_int is not None and total_cost > max_cost_int:
            # Shouldn't happen due to gating, but keep it safe.
            total_cost = max_cost_int
            budget_limited = True
            if "budget_cap" not in skipped_steps:
                skipped_steps.append("budget_cap")

        result = await self._get_final_analysis(
            fact,
            tier1_context,
            tier1_sources[:10],
            gpt_model,
            total_cost,
            lang,
            analysis_mode,
            claim_decomposition=claim_decomposition,
            content_lang=content_lang or lang,
            search_meta=search_meta,
        )
        result["search"] = search_meta
        result["search_cache_hit"] = bool(cache_hit_any)
        result["checks"] = {
            "engine_version": ENGINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "search_strategy": SEARCH_STRATEGY_VERSION,
        }
        if include_internal:
            result["_internal"] = {"context": tier1_context, "sources": tier1_sources[:10]}
        Trace.event(
            "verify.result",
            {
                "path": "analysis",
                "verified_score": result.get("verified_score"),
                "confidence_score": result.get("confidence_score"),
                "cost": result.get("cost"),
                "search": result.get("search"),
            },
        )
        Trace.stop()
        return result
