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

import argparse
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
import re
import asyncio
from datetime import date
from urllib.parse import unquote, urlparse

# --- Конфигурация источников ---

# 1. Избранные статьи (фундаментальная база)
# Уровень 4 (~10k) или локальный эквивалент. Для некоторых языков список разбит на подстраницы.
VITal_ARTICLES_URLS = {
    "ru": {
        3: "https://ru.wikipedia.org/wiki/%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F:%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D1%81%D1%82%D0%B0%D1%82%D0%B5%D0%B9,_%D0%BA%D0%BE%D1%82%D0%BE%D1%80%D1%8B%D0%B5_%D0%B4%D0%BE%D0%BB%D0%B6%D0%BD%D1%8B_%D0%B1%D1%8B%D1%82%D1%8C_%D0%B2%D0%BE_%D0%B2%D1%81%D0%B5%D1%85_%D1%8F%D0%B7%D1%8B%D0%BA%D0%BE%D0%B2%D1%8B%D1%85_%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D1%8F%D1%85/1000",
        4: "https://ru.wikipedia.org/wiki/%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F:%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D1%81%D1%82%D0%B0%D1%82%D0%B5%D0%B9,_%D0%BA%D0%BE%D1%82%D0%BE%D1%80%D1%8B%D0%B5_%D0%B4%D0%BE%D0%BB%D0%B6%D0%BD%D1%8B_%D0%B1%D1%8B%D1%82%D1%8C_%D0%B2%D0%BE_%D0%B2%D1%81%D0%B5%D1%85_%D1%8F%D0%B7%D1%8B%D0%BA%D0%BE%D0%B2%D1%8B%D1%85_%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D1%8F%D1%85/10000"
    },
    "en": {
        3: "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/3",
        4: "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4"
    },
    "uk": {
        3: "https://uk.wikipedia.org/wiki/%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F:%D0%A1%D1%82%D0%B0%D1%82%D1%82%D1%96,_%D1%8F%D0%BA%D1%96_%D0%BF%D0%BE%D0%B2%D0%B8%D0%BD%D0%BD%D1%96_%D0%B1%D1%83%D1%82%D0%B8_%D0%B2_%D1%83%D1%81%D1%96%D1%85_%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F%D1%85",
        4: "https://uk.wikipedia.org/wiki/%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F:%D0%A1%D1%82%D0%B0%D1%82%D1%82%D1%96,_%D1%8F%D0%BA%D1%96_%D0%BF%D0%BE%D0%B2%D0%B8%D0%BD%D0%BD%D1%96_%D0%B1%D1%83%D1%82%D0%B8_%D0%B2_%D1%83%D1%81%D1%96%D1%85_%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F%D1%85/%D0%A0%D0%BE%D0%B7%D1%88%D0%B8%D1%80%D0%B5%D0%BD%D0%B8%D0%B9"
    },
    "fr": {
        3: "https://fr.wikipedia.org/wiki/Wikip%C3%A9dia:Articles_vitaux/Niveau_3",
        4: "https://fr.wikipedia.org/wiki/Wikip%C3%A9dia:Articles_vitaux/Niveau_4"
    },
    "de": {
        3: "https://de.wikipedia.org/wiki/Wikipedia:Artikel,_die_es_in_allen_Wikipedias_geben_sollte/Erweitert",
        4: "https://de.wikipedia.org/wiki/Wikipedia:Artikel,_die_es_in_allen_Wikipedias_geben_sollte"
    },
    "es": {
        3: "https://es.wikipedia.org/wiki/Wikipedia:Lista_de_art%C3%ADculos_que_toda_Wikipedia_deber%C3%ADa_tener",
        4: "https://es.wikipedia.org/wiki/Wikipedia:Lista_de_art%C3%ADculos_que_toda_Wikipedia_deber%C3%ADa_tener/Expandida"
    },
    "ja": {
        3: "https://ja.wikipedia.org/wiki/Wikipedia:%E3%81%99%E3%81%B9%E3%81%A6%E3%81%AE%E8%A8%80%E8%AA%9E%E7%89%88%E3%81%AB%E3%81%82%E3%82%8B%E3%81%B9%E3%81%8D%E9%A0%85%E7%9B%AE%E3%81%AE%E4%B8%80%E8%A6%A7/1000",
        4: "https://ja.wikipedia.org/wiki/Wikipedia:%E3%81%99%E3%81%B9%E3%81%A6%E3%81%AE%E8%A8%80%E8%AA%9E%E7%89%88%E3%81%AB%E3%81%82%E3%82%8B%E3%81%B9%E3%81%8D%E9%A0%85%E7%9B%AE%E3%81%AE%E4%B8%80%E8%A6%A7/1%E4%B8%87%E3%81%AE%E9%A0%85%E7%9B%AE"
    },
    "zh": {
        3: "https://zh.wikipedia.org/wiki/Wikipedia:%E5%9F%BA%E7%A4%8E%E6%A2%9D%E7%9B%AE",
        4: "https://zh.wikipedia.org/wiki/Wikipedia:%E5%9F%BA%E7%A4%8E%E6%A2%9D%E7%9B%AE/%E6%93%B4%E5%B1%95"
    }
}

# 2. Популярные статьи (актуальная база)
TOP_ARTICLES_API_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{project}/all-access/{year}/{month}/all-days"


# --- Функции-помощники ---

def project_for_lang(lang: str) -> str:
    return f"{(lang or 'en').strip().lower()}.wikipedia.org"


def latest_complete_month(today: date | None = None) -> tuple[int, int]:
    d = today or date.today()
    year, month = d.year, d.month
    return (year - 1, 12) if month == 1 else (year, month - 1)


def prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def normalize_title(article: str) -> str:
    return unquote(article or "").replace("_", " ").strip()


# --- Функции сбора данных ---

async def scrape_vital_articles(session: httpx.AsyncClient, lang: str, level: int) -> set[str]:
    """Скачивает и рекурсивно обходит главную страницу уровня и её подстраницы,
    собирая ссылки на статьи основного пространства.
    """
    lang_urls = VITal_ARTICLES_URLS.get(lang)
    if not lang_urls or level not in lang_urls:
        print(f"⚠️ URL для избранных статей языка '{lang}' уровня '{level}' не найден, пропуск.")
        return set()

    start_url = lang_urls[level]
    parsed = urlparse(start_url)
    base_origin = f"{parsed.scheme}://{parsed.netloc}"
    base_path = parsed.path.rstrip('/')

    print(f"--- Сбор избранных статей для '{lang}' (Уровень {level}) с рекурсивным обходом... ---")

    titles: set[str] = set()
    visited: set[str] = set()
    queue: list[str] = [base_path]

    def is_mainspace_href(href: str) -> bool:
        return bool(re.match(r'^/wiki/[^:]+$', href))

    while queue:
        path = queue.pop(0)
        if path in visited:
            continue
        visited.add(path)

        page_url = base_origin + path
        try:
            resp = await session.get(page_url)
            resp.raise_for_status()
        except Exception as e:
            print(f"  ✗ Не удалось загрузить {page_url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, 'lxml')

        for a in soup.select('a[href^="/wiki/"]'):
            href = (a.get('href') or '').split('#', 1)[0]
            title_attr = a.get('title') or ''

            if href.startswith(base_path + '/'):
                if href not in visited and href not in queue:
                    queue.append(href)
                continue

            if is_mainspace_href(href):
                t = title_attr.replace(' (страница не существует)', '').strip() if title_attr else normalize_title(
                    href.replace('/wiki/', '', 1))
                if t:
                    titles.add(t)

    print(f"--- Найдено {len(titles)} избранных статей (после обхода подстраниц).")
    return titles


async def fetch_top_articles_for_month(session: httpx.AsyncClient, lang: str, year: int, month: int) -> list[str]:
    """Скачивает список топ-статей за один месяц."""
    project = project_for_lang(lang)
    url = TOP_ARTICLES_API_BASE.format(project=project, year=year, month=f"{month:02d}")
    try:
        r = await session.get(url)
        if r.status_code == 404: return []
        r.raise_for_status()
        data = r.json()
        articles = (data.get("items") or [{}])[0].get("articles") or []
        service_titles = {"Main Page", "Заглавная страница", "Special:Search"}
        titles = [normalize_title(a.get("article")) for a in articles]
        return [t for t in titles if t and t not in service_titles]
    except Exception:
        return []


async def get_top_articles(session: httpx.AsyncClient, lang: str, num_months: int) -> set[str]:
    """Асинхронно собирает топ-статьи за несколько последних месяцев."""
    print(f"--- Сбор популярных статей за последние {num_months} месяцев...")
    year, month = latest_complete_month()
    tasks = []
    for _ in range(num_months):
        tasks.append(fetch_top_articles_for_month(session, lang, year, month))
        year, month = prev_month(year, month)

    results = await asyncio.gather(*tasks)
    titles = {title for month_list in results for title in month_list}
    print(f"--- Найдено {len(titles)} уникальных популярных статей.")
    return titles


async def main():
    parser = argparse.ArgumentParser(description="Сбор и объединение списков избранных и популярных статей Википедии.")
    parser.add_argument("--lang", type=str, required=True, help="Код языка (например, 'ru', 'en', 'uk', 'fr')")
    parser.add_argument("--level", type=int, default=4,
                        help="Уровень важности избранных статей (например, 4 для ~10k).")
    parser.add_argument("--months", type=int, default=12,
                        help="Сколько последних месяцев использовать для сбора популярных статей.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Папка для сохранения итогового списка.")
    args = parser.parse_args()

    headers = {"User-Agent": "Spectrue/1.0 (contact: dev@spectrue.app)"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
        vital_task = scrape_vital_articles(client, args.lang, args.level)
        top_task = get_top_articles(client, args.lang, args.months)

        vital_articles, top_articles = await asyncio.gather(vital_task, top_task)

    combined_titles = sorted(list(vital_articles.union(top_articles)))

    if not combined_titles:
        print("❌ Не удалось собрать ни одной статьи.")
        return

    output_file = f"data/wiki_sets/master_list_{args.lang}.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for title in combined_titles:
            f.write(title + "\n")

    print(f"\n✅ Готово! Итоговый список из {len(combined_titles)} уникальных статей сохранён в: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
