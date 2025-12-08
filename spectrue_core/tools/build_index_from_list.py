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
import re
import sys
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import wikipedia
from tqdm import tqdm
import faiss

# Добавляем корень проекта в sys.path для импорта spectrue_api
project_root = Path(__file__).resolve().parents[2]  # Go up from tools/ to SpectrueBack/
sys.path.insert(0, str(project_root))
from spectrue_api.verifier.local_embedder import LocalEmbedder

# --- Константы ---
DEFAULT_BATCH = 64
DEFAULT_THREADS = 8
DEFAULT_EMBED_WORKERS = 4
DEFAULT_MIN_WORDS = 30
EMBEDDER = None


# --- Функции-помощники ---

def read_titles_from_file(filepath: Path) -> List[str]:
    """Читает названия статей из файла, по одному на строку."""
    if not filepath.exists():
        print(f"❌ Файл со списком статей не найден: {filepath}")
        print(f"Сначала запустите 'collect_articles.py --lang {filepath.stem.split('_')[-1]}' для его создания.")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_paragraphs(text: str, min_words: int) -> List[str]:
    paras = re.split(r"\n{2,}", text)
    return [p.strip() for p in paras if len(p.split()) >= min_words]


def _fetch_page_content(title: str) -> Optional[str]:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        return None
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке статьи '{title}': {e}")
        return None


def _embed_batch_with_retry(batch: List[str], max_retries: int = 6) -> List[List[float]]:
    import time
    import random
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return EMBEDDER.embed(batch)
        except Exception:
            wait = delay * (1 + 0.25 * random.random())
            if attempt < max_retries - 1:
                print(f"⏳ Ошибка эмбеддинга: пауза {wait:.2f}s, попытка {attempt + 1}/{max_retries}")
                time.sleep(wait)
                delay *= 2
                continue
            raise


# --- Основные функции ---

def build_embeddings(lang: str, titles: List[str], outfile: Path, threads: int, embed_workers: int, batch_size: int,
                     min_words: int):
    wikipedia.set_lang(lang)

    paragraphs: List[str] = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(_fetch_page_content, t): t for t in titles}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Загрузка статей ({lang})"):
            content = fut.result()
            if content:
                paragraphs.extend(split_paragraphs(content, min_words))

    if not paragraphs:
        print("❌ Не удалось собрать ни одного подходящего параграфа.")
        return

    print(f"--- Собрано {len(paragraphs)} параграфов. Создание эмбеддингов... ---")

    batches = [paragraphs[i:i + batch_size] for i in range(0, len(paragraphs), batch_size)]
    embeds: List[Optional[List[List[float]]]] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=embed_workers) as pool:
        future_to_idx = {pool.submit(_embed_batch_with_retry, batch): idx for idx, batch in enumerate(batches)}
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Создание эмбеддингов"):
            idx = future_to_idx[fut]
            try:
                embeds[idx] = fut.result()
            except Exception as e:
                print(f"⚠️ Ошибка эмбеддинга в батче {idx}: {e}")

    flat_embeds: List[List[float]] = []
    valid_paragraphs: List[str] = []
    for i, batch_embeds in enumerate(embeds):
        if batch_embeds:
            flat_embeds.extend(batch_embeds)
            valid_paragraphs.extend(batches[i])

    if outfile.exists():
        print(f"--- Добавление к существующему файлу: {outfile} ---")
        try:
            old_data = np.load(outfile, allow_pickle=True)
            old_paragraphs = list(old_data["paragraphs"])
            old_embeds = list(old_data["embeds"])
            print(f"--- Загружено {len(old_paragraphs)} существующих параграфов. ---")

            seen_set = set(old_paragraphs)
            new_added = 0
            for p, e in zip(valid_paragraphs, flat_embeds):
                if p not in seen_set:
                    old_paragraphs.append(p)
                    old_embeds.append(e)
                    seen_set.add(p)
                    new_added += 1

            print(f"--- Добавлено {new_added} новых уникальных параграфов. ---")
            final_paragraphs, final_embeds = old_paragraphs, old_embeds
        except Exception as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось прочитать существующий файл, он будет перезаписан. Ошибка: {e}")
            final_paragraphs, final_embeds = valid_paragraphs, flat_embeds
    else:
        final_paragraphs, final_embeds = valid_paragraphs, flat_embeds

    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outfile,
        paragraphs=np.array(final_paragraphs, dtype=object),
        embeds=np.array(final_embeds, dtype="float32"),
    )
    print(f"✅ Сохранено {len(final_paragraphs)} параграфов в {outfile}")


def build_faiss_index(npz_path: Path):
    index_path = npz_path.with_name(npz_path.name.replace("paragraphs", "faiss")).with_suffix(".index")

    if not npz_path.exists():
        print(f"❌ Файл с эмбеддингами не найден: {npz_path}")
        return

    print(f"--- Создание FAISS-индекса для {npz_path.name} ---")
    with np.load(npz_path, allow_pickle=True) as data:
        embeds = data["embeds"]

    if embeds.shape[0] == 0:
        print("❌ Файл с эмбеддингами пуст. Пропуск создания FAISS-индекса.")
        return

    if embeds.dtype != 'float32':
        embeds = embeds.astype('float32')

    num_vectors, dimension = embeds.shape
    print(f"Найдено {num_vectors} векторов размерности {dimension}.")

    faiss.normalize_L2(embeds)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeds)

    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS-индекс сохранен в {index_path}. Всего векторов: {index.ntotal}")


def main():
    global EMBEDDER

    parser = argparse.ArgumentParser(description="Создание RAG-индекса из списка статей Википедии.")
    parser.add_argument("--lang", type=str, required=True, help="Код языка (например, 'ru', 'en')")
    parser.add_argument("--limit", type=int, default=1000, help="Сколько статей из списка обработать за один запуск.")
    parser.add_argument("--skip", type=int, default=0, help="Сколько статей из начала списка пропустить.")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--embed-workers", type=int, default=DEFAULT_EMBED_WORKERS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    args = parser.parse_args()

    list_filepath = Path(f"data/wiki_sets/master_list_{args.lang}.txt")
    npz_outfile = Path(f"data/wiki_sets/wiki_paragraphs_master_{args.lang}.npz")

    all_titles = read_titles_from_file(list_filepath)

    start_index = args.skip
    end_index = args.skip + args.limit
    titles_to_process = all_titles[start_index:end_index]

    if not titles_to_process:
        print("--- Нет статей для обработки. Пересоздаем FAISS-индекс на случай изменений. ---")
        build_faiss_index(npz_path=npz_outfile)
        return

    print(
        f"--- Будет обработано {len(titles_to_process)} статей (с {start_index + 1} по {end_index}) для языка '{args.lang}' ---")

    EMBEDDER = LocalEmbedder()

    build_embeddings(
        lang=args.lang, titles=titles_to_process, outfile=npz_outfile,
        threads=args.threads, embed_workers=args.embed_workers,
        batch_size=args.batch, min_words=args.min_words
    )

    build_faiss_index(npz_path=npz_outfile)

    print(f"\n--- ✨ Готово! Чтобы обработать следующую порцию, запустите с --skip {end_index} ---")


if __name__ == "__main__":
    main()