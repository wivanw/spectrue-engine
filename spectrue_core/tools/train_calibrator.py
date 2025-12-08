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

import pandas as pd
import json
import pickle
import asyncio  # <-- Импортируем asyncio
from tqdm import tqdm  # tqdm для красивого прогресс-бара

from sklearn.isotonic import IsotonicRegression
from spectrue_api.core.spectrue_core import SpectrueCore


async def get_raw_score(verifier, text: str, semaphore: asyncio.Semaphore):
    """
    Асинхронная обертка для получения оценки.
    Использует семафор для ограничения одновременных запросов.
    """
    async with semaphore:
        result_dict = await verifier.verify_fact(
            text,
            "advanced",
            "gpt-5-nano",
            "ru"
        )
        # Возвращаем только итоговую оценку из словаря
        return result_dict.get('final_score', 0.5)


async def main():
    """
    Основная асинхронная функция, которая выполняет всю работу.
    """
    print("Загрузка сэмплов из data/ground_truth.jsonl...")
    try:
        with open('data/ground_truth.jsonl', 'r', encoding='utf-8') as f:
            rows = [json.loads(l) for l in f]
        df = pd.DataFrame(rows)
        print(f"Загружено {len(df)} сэмплов.")
    except FileNotFoundError:
        print("❌ Файл data/ground_truth.jsonl не найден. Завершение.")
        return

    spectrue = SpectrueCore(nlp_models={})
    texts_to_process = df["text"].tolist()

    # --- Ключевые изменения для асинхронности ---

    # 1. Семафор: Ограничитель одновременных запросов к API.
    #    Это нужно, чтобы не получить бан от OpenAI за слишком частые запросы.
    #    Значение 15 означает, что не более 15 запросов будут выполняться ОДНОВРЕМЕННО.
    CONCURRENT_REQUESTS = 15
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # 2. Создаем список асинхронных задач
    tasks = [get_raw_score(spectrue.verifier, text, semaphore) for text in texts_to_process]

    # 3. Запускаем задачи и получаем результаты с прогресс-баром
    print(f"Получение оценок для {len(texts_to_process)} фраз (параллельно по {CONCURRENT_REQUESTS})...")
    raw_scores = []
    # tqdm(asyncio.as_completed(tasks)) позволяет обновлять прогресс-бар
    # по мере завершения КАЖДОЙ задачи, а не ждать, пока все завершатся.
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await future
        raw_scores.append(result)

    # --- Конец изменений ---

    print("\nОбучение калибратора...")
    df["raw_score"] = raw_scores

    # Убедимся, что нет NaN/inf значений
    df.dropna(subset=['raw_score', 'verified'], inplace=True)

    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(df["raw_score"], df["verified"])

    # Сохраняем модель калибратора
    output_path = "data/iso_calibrator.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(iso_calibrator, f)

    print(f"✅ Калибратор обучен и сохранен в {output_path}")


if __name__ == "__main__":
    # Запускаем основную асинхронную функцию
    asyncio.run(main())