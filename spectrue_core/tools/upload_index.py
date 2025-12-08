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
import os
from pathlib import Path
from google.cloud import storage

def upload_to_gcs(lang: str, bucket_name: str):
    """
    Загружает созданные файлы индекса для указанного языка в Google Cloud Storage.
    Локально берет из 'data/wiki_sets', в облако кладет в 'data'.
    """
    print(f"--- Загрузка артефактов для языка '{lang}' в GCS бакет '{bucket_name}'... ---")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Словарь: локальный путь -> путь в GCS
        files_to_upload = {
            Path(f"data/wiki_sets/wiki_paragraphs_master_{lang}.npz"): f"data/wiki_paragraphs_master_{lang}.npz",
            Path(f"data/wiki_sets/wiki_faiss_master_{lang}.index"): f"data/wiki_faiss_master_{lang}.index"
        }

        for local_path, gcs_path in files_to_upload.items():
            if local_path.exists():
                blob = bucket.blob(gcs_path)
                print(f"Uploading {local_path} to {gcs_path}...")
                blob.upload_from_filename(str(local_path))
                print("Upload complete.")
            else:
                print(f"ОШИБКА: Локальный файл не найден, пропуск загрузки: {local_path}")
                print("Убедитесь, что вы сначала создали индекс с помощью 'build_index_from_list.py'.")
                return

    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить файлы в GCS. Убедитесь, что вы авторизованы (`gcloud auth application-default login`). Ошибка: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка файлов RAG-индекса в Google Cloud Storage.")
    parser.add_argument("--lang", type=str, required=True, help="Код языка для загрузки (например, 'ru', 'en')")
    args = parser.parse_args()

    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if gcs_bucket_name:
        upload_to_gcs(lang=args.lang, bucket_name=gcs_bucket_name)
    else:
        print("ОШИБКА: Переменная окружения GCS_BUCKET_NAME не задана. Некуда загружать.")