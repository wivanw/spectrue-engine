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

import faiss
import numpy as np
from pathlib import Path
from spectrue_core.verification.local_embedder import LocalEmbedder
from spectrue_core.config import SpectrueConfig
import threading
from google.cloud import storage

_rag_verifier_instance = None
_rag_verifier_lock = threading.Lock()


def get_rag_verifier(config: SpectrueConfig = None):
    global _rag_verifier_instance
    with _rag_verifier_lock:
        if _rag_verifier_instance is None:
            if config is None:
                # Fallback for legacy calls or tests without config
                # In production, this should always be called with config first
                config = SpectrueConfig() 
            _rag_verifier_instance = RAGVerifier(config)
    return _rag_verifier_instance


class RAGVerifier:
    def __init__(self, config: SpectrueConfig):
        self.config = config
        
        if config.project_root:
             self.project_root = Path(config.project_root)
        else:
            # Default to repo root if not specified (assuming standard layout)
            # spectrue-core-engine/spectrue_core/verification/rag_verifier.py -> spectrue-core-engine/
            self.project_root = Path(__file__).resolve().parents[3]
            
        self.embedder = LocalEmbedder()

        self.indexes = {}
        self.paragraphs = {}
        self._lock = threading.Lock()

        # Init GCS
        self.gcs_bucket_name = config.gcs_bucket_name
        self.storage_client = None
        self.gcs_bucket = None

    def _get_gcs_bucket(self):
        if self.gcs_bucket is None and self.gcs_bucket_name:
            print(f"--- Connecting to GCS bucket: {self.gcs_bucket_name} ---")
            self.storage_client = storage.Client()
            self.gcs_bucket = self.storage_client.bucket(self.gcs_bucket_name)
        return self.gcs_bucket

    def _download_from_gcs(self, lang: str):
        """
        Скачивает файлы индекса для языка из GCS, если они там есть.
        Пробует искать в папке data/ и в корне бакета.
        """
        gcs_bucket = self._get_gcs_bucket()
        if not gcs_bucket:
            print("--- GCS_BUCKET_NAME not set. RAG will only use local files. ---")
            return False

        print(f"--- Checking GCS for index files for language '{lang}'... ---")

        # Локальная папка для хранения
        local_dir = self.project_root / "data" / "wiki_sets"
        local_dir.mkdir(parents=True, exist_ok=True)

        files_to_download = {
            f"wiki_faiss_master_{lang}.index": local_dir / f"wiki_faiss_master_{lang}.index",
            f"wiki_paragraphs_master_{lang}.npz": local_dir / f"wiki_paragraphs_master_{lang}.npz"
        }

        all_success = True
        try:
            for filename, local_path in files_to_download.items():
                if not local_path.exists():
                    found = False
                    # Try prefixes: data/ then root
                    for prefix in ["data/", ""]:
                        gcs_path = f"{prefix}{filename}"
                        blob = gcs_bucket.blob(gcs_path)
                        if blob.exists():
                            print(f"Downloading {gcs_path} to {local_path}...")
                            blob.download_to_filename(str(local_path))
                            print("Download complete.")
                            found = True
                            break
                    
                    if not found:
                        print(f"File {filename} not found in GCS (checked 'data/' and root).")
                        print("--- Debug: Listing files in bucket (first 20) ---")
                        try:
                            blobs = list(gcs_bucket.list_blobs(max_results=20))
                            for b in blobs:
                                print(f" - {b.name}")
                        except Exception as list_err:
                            print(f"Failed to list blobs: {list_err}")
                        print("--- End Debug ---")
                        all_success = False
            return all_success
        except Exception as e:
            print(f"Failed to download from GCS: {e}")
            return False

    def _load_index_for_lang(self, lang: str):
        """
        Загружает индекс и параграфы для указанного языка.
        Если файлы отсутствуют локально, пытается скачать их из GCS.
        """
        with self._lock:
            if lang in self.indexes:
                return

            local_index_path = self.project_root / "data" / "wiki_sets" / f"wiki_faiss_master_{lang}.index"
            local_paragraphs_path = self.project_root / "data" / "wiki_sets" / f"wiki_paragraphs_master_{lang}.npz"

            # Если файлов нет локально, пытаемся скачать из GCS
            if not local_index_path.exists() or not local_paragraphs_path.exists():
                self._download_from_gcs(lang)

            # После попытки скачивания, снова проверяем наличие
            if not local_index_path.exists() or not local_paragraphs_path.exists():
                print(f"ПРЕДУПРЕЖДЕНИЕ: Файлы индекса для языка '{lang}' не найдены ни локально, ни в GCS.")
                self.indexes[lang] = None
                self.paragraphs[lang] = []
                return

            try:
                print(f"--- Loading RAG index for '{lang}' from local files... ---")
                self.indexes[lang] = faiss.read_index(str(local_index_path))
                npz_data = np.load(local_paragraphs_path, allow_pickle=True, mmap_mode='r')
                self.paragraphs[lang] = npz_data["paragraphs"]
                print(f"--- RAG index for '{lang}' loaded successfully. ---")
            except Exception as e:
                print(f"ОШИБКА при загрузке индекса для '{lang}': {e}")
                self.indexes[lang] = None
                self.paragraphs[lang] = []

    def is_ready(self, lang: str) -> bool:
        if lang not in self.indexes:
            self._load_index_for_lang(lang)

        return self.indexes.get(lang) is not None and len(self.paragraphs.get(lang, [])) > 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts: return []
        try:
            return self.embedder.embed(texts)
        except Exception as e:
            print(f"Ошибка при получении эмбеддингов: {e}")
            return []

    def verify(self, fact: str, lang: str, top_k: int = 3):
        if not self.is_ready(lang):
            return None

        index = self.indexes[lang]
        paragraphs = self.paragraphs[lang]
        embeds = self._embed([fact])
        if not embeds:
            return None

        X = np.array(embeds, dtype="float32")

        faiss.normalize_L2(X)
        if not isinstance(index, faiss.IndexFlatIP):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Индекс для языка '{lang}' не является IndexFlatIP.")

        distances, indices = index.search(X, top_k)

        results: list[dict] = []
        for row in range(len([fact])):
            idxs = indices[row] if row < indices.shape[0] else []
            dists = distances[row] if row < distances.shape[0] else []
            if len(idxs) == 0 or idxs[0] == -1:
                results.append({"score": 0.5, "relevance": 0.0, "sources": []})
                continue

            relevance_scores = [max(0.0, min(float(d), 1.0)) for d in dists]
            avg_relevance = (sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 0.0

            valid_pairs = [(i, paragraphs[i]) for i in idxs if i != -1]
            sources = [{
                "link": f"wiki_rag_source_{i}",
                "snippet": p,
                "title": f"Wikipedia ({lang.upper()})",
                "origin": "RAG"
            } for i, p in valid_pairs]

            results.append({"relevance": avg_relevance, "sources": sources})

        return results[0] if results else None