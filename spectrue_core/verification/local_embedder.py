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

from fastembed import TextEmbedding
from typing import List


class LocalEmbedder:
    """
    Обертка для работы с локальной моделью эмбеддингов через fastembed.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Инициализирует модель. Модель скачивается при первом запуске.
        """
        self.model_name = model_name
        self.model = None

    def _get_model(self):
        if self.model is None:
            print(f"--- Loading local embedding model: {self.model_name} (once)... ---")
            self.model = TextEmbedding(model_name=self.model_name)
            print("--- Local embedding model loaded. ---")
        return self.model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов.
        """
        if not texts:
            return []

        try:
            # .embed() возвращает numpy array, конвертируем его в обычный список
            model = self._get_model()
            embeddings = model.embed(texts)
            return [arr.tolist() for arr in embeddings]
        except Exception as e:
            print(f"Error during local embedding: {e}")
            return []