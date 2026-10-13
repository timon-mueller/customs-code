import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticIndex:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.records: list[dict] = []

    def build(self, records: list[dict]) -> None:
        """Encode all records and store normalized embeddings."""
        self.records = records
        descriptions = [r["description"] for r in records]
        self.embeddings = self.model.encode(
            descriptions,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=256,
        )

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(index_dir, "records.json"), "w", encoding="utf-8") as f:
            json.dump(self.records, f)

    def load(self, index_dir: str) -> None:
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "records.json"), encoding="utf-8") as f:
            self.records = json.load(f)

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """Return top-k (record, cosine_similarity) pairs for a query."""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores = (query_emb @ self.embeddings.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.records[i], float(scores[i])) for i in top_indices]

    def predict(self, query: str, top_k: int = 10) -> tuple[str, float, list[tuple[dict, float]]]:
        """
        Predict the customs chapter for a query.

        Returns (chapter, confidence, top_k_matches).
        Confidence is the mean cosine similarity of the winning chapter's matches.
        """
        matches = self.search(query, top_k=top_k)

        # Aggregate similarity scores per chapter
        chapter_scores: dict[str, list[float]] = {}
        for record, score in matches:
            ch = record["chapter"]
            chapter_scores.setdefault(ch, []).append(score)

        # Pick the chapter with the highest total similarity
        best_chapter = max(chapter_scores, key=lambda ch: sum(chapter_scores[ch]))
        confidence = float(np.mean(chapter_scores[best_chapter]))

        return best_chapter, confidence, matches
