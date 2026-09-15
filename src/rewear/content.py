"""Phase 2: content-based signals for cold-start and the demo.

Two embedding backends, chosen at runtime:

  "st"    sentence-transformers (all-MiniLM-L6-v2 by default) on the product text.
          Better semantics; needs the model download once (~90MB).
  "tfidf" scikit-learn TF-IDF + TruncatedSVD. No downloads, runs anywhere, and is
          the CI backend. Surprisingly decent on H&M because descriptions are formulaic.

Image embeddings (CLIP) live in `image_embeddings()` and are optional —
they're the thing that makes the demo look good, not what moves MAP@12.

Usage:
    emb = ContentEmbedder(backend="tfidf").fit(articles)
    similar = emb.most_similar("0706016001", k=10)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

TEXT_COLS = ["prod_name", "product_type_name", "graphical_appearance_name", "colour_group_name", "department_name", "detail_desc"]


def article_text(articles: pd.DataFrame) -> pd.Series:
    """Concatenate the descriptive columns into one string per article."""
    cols = [c for c in TEXT_COLS if c in articles.columns]
    return articles[cols].fillna("").astype(str).agg(". ".join, axis=1)


class ContentEmbedder:
    def __init__(self, backend: str = "tfidf", dim: int = 128, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if backend not in {"tfidf", "st"}:
            raise ValueError("backend must be 'tfidf' or 'st'")
        self.backend = backend
        self.dim = dim
        self.model_name = model_name
        self.index: pd.Index | None = None
        self.vectors: np.ndarray | None = None

    def fit(self, articles: pd.DataFrame) -> ContentEmbedder:
        texts = article_text(articles).tolist()
        self.index = pd.Index(articles["article_id"].astype(str), name="article_id")

        if self.backend == "st":
            from sentence_transformers import SentenceTransformer  # lazy import: optional dependency

            model = SentenceTransformer(self.model_name)
            vecs = model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        else:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
            n_comp = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
            vecs = TruncatedSVD(n_components=max(n_comp, 2), random_state=0).fit_transform(tfidf)
            vecs = normalize(vecs)

        self.vectors = np.asarray(vecs, dtype=np.float32)
        return self

    def most_similar(self, article_id: str, k: int = 10) -> list[tuple[str, float]]:
        assert self.vectors is not None, "call fit() first"
        i = self.index.get_loc(article_id)
        sims = self.vectors @ self.vectors[i]
        sims[i] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.index[j], float(sims[j])) for j in top]

    def user_vector(self, article_ids: list[str], weights: np.ndarray | None = None) -> np.ndarray:
        """Taste vector = weighted mean of the embeddings of what the user bought."""
        pos = self.index.get_indexer(article_ids)
        pos = pos[pos >= 0]
        if len(pos) == 0:
            return np.zeros(self.vectors.shape[1], dtype=np.float32)
        w = np.ones(len(pos)) if weights is None else np.asarray(weights)[: len(pos)]
        v = (self.vectors[pos] * w[:, None]).sum(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def recommend_for_history(self, article_ids: list[str], k: int = 12, exclude_owned: bool = True) -> list[str]:
        """Content-only recommendations: nearest items to the user's taste vector."""
        u = self.user_vector(article_ids)
        sims = self.vectors @ u
        if exclude_owned:
            owned = self.index.get_indexer(article_ids)
            sims[owned[owned >= 0]] = -np.inf
        top = np.argsort(-sims)[:k]
        return [self.index[j] for j in top]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.vectors)
        pd.Series(self.index).to_csv(path.with_suffix(".ids.csv"), index=False)


def image_embeddings(articles: pd.DataFrame, images_dir: str | Path, model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 64) -> np.ndarray:
    """CLIP image embeddings for every article with an image on disk.

    Kaggle stores images at images/<first 3 digits>/<article_id>.jpg. Missing
    images get a zero vector so the array lines up with `articles`.
    Optional dependency: pip install torch transformers pillow
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    images_dir = Path(images_dir)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)

    ids = articles["article_id"].astype(str).tolist()
    out = np.zeros((len(ids), model.config.projection_dim), dtype=np.float32)
    batch, batch_pos = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            inputs = proc(images=batch, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
        out[batch_pos] = feats
        batch.clear()
        batch_pos.clear()

    for i, aid in enumerate(ids):
        p = images_dir / aid[:3] / f"{aid}.jpg"
        if p.exists():
            batch.append(Image.open(p).convert("RGB"))
            batch_pos.append(i)
            if len(batch) >= batch_size:
                flush()
    flush()
    return out
