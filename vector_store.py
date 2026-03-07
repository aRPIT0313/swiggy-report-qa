"""Vector Store for Swiggy RAG - Pure numpy cosine similarity search."""
import numpy as np
import pickle
from typing import Optional, List, Dict


class VectorStore:
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Dict] = []
        self.n_chunks: int = 0
        self.dim: int = 0

    def add(self, chunks: List[Dict], embeddings: np.ndarray):
        assert len(chunks) == len(embeddings)
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.n_chunks = len(chunks)
        self.dim = embeddings.shape[1]
        print(f"[VectorStore] Added {self.n_chunks} chunks | dim={self.dim}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.embeddings is None or self.n_chunks == 0:
            return []
        q = query_embedding.flatten().astype(np.float32)
        scores = self.embeddings @ q
        k = min(top_k, self.n_chunks)
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        return [dict(self.chunks[i], score=float(scores[i])) for i in top_indices]

    def search_mmr(self, query_embedding: np.ndarray, top_k: int = 5, lambda_param: float = 0.7) -> List[Dict]:
        candidates = self.search(query_embedding, top_k=min(top_k * 4, self.n_chunks))
        if len(candidates) <= top_k:
            return candidates

        # Build candidate embedding matrix by matching chunk_ids
        chunk_id_to_idx = {c['chunk_id']: i for i, c in enumerate(self.chunks)}
        cand_embs = np.array([
            self.embeddings[chunk_id_to_idx[c['chunk_id']]] for c in candidates
        ], dtype=np.float32)

        selected_idxs = [0]
        selected_embs = [cand_embs[0]]
        remaining = list(range(1, len(candidates)))

        while len(selected_idxs) < top_k and remaining:
            sel_mat = np.array(selected_embs, dtype=np.float32)
            best_i, best_score = None, -np.inf
            for ri in remaining:
                rel = candidates[ri]['score']
                red = float(np.max(cand_embs[ri] @ sel_mat.T))
                score = lambda_param * rel - (1 - lambda_param) * red
                if score > best_score:
                    best_score, best_i = score, ri
            selected_idxs.append(best_i)
            selected_embs.append(cand_embs[best_i])
            remaining.remove(best_i)

        return [candidates[i] for i in selected_idxs]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings, 'chunks': self.chunks,
                         'n_chunks': self.n_chunks, 'dim': self.dim}, f)
        print(f"[VectorStore] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        store = cls()
        store.embeddings = state['embeddings']
        store.chunks = state['chunks']
        store.n_chunks = state['n_chunks']
        store.dim = state['dim']
        print(f"[VectorStore] Loaded {store.n_chunks} chunks from {path}")
        return store

    def stats(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "embedding_dim": self.dim,
            "size_mb": round(self.embeddings.nbytes / 1024 / 1024, 2) if self.embeddings is not None else 0
        }
