import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import pickle


class EmbeddingEngine:
    """
    Simple TF-IDF + SVD embedding engine for document vectors.
    Produces dense embeddings using LSA (no GPU required).
    """
    
    def __init__(self, n_components=256):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[a-zA-Z0-9][a-zA-Z0-9\-\.]*\b'
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False
    
    def fit_transform(self, texts):
        print(f"Fitting TF-IDF on {len(texts)} chunks...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF shape: {tfidf_matrix.shape}")

        max_comp = min(tfidf_matrix.shape) - 1
        if self.n_components > max_comp:
            print(f"Adjusted SVD components to {max_comp}")
            self.svd = TruncatedSVD(n_components=max_comp, random_state=42)
        
        embeddings = self.svd.fit_transform(tfidf_matrix)
        embeddings = normalize(embeddings, norm='l2')
        explained = self.svd.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained:.2%}")
        print(f"Embeddings shape: {embeddings.shape}")

        self.is_fitted = True
        return embeddings
    
    def transform(self, texts):
        if not self.is_fitted:
            raise RuntimeError("Fit the engine before calling transform.")
        tfidf_matrix = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf_matrix)
        return normalize(embeddings, norm='l2')
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'svd': self.svd,
                'n_components': self.n_components,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Engine saved to {path}")
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        engine = cls(n_components=state['n_components'])
        engine.vectorizer = state['vectorizer']
        engine.svd = state['svd']
        engine.is_fitted = state['is_fitted']
        print(f"Engine loaded from {path}")
        return engine