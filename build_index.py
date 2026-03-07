import argparse
import time
from pathlib import Path
from pdf_processor import process_pdf, save_chunks, load_chunks
from embeddings import EmbeddingEngine
from vector_store import VectorStore

DATA_DIR = Path(__file__).parent / "data"


def build_index(pdf_path, chunk_size=600, overlap=100, n_components=256, force_rebuild=False):
    DATA_DIR.mkdir(exist_ok=True)
    
    chunks_path = DATA_DIR / "chunks.json"
    embeddings_path = DATA_DIR / "embedding_engine.pkl"
    vector_store_path = DATA_DIR / "vector_store.pkl"
    
    # PDF processing
    if chunks_path.exists() and not force_rebuild:
        print("Loading cached chunks...")
        chunks = load_chunks(str(chunks_path))
        print(f"Loaded {len(chunks)} chunks")
    else:
        print("Processing PDF...")
        t0 = time.time()
        chunks = process_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
        save_chunks(chunks, str(chunks_path))
        print(f"PDF processed in {time.time()-t0:.1f}s | {len(chunks)} chunks")
    
    if not chunks:
        raise ValueError("No chunks found. Check the PDF path.")
    
    # Generate embeddings
    print("Generating embeddings...")
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    engine = EmbeddingEngine(n_components=n_components)
    embeddings = engine.fit_transform(texts)
    engine.save(str(embeddings_path))
    print(f"Embeddings done in {time.time()-t0:.1f}s | shape={embeddings.shape}")
    
    # Build vector store
    print("Building vector store...")
    t0 = time.time()
    store = VectorStore()
    store.add(chunks, embeddings)
    store.save(str(vector_store_path))
    print(f"Vector store built in {time.time()-t0:.1f}s")
    
    stats = store.stats()
    print("="*40)
    print("Index build complete")
    print(f"  Chunks:        {stats['n_chunks']}")
    print(f"  Embedding dim: {stats['embedding_dim']}")
    print(f"  Index size:    {stats['size_mb']} MB")
    print("="*40)
    
    return engine, store


def load_index():
    embeddings_path = DATA_DIR / "embedding_engine.pkl"
    vector_store_path = DATA_DIR / "vector_store.pkl"
    
    if not embeddings_path.exists() or not vector_store_path.exists():
        raise FileNotFoundError("Index not found. Run build_index.py first.")
    
    engine = EmbeddingEngine.load(str(embeddings_path))
    store = VectorStore.load(str(vector_store_path))
    return engine, store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG index from Swiggy Annual Report PDF")
    parser.add_argument("--pdf", required=True, help="Path to the Swiggy Annual Report PDF")
    parser.add_argument("--chunk-size", type=int, default=600, help="Characters per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks")
    parser.add_argument("--components", type=int, default=256, help="LSA embedding dimensions")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache exists")
    
    args = parser.parse_args()
    
    if not Path(args.pdf).exists():
        print(f"ERROR: PDF not found at: {args.pdf}")
        exit(1)
    
    build_index(
        pdf_path=args.pdf,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        n_components=args.components,
        force_rebuild=args.force
    )