import os, time, traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from build_index import load_index, build_index
from rag_engine import RAGEngine

app = Flask(__name__)
rag_engine = None
store_stats = {}
index_status = "not_loaded"


def initialize_rag():
    global rag_engine, store_stats, index_status
    try:
        print("Loading RAG index...")
        engine, store = load_index()
        rag_engine = RAGEngine(engine, store, top_k=6, use_mmr=True)
        store_stats = store.stats()
        index_status = "ready"
        print(f"RAG ready | {store_stats}")
    except FileNotFoundError:
        index_status = "not_indexed"
        print("Index not found. Run: python build_index.py --pdf /path/to/Annual-Report-FY-2023-24.pdf")
    except Exception:
        index_status = "error"
        print("Error loading index:")
        traceback.print_exc()


@app.route("/")
def home():
    safe_stats = store_stats if store_stats else {"n_chunks": 0, "embedding_dim": 0, "size_mb": 0}
    return render_template("index.html", status=index_status, stats=safe_stats)


@app.route("/api/query", methods=["POST"])
def query():
    if index_status != "ready":
        return jsonify({"error": f"RAG index not ready (status: {index_status})"}), 503

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        t0 = time.time()
        result = rag_engine.query(question)
        elapsed = round(time.time() - t0, 2)

        chunks = [{
            "text": c["text"],
            "source": c["source"],
            "page_num": c["page_num"],
            "score": round(c["score"], 4),
            "chunk_id": c["chunk_id"]
        } for c in result["retrieved_chunks"]]

        return jsonify({
            "question": question,
            "answer": result["answer"],
            "retrieved_chunks": chunks,
            "n_chunks": result["n_chunks_retrieved"],
            "time_seconds": elapsed
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Something went wrong while querying."}), 500


@app.route("/api/status")
def api_status():
    return jsonify({"status": index_status, "stats": store_stats})


@app.route("/api/index", methods=["POST"])
def api_build_index():
    data = request.get_json(silent=True) or {}
    pdf_path = data.get("pdf_path", "")
    if not pdf_path or not Path(pdf_path).exists():
        return jsonify({"error": f"PDF not found at: {pdf_path}"}), 400

    global rag_engine, store_stats, index_status
    try:
        index_status = "building"
        engine, store = build_index(pdf_path, force_rebuild=True)
        rag_engine = RAGEngine(engine, store, top_k=6, use_mmr=True)
        store_stats = store.stats()
        index_status = "ready"
        return jsonify({"status": "success", "stats": store_stats})
    except Exception:
        index_status = "error"
        traceback.print_exc()
        return jsonify({"error": "Failed to build index."}), 500


if __name__ == "__main__":
    initialize_rag()
    print(f"Status: {index_status}")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))