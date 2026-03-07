import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a Question Answering assistant for the Swiggy Annual Report.
Answer strictly using the provided context. If answer is not in the context, say:
"This information is not available in the provided sections of the Swiggy Annual Report."
Be concise and factual. Quote page numbers if available."""

class RAGEngine:
    def __init__(self, embedding_engine, vector_store, top_k=6, use_mmr=True, min_score=0.05):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.min_score = min_score

    def retrieve(self, query):
        q_emb = self.embedding_engine.transform([query])
        if self.use_mmr:
            results = self.vector_store.search_mmr(q_emb[0], top_k=self.top_k)
        else:
            results = self.vector_store.search(q_emb[0], top_k=self.top_k)
        return [r for r in results if r['score'] >= self.min_score]

    def build_context(self, chunks):
        if not chunks:
            return "No relevant context found in the Swiggy Annual Report."
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Excerpt {i} | {c['source']} | Score: {c['score']:.3f}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def generate_answer(self, question, context):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")

        user_msg = f"""Context from Swiggy Annual Report:
{context}

QUESTION: {question}

Answer only using the context above."""

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }

        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Groq response format: {data}")

    def query(self, question):
        print(f"\n[RAG] Query: {question[:80]}...")
        retrieved = self.retrieve(question)
        print(f"[RAG] Retrieved {len(retrieved)} chunks")
        context = self.build_context(retrieved)
        print("[RAG] Generating answer via Groq API...")
        answer = self.generate_answer(question, context)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "context": context,
            "n_chunks_retrieved": len(retrieved)
        }