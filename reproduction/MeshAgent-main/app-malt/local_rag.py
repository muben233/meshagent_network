"""
Local constraint retriever — FAISS + OpenAI text-embedding-ada-002.
Matches the paper's Azure Cognitive Search logic with ada-002.
"""

import json
import os
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import faiss

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com"),
        )
    return _client


class ConstraintRetriever:
    def __init__(self, constraints_path: Path):
        with open(constraints_path, "r") as f:
            self.constraints = json.load(f)

        self.texts = [c["constraint"] for c in self.constraints]
        self.n = len(self.texts)

        client = _get_client()
        print(f"[local_rag] Embedding {self.n} constraints with ada-002 ...")
        embeddings = []
        for text in self.texts:
            r = client.embeddings.create(model="text-embedding-ada-002", input=text)
            embeddings.append(r.data[0].embedding)

        emb = np.array(embeddings).astype(np.float32)
        # Normalize for cosine similarity
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        print(f"[local_rag] Index ready: {self.n} constraints, dim={self.dim}")

    def retrieve(self, query: str, top_k: int = 9) -> str:
        client = _get_client()
        r = client.embeddings.create(model="text-embedding-ada-002", input=query)
        q = np.array([r.data[0].embedding]).astype(np.float32)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        scores, indices = self.index.search(q, min(top_k, self.n))
        selected = [self.texts[int(indices[0][i])] for i in range(min(top_k, self.n))]
        return " ".join(selected)


_retriever = None

def retrieve_constraints(query: str, top_k: int = 9) -> str:
    global _retriever
    if _retriever is None:
        p = Path(__file__).parent / "data" / "rag_constraints.json"
        _retriever = ConstraintRetriever(p)
    return _retriever.retrieve(query, top_k)
