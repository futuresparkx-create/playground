# memory/memory.py
# Real LanceDB Memory Layer

import lancedb
from sentence_transformers import SentenceTransformer

class MemoryStore:
    """
    Safe LanceDB memory with embeddings.
    Stores episodes, operational data, and semantic vectors.
    Never auto-invokes actions.
    """

    def __init__(self, path: str = "./lancedb"):
        self.db = lancedb.connect(path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Ensure tables exist
        self.episodes = self.db.open_table("episodes", create_if_missing=True)
        self.semantic = self.db.open_table("semantic", create_if_missing=True)
        self.ops = self.db.open_table("ops", create_if_missing=True)

    def log_episode(self, data: dict):
        self.episodes.add([data])

    def store_semantic(self, embedding, content: dict):
        row = {"vector": embedding.tolist(), "content": content}
        self.semantic.add([row])

    def store_ops(self, config: dict):
        self.ops.add([config])

    def embed(self, text: str):
        return self.embedder.encode(text)

    def fetch_semantic(self, query: str, k: int = 5):
        e = self.embed(query)
        results = self.semantic.search(e).limit(k).to_list()
        return results