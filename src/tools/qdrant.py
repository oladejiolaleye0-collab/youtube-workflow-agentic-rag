import os
import time
import uuid
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Environment / config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pplx-embed-v1-4b")

# Collection focused on workflow/guide cache
QUERY_CACHE_COLLECTION = "workflow_query_cache"
EMBED_DIM = 3416  # match pplx-embed-v1-4b output dim
SIMILARITY_THRESHOLD = 0.93


class QdrantTool:
    """
    Semantic cache for workflow/guide objects, backed by Qdrant + Perplexity embeddings.

    Public API (kept same as before so HybridRAGAgent works unchanged):
      - add_guide(content: str, metadata: dict)
      - search_similar(query: str, limit: int = 3) -> List[str]
    """

    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )
        self._ensure_collection()
        print("✅ QdrantTool: using Qdrant semantic cache")

    # ---------- internal helpers ----------

    def _ensure_collection(self) -> None:
        """Ensure the semantic cache collection exists with correct vector config."""
        try:
            self.client.get_collection(QUERY_CACHE_COLLECTION)
        except Exception:
            # Create collection if missing (ok to recreate since this is just a cache)
            self.client.recreate_collection(
                collection_name=QUERY_CACHE_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBED_DIM,
                    distance=Distance.COSINE,
                ),
            )

    def _embed(self, text: str) -> List[float]:
        """
        Get an embedding from Perplexity's pplx-embed-v1-4b.
        Returns a list[float] of length EMBED_DIM.
        """
        if not PERPLEXITY_API_KEY:
            raise RuntimeError("PERPLEXITY_API_KEY is not set in environment")

        url = "https://api.perplexity.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        embedding = data["data"][0]["embedding"]
        if len(embedding) != EMBED_DIM:
            raise ValueError(
                f"Embedding dim {len(embedding)} != expected {EMBED_DIM}. "
                "Check EMBEDDING_MODEL and Qdrant collection config."
            )
        return embedding

    # ---------- public API used by HybridRAGAgent ----------

    def add_guide(self, content: str, metadata: dict) -> None:
        """
        Store a combined workflow/guide in the semantic cache.

        `content` can be a plain text guide or a serialized workflow (e.g. JSON string).
        """
        vector = self._embed(content)

        payload: Dict[str, Any] = {
            "content": content,
            "task": metadata.get("task", "unknown"),
            "source": metadata.get("source", "youtube_workflow"),
            "video_url": metadata.get("video_url"),
            "timestamp": metadata.get("timestamp", time.time()),
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload,
        )

        self.client.upsert(
            collection_name=QUERY_CACHE_COLLECTION,
            points=[point],
        )
        print(f"💾 QdrantTool: cached guide for task='{payload['task']}'")

    def search_similar(self, query: str, limit: int = 3) -> List[str]:
        """
        Semantic cache lookup by query/task string.
        Returns a list of guide contents (strings).
        """
        vector = self._embed(query)

        res = self.client.query_points(
            collection_name=QUERY_CACHE_COLLECTION,
            query=vector,
            with_payload=True,
            limit=limit,
        )

        results: List[str] = []
        for point in res.points:
            score = point.score
            if score is not None and score < SIMILARITY_THRESHOLD:
                continue
            payload = point.payload or {}
            content = payload.get("content")
            if content:
                results.append(content)

        return results


if __name__ == "__main__":
    # Simple manual test: run `python qdrant.py` after setting env vars
    q = QdrantTool()
    q.add_guide(
        "1. Power off PS5 2. Remove screws 3. Clean old paste 4. Apply new paste",
        {"task": "ps5 thermal paste", "source": "demo"},
    )
    res = q.search_similar("how to replace thermal paste on PS5")
    print(f"Found {len(res)} cached guides:")
    for i, r in enumerate(res):
        print(f"{i}: {r[:80]}...")