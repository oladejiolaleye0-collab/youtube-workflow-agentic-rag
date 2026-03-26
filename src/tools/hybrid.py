import os
import sys
from datetime import datetime
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.youtube import YoutubeTool
from tools.qdrant import QdrantTool


class HybridRAGAgent:
    """
    Lightweight YouTube → workflow RAG agent.

    - Treats the incoming `task` as either:
        * a YouTube URL, or
        * a natural-language description (still cached semantically)
    - Uses Qdrant as a semantic cache of workflows/guides.
    """

    def __init__(self):
        self.youtube = YoutubeTool()
        self.qdrant = QdrantTool()
        print("✅ HybridRAGAgent: YouTube workflow mode")

    def process_task(self, task: str) -> Dict[str, Any]:
        print(f"\n🤖 CRONY: Processing '{task}'")

        # 1) Try semantic cache first
        cached = self.qdrant.search_similar(task)
        if cached:
            print("📚 CACHE HIT! (50ms)")
            return {
                "answer": self._format_cached_guides(cached),
                "sources": cached,
                "type": "cache",
                "latency": "INSTANT",
            }

        # 2) Fallback: fetch from YouTube + build workflow
        print("🔍 CACHE MISS - YouTube live fetch...")
        return self._live_fetch_and_cache(task)

    # ---------- internal helpers ----------

    def _live_fetch_and_cache(self, task: str) -> Dict[str, Any]:
        """
        Interpret `task` as a YouTube URL, fetch transcript, extract workflow, then cache.
        """
        from tools.workflow_extractor import WorkflowExtractor  # lazy import

        video_url = task.strip()
        chunks = self.youtube.fetch_chunks(video_url)
        if not chunks:
            return {
                "answer": "No transcript or content found for this video.",
                "type": "youtube_error",
                "sources": [],
                "source_count": 0,
                "latency": "N/A",
            }

        extractor = WorkflowExtractor()
        workflow_text = extractor.extract(chunks, video_url=video_url)

        self.qdrant.add_guide(
            workflow_text,
            {
                "task": task,
                "source": "youtube",
                "video_url": video_url,
                "timestamp": datetime.now().isoformat(),
            },
        )
        print("💾 NEW YOUTUBE WORKFLOW CACHED")

        return {
            "answer": workflow_text,
            "type": "youtube_live",
            "sources": [
                {
                    "source": "youtube",
                    "url": video_url,
                    "content": workflow_text,
                }
            ],
            "source_count": 1,
            "latency": "3-8s (one-time)",
        }

    def _format_cached_guides(self, cached_guides) -> str:
        return (
            "**EXPERT WORKFLOWS FROM CACHE:**\n\n"
            + "\n\n---\n\n".join(
                [
                    f"**Guide {i+1}:**\n{guide[:300]}..."
                    for i, guide in enumerate(cached_guides[:2])
                ]
            )
        )