import requests
from typing import List, Dict
from tools.youtube import YoutubeTool


class MultiSourceFetcher:
    """
    Lightweight wrapper around YoutubeTool for this project.

    For now, we:
      - treat the user's `task` as a YouTube URL if it looks like one
      - otherwise assume `task` is a natural-language description and
        let higher layers decide how to map it to a URL
    """

    def __init__(self):
        self.youtube = YoutubeTool()

    def fetch_all_sources(self, task: str) -> List[Dict]:
        """
        Return a single 'youtube' source containing raw transcript chunks.

        HybridRAGAgent expects a list[dict] with keys: source, content, (optional) url.
        """
        video_url = task.strip()

        chunks = self.youtube.fetch_chunks(video_url)
        if not chunks:
            return []

        # Join chunks for now; later you can keep them separate or structured
        combined_content = "\n\n".join(chunks)

        return [
            {
                "source": "youtube",
                "url": video_url,
                "content": combined_content,
                "quality": 0.90,
            }
        ]