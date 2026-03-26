from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import requests

load_dotenv()

WORKFLOW_MODEL = os.getenv("WORKFLOW_MODEL", "sonar-pro")  # or any chat model
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


class WorkflowExtractor:
    """
    Turn raw YouTube transcript chunks into a structured, step-by-step workflow.

    v0: single-shot prompt that returns a markdown-ish text structure:
        - Title
        - Goal
        - Prerequisites / tools
        - Ordered steps
        - Warnings / tips
    """

    def __init__(self):
        if not PERPLEXITY_API_KEY:
            raise RuntimeError("PERPLEXITY_API_KEY is not set in environment")

    def _call_llm(self, prompt: str) -> str:
        """
        Minimal wrapper around Perplexity chat completions API.
        Swap this out if you change providers.
        """
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": WORKFLOW_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You convert technical tutorials into clear, step-by-step procedures. "
                        "Always respond in structured markdown with explicit numbered steps."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.2,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def extract(self, transcript_chunks: List[str], video_url: Optional[str] = None) -> str:
        """
        Main entrypoint: takes list[str] chunks and returns a single workflow string.
        """
        if not transcript_chunks:
            return "No transcript available to build a workflow."

        # Keep it simple: join with separators
        transcript_text = "\n\n".join(transcript_chunks[:20])  # hard cap to keep prompt small

        url_str = f"\nSource video: {video_url}" if video_url else ""
        prompt = f"""
You are given the transcript of a technical tutorial video about a physical or technical task.

Your job is to convert it into a clear, actionable, step-by-step workflow that another person can follow.

Requirements:
- Start with a short title line: "Title: ..."
- Then a one-sentence goal line: "Goal: ..."
- Then a bullet list of prerequisites / tools / materials if mentioned.
- Then a numbered list of steps (1., 2., 3., ...). Each step should be a single clear action or small group of closely related actions.
- Preserve any safety warnings or important cautions in a separate "Warnings" section.
- End with a short "Notes" section for variations, tips, or troubleshooting if the transcript mentions them.
- Do NOT invent tools or steps that are not implied by the transcript.

Transcript:
---
{transcript_text}
---

{url_str}
Now output only the workflow in markdown following the structure above.
"""
        try:
            workflow = self._call_llm(prompt)
        except Exception as e:
            # Fallback: return raw transcript if the extractor fails
            return f"Workflow extraction failed, showing raw transcript instead:\n\n{transcript_text[:4000]}"

        return workflow


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