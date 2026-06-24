"""Conversation history store (JSON-backed).

Powers the web UI's chat history: multiple named conversations, each a list of
role/content messages, persisted to the workspace so they survive restarts.
Dependency-free and tiny, in keeping with SABI's offline, low-RAM design.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional


def _now() -> float:
    return time.time()


class ConversationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: Dict = {"order": [], "conversations": {}}
        self.load()

    # ----------------------------------------------------------------- IO
    def load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {"order": [], "conversations": {}}
        self._data.setdefault("order", [])
        self._data.setdefault("conversations", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- queries
    def list(self) -> List[Dict]:
        """Return conversation summaries (no message bodies), newest first."""
        out = []
        for cid in self._data["order"]:
            conv = self._data["conversations"].get(cid)
            if conv:
                out.append({
                    "id": conv["id"],
                    "title": conv["title"],
                    "updated": conv["updated"],
                    "message_count": len(conv["messages"]),
                })
        out.sort(key=lambda c: c["updated"], reverse=True)
        return out

    def get(self, cid: str) -> Optional[Dict]:
        return self._data["conversations"].get(cid)

    # ------------------------------------------------------------- mutations
    def create(self, title: str = "New chat") -> Dict:
        cid = uuid.uuid4().hex[:12]
        conv = {"id": cid, "title": title, "created": _now(),
                "updated": _now(), "messages": []}
        self._data["conversations"][cid] = conv
        self._data["order"].insert(0, cid)
        self.save()
        return conv

    def add_message(self, cid: str, role: str, content: str,
                    meta: Optional[Dict] = None) -> Optional[Dict]:
        conv = self._data["conversations"].get(cid)
        if not conv:
            return None
        msg = {"role": role, "content": content, "t": _now()}
        if meta:
            msg["meta"] = meta
        conv["messages"].append(msg)
        conv["updated"] = _now()
        # Auto-title from the first user message.
        if role == "user" and conv["title"] in ("New chat", "", None):
            conv["title"] = (content[:40] + "…") if len(content) > 40 else content
        self.save()
        return msg

    def rename(self, cid: str, title: str) -> bool:
        conv = self._data["conversations"].get(cid)
        if not conv:
            return False
        conv["title"] = title
        conv["updated"] = _now()
        self.save()
        return True

    def delete(self, cid: str) -> bool:
        if cid in self._data["conversations"]:
            del self._data["conversations"][cid]
            self._data["order"] = [x for x in self._data["order"] if x != cid]
            self.save()
            return True
        return False
