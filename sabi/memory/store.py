"""JSON-backed memory store.

Tracks conversation turns and task history in a single local JSON file. Kept
deliberately small and dependency-free so it loads instantly and uses almost
no RAM - important under the 7 GB ceiling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


class MemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: Dict[str, Any] = {"turns": [], "tasks": [], "meta": {}}
        self.load()

    # ------------------------------------------------------------- IO
    def load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                # Corrupt memory should never block startup.
                self._data = {"turns": [], "tasks": [], "meta": {}}
        self._data.setdefault("turns", [])
        self._data.setdefault("tasks", [])
        self._data.setdefault("meta", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- turns
    def add_turn(self, role: str, content: str, intent: str = "") -> None:
        self._data["turns"].append(
            {"t": time.time(), "role": role, "content": content, "intent": intent}
        )
        self.save()

    def recent_turns(self, n: int = 6) -> List[Dict[str, Any]]:
        return self._data["turns"][-n:]

    def history_text(self, n: int = 6) -> str:
        lines = []
        for turn in self.recent_turns(n):
            lines.append(f"{turn['role'].upper()}: {turn['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------- tasks
    def add_task(self, summary: str, status: str = "done", detail: str = "") -> None:
        self._data["tasks"].append(
            {"t": time.time(), "summary": summary, "status": status, "detail": detail}
        )
        self.save()

    def tasks(self) -> List[Dict[str, Any]]:
        return list(self._data["tasks"])

    # ------------------------------------------------------------- misc
    def clear(self) -> None:
        self._data = {"turns": [], "tasks": [], "meta": {}}
        self.save()

    def stats(self) -> Dict[str, int]:
        return {"turns": len(self._data["turns"]), "tasks": len(self._data["tasks"])}
