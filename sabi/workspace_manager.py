"""Workspace manager.

Manages the local ``sabi_workspace`` sandbox where SABI reads and writes
generated projects. Keeps a hidden ``.sabi`` folder for memory and the vector
store, and provides helpers for listing and creating project folders.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List


class WorkspaceManager:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.root / ".sabi"
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def projects(self) -> List[str]:
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and p.name != ".sabi"
        )

    def create_project(self, name: str) -> Path:
        target = (self.root / name).resolve()
        if self.root not in target.parents:
            raise ValueError("project path escapes workspace")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def path_for(self, name: str) -> Path:
        return self.root / name

    def reset(self, keep_meta: bool = True) -> None:
        for child in self.root.iterdir():
            if keep_meta and child.name == ".sabi":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
