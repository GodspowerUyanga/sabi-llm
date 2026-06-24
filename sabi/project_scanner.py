"""Project scanner.

On startup SABI scans the current directory and auto-detects the project
context (Git, Python, Node.js, Next.js, package managers, virtual environments,
existing files) so it can act intelligently without manual configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ProjectInfo:
    root: str
    is_git: bool = False
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    package_managers: List[str] = field(default_factory=list)
    has_venv: bool = False
    file_count: int = 0

    def summary(self) -> str:
        parts = []
        parts.append("git" if self.is_git else "no-git")
        if self.languages:
            parts.append("langs=" + ",".join(self.languages))
        if self.frameworks:
            parts.append("frameworks=" + ",".join(self.frameworks))
        if self.package_managers:
            parts.append("pkg=" + ",".join(self.package_managers))
        if self.has_venv:
            parts.append("venv")
        parts.append(f"{self.file_count} files")
        return " | ".join(parts)


def scan(path: str | Path = ".") -> ProjectInfo:
    root = Path(path).resolve()
    info = ProjectInfo(root=str(root))

    names = {p.name for p in root.iterdir()} if root.exists() else set()

    info.is_git = ".git" in names

    # Languages / package managers
    if "package.json" in names:
        info.languages.append("javascript")
        info.package_managers.append("npm")
    if "yarn.lock" in names:
        info.package_managers.append("yarn")
    if "pnpm-lock.yaml" in names:
        info.package_managers.append("pnpm")
    if any(n in names for n in ("requirements.txt", "pyproject.toml", "setup.py", "Pipfile")):
        info.languages.append("python")
        if "pyproject.toml" in names:
            info.package_managers.append("pip/poetry")
        elif "Pipfile" in names:
            info.package_managers.append("pipenv")
        else:
            info.package_managers.append("pip")
    if "go.mod" in names:
        info.languages.append("go")
    if "Cargo.toml" in names:
        info.languages.append("rust")

    # Frameworks
    if "next.config.js" in names or "next.config.mjs" in names or "next.config.ts" in names:
        info.frameworks.append("next.js")
    if "package.json" in names:
        try:
            content = (root / "package.json").read_text(encoding="utf-8", errors="replace")
            for fw in ("next", "react", "express", "vue", "svelte"):
                if f'"{fw}"' in content and fw not in info.frameworks:
                    info.frameworks.append(fw)
        except Exception:
            pass

    # Virtual environments
    info.has_venv = any(n in names for n in (".venv", "venv", "env", ".env-venv"))

    # File count (shallow)
    try:
        info.file_count = sum(1 for p in root.iterdir() if p.is_file())
    except Exception:
        info.file_count = 0

    # de-dup
    info.languages = sorted(set(info.languages))
    info.frameworks = sorted(set(info.frameworks))
    info.package_managers = sorted(set(info.package_managers))
    return info
