"""File-operation tools (sandboxed to the workspace)."""

from __future__ import annotations

from .base import Tool, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace."

    def run(self, path: str, **_) -> ToolResult:
        try:
            target = self._safe_path(path)
            if not target.exists():
                return ToolResult(False, error=f"file not found: {path}")
            text = target.read_text(encoding="utf-8", errors="replace")
            return ToolResult(True, output=text, data={"bytes": len(text)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, error=str(exc))


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a text file in the workspace."

    def run(self, path: str, content: str = "", **_) -> ToolResult:
        try:
            target = self._safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(True, output=f"wrote {len(content)} chars to {path}",
                              data={"path": str(target)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, error=str(exc))


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories under a workspace path."

    def run(self, path: str = ".", **_) -> ToolResult:
        try:
            target = self._safe_path(path)
            if not target.exists():
                return ToolResult(False, error=f"path not found: {path}")
            entries = sorted(
                (("d " if p.is_dir() else "f ") + p.name) for p in target.iterdir()
            )
            return ToolResult(True, output="\n".join(entries),
                              data={"count": len(entries)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, error=str(exc))
