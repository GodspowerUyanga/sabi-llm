"""Local tool layer: file, shell and workspace operations.

All tools operate on the local filesystem only - no network access - in
keeping with SABI's offline-first design.
"""

from .base import Tool, ToolResult, ToolRegistry
from .file_tools import ReadFileTool, WriteFileTool, ListDirTool
from .shell_tools import ShellTool
from .workspace_tools import ScaffoldTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "ShellTool",
    "ScaffoldTool",
    "default_registry",
]


def default_registry(workspace):
    """Build a registry pre-populated with the standard local tools."""
    reg = ToolRegistry()
    reg.register(ReadFileTool(workspace))
    reg.register(WriteFileTool(workspace))
    reg.register(ListDirTool(workspace))
    reg.register(ShellTool(workspace))
    reg.register(ScaffoldTool(workspace))
    return reg
