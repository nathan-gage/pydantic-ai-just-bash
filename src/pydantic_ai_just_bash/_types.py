from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, TypeAlias

from just_bash import (
    ExecutionLimits,
    FileInit,
    InMemoryFs,
    LazyFile,
    MountableFs,
    OverlayFs,
    ReadWriteFs,
)
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.tools import ToolDefinition

FileValue: TypeAlias = str | bytes
LazyFileProvider: TypeAlias = Callable[[], FileValue | Awaitable[FileValue]]
HelpArgumentRenamer: TypeAlias = str | Callable[[str, str], str] | None
SpecHelpArgumentRenamer: TypeAlias = str | None

JustBashExecutionLimits: TypeAlias = ExecutionLimits
"""Execution limits accepted by both the Python API and specs."""

JustBashFileSystemConfig: TypeAlias = InMemoryFs | OverlayFs | ReadWriteFs | MountableFs
"""Filesystem config types accepted by both the Python API and specs."""

JustBashInitialFileValue: TypeAlias = FileValue | FileInit | LazyFile | LazyFileProvider
"""Initial file values accepted by the Python API.

This includes callback-based lazy file providers, which are supported when the
capability is configured directly in Python.
"""

SpecFileSystemConfig: TypeAlias = JustBashFileSystemConfig
"""Filesystem config types that are safe to construct from specs."""

SpecInitialFileValue: TypeAlias = FileValue | FileInit | LazyFile
"""Initial file values that are safe to construct from YAML/JSON specs.

`just-py-bash` also accepts callback-based lazy providers at runtime, but those are
not JSON/YAML spec-serializable.
"""

JustBashToolSelectorFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT], ToolDefinition],
    bool | Awaitable[bool],
]
"""Runtime tool-selector callback accepted by JustBash.

This mirrors Pydantic AI's `ToolSelectorFunc`, but avoids its forward-ref annotation
so `JustBash` can remain a plain pydantic dataclass without a rebuild step.
"""

JustBashToolSelector: TypeAlias = Literal['all'] | Sequence[str] | dict[str, Any] | JustBashToolSelectorFunc[AgentDepsT]
"""Tool selector accepted by the runtime Python API."""

SpecToolSelector: TypeAlias = Literal['all'] | Sequence[str] | dict[str, Any]
"""Tool selector forms that are safe to construct from YAML/JSON specs."""
