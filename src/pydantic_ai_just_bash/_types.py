from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from os import PathLike
from typing import Any, Literal, TypeAlias

from just_bash import (
    BashLogger,
    DefenseInDepthConfig,
    ExecutionLimits,
    FeatureCoverageWriter,
    FetchCallback,
    FileInit,
    InMemoryFs,
    LazyFile,
    MountableFs,
    OverlayFs,
    ReadWriteFs,
    TraceCallback,
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

JustBashFetchCallback: TypeAlias = FetchCallback
"""Custom fetch callback accepted by the Python API."""

JustBashLogger: TypeAlias = BashLogger
"""Logger hook accepted by the Python API."""

JustBashTraceCallback: TypeAlias = TraceCallback
"""Trace callback accepted by the Python API."""

JustBashDefenseInDepth: TypeAlias = bool | DefenseInDepthConfig
"""Defense-in-depth config accepted by the Python API and specs.

Callback-based `on_violation` hooks are Python-only even though the rest of the
configuration object can be represented in specs.
"""

JustBashCoverageWriter: TypeAlias = FeatureCoverageWriter
"""Coverage writer hook accepted by the Python API."""

JustBashBackendPath: TypeAlias = str | PathLike[str]
"""Path-like backend artifact override accepted by the Python API."""

SpecFileSystemConfig: TypeAlias = JustBashFileSystemConfig
"""Filesystem config types that are safe to construct from specs."""

SpecInitialFileValue: TypeAlias = FileValue | FileInit | LazyFile
"""Initial file values that are safe to construct from YAML/JSON specs.

`just-py-bash` also accepts callback-based lazy providers at runtime, but those are
not JSON/YAML spec-serializable.
"""

SpecDefenseInDepth: TypeAlias = bool | DefenseInDepthConfig
"""Defense-in-depth config that is safe to construct from YAML/JSON specs.

Specs can represent the boolean form and data-only `DefenseInDepthConfig` values,
but not callback-based `on_violation` hooks.
"""

SpecBackendPath: TypeAlias = str
"""Backend artifact path values that are safe to construct from specs."""

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
