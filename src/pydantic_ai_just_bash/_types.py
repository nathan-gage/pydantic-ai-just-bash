from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from just_bash import FileInit, InMemoryFs, LazyFile, MountableFs, OverlayFs, ReadWriteFs

FileValue: TypeAlias = str | bytes
LazyFileProvider: TypeAlias = Callable[[], FileValue | Awaitable[FileValue]]

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
