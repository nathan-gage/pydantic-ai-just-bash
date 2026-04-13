from __future__ import annotations

from typing import TypeAlias

from just_bash import FileInit, InMemoryFs, LazyFile, MountableFs, OverlayFs, ReadWriteFs

PublicFileSystemConfig: TypeAlias = InMemoryFs | OverlayFs | ReadWriteFs | MountableFs
"""Filesystem config types exposed through just_bash's public API."""

PublicInitialFileValue: TypeAlias = str | bytes | FileInit | LazyFile
"""Spec-safe/public initial file values.

`just-py-bash` also accepts callback-based lazy providers at runtime, but those are
not JSON/YAML spec-serializable.
"""
