from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._capability import JustBash
from ._toolset import JustBashExecutionResult, JustBashToolset

try:
    __version__ = version('pydantic-ai-just-bash')
except PackageNotFoundError:  # pragma: no cover
    __version__ = '0.0.0'

__all__ = [
    'JustBash',
    'JustBashExecutionResult',
    'JustBashToolset',
    '__version__',
]
