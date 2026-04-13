from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, ToolSelector
from pydantic_ai.toolsets import AbstractToolset

from ._toolset import JustBashToolset


@dataclass
class JustBash(AbstractCapability[AgentDepsT]):
    """Capability that adds a persistent just-bash executor to an agent.

    It wraps the assembled toolset with :class:`JustBashToolset`, exposing a
    `just_bash` tool whose shell can call wrapped tools as commands.
    """

    tool_name: str = 'just_bash'
    command_prefix: str = ''
    helper_prefix: str = 'pai_'
    exposed_tools: ToolSelector[AgentDepsT] = 'all'
    instructions: str | None = None
    files: Mapping[str, Any] | None = None
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    fs: Any = None
    python: bool = False
    javascript: bool | Any = False
    commands: Sequence[str] | None = None
    network: Any = None
    process_info: Any = None
    node_command: Sequence[str] | None = None
    js_entry: str | None = None
    package_json: str | None = None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return JustBashToolset(
            wrapped=toolset,
            tool_name=self.tool_name,
            command_prefix=self.command_prefix,
            helper_prefix=self.helper_prefix,
            exposed_tools=self.exposed_tools,
            instructions=self.instructions,
            files=self.files,
            env=self.env,
            cwd=self.cwd,
            fs=self.fs,
            python=self.python,
            javascript=self.javascript,
            commands=self.commands,
            network=self.network,
            process_info=self.process_info,
            node_command=self.node_command,
            js_entry=self.js_entry,
            package_json=self.package_json,
        )
