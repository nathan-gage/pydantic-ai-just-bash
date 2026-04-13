from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from just_bash import JavaScriptConfig, NetworkConfig, ProcessInfo
from pydantic.dataclasses import dataclass
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from ._toolset import JustBashToolset
from ._types import (
    JustBashFileSystemConfig,
    JustBashInitialFileValue,
    JustBashToolSelector,
    SpecFileSystemConfig,
    SpecInitialFileValue,
    SpecToolSelector,
)


@dataclass
class JustBash(AbstractCapability[AgentDepsT]):
    """Capability that adds a persistent just-bash executor to an agent.

    It wraps the assembled toolset with :class:`JustBashToolset`, exposing a
    `just_bash` tool whose shell can call wrapped tools as commands.
    """

    tool_name: str = 'just_bash'
    command_prefix: str = ''
    helper_prefix: str = 'pai_'
    exposed_tools: JustBashToolSelector[AgentDepsT] = 'all'
    instructions: str | None = None
    files: Mapping[str, JustBashInitialFileValue] | None = None
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    fs: JustBashFileSystemConfig | None = None
    python: bool = False
    javascript: bool | JavaScriptConfig = False
    commands: Sequence[str] | None = None
    network: NetworkConfig | None = None
    process_info: ProcessInfo | None = None
    node_command: Sequence[str] | None = None
    js_entry: str | None = None
    package_json: str | None = None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'JustBash'

    @classmethod
    def from_spec(
        cls,
        *,
        tool_name: str = 'just_bash',
        command_prefix: str = '',
        helper_prefix: str = 'pai_',
        exposed_tools: SpecToolSelector = 'all',
        instructions: str | None = None,
        files: Mapping[str, SpecInitialFileValue] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        fs: SpecFileSystemConfig | None = None,
        python: bool = False,
        javascript: bool | JavaScriptConfig = False,
        commands: Sequence[str] | None = None,
        network: NetworkConfig | None = None,
        process_info: ProcessInfo | None = None,
        node_command: Sequence[str] | None = None,
        js_entry: str | None = None,
        package_json: str | None = None,
    ) -> Self:
        return cls(
            tool_name=tool_name,
            command_prefix=command_prefix,
            helper_prefix=helper_prefix,
            exposed_tools=exposed_tools,
            instructions=instructions,
            files=files,
            env=env,
            cwd=cwd,
            fs=fs,
            python=python,
            javascript=javascript,
            commands=commands,
            network=network,
            process_info=process_info,
            node_command=node_command,
            js_entry=js_entry,
            package_json=package_json,
        )

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
