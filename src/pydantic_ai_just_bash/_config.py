from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from just_bash import (
    AsyncBash,
    AsyncCustomCommands,
    JavaScriptConfig,
    NetworkConfig,
    ProcessInfo,
)
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from ._types import (
    HelpArgumentRenamer,
    JustBashExecutionLimits,
    JustBashFileSystemConfig,
    JustBashInitialFileValue,
    JustBashToolSelector,
)

if TYPE_CHECKING:
    from ._toolset import JustBashToolset


@dataclass(frozen=True, slots=True)
class JustBashToolsetConfig(Generic[AgentDepsT]):
    tool_name: str = 'bash'
    command_prefix: str = ''
    helper_prefix: str = 'bash_'
    exposed_tools: JustBashToolSelector[AgentDepsT] = 'all'
    expose_wrapped_tools: bool = True
    instructions: str | None = None
    help_flag_name: str = 'help'
    rename_help_argument: HelpArgumentRenamer = None
    files: Mapping[str, JustBashInitialFileValue] | None = None
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    fs: JustBashFileSystemConfig | None = None
    execution_limits: JustBashExecutionLimits | None = None
    python: bool = False
    javascript: bool | JavaScriptConfig = False
    commands: Sequence[str] | None = None
    network: NetworkConfig | None = None
    process_info: ProcessInfo | None = None
    node_command: Sequence[str] | None = None
    js_entry: str | None = None
    package_json: str | None = None

    def build_toolset(self, wrapped: AbstractToolset[AgentDepsT]) -> JustBashToolset[AgentDepsT]:
        from ._toolset import JustBashToolset

        return JustBashToolset(
            wrapped=wrapped,
            tool_name=self.tool_name,
            command_prefix=self.command_prefix,
            helper_prefix=self.helper_prefix,
            exposed_tools=self.exposed_tools,
            expose_wrapped_tools=self.expose_wrapped_tools,
            instructions=self.instructions,
            help_flag_name=self.help_flag_name,
            rename_help_argument=self.rename_help_argument,
            files=self.files,
            env=self.env,
            cwd=self.cwd,
            fs=self.fs,
            execution_limits=self.execution_limits,
            python=self.python,
            javascript=self.javascript,
            commands=self.commands,
            network=self.network,
            process_info=self.process_info,
            node_command=self.node_command,
            js_entry=self.js_entry,
            package_json=self.package_json,
        )

    def build_bash(self, *, custom_commands: AsyncCustomCommands) -> AsyncBash:
        return AsyncBash(
            files=self.files,
            env=self.env,
            cwd=self.cwd,
            fs=self.fs,
            execution_limits=self.execution_limits,
            python=self.python,
            javascript=self.javascript,
            commands=self.commands,
            network=self.network,
            process_info=self.process_info,
            node_command=self.node_command,
            js_entry=self.js_entry,
            package_json=self.package_json,
            custom_commands=custom_commands,
        )
