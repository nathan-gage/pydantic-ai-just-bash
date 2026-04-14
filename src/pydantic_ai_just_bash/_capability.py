from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from just_bash import ExecutionLimits, JavaScriptConfig, NetworkConfig, ProcessInfo
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from ._config import JustBashToolsetConfig
from ._types import (
    HelpArgumentRenamer,
    JustBashBackendPath,
    JustBashCoverageWriter,
    JustBashDefenseInDepth,
    JustBashFetchCallback,
    JustBashFileSystemConfig,
    JustBashInitialFileValue,
    JustBashLogger,
    JustBashToolSelector,
    JustBashTraceCallback,
    SpecBackendPath,
    SpecDefenseInDepth,
    SpecFileSystemConfig,
    SpecInitialFileValue,
    SpecToolSelector,
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class JustBash(AbstractCapability[AgentDepsT]):
    """Capability that adds a persistent just-bash executor and bash helpers.

    It wraps the assembled toolset with :class:`JustBashToolset`, exposing a
    `bash` tool by default plus helper tools such as `bash_list_tools`.

    Wrapped tools remain directly visible by default; set
    `expose_wrapped_tools=False` for shell-only mode.
    """

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
    execution_limits: ExecutionLimits | None = None
    python: bool = False
    javascript: bool | JavaScriptConfig = False
    commands: Sequence[str] | None = None
    fetch: JustBashFetchCallback | None = None
    logger: JustBashLogger | None = None
    trace: JustBashTraceCallback | None = None
    defense_in_depth: JustBashDefenseInDepth | None = None
    coverage: JustBashCoverageWriter | None = None
    network: NetworkConfig | None = None
    process_info: ProcessInfo | None = None
    node_command: Sequence[str] | None = None
    js_entry: JustBashBackendPath | None = None
    package_json: JustBashBackendPath | None = None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'JustBash'

    @classmethod
    def from_spec(
        cls,
        *,
        tool_name: str = 'bash',
        command_prefix: str = '',
        helper_prefix: str = 'bash_',
        exposed_tools: SpecToolSelector = 'all',
        expose_wrapped_tools: bool = True,
        instructions: str | None = None,
        help_flag_name: str = 'help',
        rename_help_argument: str | None = None,
        files: Mapping[str, SpecInitialFileValue] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        fs: SpecFileSystemConfig | None = None,
        execution_limits: ExecutionLimits | None = None,
        python: bool = False,
        javascript: bool | JavaScriptConfig = False,
        commands: Sequence[str] | None = None,
        fetch: None = None,
        logger: None = None,
        trace: None = None,
        defense_in_depth: SpecDefenseInDepth | None = None,
        coverage: None = None,
        network: NetworkConfig | None = None,
        process_info: ProcessInfo | None = None,
        node_command: Sequence[str] | None = None,
        js_entry: SpecBackendPath | None = None,
        package_json: SpecBackendPath | None = None,
    ) -> Self:
        return cls(
            tool_name=tool_name,
            command_prefix=command_prefix,
            helper_prefix=helper_prefix,
            exposed_tools=exposed_tools,
            expose_wrapped_tools=expose_wrapped_tools,
            instructions=instructions,
            help_flag_name=help_flag_name,
            rename_help_argument=rename_help_argument,
            files=files,
            env=env,
            cwd=cwd,
            fs=fs,
            execution_limits=execution_limits,
            python=python,
            javascript=javascript,
            commands=commands,
            fetch=fetch,
            logger=logger,
            trace=trace,
            defense_in_depth=defense_in_depth,
            coverage=coverage,
            network=network,
            process_info=process_info,
            node_command=node_command,
            js_entry=js_entry,
            package_json=package_json,
        )

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return self._toolset_config().build_toolset(toolset)

    def _toolset_config(self) -> JustBashToolsetConfig[AgentDepsT]:
        return JustBashToolsetConfig(
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
            fetch=self.fetch,
            logger=self.logger,
            trace=self.trace,
            defense_in_depth=self.defense_in_depth,
            coverage=self.coverage,
            network=self.network,
            process_info=self.process_info,
            node_command=self.node_command,
            js_entry=self.js_entry,
            package_json=self.package_json,
        )
