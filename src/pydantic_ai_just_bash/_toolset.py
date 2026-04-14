from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic

from just_bash import AsyncBash, AsyncCustomCommandContext, JavaScriptConfig, NetworkConfig, ProcessInfo
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Tool
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import ModelRetry, ToolRetryError, UserError
from pydantic_ai.messages import InstructionPart, ToolCallPart, ToolReturn
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition, matches_tool_selector
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from ._config import JustBashToolsetConfig
from ._json_schema import copy_json_object, required_names, schema_properties
from ._shell_cli import (
    ShellCliError,
    ShellCommandSurface,
    bind_command_args,
    command_help_hint,
    is_help_request,
    render_help,
)
from ._types import (
    HelpArgumentRenamer,
    JustBashBackendPath,
    JustBashCoverageWriter,
    JustBashDefenseInDepth,
    JustBashExecutionLimits,
    JustBashFetchCallback,
    JustBashFileSystemConfig,
    JustBashInitialFileValue,
    JustBashLogger,
    JustBashToolSelector,
    JustBashTraceCallback,
)

_ANY_JSON_TA = TypeAdapter(Any)
_MAX_SEARCH_RESULTS = 10


class BashExecutionResult(BaseModel):
    stdout: str = ''
    stderr: str = ''
    exit_code: int
    ok: bool


class BashCommandInfo(BaseModel):
    tool_name: str
    command: str
    description: str | None = None


class BashListToolsResult(BaseModel):
    commands: list[BashCommandInfo] = Field(default_factory=list)
    hidden_count: int = 0


class BashSearchToolsResult(BaseModel):
    matches: list[BashCommandInfo] = Field(default_factory=list)


class BashDescribeToolResult(BaseModel):
    tool_name: str
    command: str
    description: str | None = None
    hidden: bool
    help_text: str
    parameters_json_schema: dict[str, Any]
    signature: str | None = None


@dataclass
class _SearchIndexEntry:
    tool_name: str
    command_name: str
    description: str | None = None

    @property
    def searchable_text(self) -> str:
        parts = [self.tool_name.lower(), self.command_name.lower()]
        if self.description:
            parts.append(self.description.lower())
        return ' '.join(parts)


@dataclass
class _CommandBinding(Generic[AgentDepsT]):
    tool_name: str
    command_name: str
    tool: ToolsetTool[AgentDepsT]
    shell_tool_def: ToolDefinition
    shell_to_actual_arg_names: dict[str, str]
    actual_to_shell_arg_names: dict[str, str]
    hidden: bool


@dataclass
class _ShellState(Generic[AgentDepsT]):
    ctx: RunContext[AgentDepsT]
    tool_manager: ToolManager[AgentDepsT]
    bindings_by_tool: dict[str, _CommandBinding[AgentDepsT]]
    bindings_by_command: dict[str, _CommandBinding[AgentDepsT]]
    search_index: list[_SearchIndexEntry]


class JustBashToolset(WrapperToolset[AgentDepsT]):
    """Wrap a toolset and expose a persistent just-bash executor plus bash helpers.

    The default public tools are:

    - `bash`
    - `bash_list_tools`
    - `bash_search_tools`
    - `bash_describe_tool`

    Wrapped Pydantic AI tools are available inside the shell as commands. By default
    they also remain directly visible to the model, but you can hide them from the
    public tool list with `expose_wrapped_tools=False` for a shell-only mode.

    Deferred tools remain hidden from the wrapper's public listings until discovered,
    following a progressive-disclosure model.
    """

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_name: str = 'bash',
        command_prefix: str = '',
        helper_prefix: str = 'bash_',
        exposed_tools: JustBashToolSelector[AgentDepsT] = 'all',
        expose_wrapped_tools: bool = True,
        instructions: str | None = None,
        help_flag_name: str = 'help',
        rename_help_argument: HelpArgumentRenamer = None,
        files: Mapping[str, JustBashInitialFileValue] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        fs: JustBashFileSystemConfig | None = None,
        execution_limits: JustBashExecutionLimits | None = None,
        python: bool = False,
        javascript: bool | JavaScriptConfig = False,
        commands: Sequence[str] | None = None,
        fetch: JustBashFetchCallback | None = None,
        logger: JustBashLogger | None = None,
        trace: JustBashTraceCallback | None = None,
        defense_in_depth: JustBashDefenseInDepth | None = None,
        coverage: JustBashCoverageWriter | None = None,
        network: NetworkConfig | None = None,
        process_info: ProcessInfo | None = None,
        node_command: Sequence[str] | None = None,
        js_entry: JustBashBackendPath | None = None,
        package_json: JustBashBackendPath | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.tool_name = tool_name
        self.command_prefix = command_prefix
        self.helper_prefix = helper_prefix
        self.exposed_tools = exposed_tools
        self.expose_wrapped_tools = expose_wrapped_tools
        self.instructions = instructions
        self.help_flag_name = help_flag_name
        self.rename_help_argument = rename_help_argument
        self.files = files
        self.env = env
        self.cwd = cwd
        self.fs = fs
        self.execution_limits = execution_limits
        self.python = python
        self.javascript = javascript
        self.commands = commands
        self.fetch = fetch
        self.logger = logger
        self.trace = trace
        self.defense_in_depth = defense_in_depth
        self.coverage = coverage
        self.network = network
        self.process_info = process_info
        self.node_command = node_command
        self.js_entry = js_entry
        self.package_json = package_json

        self.list_tools_name = self._helper_name('list_tools')
        self.search_tools_name = self._helper_name('search_tools')
        self.describe_tool_name = self._helper_name('describe_tool')
        self.call_tool_name = self._helper_name('call_tool')

        self._validate_configuration()

        self._bash: AsyncBash | None = None
        self._shell_state: _ShellState[AgentDepsT] | None = None
        self._discovered_tools: set[str] = set()

        self._bash_tool = Tool(
            self._run_bash,
            takes_ctx=True,
            name=self.tool_name,
            description=self._bash_tool_description(),
            sequential=True,
            include_return_schema=True,
        )
        self._list_tools_tool = Tool(
            self._list_tools,
            takes_ctx=True,
            name=self.list_tools_name,
            description='List the shell-visible commands exposed by the bash wrapper.',
            sequential=True,
            include_return_schema=True,
        )
        self._search_tools_tool = Tool(
            self._search_tools,
            takes_ctx=True,
            name=self.search_tools_name,
            description=(
                'Search hidden shell commands backed by deferred Pydantic AI tools. '
                'Use specific keywords to discover commands that are not yet visible in the bash environment.'
            ),
            sequential=True,
            include_return_schema=True,
        )
        self._describe_tool_tool = Tool(
            self._describe_tool,
            takes_ctx=True,
            name=self.describe_tool_name,
            description='Describe a shell-exposed command, including generated CLI help and argument schema.',
            sequential=True,
            include_return_schema=True,
        )

    @property
    def id(self) -> str | None:
        return None

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.wrapped.label})'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_wrapped = await self.wrapped.for_run(ctx)
        return self._copy_for_run(new_wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        self.wrapped = await self.wrapped.for_run_step(ctx)
        return self

    async def __aenter__(self) -> JustBashToolset[AgentDepsT]:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            await self._close_bash()
        finally:
            wrapped_result = await self.wrapped.__aexit__(*args)
        return wrapped_result

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        wrapped_instructions = await self.wrapped.get_instructions(ctx)
        instructions: list[str | InstructionPart] = []
        if wrapped_instructions is not None:
            if isinstance(wrapped_instructions, (str, InstructionPart)):
                instructions.append(wrapped_instructions)
            else:
                instructions.extend(wrapped_instructions)

        extra = self.instructions or self._default_instructions()
        if extra:
            instructions.append(extra)

        return instructions or None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        wrapped_tools = dict(await self.wrapped.get_tools(ctx))
        reserved_names = {
            self.tool_name,
            self.list_tools_name,
            self.search_tools_name,
            self.describe_tool_name,
        }
        conflicting = reserved_names.intersection(wrapped_tools)
        if conflicting:
            names = ', '.join(sorted(repr(name) for name in conflicting))
            raise UserError(
                f'Reserved bash wrapper tool name conflict: {names}. '
                'Rename the wrapped tool or configure the wrapper names differently.'
            )

        bindings_by_command: dict[str, _CommandBinding[AgentDepsT]] = {}
        for tool_name, tool in wrapped_tools.items():
            if not await matches_tool_selector(self.exposed_tools, ctx, tool.tool_def):
                continue
            command_name = self._tool_command_name(tool_name)
            self._validate_command_name(command_name, tool_name, bindings_by_command)
            binding = self._build_command_binding(tool_name, command_name, tool)
            bindings_by_command[command_name] = binding

        tools = dict(wrapped_tools) if self.expose_wrapped_tools else {}
        for public_tool in self._public_tools():
            toolset_tool = await self._as_toolset_tool(ctx, public_tool)
            tools[toolset_tool.tool_def.name] = toolset_tool
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == self.tool_name:
            return await self._run_bash(ctx, **tool_args)
        if name == self.list_tools_name:
            return await self._list_tools(ctx)
        if name == self.search_tools_name:
            return await self._search_tools(ctx, **tool_args)
        if name == self.describe_tool_name:
            return await self._describe_tool(ctx, **tool_args)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self._toolset_config().build_toolset(self.wrapped.visit_and_replace(visitor))

    async def _run_bash(
        self,
        ctx: RunContext[AgentDepsT],
        script: str,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        replace_env: bool = False,
        raw_script: bool = False,
        args: list[str] | None = None,
        timeout: float | None = None,
        reset_session: bool = False,
    ) -> BashExecutionResult:
        shell_state = await self._build_shell_state(ctx)
        self._shell_state = shell_state

        bash = await self._ensure_bash(shell_state, reset_session=reset_session)
        result = await bash.exec(
            self._script_with_shell_command_prelude(shell_state, script),
            stdin=stdin,
            cwd=cwd,
            env=env,
            replace_env=replace_env,
            raw_script=raw_script,
            args=args,
            timeout=timeout,
        )
        return BashExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            ok=result.ok,
        )

    async def _list_tools(self, ctx: RunContext[AgentDepsT]) -> BashListToolsResult:
        shell_state = await self._build_shell_state(ctx)
        commands = [self._binding_summary(binding) for binding in self._visible_bindings(shell_state)]
        hidden_count = len([binding for binding in shell_state.bindings_by_tool.values() if binding.hidden])
        return BashListToolsResult(commands=commands, hidden_count=hidden_count)

    async def _search_tools(self, ctx: RunContext[AgentDepsT], keywords: str) -> BashSearchToolsResult:
        shell_state = await self._build_shell_state(ctx)
        matches = self._discover_bindings(shell_state, keywords)
        return BashSearchToolsResult(matches=[self._binding_summary(binding) for binding in matches])

    async def _describe_tool(self, ctx: RunContext[AgentDepsT], name: str) -> BashDescribeToolResult:
        shell_state = await self._build_shell_state(ctx)
        binding = self._resolve_binding(name, shell_state)
        if binding is None:
            raise ModelRetry(f'Unknown shell tool or command: {name!r}.')

        return BashDescribeToolResult(
            tool_name=binding.tool_name,
            command=binding.command_name,
            description=binding.shell_tool_def.description,
            hidden=binding.hidden,
            help_text=self._help_text(binding),
            parameters_json_schema=copy_json_object(binding.shell_tool_def.parameters_json_schema),
            signature=self._render_signature(binding),
        )

    async def _build_shell_state(self, ctx: RunContext[AgentDepsT]) -> _ShellState[AgentDepsT]:
        shell_ctx = replace(ctx)
        root_capability = ctx.tool_manager.root_capability if ctx.tool_manager is not None else None
        tool_manager = await ToolManager(
            toolset=self.wrapped,
            root_capability=root_capability,
        ).for_run_step(shell_ctx)

        bindings_by_tool: dict[str, _CommandBinding[AgentDepsT]] = {}
        bindings_by_command: dict[str, _CommandBinding[AgentDepsT]] = {}
        search_index: list[_SearchIndexEntry] = []

        for tool_name, tool in (tool_manager.tools or {}).items():
            if not await matches_tool_selector(self.exposed_tools, shell_ctx, tool.tool_def):
                continue

            command_name = self._tool_command_name(tool_name)
            self._validate_command_name(command_name, tool_name, bindings_by_command)
            binding = self._build_command_binding(tool_name, command_name, tool)
            bindings_by_tool[tool_name] = binding
            bindings_by_command[command_name] = binding
            if binding.hidden:
                search_index.append(
                    _SearchIndexEntry(
                        tool_name=tool_name,
                        command_name=command_name,
                        description=binding.shell_tool_def.description,
                    )
                )

        return _ShellState(
            ctx=shell_ctx,
            tool_manager=tool_manager,
            bindings_by_tool=bindings_by_tool,
            bindings_by_command=bindings_by_command,
            search_index=search_index,
        )

    async def _ensure_bash(
        self,
        shell_state: _ShellState[AgentDepsT],
        *,
        reset_session: bool,
    ) -> AsyncBash:
        if reset_session:
            self._discovered_tools.clear()
            await self._close_bash()

        if self._bash is None:
            self._bash = self._toolset_config().build_bash(custom_commands=self._custom_commands())

        return self._bash

    async def _close_bash(self) -> None:
        if self._bash is not None:
            bash = self._bash
            self._bash = None
            await bash.close()

    def _custom_commands(self) -> dict[str, Any]:
        return {
            self.list_tools_name: self._cmd_list_tools,
            self.describe_tool_name: self._cmd_describe_tool,
            self.call_tool_name: self._cmd_call_tool,
            self.search_tools_name: self._cmd_search_tools,
        }

    async def _cmd_call_tool(self, args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
        if not args:
            return self._cli_error_result(
                self.call_tool_name,
                'missing tool or command name.',
                exit_code=2,
                usage=f'usage: {self.call_tool_name} <tool-or-command> [tool args...]',
            )

        shell_state = self._require_shell_state()
        binding = self._resolve_binding(args[0], shell_state)
        if binding is None:
            return self._cli_error_result(
                self.call_tool_name,
                f'unknown tool or command {args[0]!r}.',
                exit_code=127,
                hint=(
                    f'Use {self.list_tools_name} to list visible commands or '
                    f'{self.search_tools_name} <keywords> to discover deferred ones.'
                ),
            )
        return await self._invoke_binding(binding, args[1:], ctx)

    async def _cmd_describe_tool(self, args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
        del ctx
        if len(args) != 1:
            return self._cli_error_result(
                self.describe_tool_name,
                'expected exactly one tool or command name.',
                exit_code=2,
                usage=f'usage: {self.describe_tool_name} <tool-or-command>',
            )

        shell_state = self._require_shell_state()
        binding = self._resolve_binding(args[0], shell_state)
        if binding is None:
            return self._cli_error_result(
                self.describe_tool_name,
                f'unknown tool or command {args[0]!r}.',
                exit_code=127,
            )

        return {
            'stdout': self._help_text(binding),
            'stderr': '',
            'exit_code': 0,
        }

    async def _cmd_list_tools(self, args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
        del args, ctx
        shell_state = self._require_shell_state()
        visible_bindings = self._visible_bindings(shell_state)
        lines = [binding.command_name for binding in visible_bindings]
        hidden_count = len([binding for binding in shell_state.bindings_by_tool.values() if binding.hidden])
        if hidden_count:
            lines.append(f'[{hidden_count} hidden tool commands available via {self.search_tools_name} <keywords>]')
        lines.append(f'[generic helper: {self.call_tool_name} <tool-or-command> --json {{...}}]')
        stdout = '\n'.join(lines)
        if stdout:
            stdout += '\n'
        return {'stdout': stdout, 'stderr': '', 'exit_code': 0}

    async def _cmd_search_tools(self, args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
        shell_state = self._require_shell_state()
        if not shell_state.search_index:
            return {'stdout': 'No hidden tools are available.\n', 'stderr': '', 'exit_code': 0}

        keywords = ' '.join(args).strip() or ctx.stdin.strip()
        if not keywords:
            return self._cli_error_result(
                self.search_tools_name,
                'missing search keywords.',
                exit_code=2,
                usage=f'usage: {self.search_tools_name} <keywords>',
            )

        matches = self._discover_bindings(shell_state, keywords)
        if not matches:
            return {
                'stdout': 'No matching tools found. The tools you need may not be available.\n',
                'stderr': '',
                'exit_code': 0,
            }

        payload: Any = [summary.model_dump() for summary in [self._binding_summary(binding) for binding in matches]]
        stdout = _ANY_JSON_TA.dump_json(payload).decode()
        return {'stdout': stdout, 'stderr': '', 'exit_code': 0}

    async def _invoke_binding(
        self,
        binding: _CommandBinding[AgentDepsT],
        args: list[str],
        ctx: AsyncCustomCommandContext,
    ) -> dict[str, Any]:
        shell_state = self._require_shell_state()
        if binding.hidden:
            return self._cli_error_result(
                binding.command_name,
                'command is hidden until it is discovered.',
                exit_code=127,
                hint=f'Use {self.search_tools_name} <keywords> to discover it first.',
            )

        if is_help_request(args, self.help_flag_name):
            return {
                'stdout': self._help_text(binding),
                'stderr': '',
                'exit_code': 0,
            }

        try:
            shell_args = bind_command_args(self._shell_surface(binding), args, ctx.stdin)
            actual_args = self._shell_args_to_actual_args(binding, shell_args)
        except ShellCliError as exc:
            return self._cli_error_result(
                binding.command_name,
                str(exc),
                exit_code=2,
                usage=exc.usage,
                hint=exc.hint or command_help_hint(binding.command_name, self.help_flag_name),
            )
        except ValueError as exc:
            return self._cli_error_result(
                binding.command_name,
                str(exc),
                exit_code=2,
                hint=command_help_hint(binding.command_name, self.help_flag_name),
            )

        try:
            result = await shell_state.tool_manager.handle_call(
                ToolCallPart(tool_name=binding.tool_name, args=actual_args)
            )
        except ToolRetryError as exc:
            return self._tool_retry_error_result(binding, exc)
        except Exception as exc:
            return self._cli_error_result(
                binding.command_name,
                self._execution_error_message(exc),
                exit_code=1,
            )

        return {
            'stdout': self._serialize_tool_result(result),
            'stderr': '',
            'exit_code': 0,
        }

    def _help_text(self, binding: _CommandBinding[AgentDepsT]) -> str:
        return render_help(self._shell_surface(binding))

    def _shell_surface(self, binding: _CommandBinding[AgentDepsT]) -> ShellCommandSurface:
        return ShellCommandSurface(
            command_name=binding.command_name,
            help_flag_name=self.help_flag_name,
            tool_def=binding.shell_tool_def,
            actual_to_shell_arg_names=binding.actual_to_shell_arg_names,
            signature=self._render_signature(binding),
        )

    def _render_signature(self, binding: _CommandBinding[AgentDepsT]) -> str | None:
        tool_def = binding.shell_tool_def
        if tool_def.function_signature is None:
            return None
        return tool_def.function_signature.render(
            '...',
            name=binding.command_name,
            description=tool_def.description,
        )

    def _serialize_tool_result(self, result: Any) -> str:
        if isinstance(result, ToolReturn):
            result = result.return_value
        if result is None:
            return ''
        if isinstance(result, str):
            return result
        return _ANY_JSON_TA.dump_json(result).decode()

    def _default_instructions(self) -> str:
        visibility = (
            'Wrapped tools remain directly callable as normal Pydantic AI tools. '
            if self.expose_wrapped_tools
            else 'Wrapped tools are only available inside the shell and through the bash helper tools. '
        )
        return (
            f'Use {self.tool_name} when it is helpful to orchestrate multiple tool calls through a shell-like workflow. '
            'Inside the shell, wrapped tools are exposed as commands. '
            f'{visibility}'
            f'Use {self.list_tools_name} to list visible commands, '
            f'{self.describe_tool_name} <tool-or-command> for generated command help, '
            f'{self.search_tools_name} <keywords> to discover deferred tools, and '
            f'{self.call_tool_name} <tool-or-command> --json {{...}} as a generic fallback. '
            f'Use --{self.help_flag_name} or -h on shell-exposed commands for command help. '
            'Prefer shell-style flags for simple values and use JSON as the escape hatch for complex inputs.'
        )

    def _bash_tool_description(self) -> str:
        return (
            'Execute a bash script inside a persistent just-bash virtual shell. '
            'The shell keeps its virtual filesystem for the current agent run. '
            'Wrapped Pydantic AI tools are available inside the shell as commands. '
            'Use this when shell composition, pipelines, text processing, or iterative command workflows are useful.'
        )

    def _helper_name(self, suffix: str) -> str:
        return f'{self.helper_prefix}{suffix}'

    def _tool_command_name(self, tool_name: str) -> str:
        return f'{self.command_prefix}{tool_name}' if self.command_prefix else tool_name

    def _validate_configuration(self) -> None:
        if not self.help_flag_name:
            raise UserError('help_flag_name must not be empty.')
        if self.help_flag_name.startswith('-'):
            raise UserError('help_flag_name must not start with a dash.')

    def _validate_command_name(
        self,
        command_name: str,
        tool_name: str,
        bindings_by_command: Mapping[str, _CommandBinding[AgentDepsT]],
    ) -> None:
        helper_names = {
            self.list_tools_name,
            self.describe_tool_name,
            self.call_tool_name,
            self.search_tools_name,
            self.tool_name,
        }
        if command_name in helper_names:
            raise UserError(
                f'Tool {tool_name!r} maps to reserved bash helper command {command_name!r}. '
                'Rename the tool or change `command_prefix`/`helper_prefix`.'
            )
        if command_name in bindings_by_command:
            raise UserError(
                f'Wrapped tools map to the same shell command {command_name!r}. '
                'Rename a tool or choose a different `command_prefix`.'
            )

    def _build_command_binding(
        self,
        tool_name: str,
        command_name: str,
        tool: ToolsetTool[AgentDepsT],
    ) -> _CommandBinding[AgentDepsT]:
        shell_tool_def, shell_to_actual_arg_names, actual_to_shell_arg_names = self._build_shell_tool_def(
            tool_name,
            tool.tool_def,
        )
        hidden = tool.tool_def.defer_loading and tool_name not in self._discovered_tools
        return _CommandBinding(
            tool_name=tool_name,
            command_name=command_name,
            tool=tool,
            shell_tool_def=shell_tool_def,
            shell_to_actual_arg_names=shell_to_actual_arg_names,
            actual_to_shell_arg_names=actual_to_shell_arg_names,
            hidden=hidden,
        )

    def _build_shell_tool_def(
        self,
        tool_name: str,
        tool_def: ToolDefinition,
    ) -> tuple[ToolDefinition, dict[str, str], dict[str, str]]:
        schema = copy_json_object(tool_def.parameters_json_schema)
        properties = schema_properties(schema)
        required = required_names(schema)
        shell_to_actual_arg_names = {name: name for name in properties}
        actual_to_shell_arg_names = {name: name for name in properties}

        if self.help_flag_name in properties:
            if self.rename_help_argument is None:
                raise UserError(
                    f'Tool {tool_name!r} has an argument named {self.help_flag_name!r}, which conflicts with the reserved '
                    f'--{self.help_flag_name} help flag. Configure `rename_help_argument` or choose a different `help_flag_name`.'
                )
            renamed = self._renamed_help_argument_name(tool_name, self.help_flag_name)
            if not renamed:
                raise UserError(f'Renamed help argument for tool {tool_name!r} must not be empty.')
            if renamed.startswith('-'):
                raise UserError(f'Renamed help argument for tool {tool_name!r} must not start with a dash.')
            if renamed == self.help_flag_name:
                raise UserError(
                    f'Renamed help argument for tool {tool_name!r} resolved back to the reserved help flag name {renamed!r}.'
                )
            if renamed in properties:
                raise UserError(
                    f'Renamed help argument {renamed!r} for tool {tool_name!r} conflicts with an existing argument.'
                )
            properties[renamed] = properties.pop(self.help_flag_name)
            required = [renamed if item == self.help_flag_name else item for item in required]
            shell_to_actual_arg_names = {name: name for name in properties}
            shell_to_actual_arg_names[renamed] = self.help_flag_name
            actual_to_shell_arg_names = {actual: shell for shell, actual in shell_to_actual_arg_names.items()}

        schema['properties'] = properties
        if required:
            schema['required'] = required
        elif 'required' in schema:
            del schema['required']

        shell_tool_def = replace(tool_def, parameters_json_schema=schema, function_signature=None)
        return shell_tool_def, shell_to_actual_arg_names, actual_to_shell_arg_names

    def _renamed_help_argument_name(self, tool_name: str, arg_name: str) -> str:
        renamer = self.rename_help_argument
        if callable(renamer):
            return renamer(tool_name, arg_name)
        assert renamer is not None
        if '{tool_name}' in renamer or '{arg_name}' in renamer:
            return renamer.format(tool_name=tool_name, arg_name=arg_name)
        return renamer

    async def _as_toolset_tool(self, ctx: RunContext[AgentDepsT], tool: Tool[AgentDepsT]) -> ToolsetTool[AgentDepsT]:
        max_retries = 1
        tool_def = await tool.prepare_tool_def(
            replace(
                ctx,
                tool_name=tool.name,
                retry=ctx.retries.get(tool.name, 0),
                max_retries=max_retries,
            )
        )
        if tool_def is None:  # pragma: no cover
            raise RuntimeError(f'Public tool {tool.name!r} unexpectedly resolved to None.')
        return ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=max_retries,
            args_validator=tool.function_schema.validator,
        )

    def _public_tools(self) -> tuple[Tool[AgentDepsT], ...]:
        return (
            self._bash_tool,
            self._list_tools_tool,
            self._search_tools_tool,
            self._describe_tool_tool,
        )

    def _binding_summary(self, binding: _CommandBinding[AgentDepsT]) -> BashCommandInfo:
        return BashCommandInfo(
            tool_name=binding.tool_name,
            command=binding.command_name,
            description=binding.shell_tool_def.description,
        )

    def _script_with_shell_command_prelude(
        self,
        shell_state: _ShellState[AgentDepsT],
        script: str,
    ) -> str:
        visible_bindings = self._visible_bindings(shell_state)
        if not visible_bindings:
            return script

        prelude_lines = ['shopt -s expand_aliases']
        for binding in visible_bindings:
            alias_name = shlex.quote(binding.command_name)
            alias_target = shlex.quote(shlex.join([self.call_tool_name, binding.command_name]))
            prelude_lines.append(f'alias {alias_name}={alias_target}')
        return '\n'.join([*prelude_lines, script])

    def _visible_bindings(self, shell_state: _ShellState[AgentDepsT]) -> list[_CommandBinding[AgentDepsT]]:
        return sorted(
            [binding for binding in shell_state.bindings_by_tool.values() if not binding.hidden],
            key=lambda binding: binding.command_name,
        )

    def _discover_bindings(
        self,
        shell_state: _ShellState[AgentDepsT],
        keywords: str,
    ) -> list[_CommandBinding[AgentDepsT]]:
        if not keywords.strip():
            raise ModelRetry('Please provide search keywords.')

        terms = keywords.lower().split()
        matches: list[_CommandBinding[AgentDepsT]] = []
        for entry in shell_state.search_index:
            binding = shell_state.bindings_by_tool[entry.tool_name]
            if not binding.hidden:
                continue
            if any(term in entry.searchable_text for term in terms):
                self._discovered_tools.add(binding.tool_name)
                binding.hidden = False
                matches.append(binding)
                if len(matches) >= _MAX_SEARCH_RESULTS:
                    break
        return matches

    def _resolve_binding(self, name: str, shell_state: _ShellState[AgentDepsT]) -> _CommandBinding[AgentDepsT] | None:
        if name in shell_state.bindings_by_command:
            return shell_state.bindings_by_command[name]
        return shell_state.bindings_by_tool.get(name)

    def _shell_args_to_actual_args(
        self,
        binding: _CommandBinding[AgentDepsT],
        shell_args: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {binding.shell_to_actual_arg_names.get(key, key): value for key, value in shell_args.items()}

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

    def _copy_for_run(self, wrapped: AbstractToolset[AgentDepsT]) -> JustBashToolset[AgentDepsT]:
        return self._toolset_config().build_toolset(wrapped)

    def _require_shell_state(self) -> _ShellState[AgentDepsT]:
        if self._shell_state is None:  # pragma: no cover
            raise RuntimeError('No active just-bash shell state is available.')
        return self._shell_state

    def _execution_error_message(self, exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return f'execution failed: {message}'
        return f'execution failed with {exc.__class__.__name__}.'

    def _tool_retry_error_result(self, binding: _CommandBinding[AgentDepsT], exc: ToolRetryError) -> dict[str, Any]:
        content = exc.tool_retry.content
        if isinstance(content, list):
            lines = [f'{binding.command_name}: invalid arguments.']
            lines.extend(f'  {detail}' for detail in self._tool_retry_details(content))
            lines.append(command_help_hint(binding.command_name, self.help_flag_name))
            return self._error_result('\n'.join(lines), exit_code=2)
        return self._cli_error_result(
            binding.command_name,
            self._execution_error_message(exc),
            exit_code=1,
            hint=command_help_hint(binding.command_name, self.help_flag_name),
        )

    def _tool_retry_details(self, content: list[Any]) -> list[str]:
        details: list[str] = []
        for item in content:
            if isinstance(item, dict):
                loc = '.'.join(str(part) for part in item.get('loc') or ())
                msg = item.get('msg')
                if loc and isinstance(msg, str):
                    details.append(f'{loc}: {msg}')
                    continue
                if isinstance(msg, str):
                    details.append(msg)
                    continue
            details.append(str(item))
        return details

    def _cli_error_result(
        self,
        command_name: str,
        message: str,
        *,
        exit_code: int,
        usage: str | None = None,
        hint: str | None = None,
    ) -> dict[str, Any]:
        lines = [f'{command_name}: {message}']
        if usage:
            lines.append(usage)
        if hint:
            lines.append(hint)
        return self._error_result('\n'.join(lines), exit_code=exit_code)

    def _error_result(self, message: str, *, exit_code: int) -> dict[str, Any]:
        return {'stdout': '', 'stderr': f'{message}\n', 'exit_code': exit_code}
