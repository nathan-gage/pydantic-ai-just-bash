from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, cast

from just_bash import AsyncBash, AsyncCustomCommandContext, JavaScriptConfig, NetworkConfig, ProcessInfo
from pydantic import BaseModel, TypeAdapter
from pydantic_ai import Tool
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart, ToolCallPart, ToolReturn
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

from ._types import JustBashFileSystemConfig, JustBashInitialFileValue

_ANY_JSON_TA = TypeAdapter(Any)
_MAX_SEARCH_RESULTS = 10


class JustBashExecutionResult(BaseModel):
    stdout: str = ''
    stderr: str = ''
    exit_code: int
    ok: bool


@dataclass
class _ShellSearchResult:
    tool_name: str
    command: str | None = None
    description: str | None = None


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
class _ShellState(Generic[AgentDepsT]):
    ctx: RunContext[AgentDepsT]
    tool_manager: ToolManager[AgentDepsT]
    command_to_tool: dict[str, str]
    hidden_tools: set[str]
    search_index: list[_SearchIndexEntry]


class JustBashToolset(AbstractToolset[AgentDepsT]):
    """Wrap a toolset and expose a persistent just-bash executor as a tool.

    The `just_bash` tool executes scripts inside a long-lived just-bash session.
    Wrapped Pydantic AI tools are available inside that session as custom shell
    commands, with deferred tools hidden until discovered via a shell-native
    search helper inspired by Pydantic AI's tool-search system.
    """

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_name: str = 'just_bash',
        command_prefix: str = '',
        helper_prefix: str = 'pai_',
        exposed_tools: ToolSelector[AgentDepsT] = 'all',
        instructions: str | None = None,
        files: Mapping[str, JustBashInitialFileValue] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        fs: JustBashFileSystemConfig | None = None,
        python: bool = False,
        javascript: bool | JavaScriptConfig = False,
        commands: Sequence[str] | None = None,
        network: NetworkConfig | None = None,
        process_info: ProcessInfo | None = None,
        node_command: Sequence[str] | None = None,
        js_entry: str | None = None,
        package_json: str | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.tool_name = tool_name
        self.command_prefix = command_prefix
        self.helper_prefix = helper_prefix
        self.exposed_tools = exposed_tools
        self.instructions = instructions
        self.files = files
        self.env = env
        self.cwd = cwd
        self.fs = fs
        self.python = python
        self.javascript = javascript
        self.commands = commands
        self.network = network
        self.process_info = process_info
        self.node_command = node_command
        self.js_entry = js_entry
        self.package_json = package_json

        self._bash: AsyncBash | None = None
        self._shell_state: _ShellState[AgentDepsT] | None = None
        self._discovered_tools: set[str] = set()
        self._bound_tool_commands: set[str] = set()
        self._tool = Tool(
            self._run_just_bash,
            takes_ctx=True,
            name=self.tool_name,
            description=self._tool_description(),
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
        tools = dict(await self.wrapped.get_tools(ctx))
        if self.tool_name in tools:
            raise UserError(
                f'Tool name {self.tool_name!r} is reserved by JustBashToolset. '
                'Rename the wrapped tool or change `tool_name`.'
            )

        max_retries = 1
        tool_def = await self._tool.prepare_tool_def(
            replace(
                ctx,
                tool_name=self.tool_name,
                retry=ctx.retries.get(self.tool_name, 0),
                max_retries=max_retries,
            )
        )
        if tool_def is None:  # pragma: no cover
            return tools

        tools[self.tool_name] = ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=max_retries,
            args_validator=self._tool.function_schema.validator,
        )
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if name == self.tool_name:
            return await self._run_just_bash(ctx, **tool_args)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Any) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(self, visitor: Any) -> AbstractToolset[AgentDepsT]:
        return self.__class__(
            wrapped=self.wrapped.visit_and_replace(visitor),
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

    async def _run_just_bash(
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
    ) -> JustBashExecutionResult:
        shell_state = await self._build_shell_state(ctx)
        self._shell_state = shell_state

        bash = await self._ensure_bash(shell_state, reset_session=reset_session)
        result = await bash.exec(
            script,
            stdin=stdin,
            cwd=cwd,
            env=env,
            replace_env=replace_env,
            raw_script=raw_script,
            args=args,
            timeout=timeout,
        )
        return JustBashExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            ok=result.ok,
        )

    async def _build_shell_state(self, ctx: RunContext[AgentDepsT]) -> _ShellState[AgentDepsT]:
        shell_ctx = replace(ctx)
        root_capability = ctx.tool_manager.root_capability if ctx.tool_manager is not None else None
        tool_manager = await ToolManager(
            toolset=self.wrapped,
            root_capability=root_capability,
        ).for_run_step(shell_ctx)

        available_tools = tool_manager.tools or {}
        command_to_tool: dict[str, str] = {}
        hidden_tools: set[str] = set()
        search_index: list[_SearchIndexEntry] = []

        for tool_name, tool in available_tools.items():
            if not await matches_tool_selector(self.exposed_tools, shell_ctx, tool.tool_def):
                continue

            command_name = self._tool_command_name(tool_name)
            self._validate_command_name(command_name, tool_name, command_to_tool)
            command_to_tool[command_name] = tool_name

            if tool.tool_def.defer_loading and tool_name not in self._discovered_tools:
                hidden_tools.add(tool_name)
                search_index.append(
                    _SearchIndexEntry(
                        tool_name=tool_name,
                        command_name=command_name,
                        description=tool.tool_def.description,
                    )
                )

        return _ShellState(
            ctx=shell_ctx,
            tool_manager=tool_manager,
            command_to_tool=command_to_tool,
            hidden_tools=hidden_tools,
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
            self._bound_tool_commands.clear()

        if self._bash is None:
            self._bound_tool_commands = set(shell_state.command_to_tool)
            self._bash = AsyncBash(
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
                custom_commands=self._custom_commands(),
            )

        return self._bash

    async def _close_bash(self) -> None:
        if self._bash is not None:
            bash = self._bash
            self._bash = None
            await bash.close()

    def _custom_commands(self) -> dict[str, Any]:
        commands: dict[str, Any] = {
            self._helper_name('list_tools'): self._cmd_list_tools,
            self._helper_name('describe_tool'): self._cmd_describe_tool,
            self._helper_name('call_tool'): self._cmd_call_tool,
            self._helper_name('search_tools'): self._cmd_search_tools,
        }
        for command_name in sorted(self._bound_tool_commands):
            commands[command_name] = self._make_bound_tool_command(command_name)
        return commands

    def _make_bound_tool_command(self, command_name: str) -> Any:
        async def _command(args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
            return await self._cmd_bound_tool(command_name, args, ctx)

        return _command

    async def _cmd_bound_tool(
        self,
        command_name: str,
        args: list[str],
        ctx: AsyncCustomCommandContext,
    ) -> dict[str, Any]:
        shell_state = self._require_shell_state()
        tool_name = shell_state.command_to_tool.get(command_name)
        if tool_name is None:
            return self._error_result(
                f'Command {command_name!r} is not available in the current run step. '
                f'Use {self._helper_name("call_tool")} for generic access.',
                exit_code=127,
            )
        return await self._invoke_tool(tool_name, command_name, args, ctx)

    async def _cmd_call_tool(self, args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
        if not args:
            return self._error_result(
                f'Usage: {self._helper_name("call_tool")} <tool-or-command> [tool args...]',
                exit_code=2,
            )

        shell_state = self._require_shell_state()
        name = args[0]
        tool_name = self._resolve_tool_name(name, shell_state)
        if tool_name is None:
            return self._error_result(f'Unknown tool or command: {name!r}', exit_code=127)

        command_name = self._tool_command_name(tool_name)
        return await self._invoke_tool(tool_name, command_name, args[1:], ctx)

    async def _cmd_describe_tool(self, args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
        del ctx
        if len(args) != 1:
            return self._error_result(
                f'Usage: {self._helper_name("describe_tool")} <tool-or-command>',
                exit_code=2,
            )

        shell_state = self._require_shell_state()
        tool_name = self._resolve_tool_name(args[0], shell_state)
        if tool_name is None:
            return self._error_result(f'Unknown tool or command: {args[0]!r}', exit_code=127)

        tool = (shell_state.tool_manager.tools or {}).get(tool_name)
        if tool is None:
            return self._error_result(f'Tool {tool_name!r} is not available in the current run step.', exit_code=127)

        command_name = self._tool_command_name(tool_name)
        return {
            'stdout': self._render_help(tool.tool_def, command_name),
            'stderr': '',
            'exit_code': 0,
        }

    async def _cmd_list_tools(self, args: list[str], ctx: AsyncCustomCommandContext) -> dict[str, Any]:
        del args, ctx
        shell_state = self._require_shell_state()
        visible_commands = sorted(
            command_name
            for command_name, tool_name in shell_state.command_to_tool.items()
            if tool_name not in shell_state.hidden_tools and command_name in self._bound_tool_commands
        )

        lines = list(visible_commands)
        hidden_count = sum(
            1
            for tool_name in shell_state.hidden_tools
            if self._tool_command_name(tool_name) in self._bound_tool_commands
        )
        if hidden_count:
            lines.append(
                f'[{hidden_count} hidden tool commands available via {self._helper_name("search_tools")} <keywords>]'
            )
        lines.append(f'[generic helper: {self._helper_name("call_tool")} <tool-or-command> --json {{...}}]')

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
            return self._error_result(
                f'Usage: {self._helper_name("search_tools")} <keywords>',
                exit_code=2,
            )

        terms = keywords.lower().split()
        matches: list[_ShellSearchResult] = []
        discovered_now: list[str] = []
        for entry in shell_state.search_index:
            if any(term in entry.searchable_text for term in terms):
                self._discovered_tools.add(entry.tool_name)
                discovered_now.append(entry.tool_name)
                matches.append(
                    _ShellSearchResult(
                        tool_name=entry.tool_name,
                        command=entry.command_name if entry.command_name in self._bound_tool_commands else None,
                        description=entry.description,
                    )
                )
                if len(matches) >= _MAX_SEARCH_RESULTS:
                    break

        if not matches:
            return {
                'stdout': 'No matching tools found. The tools you need may not be available.\n',
                'stderr': '',
                'exit_code': 0,
            }

        hint = ''
        if any(match.command is None for match in matches):
            hint = (
                f'\nUse {self._helper_name("call_tool")} <tool_name> --json {{...}} '
                'for tools without a direct shell command in the current session.'
            )
        payload = cast(
            Any,
            [
                {
                    'tool_name': match.tool_name,
                    'command': match.command,
                    'description': match.description,
                }
                for match in matches
            ],
        )
        stdout = _ANY_JSON_TA.dump_json(payload).decode() + hint
        return {'stdout': stdout, 'stderr': '', 'exit_code': 0}

    async def _invoke_tool(
        self,
        tool_name: str,
        command_name: str,
        args: list[str],
        ctx: AsyncCustomCommandContext,
    ) -> dict[str, Any]:
        shell_state = self._require_shell_state()
        tool = (shell_state.tool_manager.tools or {}).get(tool_name)
        if tool is None:
            return self._error_result(f'Tool {tool_name!r} is not available in the current run step.', exit_code=127)

        if tool_name in shell_state.hidden_tools and tool_name not in self._discovered_tools:
            return self._error_result(
                f'Tool {tool_name!r} is hidden. Use {self._helper_name("search_tools")} <keywords> to discover it first.',
                exit_code=127,
            )

        if args and args[0] in {'--help', '-h'}:
            return {
                'stdout': self._render_help(tool.tool_def, command_name),
                'stderr': '',
                'exit_code': 0,
            }

        try:
            parsed_args = self._bind_command_args(tool.tool_def, args, ctx.stdin)
        except ValueError as exc:
            return self._error_result(str(exc), exit_code=2)

        try:
            result = await shell_state.tool_manager.handle_call(ToolCallPart(tool_name=tool_name, args=parsed_args))
        except Exception as exc:
            return self._error_result(str(exc), exit_code=1)

        return {
            'stdout': self._serialize_tool_result(result),
            'stderr': '',
            'exit_code': 0,
        }

    def _bind_command_args(self, tool_def: ToolDefinition, args: list[str], stdin: str) -> dict[str, Any]:
        properties = self._schema_properties(tool_def)
        if not properties:
            if args:
                raise ValueError(f'{tool_def.name!r} does not take any arguments.')
            return {}

        parsed: dict[str, Any] = {}
        positionals: list[str] = []
        json_payload: str | None = None
        stdin_json = False

        i = 0
        while i < len(args):
            token = args[i]
            if token == '--stdin-json':
                stdin_json = True
                i += 1
                continue
            if token == '--json':
                if i + 1 >= len(args):
                    raise ValueError('Expected a JSON object after --json.')
                json_payload = args[i + 1]
                i += 2
                continue
            if token.startswith('--json='):
                json_payload = token.split('=', 1)[1]
                i += 1
                continue
            if token.startswith('--no-'):
                prop_name = self._resolve_flag_name(token[5:], properties)
                parsed[prop_name] = False
                i += 1
                continue
            if token.startswith('--'):
                name, inline_value = self._split_flag(token)
                prop_name = self._resolve_flag_name(name, properties)
                prop_schema = properties[prop_name]
                if inline_value is None:
                    if self._is_boolean_schema(prop_schema):
                        next_token = args[i + 1] if i + 1 < len(args) else None
                        if next_token is None or next_token.startswith('--'):
                            parsed[prop_name] = True
                            i += 1
                            continue
                    if i + 1 >= len(args):
                        raise ValueError(f'Expected a value after --{name}.')
                    inline_value = args[i + 1]
                    i += 2
                else:
                    i += 1
                parsed[prop_name] = self._coerce_flag_value(prop_schema, inline_value)
                continue

            positionals.append(token)
            i += 1

        if json_payload is not None:
            if parsed or positionals or stdin_json:
                raise ValueError('Do not mix --json with flags, positionals, or --stdin-json.')
            return self._parse_json_object(json_payload)

        if stdin_json:
            if parsed or positionals:
                raise ValueError('Do not mix --stdin-json with flags or positional arguments.')
            return self._parse_json_object(stdin)

        if parsed:
            if positionals:
                raise ValueError('Do not mix named flags with positional arguments.')
            return parsed

        if positionals:
            if len(properties) == 1:
                prop_name, prop_schema = next(iter(properties.items()))
                return {prop_name: self._coerce_single_argument(prop_schema, positionals)}
            if len(positionals) == 1 and positionals[0].lstrip().startswith('{'):
                return self._parse_json_object(positionals[0])
            raise ValueError(
                'This tool has multiple parameters. Use named flags like --param value or pass a JSON object with --json.'
            )

        if stdin:
            if len(properties) == 1:
                prop_name, prop_schema = next(iter(properties.items()))
                return {prop_name: self._coerce_stdin_argument(prop_schema, stdin)}
            return self._parse_json_object(stdin)

        return {}

    def _render_help(self, tool_def: ToolDefinition, command_name: str) -> str:
        lines = [f'Command: {command_name}']
        if tool_def.description:
            lines.extend(['', tool_def.description])

        lines.extend(
            [
                '',
                'Input forms:',
                f'  {command_name} --json "{{...}}"',
                f'  {command_name} --field value [--other value ...]',
                f'  {command_name} <value>  # for single-argument tools',
                f'  printf "{{...}}" | {command_name} --stdin-json',
            ]
        )

        if tool_def.function_signature is not None:
            lines.extend(['', 'Signature:', tool_def.render_signature('...')])

        lines.extend(
            [
                '',
                'JSON schema:',
                json.dumps(tool_def.parameters_json_schema, indent=2, sort_keys=True),
            ]
        )
        return '\n'.join(lines) + '\n'

    def _serialize_tool_result(self, result: Any) -> str:
        if isinstance(result, ToolReturn):
            result = result.return_value
        if result is None:
            return ''
        if isinstance(result, str):
            return result
        return _ANY_JSON_TA.dump_json(result).decode()

    def _tool_description(self) -> str:
        return (
            'Execute a bash script inside a persistent just-bash virtual shell. '
            'The shell keeps its virtual filesystem for the current agent run. '
            'Wrapped Pydantic AI tools are available inside the shell as commands. '
            'Use this when shell composition, pipelines, text processing, or iterative command workflows are useful.'
        )

    def _default_instructions(self) -> str:
        return (
            f'Use {self.tool_name} when it is helpful to orchestrate multiple tool calls through a shell-like workflow. '
            'Inside the shell, wrapped tools are exposed as commands. '
            f'Use {self._helper_name("list_tools")} to list visible commands, '
            f'{self._helper_name("describe_tool")} <tool-or-command> for usage details, '
            f'{self._helper_name("search_tools")} <keywords> to discover deferred tools, and '
            f'{self._helper_name("call_tool")} <tool-or-command> --json {{...}} as a generic fallback. '
            'For single-argument string tools, positional arguments usually work; for complex inputs, prefer --json.'
        )

    def _helper_name(self, suffix: str) -> str:
        return f'{self.helper_prefix}{suffix}'

    def _tool_command_name(self, tool_name: str) -> str:
        return f'{self.command_prefix}{tool_name}' if self.command_prefix else tool_name

    def _validate_command_name(
        self,
        command_name: str,
        tool_name: str,
        command_to_tool: dict[str, str],
    ) -> None:
        helper_names = {
            self._helper_name('list_tools'),
            self._helper_name('describe_tool'),
            self._helper_name('call_tool'),
            self._helper_name('search_tools'),
            self.tool_name,
        }
        if command_name in helper_names:
            raise UserError(
                f'Tool {tool_name!r} maps to reserved just-bash helper command {command_name!r}. '
                'Rename the tool or change `command_prefix`/`helper_prefix`.'
            )
        if command_name in command_to_tool:
            raise UserError(
                f'Wrapped tools map to the same shell command {command_name!r}. '
                'Rename a tool or choose a different `command_prefix`.'
            )

    def _resolve_tool_name(self, name: str, shell_state: _ShellState[AgentDepsT]) -> str | None:
        if name in shell_state.command_to_tool:
            return shell_state.command_to_tool[name]
        tools = shell_state.tool_manager.tools or {}
        if name in tools:
            return name
        return None

    def _require_shell_state(self) -> _ShellState[AgentDepsT]:
        if self._shell_state is None:  # pragma: no cover
            raise RuntimeError('No active just-bash shell state is available.')
        return self._shell_state

    def _copy_for_run(self, wrapped: AbstractToolset[AgentDepsT]) -> JustBashToolset[AgentDepsT]:
        return self.__class__(
            wrapped=wrapped,
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

    def _schema_properties(self, tool_def: ToolDefinition) -> dict[str, dict[str, Any]]:
        properties = tool_def.parameters_json_schema.get('properties') or {}
        return cast(dict[str, dict[str, Any]], properties)

    def _split_flag(self, token: str) -> tuple[str, str | None]:
        name = token[2:]
        if '=' in name:
            flag_name, value = name.split('=', 1)
            return flag_name, value
        return name, None

    def _resolve_flag_name(self, flag_name: str, properties: Mapping[str, Any]) -> str:
        candidates = [flag_name, flag_name.replace('-', '_'), flag_name.replace('_', '-')]
        for candidate in candidates:
            if candidate in properties:
                return candidate
        raise ValueError(f'Unknown argument --{flag_name}.')

    def _coerce_flag_value(self, prop_schema: Mapping[str, Any], raw_value: str) -> Any:
        if self._expects_json_payload(prop_schema):
            return json.loads(raw_value)
        return raw_value

    def _coerce_single_argument(self, prop_schema: Mapping[str, Any], values: list[str]) -> Any:
        if self._is_array_schema(prop_schema):
            return values
        if self._expects_json_payload(prop_schema) and len(values) == 1:
            return json.loads(values[0])
        if self._is_string_schema(prop_schema):
            return ' '.join(values)
        if len(values) == 1:
            return values[0]
        return values

    def _coerce_stdin_argument(self, prop_schema: Mapping[str, Any], stdin: str) -> Any:
        if self._expects_json_payload(prop_schema):
            return json.loads(stdin)
        if self._is_array_schema(prop_schema):
            return [line for line in stdin.splitlines() if line]
        return stdin

    def _parse_json_object(self, payload: str) -> dict[str, Any]:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON payload: {exc.msg}') from exc
        if not isinstance(parsed, dict):
            raise ValueError('Expected a JSON object.')
        return cast(dict[str, Any], parsed)

    def _expects_json_payload(self, schema: Mapping[str, Any]) -> bool:
        schema_type = schema.get('type')
        if schema_type in {'object', 'array'}:
            return True
        for key in ('anyOf', 'oneOf', 'allOf'):
            variants = schema.get(key)
            if isinstance(variants, list) and any(
                self._expects_json_payload(cast(Mapping[str, Any], item)) for item in variants
            ):
                return True
        return False

    def _is_array_schema(self, schema: Mapping[str, Any]) -> bool:
        schema_type = schema.get('type')
        return schema_type == 'array'

    def _is_string_schema(self, schema: Mapping[str, Any]) -> bool:
        schema_type = schema.get('type')
        if schema_type == 'string':
            return True
        if schema_type is None and 'enum' in schema:
            return True
        return False

    def _is_boolean_schema(self, schema: Mapping[str, Any]) -> bool:
        schema_type = schema.get('type')
        if schema_type == 'boolean':
            return True
        for key in ('anyOf', 'oneOf', 'allOf'):
            variants = schema.get(key)
            if isinstance(variants, list) and any(
                self._is_boolean_schema(cast(Mapping[str, Any], item)) for item in variants
            ):
                return True
        return False

    def _error_result(self, message: str, *, exit_code: int) -> dict[str, Any]:
        return {'stdout': '', 'stderr': f'{message}\n', 'exit_code': exit_code}
