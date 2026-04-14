from __future__ import annotations

from collections.abc import Callable

import pytest
from just_bash import LazyFile
from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_just_bash import (
    BashDescribeToolResult,
    BashExecutionResult,
    BashListToolsResult,
    BashSearchToolsResult,
    JustBash,
    JustBashToolset,
)
from tests._helpers import build_run_context, build_shell_harness, open_bash

pytestmark = pytest.mark.anyio


def only_run_steps(*allowed_steps: int) -> Callable[[object, ToolDefinition], ToolDefinition | None]:
    allowed = frozenset(allowed_steps)

    def prepare(ctx: object, tool_def: ToolDefinition) -> ToolDefinition | None:
        run_step = getattr(ctx, 'run_step', None)
        return tool_def if run_step in allowed else None

    return prepare


async def test_bash_toolset_uses_new_default_public_names() -> None:
    wrapped = JustBashToolset(FunctionToolset[None]())
    tools = await wrapped.get_tools(build_run_context())

    assert 'bash' in tools
    assert 'bash_list_tools' in tools
    assert 'bash_search_tools' in tools
    assert 'bash_describe_tool' in tools
    assert 'just_bash' not in tools


async def test_bash_tool_executes_visible_tool_command() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with open_bash(toolset) as shell:
        result = await shell.run('greet world')

    assert isinstance(result, BashExecutionResult)
    assert result.stdout == 'hello, world'
    assert result.stderr == ''
    assert result.exit_code == 0
    assert result.ok is True


async def test_bash_toolset_supports_json_and_flag_binding() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with open_bash(toolset) as shell:
        flag_result = await shell.run('add --a 2 --b 5')
        json_result = await shell.run('add --json \'{"a": 3, "b": 4}\'')

    assert flag_result.stdout == '7'
    assert json_result.stdout == '7'


async def test_bash_toolset_supports_boolean_flags_and_array_positionals() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def flag_to_int(enabled: bool) -> int:
        """Convert a boolean flag to an integer."""
        return 1 if enabled else 0

    @toolset.tool_plain
    def join_lines(lines: list[str]) -> str:
        """Join stdin lines."""
        return ','.join(lines)

    async with open_bash(toolset) as shell:
        enabled_result = await shell.run('flag_to_int --enabled')
        disabled_result = await shell.run('flag_to_int --no-enabled')
        positional_result = await shell.run('join_lines one two')

    assert enabled_result.stdout == '1'
    assert disabled_result.stdout == '0'
    assert positional_result.stdout == 'one,two'


async def test_bash_tool_persists_filesystem_across_calls() -> None:
    async with open_bash(FunctionToolset[None]()) as shell:
        write_result = await shell.run("printf 'hello from fs' > note.txt")
        read_result = await shell.run('cat note.txt')

    assert write_result.exit_code == 0
    assert read_result.stdout == 'hello from fs'


async def test_bash_toolset_can_hide_wrapped_tools_but_keep_shell_commands() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with JustBashToolset(toolset, expose_wrapped_tools=False) as wrapped:
        tools = await wrapped.get_tools(build_run_context())
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context())
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'greet world'}))

    assert 'greet' not in tools
    assert 'bash' in tools
    assert result.stdout == 'hello, world'


async def test_bash_list_tools_exposed_as_top_level_tool() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def visible() -> str:
        """Visible tool."""
        return 'visible'

    @toolset.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:
        """Look up a stock price."""
        return symbol

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context())
        result = await manager.handle_call(ToolCallPart(tool_name='bash_list_tools', args={}))

    assert isinstance(result, BashListToolsResult)
    assert [command.command for command in result.commands] == ['visible']
    assert result.hidden_count == 1


async def test_bash_search_tools_exposed_as_top_level_tool_and_unhides_commands() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:
        """Look up a stock price by ticker symbol."""
        return f'{symbol}=150.00'

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context())
        search_result = await manager.handle_call(
            ToolCallPart(tool_name='bash_search_tools', args={'keywords': 'stock'})
        )
        bash_result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'stock_lookup AAPL'}))

    assert isinstance(search_result, BashSearchToolsResult)
    assert [match.command for match in search_result.matches] == ['stock_lookup']
    assert bash_result.stdout == 'AAPL=150.00'


async def test_bash_describe_tool_returns_generated_help() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context())
        result = await manager.handle_call(ToolCallPart(tool_name='bash_describe_tool', args={'name': 'greet'}))

    assert isinstance(result, BashDescribeToolResult)
    assert result.command == 'greet'
    assert result.hidden is False
    assert '--help' in result.help_text
    assert 'Greet a user.' in result.help_text
    assert 'JSON SCHEMA' in result.help_text


async def test_shell_helpers_use_bash_names() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def visible() -> str:
        """Visible tool."""
        return 'visible'

    @toolset.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:
        """Look up a stock price."""
        return symbol

    async with open_bash(toolset) as shell:
        list_result = await shell.run('bash_list_tools')
        search_result = await shell.run('bash_search_tools stock && stock_lookup AAPL')

    assert 'visible' in list_result.stdout
    assert 'bash_search_tools' in list_result.stdout
    assert 'AAPL' in search_result.stdout


async def test_hidden_shell_commands_can_appear_in_shell_introspection_before_discovery() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:
        """Look up a stock price by ticker symbol."""
        return f'{symbol}=150.00'

    async with open_bash(toolset) as shell:
        before_discovery = await shell.run('type -t stock_lookup')
        hidden_call = await shell.run('stock_lookup AAPL')
        discovered_call = await shell.run('bash_search_tools stock >/dev/null && stock_lookup AAPL')
        after_discovery = await shell.run('type -t stock_lookup')

    assert before_discovery.stdout.strip() == 'alias'
    assert hidden_call.exit_code == 127
    assert 'command is hidden until it is discovered' in hidden_call.stderr
    assert discovered_call.stdout == 'AAPL=150.00'
    assert after_discovery.stdout.strip() == 'alias'


async def test_bash_bound_commands_support_help_flag() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with open_bash(toolset) as shell:
        result = await shell.run('greet --help')

    assert result.exit_code == 0
    assert 'USAGE' in result.stdout
    assert '--help' in result.stdout
    assert 'Greet a user.' in result.stdout


async def test_bash_bound_commands_support_double_dash_for_literal_values() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def show_value(text: str) -> str:
        """Show a literal value."""
        return text

    async with open_bash(toolset) as shell:
        result = await shell.run('show_value -- --help')

    assert result.exit_code == 0
    assert result.stdout == '--help'


async def test_bash_command_errors_suggest_json_for_multi_parameter_positionals() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with open_bash(toolset) as shell:
        result = await shell.run('add 2 5')

    assert result.exit_code == 2
    assert 'add: this command has multiple parameters.' in result.stderr
    assert '--json' in result.stderr
    assert "Try 'add --help' for usage." in result.stderr


async def test_bash_validation_errors_are_cli_style() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def repeat(count: int) -> str:
        """Repeat a counter value."""
        return str(count)

    async with open_bash(toolset) as shell:
        result = await shell.run('repeat --count nope')

    assert result.exit_code == 2
    assert 'repeat: invalid arguments.' in result.stderr
    assert 'count:' in result.stderr
    assert 'validation error' not in result.stderr.lower()


async def test_bash_wrapper_errors_when_tool_has_reserved_help_argument() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def explain(topic: str, help: str) -> str:
        """Explain a topic."""
        return f'{topic}:{help}'

    wrapped = JustBashToolset(toolset)

    with pytest.raises(UserError, match='reserved --help help flag'):
        await wrapped.get_tools(build_run_context())


async def test_bash_wrapper_can_rename_help_argument_for_shell_commands() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def explain(topic: str, help: str) -> str:
        """Explain a topic."""
        return f'{topic}:{help}'

    async with JustBashToolset(toolset, rename_help_argument='{tool_name}_{arg_name}') as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context())
        describe_result = await manager.handle_call(
            ToolCallPart(tool_name='bash_describe_tool', args={'name': 'explain'})
        )
        bash_result = await manager.handle_call(
            ToolCallPart(
                tool_name='bash',
                args={'script': 'explain --topic shell --explain_help details'},
            )
        )

    assert '--explain_help' in describe_result.help_text
    assert '--help' in describe_result.help_text
    assert 'NOTES' in describe_result.help_text
    assert "original tool argument 'help' is exposed as --explain_help" in describe_result.help_text
    assert bash_result.stdout == 'shell:details'


async def test_bash_wrapper_supports_custom_help_flag_name() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with open_bash(toolset, help_flag_name='usage') as shell:
        result = await shell.run('greet --usage')

    assert result.exit_code == 0
    assert '--usage' in result.stdout
    assert '-h' in result.stdout


def test_bash_python_api_accepts_callback_lazy_file_provider() -> None:
    capability = JustBash(
        files={
            '/workspace/generated.txt': LazyFile(provider=lambda: 'generated at session start\n'),
        }
    )

    assert capability.files is not None
    lazy_file = capability.files['/workspace/generated.txt']
    assert isinstance(lazy_file, LazyFile)
    assert callable(lazy_file.provider)


async def test_bash_capability_exposes_bash_public_tools_to_agent_by_default() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, capabilities=[JustBash()])

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    await agent.run('What tools are available?')
    params = model.last_model_request_parameters
    assert params is not None

    tool_names = [tool.name for tool in params.function_tools]
    assert 'bash' in tool_names
    assert 'bash_list_tools' in tool_names
    assert 'bash_search_tools' in tool_names
    assert 'bash_describe_tool' in tool_names
    assert 'greet' in tool_names


async def test_bash_capability_forwards_wrapper_configuration() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def echo(text: str) -> str:
        """Echo text."""
        return text

    capability = JustBash(tool_name='shellbox', command_prefix='cmd_', helper_prefix='jb_')
    wrapped = capability.get_wrapper_toolset(toolset)

    assert isinstance(wrapped, JustBashToolset)

    async with wrapped:
        shell = await build_shell_harness(wrapped)
        result = await shell.run('jb_list_tools && cmd_echo hi')

    assert 'cmd_echo' in result.stdout
    assert result.stdout.strip().endswith('hi')


async def test_bash_capability_supports_shell_only_mode() -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, capabilities=[JustBash(expose_wrapped_tools=False)])

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    await agent.run('What tools are available?')
    params = model.last_model_request_parameters
    assert params is not None

    tool_names = [tool.name for tool in params.function_tools]
    assert 'bash' in tool_names
    assert 'bash_list_tools' in tool_names
    assert 'bash_search_tools' in tool_names
    assert 'bash_describe_tool' in tool_names
    assert 'greet' not in tool_names


async def test_bash_reset_session_clears_filesystem() -> None:
    async with open_bash(FunctionToolset[None]()) as shell:
        await shell.run("printf 'hello' > note.txt")
        result = await shell.run('cat note.txt', reset_session=True)

    assert result.exit_code != 0
    assert 'note.txt' in result.stderr


async def test_bash_refreshes_visible_commands_across_run_steps_without_resetting_session() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(prepare=only_run_steps(0))
    def first_tool() -> str:
        """Return the first-step tool marker."""
        return 'first\n'

    @toolset.tool_plain(prepare=only_run_steps(1))
    def second_tool() -> str:
        """Return the second-step tool marker."""
        return 'second\n'

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(0))
        step0_result = await manager.handle_call(
            ToolCallPart(tool_name='bash', args={'script': "printf 'persisted\\n' > note.txt && first_tool"})
        )

        manager = await manager.for_run_step(build_run_context(1))
        list_result = await manager.handle_call(ToolCallPart(tool_name='bash_list_tools', args={}))
        step1_result = await manager.handle_call(
            ToolCallPart(tool_name='bash', args={'script': 'cat note.txt && second_tool'})
        )
        missing_old_command = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'first_tool'}))

    assert isinstance(step0_result, BashExecutionResult)
    assert isinstance(list_result, BashListToolsResult)
    assert isinstance(step1_result, BashExecutionResult)
    assert isinstance(missing_old_command, BashExecutionResult)
    assert step0_result.stdout == 'first\n'
    assert [command.command for command in list_result.commands] == ['second_tool']
    assert step1_result.stdout == 'persisted\nsecond\n'
    assert missing_old_command.exit_code != 0
    assert 'first_tool' in missing_old_command.stderr


async def test_bash_preserves_discovered_deferred_commands_across_run_step_refresh() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(prepare=only_run_steps(0))
    def first_tool() -> str:
        """Return the first-step tool marker."""
        return 'first\n'

    @toolset.tool_plain(prepare=only_run_steps(1))
    def second_tool() -> str:
        """Return the second-step tool marker."""
        return 'second\n'

    @toolset.tool_plain(defer_loading=True, prepare=only_run_steps(0, 1))
    def stock_lookup(symbol: str) -> str:
        """Look up a stock price by ticker symbol."""
        return f'{symbol}=150.00\n'

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(0))
        discover_and_call = await manager.handle_call(
            ToolCallPart(
                tool_name='bash',
                args={
                    'script': 'bash_search_tools stock >/dev/null && stock_lookup AAPL && printf "saved\\n" > note.txt'
                },
            )
        )

        manager = await manager.for_run_step(build_run_context(1))
        list_result = await manager.handle_call(ToolCallPart(tool_name='bash_list_tools', args={}))
        step1_result = await manager.handle_call(
            ToolCallPart(tool_name='bash', args={'script': 'cat note.txt && stock_lookup MSFT && second_tool'})
        )

    assert isinstance(discover_and_call, BashExecutionResult)
    assert isinstance(list_result, BashListToolsResult)
    assert isinstance(step1_result, BashExecutionResult)
    assert discover_and_call.stdout == 'AAPL=150.00\n'
    assert [command.command for command in list_result.commands] == ['second_tool', 'stock_lookup']
    assert step1_result.stdout == 'saved\nMSFT=150.00\nsecond\n'
