from __future__ import annotations

from typing import TypeVar

import pytest
from just_bash import LazyFile
from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.usage import RunUsage

from pydantic_ai_just_bash import (
    BashDescribeToolResult,
    BashExecutionResult,
    BashListToolsResult,
    BashSearchToolsResult,
    JustBash,
    JustBashToolset,
)

pytestmark = pytest.mark.anyio

T = TypeVar('T')


def build_run_context(deps: T, run_step: int = 0) -> RunContext[T]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


async def test_bash_toolset_uses_new_default_public_names() -> None:
    wrapped = JustBashToolset(FunctionToolset[None]())
    tools = await wrapped.get_tools(build_run_context(None))

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

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'greet world'}))

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

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        flag_result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'add --a 2 --b 5'}))
        json_result = await manager.handle_call(
            ToolCallPart(tool_name='bash', args={'script': 'add --json \'{"a": 3, "b": 4}\''})
        )

    assert flag_result.stdout == '7'
    assert json_result.stdout == '7'


async def test_bash_tool_persists_filesystem_across_calls() -> None:
    async with JustBashToolset(FunctionToolset[None]()) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': "printf 'hello from fs' > note.txt"}))
        read_result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'cat note.txt'}))

    assert read_result.stdout == 'hello from fs'


async def test_bash_toolset_can_hide_wrapped_tools_but_keep_shell_commands() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with JustBashToolset(toolset, expose_wrapped_tools=False) as wrapped:
        tools = await wrapped.get_tools(build_run_context(None))
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
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
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
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
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
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
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
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

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        list_result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'bash_list_tools'}))
        search_result = await manager.handle_call(
            ToolCallPart(tool_name='bash', args={'script': 'bash_search_tools stock && stock_lookup AAPL'})
        )

    assert 'visible' in list_result.stdout
    assert 'bash_search_tools' in list_result.stdout
    assert 'AAPL' in search_result.stdout


async def test_bash_bound_commands_support_help_flag() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'greet --help'}))

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

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'show_value -- --help'}))

    assert result.exit_code == 0
    assert result.stdout == '--help'


async def test_bash_command_errors_suggest_json_for_multi_parameter_positionals() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'add 2 5'}))

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

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'repeat --count nope'}))

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
        await wrapped.get_tools(build_run_context(None))


async def test_bash_wrapper_can_rename_help_argument_for_shell_commands() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def explain(topic: str, help: str) -> str:
        """Explain a topic."""
        return f'{topic}:{help}'

    async with JustBashToolset(
        toolset,
        rename_help_argument='{tool_name}_{arg_name}',
    ) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
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

    async with JustBashToolset(toolset, help_flag_name='usage') as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': 'greet --usage'}))

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
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        result = await manager.handle_call(
            ToolCallPart(tool_name='shellbox', args={'script': 'jb_list_tools && cmd_echo hi'})
        )

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
    async with JustBashToolset(FunctionToolset[None]()) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))
        await manager.handle_call(ToolCallPart(tool_name='bash', args={'script': "printf 'hello' > note.txt"}))
        result = await manager.handle_call(
            ToolCallPart(tool_name='bash', args={'script': 'cat note.txt', 'reset_session': True})
        )

    assert result.exit_code != 0
    assert 'note.txt' in result.stderr
