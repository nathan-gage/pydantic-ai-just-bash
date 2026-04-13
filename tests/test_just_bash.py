from __future__ import annotations

from typing import TypeVar

import pytest
from just_bash import LazyFile
from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.usage import RunUsage

from pydantic_ai_just_bash import JustBash, JustBashExecutionResult, JustBashToolset

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


async def test_just_bash_toolset_executes_visible_tool_command() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))

        result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={
                    'script': 'greet world',
                },
            )
        )

    assert isinstance(result, JustBashExecutionResult)
    assert result.stdout == 'hello, world'
    assert result.stderr == ''
    assert result.exit_code == 0
    assert result.ok is True


async def test_just_bash_toolset_supports_json_and_flag_binding() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))

        flag_result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={
                    'script': 'add --a 2 --b 5',
                },
            )
        )
        json_result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={
                    'script': 'add --json \'{"a": 3, "b": 4}\'',
                },
            )
        )

    assert flag_result.stdout == '7'
    assert json_result.stdout == '7'


async def test_just_bash_toolset_persists_filesystem_across_calls() -> None:
    async with JustBashToolset(FunctionToolset[None]()) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))

        write_result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': "printf 'hello from fs' > note.txt"},
            )
        )
        read_result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': 'cat note.txt'},
            )
        )

    assert write_result.exit_code == 0
    assert read_result.stdout == 'hello from fs'


async def test_just_bash_toolset_hides_deferred_tools_until_shell_search() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:
        """Look up a stock price by ticker symbol."""
        return f'{symbol}=150.00'

    async with JustBashToolset(toolset) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))

        hidden_result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': 'stock_lookup AAPL'},
            )
        )
        discovered_result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': 'pai_search_tools stock && stock_lookup AAPL'},
            )
        )
        persisted_result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': 'stock_lookup MSFT'},
            )
        )

    assert hidden_result.exit_code != 0
    assert 'hidden' in hidden_result.stderr
    assert discovered_result.exit_code == 0
    assert 'AAPL=150.00' in discovered_result.stdout
    assert persisted_result.stdout == 'MSFT=150.00'


async def test_just_bash_toolset_generic_helper_can_call_selected_tools() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def echo(text: str) -> str:
        """Echo text."""
        return text

    async with JustBashToolset(toolset, command_prefix='tool.') as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))

        result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': 'pai_call_tool echo --json \'{"text": "hi"}\''},
            )
        )

    assert result.stdout == 'hi'


async def test_just_bash_toolset_can_filter_exposed_tools() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def visible() -> str:
        """Visible tool."""
        return 'visible'

    @toolset.tool_plain
    def hidden() -> str:
        """Hidden tool."""
        return 'hidden'

    async with JustBashToolset(toolset, exposed_tools=['visible']) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))

        result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': 'pai_list_tools'},
            )
        )

    assert 'visible' in result.stdout
    assert 'hidden' not in result.stdout


def test_just_bash_python_api_accepts_callback_lazy_file_provider() -> None:
    capability = JustBash(
        files={
            '/workspace/generated.txt': LazyFile(provider=lambda: 'generated at session start\n'),
        }
    )

    assert capability.files is not None
    lazy_file = capability.files['/workspace/generated.txt']
    assert isinstance(lazy_file, LazyFile)
    assert callable(lazy_file.provider)


async def test_just_bash_capability_adds_tool_to_agent() -> None:
    model = TestModel()
    agent = Agent(model, capabilities=[JustBash()])

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet a user."""
        return f'hello, {name}'

    await agent.run('What tools are available?')
    params = model.last_model_request_parameters
    assert params is not None

    tool_names = [tool.name for tool in params.function_tools]
    assert 'just_bash' in tool_names
    assert 'greet' in tool_names


async def test_just_bash_reset_session_clears_filesystem() -> None:
    async with JustBashToolset(FunctionToolset[None]()) as wrapped:
        manager = await ToolManager[None](wrapped).for_run_step(build_run_context(None))

        await manager.handle_call(ToolCallPart(tool_name='just_bash', args={'script': "printf 'hello' > note.txt"}))
        result = await manager.handle_call(
            ToolCallPart(
                tool_name='just_bash',
                args={'script': 'cat note.txt', 'reset_session': True},
            )
        )

    assert result.exit_code != 0
    assert 'note.txt' in result.stderr
