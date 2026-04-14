from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, TypedDict, Unpack

from pydantic_ai import ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_just_bash import BashExecutionResult, JustBashToolset


class BashCallArgs(TypedDict, total=False):
    stdin: str
    cwd: str
    env: dict[str, str]
    replace_env: bool
    raw_script: bool
    args: list[str]
    timeout: float
    reset_session: bool


@dataclass(slots=True)
class BashHarness:
    manager: ToolManager[None]
    tool_name: str

    async def run(self, script: str, /, **tool_args: Unpack[BashCallArgs]) -> BashExecutionResult:
        result = await self.manager.handle_call(
            ToolCallPart(tool_name=self.tool_name, args={'script': script, **tool_args})
        )
        assert isinstance(result, BashExecutionResult)
        return result


def build_run_context(run_step: int = 0) -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


async def build_shell_harness(wrapped: JustBashToolset[None]) -> BashHarness:
    manager = await ToolManager[None](wrapped).for_run_step(build_run_context())
    return BashHarness(manager=manager, tool_name=wrapped.tool_name)


@asynccontextmanager
async def open_bash(
    toolset: AbstractToolset[None],
    *,
    tool_name: str = 'bash',
    command_prefix: str = '',
    helper_prefix: str = 'bash_',
    exposed_tools: Literal['all'] | Sequence[str] = 'all',
    expose_wrapped_tools: bool = True,
    help_flag_name: str = 'help',
) -> AsyncIterator[BashHarness]:
    async with JustBashToolset(
        toolset,
        tool_name=tool_name,
        command_prefix=command_prefix,
        helper_prefix=helper_prefix,
        exposed_tools=exposed_tools,
        expose_wrapped_tools=expose_wrapped_tools,
        help_flag_name=help_flag_name,
    ) as wrapped:
        yield await build_shell_harness(wrapped)
