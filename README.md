# pydantic-ai-just-bash

`pydantic-ai-just-bash` is a small extension package for [Pydantic AI](https://ai.pydantic.dev/) that adds a persistent [`just-py-bash`](https://github.com/nathan-gage/just-py-bash) execution environment to an agent.

The extension exposes a `just_bash` tool and binds wrapped Pydantic AI tools into that shell as commands.

That gives you a workflow like:

- the model calls `just_bash`
- the shell runs inside `just-py-bash`
- wrapped Pydantic AI tools are callable from the shell as commands
- deferred tools stay hidden until discovered via a shell-side search helper
- the shell filesystem persists across `just_bash` calls for the lifetime of the agent run

## Install

```bash
uv add pydantic-ai-just-bash
```

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_just_bash import JustBash

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[JustBash()],
)


@agent.tool_plain
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@agent.tool_plain(defer_loading=True)
def stock_lookup(symbol: str) -> str:
    """Look up a stock price."""
    return f"{symbol}=150.00"
```

Inside `just_bash`, the model can do things like:

```bash
pai_list_tools
get_weather Paris
pai_search_tools stock
stock_lookup AAPL
```

## Public API

### `JustBash`

A capability that wraps the agent's assembled toolset and adds a `just_bash` tool.

```python
from pydantic_ai_just_bash import JustBash

cap = JustBash(
    tool_name='just_bash',
    command_prefix='',
    helper_prefix='pai_',
    python=True,
)
```

### `JustBashToolset`

A lower-level wrapper if you want to wrap a specific toolset directly.

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai_just_bash import JustBashToolset

base = FunctionToolset()


@base.tool_plain
def echo(text: str) -> str:
    return text


agent = Agent('openai:gpt-5.2', toolsets=[JustBashToolset(base)])
```

## Shell helpers

By default the shell gets these helper commands:

- `pai_list_tools`
- `pai_describe_tool <tool-or-command>`
- `pai_call_tool <tool-or-command> --json '{...}'`
- `pai_search_tools <keywords>`

## Argument binding rules

Wrapped tools are still validated by Pydantic AI.

The shell command adapter accepts a few convenient input forms:

1. JSON object

```bash
my_tool --json '{"a": 1, "b": 2}'
```

2. Named flags

```bash
my_tool --a 1 --b 2
```

3. Single positional value for single-argument tools

```bash
get_weather Paris
```

4. JSON via stdin

```bash
echo '{"a": 1, "b": 2}' | my_tool --stdin-json
```

Use `--help` on a bound command, or `pai_describe_tool`, to inspect the generated signature and JSON schema.

## Notes

- The shell session is persistent for a run, so virtual filesystem changes carry across `just_bash` calls.
- The shell command set is captured on first shell use in a run. If your wrapped tools change dynamically and you want a fresh command set, call `just_bash` with `reset_session=True`.
- Deferred tools are hidden in the shell until discovered with `pai_search_tools`.
- Direct shell commands return the tool result. If a tool returns `ToolReturn`, the shell uses its `return_value`.
