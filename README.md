# pydantic-ai-just-bash

[![Source](https://img.shields.io/badge/source-GitHub-black?logo=github)](https://github.com/nathan-gage/pydantic-ai-just-bash)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`pydantic-ai-just-bash` is a small extension package for [Pydantic AI](https://ai.pydantic.dev/) that gives an agent a persistent [`just-py-bash`](https://github.com/nathan-gage/just-py-bash) shell.

It adds a `just_bash` tool and binds wrapped Pydantic AI tools into that shell as commands, so a model can mix shell workflows with normal tool calls while keeping a long-lived virtual filesystem for the lifetime of the agent run.

## Why use it?

- Give an agent a persistent bash-like environment without a real OS shell
- Expose normal Pydantic AI tools as shell commands
- Keep deferred tools hidden until the model discovers them with shell-side search
- Reuse `just-py-bash` session controls like `files`, `env`, `cwd`, `fs`, `python`, and `javascript`
- Return structured execution results instead of raw subprocess plumbing

## Install

> Requires Python 3.11+

| Use case | Command |
| --- | --- |
| Install from PyPI with `uv` | `uv add pydantic-ai-just-bash` |
| Install from PyPI with `pip` | `pip install pydantic-ai-just-bash` |
| Install a cloned checkout in editable mode | `pip install -e .` |

Install name: `pydantic-ai-just-bash`  
Import name: `pydantic_ai_just_bash`

This package only adds the shell capability. Install whatever Pydantic AI model/provider extras you already need separately.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_just_bash import JustBash

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[JustBash(python=True)],
)


@agent.tool_plain
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f'Sunny in {city}'


@agent.tool_plain(defer_loading=True)
def stock_lookup(symbol: str) -> str:
    """Look up a stock price."""
    return f'{symbol}=150.00'
```

Inside `just_bash`, the model can do things like:

```bash
pai_list_tools
get_weather Paris
printf 'draft note' > note.txt
cat note.txt
pai_search_tools stock
stock_lookup AAPL
```

The shell session and virtual filesystem persist across `just_bash` calls for the lifetime of the agent run.

## API overview

### `JustBash`

Use `JustBash` as an agent capability. It wraps the assembled toolset and injects a `just_bash` tool.

```python
from pydantic_ai_just_bash import JustBash

capability = JustBash(
    tool_name='just_bash',
    command_prefix='',
    helper_prefix='pai_',
    python=True,
)
```

Common configuration knobs include:

- `tool_name`
- `command_prefix`
- `helper_prefix`
- `exposed_tools`
- `instructions`
- `files`, `env`, `cwd`, `fs`
- `python`, `javascript`, `commands`, `network`, `process_info`
- `node_command`, `js_entry`, `package_json`

### `JustBashToolset`

Use `JustBashToolset` if you want to wrap a specific toolset directly instead of using a capability.

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai_just_bash import JustBashToolset

base = FunctionToolset()


@base.tool_plain
def echo(text: str) -> str:
    return text


agent = Agent('openai:gpt-5.2', toolsets=[JustBashToolset(base)])
```

### `JustBashExecutionResult`

The `just_bash` tool returns a structured result object with:

- `stdout`
- `stderr`
- `exit_code`
- `ok`

## Shell helpers

By default the shell gets these helper commands:

| Command | Purpose |
| --- | --- |
| `pai_list_tools` | List currently visible shell-bound tools |
| `pai_describe_tool <tool-or-command>` | Show tool metadata and argument schema |
| `pai_call_tool <tool-or-command> --json '{...}'` | Call a bound tool explicitly with JSON |
| `pai_search_tools <keywords>` | Discover deferred or hidden tools by keyword |

## Argument binding

Wrapped tools are still validated by Pydantic AI. The shell adapter accepts a few convenient input forms:

| Form | Example |
| --- | --- |
| JSON object | `my_tool --json '{"a": 1, "b": 2}'` |
| Named flags | `my_tool --a 1 --b 2` |
| Single positional value for single-argument tools | `get_weather Paris` |
| JSON via stdin | `echo '{"a": 1, "b": 2}' | my_tool --stdin-json` |

Use `--help` on a bound command, or `pai_describe_tool`, to inspect the generated signature and JSON schema.

## Behavior notes

- The shell session is persistent for a run, so virtual filesystem changes carry across `just_bash` calls.
- The set of bound commands is captured on first shell use in a run. If wrapped tools change dynamically and you want a fresh command set, call `just_bash(..., reset_session=True)`.
- Deferred tools are hidden in the shell until discovered with `pai_search_tools`.
- Direct shell commands return the tool result. If a wrapped tool returns `ToolReturn`, the shell uses its `return_value`.

## Development

```bash
make install
make all-ci
uv build
```

Common commands:

| Task | Command |
| --- | --- |
| Install dev + lint tooling | `make install` |
| Format code and config | `make format` |
| Check formatting only | `make format-check` |
| Lint | `make lint` |
| Type-check | `make typecheck` |
| Test | `make test` |
| Run the full local CI suite | `make all-ci` |

## Project status

This package is still early-stage and the public API may evolve as the shell command model settles. See [ROADMAP.md](ROADMAP.md).

## Related projects

- [`pydantic-ai`](https://github.com/pydantic/pydantic-ai)
- [`just-py-bash`](https://github.com/nathan-gage/just-py-bash)
