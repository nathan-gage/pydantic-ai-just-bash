# pydantic-ai-just-bash

[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-just-bash.svg)](https://pypi.org/project/pydantic-ai-just-bash/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`pydantic-ai-just-bash` is a small extension package for [Pydantic AI](https://ai.pydantic.dev/) that gives an agent a persistent [`just-py-bash`](https://github.com/nathan-gage/just-py-bash) shell.

The extension exposes a `bash` tool plus helper tools like `bash_list_tools`, `bash_search_tools`, and `bash_describe_tool`, and binds wrapped Pydantic AI tools into that shell as commands.

By default, wrapped tools stay directly visible to the model as normal Pydantic AI tools. If you want a shell-only interface, set `expose_wrapped_tools=False`.

It is designed to let an agent mix shell workflows with normal Pydantic AI tool calls while keeping a long-lived virtual filesystem for the lifetime of the agent run.

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
| Install spec/YAML support too | `uv add 'pydantic-ai-just-bash[spec]'` |
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

Inside `bash`, the model can do things like:

```bash
bash_list_tools
get_weather Paris
printf 'draft note' > note.txt
cat note.txt
bash_search_tools stock
stock_lookup AAPL
```

The shell session and virtual filesystem persist across `bash` calls for the lifetime of the agent run.

## API overview

### `JustBash`

Use `JustBash` as an agent capability. It wraps the assembled toolset and injects a `bash` tool plus bash helper tools.

```python
from pydantic_ai_just_bash import JustBash

capability = JustBash(
    tool_name='bash',
    command_prefix='',
    helper_prefix='bash_',
    python=True,
)
```

Common configuration knobs include:

- `tool_name`
- `command_prefix`
- `helper_prefix`
- `exposed_tools`
- `expose_wrapped_tools`
- `instructions`
- `help_flag_name`
- `rename_help_argument`
- `files`, `env`, `cwd`, `fs`, `execution_limits`
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

### Result models

The package exports structured result models for the public helper tools:

- `BashExecutionResult`
- `BashListToolsResult`
- `BashSearchToolsResult`
- `BashDescribeToolResult`
- `BashCommandInfo`

## Agent specs and YAML

`JustBash` can be loaded from `Agent.from_spec(...)` and `Agent.from_file(...)`.

```yaml
model: test
capabilities:
  - JustBash:
      tool_name: bash
      helper_prefix: bash_
      python: true
      files:
        /workspace/seed.txt: hello from spec
        /workspace/lazy.txt:
          provider: lazy content from spec
```

Use the `spec` extra if you want the package to carry the YAML/spec dependencies itself:

```bash
uv add 'pydantic-ai-just-bash[spec]'
```

### Spec-safe configuration surface

The current spec-safe surface includes the public `JustBash` fields, including:

- shell naming/config fields like `tool_name`, `command_prefix`, `helper_prefix`, `instructions`, `help_flag_name`, and `expose_wrapped_tools`
- runtime/session fields like `env`, `cwd`, `execution_limits`, `python`, `javascript`, `commands`, `network`, `process_info`, `node_command`, `js_entry`, and `package_json`
- filesystem configuration via `files` and `fs`
- spec-friendly file values such as plain text/bytes, `FileInit`, and `LazyFile` with a static `provider` value
- string-based `rename_help_argument` values

### Python-only configuration surface

Some configuration remains Python-only because it depends on runtime callables rather than JSON/YAML data.

The current Python-only surface includes:

- callable `exposed_tools` selectors
- callback-based lazy file providers, either passed directly or wrapped in `LazyFile(...)`
- callable `rename_help_argument` values

For example:

```python
from just_bash import LazyFile
from pydantic_ai_just_bash import JustBash

cap = JustBash(
    files={
        '/workspace/generated.txt': LazyFile(provider=lambda: 'generated at session start\n'),
    },
)
```

That callable form is supported when you configure `JustBash(...)` in Python, but it is not spec/YAML-serializable.

## Public helper tools

At the agent level, the wrapper exposes:

- `bash`
- `bash_list_tools`
- `bash_search_tools`
- `bash_describe_tool`

Inside the shell, the same helper concepts are available as commands:

- `bash_list_tools`
- `bash_describe_tool <tool-or-command>`
- `bash_call_tool <tool-or-command> --json '{...}'`
- `bash_search_tools <keywords>`

## Tool visibility model

Wrapped tools remain directly visible to the model by default, so an agent can either call them normally or use them through `bash`.

If you want the model to go through the shell interface only, set:

```python
JustBash(expose_wrapped_tools=False)
```

In shell-only mode, the model still sees `bash`, `bash_list_tools`, `bash_search_tools`, and `bash_describe_tool`, but the wrapped tools themselves are omitted from the public agent tool list.

## Argument binding

Wrapped tools are still validated by Pydantic AI, but the shell adapter tries to behave like a small CLI layer first:

| Form | Example |
| --- | --- |
| Named flags for simple values | `my_tool --a 1 --b 2` |
| Booleans with `--flag` / `--no-flag` | `my_tool --verbose --no-cache` |
| Single positional value for single-argument tools | `get_weather Paris` |
| `--` to stop option parsing | `show_value -- --help` |
| JSON object escape hatch | `my_tool --json '{"a": 1, "b": 2}'` |
| JSON via stdin | `echo '{"a": 1, "b": 2}' | my_tool --stdin-json` |

Use shell-style flags for simple scalar values. Prefer `--json` or `--stdin-json` for objects, arrays, or anything that becomes awkward to quote safely.

When a command has multiple parameters, bare positional input is rejected with a CLI-style error that points back to `--help` and `--json`.

Use `--help` or `-h` on a bound command, or `bash_describe_tool`, to inspect generated command help, rendered signatures, renamed arguments, and JSON schema.

## Help flag behavior

By default, shell-exposed commands reserve `--help` and `-h` for generated command help.

If a wrapped tool has a real argument named `help`, wrapper creation fails by default with a configuration error. This makes the collision explicit instead of silently changing command behavior.

You can override that behavior with:

- `help_flag_name='usage'` to reserve a different help flag instead of `--help`
- `rename_help_argument=...` to rename the wrapped tool's conflicting shell argument

For example:

```python
JustBash(rename_help_argument='{tool_name}_{arg_name}')
```

would expose a tool argument named `help` as `--<tool_name>_help` in the shell, and generated help text will explain that rename explicitly.

## Behavior notes

- The shell session is persistent for a run, so virtual filesystem changes carry across `bash` calls.
- Wrapped tools stay directly visible by default; set `expose_wrapped_tools=False` for shell-only mode.
- The set of bound commands is captured on first shell use in a run. If wrapped tools change dynamically and you want a fresh command set, call `bash(..., reset_session=True)`.
- Deferred tools are hidden in the shell until discovered with `bash_search_tools`.
- Shell command failures are formatted as CLI-style stderr messages instead of leaking raw framework internals where possible.
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
