from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from just_bash import ExecutionLimits, FileInit, InMemoryFs, JavaScriptConfig, LazyFile
from pydantic_ai import Agent
from pydantic_ai._spec import CapabilitySpec
from pydantic_ai.capabilities._ordering import collect_leaves

from pydantic_ai_just_bash import JustBash

EXECUTION_LIMITS = ExecutionLimits(
    max_call_depth=None,
    max_command_count=None,
    max_loop_iterations=None,
    max_awk_iterations=None,
    max_sed_iterations=None,
    max_jq_iterations=None,
    max_sqlite_timeout_ms=None,
    max_python_timeout_ms=None,
    max_js_timeout_ms=None,
    max_glob_operations=None,
    max_string_length=None,
    max_array_elements=None,
    max_heredoc_size=None,
    max_substitution_depth=None,
    max_brace_expansion_results=None,
    max_output_size=4096,
    max_file_descriptors=None,
    max_source_depth=None,
)

ALL_SPEC_ARGS: dict[str, Any] = {
    'tool_name': 'shellbox',
    'command_prefix': 'cmd_',
    'helper_prefix': 'bash_',
    'exposed_tools': ['visible_tool', 'other_tool'],
    'expose_wrapped_tools': False,
    'instructions': 'Use the shell carefully.',
    'help_flag_name': 'usage',
    'rename_help_argument': '{tool_name}_{arg_name}',
    'files': {
        '/workspace/plain.txt': 'plain text\n',
        '/workspace/init.txt': FileInit(content='seeded\n', mode=0o640),
        '/workspace/lazy.txt': LazyFile(provider='lazy text\n'),
    },
    'env': {'MODE': 'test', 'DEBUG': '1'},
    'cwd': '/workspace',
    'fs': InMemoryFs(files={'/workspace/from_fs.txt': 'from fs\n'}),
    'execution_limits': EXECUTION_LIMITS,
    'python': True,
    'javascript': JavaScriptConfig(bootstrap='globalThis.answer = 42;'),
    'commands': ['echo', 'cat'],
    'network': {
        'allowedUrlPrefixes': ['https://example.com'],
        'allowedMethods': ['GET'],
        'timeoutMs': 5000,
    },
    'process_info': {'pid': 123, 'uid': 501},
    'node_command': ['node'],
    'js_entry': 'dist/index.js',
    'package_json': 'package.json',
}


def _extract_just_bash(agent: Agent[Any, Any]) -> JustBash[Any]:
    leaves = collect_leaves(agent.root_capability)
    for leaf in leaves:
        if isinstance(leaf, JustBash):
            return leaf
    raise AssertionError('JustBash capability not found in agent root capability tree')


def test_just_bash_has_a_serialization_name() -> None:
    assert JustBash.get_serialization_name() == 'JustBash'


def test_all_just_bash_fields_are_covered_by_spec_roundtrip_samples() -> None:
    assert set(ALL_SPEC_ARGS) == {field.name for field in fields(JustBash)}


def test_from_spec_signature_covers_all_just_bash_fields() -> None:
    params = {
        name
        for name, param in inspect.signature(JustBash.from_spec).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert params == {field.name for field in fields(JustBash)}


def test_agent_from_spec_supports_all_just_bash_arguments() -> None:
    serialized_capability = CapabilitySpec(
        name='JustBash',
        arguments=ALL_SPEC_ARGS,
    ).model_dump(mode='python', context={'use_short_form': True})

    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [serialized_capability],
        },
        custom_capability_types=[JustBash],
    )

    capability = _extract_just_bash(agent)
    for field_name, expected in ALL_SPEC_ARGS.items():
        assert getattr(capability, field_name) == expected


def test_agent_from_spec_supports_default_just_bash() -> None:
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': ['JustBash'],
        },
        custom_capability_types=[JustBash],
    )

    capability = _extract_just_bash(agent)
    assert capability == JustBash()


def test_agent_from_file_yaml_supports_all_just_bash_arguments(tmp_path: Path) -> None:
    serialized_capability = CapabilitySpec(
        name='JustBash',
        arguments=ALL_SPEC_ARGS,
    ).model_dump(mode='python', context={'use_short_form': True})

    spec = {
        'model': 'test',
        'capabilities': [serialized_capability],
    }
    spec_path = tmp_path / 'agent.yaml'
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')

    agent = Agent.from_file(spec_path, custom_capability_types=[JustBash])

    capability = _extract_just_bash(agent)
    for field_name, expected in ALL_SPEC_ARGS.items():
        assert getattr(capability, field_name) == expected
