from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from just_bash import (
    DefenseInDepthConfig,
    ExecutionLimits,
    FetchRequest,
    FetchResult,
    FileInit,
    InMemoryFs,
    JavaScriptConfig,
    LazyFile,
    MountableFs,
    MountConfig,
    NetworkConfig,
    OverlayFs,
    ProcessInfo,
    ReadWriteFs,
    TraceEvent,
)
from just_bash._bridge import resolve_backend_artifacts
from pydantic_ai import FunctionToolset

from pydantic_ai_just_bash import JustBashToolset
from tests._helpers import build_shell_harness

pytestmark = pytest.mark.anyio


def make_execution_limits(
    *,
    max_output_size: int | None = None,
    max_python_timeout_ms: int | None = None,
    max_js_timeout_ms: int | None = None,
) -> ExecutionLimits:
    return ExecutionLimits(
        max_call_depth=None,
        max_command_count=None,
        max_loop_iterations=None,
        max_awk_iterations=None,
        max_sed_iterations=None,
        max_jq_iterations=None,
        max_sqlite_timeout_ms=None,
        max_python_timeout_ms=max_python_timeout_ms,
        max_js_timeout_ms=max_js_timeout_ms,
        max_glob_operations=None,
        max_string_length=None,
        max_array_elements=None,
        max_heredoc_size=None,
        max_substitution_depth=None,
        max_brace_expansion_results=None,
        max_output_size=max_output_size,
        max_file_descriptors=None,
        max_source_depth=None,
    )


@dataclass(slots=True)
class RecordingLogger:
    infos: list[tuple[str, dict[str, object] | None]] = field(default_factory=list)
    debugs: list[tuple[str, dict[str, object] | None]] = field(default_factory=list)

    def info(self, message: str, data: Mapping[str, object] | None = None) -> object:
        self.infos.append((message, None if data is None else dict(data)))
        return None

    def debug(self, message: str, data: Mapping[str, object] | None = None) -> object:
        self.debugs.append((message, None if data is None else dict(data)))
        return None


@dataclass(slots=True)
class RecordingCoverage:
    hits: list[str] = field(default_factory=list)

    def hit(self, feature: str) -> None:
        self.hits.append(feature)


async def test_bash_wrapper_supports_session_options_and_per_exec_overrides() -> None:
    async with JustBashToolset(
        FunctionToolset[None](),
        files={'/workspace/seed.txt': 'seeded\n'},
        env={'MODE': 'default', 'KEEP': 'yes'},
        cwd='/workspace',
        python=True,
        javascript=True,
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        default_result = await shell.run('pwd && printf "%s|%s|" "$MODE" "$KEEP" && cat seed.txt')
        stdin_result = await shell.run('cat', stdin='from stdin')
        env_override_result = await shell.run(
            'printf "%s|%s" "$MODE" "${KEEP:-missing}"',
            env={'MODE': 'override'},
            replace_env=True,
        )
        cwd_override_result = await shell.run('pwd', cwd='/')
        args_result = await shell.run('python -c "import sys; print(sys.argv[1:])"', args=['one', 'two'])
        timeout_result = await shell.run('js-exec -c "while(true){}"', timeout=0.01)

    assert default_result.stdout == '/workspace\ndefault|yes|seeded\n'
    assert stdin_result.stdout == 'from stdin'
    assert env_override_result.stdout == 'override|missing'
    assert cwd_override_result.stdout == '/\n'
    assert args_result.stdout == "['one', 'two']\n"
    assert timeout_result.exit_code == 1
    assert 'interrupted' in timeout_result.stderr


async def test_bash_wrapper_supports_python_javascript_and_command_allowlists() -> None:
    async with JustBashToolset(
        FunctionToolset[None](),
        python=True,
        javascript=JavaScriptConfig(bootstrap='globalThis.answer = 42;'),
        commands=['echo'],
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        echo_result = await shell.run('echo allowed')
        cat_result = await shell.run('cat missing.txt')
        python_result = await shell.run('python -c "print(2 + 3)"')
        javascript_result = await shell.run('js-exec -c "console.log(globalThis.answer)"')

    assert echo_result.stdout == 'allowed\n'
    assert cat_result.exit_code == 127
    assert cat_result.stderr == 'bash: cat: command not found\n'
    assert python_result.stdout == '5\n'
    assert javascript_result.stdout == '42\n'


async def test_bash_wrapper_supports_execution_limits() -> None:
    async with JustBashToolset(
        FunctionToolset[None](),
        execution_limits=make_execution_limits(max_output_size=5),
    ) as wrapped:
        shell = await build_shell_harness(wrapped)
        output_limit_result = await shell.run('printf 123456')

    async with JustBashToolset(
        FunctionToolset[None](),
        execution_limits=make_execution_limits(max_python_timeout_ms=1),
        python=True,
    ) as wrapped:
        shell = await build_shell_harness(wrapped)
        python_limit_result = await shell.run('python -c "import time; time.sleep(0.05)"')

    async with JustBashToolset(
        FunctionToolset[None](),
        execution_limits=make_execution_limits(max_js_timeout_ms=1),
        javascript=True,
    ) as wrapped:
        shell = await build_shell_harness(wrapped)
        javascript_limit_result = await shell.run('js-exec -c "while(true){}"')

    assert output_limit_result.exit_code == 126
    assert 'maxOutputSize' in output_limit_result.stderr
    assert python_limit_result.exit_code == 124
    assert 'exceeded 1ms limit' in python_limit_result.stderr
    assert javascript_limit_result.exit_code == 124
    assert 'exceeded 1ms limit' in javascript_limit_result.stderr


async def test_bash_wrapper_supports_fetch_logger_and_coverage_hooks() -> None:
    logger = RecordingLogger()
    coverage = RecordingCoverage()
    fetch_requests: list[FetchRequest] = []

    async def fetch(request: FetchRequest) -> FetchResult:
        fetch_requests.append(request)
        return FetchResult(status=200, body='hooked fetch', url=request.url)

    async with JustBashToolset(
        FunctionToolset[None](),
        javascript=True,
        fetch=fetch,
        logger=logger,
        coverage=coverage,
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        echo_result = await shell.run('echo hello')
        fetch_result = await shell.run(
            'js-exec -c "fetch(\'https://example.test/data\').then(r=>r.text()).then(t=>console.log(t))"'
        )

    assert echo_result.stdout == 'hello\n'
    assert fetch_result.stdout == 'hooked fetch\n'
    assert [request.url for request in fetch_requests] == ['https://example.test/data']
    assert any(message == 'exec' and data == {'command': 'echo hello'} for message, data in logger.infos)
    assert any(message == 'exit' and data == {'exitCode': 0} for message, data in logger.infos)
    assert any(message == 'stdout' and data == {'output': 'hello\n'} for message, data in logger.debugs)
    assert 'bash:builtin:echo' in coverage.hits


async def test_bash_wrapper_passes_through_backend_overrides_and_opaque_runtime_options() -> None:
    artifacts = resolve_backend_artifacts()
    node_path = shutil.which('node')
    if node_path is None:  # pragma: no cover
        pytest.skip('Node.js is required for just-bash runtime tests.')

    trace_events: list[TraceEvent] = []

    async def trace(event: TraceEvent) -> None:
        trace_events.append(event)

    defense_in_depth = DefenseInDepthConfig(
        enabled=True,
        audit_mode=True,
        exclude_violation_types=['process_stdout'],
    )
    network: NetworkConfig = {'allowedUrlPrefixes': ['https://example.test']}
    process_info: ProcessInfo = {'pid': 321, 'ppid': 99}

    async with JustBashToolset(
        FunctionToolset[None](),
        trace=trace,
        defense_in_depth=defense_in_depth,
        network=network,
        process_info=process_info,
        node_command=(node_path,),
        js_entry=artifacts.js_entry,
        package_json=artifacts.package_json,
    ) as wrapped:
        shell = await build_shell_harness(wrapped)
        process_result = await shell.run('echo $$:$PPID')

        bash = wrapped._bash
        assert bash is not None

    assert process_result.stdout == '321:99\n'
    assert bash._options.trace is trace
    assert bash._options.defense_in_depth == defense_in_depth
    assert bash._options.network == network
    assert bash._options.process_info == process_info
    assert bash._node_command == (node_path,)
    assert bash._js_entry == artifacts.js_entry
    assert bash._package_json == artifacts.package_json
    assert trace_events == []


async def test_bash_wrapper_supports_fileinit_and_lazy_files() -> None:
    provider_calls: list[str] = []

    def generated_file() -> str:
        provider_calls.append('generated')
        return 'generated once\n'

    async with JustBashToolset(
        FunctionToolset[None](),
        files={
            '/workspace/plain.txt': 'plain\n',
            '/workspace/init.txt': FileInit(content='initialized\n', mode=0o640),
            '/workspace/lazy.txt': LazyFile(provider='lazy\n'),
            '/workspace/generated.txt': generated_file,
        },
        python=True,
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        plain_result = await shell.run('cat /workspace/plain.txt')
        init_result = await shell.run('cat /workspace/init.txt')
        lazy_result = await shell.run('cat /workspace/lazy.txt')
        generated_once_result = await shell.run('cat /workspace/generated.txt')
        generated_twice_result = await shell.run('cat /workspace/generated.txt')
        mode_result = await shell.run(
            'python -c "import os; print(oct(os.stat(\'/workspace/init.txt\').st_mode & 0o777))"'
        )

    assert plain_result.stdout == 'plain\n'
    assert init_result.stdout == 'initialized\n'
    assert lazy_result.stdout == 'lazy\n'
    assert generated_once_result.stdout == 'generated once\n'
    assert generated_twice_result.stdout == 'generated once\n'
    assert mode_result.stdout == '0o640\n'
    assert provider_calls == ['generated']


async def test_bash_wrapper_supports_inmemory_and_overlay_filesystems(tmp_path: Path) -> None:
    async with JustBashToolset(
        FunctionToolset[None](),
        fs=InMemoryFs(files={'/workspace/from_fs.txt': 'from fs\n'}),
        cwd='/workspace',
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        in_memory_read_result = await shell.run('cat from_fs.txt')
        await shell.run("printf 'persisted in memory' > note.txt")
        in_memory_persisted_result = await shell.run('cat note.txt')

    overlay_root = tmp_path / 'overlay-root'
    overlay_root.mkdir()
    (overlay_root / 'seed.txt').write_text('seed', encoding='utf-8')

    async with JustBashToolset(
        FunctionToolset[None](),
        fs=OverlayFs(root=str(overlay_root), mount_point='/workspace'),
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        overlay_host_result = await shell.run('cat /workspace/seed.txt')
        await shell.run("printf 'overlay only' > /workspace/virtual.txt")
        overlay_persisted_result = await shell.run('cat /workspace/virtual.txt')

    assert in_memory_read_result.stdout == 'from fs\n'
    assert in_memory_persisted_result.stdout == 'persisted in memory'
    assert overlay_host_result.stdout == 'seed'
    assert overlay_persisted_result.stdout == 'overlay only'
    assert not (overlay_root / 'virtual.txt').exists()


async def test_bash_wrapper_supports_readwrite_and_mountable_workspace_filesystems(tmp_path: Path) -> None:
    readwrite_root = tmp_path / 'readwrite-root'
    readwrite_root.mkdir()

    async with JustBashToolset(
        FunctionToolset[None](),
        fs=ReadWriteFs(root=str(readwrite_root)),
        cwd='/',
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        await shell.run("printf 'host persisted' > /written.txt")
        readwrite_result = await shell.run('cat /written.txt')

    workspace_root = tmp_path / 'workspace-root'
    workspace_root.mkdir()
    (workspace_root / 'host.txt').write_text('host file\n', encoding='utf-8')

    async with JustBashToolset(
        FunctionToolset[None](),
        fs=MountableFs(
            base=InMemoryFs(files={'/scratch.txt': 'scratch\n'}),
            mounts=[
                MountConfig(
                    mount_point='/workspace',
                    filesystem=ReadWriteFs(root=str(workspace_root)),
                )
            ],
        ),
        cwd='/workspace',
    ) as wrapped:
        shell = await build_shell_harness(wrapped)

        mounted_host_result = await shell.run('cat host.txt')
        await shell.run("printf 'mounted persisted' > mounted.txt")
        scratch_result = await shell.run('cat /scratch.txt')
        mounted_persisted_result = await shell.run('cat mounted.txt')

    assert readwrite_result.stdout == 'host persisted'
    assert (readwrite_root / 'written.txt').read_text(encoding='utf-8') == 'host persisted'
    assert mounted_host_result.stdout == 'host file\n'
    assert scratch_result.stdout == 'scratch\n'
    assert mounted_persisted_result.stdout == 'mounted persisted'
    assert (workspace_root / 'mounted.txt').read_text(encoding='utf-8') == 'mounted persisted'
