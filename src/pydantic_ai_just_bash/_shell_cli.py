from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.tools import ToolDefinition

from ._json_schema import JsonObject, schema_properties, schema_variants


class ShellCliError(ValueError):
    def __init__(self, message: str, *, usage: str | None = None, hint: str | None = None):
        self.usage = usage
        self.hint = hint
        super().__init__(message)


@dataclass(frozen=True)
class ShellCommandSurface:
    command_name: str
    help_flag_name: str
    tool_def: ToolDefinition
    actual_to_shell_arg_names: Mapping[str, str]
    signature: str | None = None


def bind_command_args(surface: ShellCommandSurface, args: Sequence[str], stdin: str) -> JsonObject:
    return _ShellArgumentParser(surface, list(args), stdin).bind()


def render_help(surface: ShellCommandSurface) -> str:
    tool_def = surface.tool_def
    properties = schema_properties(tool_def.parameters_json_schema)

    lines = ['NAME', f'  {surface.command_name}']
    if tool_def.description:
        lines.append(f'  {tool_def.description}')

    lines.extend(['', 'USAGE'])
    usage_lines = [
        f'  {surface.command_name} [--{surface.help_flag_name}] [--json "{{...}}"]',
        f'  {surface.command_name} [--flag value ...]',
    ]
    if properties:
        usage_lines.append(f'  printf "{{...}}" | {surface.command_name} --stdin-json')
    if len(properties) == 1:
        usage_lines.append(f'  {surface.command_name} <value>')
        usage_lines.append(f'  {surface.command_name} -- <value-starting-with-dash>')
    lines.extend(usage_lines)

    lines.extend(['', 'OPTIONS'])
    lines.append(f'  -h, --{surface.help_flag_name}\n      Show this help message and exit.')
    lines.append('  --json JSON\n      Provide the full argument object as JSON. Prefer this for complex values.')
    lines.append('  --stdin-json\n      Read the full argument object as JSON from stdin.')
    lines.append('  --\n      Stop option parsing and treat the remaining tokens as positional values.')
    for name, schema in properties.items():
        flag_lines = [f'  --{name}']
        if _is_boolean_schema(schema):
            flag_lines[0] += f', --no-{name}'
        flag_lines[0] += f'\n      {_schema_type_label(schema)}'
        description = schema.get('description')
        if isinstance(description, str) and description:
            flag_lines.append(f'      {description}')
        lines.append('\n'.join(flag_lines))

    renamed_arguments = [
        (actual_name, shell_name)
        for actual_name, shell_name in surface.actual_to_shell_arg_names.items()
        if actual_name != shell_name
    ]
    if renamed_arguments:
        lines.extend(['', 'NOTES'])
        for actual_name, shell_name in renamed_arguments:
            if actual_name == surface.help_flag_name:
                lines.append(
                    f'  The original tool argument {actual_name!r} is exposed as --{shell_name} because '
                    f'--{surface.help_flag_name} is reserved for generated command help.'
                )
            else:
                lines.append(f'  The original tool argument {actual_name!r} is exposed in the shell as --{shell_name}.')

    if surface.signature:
        lines.extend(['', 'SIGNATURE', surface.signature])

    lines.extend(
        [
            '',
            'JSON SCHEMA',
            json.dumps(tool_def.parameters_json_schema, indent=2, sort_keys=True),
            '',
        ]
    )
    return '\n'.join(lines)


def is_help_request(args: Sequence[str], help_flag_name: str) -> bool:
    long_flag = f'--{help_flag_name}'
    for arg in args:
        if arg == '--':
            return False
        if arg in {'-h', long_flag}:
            return True
    return False


def command_help_hint(command_name: str, help_flag_name: str) -> str:
    return f"Try '{command_name} --{help_flag_name}' for usage."


class _ShellArgumentParser:
    def __init__(self, surface: ShellCommandSurface, args: list[str], stdin: str):
        self.surface = surface
        self.args = args
        self.stdin = stdin
        self.properties = schema_properties(surface.tool_def.parameters_json_schema)
        self.parsed: JsonObject = {}
        self.positionals: list[str] = []
        self.json_payload: str | None = None
        self.stdin_json = False

    def bind(self) -> JsonObject:
        if not self.properties:
            if self.args:
                raise ShellCliError('this command does not take any arguments.')
            return {}

        self._consume_tokens()

        if self.json_payload is not None:
            return self._bind_from_json_payload()
        if self.stdin_json:
            return self._bind_from_stdin_json()
        if self.parsed:
            return self._bind_from_flags()
        if self.positionals:
            return self._bind_from_positionals()
        if self.stdin:
            return self._bind_from_stdin()
        return {}

    def _consume_tokens(self) -> None:
        i = 0
        while i < len(self.args):
            token = self.args[i]
            if token == '--':
                self.positionals.extend(self.args[i + 1 :])
                return
            if token == '--stdin-json':
                self.stdin_json = True
                i += 1
                continue
            if token == '--json':
                if i + 1 >= len(self.args):
                    raise ShellCliError('option --json requires a JSON object.')
                self.json_payload = self.args[i + 1]
                i += 2
                continue
            if token.startswith('--json='):
                self.json_payload = token.split('=', 1)[1]
                i += 1
                continue
            if token.startswith('--no-'):
                prop_name = self._resolve_flag_name(token[5:])
                self.parsed[prop_name] = False
                i += 1
                continue
            if token.startswith('--'):
                i = self._consume_long_flag(token, i)
                continue

            self.positionals.append(token)
            i += 1

    def _consume_long_flag(self, token: str, index: int) -> int:
        name, inline_value = self._split_flag(token)
        prop_name = self._resolve_flag_name(name)
        prop_schema = self.properties[prop_name]
        if inline_value is None:
            if _is_boolean_schema(prop_schema):
                next_token = self.args[index + 1] if index + 1 < len(self.args) else None
                if next_token is None or next_token.startswith('--'):
                    self.parsed[prop_name] = True
                    return index + 1
            if index + 1 >= len(self.args):
                raise ShellCliError(f'option --{name} requires a value.')
            inline_value = self.args[index + 1]
            index += 2
        else:
            index += 1
        self.parsed[prop_name] = self._coerce_flag_value(prop_schema, inline_value, flag_name=prop_name)
        return index

    def _bind_from_json_payload(self) -> JsonObject:
        if self.parsed or self.positionals or self.stdin_json:
            raise ShellCliError('do not mix --json with flags, positional values, or --stdin-json.')
        parsed_json = self._parse_json_object(self.json_payload or '', source='--json')
        self._validate_shell_json_keys(parsed_json)
        return parsed_json

    def _bind_from_stdin_json(self) -> JsonObject:
        if self.parsed or self.positionals:
            raise ShellCliError('do not mix --stdin-json with flags or positional values.')
        parsed_json = self._parse_json_object(self.stdin, source='stdin')
        self._validate_shell_json_keys(parsed_json)
        return parsed_json

    def _bind_from_flags(self) -> JsonObject:
        if self.positionals:
            raise ShellCliError('do not mix named flags with positional values.')
        return dict(self.parsed)

    def _bind_from_positionals(self) -> JsonObject:
        if len(self.properties) == 1:
            prop_name, prop_schema = next(iter(self.properties.items()))
            return {prop_name: self._coerce_single_argument(prop_schema, self.positionals)}
        if len(self.positionals) == 1 and self.positionals[0].lstrip().startswith('{'):
            parsed_json = self._parse_json_object(self.positionals[0], source='positional JSON input')
            self._validate_shell_json_keys(parsed_json)
            return parsed_json
        raise ShellCliError(
            'this command has multiple parameters. Use named flags like --param value or pass a JSON object with --json.'
        )

    def _bind_from_stdin(self) -> JsonObject:
        if len(self.properties) == 1:
            prop_name, prop_schema = next(iter(self.properties.items()))
            return {prop_name: self._coerce_stdin_argument(prop_schema, self.stdin)}
        parsed_json = self._parse_json_object(self.stdin, source='stdin')
        self._validate_shell_json_keys(parsed_json)
        return parsed_json

    def _resolve_flag_name(self, flag_name: str) -> str:
        candidates = [flag_name, flag_name.replace('-', '_'), flag_name.replace('_', '-')]
        for candidate in candidates:
            if candidate in self.properties:
                return candidate
        raise ShellCliError(f'unknown option --{flag_name}.')

    def _validate_shell_json_keys(self, parsed_json: Mapping[str, Any]) -> None:
        renamed_actual_names = {
            actual_name
            for actual_name, shell_name in self.surface.actual_to_shell_arg_names.items()
            if actual_name != shell_name
        }
        invalid_names = sorted(name for name in parsed_json if name in renamed_actual_names)
        if invalid_names:
            invalid_name = invalid_names[0]
            shell_name = self.surface.actual_to_shell_arg_names[invalid_name]
            raise ShellCliError(
                f'argument {invalid_name!r} is exposed in the shell as --{shell_name}. Use --{shell_name} instead.'
            )

    def _coerce_flag_value(self, prop_schema: Mapping[str, Any], raw_value: str, *, flag_name: str) -> Any:
        if _expects_json_payload(prop_schema):
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError as exc:
                raise ShellCliError(
                    f'option --{flag_name} expects valid JSON: {exc.msg}.',
                    hint='Use --json for complex values when shell quoting gets awkward.',
                ) from exc
        return raw_value

    def _coerce_single_argument(self, prop_schema: Mapping[str, Any], values: list[str]) -> Any:
        if _is_array_schema(prop_schema):
            return values
        if _expects_json_payload(prop_schema):
            if len(values) != 1:
                raise ShellCliError('complex values must be passed as JSON. Use --json or --stdin-json.')
            try:
                return json.loads(values[0])
            except json.JSONDecodeError as exc:
                raise ShellCliError(
                    f'complex positional input must be valid JSON: {exc.msg}.',
                    hint='Use --json for object or array values.',
                ) from exc
        if _is_string_schema(prop_schema):
            return ' '.join(values)
        if len(values) == 1:
            return values[0]
        return values

    def _coerce_stdin_argument(self, prop_schema: Mapping[str, Any], stdin: str) -> Any:
        if _expects_json_payload(prop_schema):
            try:
                return json.loads(stdin)
            except json.JSONDecodeError as exc:
                raise ShellCliError(
                    f'stdin must contain valid JSON for this command: {exc.msg}.',
                    hint='Use --stdin-json for object input or --json for inline JSON.',
                ) from exc
        if _is_array_schema(prop_schema):
            return [line for line in stdin.splitlines() if line]
        return stdin

    def _parse_json_object(self, payload: str, *, source: str) -> JsonObject:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ShellCliError(f'{source} must be valid JSON: {exc.msg}.') from exc

        if not isinstance(parsed, dict):
            raise ShellCliError(f'{source} must be a JSON object.')

        result: JsonObject = {}
        for key, value in parsed.items():
            if not isinstance(key, str):
                raise ShellCliError(f'{source} must use string keys.')
            result[key] = value
        return result

    @staticmethod
    def _split_flag(token: str) -> tuple[str, str | None]:
        name = token[2:]
        if '=' in name:
            flag_name, value = name.split('=', 1)
            return flag_name, value
        return name, None


def _expects_json_payload(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get('type')
    if schema_type in {'object', 'array'}:
        return True
    for key in ('anyOf', 'oneOf', 'allOf'):
        if any(_expects_json_payload(item) for item in schema_variants(schema, key)):
            return True
    return False


def _is_array_schema(schema: Mapping[str, Any]) -> bool:
    return schema.get('type') == 'array'


def _is_string_schema(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get('type')
    if schema_type == 'string':
        return True
    return schema_type is None and 'enum' in schema


def _is_boolean_schema(schema: Mapping[str, Any]) -> bool:
    schema_type = schema.get('type')
    if schema_type == 'boolean':
        return True
    for key in ('anyOf', 'oneOf', 'allOf'):
        if any(_is_boolean_schema(item) for item in schema_variants(schema, key)):
            return True
    return False


def _schema_type_label(schema: Mapping[str, Any]) -> str:
    schema_type = schema.get('type')
    if isinstance(schema_type, str):
        return f'Type: {schema_type}.'
    if any(key in schema for key in ('anyOf', 'oneOf', 'allOf')):
        return 'Type: complex.'
    return 'Type: value.'
