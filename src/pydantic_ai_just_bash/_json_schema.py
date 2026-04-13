from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

JsonObject = dict[str, Any]
JsonSchemaProperties = dict[str, JsonObject]


def copy_json_object(value: Mapping[str, Any]) -> JsonObject:
    copied = deepcopy(dict(value))
    return json_object(copied, label='JSON object')


def json_object(value: Any, *, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f'{label} must be a JSON object.')

    result: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f'{label} keys must be strings.')
        result[key] = item
    return result


def schema_properties(schema: Mapping[str, Any]) -> JsonSchemaProperties:
    raw_properties = schema.get('properties')
    if not isinstance(raw_properties, Mapping):
        return {}

    properties: JsonSchemaProperties = {}
    for name, raw_schema in raw_properties.items():
        if not isinstance(name, str) or not isinstance(raw_schema, Mapping):
            continue
        properties[name] = copy_json_object(raw_schema)
    return properties


def required_names(schema: Mapping[str, Any]) -> list[str]:
    raw_required = schema.get('required')
    if not isinstance(raw_required, list):
        return []
    return [item for item in raw_required if isinstance(item, str)]


def schema_variants(schema: Mapping[str, Any], key: str) -> list[JsonObject]:
    raw_variants = schema.get(key)
    if not isinstance(raw_variants, list):
        return []

    variants: list[JsonObject] = []
    for item in raw_variants:
        if isinstance(item, Mapping):
            variants.append(copy_json_object(item))
    return variants
