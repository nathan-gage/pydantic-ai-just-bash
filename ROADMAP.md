# Roadmap

Ordered roughly by priority.

## Public API and naming

- [x] Ship a first-cut capability and toolset wrapper
- [x] Make `bash` the default top-level tool name
- [x] Rename shell helpers from `pai_*` to `bash_*`
- [x] Expose `bash_list_tools` as a top-level Pydantic AI tool
- [x] Expose `bash_search_tools` as a top-level Pydantic AI tool
- [x] Expose `bash_describe_tool` as a top-level Pydantic AI tool
- [x] Decide and document whether wrapped tools remain directly visible by default, or whether a shell-only mode should also be supported

## Bash-like command behavior

- [ ] Make wrapped tools feel like plausible shell commands rather than framework helpers
- [x] Support `some_tool --help` and `some_tool -h` in a familiar way
- [x] Generate help text from `ToolDefinition` in a CLI-style format
- [x] Include usage, argument help, rendered signature, and JSON fallback forms in generated help
- [x] Define the behavior for tools that have an actual argument named `help`
- [x] Make that `help` collision behavior explicit, documented, and tested
- [x] Keep argument binding rules bash-like, with JSON as the escape hatch for complex values
- [x] Improve command-line error messages so they read like CLI errors, not framework internals

## Tool discovery and deferred loading

- [x] Support shell-side discovery of deferred tools in the first cut
- [ ] Finalize the public shape of `bash_search_tools`
- [x] Make list/search/describe behavior consistent between top-level tools and shell commands
- [ ] Keep the progressive-disclosure behavior aligned with Pydantic AI's deferred loading model
- [ ] Make deferred-tool behavior clearly documented from both the agent side and the shell side

## just-py-bash runtime support

- [x] Explicitly support the core `just-py-bash` session options on the wrapper API
- [x] Support and test `files`
- [x] Support and test `env`
- [x] Support and test `cwd`
- [x] Support and test `fs`
- [x] Support and test `execution_limits`
- [x] Support and test `python`
- [x] Support and test `javascript`
- [x] Support and test `commands`
- [x] Support and test `fetch`
- [x] Support and test `logger`
- [x] Support and test `trace`
- [x] Support and test `defense_in_depth`
- [x] Support and test `coverage`
- [x] Support and test `network`
- [x] Support and test `process_info`
- [x] Support and test `node_command`
- [x] Support and test `js_entry`
- [x] Support and test `package_json`
- [x] Make the `bash(...)` tool args map cleanly and intentionally to per-exec `just-py-bash` options

## Filesystem support

- [x] Make filesystem support a first-class part of the package contract
- [x] Explicitly support and test `FileInit`
- [x] Explicitly support and test `LazyFile`
- [x] Explicitly support and test `InMemoryFs`
- [x] Explicitly support and test `OverlayFs`
- [x] Explicitly support and test `ReadWriteFs`
- [x] Explicitly support and test `MountableFs`
- [x] Explicitly support and test `MountConfig`
- [x] Add dedicated coverage for lazy file providers
- [x] Add dedicated coverage for mounted workspace-style setups
- [x] Keep persistent virtual filesystem behavior stable across multiple `bash` calls

## Dynamic toolsets

- [x] Support dynamic toolsets without requiring shell session reset
- [x] Refresh shell-visible commands as wrapped tool availability changes across run steps
- [x] Preserve the `just-py-bash` session and virtual filesystem during command refresh
- [x] Ensure deferred/discovered tools continue to work correctly with refreshed command sets
- [x] Add tests for context-dependent toolsets
- [x] Add tests for tool availability changing over time without losing shell state

## Agent specs and YAML

- [x] Make the capability usable from `Agent.from_spec(...)`
- [x] Add proper serialization support for the capability
- [x] Define which configuration fields are spec-safe
- [x] Document any Python-only configuration surface
- [x] Add tests for `Agent.from_spec(...)`
- [x] Add tests for file/YAML-based agent configuration

## Documentation and examples

- [x] Update the README to reflect the final public API
- [x] Document the command model clearly
- [x] Document the generated `--help` behavior clearly
- [ ] Document deferred discovery and dynamic toolset behavior clearly
- [ ] Add examples for workflows people are likely to copy
- [ ] Add examples for persistent shell + wrapped tools
- [ ] Add examples for deferred tool discovery
- [ ] Add examples for filesystem and lazy-file bootstrapping
- [ ] Add examples for mounted workspaces
- [ ] Add examples for spec/YAML configuration

## Packaging and release readiness

- [ ] Review dependency strategy for published installs
- [ ] Review package metadata and classifiers
- [ ] Make sure local development and published usage are both straightforward
- [ ] Tighten release/readme polish once the public API settles
