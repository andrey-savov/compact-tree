# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- LOUDS-based succinct trie implementation
- DAWG-style key/value deduplication
- Binary serialization format (v2)
- Support for local and remote storage via fsspec
- Comprehensive test suite
- Documentation and examples

## [0.1.0] - 2026-02-07

### Added
- Initial implementation of CompactTree
- `from_dict()` class method for building from nested Python dicts
- `serialize()` and deserialization via `__init__(url)`
- `to_dict()` method to convert back to plain Python dict
- Dict-like interface (`__getitem__`, `__contains__`)
- Support for nested dictionary traversal
- Binary format with magic header and version
- LOUDS navigation with Poppy rank/select
- Key and value deduplication for space efficiency

### Dependencies
- bitarray >= 2.0.0
- succinct >= 0.1.0
- fsspec >= 2021.0.0

[Unreleased]: https://github.com/andrey-savov/compact-tree/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/andrey-savov/compact-tree/releases/tag/v0.1.0
