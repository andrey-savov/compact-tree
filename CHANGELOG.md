# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Python 3.13 support and testing

## [0.0.2] - 2026-02-08

### Changed
- Improved fsspec integration using `url_to_fs()` for proper URL parsing
- Eliminated memoryview in favor of bytes for better cloud storage compatibility
- Sequential file reading during deserialization with immediate file closure
- Enhanced compatibility with buffered cloud storage backends

## [0.0.1] - 2026-02-07

### Added
- Initial implementation of CompactTree
- `from_dict()` class method for building from nested Python dicts
- `serialize()` and deserialization via `__init__(url)`
- `to_dict()` method to convert back to plain Python dict
- Dict-like interface (`__getitem__`, `__contains__`)
- Support for nested dictionary traversal
- Binary format with magic header and version
- LOUDS navigation with Poppy rank/select
- LOUDS-based succinct trie implementation
- DAWG-style key/value deduplication
- Key and value deduplication for space efficiency
- Support for local and remote storage via fsspec
- Comprehensive test suite
- Documentation and examples

### Dependencies
- bitarray >= 2.0.0
- succinct >= 0.1.0
- fsspec >= 2021.0.0

[Unreleased]: https://github.com/andrey-savov/compact-tree/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/andrey-savov/compact-tree/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/andrey-savov/compact-tree/releases/tag/0.0.1
