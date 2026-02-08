# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-08

### Added
- Initial implementation of CompactTree
- `from_dict()` class method for building from nested Python dicts
- `serialize()` and deserialization via `__init__(url)`
- `to_dict()` method to convert back to plain Python dict
- Dict-like interface (`__getitem__`, `__contains__`, `__iter__`, `__len__`)
- Support for nested dictionary traversal via `_Node` interface
- Binary serialization format (v2) with magic header and version
- LOUDS-based succinct trie implementation with Poppy rank/select
- DAWG-style key/value deduplication for space efficiency
- Support for local and remote storage via fsspec
- Cloud storage compatibility with sequential file reading
- Python 3.9, 3.10, 3.11, 3.12, and 3.13 support
- Comprehensive test suite with 32 tests
- Documentation and examples
- Context manager support (`__enter__`, `__exit__`)

### Technical Details
- Proper fsspec integration using `url_to_fs()` for URL parsing
- Bytes-based storage (no memoryview) for cloud storage compatibility
- Sequential file reading during deserialization with immediate file closure
- Enhanced compatibility with buffered cloud storage backends

### Dependencies
- bitarray >= 2.0.0
- succinct >= 0.0.7
- fsspec >= 2021.0.0

[Unreleased]: https://github.com/andrey-savov/compact-tree/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/andrey-savov/compact-tree/releases/tag/v0.1.0
