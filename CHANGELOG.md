# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-15

### Added

- Gzip compression support for `serialize()` and `__init__()` via `storage_options={"compression": "gzip"}`
- Pickle support (`__reduce__` / `_unpickle_from_bytes`) for serialization with Python's `pickle` module
- `__repr__()` for `CompactTree` returns `CompactTree.from_dict(...)` representation
- `__str__()` for `CompactTree` returns dict-like string representation
- `__repr__()` and `__str__()` for `_Node` sub-tree objects
- Tests for compression, pickle, and string representations

## [0.2.0] - 2026-02-10

### Changed

- Renamed PyPI package from `compact-tree` to `savov-compact-tree`
- Refactored internal storage from `memoryview` to `bytes` for cloud storage compatibility
- Added Python 3.13 support

### Technical Details

- Proper fsspec integration using `url_to_fs()` for URL parsing
- Bytes-based storage for compatibility with buffered cloud storage backends
- Sequential file reading during deserialization with immediate file closure

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
- Python 3.9, 3.10, 3.11, and 3.12 support
- Comprehensive test suite
- Documentation and examples
- Context manager support (`__enter__`, `__exit__`)

### Dependencies

- bitarray >= 2.0.0
- succinct >= 0.0.7
- fsspec >= 2021.0.0

[Unreleased]: https://github.com/andrey-savov/compact-tree/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/andrey-savov/compact-tree/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/andrey-savov/compact-tree/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/andrey-savov/compact-tree/releases/tag/v0.1.0
