# compact-tree

[![Tests](https://github.com/andrey-savov/compact-tree/workflows/Tests/badge.svg)](https://github.com/andrey-savov/compact-tree/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact, read-only nested dictionary backed by a DAWG-style radix trie.

`CompactTree` stores a nested Python `dict` using a path-compressed radix trie with DAWG-style key/value deduplication, enabling low-memory random access and efficient serialization.

## Features

- **Memory-efficient**: DAWG-style deduplication via two `MarisaTrie` instances (one for keys, one for values)
- **Flat variant**: `CompactTreeFlat` stores all leaf paths as flat `tuple[str, …] → val_id` mappings — no intermediate nodes, `get_path()` only, ~38% faster than `CompactTree.get_path()` for full-path lookups
- **Fast lookups**: Plain list-indexing over parallel arrays — no rank/select overhead
- **C-accelerated lookups**: Optional `_marisa_ext` C extension provides `TrieIndex` and `TreeIndex` for ~5–10× faster uncached queries
- **Multi-level traversal**: `get_path(*keys)` descends multiple levels in a single C call, eliminating intermediate `_Node` allocations
- **Shared trie**: `shared_trie=True` uses a single `MarisaTrie` for both keys and values, saving memory when the vocabularies overlap
- **High-performance builds**: 7.3s for a 6.2M-leaf, 173K-key tree (v2.0.0)
- **Fast serialization**: 14/s at 173K keys, 77 MiB files
- **Serializable**: Save and load from disk with efficient binary format
- **Gzip compression**: Optional gzip compression for smaller files on disk
- **Pickle support**: Fully serializable via Python's `pickle` module
- **Read-only**: Optimized for lookup-heavy workloads
- **Storage-agnostic**: Works with local files and remote storage via `fsspec`
- **Dict-like interface**: Supports `[]`, `in`, `len()`, iteration, `repr()`, and `str()`

## Installation

```bash
pip install savov-compact-tree
```

Or install from source:

```bash
git clone https://github.com/andrey-savov/compact-tree.git
cd compact-tree
pip install -e .
```

## Quick Start

```python
from compact_tree import CompactTree

# Build from a nested dict
tree = CompactTree.from_dict({
    "a": {
        "x": "1",
        "y": "2"
    },
    "b": "3"
})

# Access like a normal dict
print(tree["a"]["x"])   # "1"
print(tree["b"])        # "3"
print("a" in tree)      # True
print(len(tree))        # 2
print(list(tree))       # ["a", "b"]

# String representations
print(str(tree))        # {'a': {'x': '1', 'y': '2'}, 'b': '3'}
print(repr(tree))       # CompactTree.from_dict({'a': {'x': '1', ...}, 'b': '3'})

# Serialize to file
tree.serialize("tree.ctree")

# Load from file
loaded_tree = CompactTree("tree.ctree")

# Serialize with gzip compression
tree.serialize("tree.ctree.gz", compression="gzip")
loaded_gz = CompactTree("tree.ctree.gz", compression="gzip")

# Shared trie: keys and values share one MarisaTrie (saves memory when vocabularies overlap)
tree_shared = CompactTree.from_dict({"a": "b", "b": "a"}, shared_trie=True)

# Pickle support
import pickle
data = pickle.dumps(tree)
tree2 = pickle.loads(data)

# Convert back to plain dict
plain_dict = loaded_tree.to_dict()

# Multi-level lookup in a single C call (when C extension is compiled)
result = tree.get_path("a", "x")   # equivalent to tree["a"]["x"] but faster
```

### CompactTreeFlat — flat full-path lookup

When you always know the full path depth at lookup time and don't need intermediate
navigation, use `CompactTreeFlat` for ~38% faster `get_path()` calls:

```python
from compact_tree_flat import CompactTreeFlat

d = {"a": {"x": "1"}, "b": {"x": "2", "y": "3"}}
tree = CompactTreeFlat.from_dict(d)

tree.get_path("a", "x")          # "1" — single dict lookup + restore_key, no trie traversal
tree.get_path("b", "y")          # "3"
("b", "x") in tree               # True — __contains__ takes a tuple
len(tree)                         # 3  (total leaf paths)
tree.to_dict()                   # {"a": {"x": "1"}, "b": {"x": "2", "y": "3"}}

# Serialize / deserialize (CTFlt v1 binary format)
tree.serialize("tree.ctflat")
tree2 = CompactTreeFlat("tree.ctflat")

tree.serialize("tree.ctflat.gz", compression="gzip")
tree3 = CompactTreeFlat("tree.ctflat.gz", compression="gzip")

import pickle
tree4 = pickle.loads(pickle.dumps(tree))
```

## How It Works

### MarisaTrie

`MarisaTrie` is a compact word-to-index mapping backed by a path-compressed radix trie with subtree word counts for minimal perfect hashing (MPH). `CompactTree` uses two `MarisaTrie` instances — one for keys and one for values — to provide DAWG-style deduplication.

- **Path compression**: single-child edges are merged for compactness
- **Dense indexing**: every unique word gets an index in `[0, N)`
- **Reverse lookup**: recover the original word from its index
- **Bulk enumeration**: `to_dict()` returns `{word: index}` for all words in O(N); the first call after construction returns a pre-built mapping (zero trie traversals) and frees it immediately
- **Per-instance LRU cache**: `functools.lru_cache` on `index()` lookups, automatically sized to the vocabulary; cache size preserved through serialization

At query time, navigation uses plain Python parallel lists (`_node_labels`, `_node_children`, `_node_counts`, `_node_terminal`) — no rank/select overhead.

### DAWG-Style Deduplication

- **Keys** are collected, sorted, and deduplicated via a `MarisaTrie`
- **Values** (leaves) are similarly deduplicated via a second `MarisaTrie`
- Edge labels store integer IDs rather than raw strings
- The same key or value appearing at multiple levels is stored only once

## Architecture

```
CompactTree
  |
  +-- _child_start : array.array('I')  child start offsets (CSR), one per node
  +-- _child_count : array.array('I')  child counts (CSR), one per node
  +-- elbl         : array.array('I')  edge labels (uint32 key ids, one per node)
  +-- vcol         : array.array('I')  value column (uint32: value id or 0xFFFFFFFF for internal nodes)
  +-- _key_trie    : MarisaTrie        key vocabulary (word <-> dense index)
  +-- _val_trie    : MarisaTrie        value vocabulary (word <-> dense index)
  +-- _shared_trie : bool              True when a single MarisaTrie covers keys and values
  +-- _c_tree      : TreeIndex | None  optional C-level traversal helper (None if extension not built)
```

```
CompactTreeFlat
  |
  +-- _key_dict      : dict[tuple[str, ...], int]  full path → val_id
  +-- _val_trie      : MarisaTrie                  value vocabulary
  +-- _val_vocab_size: int                          lru_cache size hint
```

`MarisaTrie` additionally exposes:
- `_c_index` (`TrieIndex`) — C-level lookup helper, set by `_build_c_index()` when the extension is available.
- `_first_char_maps` / `_prefix_counts` / `_node_label_lens` — navigation tables built by `_build_navigation_tables()` for O(1) child dispatch and MPH index accumulation.

Each non-root node `v` (0-indexed) occupies a slot in both `elbl` (its edge label / key id) and `vcol` (its value id, or the sentinel `0xFFFFFFFF` for internal nodes).

Child navigation uses CSR (Compressed Sparse Row) arrays: `_child_start[v]` is the start offset and `_child_count[v]` is the count of children of node `v`.

## Binary Format

### CompactTree — format v6

```
Magic   : 5 bytes    "CTree"
Version : 8 bytes    uint64 LE  (always 6)
Header  : 8 × 8 bytes  shared_flag, keys_trie_len, val_trie_len, child_count_len,
                         vcol_len, elbl_len, key_vocab_size, val_vocab_size
Payload : keys_trie_bytes | val_trie_bytes (empty when shared)
          | child_count_bytes | vcol_bytes | elbl_bytes
```

When `shared_flag=1` (built with `shared_trie=True`), `val_trie_len=0` and no
value-trie blob is written; on load both `_key_trie` and `_val_trie` point to the
same deserialized `MarisaTrie`.

Files written in v5 or earlier are **not** supported.

### CompactTreeFlat — format CTFlt v1

```
Magic   : 5 bytes    "CTFlt"
Version : 8 bytes    uint64 LE  (always 1)
Header  : 4 × 8 bytes  n_paths, val_trie_len, paths_buf_len, val_vocab_size
Payload : val_trie_bytes | paths_buf
```

## Dependencies

- `bitarray` — Bit-packed boolean arrays (used for terminal flags in MarisaTrie serialization)
- `fsspec` — Filesystem abstraction for local and remote storage

## Testing

```bash
pytest test_compact_tree.py test_marisa_trie.py
```

### Benchmarks

Run performance benchmarks with `pytest-benchmark`:

```bash
pytest test_compact_tree.py::TestLoadPerformance --benchmark-only -v
```

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed results and [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for optimization history.

## Performance

Benchmark: 3-level nested dict, shape `{L0=9, L1=4, L2=173,000}`, 6.2M leaf entries.
Windows 11, Intel Core Ultra 9 285H, Python 3.14.3, 10 s timed loops.

| Target | Lookups / s | µs / lookup | vs dict |
|---|---|---|---|
| `dict[k0][k1][k2]` | 228,998 | 4.4 | 1.0× (baseline) |
| `CompactTreeFlat.get_path()` | 194,220 | 5.1 | **0.85×** |
| `CompactTree.get_path()` (C ext) | 141,224 | 7.1 | 0.62× |
| `CompactTree[k0][k1][k2]` (no cache) | 124,384 | 8.0 | 0.54× |

See [PERFORMANCE.md](PERFORMANCE.md) for full comparisons including PyArrow.

*`CompactTreeFlat.get_path()` is ~38% faster than `CompactTree.get_path()` because a
full-path lookup reduces to one Python `dict.__getitem__` + one `restore_key` call.*

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
