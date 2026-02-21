# compact-tree

[![Tests](https://github.com/andrey-savov/compact-tree/workflows/Tests/badge.svg)](https://github.com/andrey-savov/compact-tree/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact, read-only nested dictionary backed by a DAWG-style radix trie.

`CompactTree` stores a nested Python `dict` using a path-compressed radix trie with DAWG-style key/value deduplication, enabling low-memory random access and efficient serialization.

## Features

- **Memory-efficient**: DAWG-style deduplication via two `MarisaTrie` instances (one for keys, one for values)
- **Fast lookups**: Plain list-indexing over parallel arrays — no rank/select overhead
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
tree.serialize("tree.ctree.gz", storage_options={"compression": "gzip"})
loaded_gz = CompactTree("tree.ctree.gz", storage_options={"compression": "gzip"})

# Pickle support
import pickle
data = pickle.dumps(tree)
tree2 = pickle.loads(data)

# Convert back to plain dict
plain_dict = loaded_tree.to_dict()
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
```

Each non-root node `v` (0-indexed) occupies a slot in both `elbl` (its edge label / key id) and `vcol` (its value id, or the sentinel `0xFFFFFFFF` for internal nodes).

Child navigation uses CSR (Compressed Sparse Row) arrays: `_child_start[v]` is the start offset and `_child_count[v]` is the count of children of node `v`.

## Binary Format (v5)

```
Magic   : 5 bytes    "CTree"
Version : 8 bytes    uint64 LE  (always 5)
Header  : 7 × 8 bytes  lengths of: keys_trie, val_trie, child_count,
                         vcol, elbl, key_vocab_size, val_vocab_size
Payload : keys_trie_bytes | val_trie_bytes | child_count_bytes
          | vcol_bytes | elbl_bytes
```

`keys_trie_bytes` and `val_trie_bytes` are serialized `MarisaTrie` instances (CSR format). `child_count_bytes`, `vcol_bytes`, and `elbl_bytes` are packed `uint32` arrays. `key_vocab_size` and `val_vocab_size` record the LRU cache sizes used during `from_dict` and are restored on load so query-time caches are immediately correctly sized.

Files written in v4 or earlier (LOUDS-based) are **not** supported. Use v1.x to migrate old files if needed.

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

| Metric | v2.0.0 |
|---|---|
| `from_dict` build time | 7.3s |
| Lookup throughput | 67,889/s (14.7 µs/lookup) |
| Serialize | 14.0/s (71.6 ms), 77.2 MiB |
| Deserialize | 1.0/s (999 ms) |

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
